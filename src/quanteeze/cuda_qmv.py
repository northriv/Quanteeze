# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A decode's 4-bit projections on MLX's CUDA backend in fewer instructions: MLX's own qmv, the same
arithmetic in the same order, bit for bit.

MLX's qmv (quantized/qmm/qmv.cu) computes a row of a one-row projection with a warp: each lane takes
16 inputs a step, a step every 512, dequantizes their weights with cutlass's converters (w * scale,
then + bias, each rounded to bfloat16) and multiply-adds them into its 16 bfloat16 sums, then adds
the lane's sums in float32 and cg::reduces the warp's.  Its converters turn each 4-bit value into
bfloat16 through an integer-to-float and a float-to-bfloat16 conversion, a quarter of the A100's
rate, and that, not the bytes, was the projections' time.  Here a pair of 4-bit values becomes
a bfloat16 pair exactly (their bits under 128's, less 128) and the scale, the bias and the sums are
bfloat16x2 ops, each rounding once as cutlass's do (fusing w * scale + bias into one fma does not
match).  A pair's nibbles by one byte_perm (qz_nibbles) took 3-8% more off.

install() takes mx.quantized_matmul's one-row calls (a decode's: the hyper-connections', the
attentions', the shared experts' projections) of an affine 4-bit bfloat16 projection with biases,
in groups of 32, 64 or 128 (2 and 8 bits, and float16, stay MLX's).  None is taken on trust: the
first CHECKS times a shape is met its output is compared bit for bit with MLX's, and it serves the
shape only if they are the same; a call traced into a compiled graph (quanteeze.arch.qwen4_exp's
routers and shared experts) is MLX's until its shape's checks are made.  QUANTEEZE_CUDA_QMV=off
leaves mx.quantized_matmul as it is.

HEADER also gives the fused kernels (quanteeze.hyper, quanteeze.cuda_moe) a row of their own:
qz_qmv4_row_ahead, qz_qmv4_row with every step's weights read before the first is computed (its
input in shared memory: a row of 10240 had waited on its 20 steps one after another), and
qz_qmv8_row, MLX's 8-bit qmv (8 inputs a lane a step of 256, each weight exactly to bfloat16, times
its scale and plus its bias rounded there, multiply-added in float32).
"""
# Portions of the CUDA kernel arithmetic and layout below are adapted from Apple MLX
# (Copyright © 2023-2026 Apple Inc., MIT License — see LICENSE).
import sys
from functools import lru_cache

import mlx.core as mx

from . import cuda_kernels

CHECKS = 3
ON = [True]                     # QUANTEEZE_CUDA_QMV
CHECKS = 3
ON = [True]                     # QUANTEEZE_CUDA_QMV
ROWS_ON = [True]                # QUANTEEZE_CUDA_QMV_ROWS: a call of 2 to ROWS_MOST rows through qmv_rows()
ROWS = ROWS_ON
ROWS_MOST = 7                   # MLX 0.32.4 computes up to 7 rows as one-row qmvs; 8 and more by its qmm
BITS = (4, 8)                   # bits computes() takes (qz_qmv4_row(s), qz_qmv8_row(s))
ROWS_LEAST = {4: 1 << 21, 8: 1 << 22}   # minimum weights * (rows - 1) for qmv_rows()
ONE_LEAST = [1 << 24]           # minimum weights for one-row 8-bit qmv() (None: leave to MLX)
STATS = dict(served=0, rows=0, checked=0, mismatched=0)
CHOICE = {}                     # (K, N, group size[, bits, rows]): whether it matched MLX's
CHECKED = {}
SAID = []
_ORIGINAL = []                  # mx.quantized_matmul as MLX gave it

HEADER = r"""
#include <cooperative_groups/reduce.h>

// the pair p of a word v's 8 nibbles (inputs 2p and 2p + 1) under bits 0 and 16: byte p of v and of
// v >> 4, their low nibbles (one byte_perm and a mask where shifts and masks took four or five)
__device__ __forceinline__ uint32_t qz_nibbles(uint32_t v, int p) {
  return __byte_perm(v, v >> 4, static_cast<unsigned>(p | ((4 + p) << 8))) & 0x000F000Fu;
}

// MLX's qmv_kernel_impl for a row of K 4-bit inputs in groups of GS: lane l takes inputs l * 16 to
// l * 16 + 15 of each 512, its pairs (2p, 2p + 1) dequantized and multiply-added as bfloat16x2.
template <int K, int GS>
__device__ __forceinline__ float qz_qmv4_row(const __nv_bfloat16* x, const uint32_t* w, const __nv_bfloat16* scales,
                                             const __nv_bfloat16* biases, int row) {
  auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  constexpr int per_warp = 512;
  constexpr int ITERS = K / per_warp;
  constexpr int rest = K % per_warp;
  const uint32_t* wr = w + static_cast<int64_t>(row) * (K / 8);
  scales += static_cast<int64_t>(row) * (K / GS);
  biases += static_cast<int64_t>(row) * (K / GS);
  const int lane = warp.thread_rank();
  __nv_bfloat162 acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = __float2bfloat162_rn(0.0f);
  }
  const uint32_t magic_bits = 0x43004300u;              // 128 in each half
  const __nv_bfloat162 magic = *reinterpret_cast<const __nv_bfloat162*>(&magic_bits);
  struct Load { uint2 ww; uint4 xa; uint4 xb; __nv_bfloat16 s; __nv_bfloat16 b; };
  auto load = [&](int idx) {
    Load l;
    l.ww = *reinterpret_cast<const uint2*>(wr + idx / 8);
    l.xa = *reinterpret_cast<const uint4*>(x + idx);
    l.xb = *reinterpret_cast<const uint4*>(x + idx + 8);
    l.s = scales[idx / GS];
    l.b = biases[idx / GS];
    return l;
  };
  auto compute = [&](const Load& l) {
    __nv_bfloat162 s2 = __bfloat162bfloat162(l.s);
    __nv_bfloat162 b2 = __bfloat162bfloat162(l.b);
    uint32_t xs[8] = {l.xa.x, l.xa.y, l.xa.z, l.xa.w, l.xb.x, l.xb.y, l.xb.z, l.xb.w};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      uint32_t v = j ? l.ww.y : l.ww.x;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        uint32_t bits = qz_nibbles(v, p) | magic_bits;
        __nv_bfloat162 q = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits), magic);
        __nv_bfloat162 wd = __hadd2(__hmul2(q, s2), b2);
        __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(&xs[j * 4 + p]);
        acc[j * 4 + p] = __hfma2(xv, wd, acc[j * 4 + p]);
      }
    }
  };
#pragma unroll
  for (int r = 0; r < ITERS; ++r) {
    compute(load(lane * 16 + r * per_warp));
  }
  if constexpr (rest != 0) {
    int idx = lane * 16 + K - rest;
    if (idx < K) {
      compute(load(idx));
    }
  }
  float sum{0};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum += __low2float(acc[i]);
    sum += __high2float(acc[i]);
  }
  return cooperative_groups::reduce(warp, sum, cooperative_groups::plus<float>{});
}

// qz_qmv4_row with every step's weights, scales and biases read before the first is computed (the
// input, in shared memory, read as it is used): the same arithmetic in the same order.
template <int K, int GS>
__device__ __forceinline__ float qz_qmv4_row_ahead(const __nv_bfloat16* x, const uint32_t* w,
                                                   const __nv_bfloat16* scales, const __nv_bfloat16* biases,
                                                   int row) {
  auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  constexpr int per_warp = 512;
  constexpr int ITERS = K / per_warp;
  constexpr int rest = K % per_warp;
  constexpr int STEPS = ITERS + (rest != 0);
  const uint32_t* wr = w + static_cast<int64_t>(row) * (K / 8);
  scales += static_cast<int64_t>(row) * (K / GS);
  biases += static_cast<int64_t>(row) * (K / GS);
  const int lane = warp.thread_rank();
  uint2 ww[STEPS];
  __nv_bfloat16 ss[STEPS], bb[STEPS];
#pragma unroll
  for (int r = 0; r < STEPS; ++r) {
    const int idx = lane * 16 + (r < ITERS ? r * per_warp : K - rest);
    if (r < ITERS || idx < K) {
      ww[r] = *reinterpret_cast<const uint2*>(wr + idx / 8);
      ss[r] = scales[idx / GS];
      bb[r] = biases[idx / GS];
    }
  }
  __nv_bfloat162 acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = __float2bfloat162_rn(0.0f);
  }
  const uint32_t magic_bits = 0x43004300u;
  const __nv_bfloat162 magic = *reinterpret_cast<const __nv_bfloat162*>(&magic_bits);
#pragma unroll
  for (int r = 0; r < STEPS; ++r) {
    const int idx = lane * 16 + (r < ITERS ? r * per_warp : K - rest);
    if (r < ITERS || idx < K) {
      const uint4 xa = *reinterpret_cast<const uint4*>(x + idx);
      const uint4 xb = *reinterpret_cast<const uint4*>(x + idx + 8);
      const uint32_t xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
      const __nv_bfloat162 s2 = __bfloat162bfloat162(ss[r]);
      const __nv_bfloat162 b2 = __bfloat162bfloat162(bb[r]);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint32_t v = j ? ww[r].y : ww[r].x;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          uint32_t bits = qz_nibbles(v, p) | magic_bits;
          __nv_bfloat162 q = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits), magic);
          __nv_bfloat162 wd = __hadd2(__hmul2(q, s2), b2);
          __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(&xs[j * 4 + p]);
          acc[j * 4 + p] = __hfma2(xv, wd, acc[j * 4 + p]);
        }
      }
    }
  }
  float sum{0};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum += __low2float(acc[i]);
    sum += __high2float(acc[i]);
  }
  return cooperative_groups::reduce(warp, sum, cooperative_groups::plus<float>{});
}

// MLX's qmv_kernel_impl for a row of K 8-bit inputs: lane l takes inputs l * 8 to l * 8 + 7 of each
// 256, each weight dequantized in bfloat16 (exactly to bfloat16, then w * scale and + bias rounded
// each), and multiply-adds them as float32 into its 8 float32 sums; the sums added in order, the
// warp's cg::reduced.
template <int K, int GS, bool STREAM = false>
__device__ __forceinline__ float qz_qmv8_row(const __nv_bfloat16* x, const uint32_t* w, const __nv_bfloat16* scales,
                                             const __nv_bfloat16* biases, int row) {
  auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  constexpr int per_warp = 256;
  constexpr int ITERS = K / per_warp;
  constexpr int rest = K % per_warp;
  const uint8_t* wr = reinterpret_cast<const uint8_t*>(w) + static_cast<int64_t>(row) * K;
  scales += static_cast<int64_t>(row) * (K / GS);
  biases += static_cast<int64_t>(row) * (K / GS);
  const int lane = warp.thread_rank();
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }
  auto tile = [&](int idx) {
    const uint2* wp = reinterpret_cast<const uint2*>(wr + idx);
    const uint2 ww = STREAM ? __ldcs(wp) : *wp;
    const uint4 xv = *reinterpret_cast<const uint4*>(x + idx);
    const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
    const __nv_bfloat162 s2 = __bfloat162bfloat162(scales[idx / GS]);
    const __nv_bfloat162 b2 = __bfloat162bfloat162(biases[idx / GS]);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const uint32_t v = (p < 2 ? ww.x : ww.y) >> (16 * (p & 1));
      const float f0 = __uint_as_float(0x4B000000u | (v & 0xFFu)) - 8388608.0f;          // exactly
      const float f1 = __uint_as_float(0x4B000000u | ((v >> 8) & 0xFFu)) - 8388608.0f;
      const __nv_bfloat162 wd = __hadd2(__hmul2(__floats2bfloat162_rn(f0, f1), s2), b2);
      const float2 wf = __bfloat1622float2(wd);
      const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xs[p]));
      acc[2 * p] = __fmaf_rn(xf.x, wf.x, acc[2 * p]);
      acc[2 * p + 1] = __fmaf_rn(xf.y, wf.y, acc[2 * p + 1]);
    }
  };
#pragma unroll
  for (int r = 0; r < ITERS; ++r) {
    tile(lane * 8 + r * per_warp);
  }
  if constexpr (rest != 0) {
    int idx = lane * 8 + K - rest;
    if (idx < K) {
      tile(idx);
    }
  }
  float sum{0};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum += acc[i];
  }
  return cooperative_groups::reduce(warp, sum, cooperative_groups::plus<float>{});
}

// qz_qmv4_row for R rows of x (row r at x + r * K) through the same weight row: each step's weights
// dequantized once, then multiply-added into every row's own sums in the one-row order.
template <int K, int GS, int R>
__device__ __forceinline__ void qz_qmv4_rows(const __nv_bfloat16* x, const uint32_t* w, const __nv_bfloat16* scales,
                                             const __nv_bfloat16* biases, int row, float (&out)[R]) {
  auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  constexpr int per_warp = 512;
  constexpr int ITERS = K / per_warp;
  constexpr int rest = K % per_warp;
  const uint32_t* wr = w + static_cast<int64_t>(row) * (K / 8);
  scales += static_cast<int64_t>(row) * (K / GS);
  biases += static_cast<int64_t>(row) * (K / GS);
  const int lane = warp.thread_rank();
  __nv_bfloat162 acc[R][8];
#pragma unroll
  for (int t = 0; t < R; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[t][i] = __float2bfloat162_rn(0.0f);
    }
  }
  const uint32_t magic_bits = 0x43004300u;
  const __nv_bfloat162 magic = *reinterpret_cast<const __nv_bfloat162*>(&magic_bits);
  auto step = [&](int idx) {
    const uint2 ww = *reinterpret_cast<const uint2*>(wr + idx / 8);
    const __nv_bfloat162 s2 = __bfloat162bfloat162(scales[idx / GS]);
    const __nv_bfloat162 b2 = __bfloat162bfloat162(biases[idx / GS]);
    __nv_bfloat162 wd[8];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const uint32_t v = j ? ww.y : ww.x;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        uint32_t bits = qz_nibbles(v, p) | magic_bits;
        __nv_bfloat162 q = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits), magic);
        wd[j * 4 + p] = __hadd2(__hmul2(q, s2), b2);
      }
    }
#pragma unroll
    for (int t = 0; t < R; ++t) {
      const uint4 xa = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(t) * K + idx);
      const uint4 xb = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(t) * K + idx + 8);
      const uint32_t xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(&xs[i]);
        acc[t][i] = __hfma2(xv, wd[i], acc[t][i]);
      }
    }
  };
#pragma unroll
  for (int r = 0; r < ITERS; ++r) {
    step(lane * 16 + r * per_warp);
  }
  if constexpr (rest != 0) {
    int idx = lane * 16 + K - rest;
    if (idx < K) {
      step(idx);
    }
  }
#pragma unroll
  for (int t = 0; t < R; ++t) {
    float sum{0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      sum += __low2float(acc[t][i]);
      sum += __high2float(acc[t][i]);
    }
    out[t] = cooperative_groups::reduce(warp, sum, cooperative_groups::plus<float>{});
  }
}

// qz_qmv8_row for R rows of x (row r at x + r * K) through the same weight row.
template <int K, int GS, int R>
__device__ __forceinline__ void qz_qmv8_rows(const __nv_bfloat16* x, const uint32_t* w, const __nv_bfloat16* scales,
                                             const __nv_bfloat16* biases, int row, float (&out)[R]) {
  auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  constexpr int per_warp = 256;
  constexpr int ITERS = K / per_warp;
  constexpr int rest = K % per_warp;
  const uint8_t* wr = reinterpret_cast<const uint8_t*>(w) + static_cast<int64_t>(row) * K;
  scales += static_cast<int64_t>(row) * (K / GS);
  biases += static_cast<int64_t>(row) * (K / GS);
  const int lane = warp.thread_rank();
  float acc[R][8];
#pragma unroll
  for (int t = 0; t < R; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[t][i] = 0.0f;
    }
  }
  auto tile = [&](int idx) {
    const uint2 ww = *reinterpret_cast<const uint2*>(wr + idx);
    const __nv_bfloat162 s2 = __bfloat162bfloat162(scales[idx / GS]);
    const __nv_bfloat162 b2 = __bfloat162bfloat162(biases[idx / GS]);
    float2 wf[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const uint32_t v = (p < 2 ? ww.x : ww.y) >> (16 * (p & 1));
      const float f0 = __uint_as_float(0x4B000000u | (v & 0xFFu)) - 8388608.0f;
      const float f1 = __uint_as_float(0x4B000000u | ((v >> 8) & 0xFFu)) - 8388608.0f;
      const __nv_bfloat162 wd = __hadd2(__hmul2(__floats2bfloat162_rn(f0, f1), s2), b2);
      wf[p] = __bfloat1622float2(wd);
    }
#pragma unroll
    for (int t = 0; t < R; ++t) {
      const uint4 xv = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(t) * K + idx);
      const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xs[p]));
        acc[t][2 * p] = __fmaf_rn(xf.x, wf[p].x, acc[t][2 * p]);
        acc[t][2 * p + 1] = __fmaf_rn(xf.y, wf[p].y, acc[t][2 * p + 1]);
      }
    }
  };
#pragma unroll
  for (int r = 0; r < ITERS; ++r) {
    tile(lane * 8 + r * per_warp);
  }
  if constexpr (rest != 0) {
    int idx = lane * 8 + K - rest;
    if (idx < K) {
      tile(idx);
    }
  }
#pragma unroll
  for (int t = 0; t < R; ++t) {
    float sum{0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      sum += acc[t][i];
    }
    out[t] = cooperative_groups::reduce(warp, sum, cooperative_groups::plus<float>{});
  }
}
"""

# A warp a row, 8 rows a block, as MLX's qmv launches it.
SOURCE = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int row = blockIdx.x * 8 + warp.meta_group_rank();
    if (row >= N_OUT) {
      return;
    }
    float sum = qz_qmv4_row<K_IN, GS>(x, w, scales, biases, row);
    if (warp.thread_rank() == 0) {
      y[row] = __float2bfloat16_rn(sum);
    }
"""

ROWS_SOURCE = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int row = blockIdx.x * 8 + warp.meta_group_rank();
    if (row >= N_OUT) {
      return;
    }
    float sums[R];
    if constexpr (BITS == 8) {
      qz_qmv8_rows<K_IN, GS, R>(x, w, scales, biases, row, sums);
    } else {
      qz_qmv4_rows<K_IN, GS, R>(x, w, scales, biases, row, sums);
    }
    if (warp.thread_rank() == 0) {
#pragma unroll
      for (int t = 0; t < R; ++t) {
        y[static_cast<int64_t>(t) * N_OUT + row] = __float2bfloat16_rn(sums[t]);
      }
    }
"""


@lru_cache(maxsize=None)
def _kernel(bits=4):
    if bits == 8:
        return mx.fast.cuda_kernel(
            name='quanteeze_qmv8',
            input_names=['x', 'w', 'scales', 'biases'],
            output_names=['y'],
            source=SOURCE.replace('qz_qmv4_row<K_IN, GS>', 'qz_qmv8_row<K_IN, GS, true>'),
            header=HEADER,
        )
    return mx.fast.cuda_kernel(name='quanteeze_qmv4', input_names=['x', 'w', 'scales', 'biases'],
                               output_names=['y'], source=SOURCE, header=HEADER)


def qmv(x, w, scales, biases, group_size, bits=4):
    """mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size, bits) of one row x (4 or 8 bits)."""
    n, k = w.shape[0], x.shape[-1]
    return _kernel(int(bits))(
        inputs=[x, w, scales, biases],
        template=[('GS', int(group_size)), ('N_OUT', n), ('K_IN', k)],
        grid=(32 * -(-n // 8), 8, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(maxsize=None)
def _rows_kernel():
    return mx.fast.cuda_kernel(name='quanteeze_qmv_rows', input_names=['x', 'w', 'scales', 'biases'],
                               output_names=['y'], source=ROWS_SOURCE, header=HEADER)


def qmv_rows(x, w, scales, biases, group_size, bits):
    """mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size, bits) of 2 to ROWS_MOST rows."""
    n, k = w.shape[0], x.shape[-1]
    rows = x.size // k
    return _rows_kernel()(
        inputs=[x, w, scales, biases],
        template=[('GS', int(group_size)), ('N_OUT', n), ('K_IN', k), ('R', rows), ('BITS', int(bits))],
        grid=(32 * -(-n // 8), 8, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]


def computes(x, w, scales, biases, transpose, group_size, bits, mode):
    """Whether qz_qmv4_row (4 bits) or qz_qmv8_row (8 bits) computes each row of this call."""
    return (transpose is True and mode == 'affine' and bits in BITS and biases is not None
            and group_size in (32, 64, 128) and x.dtype == mx.bfloat16 and scales.dtype == mx.bfloat16
            and biases.dtype == mx.bfloat16 and w.ndim == 2 and w.dtype == mx.uint32 and x.ndim >= 1
            and x.shape[-1] == scales.shape[-1] * group_size == w.shape[-1] * 32 // bits
            and scales.shape[0] == w.shape[0] and biases.shape == scales.shape)


def one_row(x, w, scales, biases, transpose, group_size, bits, mode):
    """Whether qz_qmv4_row or qz_qmv8_row computes this call of one row."""
    return x.ndim >= 1 and x.size == x.shape[-1] and computes(x, w, scales, biases, transpose, group_size, bits, mode)


def takes(x, w, scales, biases, transpose, group_size, bits, mode):
    """Whether qmv() computes this call as MLX's qmv does: one row of bfloat16, an affine 4-bit
    projection with biases in groups of 32, 64 or 128, the weights a matrix."""
    return bits == 4 and one_row(x, w, scales, biases, transpose, group_size, bits, mode)


def takes8(x, w, scales, biases, transpose, group_size, bits, mode):
    """Whether qmv(..., bits=8) computes this one-row affine 8-bit call (at least ONE_LEAST weights)."""
    return (bits == 8 and ONE_LEAST[0] is not None and w.ndim == 2 and w.shape[0] * w.shape[-1] * 4 >= ONE_LEAST[0]
            and one_row(x, w, scales, biases, transpose, group_size, bits, mode))


def takes_rows(x, w, scales, biases, transpose, group_size, bits, mode):
    """Whether qmv_rows() computes this call: 2 to ROWS_MOST rows through an affine 4- or 8-bit projection."""
    return (computes(x, w, scales, biases, transpose, group_size, bits, mode) and x.shape[-1] > 0
            and 1 < x.size // x.shape[-1] <= ROWS_MOST)


def quantized_matmul(x, w, scales, biases=None, transpose=True, group_size=None, bits=None, mode='affine', *,
                     stream=None):
    """mx.quantized_matmul: one-row calls (4/8-bit) through qmv() and 2..ROWS_MOST rows through qmv_rows()."""
    def theirs():
        return _ORIGINAL[0](x, w, scales, biases, transpose, group_size, bits, mode, stream=stream)
    if not ON[0] or stream is not None:
        return theirs()
    if takes(x, w, scales, biases, transpose, group_size, bits, mode):
        return _served((x.shape[-1], w.shape[0], int(group_size)), lambda: qmv(x, w, scales, biases, group_size), theirs)
    if takes8(x, w, scales, biases, transpose, group_size, bits, mode):
        return _served((x.shape[-1], w.shape[0], int(group_size), 8),
                       lambda: qmv(x, w, scales, biases, group_size, 8), theirs)
    if (ROWS_ON[0] and takes_rows(x, w, scales, biases, transpose, group_size, bits, mode)
            and w.shape[0] * x.shape[-1] * (x.size // x.shape[-1] - 1) >= ROWS_LEAST[bits]):
        rows = x.size // x.shape[-1]
        return _served((x.shape[-1], w.shape[0], int(group_size), int(bits), rows),
                       lambda: qmv_rows(x, w, scales, biases, group_size, bits), theirs, 'rows')
    return theirs()


def _served(key, mine, theirs, counted='served'):
    """mine() where this shape matched theirs() bit for bit the first CHECKS times it was met, else
    theirs(); a call being traced into a compiled graph (whose values cannot be read) theirs() until
    the checks are made outside one."""
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS[counted] += 1
            return mine()
        return theirs()
    want = theirs()
    try:
        got = mine()
        same = mx.array_equal(got, want).item()
    except Exception as error:
        if 'transformations' in str(error):                 # traced (mx.compile): nothing to compare yet
            return want
        if not SAID:                                        # a kernel that does not build: MLX's
            SAID.append(True)
            print(f'quanteeze: the decode\'s qmv kernel unavailable ({type(error).__name__}: {error})',
                  file=sys.stderr, flush=True)
        same = False
    STATS['checked'] += 1
    STATS['mismatched'] += not same
    CHOICE[key], CHECKED[key] = bool(same), (seen + 1 if same else CHECKS)
    return want


def install():
    """mx.quantized_matmul through quantized_matmul() from now on, where MLX runs on CUDA; whether it is."""
    if not cuda_kernels.available():
        return False
    if not _ORIGINAL:
        _ORIGINAL.append(mx.quantized_matmul)
        mx.quantized_matmul = quantized_matmul
    return True


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
