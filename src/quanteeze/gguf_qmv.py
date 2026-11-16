# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Metal and CUDA gather-qmv and tiled gather-mm kernels for raw GGUF / ggml block expert projections.

Runs pre-distributed and uniform GGUF expert tensors (`IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, `Q2_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q4_0`, `Q4_1`,
`Q5_0`, `Q5_1`, `Q8_0`, `F16`, `BF16`) directly from stacked uint8 ggml blocks without unpacking
weights to bfloat16/float32 memory.

Decode / verify (`takes` / `gather`): one SIMD group / warp (32 lanes) per routed slot and `ROWS`
outputs.  Prefill (`takes_rows` / `gather_rows`): tiled 32x32 threadgroup/block kernel over
expert-sorted rows (`simdgroup_matrix` on Metal, 128-thread register-tiled FP32 GEMM on CUDA).
Every shape & ggml type is checked against `unpacked()` for the first `CHECKS` calls
(`TOLERANCE = 1e-2`) before serving directly.
"""
import sys
from functools import lru_cache

import mlx.core as mx
import numpy as np

from . import cuda_kernels, iq_experts

CHECKS = 3
TOLERANCE = 1e-2
MAX_ROWS = 256
ROWS = 4
WARPS = 4
TILE = 32
ON = [True]                                     # synced with QUANTEEZE_KQUANT_QMV in serve
STATS = dict(served=0, checked=0, mismatched=0)
CHOICE, CHECKED, SAID = {}, {}, []
_DUMMY_GRID = None

_HEADER = r"""
template <typename P>
inline ushort u16_le(P p) {
  return ushort(p[0]) | (ushort(p[1]) << 8);
}
template <typename P>
inline uint u32_le(P p) {
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}
template <typename P>
inline float f16_le(P p) {
  return float(as_type<half>(u16_le(p)));
}
inline uint ksign_byte(uint s7) {
  return s7 | ((metal::popcount(s7) & 1u) << 7);
}

static constant float IQ2_MAG[4] = {8.0f, 25.0f, 43.0f, 0.0f};
static constant float IQ3_MAG[8] = {4.0f, 12.0f, 20.0f, 28.0f, 36.0f, 44.0f, 52.0f, 62.0f};
static constant float IQ4NL_MAG[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f
};
static constant float MXFP4_MAG[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};

template <int KIND, typename BlkPtr, typename GridPtr>
inline void decode_octad(BlkPtr blk, GridPtr grid, uint octad, thread float* w8) {
  const uint v0 = octad * 8u;
  if (KIND == 1) {                                       // F16 (block 32)
    for (int i = 0; i < 8; i++) w8[i] = f16_le(blk + 2u * (v0 + i));
  } else if (KIND == 30) {                               // BF16 (block 32)
    for (int i = 0; i < 8; i++) w8[i] = as_type<float>(uint(u16_le(blk + 2u * (v0 + i))) << 16);
  } else if (KIND == 2) {                                // Q4_0 (block 32)
    const float d = f16_le(blk);
    auto qs = blk + 2u + (octad < 2u ? v0 : v0 - 16u);
    const uint shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; i++) w8[i] = d * float(int((qs[i] >> shift) & 15u) - 8);
  } else if (KIND == 3) {                                // Q4_1 (block 32)
    const float d = f16_le(blk), m = f16_le(blk + 2u);
    auto qs = blk + 4u + (octad < 2u ? v0 : v0 - 16u);
    const uint shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; i++) w8[i] = d * float((qs[i] >> shift) & 15u) + m;
  } else if (KIND == 6) {                                // Q5_0 (block 32)
    const float d = f16_le(blk);
    const uint qhb = blk[2u + octad];
    auto qs = blk + 6u + (octad < 2u ? v0 : v0 - 16u);
    const uint shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; i++) {
      const uint q = ((qs[i] >> shift) & 15u) | (((qhb >> i) & 1u) << 4);
      w8[i] = d * float(int(q) - 16);
    }
  } else if (KIND == 7) {                                // Q5_1 (block 32)
    const float d = f16_le(blk), m = f16_le(blk + 2u);
    const uint qhb = blk[4u + octad];
    auto qs = blk + 8u + (octad < 2u ? v0 : v0 - 16u);
    const uint shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; i++) {
      const uint q = ((qs[i] >> shift) & 15u) | (((qhb >> i) & 1u) << 4);
      w8[i] = d * float(q) + m;
    }
  } else if (KIND == 8) {                                // Q8_0 (block 32)
    const float d = f16_le(blk);
    for (int i = 0; i < 8; i++) w8[i] = d * float(as_type<int8_t>(blk[2u + v0 + i]));
  } else if (KIND == 20) {                               // IQ4_NL (block 32)
    const float d = f16_le(blk);
    auto qs = blk + 2u + (octad < 2u ? v0 : v0 - 16u);
    const uint shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; i++) w8[i] = d * IQ4NL_MAG[(qs[i] >> shift) & 15u];
  } else if (KIND == 39) {                               // MXFP4 (block 32, 17B)
    const uint e = blk[0];
    const float d = as_type<float>(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
    auto qs = blk + 1u + (octad < 2u ? v0 : v0 - 16u);
    const uint shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; i++) w8[i] = d * MXFP4_MAG[(qs[i] >> shift) & 15u];
  } else if (KIND == 42) {                               // Q2_0 (block 64)
    const float d = f16_le(blk);
    auto qs = blk + 2u + 2u * octad;
    for (int i = 0; i < 8; i++) {
      const uint q = (qs[i >> 2] >> (2u * (i & 3))) & 3u;
      w8[i] = d * float(int(q) - 1);
    }
  } else if (KIND == 10) {                               // Q2_K (block 256)
    const uint sub = octad >> 1, h = v0 >> 7, j = (v0 & 127u) >> 5, l0 = v0 & 31u;
    const uint s = blk[sub];
    const float dl = f16_le(blk + 80u) * float(s & 15u);
    const float ml = f16_le(blk + 82u) * float(s >> 4);
    auto qs = blk + 16u + 32u * h + l0;
    for (int i = 0; i < 8; i++) w8[i] = dl * float((qs[i] >> (2u * j)) & 3u) - ml;
  } else if (KIND == 11) {                               // Q3_K (block 256)
    const uint sub = octad >> 1, h = v0 >> 7, j = (v0 & 127u) >> 5, l0 = v0 & 31u, hb = v0 >> 5;
    const uint low = sub < 8u ? (blk[96u + sub] & 15u) : (blk[96u + sub - 8u] >> 4);
    const uint high = (blk[104u + (sub & 3u)] >> (2u * (sub >> 2))) & 3u;
    const float dl = f16_le(blk + 108u) * float(int(low | (high << 4)) - 32);
    auto qs = blk + 32u + 32u * h + l0;
    for (int i = 0; i < 8; i++) {
      int q = int((qs[i] >> (2u * j)) & 3u);
      if (((blk[l0 + i] >> hb) & 1u) == 0u) q -= 4;
      w8[i] = dl * float(q);
    }
  } else if (KIND == 12 || KIND == 13) {                 // Q4_K / Q5_K (block 256)
    const uint sb = octad >> 2, l0 = v0 & 31u;
    auto sp = blk + 4u;
    uint sc, mn;
    if (sb < 4u) {
      sc = sp[sb] & 63u;
      mn = sp[sb + 4u] & 63u;
    } else {
      sc = (sp[sb + 4u] & 15u) | ((sp[sb - 4u] >> 2) & 48u);
      mn = (sp[sb + 4u] >> 4) | ((sp[sb] >> 2) & 48u);
    }
    const float dd = f16_le(blk) * float(sc);
    const float dm = f16_le(blk + 2u) * float(mn);
    auto qs = blk + (KIND == 12 ? 16u : 48u) + 32u * (sb >> 1) + l0;
    const uint shift = 4u * (sb & 1u);
    for (int i = 0; i < 8; i++) {
      uint q = (qs[i] >> shift) & 15u;
      if (KIND == 13) q |= ((uint(blk[16u + l0 + i]) >> sb) & 1u) << 4;
      w8[i] = dd * float(q) - dm;
    }
  } else if (KIND == 14) {                               // Q6_K (block 256)
    const uint sub = octad >> 1, h = v0 >> 7, rem = v0 & 127u;
    const float dsc = f16_le(blk + 208u) * float(as_type<int8_t>(blk[192u + sub]));
    const uint ql_shift = 4u * (rem >> 6), qh_shift = 2u * (rem >> 5);
    auto qlp = blk + 64u * h + (rem & 63u);
    auto qhp = blk + 128u + 32u * h + (rem & 31u);
    for (int i = 0; i < 8; i++) {
      const uint q = ((qlp[i] >> ql_shift) & 15u) | (((qhp[i] >> qh_shift) & 3u) << 4);
      w8[i] = dsc * float(int(q) - 32);
    }
  } else if (KIND == 16) {                               // IQ2_XXS (block 256)
    const uint ib32 = octad >> 2, l = octad & 3u;
    const float d = f16_le(blk);
    const uint aux = u32_le(blk + 2u + 8u * ib32 + 4u);
    const float db = d * (0.5f + float(aux >> 28)) * 0.25f;
    const uint sbyte = ksign_byte((aux >> (7u * l)) & 127u);
    const uint16_t g = grid[blk[2u + 8u * ib32 + l]];
    for (int i = 0; i < 8; i++) {
      const float sign = ((sbyte >> i) & 1u) ? -1.0f : 1.0f;
      w8[i] = db * IQ2_MAG[(g >> (2u * i)) & 3u] * sign;
    }
  } else if (KIND == 17) {                               // IQ2_XS (block 256)
    const uint ib32 = octad >> 2, l = octad & 3u;
    const float d = f16_le(blk);
    const uint q16 = u16_le(blk + 2u + 2u * octad);
    const uint s = blk[66u + ib32];
    const float db = d * (0.5f + float(l < 2u ? (s & 15u) : (s >> 4))) * 0.25f;
    const uint sbyte = ksign_byte(q16 >> 9);
    const uint16_t g = grid[q16 & 511u];
    for (int i = 0; i < 8; i++) {
      const float sign = ((sbyte >> i) & 1u) ? -1.0f : 1.0f;
      w8[i] = db * IQ2_MAG[(g >> (2u * i)) & 3u] * sign;
    }
  } else if (KIND == 22) {                               // IQ2_S (block 256)
    const uint ib32 = octad >> 2, l = octad & 3u;
    const float d = f16_le(blk);
    const uint idx10 = uint(blk[2u + octad]) | (((uint(blk[66u + ib32]) >> (2u * l)) & 3u) << 8);
    const uint sbyte = blk[34u + octad];
    const uint s = blk[74u + ib32];
    const float db = d * (0.5f + float(l < 2u ? (s & 15u) : (s >> 4))) * 0.25f;
    const uint16_t g = grid[idx10];
    for (int i = 0; i < 8; i++) {
      const float sign = ((sbyte >> i) & 1u) ? -1.0f : 1.0f;
      w8[i] = db * IQ2_MAG[(g >> (2u * i)) & 3u] * sign;
    }
  } else if (KIND == 19) {                               // IQ1_S (block 256)
    const uint ib32 = octad >> 2, l = octad & 3u;
    const float d = f16_le(blk);
    const uint qh = u16_le(blk + 34u + 2u * ib32);
    const float dl = d * float(2u * ((qh >> 12) & 7u) + 1u);
    const float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
    const uint idx11 = uint(blk[2u + octad]) | (((qh >> (3u * l)) & 7u) << 8);
    const uint16_t g = grid[idx11];
    for (int i = 0; i < 8; i++) {
      w8[i] = dl * (float(int((g >> (2u * i)) & 3u) - 1) + delta);
    }
  } else if (KIND == 29) {                               // IQ1_M (block 256)
    const uint ib16 = octad >> 1;
    const uint qhb = uint(blk[32u + ib16]) >> (4u * (octad & 1u));
    const uint sc0 = u16_le(blk + 48u), sc1 = u16_le(blk + 50u), sc2 = u16_le(blk + 52u), sc3 = u16_le(blk + 54u);
    const uint16_t d_bits = uint16_t(
        ((sc0 >> 12) & 0x000fu) | ((sc1 >> 8) & 0x00f0u)
        | ((sc2 >> 4) & 0x0f00u) | (sc3 & 0xf000u));
    const float d = float(as_type<half>(d_bits));
    const uint sc_word = u16_le(blk + 48u + 2u * (ib16 >> 2));
    const float dl = d * float(2u * ((sc_word >> (3u * (ib16 & 3u))) & 7u) + 1u);
    const float delta = (qhb & 8u) ? -0.125f : 0.125f;
    const uint16_t g = grid[uint(blk[octad]) | ((qhb & 7u) << 8)];
    for (int i = 0; i < 8; i++) {
      w8[i] = dl * (float(int((g >> (2u * i)) & 3u) - 1) + delta);
    }
  } else if (KIND == 18) {                               // IQ3_XXS (block 256)
    const uint ib32 = octad >> 2, l = octad & 3u;
    const float d = f16_le(blk);
    const uint aux = u32_le(blk + 66u + 4u * ib32);
    const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
    const uint sbyte = ksign_byte((aux >> (7u * l)) & 127u);
    const uint16_t g0 = grid[blk[2u + 8u * ib32 + 2u * l]];
    const uint16_t g1 = grid[blk[2u + 8u * ib32 + 2u * l + 1u]];
    for (int i = 0; i < 4; i++) {
      const float s0 = ((sbyte >> i) & 1u) ? -1.0f : 1.0f;
      const float s1 = ((sbyte >> (i + 4)) & 1u) ? -1.0f : 1.0f;
      w8[i] = db * IQ3_MAG[(g0 >> (4u * i)) & 7u] * s0;
      w8[i + 4] = db * IQ3_MAG[(g1 >> (4u * i)) & 7u] * s1;
    }
  } else if (KIND == 23) {                               // IQ4_XS (block 256)
    const uint ib32 = octad >> 2, l = octad & 3u;
    const float d = f16_le(blk);
    const uint high = u16_le(blk + 2u);
    const uint low_b = blk[4u + (ib32 >> 1)];
    const uint low = (ib32 & 1u) ? (low_b >> 4) : (low_b & 15u);
    const uint ls = low | (((high >> (2u * ib32)) & 3u) << 4);
    const float dl = d * float(int(ls) - 32);
    auto qs = blk + 8u + 16u * ib32 + 8u * (l & 1u);
    const uint shift = l < 2u ? 0u : 4u;
    for (int i = 0; i < 8; i++) w8[i] = dl * IQ4NL_MAG[(qs[i] >> shift) & 15u];
  }
}
"""

_SOURCE = r"""
  const uint lane = thread_position_in_grid.x;
  const uint first = thread_position_in_grid.y * ROWS;
  const uint r = thread_position_in_grid.z;
  const int e = idx[r];
  const device T* xr = x + (ulong)r * K_SIZE;
  float acc[ROWS];
  for (int row = 0; row < ROWS; row++) acc[row] = 0.0f;
  for (uint k0 = 0; k0 < K_SIZE; k0 += 256u) {
    const uint v0 = k0 + lane * 8u;
    if (v0 < K_SIZE) {
      float xv[8];
      for (int i = 0; i < 8; i++) xv[i] = float(xr[v0 + i]);
      const uint b = v0 / PB;
      const uint octad = (v0 % PB) / 8u;
      for (int row = 0; row < ROWS; row++) {
        const uint n = first + row;
        if (n >= N_SIZE) break;
        const ulong at = ((ulong)e * N_SIZE + n) * K_BLOCKS + b;
        float w8[8];
        decode_octad<KIND>(w + at * BB, grid, octad, w8);
        float dot = 0.0f;
        for (int i = 0; i < 8; i++) dot += xv[i] * w8[i];
        acc[row] += dot;
      }
    }
  }
  for (int row = 0; row < ROWS; row++) {
    const uint n = first + row;
    const float total = simd_sum(acc[row]);
    if (lane == 0u && n < N_SIZE) y[(ulong)r * N_SIZE + n] = static_cast<T>(total);
  }
"""

_TILED = r"""
  constexpr int BM = 32, BN = 32, BK = 32, LD = 36;
  threadgroup float Xs[BM * LD];
  threadgroup float Ws[BN * LD];
  threadgroup float Ys[BM * LD];
  const uint tid = thread_index_in_threadgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint n0 = threadgroup_position_in_grid.x * BN;
  const uint tile = threadgroup_position_in_grid.y;
  const int e = tiles[tile * 3], r0 = tiles[tile * 3 + 1], count = tiles[tile * 3 + 2];
  const uint lr = tid / 4u, seg = (tid % 4u) * 8u;
  const uint sr = (sg / 2u) * 16u, sn = (sg % 2u) * 16u;
  simdgroup_float8x8 acc[2][2];
  for (int i = 0; i < 2; i++) for (int jj = 0; jj < 2; jj++) acc[i][jj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const ulong row = ((ulong)e * N_SIZE + n0 + lr) * K_BLOCKS;
  for (uint k0 = 0; k0 < K_SIZE; k0 += BK) {
    for (int i = 0; i < 8; i++)
      Xs[lr * LD + seg + i] = (int)lr < count ? float(x[(ulong)(r0 + lr) * K_SIZE + k0 + seg + i]) : 0.0f;
    if (n0 + lr < N_SIZE) {
      const uint col = k0 + seg;
      const uint b = col / PB;
      const uint octad = (col % PB) / 8u;
      const ulong at = row + b;
      float w8[8];
      decode_octad<KIND>(w + at * BB, grid, octad, w8);
      for (int i = 0; i < 8; i++) Ws[lr * LD + seg + i] = w8[i];
    } else {
      for (int i = 0; i < 8; i++) Ws[lr * LD + seg + i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 a[2], b[2];
      for (int i = 0; i < 2; i++) simdgroup_load(a[i], Xs + (sr + 8u * i) * LD + kk, LD);
      for (int jj = 0; jj < 2; jj++) simdgroup_load(b[jj], Ws + (sn + 8u * jj) * LD + kk, LD, ulong2(0, 0), true);
      for (int i = 0; i < 2; i++)
        for (int jj = 0; jj < 2; jj++) simdgroup_multiply_accumulate(acc[i][jj], a[i], b[jj], acc[i][jj]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = 0; i < 2; i++)
    for (int jj = 0; jj < 2; jj++) simdgroup_store(acc[i][jj], Ys + (sr + 8u * i) * LD + sn + 8u * jj, LD);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((int)lr < count && n0 + seg < N_SIZE) {
    for (int i = 0; i < 8; i++) {
      if (n0 + seg + i < N_SIZE) {
        y[(ulong)(r0 + lr) * N_SIZE + n0 + seg + i] = static_cast<T>(Ys[lr * LD + seg + i]);
      }
    }
  }
"""


def kind_of(experts):
    """Return integer ggml type for `experts` (`mode == 'gguf:<name>'`), or `None`."""
    mode = getattr(experts, 'mode', None)
    if not isinstance(mode, str) or not mode.startswith('gguf:'):
        return None
    return iq_experts.BY_NAME.get(mode[5:].upper())


def _grid_for(kind):
    global _DUMMY_GRID
    grids = iq_experts.compact_grids()
    if kind in grids:
        return grids[kind]
    if _DUMMY_GRID is None:
        _DUMMY_GRID = mx.zeros((8,), dtype=mx.uint16)
        mx.eval(_DUMMY_GRID)
    return _DUMMY_GRID


def _gpu_available():
    return mx.default_device() == mx.gpu and (mx.metal.is_available() or cuda_kernels.available())


_CUDA_HEADER = r"""
template <typename P>
__device__ __forceinline__ uint16_t qz_u16_le(P p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
template <typename P>
__device__ __forceinline__ uint32_t qz_u32_le(P p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
      | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
template <typename P>
__device__ __forceinline__ float qz_f16_le(P p) {
  return __half2float(__ushort_as_half(qz_u16_le(p)));
}
__device__ __forceinline__ uint32_t qz_ksign_byte(uint32_t s7) {
  return s7 | ((static_cast<uint32_t>(__popc(s7)) & 1u) << 7);
}
__device__ __forceinline__ void qz_syncwarp() {
#ifdef __CUDACC__
  __syncwarp();
#endif
}
template <bool ALIGNED4, typename P>
__device__ __forceinline__ uint32_t qz_word4(P p) {
  if constexpr (ALIGNED4) {
#ifdef __CUDACC__
    return *reinterpret_cast<const uint32_t*>(p);
#else
    if ((reinterpret_cast<uintptr_t>(p) & 3u) != 0u) __builtin_trap();
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
#endif
  }
  return qz_u32_le(p);
}

template <int BB>
__device__ __forceinline__ void qz_load_bytes(uint8_t* dst, const uint8_t* src, uint32_t nbytes, uint32_t lane) {
  if constexpr ((BB % 16) == 0) {
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    const uint32_t n4 = nbytes >> 4;
    for (uint32_t i = lane; i < n4; i += 32u) d4[i] = s4[i];
  } else if constexpr ((BB % 8) == 0) {
    uint2* d2 = reinterpret_cast<uint2*>(dst);
    const uint2* s2 = reinterpret_cast<const uint2*>(src);
    const uint32_t n2 = nbytes >> 3;
    for (uint32_t i = lane; i < n2; i += 32u) d2[i] = s2[i];
  } else if constexpr ((BB % 4) == 0) {
    uint32_t* dw = reinterpret_cast<uint32_t*>(dst);
    const uint32_t* sw = reinterpret_cast<const uint32_t*>(src);
    const uint32_t nw = nbytes >> 2;
    for (uint32_t i = lane; i < nw; i += 32u) dw[i] = sw[i];
  } else if constexpr ((BB % 2) == 0) {
    uint16_t* dh = reinterpret_cast<uint16_t*>(dst);
    const uint16_t* sh = reinterpret_cast<const uint16_t*>(src);
    const uint32_t mis = (reinterpret_cast<uintptr_t>(src) >> 1) & 1u;
    if (mis == 0u) {
      uint32_t* dw = reinterpret_cast<uint32_t*>(dst);
      const uint32_t* sw = reinterpret_cast<const uint32_t*>(src);
      const uint32_t nw = nbytes >> 2;
      for (uint32_t i = lane; i < nw; i += 32u) dw[i] = sw[i];
      if ((nbytes & 2u) != 0u && lane == 0u) dh[(nbytes >> 1) - 1u] = sh[(nbytes >> 1) - 1u];
    } else if (nbytes >= 2u) {
      if (lane == 0u) dh[0] = sh[0];
      const uint32_t rem = nbytes - 2u;
      const uint32_t nw = rem >> 2;
      const uint32_t* sw = reinterpret_cast<const uint32_t*>(src + 2u);
      for (uint32_t i = lane; i < nw; i += 32u) {
        const uint32_t v = sw[i];
        dh[1u + 2u * i] = static_cast<uint16_t>(v);
        dh[2u + 2u * i] = static_cast<uint16_t>(v >> 16);
      }
      if ((rem & 2u) != 0u && lane == 0u) dh[(nbytes >> 1) - 1u] = sh[(nbytes >> 1) - 1u];
    }
  } else {
    for (uint32_t i = lane; i < nbytes; i += 32u) dst[i] = src[i];
  }
}

template <int KIND, bool ALIGNED4 = false, typename BlkPtr, typename GridPtr>
__device__ __forceinline__ void qz_decode_octad(BlkPtr blk, GridPtr grid, uint32_t octad, float* w8) {
  constexpr float IQ2_MAG[4] = {8.0f, 25.0f, 43.0f, 0.0f};
  constexpr float IQ3_MAG[8] = {4.0f, 12.0f, 20.0f, 28.0f, 36.0f, 44.0f, 52.0f, 62.0f};
  constexpr float IQ4NL_MAG[16] = {
      -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
      1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f
  };
  constexpr float MXFP4_MAG[16] = {
      0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
      0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
  };
  const uint32_t v0 = octad * 8u;
  if constexpr (KIND == 1) {                             // F16 (block 32)
    for (int i = 0; i < 8; ++i) w8[i] = qz_f16_le(blk + 2u * (v0 + i));
  } else if constexpr (KIND == 30) {                     // BF16 (block 32)
    for (int i = 0; i < 8; ++i) {
      w8[i] = __uint_as_float(static_cast<uint32_t>(qz_u16_le(blk + 2u * (v0 + i))) << 16);
    }
  } else if constexpr (KIND == 2) {                      // Q4_0 (block 32)
    const float d = qz_f16_le(blk);
    BlkPtr qs = blk + 2u + (octad < 2u ? v0 : v0 - 16u);
    const uint32_t shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; ++i) w8[i] = d * static_cast<float>(static_cast<int>((qs[i] >> shift) & 15u) - 8);
  } else if constexpr (KIND == 3) {                      // Q4_1 (block 32)
    const float d = qz_f16_le(blk), m = qz_f16_le(blk + 2u);
    BlkPtr qs = blk + 4u + (octad < 2u ? v0 : v0 - 16u);
    const uint32_t shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; ++i) w8[i] = d * static_cast<float>((qs[i] >> shift) & 15u) + m;
  } else if constexpr (KIND == 6) {                      // Q5_0 (block 32)
    const float d = qz_f16_le(blk);
    const uint32_t qhb = blk[2u + octad];
    BlkPtr qs = blk + 6u + (octad < 2u ? v0 : v0 - 16u);
    const uint32_t shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; ++i) {
      const uint32_t q = ((qs[i] >> shift) & 15u) | (((qhb >> i) & 1u) << 4);
      w8[i] = d * static_cast<float>(static_cast<int>(q) - 16);
    }
  } else if constexpr (KIND == 7) {                      // Q5_1 (block 32)
    const float d = qz_f16_le(blk), m = qz_f16_le(blk + 2u);
    const uint32_t qhb = blk[4u + octad];
    BlkPtr qs = blk + 8u + (octad < 2u ? v0 : v0 - 16u);
    const uint32_t shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; ++i) {
      const uint32_t q = ((qs[i] >> shift) & 15u) | (((qhb >> i) & 1u) << 4);
      w8[i] = d * static_cast<float>(q) + m;
    }
  } else if constexpr (KIND == 8) {                      // Q8_0 (block 32)
    const float d = qz_f16_le(blk);
    for (int i = 0; i < 8; ++i) w8[i] = d * static_cast<float>(static_cast<int8_t>(blk[2u + v0 + i]));
  } else if constexpr (KIND == 20) {                     // IQ4_NL (block 32)
    const float d = qz_f16_le(blk);
    BlkPtr qs = blk + 2u + (octad < 2u ? v0 : v0 - 16u);
    const uint32_t shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; ++i) w8[i] = d * IQ4NL_MAG[(qs[i] >> shift) & 15u];
  } else if constexpr (KIND == 39) {                     // MXFP4 (block 32, 17B)
    const uint32_t e = blk[0];
    const float d = __uint_as_float(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
    BlkPtr qs = blk + 1u + (octad < 2u ? v0 : v0 - 16u);
    const uint32_t shift = octad < 2u ? 0u : 4u;
    for (int i = 0; i < 8; ++i) w8[i] = d * MXFP4_MAG[(qs[i] >> shift) & 15u];
  } else if constexpr (KIND == 42) {                     // Q2_0 (block 64)
    const float d = qz_f16_le(blk);
    BlkPtr qs = blk + 2u + 2u * octad;
    for (int i = 0; i < 8; ++i) {
      const uint32_t q = (qs[i >> 2] >> (2u * (i & 3))) & 3u;
      w8[i] = d * static_cast<float>(static_cast<int>(q) - 1);
    }
  } else if constexpr (KIND == 10) {                     // Q2_K (block 256)
    const uint32_t sub = octad >> 1, h = v0 >> 7, j = (v0 & 127u) >> 5, l0 = v0 & 31u;
    const uint32_t s = blk[sub];
    const float dl = qz_f16_le(blk + 80u) * static_cast<float>(s & 15u);
    const float ml = qz_f16_le(blk + 82u) * static_cast<float>(s >> 4);
    BlkPtr qs = blk + 16u + 32u * h + l0;
    const uint32_t q0 = qz_word4<ALIGNED4>(qs);
    const uint32_t q1 = qz_word4<ALIGNED4>(qs + 4u);
    const uint32_t sh = 2u * j;
    for (int i = 0; i < 4; ++i) {
      w8[i] = dl * static_cast<float>((q0 >> (sh + 8u * i)) & 3u) - ml;
      w8[i + 4] = dl * static_cast<float>((q1 >> (sh + 8u * i)) & 3u) - ml;
    }
  } else if constexpr (KIND == 11) {                     // Q3_K (block 256)
    const uint32_t sub = octad >> 1, h = v0 >> 7, j = (v0 & 127u) >> 5, l0 = v0 & 31u, hb = v0 >> 5;
    const uint32_t low = sub < 8u ? (blk[96u + sub] & 15u) : (blk[96u + sub - 8u] >> 4);
    const uint32_t high = (blk[104u + (sub & 3u)] >> (2u * (sub >> 2))) & 3u;
    const float dl = qz_f16_le(blk + 108u) * static_cast<float>(static_cast<int>(low | (high << 4)) - 32);
    BlkPtr qs = blk + 32u + 32u * h + l0;
    const uint32_t q0 = qz_word4<ALIGNED4>(qs);
    const uint32_t q1 = qz_word4<ALIGNED4>(qs + 4u);
    const uint32_t h0 = qz_word4<ALIGNED4>(blk + l0);
    const uint32_t h1 = qz_word4<ALIGNED4>(blk + l0 + 4u);
    const uint32_t sh = 2u * j;
    for (int i = 0; i < 4; ++i) {
      int qa = static_cast<int>((q0 >> (sh + 8u * i)) & 3u);
      if (((h0 >> (hb + 8u * i)) & 1u) == 0u) qa -= 4;
      w8[i] = dl * static_cast<float>(qa);
      int qb = static_cast<int>((q1 >> (sh + 8u * i)) & 3u);
      if (((h1 >> (hb + 8u * i)) & 1u) == 0u) qb -= 4;
      w8[i + 4] = dl * static_cast<float>(qb);
    }
  } else if constexpr (KIND == 12 || KIND == 13) {       // Q4_K / Q5_K (block 256)
    const uint32_t sb = octad >> 2, l0 = v0 & 31u;
    BlkPtr sp = blk + 4u;
    uint32_t sc, mn;
    if (sb < 4u) {
      sc = sp[sb] & 63u;
      mn = sp[sb + 4u] & 63u;
    } else {
      sc = (sp[sb + 4u] & 15u) | ((sp[sb - 4u] >> 2) & 48u);
      mn = (sp[sb + 4u] >> 4) | ((sp[sb] >> 2) & 48u);
    }
    const float dd = qz_f16_le(blk) * static_cast<float>(sc);
    const float dm = qz_f16_le(blk + 2u) * static_cast<float>(mn);
    BlkPtr qs = blk + (KIND == 12 ? 16u : 48u) + 32u * (sb >> 1) + l0;
    const uint32_t shift = 4u * (sb & 1u);
    const uint32_t q0 = qz_word4<ALIGNED4>(qs);
    const uint32_t q1 = qz_word4<ALIGNED4>(qs + 4u);
    uint32_t h0 = 0u, h1 = 0u;
    if constexpr (KIND == 13) {
      h0 = qz_word4<ALIGNED4>(blk + 16u + l0);
      h1 = qz_word4<ALIGNED4>(blk + 20u + l0);
    }
    for (int i = 0; i < 4; ++i) {
      uint32_t qa = (q0 >> (shift + 8u * i)) & 15u;
      uint32_t qb = (q1 >> (shift + 8u * i)) & 15u;
      if constexpr (KIND == 13) {
        qa |= ((h0 >> (sb + 8u * i)) & 1u) << 4;
        qb |= ((h1 >> (sb + 8u * i)) & 1u) << 4;
      }
      w8[i] = dd * static_cast<float>(qa) - dm;
      w8[i + 4] = dd * static_cast<float>(qb) - dm;
    }
  } else if constexpr (KIND == 14) {                     // Q6_K (block 256)
    const uint32_t sub = octad >> 1, h = v0 >> 7, rem = v0 & 127u;
    const float dsc = qz_f16_le(blk + 208u) * static_cast<float>(static_cast<int8_t>(blk[192u + sub]));
    const uint32_t ql_shift = 4u * (rem >> 6), qh_shift = 2u * (rem >> 5);
    BlkPtr qlp = blk + 64u * h + (rem & 63u);
    BlkPtr qhp = blk + 128u + 32u * h + (rem & 31u);
    const uint32_t ql0 = qz_word4<ALIGNED4>(qlp);
    const uint32_t ql1 = qz_word4<ALIGNED4>(qlp + 4u);
    const uint32_t qh0 = qz_word4<ALIGNED4>(qhp);
    const uint32_t qh1 = qz_word4<ALIGNED4>(qhp + 4u);
    for (int i = 0; i < 4; ++i) {
      const uint32_t qa = ((ql0 >> (ql_shift + 8u * i)) & 15u) | (((qh0 >> (qh_shift + 8u * i)) & 3u) << 4);
      const uint32_t qb = ((ql1 >> (ql_shift + 8u * i)) & 15u) | (((qh1 >> (qh_shift + 8u * i)) & 3u) << 4);
      w8[i] = dsc * static_cast<float>(static_cast<int>(qa) - 32);
      w8[i + 4] = dsc * static_cast<float>(static_cast<int>(qb) - 32);
    }
  } else if constexpr (KIND == 16) {                     // IQ2_XXS (block 256)
    const uint32_t ib32 = octad >> 2, l = octad & 3u;
    const float d = qz_f16_le(blk);
    const uint32_t aux = qz_u32_le(blk + 2u + 8u * ib32 + 4u);
    const float db = d * (0.5f + static_cast<float>(aux >> 28)) * 0.25f;
    const uint32_t sbyte = qz_ksign_byte((aux >> (7u * l)) & 127u);
    const uint16_t g = grid[blk[2u + 8u * ib32 + l]];
    for (int i = 0; i < 8; ++i) {
      const float sign = ((sbyte >> i) & 1u) ? -1.0f : 1.0f;
      w8[i] = db * IQ2_MAG[(g >> (2u * i)) & 3u] * sign;
    }
  } else if constexpr (KIND == 17) {                     // IQ2_XS (block 256)
    const uint32_t ib32 = octad >> 2, l = octad & 3u;
    const float d = qz_f16_le(blk);
    const uint32_t q16 = qz_u16_le(blk + 2u + 2u * octad);
    const uint32_t s = blk[66u + ib32];
    const float db = d * (0.5f + static_cast<float>(l < 2u ? (s & 15u) : (s >> 4))) * 0.25f;
    const uint32_t sbyte = qz_ksign_byte(q16 >> 9);
    const uint16_t g = grid[q16 & 511u];
    for (int i = 0; i < 8; ++i) {
      const float sign = ((sbyte >> i) & 1u) ? -1.0f : 1.0f;
      w8[i] = db * IQ2_MAG[(g >> (2u * i)) & 3u] * sign;
    }
  } else if constexpr (KIND == 22) {                     // IQ2_S (block 256)
    const uint32_t ib32 = octad >> 2, l = octad & 3u;
    const float d = qz_f16_le(blk);
    const uint32_t idx10 = static_cast<uint32_t>(blk[2u + octad])
        | (((static_cast<uint32_t>(blk[66u + ib32]) >> (2u * l)) & 3u) << 8);
    const uint32_t sbyte = blk[34u + octad];
    const uint32_t s = blk[74u + ib32];
    const float db = d * (0.5f + static_cast<float>(l < 2u ? (s & 15u) : (s >> 4))) * 0.25f;
    const uint16_t g = grid[idx10];
    for (int i = 0; i < 8; ++i) {
      const float sign = ((sbyte >> i) & 1u) ? -1.0f : 1.0f;
      w8[i] = db * IQ2_MAG[(g >> (2u * i)) & 3u] * sign;
    }
  } else if constexpr (KIND == 19) {                     // IQ1_S (block 256)
    const uint32_t ib32 = octad >> 2, l = octad & 3u;
    const float d = qz_f16_le(blk);
    const uint32_t qh = qz_u16_le(blk + 34u + 2u * ib32);
    const float dl = d * static_cast<float>(2u * ((qh >> 12) & 7u) + 1u);
    const float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
    const uint32_t idx11 = static_cast<uint32_t>(blk[2u + octad]) | (((qh >> (3u * l)) & 7u) << 8);
    const uint16_t g = grid[idx11];
    for (int i = 0; i < 8; ++i) {
      w8[i] = dl * (static_cast<float>(static_cast<int>((g >> (2u * i)) & 3u) - 1) + delta);
    }
  } else if constexpr (KIND == 29) {                     // IQ1_M (block 256)
    const uint32_t ib16 = octad >> 1;
    const uint32_t qhb = static_cast<uint32_t>(blk[32u + ib16]) >> (4u * (octad & 1u));
    const uint32_t sc0 = qz_u16_le(blk + 48u), sc1 = qz_u16_le(blk + 50u);
    const uint32_t sc2 = qz_u16_le(blk + 52u), sc3 = qz_u16_le(blk + 54u);
    const uint16_t d_bits = static_cast<uint16_t>(
        ((sc0 >> 12) & 0x000fu) | ((sc1 >> 8) & 0x00f0u)
        | ((sc2 >> 4) & 0x0f00u) | (sc3 & 0xf000u));
    const float d = __half2float(__ushort_as_half(d_bits));
    const uint32_t sc_word = qz_u16_le(blk + 48u + 2u * (ib16 >> 2));
    const float dl = d * static_cast<float>(2u * ((sc_word >> (3u * (ib16 & 3u))) & 7u) + 1u);
    const float delta = (qhb & 8u) ? -0.125f : 0.125f;
    const uint16_t g = grid[static_cast<uint32_t>(blk[octad]) | ((qhb & 7u) << 8)];
    for (int i = 0; i < 8; ++i) {
      w8[i] = dl * (static_cast<float>(static_cast<int>((g >> (2u * i)) & 3u) - 1) + delta);
    }
  } else if constexpr (KIND == 18) {                     // IQ3_XXS (block 256)
    const uint32_t ib32 = octad >> 2, l = octad & 3u;
    const float d = qz_f16_le(blk);
    const uint32_t aux = qz_u32_le(blk + 66u + 4u * ib32);
    const float db = d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
    const uint32_t sbyte = qz_ksign_byte((aux >> (7u * l)) & 127u);
    const uint16_t g0 = grid[blk[2u + 8u * ib32 + 2u * l]];
    const uint16_t g1 = grid[blk[2u + 8u * ib32 + 2u * l + 1u]];
    for (int i = 0; i < 4; ++i) {
      const float s0 = ((sbyte >> i) & 1u) ? -1.0f : 1.0f;
      const float s1 = ((sbyte >> (i + 4)) & 1u) ? -1.0f : 1.0f;
      w8[i] = db * IQ3_MAG[(g0 >> (4u * i)) & 7u] * s0;
      w8[i + 4] = db * IQ3_MAG[(g1 >> (4u * i)) & 7u] * s1;
    }
  } else if constexpr (KIND == 23) {                     // IQ4_XS (block 256)
    const uint32_t ib32 = octad >> 2, l = octad & 3u;
    const float d = qz_f16_le(blk);
    const uint32_t high = qz_u16_le(blk + 2u);
    const uint32_t low_b = blk[4u + (ib32 >> 1)];
    const uint32_t low = (ib32 & 1u) ? (low_b >> 4) : (low_b & 15u);
    const uint32_t ls = low | (((high >> (2u * ib32)) & 3u) << 4);
    const float dl = d * static_cast<float>(static_cast<int>(ls) - 32);
    BlkPtr qs = blk + 8u + 16u * ib32 + 8u * (l & 1u);
    const uint32_t shift = l < 2u ? 0u : 4u;
    const uint32_t q0 = qz_word4<ALIGNED4>(qs);
    const uint32_t q1 = qz_word4<ALIGNED4>(qs + 4u);
    for (int i = 0; i < 4; ++i) {
      w8[i] = dl * IQ4NL_MAG[(q0 >> (shift + 8u * i)) & 15u];
      w8[i + 4] = dl * IQ4NL_MAG[(q1 >> (shift + 8u * i)) & 15u];
    }
  }
}

template <int KIND, int PB, int BB, int K_BLOCKS, int ROWS, int SBB>
__device__ __forceinline__ void qz_accum_iter(
    float (&acc)[ROWS], const float (&xv)[8], uint8_t (&ws_warp)[ROWS][SBB],
    const uint8_t* w, const uint16_t* grid, int e, uint32_t first, uint32_t n_size,
    uint32_t k0, uint32_t rem_k, uint32_t lane) {
  const uint32_t b0 = k0 / PB;
  const uint32_t rem_bytes = (rem_k / PB) * BB;
  constexpr uint32_t FULL_BYTES = (256u / PB) * BB;
  if constexpr ((BB % 16) == 0 && (FULL_BYTES >> 4) <= 16u && (ROWS % 2) == 0) {
    constexpr uint32_t N4 = FULL_BYTES >> 4;
    if (rem_k == 256u && first + ROWS <= n_size) {
      const uint32_t r_sub = lane / N4;
      const uint32_t c4 = lane % N4;
      for (int base_r = 0; base_r < ROWS; base_r += 2) {
        if (r_sub < 2u) {
          const size_t at = (static_cast<size_t>(e) * n_size + first + base_r + r_sub) * K_BLOCKS + b0;
          reinterpret_cast<uint4*>(ws_warp[base_r + r_sub])[c4] =
              reinterpret_cast<const uint4*>(w + at * BB)[c4];
        }
      }
    } else {
      for (int row = 0; row < ROWS; ++row) {
        if (first + row >= n_size) break;
        const size_t at = (static_cast<size_t>(e) * n_size + first + row) * K_BLOCKS + b0;
        qz_load_bytes<BB>(ws_warp[row], w + at * BB, rem_bytes, lane);
      }
    }
  } else {
    for (int row = 0; row < ROWS; ++row) {
      if (first + row >= n_size) break;
      const size_t at = (static_cast<size_t>(e) * n_size + first + row) * K_BLOCKS + b0;
      qz_load_bytes<BB>(ws_warp[row], w + at * BB, rem_bytes, lane);
    }
  }
  qz_syncwarp();
  const uint32_t local_v = lane * 8u;
  if (local_v < rem_k) {
    const uint32_t b_local = local_v / PB;
    const uint32_t octad = (local_v % PB) / 8u;
    constexpr bool ALIGNED4 = (PB == 256) || ((BB % 4) == 0);
    for (int row = 0; row < ROWS; ++row) {
      if (first + row >= n_size) break;
      float w8[8];
      qz_decode_octad<KIND, ALIGNED4>(ws_warp[row] + b_local * BB, grid, octad, w8);
      float dot = 0.0f;
      for (int i = 0; i < 8; ++i) dot += xv[i] * w8[i];
      acc[row] += dot;
    }
  }
  qz_syncwarp();
}
"""

_CUDA_SOURCE = r"""
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  const uint32_t lane = warp.thread_rank();
  const uint32_t wid = warp.meta_group_rank();
  const uint32_t tid = threadIdx.x;
  const uint32_t first = (blockIdx.x * WARPS + wid) * ROWS;
  const uint32_t r = blockIdx.y;
  const int e = static_cast<int>(idx[r]);
  const T* xr = x + static_cast<size_t>(r / X_DIV) * K_SIZE;
  constexpr uint32_t SBB = (((256u / PB) * BB + 15u) / 16u) * 16u;
  __shared__ __align__(16) float xs[256];
  __shared__ __align__(16) uint8_t ws[WARPS][ROWS][SBB];
  float acc[ROWS];
  for (int row = 0; row < ROWS; ++row) acc[row] = 0.0f;
  for (uint32_t k0 = 0; k0 < K_SIZE; k0 += 256u) {
    for (uint32_t i = tid; i < 256u; i += 32u * WARPS) {
      xs[i] = (k0 + i < K_SIZE) ? static_cast<float>(xr[k0 + i]) : 0.0f;
    }
    __syncthreads();
    if (first < N_SIZE) {
      const uint32_t rem_k = (k0 + 256u <= K_SIZE) ? 256u : (K_SIZE - k0);
      float xv[8];
      const uint32_t v0 = lane * 8u;
      if (v0 < rem_k) {
        for (int i = 0; i < 8; ++i) xv[i] = xs[v0 + i];
      } else {
        for (int i = 0; i < 8; ++i) xv[i] = 0.0f;
      }
      qz_accum_iter<KIND, PB, BB, K_BLOCKS, ROWS, SBB>(
          acc, xv, ws[wid], w, grid, e, first, N_SIZE, k0, rem_k, lane);
    }
    __syncthreads();
  }
  for (int row = 0; row < ROWS; ++row) {
    const uint32_t n = first + row;
    const float total = cg::reduce(warp, acc[row], QzSum{});
    if (lane == 0u && n < N_SIZE) y[static_cast<size_t>(r) * N_SIZE + n] = static_cast<T>(total);
  }
"""

_CUDA_GATE_UP = r"""
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  const uint32_t lane = warp.thread_rank();
  const uint32_t wid = warp.meta_group_rank();
  const uint32_t tid = threadIdx.x;
  const uint32_t m = wid >> 1;
  const uint32_t sub = wid & 1u;
  const uint32_t base_n = blockIdx.x * (2u * ROWS);
  const uint32_t first = base_n + sub * ROWS;
  const uint32_t r = blockIdx.y;
  const int e = static_cast<int>(idx[r]);
  const T* xr = x + static_cast<size_t>(r / X_DIV) * K_SIZE;
  constexpr uint32_t SBB_G = (((256u / PB_G) * BB_G + 15u) / 16u) * 16u;
  constexpr uint32_t SBB_U = (((256u / PB_U) * BB_U + 15u) / 16u) * 16u;
  constexpr uint32_t SBB = (SBB_G > SBB_U) ? SBB_G : SBB_U;
  __shared__ __align__(16) float xs[256];
  __shared__ __align__(16) uint8_t ws[4][ROWS][SBB];
  __shared__ __align__(16) float gu[2][2 * ROWS];
  float acc[ROWS];
  for (int row = 0; row < ROWS; ++row) acc[row] = 0.0f;
  for (uint32_t k0 = 0; k0 < K_SIZE; k0 += 256u) {
    for (uint32_t i = tid; i < 256u; i += 128u) {
      xs[i] = (k0 + i < K_SIZE) ? static_cast<float>(xr[k0 + i]) : 0.0f;
    }
    __syncthreads();
    if (first < N_SIZE) {
      const uint32_t rem_k = (k0 + 256u <= K_SIZE) ? 256u : (K_SIZE - k0);
      float xv[8];
      const uint32_t v0 = lane * 8u;
      if (v0 < rem_k) {
        for (int i = 0; i < 8; ++i) xv[i] = xs[v0 + i];
      } else {
        for (int i = 0; i < 8; ++i) xv[i] = 0.0f;
      }
      if (m == 0u) {
        qz_accum_iter<KIND_G, PB_G, BB_G, KB_G, ROWS, SBB>(
            acc, xv, ws[wid], wg, grid_g, e, first, N_SIZE, k0, rem_k, lane);
      } else {
        qz_accum_iter<KIND_U, PB_U, BB_U, KB_U, ROWS, SBB>(
            acc, xv, ws[wid], wu, grid_u, e, first, N_SIZE, k0, rem_k, lane);
      }
    }
    __syncthreads();
  }
  for (int row = 0; row < ROWS; ++row) {
    const uint32_t n = first + row;
    const float total = cg::reduce(warp, acc[row], QzSum{});
    if (lane == 0u && n < N_SIZE) gu[m][sub * ROWS + row] = total;
  }
  __syncthreads();
  if (tid < 2u * ROWS) {
    const uint32_t n = base_n + tid;
    if (n < N_SIZE) {
      const T gate_v = static_cast<T>(gu[0][tid]);
      const T up_v = static_cast<T>(gu[1][tid]);
      y[static_cast<size_t>(r) * N_SIZE + n] = Multiply{}(Multiply{}(gate_v, Sigmoid{}(gate_v)), up_v);
    }
  }
"""

_CUDA_TILED = r"""
  constexpr int BM = 32, BN = 32, BK = 32, LD = 36;
  constexpr uint32_t SBB1 = ((BB + 15u) / 16u) * 16u;
  __shared__ __align__(16) float Xs[BM * LD];
  __shared__ __align__(16) float Ws[BK * LD];
  __shared__ __align__(16) uint8_t Wraw[BN][SBB1];
  const uint32_t tid = threadIdx.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lane = tid & 31u;
  const uint32_t n0 = blockIdx.x * BN;
  const uint32_t tile = blockIdx.y;
  const int e = tiles[tile * 3], r0 = tiles[tile * 3 + 1], count = tiles[tile * 3 + 2];
  const uint32_t lr = tid / 4u, seg = (tid % 4u) * 8u;
  const uint32_t tm = (tid / 8u) * 2u, tn = (tid % 8u) * 4u;
  float acc[2][4];
  for (int rm = 0; rm < 2; ++rm)
    for (int rn = 0; rn < 4; ++rn) acc[rm][rn] = 0.0f;
  for (uint32_t k0 = 0; k0 < K_SIZE; k0 += BK) {
    if ((k0 % PB) == 0u) {
      const uint32_t b = k0 / PB;
      for (uint32_t r = wid * 8u; r < wid * 8u + 8u; ++r) {
        if (n0 + r < N_SIZE) {
          const size_t at = (static_cast<size_t>(e) * N_SIZE + n0 + r) * K_BLOCKS + b;
          qz_load_bytes<BB>(Wraw[r], w + at * BB, BB, lane);
        }
      }
      __syncthreads();
    }
    for (int i = 0; i < 8; ++i) {
      Xs[lr * LD + seg + i] = (static_cast<int>(lr) < count)
          ? static_cast<float>(x[static_cast<size_t>(r0 + lr) * K_SIZE + k0 + seg + i])
          : 0.0f;
    }
    if (n0 + lr < N_SIZE) {
      const uint32_t octad = ((k0 + seg) % PB) / 8u;
      float w8[8];
      qz_decode_octad<KIND, true>(Wraw[lr], grid, octad, w8);
      for (int i = 0; i < 8; ++i) Ws[(seg + i) * LD + lr] = w8[i];
    } else {
      for (int i = 0; i < 8; ++i) Ws[(seg + i) * LD + lr] = 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < BK; ++kk) {
      const float a0 = Xs[(tm + 0u) * LD + kk];
      const float a1 = Xs[(tm + 1u) * LD + kk];
      const float4 bv = *reinterpret_cast<const float4*>(&Ws[kk * LD + tn]);
      acc[0][0] += a0 * bv.x; acc[0][1] += a0 * bv.y; acc[0][2] += a0 * bv.z; acc[0][3] += a0 * bv.w;
      acc[1][0] += a1 * bv.x; acc[1][1] += a1 * bv.y; acc[1][2] += a1 * bv.z; acc[1][3] += a1 * bv.w;
    }
    __syncthreads();
  }
  for (int rm = 0; rm < 2; ++rm) {
    if (static_cast<int>(tm + rm) < count) {
      const size_t y_off = static_cast<size_t>(r0 + tm + rm) * N_SIZE + n0 + tn;
      for (int rn = 0; rn < 4; ++rn) {
        if (n0 + tn + rn < N_SIZE) {
          y[y_off + rn] = static_cast<T>(acc[rm][rn]);
        }
      }
    }
  }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name='quanteeze_gguf_qmv',
        input_names=['x', 'w', 'grid', 'idx'],
        output_names=['y'],
        header=_HEADER,
        source=_SOURCE,
    )


@lru_cache(maxsize=None)
def _tiled():
    return mx.fast.metal_kernel(
        name='quanteeze_gguf_mm',
        input_names=['x', 'w', 'grid', 'tiles'],
        output_names=['y'],
        header='#include <metal_simdgroup_matrix>\n' + _HEADER,
        source=_TILED,
    )


@lru_cache(maxsize=None)
def _cuda_kernel():
    return cuda_kernels.kernel(
        name='quanteeze_gguf_qmv_cuda',
        input_names=['x', 'w', 'grid', 'idx'],
        output_names=['y'],
        header=_CUDA_HEADER,
        source=_CUDA_SOURCE,
    )


@lru_cache(maxsize=None)
def _cuda_tiled():
    return cuda_kernels.kernel(
        name='quanteeze_gguf_mm_cuda',
        input_names=['x', 'w', 'grid', 'tiles'],
        output_names=['y'],
        header=_CUDA_HEADER,
        source=_CUDA_TILED,
    )


@lru_cache(maxsize=None)
def _cuda_gate_up():
    return cuda_kernels.kernel(
        name='quanteeze_gguf_gate_up_cuda',
        input_names=['x', 'wg', 'grid_g', 'wu', 'grid_u', 'idx'],
        output_names=['y'],
        header=_CUDA_HEADER,
        source=_CUDA_GATE_UP,
    )


def takes(experts, x, indices):
    """Whether this call is a single-row decode/verify call that a Metal or CUDA GGUF qmv kernel serves."""
    if not ON[0] or not _gpu_available():
        return False
    kind = kind_of(experts)
    if kind is None or experts.weight.ndim != 4 or experts.weight.dtype != mx.uint8:
        return False
    pb = iq_experts.per_block(kind)
    if x.ndim < 2 or x.shape[-2] != 1 or x.shape[-1] % pb != 0 or x.shape[-1] % 32 != 0:
        return False
    if x.ndim == 3 and x.shape[1] == 1 and indices.ndim == 1 and x.shape[0] == indices.size >= 16 and x.shape[-1] % TILE == 0:
        return False
    try:
        rows = int(np.prod(np.broadcast_shapes(tuple(x.shape[:-2]), tuple(indices.shape))))
    except ValueError:
        return False
    return 0 < rows <= MAX_ROWS


def run(experts, x, indices):
    """`x [..., 1, K]` through raw GGUF `experts` named by `indices`: `[..., 1, N]`."""
    kind = kind_of(experts)
    pb, bb = iq_experts.per_block(kind), iq_experts.BLOCK[kind]
    shape = tuple(np.broadcast_shapes(tuple(x.shape[:-2]), tuple(indices.shape)))
    k, n = x.shape[-1], experts.weight.shape[1]
    rows = int(np.prod(shape))
    template = [
        ('T', x.dtype), ('K_SIZE', k), ('K_BLOCKS', k // pb), ('N_SIZE', n),
        ('KIND', kind), ('PB', pb), ('BB', bb), ('ROWS', ROWS),
    ]
    if cuda_kernels.available() and not mx.metal.is_available():
        x_bshape = tuple(x.shape[:-2])
        x_rows = max(1, x.size // k)
        if x.size == rows * k:
            xs, x_div = mx.contiguous(x.reshape(rows, k)), 1
        elif x.size == k:
            xs, x_div = mx.contiguous(x.reshape(1, k)), rows
        elif len(x_bshape) == len(shape) and x_bshape[:-1] == shape[:-1] and x_bshape[-1] == 1 and rows % x_rows == 0:
            x_div = rows // x_rows
            xs = mx.contiguous(x.reshape(x_rows, k))
        else:
            xs, x_div = mx.contiguous(mx.broadcast_to(x, shape + (1, k)).reshape(rows, k)), 1
        if tuple(indices.shape) == shape and indices.dtype in (mx.int32, mx.uint32):
            idx = mx.contiguous(indices.reshape(rows))
        else:
            idx = mx.broadcast_to(indices, shape).reshape(rows).astype(mx.int32)
        if rows < 4:
            idx = mx.pad(idx, (0, 4 - rows))
        n_blocks = (n + WARPS * ROWS - 1) // (WARPS * ROWS)
        out_elems = cuda_kernels.roomy(rows * n, x.dtype)
        y = _cuda_kernel()(
            inputs=[xs, mx.contiguous(experts.weight), _grid_for(kind), idx],
            template=template + [('WARPS', WARPS), ('X_DIV', x_div)],
            grid=(32 * WARPS * n_blocks, rows, 1),
            threadgroup=(32 * WARPS, 1, 1),
            output_shapes=[(out_elems,)],
            output_dtypes=[x.dtype],
        )[0]
        return y[:rows * n].reshape(*shape, 1, n)
    xs = mx.contiguous(mx.broadcast_to(x, shape + (1, k)).reshape(rows, k))
    idx = mx.broadcast_to(indices, shape).reshape(rows).astype(mx.int32)
    y = _kernel()(
        inputs=[xs, mx.contiguous(experts.weight), _grid_for(kind), idx],
        template=template,
        grid=(32, (n + ROWS - 1) // ROWS, rows),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]
    return y.reshape(*shape, 1, n)


def gather(experts, x, indices, unpacked):
    """`run()` once verified within `TOLERANCE` of `unpacked()` for `CHECKS` calls, else `unpacked()`."""
    kind = kind_of(experts)
    key = (kind, tuple(x.shape), x.dtype, tuple(indices.shape), tuple(experts.weight.shape))
    return _checked(key, lambda: run(experts, x, indices), unpacked)


def takes_switch(switch, x, indices):
    """Whether `switch` can use the fused CUDA GGUF `gate_up + SwiGLU` + `down` decode path."""
    if not ON[0] or mx.metal.is_available() or mx.default_device() != mx.gpu or not cuda_kernels.available():
        return False
    if x.ndim != 3 or x.shape[1] != 1 or not 1 <= x.shape[0] <= 8:
        return False
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if indices.ndim != 3 or indices.shape[:2] != x.shape[:2] or not 0 < indices.size <= MAX_ROWS:
        return False
    if not all(hasattr(switch, a) for a in ('gate_proj', 'up_proj', 'down_proj', 'activation')):
        return False
    if type(switch.activation).__name__ != 'SwiGLU':
        return False
    gate, up, dn = switch.gate_proj, switch.up_proj, switch.down_proj
    if any('bias' in p for p in (gate, up, dn)):
        return False
    kg, ku, kd = kind_of(gate), kind_of(up), kind_of(dn)
    if kg is None or ku is None or kd is None:
        return False
    if any(p.weight.ndim != 4 or p.weight.dtype != mx.uint8 for p in (gate, up, dn)):
        return False
    hidden = x.shape[-1]
    inner = gate.weight.shape[1]
    if gate.weight.shape[:2] != up.weight.shape[:2] or dn.weight.shape[0] != gate.weight.shape[0] or dn.weight.shape[1] != hidden:
        return False
    pbg, pbu, pbd = iq_experts.per_block(kg), iq_experts.per_block(ku), iq_experts.per_block(kd)
    return (
        hidden % pbg == 0 and hidden % pbu == 0 and hidden % 32 == 0
        and gate.weight.shape[2] * pbg == hidden and up.weight.shape[2] * pbu == hidden
        and inner % pbd == 0 and inner % 32 == 0 and dn.weight.shape[2] * pbd == inner
    )


def run_switch(switch, x, indices):
    """Fused CUDA GGUF `gate_up + SwiGLU` + `down_proj` for `[B, 1, K]` and `indices [B, 1, top_k]`."""
    gate, up, dn = switch.gate_proj, switch.up_proj, switch.down_proj
    kg, ku, kd = kind_of(gate), kind_of(up), kind_of(dn)
    pbg, bbg = iq_experts.per_block(kg), iq_experts.BLOCK[kg]
    pbu, bbu = iq_experts.per_block(ku), iq_experts.BLOCK[ku]
    pbd, bbd = iq_experts.per_block(kd), iq_experts.BLOCK[kd]
    b, top_k, hidden = x.shape[0], indices.shape[-1], x.shape[-1]
    slots = b * top_k
    inner = gate.weight.shape[1]
    xs = mx.contiguous(x.reshape(b, hidden))
    if indices.dtype in (mx.int32, mx.uint32):
        idx = mx.contiguous(indices.reshape(slots))
    else:
        idx = mx.contiguous(indices.reshape(slots).astype(mx.int32))
    if slots < 4:
        idx = mx.pad(idx, (0, 4 - slots))
    n_blocks_gu = (inner + 2 * ROWS - 1) // (2 * ROWS)
    out_gu = cuda_kernels.roomy(slots * inner, x.dtype)
    h = _cuda_gate_up()(
        inputs=[xs, mx.contiguous(gate.weight), _grid_for(kg), mx.contiguous(up.weight), _grid_for(ku), idx],
        template=[
            ('T', x.dtype), ('K_SIZE', hidden), ('N_SIZE', inner), ('X_DIV', top_k),
            ('KIND_G', kg), ('PB_G', pbg), ('BB_G', bbg), ('KB_G', hidden // pbg),
            ('KIND_U', ku), ('PB_U', pbu), ('BB_U', bbu), ('KB_U', hidden // pbu),
            ('ROWS', ROWS),
        ],
        grid=(128 * n_blocks_gu, slots, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(out_gu,)],
        output_dtypes=[x.dtype],
    )[0][:slots * inner].reshape(slots, inner)
    n_blocks_d = (hidden + WARPS * ROWS - 1) // (WARPS * ROWS)
    out_d = cuda_kernels.roomy(slots * hidden, x.dtype)
    y = _cuda_kernel()(
        inputs=[h, mx.contiguous(dn.weight), _grid_for(kd), idx],
        template=[
            ('T', x.dtype), ('K_SIZE', inner), ('K_BLOCKS', inner // pbd), ('N_SIZE', hidden),
            ('KIND', kd), ('PB', pbd), ('BB', bbd), ('ROWS', ROWS), ('WARPS', WARPS), ('X_DIV', 1),
        ],
        grid=(32 * WARPS * n_blocks_d, slots, 1),
        threadgroup=(32 * WARPS, 1, 1),
        output_shapes=[(out_d,)],
        output_dtypes=[x.dtype],
    )[0]
    return y[:slots * hidden].reshape(b, 1, top_k, hidden)


def decode_switch(switch, x, indices, fallback):
    """Fused CUDA GGUF `SwitchGLU` once verified within `TOLERANCE` of `fallback()` for `CHECKS` calls."""
    gate, up, dn = switch.gate_proj, switch.up_proj, switch.down_proj
    key = ('switch', kind_of(gate), kind_of(up), kind_of(dn), tuple(x.shape), x.dtype, tuple(indices.shape), gate.weight.shape[1])
    return _checked(key, lambda: run_switch(switch, x, indices), fallback)


def _dev_array(np_arr):
    arr = mx.array(np_arr)
    if cuda_kernels.available() and not mx.metal.is_available() and mx.default_device() == mx.gpu:
        arr = arr + mx.zeros_like(arr)
    mx.eval(arr)
    return arr


def prepare_rows(indices):
    """Build `(order_mx, inv_order_mx, idx_sorted_mx, (tiles_mx, n_tiles))` once for expert-sorted rows."""
    ids = np.asarray(indices).reshape(-1).astype(np.int64)
    m = ids.size
    order = None if (m <= 1 or (np.diff(ids) >= 0).all()) else np.argsort(ids, kind='stable')
    ranked = ids if order is None else ids[order]
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
    ends = np.r_[starts[1:], m]
    tiles = np.array([(ranked[a], r, min(TILE, b - r)) for a, b in zip(starts, ends) for r in range(a, b, TILE)], np.int32)
    n_tiles = len(tiles)
    flat_tiles = tiles.reshape(-1)
    if cuda_kernels.available() and not mx.metal.is_available() and flat_tiles.size < 4:
        flat_tiles = np.pad(flat_tiles, (0, 4 - flat_tiles.size))
    tiles_mx = _dev_array(flat_tiles)
    order_mx = None if order is None else _dev_array(order.astype(np.int32))
    inv_order_mx = None if order is None else _dev_array(np.argsort(order).astype(np.int32))
    idx_sorted_mx = indices if order is None else _dev_array(ranked.astype(np.uint32))
    return order_mx, inv_order_mx, idx_sorted_mx, (tiles_mx, n_tiles)


def takes_rows(experts, x, indices):
    """Whether this call is a multi-row `x [M, 1, K]`, `indices [M]` call that a tiled Metal or CUDA GGUF kernel serves."""
    if not ON[0] or not _gpu_available():
        return False
    kind = kind_of(experts)
    if kind is None or experts.weight.ndim != 4 or experts.weight.dtype != mx.uint8:
        return False
    pb = iq_experts.per_block(kind)
    return (
        x.ndim == 3 and x.shape[1] == 1 and indices.ndim == 1 and x.shape[0] == indices.size
        and indices.size > 0 and x.shape[-1] % pb == 0 and x.shape[-1] % TILE == 0
    )


def run_rows(experts, x, indices, plan=None):
    """`x [M, 1, K]` through raw GGUF `experts` named by each row of `indices [M]`: `[M, 1, N]`."""
    kind = kind_of(experts)
    pb, bb = iq_experts.per_block(kind), iq_experts.BLOCK[kind]
    m, k, n = x.shape[0], x.shape[-1], experts.weight.shape[1]
    if plan is not None:
        tiles_mx, n_tiles = plan
        order_mx = inv_order_mx = None
    else:
        order_mx, inv_order_mx, _, (tiles_mx, n_tiles) = prepare_rows(indices)
    xs = x.reshape(m, k)
    if order_mx is not None:
        xs = xs[order_mx]
    n_col_tiles = (n + TILE - 1) // TILE
    template = [
        ('T', x.dtype), ('K_SIZE', k), ('K_BLOCKS', k // pb), ('N_SIZE', n),
        ('KIND', kind), ('PB', pb), ('BB', bb),
    ]
    if cuda_kernels.available() and not mx.metal.is_available():
        out_elems = cuda_kernels.roomy(m * n, x.dtype)
        y = _cuda_tiled()(
            inputs=[mx.contiguous(xs), mx.contiguous(experts.weight), _grid_for(kind), tiles_mx],
            template=template,
            grid=(128 * n_col_tiles, n_tiles, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(out_elems,)],
            output_dtypes=[x.dtype],
        )[0][:m * n].reshape(m, n)
    else:
        y = _tiled()(
            inputs=[mx.contiguous(xs), mx.contiguous(experts.weight), _grid_for(kind), tiles_mx],
            template=template,
            grid=(128 * n_col_tiles, n_tiles, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(m, n)],
            output_dtypes=[x.dtype],
        )[0]
    if inv_order_mx is not None:
        y = y[inv_order_mx]
    return y.reshape(m, 1, n)


def gather_rows(experts, x, indices, unpacked, plan=None):
    """`run_rows()` once verified within `TOLERANCE` of `unpacked()` for `CHECKS` calls, else `unpacked()`."""
    kind = kind_of(experts)
    key = ('rows', kind, x.shape[-1], experts.weight.shape[1], x.dtype)
    return _checked(key, lambda: run_rows(experts, x, indices, plan=plan), unpacked)


def _checked(key, mine, unpacked):
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if not choice:
            return unpacked()
        STATS['served'] += 1
        return mine()
    want = unpacked()
    STATS['checked'] += 1
    try:
        got = mine().astype(mx.float32)
        w = want.astype(mx.float32)
        near = got.shape == w.shape and (mx.abs(got - w).max() <= TOLERANCE * mx.maximum(mx.abs(w).max(), 1e-30)).item()
    except Exception as error:
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: GGUF kernel unavailable ({type(error).__name__}: {error})',
                  file=sys.stderr, flush=True)
        near = False
    if not near:
        STATS['mismatched'] += 1
    CHOICE[key], CHECKED[key] = bool(near), (seen + 1 if near else CHECKS)
    return want


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
