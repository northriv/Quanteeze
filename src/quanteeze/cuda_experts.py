# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A decode token's routed experts on CUDA in two kernels, MLX's tensor-core arithmetic bit for bit.

switch_glu() is Qwen3_5BatchInvariantForward._switch_glu for a step whose shape this takes; the
first CHECKS times a shape is met its output is compared bit for bit with the original's, and it
serves the shape only if they are the same (where MLX goes another way, as gather_qmv below 8
experts or another GPU's kernel, they would not be).  QUANTEEZE_CUDA_EXPERTS=off keeps mlx-vlm's.
A warp's rows are qz_expert_rows in HEADER, which quanteeze.cuda_moe's last kernel computes too.
"""
import sys
from functools import lru_cache

import mlx.core as mx

from . import cuda_kernels

CHECKS = 3
ON = [True]                     # QUANTEEZE_CUDA_EXPERTS
STATS = dict(served=0, checked=0, mismatched=0)
CHOICE = {}                     # (k, hidden, inner, group size): whether it matched the original
CHECKED = {}
SAID = []

MXFP4_DECODE_HEADER = r"""
__device__ __forceinline__ unsigned qz_bits(__nv_bfloat162 v) { return *reinterpret_cast<unsigned*>(&v); }
__device__ __forceinline__ __nv_bfloat162 qz_e8m0_bf162(uint8_t e) {
  const uint16_t u = (e == 0u) ? 0x0040u : (static_cast<uint16_t>(e) << 7);
  const uint32_t u2 = static_cast<uint32_t>(u) | (static_cast<uint32_t>(u) << 16);
  return *reinterpret_cast<const __nv_bfloat162*>(&u2);
}
__device__ __forceinline__ unsigned qz_deq_mxfp4(uint32_t v, int t, __nv_bfloat162 s2) {
  constexpr uint16_t E2M1[16] = {
      0x0000u, 0x3f00u, 0x3f80u, 0x3fc0u, 0x4000u, 0x4040u, 0x4080u, 0x40c0u,
      0x8000u, 0xbf00u, 0xbf80u, 0xbfc0u, 0xc000u, 0xc040u, 0xc080u, 0xc0c0u
  };
  const uint32_t byte = (v >> (8 * t)) & 0xffu;
  const uint32_t q_bits = static_cast<uint32_t>(E2M1[byte & 0xfu])
      | (static_cast<uint32_t>(E2M1[byte >> 4]) << 16);
  return qz_bits(__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q_bits), s2));
}
"""

HEADER = MXFP4_DECODE_HEADER + r"""
__device__ __forceinline__ void qz_mma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// qz_expert_rows of 2-bit weights (MLX's qmm_naive for them: the same mma.sync m16n8k16 over the 16-wide chunks
// in order, each weight w * scale, then + bias, each rounded to bfloat16): a chunk is one word, input i at bits
// 2i; lane t's pairs are inputs 2t, 2t+1 (bits 4t, 4t+2) and 2t+8, 2t+9 (bits 16+4t, 16+4t+2), rounded to
// bfloat16 under 128's bits as the 4-bit pairs are.
template <int N, int K, int GS, bool PAIR>
__device__ __forceinline__ void qz_expert_rows2(float (&c)[PAIR ? 2 : 1][4], const __nv_bfloat16* xr,
                                                const uint32_t* w, const __nv_bfloat16* scales,
                                                const __nv_bfloat16* biases, const uint32_t* w2,
                                                const __nv_bfloat16* scales2, const __nv_bfloat16* biases2,
                                                int e, int n0) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int n = n0 + g;
  constexpr int NW = PAIR ? 2 : 1;
  const uint32_t* wr[NW];
  const __nv_bfloat16* sr[NW];
  const __nv_bfloat16* br[NW];
  wr[0] = w + (static_cast<size_t>(e) * N + n) * (K / 16);
  sr[0] = scales + (static_cast<size_t>(e) * N + n) * (K / GS);
  br[0] = biases + (static_cast<size_t>(e) * N + n) * (K / GS);
  if constexpr (PAIR) {
    wr[NW - 1] = w2 + (static_cast<size_t>(e) * N + n) * (K / 16);
    sr[NW - 1] = scales2 + (static_cast<size_t>(e) * N + n) * (K / GS);
    br[NW - 1] = biases2 + (static_cast<size_t>(e) * N + n) * (K / GS);
  }
  const uint32_t magic_bits = 0x43004300u;
  const __nv_bfloat162 magic = *reinterpret_cast<const __nv_bfloat162*>(&magic_bits);
  for (int p = 0; p < NW; ++p) {
    for (int i = 0; i < 4; ++i) {
      c[p][i] = 0.0f;
    }
  }
  constexpr int CH = K / 16;
  constexpr int GR = 8;
  static_assert(CH % GR == 0, "the chunks a multiple of a group");
  uint4 cur[NW][GR / 4], nxt[NW][GR / 4];
  auto load = [&](uint4 (&dst)[NW][GR / 4], int g0) {
#pragma unroll
    for (int p = 0; p < NW; ++p) {
#pragma unroll
      for (int q = 0; q < GR / 4; ++q) {
        dst[p][q] = *reinterpret_cast<const uint4*>(wr[p] + g0 + 4 * q);
      }
    }
  };
  load(cur, 0);
  for (int g0 = 0; g0 < CH; g0 += GR) {
    if (g0 + GR < CH) {
      load(nxt, g0 + GR);
    }
#pragma unroll
    for (int cc = 0; cc < GR; ++cc) {
      const int kc = g0 + cc;
      unsigned a0 = 0u, a2 = 0u;
      if (g == 0) {
        a0 = *reinterpret_cast<const unsigned*>(xr + kc * 16 + 2 * t);
        a2 = *reinterpret_cast<const unsigned*>(xr + kc * 16 + 2 * t + 8);
      }
#pragma unroll
      for (int p = 0; p < NW; ++p) {
        const uint4 q4 = cur[p][cc / 4];
        const uint32_t v = (cc & 3) == 0 ? q4.x : (cc & 3) == 1 ? q4.y : (cc & 3) == 2 ? q4.z : q4.w;
        const __nv_bfloat162 s2 = __bfloat162bfloat162(sr[p][(kc * 16) / GS]);
        const __nv_bfloat162 b2 = __bfloat162bfloat162(br[p][(kc * 16) / GS]);
        unsigned bq[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int at = 16 * h + 4 * t;
          uint32_t bits = ((v >> at) & 0x3u) | (((v >> (at + 2)) & 0x3u) << 16) | magic_bits;
          __nv_bfloat162 q = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits), magic);
          bq[h] = qz_bits(__hadd2(__hmul2(q, s2), b2));
        }
        qz_mma(c[p], a0, 0u, a2, 0u, bq[0], bq[1]);
      }
    }
#pragma unroll
    for (int p = 0; p < NW; ++p) {
#pragma unroll
      for (int q = 0; q < GR / 4; ++q) {
        cur[p][q] = nxt[p][q];
      }
    }
  }
}

// A warp's 8 rows n0..n0+7 of expert e on the input xr [K] (PAIR: of two projections, two chains):
// lane (g, t) holds B's column n0 + g; c[p] its C fragment, row 0's columns 2t and 2t+1 in lanes 0..3.
// BITS: 4 (MLX's qmm_sm80 / MXFP4), or 2 (MLX's qmm_naive: qz_expert_rows2).
template <int N, int K, int GS, bool PAIR, bool MXFP4 = false, int BITS = 4, typename ScaleT = __nv_bfloat16>
__device__ __forceinline__ void qz_expert_rows(float (&c)[PAIR ? 2 : 1][4], const __nv_bfloat16* xr,
                                               const uint32_t* w, const ScaleT* scales,
                                               const ScaleT* biases, const uint32_t* w2,
                                               const ScaleT* scales2, const ScaleT* biases2,
                                               int e, int n0) {
  if constexpr (!MXFP4 && BITS == 2) {
    qz_expert_rows2<N, K, GS, PAIR>(
        c, xr, w, reinterpret_cast<const __nv_bfloat16*>(scales),
        reinterpret_cast<const __nv_bfloat16*>(biases), w2,
        reinterpret_cast<const __nv_bfloat16*>(scales2),
        reinterpret_cast<const __nv_bfloat16*>(biases2), e, n0);
    return;
  }
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int n = n0 + g;
  constexpr int NW = PAIR ? 2 : 1;
  const uint32_t* wr[NW];
  const ScaleT* sr[NW];
  const ScaleT* br[NW];
  wr[0] = w + (static_cast<size_t>(e) * N + n) * (K / 8);
  sr[0] = scales + (static_cast<size_t>(e) * N + n) * (K / GS);
  if constexpr (!MXFP4) {
    br[0] = biases + (static_cast<size_t>(e) * N + n) * (K / GS);
  }
  if constexpr (PAIR) {
    wr[NW - 1] = w2 + (static_cast<size_t>(e) * N + n) * (K / 8);
    sr[NW - 1] = scales2 + (static_cast<size_t>(e) * N + n) * (K / GS);
    if constexpr (!MXFP4) {
      br[NW - 1] = biases2 + (static_cast<size_t>(e) * N + n) * (K / GS);
    }
  }
  const uint32_t magic_bits = 0x43004300u;
  const __nv_bfloat162 magic = *reinterpret_cast<const __nv_bfloat162*>(&magic_bits);
  for (int p = 0; p < NW; ++p) {
    for (int i = 0; i < 4; ++i) {
      c[p][i] = 0.0f;
    }
  }
  constexpr int CH = K / 16;
  constexpr int GR = 8;
  static_assert(CH % GR == 0, "the chunks a multiple of a group");
  uint4 cur[NW][GR / 2], nxt[NW][GR / 2];
  auto load = [&](uint4 (&dst)[NW][GR / 2], int g0) {
#pragma unroll
    for (int p = 0; p < NW; ++p) {
#pragma unroll
      for (int q = 0; q < GR / 2; ++q) {
        dst[p][q] = *reinterpret_cast<const uint4*>(wr[p] + (g0 + 2 * q) * 2);
      }
    }
  };
  load(cur, 0);
  for (int g0 = 0; g0 < CH; g0 += GR) {
    if (g0 + GR < CH) {
      load(nxt, g0 + GR);
    }
#pragma unroll
    for (int cc = 0; cc < GR; ++cc) {
      const int kc = g0 + cc;
      unsigned a0 = 0u, a2 = 0u;
      if (g == 0) {
        a0 = *reinterpret_cast<const unsigned*>(xr + kc * 16 + 2 * t);
        a2 = *reinterpret_cast<const unsigned*>(xr + kc * 16 + 2 * t + 8);
      }
#pragma unroll
      for (int p = 0; p < NW; ++p) {
        const uint4 q4 = cur[p][cc / 2];
        const uint32_t w0 = (cc & 1) ? q4.z : q4.x;
        const uint32_t w1 = (cc & 1) ? q4.w : q4.y;
        unsigned bq[2];
        if constexpr (MXFP4) {
          const __nv_bfloat162 s2 = qz_e8m0_bf162(*reinterpret_cast<const uint8_t*>(&sr[p][(kc * 16) / GS]));
          bq[0] = qz_deq_mxfp4(w0, t, s2);
          bq[1] = qz_deq_mxfp4(w1, t, s2);
        } else {
          const __nv_bfloat162 s2 = __bfloat162bfloat162(
              *reinterpret_cast<const __nv_bfloat16*>(&sr[p][(kc * 16) / GS]));
          const __nv_bfloat162 b2 = __bfloat162bfloat162(
              *reinterpret_cast<const __nv_bfloat16*>(&br[p][(kc * 16) / GS]));
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            uint32_t v = h ? w1 : w0;
            uint32_t bits = ((v >> (8 * t)) & 0xFu) | (((v >> (8 * t + 4)) & 0xFu) << 16) | magic_bits;
            __nv_bfloat162 q = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits), magic);
            bq[h] = qz_bits(__hadd2(__hmul2(q, s2), b2));
          }
        }
        qz_mma(c[p], a0, 0u, a2, 0u, bq[0], bq[1]);
      }
    }
#pragma unroll
    for (int p = 0; p < NW; ++p) {
#pragma unroll
      for (int q = 0; q < GR / 2; ++q) {
        cur[p][q] = nxt[p][q];
      }
    }
  }
}

// lane t's pair of a chunk word v (inputs 2t and 2t + 1 of its 8) dequantized as qz_expert_rows does
// (the nibbles under 128's bits by one byte_perm: byte t of v and of v >> 4)
__device__ __forceinline__ unsigned qz_deq(uint32_t v, int t, __nv_bfloat162 s2, __nv_bfloat162 b2) {
  const uint32_t magic_bits = 0x43004300u;
  const __nv_bfloat162 magic = *reinterpret_cast<const __nv_bfloat162*>(&magic_bits);
  const uint32_t bits = (__byte_perm(v, v >> 4, static_cast<unsigned>(t | ((4 + t) << 8))) & 0x000F000Fu) | magic_bits;
  const __nv_bfloat162 q = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits), magic);
  return qz_bits(__hadd2(__hmul2(q, s2), b2));
}

__device__ __forceinline__ unsigned qz_smem(const void* p) {
  unsigned s;
  asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(s) : "l"(p));
  return s;
}
__device__ __forceinline__ void qz_cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(qz_smem(dst)), "l"(src));
}
__device__ __forceinline__ void qz_cp8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(qz_smem(dst)), "l"(src));
}
__device__ __forceinline__ void qz_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int R>
__device__ __forceinline__ void qz_cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(R)); }
// qz_cp_wait with a count known once a loop is unrolled (0 to 7)
__device__ __forceinline__ void qz_cp_wait_dyn(int n) {
  switch (n) {
    case 0: qz_cp_wait<0>(); break;
    case 1: qz_cp_wait<1>(); break;
    case 2: qz_cp_wait<2>(); break;
    case 3: qz_cp_wait<3>(); break;
    case 4: qz_cp_wait<4>(); break;
    case 5: qz_cp_wait<5>(); break;
    case 6: qz_cp_wait<6>(); break;
    default: qz_cp_wait<7>(); break;
  }
}

// 16 rows of a warp, chunk kc, from shared memory: the weights as mma's A (lane (g, t) rows g and
// g + 8: wrow its row g's words, ROWW on for g + 8; srow / brow its scales and biases, SBR on), the
// token as B's column 0 (x0, x1: lanes g == 0); C's rows g and g + 8 of column 0 in lanes t == 0
template <int ROWW, int SBR, int GS>
__device__ __forceinline__ void qz_rows16_chunk(float (&c)[4], const uint32_t* wrow, int cw,
                                                const __nv_bfloat16* srow, const __nv_bfloat16* brow, int kc,
                                                unsigned x0, unsigned x1, int t) {
  const uint2 w0 = *reinterpret_cast<const uint2*>(wrow + cw);
  const uint2 w8 = *reinterpret_cast<const uint2*>(wrow + 8 * ROWW + cw);
  const __nv_bfloat162 s0 = __bfloat162bfloat162(srow[(kc * 16) / GS]);
  const __nv_bfloat162 b0 = __bfloat162bfloat162(brow[(kc * 16) / GS]);
  const __nv_bfloat162 s8 = __bfloat162bfloat162(srow[8 * SBR + (kc * 16) / GS]);
  const __nv_bfloat162 b8 = __bfloat162bfloat162(brow[8 * SBR + (kc * 16) / GS]);
  qz_mma(c, qz_deq(w0.x, t, s0, b0), qz_deq(w8.x, t, s8, b8), qz_deq(w0.y, t, s0, b0), qz_deq(w8.y, t, s8, b8),
         x0, x1);
}

template <int ROWW, int SBR, int GS>
__device__ __forceinline__ void qz_rows16_chunk_mxfp4(float (&c)[4], const uint32_t* wrow, int cw,
                                                      const uint8_t* srow, int kc,
                                                      unsigned x0, unsigned x1, int t) {
  const uint2 w0 = *reinterpret_cast<const uint2*>(wrow + cw);
  const uint2 w8 = *reinterpret_cast<const uint2*>(wrow + 8 * ROWW + cw);
  const __nv_bfloat162 s0 = qz_e8m0_bf162(srow[(kc * 16) / GS]);
  const __nv_bfloat162 s8 = qz_e8m0_bf162(srow[8 * SBR + (kc * 16) / GS]);
  qz_mma(c, qz_deq_mxfp4(w0.x, t, s0), qz_deq_mxfp4(w8.x, t, s8),
         qz_deq_mxfp4(w0.y, t, s0), qz_deq_mxfp4(w8.y, t, s8), x0, x1);
}
"""

# A warp 8 rows (n0..n0+7) of slot j's expert: C's row 0 is lanes 0..3, columns 2t and 2t+1.
SOURCE = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int gw = blockIdx.x * (blockDim.x / 32) + warp.meta_group_rank();
    constexpr int BLOCKS = N / 8;
    if (gw >= SLOTS * BLOCKS) {
      return;
    }
    const int j = gw / BLOCKS;
    const int n0 = (gw % BLOCKS) * 8;
    const int lane = warp.thread_rank();
    const int t = lane & 3;
    float c[PAIR ? 2 : 1][4];
    qz_expert_rows<N, K, GS, PAIR, MXFP4, BITS>(c, x + static_cast<size_t>(X_DIV ? j / X_DIV : 0) * K, w, scales,
                                                biases, w2, scales2, biases2, static_cast<int>(idx[j]), n0);
    if (lane < 4) {
      for (int i = 0; i < 2; ++i) {
        __nv_bfloat16 gate = __float2bfloat16_rn(c[0][i]);
        __nv_bfloat16 out = gate;
        if constexpr (PAIR) {
          __nv_bfloat16 up = __float2bfloat16_rn(c[PAIR ? 1 : 0][i]);
          out = Multiply{}(Multiply{}(gate, Sigmoid{}(gate)), up);
        }
        y[static_cast<size_t>(j) * N + n0 + 2 * t + i] = out;
      }
    }
"""


# gate and up read the way a copy reads: a block 32 rows of slot j's expert of each (warps 0-1 gate's,
# 2-3 up's, 16 rows each); each warp copies (cp.async) 256 bytes of each of its rows a stage (lanes 16
# bytes apart: two rows' 512 bytes an instruction) with the stage's scales and biases into two buffers
# of its own, and computes a stage while the next lands; the token in shared memory.
STAGED_SOURCE = r"""
    constexpr int RB = 32;
    constexpr int SEGW = 64;                        // a row's words a stage (256 bytes, 32 chunks)
    constexpr int ROWW = SEGW + 4;                  // a row's words in shared memory (rows' banks apart)
    constexpr int NS = K / 512;
    constexpr int GPS = 512 / GS;                   // a stage's groups
    constexpr int SBR = MXFP4 ? 32 : (GPS + 8);     // a row's scales a stage in shared memory (16B aligned)
    constexpr int SBP = GPS / 8;                    // ... in 16-byte pieces
    constexpr int BUF = 16 * ROWW * 4 + (MXFP4 ? (16 * SBR) : (2 * 16 * SBR * 2));
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    unsigned char* mine = smem + K * 2 + wid * 2 * BUF;
    __shared__ __nv_bfloat16 gu[2][RB];
    const int j = blockIdx.x / (N / RB);
    const int r0 = (blockIdx.x % (N / RB)) * RB;
    const int m = wid / 2;
    const int wr0 = (wid % 2) * 16;
    const size_t grow = static_cast<size_t>(idx[j]) * N + r0 + wr0;
    const uint32_t* wsrc = (m ? w2 : w) + grow * (K / 8);
    const auto* ssrc = (m ? scales2 : scales) + grow * (K / GS);
    const auto* bsrc = (m ? biases2 : biases) + grow * (K / GS);
    const __nv_bfloat16* xj = x + static_cast<size_t>(j / TOP_K) * K;
    for (int p = tid; p < K / 8; p += 128) {
      qz_cp16(xs + p * 8, xj + p * 8);
    }
    auto stage = [&](int s) {
      if (s < NS) {
        unsigned char* buf = mine + (s % 2) * BUF;
        uint32_t* dst = reinterpret_cast<uint32_t*>(buf);
        if constexpr (MXFP4) {
          uint8_t* sb = reinterpret_cast<uint8_t*>(buf + 16 * ROWW * 4);
          if (lane < 16) {
            qz_cp16(sb + lane * SBR, ssrc + lane * (K / GS) + s * GPS);
          }
        } else {
          __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(buf + 16 * ROWW * 4);
          for (int p = lane; p < 2 * 16 * SBP; p += 32) {
            const int a = p / (16 * SBP);
            const int q = p % (16 * SBP);
            qz_cp16(sb + (a * 16 + q / SBP) * SBR + (q % SBP) * 8,
                    (a ? bsrc : ssrc) + (q / SBP) * (K / GS) + s * GPS + (q % SBP) * 8);
          }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int p = lane + 32 * i;
          qz_cp16(dst + (p / 16) * ROWW + (p % 16) * 4, wsrc + (p / 16) * (K / 8) + s * SEGW + (p % 16) * 4);
        }
      }
      qz_cp_commit();
    };
    stage(0);
    const uint32_t* xw = reinterpret_cast<const uint32_t*>(xs);
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < NS; ++s) {
      qz_cp_wait<0>();
      if (s == 0) {
        __syncthreads();                            // the token, every warp's copies
      } else {
        __syncwarp();
      }
      stage(s + 1);
      const unsigned char* buf = mine + (s % 2) * BUF;
      const uint32_t* wrow = reinterpret_cast<const uint32_t*>(buf) + g * ROWW;
      if constexpr (MXFP4) {
        const uint8_t* srow = reinterpret_cast<const uint8_t*>(buf + 16 * ROWW * 4) + g * SBR;
#pragma unroll 8
        for (int cc = 0; cc < 32; ++cc) {
          const int kc = s * 32 + cc;
          unsigned x0 = 0u, x1 = 0u;
          if (g == 0) {
            x0 = xw[kc * 8 + t];
            x1 = xw[kc * 8 + t + 4];
          }
          qz_rows16_chunk_mxfp4<ROWW, SBR, GS>(c, wrow, 2 * cc, srow, cc, x0, x1, t);
        }
      } else {
        const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(buf + 16 * ROWW * 4);
        const __nv_bfloat16* srow = sb + g * SBR;
        const __nv_bfloat16* brow = sb + (16 + g) * SBR;
#pragma unroll 8
        for (int cc = 0; cc < 32; ++cc) {
          const int kc = s * 32 + cc;
          unsigned x0 = 0u, x1 = 0u;
          if (g == 0) {
            x0 = xw[kc * 8 + t];
            x1 = xw[kc * 8 + t + 4];
          }
          qz_rows16_chunk<ROWW, SBR, GS>(c, wrow, 2 * cc, srow, brow, cc, x0, x1, t);
        }
      }
    }
    if (t == 0) {
      gu[m][wr0 + g] = __float2bfloat16_rn(c[0]);
      gu[m][wr0 + g + 8] = __float2bfloat16_rn(c[2]);
    }
    __syncthreads();
    if (tid < RB) {
      const __nv_bfloat16 gate = gu[0][tid];
      y[static_cast<size_t>(j) * N + r0 + tid] = Multiply{}(Multiply{}(gate, Sigmoid{}(gate)), gu[1][tid]);
    }
"""


def _staged_smem(k, group_size, mxfp4=False):
    scale_bytes = 16 * 32 if mxfp4 else 2 * 16 * (512 // group_size + 8) * 2
    return k * 2 + 4 * 2 * (16 * 68 * 4 + scale_bytes)


def staged(n, k, group_size, mxfp4=False):
    """Whether gate_up() takes the staged kernel: inputs in stages of 512, a stage's scales in 16-byte
    pieces (groups of 32 or 64), rows in blocks of 32, its shared memory within an A100's."""
    return (k % 512 == 0 and group_size in (32, 64) and n % 32 == 0
            and _staged_smem(k, group_size, mxfp4) <= 99 * 1024)


@lru_cache(maxsize=None)
def _staged_kernel(k, group_size, mxfp4=False):
    suffix = f'_{k}_{group_size}_mxfp4' if mxfp4 else f'_{k}_{group_size}'
    return cuda_kernels.kernel(f'quanteeze_experts_staged{suffix}',
                               ['x', 'w', 'scales', 'biases', 'idx', 'w2', 'scales2', 'biases2'], ['y'],
                               STAGED_SOURCE, HEADER, shared_memory=_staged_smem(k, group_size, mxfp4))


def _staged_rows_kernel(k, group_size, mxfp4=False):
    return _staged_kernel(k, group_size, mxfp4)


@lru_cache(maxsize=None)
def _kernel():
    return cuda_kernels.kernel('quanteeze_experts', ['x', 'w', 'scales', 'biases', 'idx', 'w2', 'scales2', 'biases2'],
                               ['y'], SOURCE, HEADER)


def _run(x, proj, idx, x_div, pair=None):
    slots, n, k = idx.size, proj.weight.shape[1], x.shape[-1]
    mxfp4 = getattr(proj, 'mode', None) == 'mxfp4'
    bits = 4 if mxfp4 else int(getattr(proj, 'bits', 4))
    second = pair or proj
    b1 = proj.scales if mxfp4 else proj.biases
    b2 = second.scales if mxfp4 else second.biases
    return _kernel()(
        inputs=[mx.contiguous(x), proj.weight, proj.scales, b1, idx, second.weight, second.scales, b2],
        template=[('N', n), ('K', k), ('GS', int(proj.group_size)), ('SLOTS', slots), ('X_DIV', int(x_div)),
                  ('PAIR', pair is not None), ('MXFP4', mxfp4), ('BITS', bits)],
        grid=(32 * slots * (n // 8), 1, 1), threadgroup=(256, 1, 1), output_shapes=[(slots, n)],
        output_dtypes=[x.dtype])[0]


def gate_up(switch, x, idx, per=None):
    """swiglu(gate_proj(x), up_proj(x)) of each slot's expert for token rows `x`: `[slots, inner]`."""
    gate, up = switch.gate_proj, switch.up_proj
    slots, n, k, gs = idx.size, gate.weight.shape[1], x.shape[-1], int(gate.group_size)
    mxfp4 = getattr(gate, 'mode', None) == 'mxfp4'
    rows = max(1, x.size // k)
    top_k = int(per) if per is not None else max(1, slots // rows)
    if int(getattr(gate, 'bits', 4)) != 4 or not staged(n, k, gs, mxfp4):
        return _run(x, gate, idx, 0 if rows == 1 and per is None else top_k, pair=up)
    b1 = gate.scales if mxfp4 else gate.biases
    b2 = up.scales if mxfp4 else up.biases
    return _staged_kernel(k, gs, mxfp4)(
        inputs=[mx.contiguous(x.reshape(-1)), gate.weight, gate.scales, b1, idx, up.weight, up.scales, b2],
        template=[('N', n), ('K', k), ('GS', gs), ('SLOTS', slots), ('TOP_K', top_k), ('MXFP4', mxfp4)],
        grid=(128 * slots * (n // 32), 1, 1), threadgroup=(128, 1, 1), output_shapes=[(slots, n)],
        output_dtypes=[x.dtype])[0]


def down(switch, h, idx):
    """down_proj of each slot's expert on its own row of h [slots, inner]: [slots, hidden]."""
    return _run(h, switch.down_proj, idx, 1)


def _projection(p, k, dtype):
    if type(p).__name__ not in ('QuantizedSwitchLinear', 'Projection') or 'bias' in p:
        return False
    mode = getattr(p, 'mode', None)
    if mode == 'mxfp4':
        return (p.bits == 4 and p.group_size == 32 and p.get('biases') is None
                and p.weight.dtype == mx.uint32 and p.scales.dtype == mx.uint8
                and p.weight.shape[-1] * 8 == k and p.scales.shape[-1] * 32 == k
                and p.weight.shape[1] % 8 == 0 and k % 128 == 0)
    return (mode == 'affine' and p.bits in (2, 4)
            and p.group_size in (32, 64, 128) and p.get('biases') is not None
            and p.weight.dtype == mx.uint32 and p.scales.dtype == dtype and p.biases.dtype == dtype
            and p.weight.shape[-1] * (32 // p.bits) == k and p.weight.shape[1] % 8 == 0 and k % 128 == 0)


def takes(switch, x, indices):
    """Whether switch_glu() computes this step's experts as MLX's qmm_sm80 / qmm_naive do: on CUDA, bfloat16
    batch decode (`B=1..8, T=1`) or single-request MTP verify (`B=1, T=1..8`), 8 to 64 total slots, 4-bit or
    2-bit affine or mxfp4 projections, a SwiGLU between them."""
    if not cuda_kernels.available() or x.dtype != mx.bfloat16 or x.ndim != 3:
        return False
    b, t = x.shape[0], x.shape[1]
    if not ((t == 1 and 1 <= b <= 8) or (b == 1 and 1 <= t <= 8)):
        return False
    if indices.ndim != 3 or indices.shape[:2] != x.shape[:2] or indices.shape[-1] <= 0:
        return False
    if not all(hasattr(switch, name) for name in ('gate_proj', 'up_proj', 'down_proj', 'activation')):
        return False
    if type(switch.activation).__name__ != 'SwiGLU' or not 8 <= indices.size <= 64:
        return False
    hidden = x.shape[-1]
    gate, up, dn = switch.gate_proj, switch.up_proj, switch.down_proj
    return (_projection(gate, hidden, x.dtype) and _projection(up, hidden, x.dtype)
            and gate.weight.shape == up.weight.shape and gate.group_size == up.group_size
            and gate.bits == up.bits
            and getattr(gate, 'mode', None) == getattr(up, 'mode', None)
            and _projection(dn, gate.weight.shape[1], x.dtype) and dn.weight.shape[1] == hidden)


def decode(switch, x, indices, fallback):
    """`fallback()` or custom CUDA `gate_up()` + `down()` for `[B, T, k, hidden]` once bit-verified."""
    if not ON[0] or not takes(switch, x, indices):
        return fallback()
    b, t, top_k = x.shape[0], x.shape[1], indices.shape[-1]
    rows = b * t

    def mine():
        idx = mx.contiguous(indices.reshape(-1).astype(mx.uint32))
        h = gate_up(switch, x.reshape(rows, -1), idx, per=top_k if rows > 1 else None)
        return down(switch, h, idx).reshape(b, t, top_k, -1)

    gate, dn = switch.gate_proj, switch.down_proj
    base = ((top_k, x.shape[-1], gate.weight.shape[1], int(gate.group_size))
            if rows == 1 else
            (rows, top_k, x.shape[-1], gate.weight.shape[1], int(gate.group_size)))
    key = base if gate.mode == 'affine' and dn.mode == 'affine' else base + (gate.mode, dn.mode)
    bits_pair = (int(gate.bits), int(dn.bits))
    if bits_pair != (4, 4):
        key = key + bits_pair
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS['served'] += 1
            return mine()
        return fallback()
    want = fallback()
    try:
        got = mine()
        same = mx.array_equal(got, want).item()
    except Exception as error:
        if 'transformations' in str(error):                 # traced (mx.compile): nothing to compare yet
            return want
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: the experts\' kernels unavailable ({type(error).__name__}: {error})', file=sys.stderr,
                  flush=True)
        same = False
    STATS['checked'] += 1
    STATS['mismatched'] += not same
    CHOICE[key], CHECKED[key] = bool(same), (seen + 1 if same else CHECKS)
    return want


def switch_glu(forward, switch, x, indices):
    """forward._switch_glu(switch, x, indices) [1, 1, k, hidden], through gate_up() and down() where
    this shape matched it."""
    if getattr(switch, 'demand', None) is not None:
        return switch(x, indices)
    return decode(switch, x, indices, lambda: forward._switch_glu(switch, x, indices))


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
