# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Metal gather-qmv and tiled gather-mm kernels for raw GGUF / ggml block expert projections.

Runs pre-distributed and uniform GGUF expert tensors (`IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ4_NL`, `IQ4_XS`, `Q2_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q4_0`, `Q4_1`,
`Q5_0`, `Q5_1`, `Q8_0`, `F16`, `BF16`) directly from stacked uint8 ggml blocks without unpacking
weights to bfloat16/float32 memory.

Decode / verify (`takes` / `gather`): one SIMD group (32 lanes) per routed slot and `ROWS` outputs.
Prefill (`takes_rows` / `gather_rows`): tiled 32x32 threadgroup `simdgroup_matrix` kernel over
expert-sorted rows.  Every shape & ggml type is checked against `unpacked()` for the first `CHECKS`
calls (`TOLERANCE = 1e-2`) before serving directly.
"""
import sys
from functools import lru_cache

import mlx.core as mx
import numpy as np

from . import iq_experts

CHECKS = 3
TOLERANCE = 1e-2
MAX_ROWS = 256
ROWS = 4
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
        _DUMMY_GRID = mx.zeros((1,), dtype=mx.uint16)
    return _DUMMY_GRID


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


def takes(experts, x, indices):
    """Whether this call is a single-row decode/verify call that the Metal GGUF qmv kernel serves."""
    if not ON[0] or not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    kind = kind_of(experts)
    if kind is None or experts.weight.ndim != 4 or experts.weight.dtype != mx.uint8:
        return False
    pb = iq_experts.per_block(kind)
    if x.ndim < 2 or x.shape[-2] != 1 or x.shape[-1] % pb != 0 or x.shape[-1] % 32 != 0:
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
    xs = mx.contiguous(mx.broadcast_to(x, shape + (1, k)).reshape(rows, k))
    idx = mx.broadcast_to(indices, shape).reshape(rows).astype(mx.int32)
    y = _kernel()(
        inputs=[xs, mx.contiguous(experts.weight), _grid_for(kind), idx],
        template=[
            ('T', x.dtype), ('K_SIZE', k), ('K_BLOCKS', k // pb), ('N_SIZE', n),
            ('KIND', kind), ('PB', pb), ('BB', bb), ('ROWS', ROWS),
        ],
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


def takes_rows(experts, x, indices):
    """Whether this call is a multi-row `x [M, 1, K]`, `indices [M]` call that the tiled Metal GGUF kernel serves."""
    if not ON[0] or not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    kind = kind_of(experts)
    if kind is None or experts.weight.ndim != 4 or experts.weight.dtype != mx.uint8:
        return False
    pb = iq_experts.per_block(kind)
    return (
        x.ndim == 3 and x.shape[1] == 1 and indices.ndim == 1 and x.shape[0] == indices.size
        and indices.size > 0 and x.shape[-1] % pb == 0 and x.shape[-1] % TILE == 0
    )


def run_rows(experts, x, indices):
    """`x [M, 1, K]` through raw GGUF `experts` named by each row of `indices [M]`: `[M, 1, N]`."""
    kind = kind_of(experts)
    pb, bb = iq_experts.per_block(kind), iq_experts.BLOCK[kind]
    ids = np.array(indices).astype(np.int64)
    m, k, n = ids.size, x.shape[-1], experts.weight.shape[1]
    order = None if (np.diff(ids) >= 0).all() else np.argsort(ids, kind='stable')
    ranked = ids if order is None else ids[order]
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
    ends = np.r_[starts[1:], m]
    tiles = np.array([(ranked[a], r, min(TILE, b - r)) for a, b in zip(starts, ends) for r in range(a, b, TILE)], np.int32)
    xs = x.reshape(m, k)
    if order is not None:
        xs = xs[mx.array(order.astype(np.int32))]
    n_tiles = (n + TILE - 1) // TILE
    y = _tiled()(
        inputs=[mx.contiguous(xs), mx.contiguous(experts.weight), _grid_for(kind), mx.array(tiles.reshape(-1))],
        template=[
            ('T', x.dtype), ('K_SIZE', k), ('K_BLOCKS', k // pb), ('N_SIZE', n),
            ('KIND', kind), ('PB', pb), ('BB', bb),
        ],
        grid=(128 * n_tiles, len(tiles), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]
    if order is not None:
        y = y[mx.array(np.argsort(order).astype(np.int32))]
    return y.reshape(m, 1, n)


def gather_rows(experts, x, indices, unpacked):
    """`run_rows()` once verified within `TOLERANCE` of `unpacked()` for `CHECKS` calls, else `unpacked()`."""
    kind = kind_of(experts)
    key = ('rows', kind, x.shape[-1], experts.weight.shape[1], x.dtype)
    return _checked(key, lambda: run_rows(experts, x, indices), unpacked)


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
