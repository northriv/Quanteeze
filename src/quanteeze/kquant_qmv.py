# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Metal gather-qmv and tiled gather-mm kernels for ggml's `Q2_K`, `Q3_K`, and `IQ2_XS` expert
projections (`QUANTEEZE_KQUANT_QMV`).

Decode (`takes` / `gather`): one SIMD group (32 lanes) per routed slot and `ROWS` outputs, reading
`Q2_K` / `Q3_K` / `IQ2_XS` blocks directly in float32 without unpacking to bfloat16.
Prefill (`takes_rows` / `gather_rows`): tiled 32x32 threadgroup `simdgroup_matrix` kernel over
expert-sorted rows.
Each shape is verified against `unpacked()` for the first `CHECKS` calls (`TOLERANCE = 1e-2`) before
the fused kernel serves it."""
import sys
from functools import lru_cache

import mlx.core as mx
import numpy as np

from . import iq_experts, kernel_checks

CHECKS = 3
TOLERANCE = 1e-2
MAX_ROWS = 256
ROWS = 4
ON = [True]                                     # QUANTEEZE_KQUANT_QMV
MODES = {'q2_k': 2, 'q3_k': 3, 'iq2_xs': 17}
STATS = dict(served=0, checked=0, mismatched=0)
CHOICE, CHECKED = kernel_checks.tables('kquant_qmv')
SAID = []
_DUMMY_GRID = None

_HEADER = r"""
static constant float IQ2_MAG[4] = {8.0f, 25.0f, 43.0f, 0.0f};
"""

_SOURCE = r"""
  const uint lane = thread_position_in_grid.x;
  const uint first = thread_position_in_grid.y * ROWS;
  const uint r = thread_position_in_grid.z;
  const int e = idx[r];
  const device T* xr = x + (ulong)r * (K_BLOCKS * 256);
  const uint v0 = lane * 8, sub = v0 / 16, h = v0 / 128, j = (v0 % 128) / 32, l0 = v0 % 32, hb = v0 / 32;
  float acc[ROWS];
  for (int row = 0; row < ROWS; row++) acc[row] = 0.0f;
  for (int b = 0; b < K_BLOCKS; b++) {
    float xv[8];
    float xsum = 0.0f;
    for (int i = 0; i < 8; i++) { xv[i] = float(xr[b * 256 + v0 + i]); xsum += xv[i]; }
    for (int row = 0; row < ROWS; row++) {
      const uint n = first + row;
      if (n >= N_SIZE) break;
      const ulong at = ((ulong)e * N_SIZE + n) * K_BLOCKS + b;
      const device uint8_t* blk = w + at * BODY;
      const float dd = float(d[at]);
      float dot = 0.0f;
      if (MODE == 2) {                                   // Q2_K: scales and mins, then the codes
        const uint s = blk[sub];
        for (int i = 0; i < 8; i++) dot += xv[i] * float((blk[16 + 32 * h + l0 + i] >> (2 * j)) & 3);
        acc[row] += dd * float(s & 15) * dot - float(dmin[at]) * float(s >> 4) * xsum;
      } else if (MODE == 3) {                            // Q3_K: high bits, codes, 6-bit scales
        const uint low = sub < 8 ? (blk[96 + sub] & 15) : (blk[96 + sub - 8] >> 4);
        const uint high = (blk[104 + sub % 4] >> (2 * (sub / 4))) & 3;
        for (int i = 0; i < 8; i++) {
          int q = int((blk[32 + 32 * h + l0 + i] >> (2 * j)) & 3);
          if (((blk[l0 + i] >> hb) & 1) == 0) q -= 4;
          dot += xv[i] * float(q);
        }
        acc[row] += dd * float(int(low | (high << 4)) - 32) * dot;
      } else {                                           // IQ2_XS: 64B 16-bit codes, 8B half-scales
        const uint octad = lane, ib32 = octad >> 2, l = octad & 3u;
        const uint q16 = uint(blk[2u * octad]) | (uint(blk[2u * octad + 1u]) << 8);
        const uint s = blk[64u + ib32];
        const float db = dd * (0.5f + float(l < 2u ? (s & 15u) : (s >> 4))) * 0.25f;
        const uint s7 = q16 >> 9;
        const uint sbyte = s7 | ((metal::popcount(s7) & 1u) << 7);
        const uint16_t g = grid[q16 & 511u];
        for (int i = 0; i < 8; i++) {
          const float sign = ((sbyte >> i) & 1u) ? -1.0f : 1.0f;
          dot += xv[i] * IQ2_MAG[(g >> (2u * i)) & 3u] * sign;
        }
        acc[row] += db * dot;
      }
    }
  }
  for (int row = 0; row < ROWS; row++) {
    const uint n = first + row;
    const float total = simd_sum(acc[row]);
    if (lane == 0 && n < N_SIZE) y[(ulong)r * N_SIZE + n] = static_cast<T>(total);
  }
"""


def _grid_for(mode):
    global _DUMMY_GRID
    if mode == 'iq2_xs':
        return iq_experts.compact_grids()[iq_experts.IQ2_XS]
    if _DUMMY_GRID is None:
        _DUMMY_GRID = mx.zeros((1,), dtype=mx.uint16)
    return _DUMMY_GRID


@lru_cache(maxsize=None)
def _kernel(mode):
    return mx.fast.metal_kernel(
        name=f'quanteeze_kquant_qmv_{mode}',
        input_names=['x', 'w', 'd', 'dmin', 'grid', 'idx'],
        output_names=['y'],
        header=_HEADER,
        source=_SOURCE,
    )


def _biases_of(experts):
    return experts.get('biases') if hasattr(experts, 'get') else getattr(experts, 'biases', None)


def takes(experts, x, indices):
    """Whether this call is a single-row decode that the Metal K-quant qmv kernel can serve."""
    if not ON[0] or getattr(experts, 'mode', None) not in MODES or not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    if x.ndim < 2 or x.shape[-2] != 1 or x.shape[-1] % 256 != 0:
        return False
    try:
        rows = int(np.prod(np.broadcast_shapes(tuple(x.shape[:-2]), tuple(indices.shape))))
    except ValueError:
        return False
    return 0 < rows <= MAX_ROWS


def run(experts, x, indices):
    """`x [..., 1, K]` through the experts named by `indices` (broadcast like `gather_mm`): `[..., 1, N]`."""
    from .ggml_quants import BODY
    shape = tuple(np.broadcast_shapes(tuple(x.shape[:-2]), tuple(indices.shape)))
    k, n = x.shape[-1], experts.weight.shape[1]
    rows = int(np.prod(shape))
    xs = mx.contiguous(mx.broadcast_to(x, shape + (1, k)).reshape(rows, k))
    idx = mx.broadcast_to(indices, shape).reshape(rows).astype(mx.int32)
    d = mx.contiguous(experts.scales)
    biases = _biases_of(experts)
    dmin = mx.contiguous(biases) if biases is not None else d
    y = _kernel(experts.mode)(
        inputs=[xs, mx.contiguous(experts.weight), d, dmin, _grid_for(experts.mode), idx],
        template=[
            ('T', x.dtype), ('K_BLOCKS', k // 256), ('N_SIZE', n), ('MODE', MODES[experts.mode]),
            ('ROWS', ROWS), ('BODY', BODY[experts.mode]),
        ],
        grid=(32, (n + ROWS - 1) // ROWS, rows),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]
    return y.reshape(*shape, 1, n)


def gather(experts, x, indices, unpacked):
    """`run()` once verified within `TOLERANCE` of `unpacked()` for `CHECKS` calls, else `unpacked()`."""
    key = (experts.mode, tuple(x.shape), x.dtype, tuple(indices.shape), tuple(experts.weight.shape))
    return _checked(key, lambda: run(experts, x, indices), unpacked)


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
  const uint lr = tid / 4, seg = (tid % 4) * 8;
  const uint sr = (sg / 2) * 16, sn = (sg % 2) * 16;
  simdgroup_float8x8 acc[2][2];
  for (int i = 0; i < 2; i++) for (int jj = 0; jj < 2; jj++) acc[i][jj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const ulong row = ((ulong)e * N_SIZE + n0 + lr) * (K_SIZE / 256);
  for (uint k0 = 0; k0 < K_SIZE; k0 += BK) {
    for (int i = 0; i < 8; i++)
      Xs[lr * LD + seg + i] = (int)lr < count ? float(x[(ulong)(r0 + lr) * K_SIZE + k0 + seg + i]) : 0.0f;
    const ulong at = row + k0 / 256;
    const device uint8_t* blk = w + at * BODY;
    const float dd = float(d[at]);
    const uint v0 = k0 % 256 + seg, sub = v0 / 16, h = v0 / 128, j = (v0 % 128) / 32, l0 = v0 % 32, hb = v0 / 32;
    if (MODE == 2) {                                            // Q2_K: scale and minimum, then codes
      const uint s = blk[sub];
      const float scale = dd * float(s & 15), low = float(dmin[at]) * float(s >> 4);
      for (int i = 0; i < 8; i++) {
        const uint q = (blk[16 + 32 * h + l0 + i] >> (2 * j)) & 3;
        Ws[lr * LD + seg + i] = scale * float(q) - low;
      }
    } else if (MODE == 3) {                                     // Q3_K: high bits, codes, 6-bit scales
      const uint low = sub < 8 ? (blk[96 + sub] & 15) : (blk[96 + sub - 8] >> 4);
      const uint high = (blk[104 + sub % 4] >> (2 * (sub / 4))) & 3;
      const float scale = dd * float(int(low | (high << 4)) - 32);
      for (int i = 0; i < 8; i++) {
        int q = int((blk[32 + 32 * h + l0 + i] >> (2 * j)) & 3);
        if (((blk[l0 + i] >> hb) & 1) == 0) q -= 4;
        Ws[lr * LD + seg + i] = scale * float(q);
      }
    } else {                                                    // IQ2_XS: 64B 16-bit codes, 8B half-scales
      const uint octad = v0 >> 3, ib32 = octad >> 2, l = octad & 3u;
      const uint q16 = uint(blk[2u * octad]) | (uint(blk[2u * octad + 1u]) << 8);
      const uint s = blk[64u + ib32];
      const float db = dd * (0.5f + float(l < 2u ? (s & 15u) : (s >> 4))) * 0.25f;
      const uint s7 = q16 >> 9;
      const uint sbyte = s7 | ((metal::popcount(s7) & 1u) << 7);
      const uint16_t g = grid[q16 & 511u];
      for (int i = 0; i < 8; i++) {
        const float sign = ((sbyte >> i) & 1u) ? -1.0f : 1.0f;
        Ws[lr * LD + seg + i] = db * IQ2_MAG[(g >> (2u * i)) & 3u] * sign;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 a[2], b[2];
      for (int i = 0; i < 2; i++) simdgroup_load(a[i], Xs + (sr + 8 * i) * LD + kk, LD);
      for (int jj = 0; jj < 2; jj++) simdgroup_load(b[jj], Ws + (sn + 8 * jj) * LD + kk, LD, ulong2(0, 0), true);
      for (int i = 0; i < 2; i++)
        for (int jj = 0; jj < 2; jj++) simdgroup_multiply_accumulate(acc[i][jj], a[i], b[jj], acc[i][jj]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = 0; i < 2; i++)
    for (int jj = 0; jj < 2; jj++) simdgroup_store(acc[i][jj], Ys + (sr + 8 * i) * LD + sn + 8 * jj, LD);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((int)lr < count)
    for (int i = 0; i < 8; i++) y[(ulong)(r0 + lr) * N_SIZE + n0 + seg + i] = static_cast<T>(Ys[lr * LD + seg + i]);
"""
TILE = 32


@lru_cache(maxsize=None)
def _tiled(mode):
    return mx.fast.metal_kernel(
        name=f'quanteeze_kquant_mm_{mode}',
        input_names=['x', 'w', 'd', 'dmin', 'grid', 'tiles'],
        output_names=['y'],
        header='#include <metal_simdgroup_matrix>\n' + _HEADER,
        source=_TILED,
    )


def takes_rows(experts, x, indices):
    """Whether this call is a multi-row `x [M, 1, K]`, `indices [M]` call that the tiled Metal kernel can serve."""
    if not ON[0] or getattr(experts, 'mode', None) not in MODES or not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    return (
        x.ndim == 3 and x.shape[1] == 1 and indices.ndim == 1 and x.shape[0] == indices.size
        and indices.size > 0 and experts.weight.shape[1] % TILE == 0 and x.shape[-1] % 256 == 0
    )


def run_rows(experts, x, indices):
    """`x [M, 1, K]` through the expert named by each row of `indices [M]`: `[M, 1, N]`."""
    from .ggml_quants import BODY
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
    d = mx.contiguous(experts.scales)
    biases = _biases_of(experts)
    dmin = mx.contiguous(biases) if biases is not None else d
    y = _tiled(experts.mode)(
        inputs=[mx.contiguous(xs), mx.contiguous(experts.weight), d, dmin, _grid_for(experts.mode),
                mx.array(tiles.reshape(-1))],
        template=[('T', x.dtype), ('K_SIZE', k), ('N_SIZE', n), ('MODE', MODES[experts.mode]), ('BODY', BODY[experts.mode])],
        grid=(128 * (n // TILE), len(tiles), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]
    if order is not None:
        y = y[mx.array(np.argsort(order).astype(np.int32))]
    return y.reshape(m, 1, n)


def gather_rows(experts, x, indices, unpacked):
    """`run_rows()` once verified within `TOLERANCE` of `unpacked()` for `CHECKS` calls, else `unpacked()`."""
    key = ('rows', experts.mode, x.shape[-1], experts.weight.shape[1], x.dtype)
    return _checked(key, lambda: run_rows(experts, x, indices), unpacked)


def _near(got, want):
    got, w = got.astype(mx.float32), want.astype(mx.float32)
    return (
        got.shape == w.shape
        and bool((mx.abs(got - w).max() <= TOLERANCE * mx.maximum(mx.abs(w).max(), 1e-30)).item())
    )


def _checked(key, mine, unpacked):
    return kernel_checks.served(sys.modules[__name__], key, mine, unpacked, 'k-quant kernel', same=_near)


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
