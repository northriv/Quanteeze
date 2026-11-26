# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Routed MoE experts in decode and speculative verify passes, with each expert's weights read once
for the tokens routed to it, and each row's arithmetic matching the one-token decode bit for bit.

In a decode step (`T = 1`), `fused()` runs all `k` routed slots through `gate_proj`, `up_proj`, and
`swiglu` in one Metal kernel, and `down_proj` in a second Metal kernel.  In a multi-token verify
pass (`1 < T <= verify_qmv.MAX_TOKENS`), `own()` runs `run_pair(gate_proj, up_proj)` followed by
`activation` and `run(down_proj)`, reading each chosen expert's weights once for all tokens in the
step that route to it.

Every shape is checked bit for bit against stock `SwitchGLU` (`stock()`) the first `CHECKS` times
it is met, and served by the custom kernels only when identical.  `QUANTEEZE_EXPERT_QMV=off` keeps
stock `SwitchGLU`.
"""
# Portions of the Metal kernel ports below are adapted from Apple MLX
# (Copyright © 2023-2026 Apple Inc., MIT License — see LICENSE).
from functools import lru_cache
import inspect
import re
import sys

import mlx.core as mx
import numpy as np

from . import kernel_checks, per_load, requant, verify_qmv

CHECKS = verify_qmv.CHECKS
ON = [True]                       # QUANTEEZE_EXPERT_QMV
PAIR = [True]                     # gate and up in one kernel (run_pair)
ROWS = [None]                     # max rows naming an expert a threadgroup computes at once (None: T)
STATS = dict(served=0, fused=0, checked=0, mismatched=0)
CHOICE, CHECKED = kernel_checks.tables('expert_qmv')
SAID = []


@per_load.on_let_go
def _let_go():
    """Clear per-load `ROWS` setting override on model unload."""
    ROWS[0] = None


def weighs(activation):
    """Whether a switch's activation takes the router's weights `(up, gate, weights)` (DeepSeek V4.1)."""
    try:
        return len(inspect.signature(activation).parameters) >= 3
    except (TypeError, ValueError):
        return False


_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint r = threadgroup_position_in_grid.z;
    const int e = idx[r];
    int before = 0;
    for (uint q = 0; q < r; ++q) {
      if (idx[q] == e) {
        before++;
      }
    }
    if (before % VERIFY_T != 0) {
      return;
    }
    int rows[VERIFY_T];
    int n = 0;
    for (uint q = r; q < R_SIZE; ++q) {
      if (idx[q] == e && n < VERIFY_T) {
        rows[n] = q;
        n++;
      }
    }
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int packs_per_thread = __PACKS__;
    constexpr int pack_factor = get_pack_factor<BITS, 32>();
    constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * SIMD_SIZE;
    constexpr int scale_step_per_thread = GS / values_per_thread;
    const int in_vec_size = K_SIZE;
    const int out_vec_size = N_SIZE;
    typedef float U;
    thread U x_thread[VERIFY_T][values_per_thread];
    thread U result[VERIFY_T][results_per_simdgroup];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < results_per_simdgroup; ++row) {
        result[t][row] = 0;
      }
    }
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / GS;
    const int out_row = n_tile * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
    const device uint8_t* ws = (const device uint8_t*)w + (ulong)e * out_vec_size * in_vec_size_w;
    const device T* sc = scales + (ulong)e * out_vec_size * in_vec_size_g;
    const device T* bs = biases + (ulong)e * out_vec_size * in_vec_size_g;
    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    bs += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* xs = x + simd_lid * values_per_thread;
    U sum[VERIFY_T];
    for (int k = 0; k < in_vec_size; k += block_size) {
      for (int t = 0; t < VERIFY_T; ++t) {
        if (t < n) {
          sum[t] = load_vector<T, U, values_per_thread, BITS>(xs + rows[t] * in_vec_size, x_thread[t]);
        }
      }
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        for (int t = 0; t < VERIFY_T; ++t) {
          if (t < n) {
            result[t][row] += qdot<U, values_per_thread, BITS>(wl, x_thread[t], s, b, sum[t]);
          }
        }
      }
      ws += block_size * bytes_per_pack / pack_factor;
      sc += block_size / GS;
      bs += block_size / GS;
      xs += block_size;
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      for (int t = 0; t < VERIFY_T; ++t) {
        if (t < n) {
          U s = simd_sum(result[t][row]);
          if (simd_lid == 0) {
            y[rows[t] * out_vec_size + out_row + row] = static_cast<T>(s);
          }
        }
      }
    }
"""


def _generic(source):
    for old, new in (
            ('    for (int k = 0; k < in_vec_size; k += block_size) {\n',
             '    int k = 0;\n    for (; k < in_vec_size - block_size; k += block_size) {\n'),
            ('      xs += block_size;\n    }\n',
             '      xs += block_size;\n    }\n'
             '    const int remaining = clamp(static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0,\n'
             '                                values_per_thread);\n'
             '    if (remaining > 0) {\n'
             '      for (int t = 0; t < VERIFY_T; ++t) {\n'
             '        if (t < n) {\n'
             '          sum[t] = load_vector_safe<T, U, values_per_thread, BITS>(xs + rows[t] * in_vec_size, x_thread[t],\n'
             '                                                                   remaining);\n'
             '        }\n'
             '      }\n'
             '      for (int row = 0; row < results_per_simdgroup; row++) {\n'
             '        auto tw = (const device uint8_t*)(ws + row * in_vec_size_w);\n'
             '        U ts = sc[row * in_vec_size_g];\n'
             '        U tb = bs[row * in_vec_size_g];\n'
             '        for (int t = 0; t < VERIFY_T; ++t) {\n'
             '          if (t < n) {\n'
             '            result[t][row] += qdot_safe<U, values_per_thread, BITS>(tw, x_thread[t], ts, tb, sum[t], remaining);\n'
             '          }\n'
             '        }\n'
             '      }\n'
             '    }\n')):
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


def _paired(source):
    for old, new in (
            ('    thread U result[VERIFY_T][results_per_simdgroup];\n',
             '    thread U result[VERIFY_T][results_per_simdgroup];\n'
             '    thread U result2[VERIFY_T][results_per_simdgroup];\n'),
            ('        result[t][row] = 0;\n', '        result[t][row] = 0;\n        result2[t][row] = 0;\n'),
            ('    ws += out_row * in_vec_size_w',
             '    const device uint8_t* ws2 = (const device uint8_t*)w2 + (ulong)e * out_vec_size * in_vec_size_w;\n'
             '    const device T* sc2 = scales2 + (ulong)e * out_vec_size * in_vec_size_g;\n'
             '    const device T* bs2 = biases2 + (ulong)e * out_vec_size * in_vec_size_g;\n'
             '    ws2 += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;\n'
             '    sc2 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;\n'
             '    bs2 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;\n'
             '    ws += out_row * in_vec_size_w'),
            ('            result[t][row] += qdot<U, values_per_thread, BITS>(wl, x_thread[t], s, b, sum[t]);\n',
             '            result[t][row] += qdot<U, values_per_thread, BITS>(wl, x_thread[t], s, b, sum[t]);\n'
             '            result2[t][row] += qdot<U, values_per_thread, BITS>(wl2, x_thread[t], s2, b2, sum[t]);\n'),
            ('        U b = bl[0];\n',
             '        U b = bl[0];\n'
             '        auto wl2 = (const device uint8_t*)(ws2 + row * in_vec_size_w);\n'
             '        U s2 = sc2[row * in_vec_size_g];\n'
             '        U b2 = bs2[row * in_vec_size_g];\n'),
            ('      bs += block_size / GS;\n',
             '      bs += block_size / GS;\n      ws2 += block_size * bytes_per_pack / pack_factor;\n'
             '      sc2 += block_size / GS;\n      bs2 += block_size / GS;\n'),
            ('          U s = simd_sum(result[t][row]);\n          if (simd_lid == 0) {\n'
             '            y[rows[t] * out_vec_size + out_row + row] = static_cast<T>(s);\n',
             '          U s = simd_sum(result[t][row]);\n          U s2 = simd_sum(result2[t][row]);\n'
             '          if (simd_lid == 0) {\n'
             '            y[rows[t] * out_vec_size + out_row + row] = static_cast<T>(s);\n'
             '            y2[rows[t] * out_vec_size + out_row + row] = static_cast<T>(s2);\n')):
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    if 'qdot_safe' in source:
        for old, new in (
                ('        U tb = bs[row * in_vec_size_g];\n',
                 '        U tb = bs[row * in_vec_size_g];\n'
                 '        auto tw2 = (const device uint8_t*)(ws2 + row * in_vec_size_w);\n'
                 '        U ts2 = sc2[row * in_vec_size_g];\n'
                 '        U tb2 = bs2[row * in_vec_size_g];\n'),
                ('            result[t][row] += qdot_safe<U, values_per_thread, BITS>(tw, x_thread[t], ts, tb, sum[t], remaining);\n',
                 '            result[t][row] += qdot_safe<U, values_per_thread, BITS>(tw, x_thread[t], ts, tb, sum[t], remaining);\n'
                 '            result2[t][row] += qdot_safe<U, values_per_thread, BITS>(tw2, x_thread[t], ts2, tb2, sum[t], remaining);\n')):
            assert source.count(old) == 1, old
            source = source.replace(old, new)
    return source


@lru_cache(maxsize=None)
def fp_helpers():
    kernels = verify_qmv._header_path().parent
    try:
        types = ''.join((kernels / name).read_text().replace('#pragma once', '') for name in ('fp4.h', 'fp8.h'))
        text = (kernels / 'fp_quantized.h').read_text()
    except OSError:
        return None
    start = text.find('template <int wsize = 8, int bits = 4>\ninline constexpr short get_pack_factor')
    end = text.find('inline void qouter(')
    if start < 0 or end < 0:
        return None
    body = text[start:text.rfind('template <', start, end)]
    if not all(re.search(rf'\b{name}\(', body) for name in
               ('get_pack_factor', 'get_bytes_per_pack', 'dequantize_scale', 'load_vector', 'load_vector_safe',
                'qdot', 'qdot_safe')):
        return None
    return types + body


def _fp_header(bits, group_size):
    return ('using namespace metal;\n#define MLX_MTL_CONST static constant constexpr const\n'
            'MLX_MTL_CONST int SIMD_SIZE = 32;\nMLX_MTL_CONST int QUAD_SIZE = 4;\n' + fp_helpers()
            + f'\nconstant constexpr int BITS = {int(bits)};\nconstant constexpr int GS = {int(group_size)};\n')


def _fp_source(paired, port):
    def both(first, second):
        return first + (second if paired else '')
    generic = port == 'generic'
    body = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint r = threadgroup_position_in_grid.z;
    const int e = idx[r];
    int before = 0;
    for (uint q = 0; q < r; ++q) {
      if (idx[q] == e) {
        before++;
      }
    }
    if (before % VERIFY_T != 0) {
      return;
    }
    int rows[VERIFY_T];
    int n = 0;
    for (uint q = r; q < R_SIZE; ++q) {
      if (idx[q] == e && n < VERIFY_T) {
        rows[n] = q;
        n++;
      }
    }
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int packs_per_thread = __PACKS__;
    constexpr int pack_factor = get_pack_factor<32, BITS>();
    constexpr int bytes_per_pack = get_bytes_per_pack<32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * SIMD_SIZE;
    constexpr int scale_step_per_thread = GS / values_per_thread;
    const int in_vec_size = K_SIZE;
    const int out_vec_size = N_SIZE;
    typedef float U;
    thread U x_thread[VERIFY_T][values_per_thread];
    thread U result[VERIFY_T][results_per_simdgroup];
__RESULT2__    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < results_per_simdgroup; ++row) {
__ZERO__      }
    }
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / GS;
    const int out_row = n_tile * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
__POINTERS__    const device T* xs = x + simd_lid * values_per_thread;
    int k = 0;
    for (; k < __WHOLE__; k += block_size) {
      for (int t = 0; t < VERIFY_T; ++t) {
        if (t < n) {
          load_vector<T, U, values_per_thread>(xs + rows[t] * in_vec_size, x_thread[t]);
        }
      }
__ROWS__      xs += block_size;
    }
__TAIL__    for (int row = 0; row < results_per_simdgroup; row++) {
      for (int t = 0; t < VERIFY_T; ++t) {
        if (t < n) {
__OUT__        }
      }
    }
"""
    pointers = (both('    const device uint8_t* ws = (const device uint8_t*)w + (ulong)e * out_vec_size * in_vec_size_w;\n',
                     '    const device uint8_t* ws2 = (const device uint8_t*)w2 + (ulong)e * out_vec_size * in_vec_size_w;\n')
                + both('    const device uint8_t* sc = scales + (ulong)e * out_vec_size * in_vec_size_g;\n',
                       '    const device uint8_t* sc2 = scales2 + (ulong)e * out_vec_size * in_vec_size_g;\n')
                + both('    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;\n',
                       '    ws2 += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;\n')
                + both('    sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;\n',
                       '    sc2 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;\n'))
    rows = ('      for (int row = 0; row < results_per_simdgroup; row++) {\n'
            + both('        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);\n',
                   '        auto wl2 = (const device uint8_t*)(ws2 + row * in_vec_size_w);\n')
            + both('        U s = dequantize_scale<U, GS>(sc[row * in_vec_size_g]);\n',
                   '        U s2 = dequantize_scale<U, GS>(sc2[row * in_vec_size_g]);\n')
            + '        for (int t = 0; t < VERIFY_T; ++t) {\n          if (t < n) {\n'
            + both('            result[t][row] += qdot<U, values_per_thread, BITS>(wl, x_thread[t], s);\n',
                   '            result2[t][row] += qdot<U, values_per_thread, BITS>(wl2, x_thread[t], s2);\n')
            + '          }\n        }\n      }\n'
            + both('      ws += block_size * bytes_per_pack / pack_factor;\n',
                   '      ws2 += block_size * bytes_per_pack / pack_factor;\n')
            + both('      sc += block_size / GS;\n', '      sc2 += block_size / GS;\n'))
    tail = ''
    if generic:
        tail = ('    const int remaining = clamp(static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0,\n'
                '                                values_per_thread);\n'
                '    if (remaining > 0) {\n'
                '      for (int t = 0; t < VERIFY_T; ++t) {\n        if (t < n) {\n'
                '          load_vector_safe<T, U, values_per_thread>(xs + rows[t] * in_vec_size, x_thread[t], remaining);\n'
                '        }\n      }\n'
                + rows.split('      ws += block_size')[0] + '    }\n')
    out = ('          U s = simd_sum(result[t][row]);\n'
           + ('          U s2 = simd_sum(result2[t][row]);\n' if paired else '')
           + '          if (simd_lid == 0) {\n'
           '            y[rows[t] * out_vec_size + out_row + row] = static_cast<T>(s);\n'
           + ('            y2[rows[t] * out_vec_size + out_row + row] = static_cast<T>(s2);\n' if paired else '')
           + '          }\n')
    for old, new in (('__PACKS__', '1' if generic else '2'),
                     ('__RESULT2__', '    thread U result2[VERIFY_T][results_per_simdgroup];\n' if paired else ''),
                     ('__ZERO__', '        result[t][row] = 0;\n' + ('        result2[t][row] = 0;\n' if paired else '')),
                     ('__POINTERS__', pointers), ('__WHOLE__', 'in_vec_size - block_size' if generic else 'in_vec_size'),
                     ('__ROWS__', rows), ('__TAIL__', tail), ('__OUT__', out)):
        body = body.replace(old, new)
    return body


@lru_cache(maxsize=None)
def _kernel(bits, group_size, paired=False, port='fast', mode='affine'):
    if mode != 'affine':
        tag = ('2' if paired else '') + ('' if port == 'fast' else '_generic')
        names = ['x', 'w', 'scales', 'w2', 'scales2', 'idx'] if paired else ['x', 'w', 'scales', 'idx']
        return mx.fast.metal_kernel(name=f'quanteeze_expert_{mode}{tag}_b{bits}_gs{group_size}', input_names=names,
                                    output_names=['y', 'y2'] if paired else ['y'],
                                    header=_fp_header(bits, group_size), source=_fp_source(paired, port))
    if port == 'generic':
        source = _generic(_SOURCE.replace('__PACKS__', '1'))
    else:
        source = _SOURCE.replace('__PACKS__', '1' if bits <= 2 else '2')
    tag = '' if port == 'fast' else '_generic'
    if paired:
        return mx.fast.metal_kernel(name=f'quanteeze_expert_qmv2{tag}_b{bits}_gs{group_size}',
                                    input_names=['x', 'w', 'scales', 'biases', 'w2', 'scales2', 'biases2', 'idx'],
                                    output_names=['y', 'y2'], header=verify_qmv._header(bits, group_size),
                                    source=_paired(source))
    return mx.fast.metal_kernel(name=f'quanteeze_expert_qmv{tag}_b{bits}_gs{group_size}',
                                input_names=['x', 'w', 'scales', 'biases', 'idx'], output_names=['y'],
                                header=verify_qmv._header(bits, group_size), source=source)


def port_of(experts, k):
    block = 32 * (32 // experts.bits) * (1 if experts.bits <= 2 else 2)
    if getattr(experts, 'mode', 'affine') != 'affine':
        block = 32 * 8 * 2
    return 'fast' if k % 512 == 0 and k % block == 0 else 'generic'


def applies(experts, k):
    bits = getattr(experts, 'bits', None)
    group_size = getattr(experts, 'group_size', None)
    mode = getattr(experts, 'mode', 'affine')
    if bits is None or group_size is None:
        return False
    n = experts.weight.shape[1]
    if mode == 'mxfp4':
        own = (bits == 4 and group_size == 32 and fp_helpers() is not None
               and experts.scales.dtype == mx.uint8)
    else:
        own = (mode == 'affine' and getattr(experts, 'biases', None) is not None and bits in (1, 2, 3, 4, 8)
               and verify_qmv.helpers() is not None)
    return (own and n % 8 == 0 and k % group_size == 0 and k == experts.scales.shape[-1] * group_size
            and mx.metal.is_available() and mx.default_device() == mx.gpu)


def run(experts, x, idx, tokens):
    r, k = x.shape
    n = experts.weight.shape[1]
    mode = getattr(experts, 'mode', 'affine')
    arrays = [experts.weight, experts.scales] + ([experts.biases] if mode == 'affine' else [])
    return _kernel(int(experts.bits), int(experts.group_size), False, port_of(experts, k), mode)(
        inputs=[mx.contiguous(x), *arrays, idx],
        template=[('T', x.dtype), ('VERIFY_T', int(tokens)), ('K_SIZE', int(k)), ('N_SIZE', int(n)),
                  ('R_SIZE', int(r))],
        grid=(32, 2 * (n // 8), r), threadgroup=(32, 2, 1), output_shapes=[(r, n)], output_dtypes=[x.dtype])[0]


def run_pair(first, second, x, idx, tokens):
    r, k = x.shape
    n = first.weight.shape[1]
    mode = getattr(first, 'mode', 'affine')
    arrays = [[e.weight, e.scales] + ([e.biases] if getattr(e, 'mode', 'affine') == 'affine' else [])
              for e in (first, second)]
    return tuple(_kernel(int(first.bits), int(first.group_size), True, port_of(first, k), mode)(
        inputs=[mx.contiguous(x), *arrays[0], *arrays[1], idx],
        template=[('T', x.dtype), ('VERIFY_T', int(tokens)), ('K_SIZE', int(k)), ('N_SIZE', int(n)),
                  ('R_SIZE', int(r))],
        grid=(32, 2 * (n // 8), r), threadgroup=(32, 2, 1), output_shapes=[(r, n), (r, n)],
        output_dtypes=[x.dtype, x.dtype]))


_TILE = r"""
template <typename T, int BITS, int GS, bool GENERIC>
METAL_FUNC void expert_tile(const device uint32_t* w, const device T* scales, const device T* biases,
                            int e, const device T* x, int K, int N, int out_row, uint simd_lid,
                            thread float* result) {
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = GENERIC ? 1 : (BITS <= 2 ? 1 : 2);
  constexpr int pack_factor = get_pack_factor<BITS, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = GS / values_per_thread;
  typedef float U;
  thread U x_thread[values_per_thread];
  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / GS;
  const device uint8_t* ws = (const device uint8_t*)w + (ulong)e * N * in_vec_size_w + out_row * in_vec_size_w
      + simd_lid * packs_per_thread * bytes_per_pack;
  const device T* sc = scales + (ulong)e * N * in_vec_size_g + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* bs = biases + (ulong)e * N * in_vec_size_g + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xs = x + simd_lid * values_per_thread;
  int k = 0;
  const int whole = GENERIC ? K - block_size : K;
  for (; k < whole; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, BITS>(xs, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      U s = sc[row * in_vec_size_g];
      U b = bs[row * in_vec_size_g];
      result[row] += qdot<U, values_per_thread, BITS>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / GS;
    bs += block_size / GS;
    xs += block_size;
  }
  if (GENERIC) {
    const int remaining = clamp(static_cast<int>(K - k - simd_lid * values_per_thread), 0, values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, BITS>(xs, x_thread, remaining);
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        U s = sc[row * in_vec_size_g];
        U b = bs[row * in_vec_size_g];
        result[row] += qdot_safe<U, values_per_thread, BITS>(wl, x_thread, s, b, sum, remaining);
      }
    }
  }
}
template <typename T, int BITS, int GS, bool GENERIC>
METAL_FUNC void expert_tile_pair(const device uint32_t* w, const device T* scales, const device T* biases,
                                 const device uint32_t* w2, const device T* scales2, const device T* biases2,
                                 int e, const device T* x, int K, int N, int out_row, uint simd_lid,
                                 thread float* result, thread float* result2) {
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = GENERIC ? 1 : (BITS <= 2 ? 1 : 2);
  constexpr int pack_factor = get_pack_factor<BITS, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = GS / values_per_thread;
  typedef float U;
  thread U x_thread[values_per_thread];
  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / GS;
  const ulong w_at = (ulong)e * N * in_vec_size_w + out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  const ulong g_at = (ulong)e * N * in_vec_size_g + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device uint8_t* ws = (const device uint8_t*)w + w_at;
  const device uint8_t* ws2 = (const device uint8_t*)w2 + w_at;
  const device T* sc = scales + g_at;
  const device T* sc2 = scales2 + g_at;
  const device T* bs = biases + g_at;
  const device T* bs2 = biases2 + g_at;
  const device T* xs = x + simd_lid * values_per_thread;
  int k = 0;
  const int whole = GENERIC ? K - block_size : K;
  for (; k < whole; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, BITS>(xs, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] += qdot<U, values_per_thread, BITS>((const device uint8_t*)(ws + row * in_vec_size_w), x_thread,
                                                       U(sc[row * in_vec_size_g]), U(bs[row * in_vec_size_g]), sum);
      result2[row] += qdot<U, values_per_thread, BITS>((const device uint8_t*)(ws2 + row * in_vec_size_w), x_thread,
                                                        U(sc2[row * in_vec_size_g]), U(bs2[row * in_vec_size_g]), sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    ws2 += block_size * bytes_per_pack / pack_factor;
    sc += block_size / GS;
    sc2 += block_size / GS;
    bs += block_size / GS;
    bs2 += block_size / GS;
    xs += block_size;
  }
  if (GENERIC) {
    const int remaining = clamp(static_cast<int>(K - k - simd_lid * values_per_thread), 0, values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, BITS>(xs, x_thread, remaining);
      for (int row = 0; row < results_per_simdgroup; row++) {
        result[row] += qdot_safe<U, values_per_thread, BITS>((const device uint8_t*)(ws + row * in_vec_size_w),
            x_thread, U(sc[row * in_vec_size_g]), U(bs[row * in_vec_size_g]), sum, remaining);
        result2[row] += qdot_safe<U, values_per_thread, BITS>((const device uint8_t*)(ws2 + row * in_vec_size_w),
            x_thread, U(sc2[row * in_vec_size_g]), U(bs2[row * in_vec_size_g]), sum, remaining);
      }
    }
  }
}
"""

_GATE_UP = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint r = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const int e = int(indices[r]);
    const int out_row = n_tile * 8 + simd_gid * 4;
    const device T* xr = x + (r / K_SLOTS) * D_SIZE;
    float rg[4] = {0, 0, 0, 0};
    float ru[4] = {0, 0, 0, 0};
    expert_tile_pair<T, BITS, GS, GU>(gw, gs, gb, uw, us, ub, e, xr, D_SIZE, I_SIZE, out_row,
                                      simd_lid, rg, ru);
    for (int row = 0; row < 4; row++) {
      float a = simd_sum(rg[row]);
      float b = simd_sum(ru[row]);
      if (simd_lid == 0) {
        T gate = static_cast<T>(a);
        T up = static_cast<T>(b);
        auto ex = 1 / (1 + metal::precise::exp(metal::abs(gate)));
        T sg = (gate < 0) ? ex : 1 - ex;
        T silu = gate * sg;
        y[r * I_SIZE + out_row + row] = silu * up;
      }
    }
"""

_DOWN = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint r = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const int e = int(indices[r]);
    const int out_row = n_tile * 8 + simd_gid * 4;
    const device T* xr = h + r * I_SIZE;
    float rd[4] = {0, 0, 0, 0};
    expert_tile<T, BITS, GS, GD>(dw, ds, db, e, xr, I_SIZE, D_SIZE, out_row, simd_lid, rd);
    for (int row = 0; row < 4; row++) {
      float a = simd_sum(rd[row]);
      if (simd_lid == 0) {
        y[r * D_SIZE + out_row + row] = static_cast<T>(a);
      }
    }
"""
FUSED = [True]
VERIFY = [True]


@lru_cache(maxsize=None)
def _fused_kernels():
    header = verify_qmv._header(4, 64).split('\nconstant constexpr int BITS')[0] + '\n' + _TILE
    gate_up = mx.fast.metal_kernel(name='quanteeze_experts_gate_up',
                                   input_names=['x', 'indices', 'gw', 'gs', 'gb', 'uw', 'us', 'ub'],
                                   output_names=['y'], header=header, source=_GATE_UP)
    down = mx.fast.metal_kernel(name='quanteeze_experts_down',
                                input_names=['h', 'indices', 'dw', 'ds', 'db'],
                                output_names=['y'], header=header, source=_DOWN)
    return gate_up, down


def fusable(switch, x):
    """Whether `fused()` computes this step: one-token decode or multi-token verify (`VERIFY[0]`),
    with affine `gate_proj`, `up_proj`, and `down_proj` with `bits in (1, 2, 4, 8)`."""
    projs = (switch.gate_proj, switch.up_proj, switch.down_proj)
    return ((x.shape[1] == 1 or VERIFY[0]) and verify_qmv.helpers() is not None
            and len({(getattr(p, 'bits', None), getattr(p, 'group_size', None)) for p in projs}) == 1
            and getattr(switch.gate_proj, 'bits', None) in (1, 2, 4, 8)
            and all(getattr(p, 'mode', 'affine') == 'affine' for p in projs)
            and eligible(switch, x, None))


def fused(switch, x, indices):
    """A `SwitchGLU`'s experts for `x [B, T, D]` on `indices [B, T, k]`: `[B, T, k, D]` in two
    Metal kernels (`gate+up+swiglu`, then `down`), with each token's `k` slots dispatched as rows."""
    b0, t0, k, d = x.shape[0], x.shape[1], indices.shape[-1], x.shape[-1]
    t = b0 * t0
    flat_idx = indices.reshape(-1).astype(mx.int32)
    gp, up, dp = switch.gate_proj, switch.up_proj, switch.down_proj
    i_size = gp.weight.shape[1]
    template = [
        ('T', x.dtype), ('D_SIZE', int(d)), ('I_SIZE', int(i_size)), ('K_SLOTS', int(k)),
        ('BITS', int(gp.bits)), ('GS', int(gp.group_size)),
        ('GU', port_of(gp, d) == 'generic'), ('GD', port_of(dp, i_size) == 'generic'),
    ]
    gate_up, down = _fused_kernels()
    hidden = gate_up(
        inputs=[mx.contiguous(x.reshape(-1, d)), flat_idx,
                gp.weight, gp.scales, gp.biases, up.weight, up.scales, up.biases],
        template=template, grid=(32, 2 * (i_size // 8), t * k), threadgroup=(32, 2, 1),
        output_shapes=[(t * k, i_size)], output_dtypes=[x.dtype])[0]
    y = down(
        inputs=[hidden, flat_idx, dp.weight, dp.scales, dp.biases],
        template=template, grid=(32, 2 * (d // 8), t * k), threadgroup=(32, 2, 1),
        output_shapes=[(t * k, d)], output_dtypes=[x.dtype])[0]
    return y.reshape(b0, t0, k, d)


def eligible(switch, x, indices):
    if not (x.ndim == 3 and x.dtype in (mx.bfloat16, mx.float16)
            and ((x.shape[0] == 1 and 1 <= x.shape[1] <= verify_qmv.MAX_TOKENS)
                 or (x.shape[1] == 1 and 1 <= x.shape[0] <= verify_qmv.MAX_TOKENS))):
        return False
    gp = switch.gate_proj
    bits = getattr(gp, 'bits', None)
    gs = getattr(gp, 'group_size', None)
    mode = getattr(gp, 'mode', 'affine')
    stamp = (bits, gs, mode, id(gp.weight), x.dtype, x.shape[-1])
    kept = switch.__dict__.get('_expert_qmv')
    if kept is None or kept[0] != stamp:
        d = x.shape[-1]
        ok = bits in (1, 2, 4, 8) and gp.weight.ndim == 3 and gp.weight.shape[-1] * 32 // bits == d
        if ok:
            for p in (switch.gate_proj, switch.up_proj, switch.down_proj):
                p_mode = getattr(p, 'mode', 'affine')
                if p.weight.ndim != 3 or not applies(p, p.weight.shape[-1] * 32 // p.bits) or (p_mode == 'affine' and p.scales.dtype != x.dtype):
                    ok = False
                    break
        kept = (stamp, ok, (int(bits or 0), int(gs or 0), mode))
        object.__setattr__(switch, '_expert_qmv', kept)
    return kept[1]


def key(switch, x, indices, weighed=False):
    kept = switch.__dict__.get('_expert_qmv')
    info = kept[2] if kept else (int(getattr(switch.gate_proj, 'bits', 0)),
                                 int(getattr(switch.gate_proj, 'group_size', 0)),
                                 getattr(switch.gate_proj, 'mode', 'affine'))
    t_key = x.shape[1] if x.shape[0] == 1 else (x.shape[0], x.shape[1])
    return (t_key, indices.shape[-1], x.shape[-1], x.dtype, info, weighed)


def own(switch, x, indices, weights=None):
    """A `SwitchGLU`'s experts for `x [B, T, D]` on `indices [B, T, k]`: `[B, T, k, D]`, with
    each expert's weights read once across the `B * T` tokens routed to it."""
    b0, t0, k, d = x.shape[0], x.shape[1], indices.shape[-1], x.shape[-1]
    t = b0 * t0
    rows = mx.broadcast_to(x.reshape(t, 1, d), (t, k, d)).reshape(t * k, d)
    idx = indices.astype(mx.int32).reshape(-1)
    per = t if ROWS[0] is None else max(1, min(t, ROWS[0]))
    if PAIR[0]:
        gate, up = run_pair(switch.gate_proj, switch.up_proj, rows, idx, per)
    else:
        gate, up = run(switch.gate_proj, rows, idx, per), run(switch.up_proj, rows, idx, per)
    hidden = switch.activation(up, gate) if weights is None else \
        switch.activation(up, gate, weights.reshape(t * k, 1))
    return run(switch.down_proj, hidden, idx, per).reshape(b0, t0, k, d)


def _proj_call(proj, x, indices, sorted_indices=False, rows_plan=None):
    bits = getattr(proj, 'bits', None)
    mode = getattr(proj, 'mode', 'affine')
    if bits is not None and not requant.native(bits, mode):
        def unpacked():
            arrays = [proj.weight, getattr(proj, 'scales', None), getattr(proj, 'biases', None)]
            idx = indices
            uniq, inv = np.unique(np.array(indices).reshape(-1), return_inverse=True)
            if 0 < uniq.size < proj.weight.shape[0]:
                sel = mx.array(uniq.astype(np.int32))
                arrays = [a[sel] if a is not None else None for a in arrays]
                idx = mx.array(inv.astype(np.uint32)).reshape(indices.shape)
            w = requant.dequantize(*arrays, proj.group_size, bits, mode=mode, dtype=x.dtype).astype(x.dtype)
            return mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=sorted_indices)

        prefer_rows = rows_plan is not None or (indices.ndim == 1 and indices.size >= 16)
        if isinstance(mode, str) and mode.startswith('gguf:'):
            from . import gguf_qmv
            if prefer_rows and gguf_qmv.takes_rows(proj, x, indices):
                return gguf_qmv.gather_rows(proj, x, indices, unpacked, plan=rows_plan)
            if gguf_qmv.takes(proj, x, indices):
                return gguf_qmv.gather(proj, x, indices, unpacked)
            if gguf_qmv.takes_rows(proj, x, indices):
                return gguf_qmv.gather_rows(proj, x, indices, unpacked)
        from . import kquant_qmv
        if mode in kquant_qmv.MODES:
            if prefer_rows and kquant_qmv.takes_rows(proj, x, indices):
                return kquant_qmv.gather_rows(proj, x, indices, unpacked)
            if kquant_qmv.takes(proj, x, indices):
                return kquant_qmv.gather(proj, x, indices, unpacked)
            if kquant_qmv.takes_rows(proj, x, indices):
                return kquant_qmv.gather_rows(proj, x, indices, unpacked)
        return unpacked()
    return proj(x, indices, sorted_indices=sorted_indices)


def stock(switch, x, indices, weights=None):
    """Stock `SwitchGLU` computation (with `weights` for weighted activations and 1-bit fallback)."""
    shape = indices.shape
    x_exp = mx.expand_dims(x, (-2, -3))
    indices = mx.stop_gradient(indices)
    do_sort = indices.size >= 64
    idx = indices
    order = None
    inv_order = None
    rows_plan = None
    if do_sort:
        projs = (switch.up_proj, switch.gate_proj, switch.down_proj)
        if all(isinstance(getattr(p, 'mode', None), str) and p.mode.startswith('gguf:') for p in projs):
            from . import gguf_qmv
            if gguf_qmv.ON[0] and gguf_qmv._gpu_available():
                order_mx, inv_order, idx, rows_plan = gguf_qmv.prepare_rows(indices.flatten())
                x_flat = x_exp.flatten(0, -3)
                if order_mx is None:
                    x_exp = mx.broadcast_to(
                        mx.expand_dims(x_flat, 1),
                        (x_flat.shape[0], shape[-1], *x_flat.shape[1:]),
                    ).reshape(-1, *x_flat.shape[1:])
                else:
                    x_exp = x_flat[order_mx // shape[-1]]
                if weights is not None:
                    w_flat = weights.flatten()
                    weights = w_flat if order_mx is None else w_flat[order_mx]
        if rows_plan is None:
            order = mx.argsort(indices.flatten())
            x_exp = x_exp.flatten(0, -3)[order // shape[-1]]
            idx = indices.flatten()[order]
            if weights is not None:
                weights = weights.flatten()[order]
    up = _proj_call(switch.up_proj, x_exp, idx, sorted_indices=do_sort, rows_plan=rows_plan)
    gate = _proj_call(switch.gate_proj, x_exp, idx, sorted_indices=do_sort, rows_plan=rows_plan)
    if weights is not None:
        hidden = switch.activation(up, gate, weights[..., None, None])
    else:
        hidden = switch.activation(up, gate)
    y = _proj_call(switch.down_proj, hidden, idx, sorted_indices=do_sort, rows_plan=rows_plan)
    if do_sort:
        if rows_plan is not None:
            if inv_order is not None:
                y = y[inv_order]
            y = mx.unflatten(y, 0, shape)
        else:
            y = mx.unflatten(y[mx.argsort(order)], 0, shape)
    return y.squeeze(-2)


def decode(switch, x, indices, weights=None):
    """Run a decode (`T = 1`) or verify (`1 < T <= MAX_TOKENS`) step on `switch`."""
    if ON[0] and weights is None and not mx.metal.is_available():
        from . import cuda_experts, gguf_qmv
        if cuda_experts.ON[0] and cuda_experts.takes(switch, x, indices):
            return cuda_experts.decode(switch, x, indices, lambda: stock(switch, x, indices))
        if gguf_qmv.ON[0] and gguf_qmv.takes_switch(switch, x, indices):
            return gguf_qmv.decode_switch(switch, x, indices, lambda: stock(switch, x, indices))
    if not ON[0] or not eligible(switch, x, indices):
        return stock(switch, x, indices, weights)
    k = key(switch, x, indices, weights is not None)
    if FUSED[0] and weights is None and fusable(switch, x):
        return _checked(('fused',) + k, lambda: fused(switch, x, indices), lambda: stock(switch, x, indices), 'fused')
    return _checked(k, lambda: own(switch, x, indices, weights), lambda: stock(switch, x, indices, weights), 'served')


def _checked(k, mine, theirs, stat):
    return kernel_checks.served(sys.modules[__name__], k, mine, theirs, "experts' kernel", stat=stat)


def install_switch(switch):
    """Hook a `SwitchGLU` (or `WeightedSwitchGLU`) so decode and verify passes use `decode()`."""
    cls = type(switch)
    if getattr(cls, '_quanteeze_expert_qmv', False):
        return switch
    orig_call = cls.__call__
    has_weights = weighs(switch.activation)
    has_combine = not has_weights and callable(getattr(cls, '_combine', None))

    class _FastSwitch(cls):
        _quanteeze_expert_qmv = True

        def __call__(self, x, indices, *args, **kwargs):
            w = (args[0] if args else kwargs.get('weights')) if has_weights else None
            combine = None
            if has_combine and (args or kwargs):
                c_weights = args[0] if len(args) >= 1 else kwargs.get('weights')
                c_shared = args[1] if len(args) >= 2 else kwargs.get('shared')
                c_residual = args[2] if len(args) >= 3 else kwargs.get('residual')
                if c_weights is not None or c_shared is not None or c_residual is not None:
                    combine = (c_weights, c_shared, c_residual)
            if x.ndim == 3 and ((x.shape[0] == 1 and 1 <= x.shape[1] <= verify_qmv.MAX_TOKENS)
                                or (x.shape[1] == 1 and 1 <= x.shape[0] <= verify_qmv.MAX_TOKENS)):
                y = decode(self, x, indices, w)
                return self._combine(y, *combine) if combine is not None else y
            if any(
                getattr(p, 'bits', None) is not None and not requant.native(p.bits, getattr(p, 'mode', 'affine'))
                for p in (self.gate_proj, self.up_proj, self.down_proj)
            ):
                y = stock(self, x, indices, w)
                return self._combine(y, *combine) if combine is not None else y
            return orig_call(self, x, indices, *args, **kwargs)

    _FastSwitch.__name__ = cls.__name__
    switch.__class__ = _FastSwitch
    return switch


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
