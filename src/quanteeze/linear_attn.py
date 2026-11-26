# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Flash-Next's linear attention (its gated delta layers) in a decode token with fewer kernels on the
GPU's chain, each bit for bit what it replaces.

With quanteeze.cuda_kernels.FUSE (QUANTEEZE_CUDA_FUSE) the conv's kernel also writes the cache's
window slid by the token (update_window's slice of the concatenation, which then is not made: three
copies a layer) where the cache keeps no lengths or history, and the delta rule's kernel computes the
step's decay and beta itself (mlx-vlm's compiled _compute_g_beta: exp(-exp(A_log) * softplus(a +
dt_bias)) and sigmoid(b), MLX's functors in its order; two kernels a layer).  None is taken on trust:
the first CHECKS times a shape is met, each kernel's output is compared bit for bit with mlx-vlm's ops,
and it serves the shape only if they are the same.  Covers single-request decode steps and CUDA multi-request
batch decode steps for 2 to ROWS_MOST requests (QUANTEEZE_LINEAR_ATTN_KERNELS=off: mlx-vlm's).
"""
# Portions of the Metal/CUDA kernel helpers below are adapted from Apple MLX
# (Copyright © 2023-2026 Apple Inc., MIT License — see LICENSE).
import sys
from functools import lru_cache

import mlx.core as mx

from . import cuda_kernels, kernel_checks, metal_kernels

CHECKS = 3
ROWS_MOST = 8                   # concurrent requests at most that decode_batch() takes (2 to 8)
ON = [True]                     # QUANTEEZE_LINEAR_ATTN_KERNELS
DELTA = [True]                  # QUANTEEZE_CUDA_DELTA: the delta rule's step on CUDA
SLIDE = [True]                  # slide the cached conv window inside the Metal conv kernel on decode
STATS = dict(conv=0, slide=0, norm=0, delta=0, gating=0, conv_rows=0, delta_rows=0, gating_rows=0, norm_rows=0,
             conv_batch=0, slide_batch=0, delta_batch=0, gating_batch=0, norm_batch=0, checked=0, mismatched=0, few=0)
CHOICE, CHECKED = kernel_checks.tables('linear_attn')  # (part, shape, dtype): whether it matched the original
SAID = []

SHARED = dict(fused=0)          # share_fused_linears(): projections now slices of their fused copy


def share_fused_linears(ops=None):
    """mlx-vlm fuses a gated delta layer's four input projections (in_proj_qkv, _z, _b, _a) for a
    decode token (speculative.ops.linear._decode_quantized_linears_fused): their weights, scales
    and biases concatenated once and kept on the first, the four kept besides for a prompt.  Both
    stayed in memory: Flash-Next's 36 such layers held 1.6 GB twice, taken on the first request
    from the room the experts' prompt pass had (on demand on a 16 GB GPU, a long prompt after any
    request ran out of memory).  Here, once fused, each projection's arrays become its rows of the
    fused ones (a slice on the first axis, a view of the same buffer): one copy, the same values,
    and the fused copy kept, keyed to the slices, as mlx-vlm's check of it asks."""
    if ops is None:
        from mlx_vlm.speculative.ops import linear as ops
    own = ops._decode_quantized_linears_fused
    if getattr(own, '_quanteeze_shared', False):
        return

    def fused(linears, x):
        first = linears[0] if linears else None
        before = getattr(first, '_fused_decode_linears', None)
        out = own(linears, x)
        cached = getattr(first, '_fused_decode_linears', None)
        if out is None or cached is None or cached is before:
            return out
        _, weights, scales, biases, splits = cached
        bounds = [0, *splits, weights.shape[0]]
        views = [(weights[a:b], scales[a:b], biases[a:b]) for a, b in zip(bounds, bounds[1:])]
        mx.eval(*(v for row in views for v in row))
        for linear, (w, s, b) in zip(linears, views):
            linear.weight, linear.scales, linear.biases = w, s, b
        key = tuple((id(linear.weight), id(linear.scales), id(linear.biases)) for linear in linears)
        first._fused_decode_linears = (key, weights, scales, biases, splits)
        SHARED['fused'] += 1
        return out

    fused._quanteeze_shared = True
    ops._decode_quantized_linears_fused = fused


_CONV = r"""
    uint head = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint c0 = head * 128 + lid * 4;
    T out[4];
    for (int i = 0; i < 4; ++i) {
      uint c = c0 + i;
      float acc = 0.0f;
      for (int k = 0; k < TAPS - 1; ++k) {
        acc = acc + static_cast<float>(state[k * CH + c]) * static_cast<float>(w[c * TAPS + k]);
      }
      acc = acc + static_cast<float>(x[c]) * static_cast<float>(w[c * TAPS + TAPS - 1]);
      T conv = static_cast<T>(acc);
      T s = mlx_sigmoid(conv);
      out[i] = conv * s;
    }
    if (head < 2 * HK) {
      T total = T(0);
      for (int i = 0; i < 4; ++i) {
        T sq = out[i] * out[i];
        total = sq + total;
      }
      T sum = static_cast<T>(simd_sum(static_cast<float>(total)));
      T r = static_cast<T>(metal::precise::rsqrt(static_cast<T>(sum + static_cast<T>(1e-6f))));
      for (int i = 0; i < 4; ++i) {
        T n = out[i] * r;
        y[c0 + i] = head < HK ? static_cast<T>(n * static_cast<T>(SCALE)) : n;
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        y[c0 + i] = out[i];
      }
    }
"""

_GATED_NORM = r"""
    uint head = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint c0 = head * 128 + lid * 4;
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
      float v = static_cast<float>(x[c0 + i]);
      acc = acc + v * v;
    }
    acc = simd_sum(acc);
    float inv = metal::precise::rsqrt(acc / 128.0f + EPS);
    for (int i = 0; i < 4; ++i) {
      T normed = static_cast<T>(w[lid * 4 + i]) * static_cast<T>(static_cast<float>(x[c0 + i]) * inv);
      float gate = mlx_sigmoid(static_cast<float>(z[c0 + i]));
      y[c0 + i] = static_cast<T>(static_cast<float>(normed) * gate);
    }
"""


_CONV_CUDA = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    unsigned int head = blockIdx.x;
    unsigned int t = threadIdx.x;
    T out[8];
    T acc = static_cast<T>(0.0f);
    for (int j = 0; j < 8; ++j) {
      out[j] = static_cast<T>(0.0f);
      if (t < 16) {
        unsigned int c = head * 128 + t * 8 + j;
/*STATE*/
        float a = 0.0f;
        for (int k = 0; k < TAPS - 1; ++k) {
          a = __fmaf_rn(static_cast<float>(state[k * CH + c]), static_cast<float>(w[c * TAPS + k]), a);
        }
        a = __fmaf_rn(static_cast<float>(x[c]), static_cast<float>(w[c * TAPS + TAPS - 1]), a);
        T conv = static_cast<T>(a);
        out[j] = Multiply{}(conv, Sigmoid{}(conv));
        acc = Add{}(acc, QzMul{}(out[j], out[j]));
      }
    }
    if (head < 2 * HK) {
      T total = cg::reduce(warp, acc, QzSum{});
      T r = static_cast<T>(Rsqrt{}(Add{}(total, static_cast<T>(1e-6f))));
      if (t < 16) {
        for (int j = 0; j < 8; ++j) {
          T n = Multiply{}(out[j], r);
          y[head * 128 + t * 8 + j] = head < HK ? Multiply{}(n, static_cast<T>(SCALE)) : n;
        }
      }
    } else if (t < 16) {
      for (int j = 0; j < 8; ++j) {
        y[head * 128 + t * 8 + j] = out[j];
      }
    }
"""

_GATED_NORM_CUDA = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    unsigned int head = blockIdx.x;
    unsigned int t = threadIdx.x;
    const T* xr = x + head * 128;
    float n = 0.0f;
    T xv[8];
    for (int j = 0; j < 8; ++j) {
      int c = t * 8 + j;
      xv[j] = c < 128 ? xr[c] : static_cast<T>(0.0f);
      float v = static_cast<float>(xv[j]);
      n = __fmaf_rn(v, v, n);
    }
    __shared__ float sums[2];
    float r = cg::reduce(warp, n, cg::plus<float>());
    if (warp.thread_rank() == 0) {
      sums[warp.meta_group_rank()] = r;
    }
    block.sync();
    r = warp.thread_rank() < warp.meta_group_size() ? sums[warp.thread_rank()] : 0.0f;
    r = cg::reduce(warp, r, cg::plus<float>());
    float inv = rsqrtf(__fadd_rn(__fdiv_rn(r, 128.0f), EPS));
    for (int j = 0; j < 8; ++j) {
      int c = t * 8 + j;
      if (c < 128) {
        T normed = Multiply{}(w[c], static_cast<T>(__fmul_rn(static_cast<float>(xv[j]), inv)));
        float gate = Sigmoid{}(static_cast<float>(z[head * 128 + c]));
        y[head * 128 + c] = static_cast<T>(__fmul_rn(static_cast<float>(normed), gate));
      }
    }
"""


_DELTA_CUDA = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int gw = blockIdx.x * (blockDim.x / 32) + warp.meta_group_rank();
    if (gw >= HV * DV) {
      return;
    }
    const int h = gw / DV;
    const int row = gw % DV;
    const int hk = h / (HV / HK);
    const int lane = warp.thread_rank();
    const float* s = state + static_cast<size_t>(gw) * DK;
    float* so = new_state + static_cast<size_t>(gw) * DK;
/*GATES*/
    const float vr = static_cast<float>(v[h * DV + row]);
    float s1[4], kf[4], qf[4];
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
      int c = lane * 4 + j;
      kf[j] = static_cast<float>(k[hk * DK + c]);
      qf[j] = static_cast<float>(q[hk * DK + c]);
      s1[j] = __fmul_rn(s[c], gd);
      acc = __fadd_rn(acc, __fmul_rn(s1[j], kf[j]));
    }
    const float kv = cg::reduce(warp, acc, QzSum{});
    const float delta = __fmul_rn(__fsub_rn(vr, kv), bt);
    float acc2 = 0.0f;
    for (int j = 0; j < 4; ++j) {
      int c = lane * 4 + j;
      float s2 = __fmaf_rn(kf[j], delta, s1[j]);
      so[c] = s2;
      acc2 = __fadd_rn(acc2, __fmul_rn(s2, qf[j]));
    }
    const float yv = cg::reduce(warp, acc2, QzSum{});
    if (lane == 0) {
      y[h * DV + row] = static_cast<T>(yv);
    }
"""


_GATES = r"""
    const float gd = g[h];
    const float bt = static_cast<float>(beta[h]);
"""
_GATES_COMPUTED = r"""
    const TS sp = LogAddExp{}(Add{}(static_cast<TS>(a[h]), static_cast<TS>(dtb[h])), static_cast<TS>(0.0f));
    const float gd = Exp{}(Multiply{}(Negative{}(Exp{}(static_cast<float>(alog[h]))), static_cast<float>(sp)));
    const float bt = static_cast<float>(Sigmoid{}(b[h]));
"""

_STATE = r"""
        for (int k = 0; k + 2 < TAPS; ++k) {
          state_out[k * CH + c] = state[(k + 1) * CH + c];
        }
        state_out[(TAPS - 2) * CH + c] = x[c];
"""


def _replaced(source, pairs):
    """`source` with each `(old, new, count)` replaced, verifying `old` appears exactly `count` times."""
    for old, new, count in pairs:
        if source.count(old) != count:
            raise ValueError(f'linear_attn: {old!r} met {source.count(old)} times, not {count}')
        source = source.replace(old, new)
    return source


# Multi-request batch decode on CUDA (decode_batch: inputs [B, 1, C], each request with its own conv window and state):
# _CONV_CUDA and _DELTA_CUDA arithmetic with per-request offsets indexed by blockIdx.y / bi.
_CONV_BATCH_CUDA = _replaced(_CONV_CUDA.replace('/*STATE*/', '/*STATE_BATCH*/'), [
    ('    unsigned int t = threadIdx.x;\n',
     '    unsigned int t = threadIdx.x;\n'
     '    const size_t sb = static_cast<size_t>(blockIdx.y) * (TAPS - 1) * CH;\n'
     '    const size_t xb = static_cast<size_t>(blockIdx.y) * CH;\n', 1),
    ('state[k * CH + c]', 'state[sb + k * CH + c]', 1),
    ('x[c]', 'x[xb + c]', 1),
    ('y[head * 128', 'y[xb + head * 128', 2),
])
_STATE_BATCH = _replaced(_STATE, [
    ('state_out[k * CH + c] = state[(k + 1) * CH + c]', 'state_out[sb + k * CH + c] = state[sb + (k + 1) * CH + c]', 1),
    ('state_out[(TAPS - 2) * CH + c] = x[c]', 'state_out[sb + (TAPS - 2) * CH + c] = x[xb + c]', 1),
])
_DELTA_BATCH_CUDA = _replaced(_DELTA_CUDA, [
    ('if (gw >= HV * DV)', 'if (gw >= BATCH * HV * DV)', 1),
    ('    const int h = gw / DV;\n',
     '    const int hf = gw / DV;\n    const int h = hf % HV;\n    const int bi = hf / HV;\n', 1),
    ('k[hk * DK + c]', 'k[(bi * HK + hk) * DK + c]', 1),
    ('q[hk * DK + c]', 'q[(bi * HK + hk) * DK + c]', 1),
    ('v[h * DV + row]', 'v[hf * DV + row]', 1),
    ('y[h * DV + row]', 'y[hf * DV + row]', 1),
])
_GATES_BATCH = _replaced(_GATES, [('g[h]', 'g[hf]', 1), ('beta[h]', 'beta[hf]', 1)])
_GATES_BATCH_COMPUTED = _replaced(_GATES_COMPUTED, [('(a[h])', '(a[hf])', 1), ('(b[h])', '(b[hf])', 1)])

_CONV_ROWS_CUDA = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    unsigned int head = blockIdx.x;
    unsigned int tok = blockIdx.y;
    unsigned int t = threadIdx.x;
    T out[8];
    T acc = static_cast<T>(0.0f);
    for (int j = 0; j < 8; ++j) {
      out[j] = static_cast<T>(0.0f);
      if (t < 16) {
        unsigned int c = head * 128 + t * 8 + j;
        float a = 0.0f;
        for (int k = 0; k < TAPS; ++k) {
          const int p = static_cast<int>(tok) + k;
          const T v = p < TAPS - 1 ? state[p * CH + c] : x[(p - (TAPS - 1)) * CH + c];
          a = __fmaf_rn(static_cast<float>(v), static_cast<float>(w[c * TAPS + k]), a);
        }
        T conv = static_cast<T>(a);
        out[j] = Multiply{}(conv, Sigmoid{}(conv));
        acc = Add{}(acc, QzMul{}(out[j], out[j]));
      }
    }
    T* yt = y + static_cast<size_t>(tok) * CH;
    if (head < 2 * HK) {
      T total = cg::reduce(warp, acc, QzSum{});
      T r = static_cast<T>(Rsqrt{}(Add{}(total, static_cast<T>(1e-6f))));
      if (t < 16) {
        for (int j = 0; j < 8; ++j) {
          T n = Multiply{}(out[j], r);
          yt[head * 128 + t * 8 + j] = head < HK ? Multiply{}(n, static_cast<T>(SCALE)) : n;
        }
      }
    } else if (t < 16) {
      for (int j = 0; j < 8; ++j) {
        yt[head * 128 + t * 8 + j] = out[j];
      }
    }
"""

_DELTA_ROWS_CUDA = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int gw = blockIdx.x * (blockDim.x / 32) + warp.meta_group_rank();
    if (gw >= HV * DV) {
      return;
    }
    const int h = gw / DV;
    const int row = gw % DV;
    const int hk = h / (HV / HK);
    const int lane = warp.thread_rank();
    const float* s = state + static_cast<size_t>(gw) * DK;
    float sr[4];
    for (int j = 0; j < 4; ++j) {
      sr[j] = s[lane * 4 + j];
    }
    for (int tok = 0; tok < TOKENS; ++tok) {
      const int gi = tok * HV + h;
/*GATES*/
      const float vr = static_cast<float>(v[static_cast<size_t>(gi) * DV + row]);
      const T* kt = k + static_cast<size_t>(tok * HK + hk) * DK;
      const T* qt = q + static_cast<size_t>(tok * HK + hk) * DK;
      float s1[4], kf[4], qf[4];
      float acc = 0.0f;
      for (int j = 0; j < 4; ++j) {
        int c = lane * 4 + j;
        kf[j] = static_cast<float>(kt[c]);
        qf[j] = static_cast<float>(qt[c]);
        s1[j] = __fmul_rn(sr[j], gd);
        acc = __fadd_rn(acc, __fmul_rn(s1[j], kf[j]));
      }
      const float kv = cg::reduce(warp, acc, QzSum{});
      const float delta = __fmul_rn(__fsub_rn(vr, kv), bt);
      float acc2 = 0.0f;
      for (int j = 0; j < 4; ++j) {
        sr[j] = __fmaf_rn(kf[j], delta, s1[j]);
        acc2 = __fadd_rn(acc2, __fmul_rn(sr[j], qf[j]));
      }
/*STATES*/
      const float yv = cg::reduce(warp, acc2, QzSum{});
      if (lane == 0) {
        y[static_cast<size_t>(gi) * DV + row] = static_cast<T>(yv);
      }
    }
    float* so = new_state + static_cast<size_t>(gw) * DK;
    for (int j = 0; j < 4; ++j) {
      so[lane * 4 + j] = sr[j];
    }
"""
_STATES_ROWS = r"""
      if (tok < STEPS) {
        float* st = states + (static_cast<size_t>(tok) * HV * DV + gw) * DK;
        for (int j = 0; j < 4; ++j) {
          st[lane * 4 + j] = sr[j];
        }
      }
"""
_GATES_ROWS = r"""
      const float gd = g[gi];
      const float bt = static_cast<float>(beta[gi]);
"""
_GATES_ROWS_COMPUTED = r"""
      const TS sp = LogAddExp{}(Add{}(static_cast<TS>(a[gi]), static_cast<TS>(dtb[h])), static_cast<TS>(0.0f));
      const float gd = Exp{}(Multiply{}(Negative{}(Exp{}(static_cast<float>(alog[h]))), static_cast<float>(sp)));
      const float bt = static_cast<float>(Sigmoid{}(b[gi]));
"""
_CONV_ROWS = _replaced(_CONV, [
    ('    uint lid = thread_position_in_threadgroup.x;\n',
     '    uint lid = thread_position_in_threadgroup.x;\n    int tok = int(threadgroup_position_in_grid.y);\n', 1),
    ('static_cast<float>(state[k * CH + c])',
     'static_cast<float>(tok + k < TAPS - 1 ? state[(tok + k) * CH + c] : x[(tok + k - (TAPS - 1)) * CH + c])', 1),
    ('static_cast<float>(x[c])', 'static_cast<float>(x[tok * CH + c])', 1),
    ('y[c0 + i]', 'y[tok * CH + c0 + i]', 2),
])


@lru_cache(maxsize=None)
def _conv_rows_kernel(scale):
    name = f'quanteeze_la_conv_rows_{scale!r}'.replace('.', '_').replace('-', 'm')
    if not cuda_kernels.available():
        return mx.fast.metal_kernel(name=name, input_names=['state', 'x', 'w'], output_names=['y'], source=_CONV_ROWS,
                                    header=metal_kernels.SIGMOID + f'constant constexpr float SCALE = {float(scale)!r}f;\n')
    return cuda_kernels.kernel(name, ['state', 'x', 'w'], ['y'], _CONV_ROWS_CUDA,
                               f'constexpr float SCALE = {float(scale)!r}f;\n')


@lru_cache(maxsize=None)
def _delta_rows_kernel(gating, states):
    ins = ['q', 'k', 'v'] + (['a', 'b', 'alog', 'dtb'] if gating else ['g', 'beta']) + ['state']
    source = (_DELTA_ROWS_CUDA.replace('/*GATES*/', _GATES_ROWS_COMPUTED if gating else _GATES_ROWS)
              .replace('/*STATES*/', _STATES_ROWS if states else ''))
    return cuda_kernels.kernel(f'quanteeze_la_delta_rows{"_gating" if gating else ""}{"_states" if states else ""}',
                               ins, ['y', 'new_state'] + (['states'] if states else []), source)


@lru_cache(maxsize=None)
def _conv_batch_kernel(scale, sliding=False):
    name = f'quanteeze_la_conv_batch{"_sliding" if sliding else ""}_{scale!r}'.replace('.', '_').replace('-', 'm')
    return cuda_kernels.kernel(name, ['state', 'x', 'w'], ['y'] + (['state_out'] if sliding else []),
                               _CONV_BATCH_CUDA.replace('/*STATE_BATCH*/', _STATE_BATCH if sliding else ''),
                               f'constexpr float SCALE = {float(scale)!r}f;\n')


@lru_cache(maxsize=None)
def _delta_batch_kernel(gating=False):
    if gating:
        return cuda_kernels.kernel('quanteeze_la_delta_batch_gating',
                                   ['q', 'k', 'v', 'a', 'b', 'alog', 'dtb', 'state'], ['y', 'new_state'],
                                   _DELTA_BATCH_CUDA.replace('/*GATES*/', _GATES_BATCH_COMPUTED))
    return cuda_kernels.kernel('quanteeze_la_delta_batch', ['q', 'k', 'v', 'g', 'beta', 'state'], ['y', 'new_state'],
                               _DELTA_BATCH_CUDA.replace('/*GATES*/', _GATES_BATCH))


@lru_cache(maxsize=None)
def _delta_kernel(gating=False):
    if gating:
        return cuda_kernels.kernel('quanteeze_la_delta_gating', ['q', 'k', 'v', 'a', 'b', 'alog', 'dtb', 'state'],
                                   ['y', 'new_state'], _DELTA_CUDA.replace('/*GATES*/', _GATES_COMPUTED))
    return cuda_kernels.kernel('quanteeze_la_delta', ['q', 'k', 'v', 'g', 'beta', 'state'], ['y', 'new_state'],
                               _DELTA_CUDA.replace('/*GATES*/', _GATES))


@lru_cache(maxsize=None)
def _conv_kernel(scale, sliding=False):
    name = f'quanteeze_la_conv{"_sliding" if sliding else ""}_{scale!r}'.replace('.', '_').replace('-', 'm')
    if cuda_kernels.available():
        return cuda_kernels.kernel(name, ['state', 'x', 'w'], ['y'] + (['state_out'] if sliding else []),
                                   _CONV_CUDA.replace('/*STATE*/', _STATE if sliding else ''),
                                   f'constexpr float SCALE = {float(scale)!r}f;\n')
    source = _CONV + (
        '    for (int i = 0; i < 4; ++i) {\n      uint c = c0 + i;\n' + _STATE + '    }\n' if sliding else ''
    )
    return mx.fast.metal_kernel(name=name, input_names=['state', 'x', 'w'],
                                output_names=['y'] + (['state_out'] if sliding else []), source=source,
                                header=metal_kernels.SIGMOID + f'constant constexpr float SCALE = {float(scale)!r}f;\n')


@lru_cache(maxsize=None)
def _gated_norm_kernel(eps):
    name = f'quanteeze_la_gated_norm_{eps!r}'.replace('.', '_').replace('-', 'm')
    if cuda_kernels.available():
        return cuda_kernels.kernel(name, ['x', 'z', 'w'], ['y'], _GATED_NORM_CUDA,
                                   f'constexpr float EPS = {float(eps)!r}f;\n')
    return mx.fast.metal_kernel(name=name, input_names=['x', 'z', 'w'], output_names=['y'], source=_GATED_NORM,
                                header=metal_kernels.SIGMOID + f'constant constexpr float EPS = {float(eps)!r}f;\n')


FEW = 64                        # tokens up to which a pass's gated delta rule goes token by token on CUDA (few())


def few():
    """On MLX's CUDA backend, mlx-vlm's gated delta rule for a pass of 2 to FEW tokens token by token (its own
    per-token ops, as each decode step computes it) where it took its chunked scan, which pads the tokens to a
    chunk of 64 and solves it."""
    if mx.metal.is_available():
        return False
    from mlx_vlm.models.qwen3_5 import gated_delta as gd
    if getattr(gd.gated_delta_chunked, 'quanteeze_few', False):
        return True
    chunked = gd.gated_delta_chunked

    def gated_delta_chunked(q, k, v, g, beta, state, mask=None, C=64):
        if DELTA[0] and q.shape[1] <= FEW:
            STATS['few'] += 1
            return gd.gated_delta_ops(q, k, v, g, beta, state, mask)
        return chunked(q, k, v, g, beta, state, mask, C)
    gated_delta_chunked.quanteeze_few = True
    gd.gated_delta_chunked = gated_delta_chunked
    return True


def applies(layer, inputs, mask, cache):
    """Whether this step is one the kernels compute: one token of one request with its conv state
    cached, heads of 128, a conv without bias, and a sigmoid-gated RMSNorm."""
    norm = getattr(layer, 'norm', None)
    return (inputs.shape[:2] == (1, 1) and mask is None and cache is not None and cache[0] is not None
            and cache[0].shape[0] == 1 and inputs.dtype in (mx.bfloat16, mx.float16)
            and layer.head_k_dim == 128 and layer.head_v_dim == 128 and 'bias' not in layer.conv1d
            and getattr(norm, 'activation', None) == 'sigmoid' and getattr(norm, 'weight', None) is not None
            and norm.weight.size == 128 and not layer.training)


def conv(layer, state, x, sliding=False):
    """silu(conv1d([state, x])) of one token, its q and k heads L2-normalized and q scaled: [1, 1, C];
    sliding, also the state slid by the token ([state, x][:, 1:]): (that, the state)."""
    c = x.shape[-1]
    out = _conv_kernel(layer.head_k_dim ** -0.5, sliding)(
        inputs=[state, x, layer.conv1d.weight.reshape(c, -1)],
        template=[('T', x.dtype), ('CH', c), ('HK', layer.num_k_heads), ('TAPS', layer.conv_kernel_size)],
        grid=(c // 128 * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[x.shape] + ([state.shape] if sliding else []), output_dtypes=[x.dtype] * (1 + sliding))
    return tuple(out) if sliding else out[0]


def gated_norm(norm, x, z):
    """Qwen4ExpRMSNormGated(x, z) with a sigmoid gate, in one kernel."""
    tg = 64 if cuda_kernels.available() else 32
    return _gated_norm_kernel(float(norm.eps))(
        inputs=[x, z, norm.weight], template=[('T', x.dtype)], grid=(x.size // 128 * tg, 1, 1),
        threadgroup=(tg, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def deltable(q, k, v, state, layer):
    """Whether delta() computes this step as mlx-vlm's ops do: on CUDA, one token of one request,
    heads of 128, the key heads a divisor of the value heads, a scalar decay a head, a float32 state."""
    return (cuda_kernels.available() and state is not None and state.dtype == mx.float32 and q.dtype == mx.bfloat16
            and q.shape[:2] == (1, 1) and q.shape[-1] == k.shape[-1] == v.shape[-1] == 128
            and v.shape[2] % q.shape[2] == 0 and state.shape == (1, v.shape[2], 128, 128)
            and layer.A_log.shape == (v.shape[2],))


def delta(q, k, v, g, beta, state):
    """_gated_delta_step_ops(q, k, v, g, beta, state) of one token, q and k a head each for
    v.shape[2] // q.shape[2] value heads (mx.repeat's): (y [1, 1, Hv, 128], the state after)."""
    hk, hv = q.shape[2], v.shape[2]
    return tuple(_delta_kernel()(
        inputs=[q, k, v, g, beta, state],
        template=[('T', q.dtype), ('HK', hk), ('HV', hv), ('DK', 128), ('DV', 128)],
        grid=(32 * hv * 128, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(1, 1, hv, 128), state.shape],
        output_dtypes=[q.dtype, mx.float32]))


def _softplus_dtype(a, dt_bias):
    """The dtype a + dt_bias is computed in, where the gating kernel takes it."""
    if a.dtype == dt_bias.dtype and a.dtype in (mx.bfloat16, mx.float16, mx.float32):
        return a.dtype
    return mx.float32 if dt_bias.dtype == mx.float32 and a.dtype in (mx.bfloat16, mx.float16) else None


def delta_gating(q, k, v, a, b, layer, state):
    """delta(q, k, v, *_compute_g_beta(layer.A_log, a, b, layer.dt_bias), state) in one kernel."""
    hk, hv = q.shape[2], v.shape[2]
    return tuple(_delta_kernel(True)(
        inputs=[q, k, v, a, b, layer.A_log, layer.dt_bias, state],
        template=[('T', q.dtype), ('TS', _softplus_dtype(a, layer.dt_bias)), ('HK', hk), ('HV', hv), ('DK', 128),
                  ('DV', 128)],
        grid=(32 * hv * 128, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(1, 1, hv, 128), state.shape],
        output_dtypes=[q.dtype, mx.float32]))


def _qkv(layer, out):
    return [value.reshape(1, 1, heads, width) for value, heads, width in zip(
        mx.split(out, [layer.key_dim, 2 * layer.key_dim], -1),
        [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
        [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim])]


def _conv_original(forward, layer, conv_input):
    import mlx.nn as nn
    q, k, v = _qkv(layer, nn.silu(layer.conv1d(conv_input)))
    q, k = forward._normalize_gated_delta_qk(layer, q, k)
    return q, k, v


IN_PROJ = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a')


def _in_proj(forward, layer, inputs):
    """Run the four gated-delta decode input projections via mlx-vlm's cached joined weights when intact."""
    if not isinstance(layer, dict):
        return forward._linears(tuple(getattr(layer, name) for name in IN_PROJ), inputs)
    kept = layer.__dict__.get('_quanteeze_in_proj')
    if inputs.ndim == 3 and inputs.shape[1] == 1 and kept is not None and inputs.dtype == kept[0]:
        held = kept[1]
        if all(
            layer.get(name) is lin
            and isinstance(lin, dict)
            and lin.get('weight') is w
            and lin.get('scales') is s
            and lin.get('biases') is b
            for name, lin, (w, s, b) in zip(IN_PROJ, held[0], held[1])
        ):
            _, _, weights, scales, biases, split, group_size, bits, mode = kept
            out = mx.quantized_matmul(
                inputs,
                weights,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            return tuple(mx.split(out, split, axis=-1))
        layer.__dict__.pop('_quanteeze_in_proj', None)
    linears = tuple(layer[name] for name in IN_PROJ)
    out = forward._linears(linears, inputs)
    first = linears[0]
    joined = first.get('_fused_decode_linears') if isinstance(first, dict) else None
    if (
        inputs.ndim == 3
        and inputs.shape[1] == 1
        and joined is not None
        and all(isinstance(lin, dict) and lin.get('biases') is not None and 'bias' not in lin for lin in linears)
        and joined[0] == tuple((id(lin['weight']), id(lin['scales']), id(lin['biases'])) for lin in linears)
    ):
        held = (linears, tuple((lin['weight'], lin['scales'], lin['biases']) for lin in linears))
        object.__setattr__(
            layer,
            '_quanteeze_in_proj',
            (inputs.dtype, held, joined[1], joined[2], joined[3], joined[4], first.group_size, first.bits, first.mode),
        )
    return out


def decode(forward, layer, inputs, cache):
    """Qwen3_5BatchInvariantForward._gated_delta for a step applies() takes, through conv(),
    delta() (on CUDA) and gated_norm() where they matched."""
    from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta, gated_delta_update
    mixed_qkv, z, b, a = _in_proj(forward, layer, inputs)
    z = z.reshape(1, 1, -1, layer.head_v_dim)
    state = cache[0]
    width = layer.conv_kernel_size - 1
    key = (tuple(mixed_qkv.shape), str(mixed_qkv.dtype))
    joined = []

    def conv_input():
        if not joined:
            joined.append(mx.concatenate([state, mixed_qkv], axis=1))
        return joined[0]
    if (((cuda_kernels.FUSE[0] and cuda_kernels.available()) or (SLIDE[0] and mx.metal.is_available()))
            and cache.lengths is None and not getattr(cache, 'history_capacity', 0)):
        def mine():
            y, slid = conv(layer, state, mixed_qkv, sliding=True)
            return (*_qkv(layer, y), slid)
        q, k, v, slid = _served('slide', key, mine, lambda: (*_conv_original(forward, layer, conv_input()),
                                                             mx.contiguous(conv_input()[:, -width:])))
        cache[0] = slid
    else:
        cache.update_window(0, conv_input(), width, lengths=cache.lengths)
        q, k, v = _served('conv', key, lambda: tuple(_qkv(layer, conv(layer, state, mixed_qkv))),
                          lambda: _conv_original(forward, layer, conv_input()))

    def update(initial, steps):
        def theirs():
            return gated_delta_update(q, k, v, a, b, layer.A_log, layer.dt_bias, mask=None, use_kernel=True,
                                      state=initial if initial is None or initial.shape[0] == q.shape[0] else None,
                                      state_steps=steps)
        if not DELTA[0] or steps is not None or not deltable(q, k, v, initial, layer):
            return theirs()
        if cuda_kernels.FUSE[0] and _softplus_dtype(a, layer.dt_bias) is not None and a.size == b.size == v.shape[2]:
            return _served('gating', key, lambda: delta_gating(q, k, v, a, b, layer, initial), theirs)
        return _served('delta', key, lambda: delta(q, k, v, *_compute_g_beta(layer.A_log, a, b, layer.dt_bias),
                                                   initial), theirs)
    output, _ = cache.update_recurrent(1, 1, update)
    if hasattr(cache, 'advance'):
        cache.advance(1)
        forward._helpers()._qwen3_5_advance_lengths_info(cache, 1)
    output = _served('norm', key, lambda: gated_norm(layer.norm, output, z), lambda: layer.norm(output, z))
    return forward._linear(layer.out_proj, output.reshape(1, 1, -1))


def applies_rows(layer, inputs, mask, cache):
    """Whether decode_rows() computes this step: 2 to ROWS_MOST tokens of one request (an MTP verify pass)."""
    norm = getattr(layer, 'norm', None)
    return (ON[0] and (cuda_kernels.available() or mx.metal.is_available()) and inputs.ndim == 3 and inputs.shape[0] == 1
            and 1 < inputs.shape[1] <= ROWS_MOST and mask is None and cache is not None and cache[0] is not None
            and cache[0].shape[0] == 1 and inputs.dtype == mx.bfloat16
            and layer.head_k_dim == 128 and layer.head_v_dim == 128 and 'bias' not in layer.conv1d
            and getattr(norm, 'activation', None) == 'sigmoid' and getattr(norm, 'weight', None) is not None
            and norm.weight.size == 128 and not layer.training)


def conv_rows(layer, state, x):
    """conv() of each of a pass's tokens x [1, T, C]: [1, T, C]."""
    c, t = x.shape[-1], x.shape[1]
    return _conv_rows_kernel(layer.head_k_dim ** -0.5)(
        inputs=[state, x, layer.conv1d.weight.reshape(c, -1)],
        template=[('T', x.dtype), ('CH', c), ('HK', layer.num_k_heads), ('TAPS', layer.conv_kernel_size)],
        grid=(c // 128 * 32, t, 1), threadgroup=(32, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def deltable_rows(q, k, v, state, layer):
    """deltable() for a pass of 2 to ROWS_MOST tokens of one request."""
    t = q.shape[1]
    return (cuda_kernels.available() and state is not None and state.dtype == mx.float32 and q.dtype == mx.bfloat16
            and q.shape[0] == 1 and 1 < t <= ROWS_MOST and k.shape[:2] == v.shape[:2] == (1, t)
            and q.shape[-1] == k.shape[-1] == v.shape[-1] == 128 and v.shape[2] % q.shape[2] == 0
            and state.shape == (1, v.shape[2], 128, 128) and layer.A_log.shape == (v.shape[2],))


def delta_rows(q, k, v, gates, state, steps=None, layer=None):
    """The delta rule's steps of a pass's tokens in one kernel."""
    t, hk, hv = q.shape[1], q.shape[2], v.shape[2]
    gating = layer is not None
    keep = bool(steps)
    ins = [q, k, v] + (list(gates) + [layer.A_log, layer.dt_bias] if gating else list(gates)) + [state]
    template = [('T', q.dtype), ('HK', hk), ('HV', hv), ('DK', 128), ('DV', 128), ('TOKENS', t)]
    if gating:
        template.insert(1, ('TS', _softplus_dtype(gates[0], layer.dt_bias)))
    if keep:
        template.append(('STEPS', int(steps)))
    out = _delta_rows_kernel(gating, keep)(
        inputs=ins, template=template, grid=(32 * hv * 128, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1, t, hv, 128), state.shape] + ([(1, int(steps), hv, 128, 128)] if keep else []),
        output_dtypes=[q.dtype, mx.float32] + ([mx.float32] if keep else []))
    return tuple(out)


def _qkv_rows(layer, out, t):
    return [value.reshape(1, t, heads, width) for value, heads, width in zip(
        mx.split(out, [layer.key_dim, 2 * layer.key_dim], -1),
        [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
        [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim])]


def _delta_tokens(q, k, v, a, b, layer, state, steps=None):
    """mlx-vlm's delta rule token by token, preserving states after the first `steps`."""
    from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update
    ys, kept = [], []
    for i in range(q.shape[1]):
        one = slice(i, i + 1)
        y, state = gated_delta_update(q[:, one], k[:, one], v[:, one], a[:, one], b[:, one], layer.A_log,
                                      layer.dt_bias, state=state)
        ys.append(y)
        if steps and i < steps:
            kept.append(state)
    return (mx.concatenate(ys, axis=1), state) + ((mx.stack(kept, axis=1),) if steps else ())


def _conv_tokens(forward, layer, conv_input, t):
    """_conv_original of each token on its own window, joined: q, k, v [1, T, ...]."""
    taps = layer.conv_kernel_size
    each = [_conv_original(forward, layer, conv_input[:, i:i + taps]) for i in range(t)]
    return tuple(mx.concatenate([e[j] for e in each], axis=1) for j in range(3))


def decode_rows(forward, layer, inputs, cache):
    """Qwen4ExpBatchInvariantForward._gated_delta for a verify pass of 2 to ROWS_MOST tokens."""
    from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta, gated_delta_update
    t = inputs.shape[1]
    mixed_qkv, z, b, a = forward._linears(
        (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a), inputs)
    z = z.reshape(1, t, -1, layer.head_v_dim)
    state = cache[0]
    key = (tuple(mixed_qkv.shape), str(mixed_qkv.dtype))
    conv_input = mx.concatenate([state, mixed_qkv], axis=1)
    cache.update_window(0, conv_input, layer.conv_kernel_size - 1, lengths=cache.lengths)
    q, k, v = _served('conv_rows', key, lambda: tuple(_qkv_rows(layer, conv_rows(layer, state, mixed_qkv), t)),
                      lambda: _conv_tokens(forward, layer, conv_input, t))

    def update(initial, steps):
        if not DELTA[0] or not deltable_rows(q, k, v, initial, layer):
            return gated_delta_update(q, k, v, a, b, layer.A_log, layer.dt_bias, mask=None, use_kernel=True,
                                      state=initial if initial is None or initial.shape[0] == q.shape[0] else None,
                                      state_steps=steps)

        def theirs():
            return _delta_tokens(q, k, v, a, b, layer, initial, steps)
        if cuda_kernels.FUSE[0] and _softplus_dtype(a, layer.dt_bias) is not None and a.shape == b.shape == v.shape[:3]:
            return _served('gating_rows', key + (steps,), lambda: delta_rows(q, k, v, (a, b), initial, steps, layer),
                           theirs)
        return _served('delta_rows', key + (steps,),
                       lambda: delta_rows(q, k, v, _compute_g_beta(layer.A_log, a, b, layer.dt_bias), initial, steps),
                       theirs)
    output, _ = cache.update_recurrent(1, t, update)
    if hasattr(cache, 'advance'):
        cache.advance(t)
        forward._helpers()._qwen3_5_advance_lengths_info(cache, t)
    output = _served('norm_rows', key, lambda: gated_norm(layer.norm, output, z), lambda: layer.norm(output, z))
    return forward._linear(layer.out_proj, output.reshape(1, t, -1))


def applies_batch(layer, inputs, mask, cache):
    """Whether decode_batch() computes this step: on CUDA, a decode of 2 to ROWS_MOST requests at once (inputs
    [B, 1, C]), each with its conv window cached, heads of 128, a conv without bias, and a sigmoid-gated RMSNorm."""
    norm = getattr(layer, 'norm', None)
    n = inputs.shape[0] if inputs.ndim == 3 else 0
    return (ON[0] and cuda_kernels.available() and inputs.ndim == 3 and inputs.shape[1] == 1 and 1 < n <= ROWS_MOST
            and mask is None and cache is not None and cache[0] is not None and cache[0].shape[0] == n
            and inputs.dtype == mx.bfloat16
            and layer.head_k_dim == 128 and layer.head_v_dim == 128 and 'bias' not in layer.conv1d
            and getattr(norm, 'activation', None) == 'sigmoid' and getattr(norm, 'weight', None) is not None
            and norm.weight.size == 128 and not layer.training)


def conv_batch(layer, state, x, sliding=False):
    """conv() of each request's token x [B, 1, C] on its own window state [B, TAPS - 1, C]: [B, 1, C]; sliding,
    also each window slid by its token: (that, the windows)."""
    c, n = x.shape[-1], x.shape[0]
    out = _conv_batch_kernel(layer.head_k_dim ** -0.5, sliding)(
        inputs=[state, x, layer.conv1d.weight.reshape(c, -1)],
        template=[('T', x.dtype), ('CH', c), ('HK', layer.num_k_heads), ('TAPS', layer.conv_kernel_size)],
        grid=(c // 128 * 32, n, 1), threadgroup=(32, 1, 1),
        output_shapes=[x.shape] + ([state.shape] if sliding else []), output_dtypes=[x.dtype] * (1 + sliding))
    return tuple(out) if sliding else out[0]


def deltable_batch(q, k, v, state, layer):
    """deltable() for a decode of 2 to ROWS_MOST requests at once, a state each."""
    n = q.shape[0]
    return (cuda_kernels.available() and state is not None and state.dtype == mx.float32 and q.dtype == mx.bfloat16
            and 1 < n <= ROWS_MOST and q.shape[1] == 1 and k.shape[:2] == v.shape[:2] == (n, 1)
            and q.shape[-1] == k.shape[-1] == v.shape[-1] == 128 and v.shape[2] % q.shape[2] == 0
            and state.shape == (n, v.shape[2], 128, 128) and layer.A_log.shape == (v.shape[2],))


def delta_batch(q, k, v, gates, state, layer=None):
    """delta() (or delta_gating(), given `layer`: gates then a and b) of each request's token, q and k [B, 1, Hk,
    128], v [B, 1, Hv, 128], on its own state [B, Hv, 128, 128]: (y [B, 1, Hv, 128], the states after)."""
    n, hk, hv = q.shape[0], q.shape[2], v.shape[2]
    gating = layer is not None
    ins = [q, k, v] + (list(gates) + [layer.A_log, layer.dt_bias] if gating else list(gates)) + [state]
    template = [('T', q.dtype), ('HK', hk), ('HV', hv), ('DK', 128), ('DV', 128), ('BATCH', n)]
    if gating:
        template.insert(1, ('TS', _softplus_dtype(gates[0], layer.dt_bias)))
    return tuple(_delta_batch_kernel(gating)(
        inputs=ins, template=template, grid=(32 * n * hv * 128, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n, 1, hv, 128), state.shape], output_dtypes=[q.dtype, mx.float32]))


def _qkv_batch(layer, out, n):
    return [value.reshape(n, 1, heads, width) for value, heads, width in zip(
        mx.split(out, [layer.key_dim, 2 * layer.key_dim], -1),
        [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
        [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim])]


def _conv_original_batch(forward, layer, conv_input, n):
    import mlx.nn as nn
    q, k, v = _qkv_batch(layer, nn.silu(layer.conv1d(conv_input)), n)
    q, k = forward._normalize_gated_delta_qk(layer, q, k)
    return q, k, v


def decode_batch(forward, layer, inputs, cache):
    """Qwen4ExpBatchInvariantForward._gated_delta for a step applies_batch() takes (a decode of several requests at
    once), as decode() computes one request's, each request's rows at its offset in the same kernels: conv_batch(),
    delta_batch(), gated_norm() over every request's heads.  Each part is checked first against mlx-vlm's ops for
    the batch and serves its shape only where they are the same."""
    from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta, gated_delta_update
    n = inputs.shape[0]
    mixed_qkv, z, b, a = forward._linears(
        (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a), inputs)
    z = z.reshape(n, 1, -1, layer.head_v_dim)
    state = cache[0]
    width = layer.conv_kernel_size - 1
    key = (tuple(mixed_qkv.shape), str(mixed_qkv.dtype))
    joined = []

    def conv_input():
        if not joined:
            joined.append(mx.concatenate([state, mixed_qkv], axis=1))
        return joined[0]
    if cuda_kernels.FUSE[0] and cache.lengths is None and not getattr(cache, 'history_capacity', 0):
        def mine():
            y, slid = conv_batch(layer, state, mixed_qkv, sliding=True)
            return (*_qkv_batch(layer, y, n), slid)
        q, k, v, slid = _served('slide_batch', key, mine,
                                lambda: (*_conv_original_batch(forward, layer, conv_input(), n),
                                         mx.contiguous(conv_input()[:, -width:])))
        cache[0] = slid
    else:
        cache.update_window(0, conv_input(), width, lengths=cache.lengths)
        q, k, v = _served('conv_batch', key, lambda: tuple(_qkv_batch(layer, conv_batch(layer, state, mixed_qkv), n)),
                          lambda: _conv_original_batch(forward, layer, conv_input(), n))

    def update(initial, steps):
        def theirs():
            return gated_delta_update(q, k, v, a, b, layer.A_log, layer.dt_bias, mask=None, use_kernel=True,
                                      state=initial if initial is None or initial.shape[0] == q.shape[0] else None,
                                      state_steps=steps)
        if not DELTA[0] or steps is not None or not deltable_batch(q, k, v, initial, layer):
            return theirs()
        if cuda_kernels.FUSE[0] and _softplus_dtype(a, layer.dt_bias) is not None and a.shape == b.shape == v.shape[:3]:
            return _served('gating_batch', key, lambda: delta_batch(q, k, v, (a, b), initial, layer), theirs)
        return _served('delta_batch', key,
                       lambda: delta_batch(q, k, v, _compute_g_beta(layer.A_log, a, b, layer.dt_bias), initial), theirs)
    output, _ = cache.update_recurrent(1, 1, update)
    if hasattr(cache, 'advance'):
        cache.advance(1)
        forward._helpers()._qwen3_5_advance_lengths_info(cache, 1)
    output = _served('norm_batch', key, lambda: gated_norm(layer.norm, output, z), lambda: layer.norm(output, z))
    return forward._linear(layer.out_proj, output.reshape(n, 1, -1))


def _served(part, key, mine, theirs):
    """mine() where this part's shape matched theirs() bit for bit the first CHECKS times it was
    met, else theirs()."""
    return kernel_checks.served(sys.modules[__name__], (part,) + key, mine, theirs,
                                'linear attention kernels', stat=part)


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
