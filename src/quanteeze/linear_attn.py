# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Flash-Next's linear attention (its gated delta layers) in a decode token with fewer kernels on the
GPU's chain, each bit for bit what it replaces.

With quanteeze.cuda_kernels.FUSE (QUANTEEZE_CUDA_FUSE) the conv's kernel also writes the cache's
window slid by the token (update_window's slice of the concatenation, which then is not made: three
copies a layer) where the cache keeps no lengths or history, and the delta rule's kernel computes the
step's decay and beta itself (mlx-vlm's compiled _compute_g_beta: exp(-exp(A_log) * softplus(a +
dt_bias)) and sigmoid(b), MLX's functors in its order; two kernels a layer).  None is taken on trust:
the first CHECKS times a shape is met, each kernel's output is compared bit for bit with mlx-vlm's ops,
and it serves the shape only if they are the same.  Only a decode step of one request
(QUANTEEZE_LINEAR_ATTN_KERNELS=off: mlx-vlm's).
"""
import sys
from functools import lru_cache

import mlx.core as mx

from . import cuda_kernels

CHECKS = 3
ON = [True]                     # QUANTEEZE_LINEAR_ATTN_KERNELS
DELTA = [True]                  # QUANTEEZE_CUDA_DELTA: the delta rule's step on CUDA
STATS = dict(conv=0, slide=0, norm=0, delta=0, gating=0, checked=0, mismatched=0, few=0)
CHOICE = {}                     # (part, shape, dtype): whether it matched the original
CHECKED = {}
SAID = []

_SIGMOID = """
template <typename X>
inline X mlx_sigmoid(X x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

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
    return mx.fast.metal_kernel(name=name, input_names=['state', 'x', 'w'], output_names=['y'], source=_CONV,
                                header=_SIGMOID + f'constant constexpr float SCALE = {float(scale)!r}f;\n')


@lru_cache(maxsize=None)
def _gated_norm_kernel(eps):
    name = f'quanteeze_la_gated_norm_{eps!r}'.replace('.', '_').replace('-', 'm')
    if cuda_kernels.available():
        return cuda_kernels.kernel(name, ['x', 'z', 'w'], ['y'], _GATED_NORM_CUDA,
                                   f'constexpr float EPS = {float(eps)!r}f;\n')
    return mx.fast.metal_kernel(name=name, input_names=['x', 'z', 'w'], output_names=['y'], source=_GATED_NORM,
                                header=_SIGMOID + f'constant constexpr float EPS = {float(eps)!r}f;\n')


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


def decode(forward, layer, inputs, cache):
    """Qwen3_5BatchInvariantForward._gated_delta for a step applies() takes, through conv(),
    delta() (on CUDA) and gated_norm() where they matched."""
    from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta, gated_delta_update
    mixed_qkv, z, b, a = forward._linears(
        (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a), inputs)
    z = z.reshape(1, 1, -1, layer.head_v_dim)
    state = cache[0]
    width = layer.conv_kernel_size - 1
    key = (tuple(mixed_qkv.shape), str(mixed_qkv.dtype))
    joined = []

    def conv_input():
        if not joined:
            joined.append(mx.concatenate([state, mixed_qkv], axis=1))
        return joined[0]
    if (cuda_kernels.FUSE[0] and cuda_kernels.available() and cache.lengths is None
            and not getattr(cache, 'history_capacity', 0)):
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


def _served(part, key, mine, theirs):
    """mine() where this part's shape matched theirs() bit for bit the first CHECKS times it was
    met, else theirs()."""
    key = (part,) + key
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS[part] += 1
            return mine()
        return theirs()
    want = theirs()
    STATS['checked'] += 1
    try:
        got = mine()
        pairs = zip(got, want) if isinstance(want, tuple) else [(got, want)]
        same = all(mx.array_equal(a, b).item() for a, b in pairs)
    except Exception as error:
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: linear attention kernels unavailable ({type(error).__name__}: {error})',
                  file=sys.stderr, flush=True)
        same = False
    STATS['mismatched'] += not same
    CHOICE[key], CHECKED[key] = bool(same), (seen + 1 if same else CHECKS)
    return want


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
