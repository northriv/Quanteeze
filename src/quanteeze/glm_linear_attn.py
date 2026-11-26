# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""GLM-5.3's linear attention (its KDA layers: a short conv, RMS-normed q and k, the gated delta rule
with a gate per channel) in a decode token with fewer kernels, bit for bit what mlx-vlm computes.

A layer's decode step in mlx-vlm (Glm5NextLinearAttention) is 19 kernels: five quantized projections
and fourteen others.  Three replace those fourteen:

- conv(): the window and the new input through the conv of 4 taps, silu, q and k RMS-normed and
  scaled, and the next window, where mlx-vlm runs Concatenate, Convolution, silu, the window's
  Contiguous, two RMSNorms and two Multiplies.  A threadgroup of 32 a head of 128 channels, four a
  thread: each channel's taps summed in float32 and rounded (MLX's conv), silu on bfloat16 as MLX's
  compiled silu computes it (its sigmoid as its unary op writes it, then the product), and for q and
  k mx.fast.rms_norm's arithmetic (float32 squares, simd_sum, precise rsqrt of the mean plus
  1e-6 / 128) and the scale (1/128 for q, 128^-1/2 for k) multiplied on bfloat16.
- delta(): mlx-vlm's gated delta kernel (vectorized gate) with its inputs computed in it: the gate
  as the compiled compute_g_safe computes it (exp(lower_bound * sigmoid(exp(A_log) * (a + dt_bias)))
  in float32) and beta as MLX's Sigmoid on bfloat16, where mlx-vlm runs the compiled g, Sigmoid and
  the kernel.
- gated_norm(): o_norm (mx.fast.rms_norm with its weight, eps 1e-5) times sigmoid of the gate, both
  on bfloat16, where mlx-vlm runs RMSNorm, Sigmoid and Multiply.

A layer of GLM-5.3's size (4-bit projections, random weights), 36 decode steps: the output, the
window and the state the same bits.  34 layers a step on the host: 3.22 to 2.46 ms.

Only a decode step of one request (one token, no mask, no lengths or padding, no speculative history:
those are mlx-vlm's), heads of 128.  None is taken on trust: the first CHECKS times a shape is met, the
layer runs mlx-vlm's way and this one, the output, window and state are compared bit for bit, and the
shape is served only if they are the same.  QUANTEEZE_LINEAR_ATTN_KERNELS=off keeps mlx-vlm's (as it
does Flash-Next's, quanteeze.linear_attn).
"""
import sys
from functools import lru_cache

import mlx.core as mx
from mlx_vlm.models.linear import linear

from . import kernel_checks, linear_attn, metal_kernels, per_thread

CHECKS = 3
STATS = dict(served=0, checked=0, mismatched=0)
CHOICE, CHECKED = kernel_checks.tables('glm_linear_attn')  # shape: whether it matched mlx-vlm's
SAID = []

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
      for (int k = 0; k < TAPS - 2; ++k) {
        window[k * CH + c] = state[(k + 1) * CH + c];
      }
      window[(TAPS - 2) * CH + c] = x[c];
    }
    if (head < 2 * H) {
      float acc = 0.0f;
      for (int i = 0; i < 4; ++i) {
        float v = static_cast<float>(out[i]);
        acc = acc + v * v;
      }
      acc = simd_sum(acc);
      float inv = metal::precise::rsqrt(acc / 128.0f + EPS);
      T scale = head < H ? static_cast<T>(SCALE_Q) : static_cast<T>(SCALE_K);
      for (int i = 0; i < 4; ++i) {
        y[c0 + i] = static_cast<T>(static_cast<float>(out[i]) * inv) * scale;
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        y[c0 + i] = out[i];
      }
    }
"""

_DELTA = r"""
    auto h = thread_position_in_grid.z;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    constexpr int n_per_t = Dk / 32;
    auto q_ = qkv + h * Dk;
    auto k_ = qkv + H * Dk + h * Dk;
    auto v_ = qkv + 2 * H * Dk + h * Dv;
    auto i_state = state_in + (h * Dv + dv_idx) * Dk;
    auto o_state = state_out + (h * Dv + dv_idx) * Dk;
    // the gate (compute_g_safe) and beta (MLX's Sigmoid of b), in the dtypes mlx-vlm computes them
    float state[n_per_t];
    float g[n_per_t];
    float ea = metal::precise::exp(A_log[h]);
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
      float z = ea * (static_cast<float>(a[h * Dk + s_idx]) + dt_bias[h * Dk + s_idx]);
      g[i] = metal::precise::exp(LOWER * mlx_sigmoid(z));
    }
    InT beta = mlx_sigmoid(fbg[Dk + h]);
    // mlx-vlm's gated delta step, as its kernel writes it
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] * g[i];
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);
    auto delta = (v_[dv_idx] - kv_mem) * beta;
    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] + k_[s_idx] * delta;
      out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y[h * Dv + dv_idx] = static_cast<InT>(out);
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
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
      T normed = w[lid * 4 + i] * static_cast<T>(static_cast<float>(x[c0 + i]) * inv);
      y[c0 + i] = normed * mlx_sigmoid(z[c0 + i]);
    }
"""


def _header(**values):
    return metal_kernels.SIGMOID + ''.join(f'constant constexpr float {k} = {float(v)!r}f;\n' for k, v in values.items())


def _name(base, *values):
    return '_'.join([base] + [repr(float(v)) for v in values]).replace('.', '_').replace('-', 'm')


@lru_cache(maxsize=None)
def _conv_kernel(scale, eps):
    return mx.fast.metal_kernel(name=_name('quanteeze_kda_conv', scale, eps), input_names=['state', 'x', 'w'],
                                output_names=['y', 'window'], source=_CONV,
                                header=_header(SCALE_Q=scale ** 2, SCALE_K=scale, EPS=eps))


@lru_cache(maxsize=None)
def _delta_kernel(lower):
    return mx.fast.metal_kernel(name=_name('quanteeze_kda_delta', lower),
                                input_names=['qkv', 'a', 'fbg', 'A_log', 'dt_bias', 'state_in'],
                                output_names=['y', 'state_out'], source=_DELTA, header=_header(LOWER=lower))


@lru_cache(maxsize=None)
def _gated_norm_kernel(eps):
    return mx.fast.metal_kernel(name=_name('quanteeze_kda_gated_norm', eps), input_names=['x', 'z', 'w'],
                                output_names=['y'], source=_GATED_NORM, header=_header(EPS=eps))


@lru_cache(maxsize=None)
def _metal():
    return mx.metal.is_available()


def applies(layer, x, mask, cache):
    """Whether decode() takes this step: one token of one request, no mask, a cache holding a window
    and a state, no lengths, padding or speculative history, heads of 128, bfloat16 throughout."""
    if not (linear_attn.ON[0] and _metal() and mask is None and cache is not None and x.shape[:2] == (1, 1)
            and x.dtype == mx.bfloat16 and not layer.training and layer.head_dim == 128
            and layer.lower_bound is not None):
        return False
    window, state = cache[0], cache[1]
    ch, taps = 3 * layer.projection_dim, layer.conv_kernel
    return (window is not None and state is not None and cache.lengths is None and cache.left_padding is None
            and not getattr(cache, 'history_capacity', 0) and taps >= 2
            and window.shape == (1, taps - 1, ch) and window.dtype == mx.bfloat16
            and state.shape == (1, layer.num_heads, 128, 128) and state.dtype == mx.float32
            and layer.qkv_conv.conv.weight.shape == (ch, taps, 1) and layer.o_norm.weight.dtype == mx.bfloat16
            and layer.A_log.dtype == mx.float32 and layer.dt_bias.dtype == mx.float32)


def decode(layer, x, cache):
    """A decode step of Glm5NextLinearAttention through the three kernels: its output, the cache's
    window and state updated as mlx-vlm updates them."""
    h, d, taps = layer.num_heads, layer.head_dim, layer.conv_kernel
    ch = 3 * h * d
    proj = linear(layer.qkv_proj, x)
    y, window = _conv_kernel(float(layer.scale), 1e-6 / d)(
        inputs=[cache[0], proj, layer.qkv_conv.conv.weight],
        template=[('T', proj.dtype), ('CH', ch), ('TAPS', taps), ('H', h)], grid=(32 * 3 * h, 1, 1),
        threadgroup=(32, 1, 1), output_shapes=[proj.shape, (1, taps - 1, ch)], output_dtypes=[proj.dtype] * 2)
    cache[0] = window
    fbg = linear(layer.fbg_a_proj, x)
    f_a, _, g_a = mx.split(fbg, (d, d + h), axis=-1)
    a = linear(layer.f_b_proj, f_a)
    state = cache[1]
    out, state = _delta_kernel(float(layer.lower_bound))(
        inputs=[y, a, fbg, layer.A_log, layer.dt_bias, state],
        template=[('InT', y.dtype), ('StT', state.dtype), ('Dk', d), ('Dv', d), ('H', h)], grid=(32, d, h),
        threadgroup=(32, 4, 1), output_shapes=[(1, 1, h, d), state.shape], output_dtypes=[y.dtype, state.dtype])
    cache[1] = state
    cache.advance(1)
    gate = linear(layer.g_b_proj, g_a)
    o = _gated_norm_kernel(float(layer.o_norm.eps))(
        inputs=[out, gate, layer.o_norm.weight], template=[('T', out.dtype)], grid=(32 * h, 1, 1),
        threadgroup=(32, 1, 1), output_shapes=[(1, 1, h * d)], output_dtypes=[out.dtype])[0]
    return linear(layer.o_proj, o)


def step(layer, x, cache, stock):
    """The layer's decode step: decode() where this shape matched stock() (mlx-vlm's __call__) bit for
    bit the first CHECKS times it was met, else stock()."""
    key = (tuple(layer.qkv_proj.weight.shape), layer.num_heads, layer.conv_kernel, float(layer.scale),
           float(layer.lower_bound), float(layer.o_norm.eps))
    choice, seen = kernel_checks.read(CHOICE, CHECKED, key)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS['served'] += 1
            return decode(layer, x, cache)
        return stock()
    before = (cache[0], cache[1])
    want = stock()
    after = (cache[0], cache[1])
    STATS['checked'] += 1
    try:
        cache[0], cache[1] = before
        got = decode(layer, x, cache)
        same = kernel_checks.same_bits((got, cache[0], cache[1]), (want, after[0], after[1]))
    except Exception as error:                              # a kernel that does not build: mlx-vlm's
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: GLM-5.3\'s linear attention kernels unavailable ({type(error).__name__}: {error})',
                  file=sys.stderr, flush=True)
        same = False
    cache[0], cache[1] = after
    STATS['mismatched'] += not same
    per_thread.checked(CHOICE, CHECKED, key, bool(same), CHECKS)
    return want


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
