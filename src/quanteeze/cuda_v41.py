# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Fused CUDA kernels for DeepSeek V4.1 (`QUANTEEZE_HC_KERNELS`, `QUANTEEZE_CUDA_ROUTER`):
- `split`: Hyper-connection Sinkhorn split (`hc_split_sinkhorn`) in a single CUDA kernel;
- `fp8_roundtrip`: UE8M0 FP8 fake quantization (`fake_quant_fp8_ue8m0`) in a single warp kernel;
- `hc_pre` / `hc_expand`: Hyper-connection stream collapse and expansion in one kernel each;
- `route`: Post-logit MoE top-k router (`sqrt(softplus)` + biased top-k + score renormalization) in one kernel.

Every kernel is verified for bit-exact equality against MLX's reference ops at install time before
being enabled."""
import sys

import mlx.core as mx
import numpy as np

from . import cuda_kernels, kernel_checks

ON = [True]                     # QUANTEEZE_HC_KERNELS
CHECK_ROWS = 256
STATS = dict(checked=0, mismatched=0)

SOURCE = r"""
  const long long row = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= meta[0]) return;
  const float eps = epsilon[0];
  const float* m = mixes + row * (2 + HC) * HC;
  const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
  auto lin = [](float a, float s, float b) -> float { return Add{}(Multiply{}(a, s), b); };
  for (int i = 0; i < HC; i++) {
    pre[row * HC + i] = __fadd_rn(Sigmoid{}(lin(m[i], s0, base[i])), eps);
    post[row * HC + i] = __fmul_rn(2.0f, Sigmoid{}(lin(m[HC + i], s1, base[HC + i])));
  }
  float c[HC * HC];
  for (int k = 0; k < HC * HC; k++) c[k] = lin(m[2 * HC + k], s2, base[2 * HC + k]);
  for (int r = 0; r < HC; r++) {
    float* v = c + r * HC;
    float top = v[0];
    for (int j = 1; j < HC; j++) top = fmaxf(top, v[j]);
    float e[HC];
    for (int j = 0; j < HC; j++) e[j] = __expf(__fsub_rn(v[j], top));
    float sum = e[0];
    for (int j = 1; j < HC; j++) sum = __fadd_rn(sum, e[j]);
    const float inv = __frcp_rn(sum);
    for (int j = 0; j < HC; j++) v[j] = __fadd_rn(__fmul_rn(e[j], inv), eps);
  }
  auto columns = [&]() {
    for (int j = 0; j < HC; j++) {
      float sum = c[j];
      for (int r = 1; r < HC; r++) sum = __fadd_rn(sum, c[r * HC + j]);
      sum = __fadd_rn(sum, eps);
      for (int r = 0; r < HC; r++) c[r * HC + j] = __fdiv_rn(c[r * HC + j], sum);
    }
  };
  auto rows = [&]() {
    for (int r = 0; r < HC; r++) {
      float sum = c[r * HC];
      for (int j = 1; j < HC; j++) sum = __fadd_rn(sum, c[r * HC + j]);
      sum = __fadd_rn(sum, eps);
      for (int j = 0; j < HC; j++) c[r * HC + j] = __fdiv_rn(c[r * HC + j], sum);
    }
  };
  columns();
  for (int it = 0; it < ITERS - 1; it++) { rows(); columns(); }
  for (int k = 0; k < HC * HC; k++) comb[row * HC * HC + k] = c[k];
"""

_KERNEL = []


def _gpu():
    """Whether this rank computes on a CUDA GPU (not a CPU rank in a ring on a CUDA host)."""
    return cuda_kernels.available() and mx.default_device() == mx.gpu


def available():
    """Whether V4.1 CUDA hyper-connection kernels are enabled and running on a CUDA GPU."""
    return ON[0] and _gpu()


def split(mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    """Fused `hc_split_sinkhorn(mixes, scale, base, hc_mult, sinkhorn_iters, eps)` returning `(pre, post, comb)`."""
    if not _KERNEL:
        _KERNEL.append(
            cuda_kernels.kernel(
                'quanteeze_v41_hc_split', ['mixes', 'scale', 'base', 'epsilon', 'meta'], ['pre', 'post', 'comb'], SOURCE
            )
        )
    lead = tuple(mixes.shape[:-1])
    rows = int(np.prod(lead)) if lead else 1
    return _KERNEL[0](
        inputs=[
            mixes.astype(mx.float32),
            scale.astype(mx.float32),
            base.astype(mx.float32),
            mx.array([eps], dtype=mx.float32),
            mx.array(np.array([rows], np.int64)),
        ],
        template=[('HC', int(hc_mult)), ('ITERS', int(sinkhorn_iters))],
        grid=(-(-rows // 32) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[lead + (hc_mult,), lead + (hc_mult,), lead + (hc_mult, hc_mult)],
        output_dtypes=[mx.float32] * 3,
    )


def matches(blocks, hc_mult, sinkhorn_iters, eps, rows=CHECK_ROWS, seed=0):
    from mlx_vlm.models.deepseek_v4.hyper_connection import _hc_split_sinkhorn_ops
    ops = mx.compile(lambda m, s, b: _hc_split_sinkhorn_ops(m, s, b, hc_mult, sinkhorn_iters, eps))
    rng = np.random.default_rng(seed)
    width = (2 + hc_mult) * hc_mult
    every = True
    for block in blocks:
        for part in ('attn', 'ffn'):
            scale = getattr(block, f'hc_{part}_scale').astype(mx.float32)
            base = getattr(block, f'hc_{part}_base').astype(mx.float32)
            mixes = mx.array(rng.normal(0, 3, (1, rows, width)).astype(np.float32))
            want, got = ops(mixes, scale, base), split(mixes, scale, base, hc_mult, sinkhorn_iters, eps)
            same = kernel_checks.same_bits(got, want)
            STATS['checked'] += 1
            if not same:
                STATS['mismatched'] += 1
                every = False
    return every


def choose(blocks, config):
    hc_mult, iters, eps = int(config.hc_mult), int(config.hc_sinkhorn_iters), float(config.hc_eps)
    if not available() or hc_mult != 4 or not blocks:
        return False
    ok = matches(blocks, hc_mult, iters, eps)
    print(
        f"quanteeze: DeepSeek V4.1 hyper-connection Sinkhorn ({iters} rounds) CUDA kernel: "
        + ('bit-exact match across all blocks' if ok else
           f'{STATS["mismatched"]}/{STATS["checked"]} checks differed; keeping MLX ops')
        + ' (QUANTEEZE_HC_KERNELS)',
        file=sys.stderr,
        flush=True,
    )
    return ok


FP8_SOURCE = r"""
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const float value = static_cast<float>(x[i]);
  float a = fabsf(value);
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  const float amax = fmaxf(a, 1e-4f);
  const unsigned bits = __float_as_uint(__fmul_rn(amax, 1.0f / 448.0f));
  const int e = static_cast<int>((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u);
  const float scale = __uint_as_float(static_cast<unsigned>(e + 127) << 23);
  const float v = fminf(fmaxf(__fdiv_rn(value, scale), -448.0f), 448.0f);
  const int qexp = max(static_cast<int>(__float_as_uint(fabsf(v)) >> 23) - 3, 118);
  const float step = __uint_as_float(static_cast<unsigned>(qexp) << 23);
  out[i] = static_cast<T>(__fmul_rn(__fmul_rn(rintf(__fdiv_rn(v, step)), step), scale));
"""
_FP8 = []
FP8_CHOICE = {}
_ORIGINAL = [None]


def fp8_roundtrip(x):
    if not _FP8:
        _FP8.append(cuda_kernels.kernel('quanteeze_v41_fp8_roundtrip', ['x'], ['out'], FP8_SOURCE))
    return _FP8[0](
        inputs=[x],
        template=[('T', x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def _fp8_ops(x):
    from mlx_vlm.models.deepseek_v41 import fakequant
    kept, fakequant._fp8_roundtrip_kernel = fakequant._fp8_roundtrip_kernel, None
    try:
        return fakequant.fake_quant_fp8_ue8m0(x)
    finally:
        fakequant._fp8_roundtrip_kernel = kept


def fp8_matches(x):
    got, want = fp8_roundtrip(x), _fp8_ops(x)
    return kernel_checks.same_bits(got, want)


class _AsMlxVlmKernel:
    def __call__(self, inputs, template, grid, threadgroup, output_shapes, output_dtypes, **kwargs):
        return [fp8_roundtrip(inputs[0])]


FP8_CHECK_SHAPES = [
    (1, 1, 5120), (1, 1, 1280), (1, 1, 512), (1, 1, 2304), (1, 7, 5120), (3, 5, 1280),
    (1, 1, 4, 5120), (1, 300, 512),
]


def install_fp8(seed=0):
    from mlx_vlm.models.deepseek_v41 import fakequant
    mine = isinstance(fakequant._fp8_roundtrip_kernel, _AsMlxVlmKernel)
    if not available():
        if mine:
            fakequant._fp8_roundtrip_kernel = _ORIGINAL[0]
        return False
    if mine:
        return True
    _ORIGINAL[:] = [fakequant._fp8_roundtrip_kernel]
    rng = np.random.default_rng(seed)
    ok, checked = True, 0
    for shape in FP8_CHECK_SHAPES:
        for dtype in (mx.bfloat16, mx.float32):
            for spread in (1e-6, 1e-3, 1.0, 30.0, 1e4):
                x = rng.normal(0, spread, shape).astype(np.float32)
                x[..., :32] = 0.0
                x = mx.array(x).astype(dtype)
                mx.eval(x)
                same = fp8_matches(x)
                FP8_CHOICE[(shape, str(dtype))] = FP8_CHOICE.get((shape, str(dtype)), True) and same
                ok, checked = ok and same, checked + 1
    if ok:
        fakequant._fp8_roundtrip_kernel = _AsMlxVlmKernel()
    print(
        'quanteeze: DeepSeek V4.1 FP8 fake-quant CUDA kernel: '
        + ('bit-exact match' if ok else 'mismatch; keeping MLX ops')
        + f' over {checked} checks (QUANTEEZE_HC_KERNELS)',
        file=sys.stderr,
        flush=True,
    )
    return ok


PRE_SOURCE = r"""
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long row = i / D, d = i % D;
  float s = __fmul_rn(pre[row * 4], static_cast<float>(x[row * 4 * D + d]));
  for (int h = 1; h < 4; h++)
    s = __fadd_rn(s, __fmul_rn(pre[row * 4 + h], static_cast<float>(x[(row * 4 + h) * D + d])));
  out[i] = static_cast<T>(s);
"""
EXPAND_SOURCE = r"""
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long d = i % D, j = (i / D) % 4, row = i / (4 * D);
  float mixed = __fmul_rn(comb[row * 16 + j], static_cast<float>(res[row * 4 * D + d]));
  for (int h = 1; h < 4; h++)
    mixed = __fadd_rn(mixed, __fmul_rn(comb[(row * 4 + h) * 4 + j], static_cast<float>(res[(row * 4 + h) * D + d])));
  out[i] = static_cast<T>(Add{}(Multiply{}(post[row * 4 + j], static_cast<float>(x[row * D + d])), mixed));
"""
_PREPOST = {}
PREPOST = {}


def _built(name, inputs, source):
    if name not in _PREPOST:
        _PREPOST[name] = cuda_kernels.kernel(f'quanteeze_v41_{name}', inputs, ['out'], source)
    return _PREPOST[name]


def _group(n):
    group = 256
    while n % group:
        group //= 2
    return group


def _takes(x, copies):
    return copies == 4 and x.shape[-1] % 32 == 0 and x.size > 0


def hc_pre(x, pre_mix):
    if not _takes(x, x.shape[-2]):
        return PREPOST['hc_pre'](x, pre_mix)
    D = x.shape[-1]
    n = x.size // 4
    return _built('hc_pre', ['x', 'pre'], PRE_SOURCE)(
        inputs=[x, pre_mix.astype(mx.float32)],
        template=[('T', x.dtype), ('D', D)],
        grid=(n, 1, 1),
        threadgroup=(_group(n), 1, 1),
        output_shapes=[x.shape[:-2] + (D,)],
        output_dtypes=[x.dtype],
    )[0]


def hc_expand(x, residual, post, comb):
    if not _takes(x, residual.shape[-2]):
        return PREPOST['hc_expand'](x, residual, post, comb)
    D = x.shape[-1]
    n = residual.size
    return _built('hc_expand', ['x', 'res', 'post', 'comb'], EXPAND_SOURCE)(
        inputs=[x, residual, post.astype(mx.float32), comb.astype(mx.float32)],
        template=[('T', x.dtype), ('D', D)],
        grid=(n, 1, 1),
        threadgroup=(_group(n), 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[x.dtype],
    )[0]


PREPOST_CHECK = [((1, 1), 5120), ((1, 64), 5120), ((2, 5), 1024), ((1, 3), 128)]


def prepost_matches(seed=0, shapes=PREPOST_CHECK):
    from mlx_vlm.models.deepseek_v41 import language
    if not PREPOST:
        PREPOST.update(hc_pre=language.DeepseekV41Block.hc_pre, hc_expand=language.hc_expand)
    ops_pre, ops_expand = PREPOST['hc_pre'], PREPOST['hc_expand']
    rng = np.random.default_rng(seed)
    for lead, D in shapes:
        for spread in (1.0, 30.0):
            x4 = mx.array(rng.normal(0, spread, lead + (4, D)).astype(np.float32)).astype(mx.bfloat16)
            x = mx.array(rng.normal(0, spread, lead + (D,)).astype(np.float32)).astype(mx.bfloat16)
            pre = mx.array(rng.uniform(0, 1, lead + (4,)).astype(np.float32))
            post = mx.array(rng.uniform(0, 2 * spread, lead + (4,)).astype(np.float32))
            comb = mx.array(rng.dirichlet(np.ones(4), lead + (4,)).astype(np.float32) * spread)
            if not (
                kernel_checks.same_bits(hc_pre(x4, pre), ops_pre(x4, pre))
                and kernel_checks.same_bits(hc_expand(x, x4, post, comb), ops_expand(x, x4, post, comb))
            ):
                return False
    return True


def install_prepost():
    from mlx_vlm.models.deepseek_v41 import language
    mine = language.hc_expand is hc_expand
    if not available():
        if mine:
            language.DeepseekV41Block.hc_pre = staticmethod(PREPOST['hc_pre'])
            language.hc_expand = PREPOST['hc_expand']
        return False
    if mine:
        return True
    PREPOST.clear()
    PREPOST.update(hc_pre=language.DeepseekV41Block.hc_pre, hc_expand=language.hc_expand)
    ok = prepost_matches()
    if ok:
        language.DeepseekV41Block.hc_pre = staticmethod(hc_pre)
        language.hc_expand = hc_expand
    print(
        'quanteeze: DeepSeek V4.1 hyper-connection collapse/expand CUDA kernels: '
        + ('bit-exact match' if ok else 'mismatch; keeping MLX ops')
        + ' (QUANTEEZE_HC_KERNELS)',
        file=sys.stderr,
        flush=True,
    )
    return ok


ROUTE_ON = [True]               # QUANTEEZE_CUDA_ROUTER
ROUTE = [False]
ROUTE_ROWS_BIAS = [False]       # route() with a correction bias a row (a pass with an image mask: bias_vl where the
                                # mask is set), matched at install on random masks.  The model makes the mask from the
                                # ids of every pass unless its vision is let go (no_vision), so a route() that took
                                # only the gate's own bias [E] served no pass under QUANTEEZE_V41_READS_IMAGES=on
ROUTE_SOURCE = r"""
  constexpr float INF = cuda::std::numeric_limits<float>::infinity();
  const int row = blockIdx.x, e = threadIdx.x, lane = e % 32, w = e / 32;
  const float score = Sqrt{}(LogAddExp{}(static_cast<float>(logits[row * E + e]), 0.0f));
  scores[row * E + e] = score;
  float key = -Add{}(score, static_cast<float>(bias[(bias_ndim > 1 ? row * E : 0) + e]));
  __shared__ float best_key[32];
  __shared__ int best_idx[32];
  __shared__ int chosen[K];
  for (int r = 0; r < K; r++) {
    float k = key; int i = e;
    for (int o = 16; o > 0; o >>= 1) {
      const float k2 = __shfl_xor_sync(0xffffffffu, k, o); const int i2 = __shfl_xor_sync(0xffffffffu, i, o);
      if (k2 < k || (k2 == k && i2 < i)) { k = k2; i = i2; }
    }
    if (lane == 0) { best_key[w] = k; best_idx[w] = i; }
    __syncthreads();
    if (w == 0) {
      k = lane < E / 32 ? best_key[lane] : INF; i = lane < E / 32 ? best_idx[lane] : 0x7fffffff;
      for (int o = 16; o > 0; o >>= 1) {
        const float k2 = __shfl_xor_sync(0xffffffffu, k, o); const int i2 = __shfl_xor_sync(0xffffffffu, i, o);
        if (k2 < k || (k2 == k && i2 < i)) { k = k2; i = i2; }
      }
      if (lane == 0) chosen[r] = i;
    }
    __syncthreads();
    if (e == chosen[r]) key = INF;
    __syncthreads();
  }
  if (e == 0) {
    float wv[K];
    for (int r = 0; r < K; r++) { wv[r] = scores[row * E + chosen[r]]; inds[row * K + r] = chosen[r]; }
    float a = wv[0];
    for (int r = 1; r < 4 && r < K; r++) a = __fadd_rn(a, wv[r]);
    float b = 0.0f;
    for (int r = 4; r < K; r++) b = __fadd_rn(b, wv[r]);
    const float denom = Add{}(K > 4 ? __fadd_rn(a, b) : a, 1e-20f);
    for (int r = 0; r < K; r++) weights[row * K + r] = Multiply{}(Divide{}(wv[r], denom), scale[0]);
  }
"""
_ROUTE = []
_SCALES = {}


def takes_route(logits, k):
    E = logits.shape[-1]
    return logits.dtype == mx.float32 and E % 32 == 0 and 32 <= E <= 1024 and 1 < k <= 8 and logits.size > 0


def takes_bias(logits, bias):
    """Whether route() takes this correction bias: the gate's own [E]; or a row's each in the logits' shape (a pass
    with an image mask), once that matched at install (ROUTE_ROWS_BIAS)."""
    if bias.ndim == 1:
        return bias.size == logits.shape[-1]
    return bool(ROUTE_ROWS_BIAS[0]) and tuple(bias.shape) == tuple(logits.shape)


def route(logits, bias, scale, k):
    """(inds, weights, scores) of RowGate from its logits.  bias [E], or a row's each in the logits' shape (more than
    one dimension: the kernel reads it by row)."""
    if not _ROUTE:
        _ROUTE.append(
            cuda_kernels.kernel('quanteeze_v41_route', ['logits', 'bias', 'scale'], ['inds', 'weights', 'scores'], ROUTE_SOURCE)
        )
    if scale not in _SCALES:
        _SCALES[scale] = mx.array([scale], dtype=mx.float32)
        mx.eval(_SCALES[scale])
    E = logits.shape[-1]
    rows = logits.size // E
    return _ROUTE[0](
        inputs=[logits, bias, _SCALES[scale]],
        template=[('E', E), ('K', int(k))],
        grid=(rows * E, 1, 1),
        threadgroup=(E, 1, 1),
        output_shapes=[logits.shape[:-1] + (k,), logits.shape[:-1] + (k,), logits.shape],
        output_dtypes=[mx.int32, mx.float32, mx.float32],
    )


def install_route(gates, rows=(1, 7, 33), seed=0):
    ROUTE[0] = ROUTE_ROWS_BIAS[0] = False
    if not (ROUTE_ON[0] and _gpu()) or not gates:
        return False
    rng = np.random.default_rng(seed)
    ok = True
    for gate in gates:
        for n in rows:
            x = mx.array(rng.normal(0, 1, (1, n, gate.weight.shape[-1])).astype(np.float32)).astype(mx.bfloat16)
            logits = gate.logits(x)
            if not takes_route(logits, gate.top_k):
                return False
            want = gate.by_ops(logits)
            got = route(logits, gate.bias, gate.routed_scaling_factor, gate.top_k)
            ok = ok and kernel_checks.same_bits(got, want)
        if not ok:
            break
    ROUTE[0] = ok
    ROUTE_ROWS_BIAS[0] = ok and _route_masked_matches(gates, rows, rng)
    print(
        'quanteeze: DeepSeek V4.1 router CUDA kernel: '
        + ('bit-exact match across all gates' if ok else 'mismatch; keeping MLX ops')
        + ((', with an image mask\'s bias a row too' if ROUTE_ROWS_BIAS[0] else ', not with an image mask')
           if ok else '')
        + ' (QUANTEEZE_CUDA_ROUTER)',
        file=sys.stderr,
        flush=True,
    )
    return ok


def _route_masked_matches(gates, rows, rng):
    """Whether route() with a correction bias a row (bias_vl where a random image mask is set: half the tokens, none
    and all) gives each gate's by_ops with that bias -- the picks, the weights, the scores."""
    for gate in gates:
        if getattr(gate, 'bias_vl', None) is None:
            return False
        for n, share in zip(rows, (0.5, 0.0, 1.0)):
            x = mx.array(rng.normal(0, 1, (1, n, gate.weight.shape[-1])).astype(np.float32)).astype(mx.bfloat16)
            logits = gate.logits(x)
            mask = mx.array(rng.random((1, n)) < share)
            bias = mx.where(mask[..., None], gate.bias_vl, gate.bias)
            if tuple(bias.shape) != tuple(logits.shape):
                return False
            want = gate.by_ops(logits, bias)
            got = route(logits, bias, gate.routed_scaling_factor, gate.top_k)
            if not kernel_checks.same_bits(got, want):
                return False
    return True


ATTN_ON = [True]                # QUANTEEZE_CUDA_ATTN
ATTN = {}
GLUE_SOURCE = r"""
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  const int row = blockIdx.x, t = threadIdx.x, lane = warp.thread_rank(), w = warp.meta_group_rank();
  constexpr float NEG_INF = -cuda::std::numeric_limits<float>::infinity();
  constexpr float LOWEST = cuda::std::numeric_limits<float>::lowest();
  cg::greater<float> max_op;
  cg::plus<float> plus_op;
  __shared__ float local_max[32];
  __shared__ float local_norm[32];
  const int W = window_shape[window_ndim - 1], P = pooled_shape[pooled_ndim - 1];
  auto cover = [](int n) { int need = (n + 3) / 4, b = 32; while (b < need && b < 1024) b *= 2; return b; };
  const int TW = cover(W), TP = cover(P);
  auto lse = [&](const float* in, const bool* mask, int n, int threads) -> float {
    const int nw = threads / 32;
    float vals[4];
    for (int i = 0; i < 4; i++) {
      const int c = t * 4 + i;
      vals[i] = (t < threads && c < n) ? (mask[c] ? in[c] : LOWEST) : NEG_INF;
    }
    float maxval = LOWEST, normalizer = 0.0f, prevmax = maxval;
    for (int i = 0; i < 4; i++) maxval = max_op(maxval, vals[i]);
    normalizer = normalizer * __expf(prevmax - maxval);
    for (int i = 0; i < 4; i++) normalizer = normalizer + __expf(vals[i] - maxval);
    prevmax = maxval;
    maxval = cg::reduce(warp, maxval, max_op);
    normalizer = normalizer * __expf(prevmax - maxval);
    normalizer = cg::reduce(warp, normalizer, plus_op);
    prevmax = maxval;
    block.sync();
    if (lane == 0) local_max[w] = maxval;
    block.sync();
    maxval = lane < nw ? local_max[lane] : NEG_INF;
    maxval = cg::reduce(warp, maxval, max_op);
    normalizer = normalizer * __expf(prevmax - maxval);
    block.sync();
    if (lane == 0) local_norm[w] = (w < nw) ? normalizer : 0.0f;
    block.sync();
    normalizer = lane < nw ? local_norm[lane] : 0.0f;
    normalizer = cg::reduce(warp, normalizer, plus_op);
    return Add{}(logf(normalizer), maxval);
  };
  const float n1 = lse(window + row * W, wmask, W, TW);
  const float n2 = lse(pooled + row * P, pmask, P, TP);
  const float norm = LogAddExp{}(LogAddExp{}(n1, n2), sinks[row]);
  for (int j = t; j < W; j += blockDim.x)
    wout[row * W + j] = Exp{}(Subtract{}(wmask[j] ? window[row * W + j] : LOWEST, norm));
  for (int j = t; j < P; j += blockDim.x)
    pout[row * P + j] = Exp{}(Subtract{}(pmask[j] ? pooled[row * P + j] : LOWEST, norm));
"""
_GLUE = []


def _lse_threads(n):
    need, block = -(-n // 4), 32
    while block < need and block < 1024:
        block *= 2
    return block


def attention_glue(window, wmask, pooled, pmask, sinks):
    """Fused mask + logsumexp + logaddexp + split-softmax weights for decode sparse pooled attention."""
    if not _GLUE:
        _GLUE.append(
            cuda_kernels.kernel(
                'quanteeze_v41_attention_glue',
                ['window', 'wmask', 'pooled', 'pmask', 'sinks'],
                ['wout', 'pout'],
                GLUE_SOURCE,
            )
        )
    H, W, P = window.shape[1], window.shape[-1], pooled.shape[-1]
    block = max(_lse_threads(W), _lse_threads(P))
    return _GLUE[0](
        inputs=[window, wmask, pooled, pmask, sinks],
        grid=(H * block, 1, 1),
        threadgroup=(block, 1, 1),
        output_shapes=[window.shape, pooled.shape],
        output_dtypes=[mx.float32, mx.float32],
    )


def _takes_attention(q, local_kv, pooled, topk, local_mask, pooled_mask, sinks):
    del pooled
    return (
        q.ndim == 4 and q.shape[0] == 1 and q.shape[2] == 1 and q.dtype == mx.float32 and sinks is not None
        and topk.ndim == 3 and topk.shape[:2] == (1, 1) and 0 < topk.shape[-1] <= 4096
        and 0 < local_kv.shape[-2] <= 4096
        and local_mask is not None and local_mask.dtype == mx.bool_ and local_mask.size == local_kv.shape[-2]
        and pooled_mask is not None and pooled_mask.dtype == mx.bool_ and pooled_mask.size == topk.shape[-1]
    )


def sparse_pooled_attention(q, local_kv, pooled, topk, local_mask, pooled_mask, scale, sinks):
    from mlx_vlm.models.deepseek_v4 import language as v4
    if not _takes_attention(q, local_kv, pooled, topk, local_mask, pooled_mask, sinks):
        return ATTN['original'](q, local_kv, pooled, topk, local_mask, pooled_mask, scale, sinks)
    pooled_sq = v4._gather_pooled(pooled, topk)
    q_scaled = q * scale
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = (q_bl @ pooled_sq.swapaxes(-1, -2)).transpose(0, 2, 1, 3)
    local_weights, pooled_weights = attention_glue(
        local_scores, local_mask.reshape(-1), pooled_scores, pooled_mask.reshape(-1), sinks.astype(mx.float32)
    )
    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


ATTN_CHECK = [(128, 512, 1000), (128, 37, 60), (40, 512, 700), (7, 300, 300), (128, 1, 4)]


def attention_matches(original, seed=0, cases=ATTN_CHECK, heads=8, dim=512):
    rng = np.random.default_rng(seed)
    for W, P, N in cases:
        q = mx.array(rng.normal(0, 1, (1, heads, 1, dim)).astype(np.float32))
        local_kv = mx.array(rng.normal(0, 1, (1, 1, W, dim)).astype(np.float32)).astype(mx.bfloat16)
        pooled = mx.array(rng.normal(0, 1, (1, N, dim)).astype(np.float32)).astype(mx.bfloat16)
        topk = rng.integers(0, N, (1, 1, P)).astype(np.int32)
        topk[..., -1:] = -1 if P > 1 else topk[..., -1:]
        topk = mx.array(topk)
        local_mask = mx.array(rng.random(W) < 0.9).reshape(1, 1, 1, W)
        pooled_mask = (topk != -1)[:, None]
        sinks = mx.array(rng.normal(0, 1, heads).astype(np.float32))
        args = (q, local_kv, pooled, topk, local_mask, pooled_mask, dim ** -0.5, sinks)
        if not kernel_checks.same_bits(sparse_pooled_attention(*args), original(*args)):
            return False
    return True


def install_attention():
    from mlx_vlm.models.deepseek_v41 import language
    mine = language._sparse_pooled_attention is sparse_pooled_attention
    if not (ATTN_ON[0] and _gpu()):
        if mine:
            language._sparse_pooled_attention = ATTN['original']
        ATTN['installed'] = False
        return False
    if mine:
        ATTN['installed'] = True
        return True
    ATTN['original'] = language._sparse_pooled_attention
    ok = attention_matches(ATTN['original'])
    ATTN['installed'] = ok
    if ok:
        language._sparse_pooled_attention = sparse_pooled_attention
    print(
        'quanteeze: DeepSeek V4.1 decode sparse-attention glue CUDA kernel: '
        + ('bit-exact match' if ok else 'mismatch; keeping MLX ops')
        + ' (QUANTEEZE_CUDA_ATTN)',
        file=sys.stderr,
        flush=True,
    )
    return ok


def cpp_kernels(ext):
    """Return `(fp8, glue)` CUDA kernel specs for `v41_attention`'s C++ extension matching what Python
    runs on CUDA (`fp8` when `install_fp8` succeeded, else `None`; `glue` when `install_attention` succeeded)."""
    from mlx_vlm.models.deepseek_v41 import fakequant, language
    fp8 = glue = None
    if isinstance(fakequant._fp8_roundtrip_kernel, _AsMlxVlmKernel):
        fp8 = ext.cuda_kernel_spec('quanteeze_v41_fp8_roundtrip', ['x'], ['out'], cuda_kernels.HEADER, FP8_SOURCE, 0)
    if language._sparse_pooled_attention is sparse_pooled_attention:
        glue = ext.cuda_kernel_spec(
            'quanteeze_v41_attention_glue',
            ['window', 'wmask', 'pooled', 'pmask', 'sinks'],
            ['wout', 'pout'],
            cuda_kernels.HEADER,
            GLUE_SOURCE,
            0,
        )
    return fp8, glue


def summary():
    return dict(
        hc=dict(STATS),
        route=bool(ROUTE[0]),
        fp8_checked=len(FP8_CHOICE),
        attn=bool(ATTN.get('installed', False)),
    )
