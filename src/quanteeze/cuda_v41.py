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

from . import cuda_kernels

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


def available():
    """Whether V4.1 CUDA hyper-connection kernels are enabled and running on MLX CUDA."""
    return ON[0] and cuda_kernels.available()


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
            same = all(mx.array_equal(g, w).item() for g, w in zip(got, want))
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
    return mx.array_equal(got, want).item()


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
                mx.array_equal(hc_pre(x4, pre), ops_pre(x4, pre)).item()
                and mx.array_equal(hc_expand(x, x4, post, comb), ops_expand(x, x4, post, comb)).item()
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
ROUTE_SOURCE = r"""
  constexpr float INF = cuda::std::numeric_limits<float>::infinity();
  const int row = blockIdx.x, e = threadIdx.x, lane = e % 32, w = e / 32;
  const float score = Sqrt{}(LogAddExp{}(static_cast<float>(logits[row * E + e]), 0.0f));
  scores[row * E + e] = score;
  float key = -Add{}(score, static_cast<float>(bias[e]));
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


def route(logits, bias, scale, k):
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
    ROUTE[0] = False
    if not (ROUTE_ON[0] and cuda_kernels.available()) or not gates:
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
            ok = ok and all(mx.array_equal(g, w).item() for g, w in zip(got, want))
        if not ok:
            break
    ROUTE[0] = ok
    print(
        'quanteeze: DeepSeek V4.1 router CUDA kernel: '
        + ('bit-exact match across all gates' if ok else 'mismatch; keeping MLX ops')
        + ' (QUANTEEZE_CUDA_ROUTER)',
        file=sys.stderr,
        flush=True,
    )
    return ok


def summary():
    return dict(
        hc=dict(STATS),
        route=bool(ROUTE[0]),
        fp8_checked=len(FP8_CHOICE),
    )
