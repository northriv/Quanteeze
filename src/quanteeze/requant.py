# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Uniform routed-expert quantization from a model's own weights, each group's base and scale
fit by alternating least squares.

MLX's affine format gives each group of `group_size` weights along a row a base `b` and a scale
`s`; a weight is `b + s * q` with `q` in `0 .. 2^bits - 1`.  `mx.quantize` takes `b` and `s` from
the group's minimum and maximum.  `fit()` takes them by least squares instead: from a start, it
alternates the levels -- each weight's nearest, given `b` and `s` -- with the regression of the
weights on their levels, which gives the `b` and `s` of least squared error for those levels.
Neither step can raise the error, so it falls to a local minimum in `ITERATIONS` rounds.  At 1 bit
this is 2-means, started from the centroids of the weights of each sign; above it, from the range.
The stored `b` and `s` are bf16, and the levels are chosen last for those, so they are nearest for
what is served.

`quantize()` computes both `fit()` and `pack()` in one Metal kernel, a SIMD group for each group
of weights, so lowering a model takes about the time to read the weights.  `lower_model()` lowers
every MoE layer's `SwitchGLU` in place to a uniform bit-width (`QUANTEEZE_BITS`), streaming in
chunks so memory never spikes; `Flat` lowers experts on the fly as they are read on demand from
the SSD (`quanteeze.demand`).
"""
import fcntl
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import arch, q2_experts, variables
from .q2_experts import PROJECTIONS

SCHEME = 'ls'
GROUP_SIZE = 128                      # group size for 1..3-bit least-squares quantization
ITERATIONS = 8
FORMATS = {'mxfp4': (32, 4), 'nvfp4': (16, 4), 'mxfp8': (32, 8)}     # mode: (group size, bits)
GGML_MODES = ('q3_k', 'q2_k', 'iq2_xs')
UNIFORM_MODES = ('mxfp4',) + GGML_MODES


def _norm_bits(bits):
    if isinstance(bits, str) and bits.strip().lower() in UNIFORM_MODES:
        return bits.strip().lower()
    return int(bits)


def effective_format(bits, in_dims, group_size=None):
    """Return `(int_bits, group_size, mode)` for uniform quantization spec `bits` on rows of width `in_dims`."""
    b = _norm_bits(bits)
    if b == 'mxfp4':
        if in_dims % 32 == 0:
            return 4, 32, 'mxfp4'
        return 4, 64, 'affine'
    if b in GGML_MODES:
        from . import ggml_quants
        fb = ggml_quants.FORMAT_BITS[b]
        if in_dims % 256 == 0:
            return fb, 256, b
        gs = GROUP_SIZE if in_dims % GROUP_SIZE == 0 else 64
        return fb, gs, 'affine'
    from . import fitting
    gs = fitting.group_size(b) if group_size is None else int(group_size)
    return b, gs, 'affine'


def source(path, bits, group_size=GROUP_SIZE):
    """The source string of a uniform quantization computed from `path`."""
    b = _norm_bits(bits)
    gs = 256 if b in GGML_MODES else (32 if b == 'mxfp4' else int(group_size))
    inner = str(path) if str(path).startswith(SCHEME + ':') else Path(path).expanduser().resolve()
    return f'{SCHEME}:{b}:{gs}:{inner}'


def parse(source_string):
    """(bits, group size, checkpoint path) of a least-squares or ggml source string."""
    scheme, _, rest = source_string.partition(':')
    if scheme != SCHEME:
        raise ValueError(f'{source_string}: not a {SCHEME} source')
    bits, group_size, path = rest.split(':', 2)
    return _norm_bits(bits), int(group_size), path


def _levels(g, base, scale, top):
    """Each weight's nearest level for a group's base and scale (level 0 where the scale is 0)."""
    safe = mx.where(scale > 0, scale, 1)
    return mx.where(scale > 0, mx.clip(mx.round((g - base) / safe), 0, top), 0)


def _regress(g, q):
    """The base and scale of least squared error for weights g on levels q, group by group."""
    mq, mg = q.mean(-1, keepdims=True), g.mean(-1, keepdims=True)
    var = ((q - mq) ** 2).mean(-1, keepdims=True)
    cov = ((q - mq) * (g - mg)).mean(-1, keepdims=True)
    scale = mx.where(var > 0, cov / mx.where(var > 0, var, 1), 0)
    return mg - scale * mq, scale


def fit(w, bits, group_size=GROUP_SIZE, iterations=ITERATIONS, dtype=mx.bfloat16):
    """Weights w [..., in] at `bits` in groups of `group_size`, the base and scale of each group
    by least squares: (levels [..., in] float32, scales [..., groups], biases [..., groups]),
    scales and biases in `dtype`, the levels nearest for those."""
    shape = w.shape
    g = w.astype(mx.float32).reshape(*shape[:-1], shape[-1] // group_size, group_size)
    top = 2 ** bits - 1
    if bits == 1:                                        # the centroids of each sign's weights
        q = (g >= 0).astype(mx.float32)
    else:                                                # the group's range, as mx.quantize takes it
        lo, hi = g.min(-1, keepdims=True), g.max(-1, keepdims=True)
        q = _levels(g, lo, (hi - lo) / top, top)
    for _ in range(iterations):
        base, scale = _regress(g, q)
        q = _levels(g, base, scale, top)
    base, scale = _regress(g, q)
    base, scale = base.astype(dtype), scale.astype(dtype)
    q = _levels(g, base.astype(mx.float32), scale.astype(mx.float32), top)
    return q.reshape(shape), scale[..., 0], base[..., 0]


def pack(levels, bits):
    """Levels [..., in] as MLX packs them: each row a little-endian stream of `bits`-bit
    fields in uint32 words [..., in * bits / 32]."""
    lv = levels.astype(mx.uint32)
    stream = (lv[..., None] >> mx.arange(bits, dtype=mx.uint32)) & 1                 # [..., in, bits]
    stream = stream.reshape(*lv.shape[:-1], -1, 32)                                  # [..., words, 32]
    return (stream << mx.arange(32, dtype=mx.uint32)).sum(-1).astype(mx.uint32)


def native(bits, mode='affine'):
    """Whether MLX's own kernels compute `bits`/`mode` arrays directly: ggml K/IQ formats and 1-bit
    affine on CUDA use `dequantize()` / `kquant_qmv` instead."""
    if mode in GGML_MODES or (isinstance(mode, str) and mode.startswith('gguf:')) or (isinstance(bits, str) and bits in GGML_MODES):
        return False
    return bits != 1 or mode != 'affine' or mx.metal.is_available() or mx.default_device() != mx.gpu


def unpack(weight, bits):
    """The levels [..., in] uint32 of words [..., in * bits / 32] packed as pack() packs them."""
    stream = (weight.astype(mx.uint32)[..., None] >> mx.arange(32, dtype=mx.uint32)) & 1     # [..., words, 32]
    stream = stream.reshape(*weight.shape[:-1], -1, bits)                                   # [..., in, bits]
    return (stream << mx.arange(bits, dtype=mx.uint32)).sum(-1).astype(mx.uint32)


@mx.compile
def _one_bit(weight, scales, biases):
    """1-bit affine arrays unpacked in one compiled kernel."""
    per = weight.shape[-1] // scales.shape[-1]                                       # words a group
    s = mx.repeat(scales.astype(mx.float32), per, axis=-1)[..., None]
    b = mx.repeat(biases.astype(mx.float32), per, axis=-1)[..., None]
    q = (weight[..., None] >> mx.arange(32, dtype=mx.uint32)) & 1                    # [..., words, 32]
    return (q.astype(mx.float32) * s + b).astype(scales.dtype).reshape(*weight.shape[:-1], -1)


def dequantize(weight, scales, biases=None, group_size=GROUP_SIZE, bits=4, mode='affine', dtype=None):
    """mx.dequantize, and where MLX cannot (native()), ggml K/IQ or 1-bit affine arrays unpacked here."""
    if mode in GGML_MODES:
        from . import ggml_quants
        return ggml_quants.dequantize(weight, scales, biases, mode)
    if isinstance(mode, str) and mode.startswith('gguf:'):
        from . import iq_experts
        kind = iq_experts.BY_NAME[mode[5:].upper()]
        return iq_experts.decode(
            weight.reshape(-1, weight.shape[-1]), kind,
            dtype=(mx.float32 if dtype is None else dtype),
        ).reshape(*weight.shape[:-2], weight.shape[-2] * iq_experts.per_block(kind))
    if native(bits, mode):
        return mx.dequantize(weight, scales, biases, group_size=group_size, bits=bits, mode=mode)
    if bits == 1:
        return _one_bit(weight.astype(mx.uint32), scales, biases)
    q = unpack(weight, bits).astype(mx.float32)
    q = q.reshape(*q.shape[:-1], -1, group_size)
    w = q * scales.astype(mx.float32)[..., None] + biases.astype(mx.float32)[..., None]
    return w.reshape(*w.shape[:-2], -1).astype(scales.dtype)


# fit() and pack() in one Metal kernel: one SIMD group (32 lanes) takes one group of weights,
# GS / 32 per lane, keeps them in registers through all ITERATIONS rounds, and packs the levels
# by simd_or.
_HEADER = """
template <int PER, int TOP>
inline void nearest(thread uint* q, thread const float* x, float base, float scale) {
  float safe = scale > 0 ? scale : 1.0f;
  for (int i = 0; i < PER; i++)
    q[i] = scale > 0 ? uint(metal::clamp(metal::rint((x[i] - base) / safe), 0.0f, float(TOP))) : 0u;
}
"""
_SOURCE = """
  constexpr int PER = GS / 32, TOP = (1 << BITS) - 1;
  uint group = thread_position_in_grid.x / 32, lane = thread_position_in_grid.x % 32;
  size_t at = size_t(group) * GS + lane;
  float x[PER];
  uint q[PER];
  float sx = 0, lo = INFINITY, hi = -INFINITY;
  for (int i = 0; i < PER; i++) {
    x[i] = float(w[at + 32 * i]);
    sx += x[i]; lo = metal::min(lo, x[i]); hi = metal::max(hi, x[i]);
  }
  float mg = simd_sum(sx) / GS, base, scale;
  if (BITS == 1) {                                     // the centroids of each sign's weights
    for (int i = 0; i < PER; i++) q[i] = x[i] >= 0 ? 1u : 0u;
  } else {                                             // the group's range
    base = simd_min(lo);
    scale = (simd_max(hi) - base) / TOP;
    nearest<PER, TOP>(q, x, base, scale);
  }
  for (int r = 0; r <= ITERATIONS; r++) {              // the regression, then the nearest levels
    float sq = 0, vq = 0, cq = 0;
    for (int i = 0; i < PER; i++) sq += q[i];
    float mq = simd_sum(sq) / GS;
    for (int i = 0; i < PER; i++) { float d = q[i] - mq; vq += d * d; cq += d * (x[i] - mg); }
    float var = simd_sum(vq) / GS, cov = simd_sum(cq) / GS;
    scale = var > 0 ? cov / var : 0.0f;
    base = mg - scale * mq;
    if (r == ITERATIONS) { scale = float(bfloat16_t(scale)); base = float(bfloat16_t(base)); }
    nearest<PER, TOP>(q, x, base, scale);
  }
  uint bit = lane * BITS, shift = bit % 32, word = bit / 32;
  size_t out = size_t(group) * GS * BITS / 32;
  for (int i = 0; i < PER; i++) {
    uint low = q[i] << shift, high = shift + BITS > 32 ? q[i] >> (32 - shift) : 0u;
    for (uint j = 0; j < BITS; j++) {
      uint v = simd_or((word == j ? low : 0u) | (word + 1 == j ? high : 0u));
      if (lane == 0) weight[out + i * BITS + j] = v;
    }
  }
  if (lane == 0) { scales[group] = bfloat16_t(scale); biases[group] = bfloat16_t(base); }
"""
_kernel = None


# The same on MLX's CUDA backend: a warp (32 lanes) a group of weights, the sums, the range and the
# packing's OR by shuffles across it.
_CUDA_HEADER = r"""
template <int PER, int TOP>
__device__ __forceinline__ void qz_nearest(unsigned* q, const float* x, float base, float scale) {
  float safe = scale > 0 ? scale : 1.0f;
  #pragma unroll
  for (int i = 0; i < PER; i++)
    q[i] = scale > 0 ? static_cast<unsigned>(fminf(fmaxf(rintf((x[i] - base) / safe), 0.0f), float(TOP))) : 0u;
}
__device__ __forceinline__ float qz_wsum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float qz_wmin(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fminf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float qz_wmax(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ unsigned qz_wor(unsigned v) {
  for (int o = 16; o > 0; o >>= 1) v |= __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
"""
_CUDA_SOURCE = r"""
  constexpr int PER = GS / 32, TOP = (1 << BITS) - 1;
  const long long group = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const unsigned lane = threadIdx.x % 32;
  if (group >= meta[0]) return;                        // a whole warp: blocks are whole warps
  const long long at = group * GS + lane;
  float x[PER];
  unsigned q[PER];
  float sx = 0, lo = INFINITY, hi = -INFINITY;
  #pragma unroll
  for (int i = 0; i < PER; i++) {
    x[i] = static_cast<float>(w[at + 32 * i]);
    sx += x[i]; lo = fminf(lo, x[i]); hi = fmaxf(hi, x[i]);
  }
  float mg = qz_wsum(sx) / GS, base = 0.0f, scale = 0.0f;
  if (BITS == 1) {                                     // the centroids of each sign's weights
    #pragma unroll
    for (int i = 0; i < PER; i++) q[i] = x[i] >= 0 ? 1u : 0u;
  } else {                                             // the group's range
    base = qz_wmin(lo);
    scale = (qz_wmax(hi) - base) / TOP;
    qz_nearest<PER, TOP>(q, x, base, scale);
  }
  for (int r = 0; r <= ITERATIONS; r++) {              // the regression, then the nearest levels
    float sq = 0, vq = 0, cq = 0;
    #pragma unroll
    for (int i = 0; i < PER; i++) sq += q[i];
    float mq = qz_wsum(sq) / GS;
    #pragma unroll
    for (int i = 0; i < PER; i++) { float d = q[i] - mq; vq += d * d; cq += d * (x[i] - mg); }
    float var = qz_wsum(vq) / GS, cov = qz_wsum(cq) / GS;
    scale = var > 0 ? cov / var : 0.0f;
    base = mg - scale * mq;
    if (r == ITERATIONS) {
      scale = __bfloat162float(__float2bfloat16(scale));
      base = __bfloat162float(__float2bfloat16(base));
    }
    qz_nearest<PER, TOP>(q, x, base, scale);
  }
  const unsigned bit = lane * BITS, shift = bit % 32, word = bit / 32;
  const long long out = group * GS * BITS / 32;
  #pragma unroll
  for (int i = 0; i < PER; i++) {
    unsigned low = q[i] << shift, high = shift + BITS > 32 ? q[i] >> (32 - shift) : 0u;
    for (unsigned j = 0; j < BITS; j++) {
      unsigned v = qz_wor((word == j ? low : 0u) | (word + 1 == j ? high : 0u));
      if (lane == 0) weight[out + i * BITS + j] = v;
    }
  }
  if (lane == 0) { scales[group] = __float2bfloat16(scale); biases[group] = __float2bfloat16(base); }
"""
_cuda_kernel = None
CUDA_LS = [True]                 # requant's least squares by the CUDA kernel where it builds (tests turn it off)


def _cuda():
    """Whether quantize() computes on MLX's CUDA backend by the kernel."""
    if not CUDA_LS[0] or mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    from . import cuda_kernels
    return cuda_kernels.available()


def quantize(w, bits, group_size=GROUP_SIZE, iterations=ITERATIONS):
    """Weights w [..., out, in] as MLX affine arrays `[weight uint32, scales, biases]` (bf16),
    MXFP4 `[weight uint32, scales uint8, None]` when `bits == 'mxfp4'`, or ggml K/IQ packed
    `[body uint8, d float16, dmin float16|None]` when `bits in GGML_MODES`."""
    global _kernel, _cuda_kernel
    b = _norm_bits(bits)
    if b == 'mxfp4':
        if w.shape[-1] % 32 == 0:
            wq, sq = mx.quantize(w, group_size=32, bits=4, mode='mxfp4')
            return [wq, sq, None]
        return quantize(w, 4, 64, iterations)
    if b in GGML_MODES:
        from . import ggml_quants
        if w.shape[-1] % 256 == 0:
            return list(ggml_quants.pack_arrays(w, b, iterations))
        fb, fgs, _ = effective_format(b, w.shape[-1])
        return quantize(w, fb, fgs, iterations)
    bits = int(b)
    if group_size % 32 == 0 and not mx.metal.is_available() and _cuda():
        if _cuda_kernel is None:
            from . import cuda_kernels
            _cuda_kernel = cuda_kernels.kernel('quanteeze_least_squares', ['w', 'meta'],
                                               ['weight', 'scales', 'biases'], _CUDA_SOURCE, header=_CUDA_HEADER)
        groups = w.size // group_size
        return _cuda_kernel(inputs=[mx.contiguous(w), mx.array(np.array([groups], np.int64))],
                            template=[('GS', group_size), ('BITS', bits), ('ITERATIONS', iterations)],
                            grid=(32 * max(groups, 1), 1, 1), threadgroup=(256, 1, 1),
                            output_shapes=[(*w.shape[:-1], w.shape[-1] * bits // 32),
                                           (*w.shape[:-1], w.shape[-1] // group_size),
                                           (*w.shape[:-1], w.shape[-1] // group_size)],
                            output_dtypes=[mx.uint32, mx.bfloat16, mx.bfloat16])
    if not mx.metal.is_available() or group_size % 32:
        q, scales, biases = fit(w, bits, group_size, iterations)
        return [pack(q, bits), scales, biases]
    if _kernel is None:
        _kernel = mx.fast.metal_kernel('requant_least_squares', ['w'], ['weight', 'scales', 'biases'],
                                       _SOURCE, header=_HEADER)
    groups = w.size // group_size
    return _kernel(inputs=[w], template=[('GS', group_size), ('BITS', bits), ('ITERATIONS', iterations)],
                   grid=(32 * groups, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(*w.shape[:-1], w.shape[-1] * bits // 32),
                                  (*w.shape[:-1], w.shape[-1] // group_size),
                                  (*w.shape[:-1], w.shape[-1] // group_size)],
                   output_dtypes=[mx.uint32, mx.bfloat16, mx.bfloat16])


CACHE_FORMAT = 2


def backend():
    """Where a quantization is computed: by the kernel on Metal, by the kernel on CUDA ('cuda-ls'),
    by fit() elsewhere ('cuda', 'cpu')."""
    if mx.metal.is_available():
        return 'metal'
    if _cuda():
        return 'cuda-ls'
    return 'cuda' if mx.default_device() == mx.gpu else 'cpu'


def stamp(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob('*.safetensors'))
    return [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]


class QuantCache:
    """Lowered least-squares expert weights kept on disk under `root` (`QUANTEEZE_QUANT_CACHE`), so
    a checkpoint lowered once loads straight from `.npy` files on subsequent starts."""

    NAMES = ('weight', 'scales', 'biases')

    def __init__(self, root, origin, checkpoint):
        key = {'origin': origin, 'stamp': stamp(checkpoint), 'iterations': ITERATIONS, 'backend': backend(),
               'format': CACHE_FORMAT}
        text = json.dumps(key, sort_keys=True)
        self.dir = Path(root).expanduser() / hashlib.sha256(text.encode()).hexdigest()[:16]
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / 'meta.json'
        if not meta.exists():
            tmp = self.dir / f'meta.json.{os.getpid()}'
            tmp.write_text(json.dumps(key, indent=1))
            os.replace(tmp, meta)
        self._headers, self._pool = {}, None

    def _path(self, layer, projection, name):
        return self.dir / f'{layer}.{projection}.{name}.npy'

    def _header(self, path):
        if path not in self._headers:
            with open(path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                read = np.lib.format.read_array_header_1_0 if version == (1, 0) else \
                    np.lib.format.read_array_header_2_0
                shape, fortran, dtype = read(f)
                if fortran:
                    raise ValueError(f'{path}: kept in Fortran order')
                self._headers[path] = f.tell(), shape, dtype
        return self._headers[path]

    def _rows(self, paths, experts):
        if self._pool is None:
            self._workers = q2_experts.read_threads(paths[0] if paths else self.path)
            self._pool = ThreadPoolExecutor(self._workers, thread_name_prefix='quant-cache')
        for header in [self._pool.submit(self._header, path) for path in paths if path not in self._headers]:
            header.result()
        parts = [p for p in np.array_split(np.arange(experts.size), self._workers) if p.size]
        outs, pending = [], []
        for path in paths:
            offset, shape, dtype = self._header(path)
            out = np.empty((experts.size,) + shape[1:], dtype)
            outs.append(out)
            pending += [self._pool.submit(q2_experts.read_rows, path, offset, shape, dtype, experts[p],
                                          out=out[p[0]:p[-1] + 1]) for p in parts]
        for p in pending:
            p.result()
        return outs

    def _write_rows(self, layer, projection, subset, arrays, total, done, done_path):
        """Write computed `arrays` for `subset` rows to per-tensor `.npy` memmaps and update `done`."""
        for name, a in zip(self.NAMES, arrays):
            if a is None:
                continue
            rows = np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a)
            path = self._path(layer, projection, name)
            kept = np.load(path, mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(
                path, mode='w+', dtype=rows.dtype, shape=(total,) + rows.shape[1:])
            kept[subset] = rows
            kept.flush()
            del kept
        done[subset] = True
        tmp = self.dir / f'{layer}.{projection}.done.{os.getpid()}.npy'
        np.save(tmp, done)
        os.replace(tmp, done_path)

    def get(self, layer, projection, experts, total, compute):
        experts = np.asarray(experts, np.int64)
        with open(self.dir / f'{layer}.{projection}.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            done_path = self._path(layer, projection, 'done')
            done = np.load(done_path) if done_path.exists() else np.zeros(total, bool)
            missing = np.unique(experts[~done[experts]])
            if missing.size:
                self._write_rows(layer, projection, missing, compute(missing), total, done, done_path)
            present_names = [name for name in self.NAMES if self._path(layer, projection, name).exists()]
            loaded = dict(zip(present_names, self._rows([self._path(layer, projection, n) for n in present_names], experts)))
        weight_mx = mx.array(loaded['weight'])
        scales_np = loaded['scales']
        scales_mx = mx.array(scales_np).view(mx.bfloat16) if scales_np.dtype == np.uint16 else mx.array(scales_np)
        biases_np = loaded.get('biases')
        biases_mx = (None if biases_np is None
                     else (mx.array(biases_np).view(mx.bfloat16) if biases_np.dtype == np.uint16 else mx.array(biases_np)))
        return [weight_mx, scales_mx, biases_mx]

    def ensure(self, layer, projection, total, compute, chunk=32, ahead=None):
        """Ensure all `0..total-1` expert rows for `(layer, projection)` are materialized on disk."""
        with open(self.dir / f'{layer}.{projection}.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            done_path = self._path(layer, projection, 'done')
            done = np.load(done_path) if done_path.exists() else np.zeros(total, bool)
            missing = np.flatnonzero(~done)
            for start in range(0, missing.size, chunk):
                sub = missing[start:start + chunk]
                if ahead is not None:
                    ahead(missing[start + chunk:start + 2 * chunk])
                self._write_rows(layer, projection, sub, compute(sub), total, done, done_path)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None


class Reader:
    """Computes (or loads from `QuantCache`) lowered routed-expert weights `[weight, scales, biases]`
    at uniform `bits` and `group_size` from a checkpoint."""

    def __init__(self, path, bits, group_size=GROUP_SIZE, chunk=32, cache=None):
        self.base = q2_experts.Checkpoint(path)
        self.bits = _norm_bits(bits)
        self.group_size = 256 if self.bits in GGML_MODES else (32 if self.bits == 'mxfp4' else int(group_size))
        self.path, self.chunk = self.base.path, chunk
        self.origin = source(path, self.bits, self.group_size)
        self._config = None
        if cache is None and not variables.off('QUANTEEZE_QUANT_CACHE'):
            cache = variables.get('QUANTEEZE_QUANT_CACHE')
        self.cache = QuantCache(cache, self.origin, self.base.path) if cache else None

    def count(self, layer, projection):
        return self.base.tensors[self.base.name(layer, projection)][2][0]

    def _quantization(self, layer, projection):
        """(mode, group size, bits) of the checkpoint's arrays for one layer's projection, or
        (None, None, None) when unquantized."""
        mode = self.base.mode(layer, projection)
        if mode is None:
            return None, None, None
        if mode in FORMATS:
            return (mode,) + FORMATS[mode]
        if self._config is None:
            config = json.loads((self.path / 'config.json').read_text())
            self._config = (config.get('quantization') or config.get('quantization_config')
                            or (config.get('text_config') or {}).get('quantization') or {})
        name = self.base.name(layer, projection).rsplit('.', 1)[0]
        for key in (name, name.removeprefix('model.'), name.removeprefix('language_model.')):
            if isinstance(self._config.get(key), dict):
                return 'affine', int(self._config[key]['group_size']), int(self._config[key]['bits'])
        return 'affine', int(self._config['group_size']), int(self._config['bits'])

    def experts(self, layer, projection, experts):
        from . import parallel
        parallel.require(layer, 'requant read')
        experts = np.asarray(experts, np.int64)
        if self.cache is None:
            return self._computed(layer, projection, experts)
        return self.cache.get(layer, projection, experts, self.count(layer, projection),
                              lambda ids: self._computed(layer, projection, ids))

    def _ahead(self, layer, projection, rows):
        if rows.size:
            names = [self.base.name(layer, projection, k) for k in q2_experts.NAMES
                     if self.base.has(layer, projection, k)]
            self.base.ahead([(name, rows) for name in names])

    def close(self):
        self.base.close()
        if self.cache is not None:
            self.cache.close()

    def _computed(self, layer, projection, experts):
        mode, group_size, bits = self._quantization(layer, projection)
        parts = []
        for s in range(0, experts.size, self.chunk):
            weight, scales, biases = self.base.experts(layer, projection, experts[s:s + self.chunk])
            self._ahead(layer, projection, experts[s + self.chunk:s + 2 * self.chunk])
            if mode == 'mxfp4' and self.bits == 'mxfp4':
                part = [weight, scales, None]
            else:
                w = weight if mode is None else dequantize(weight, scales, biases, group_size=group_size,
                                                           bits=bits, mode=mode)
                part = quantize(w, self.bits, self.group_size)
            mx.eval(*[t for t in part if t is not None])
            parts.append(part)
        return [mx.concatenate([p[i] for p in parts]) if parts[0][i] is not None else None for i in range(3)]


class Flat:
    """A checkpoint's routed experts all at one uniform quantization (`bits` or ggml format), each
    computed as it is read on demand from the SSD (`quanteeze.demand`)."""

    def __init__(self, path, bits, group_size=GROUP_SIZE):
        self.base = q2_experts.Checkpoint(path)
        self.bits = _norm_bits(bits)
        self.group_size = 256 if self.bits in GGML_MODES else (32 if self.bits == 'mxfp4' else int(group_size))
        self.path = self.base.path
        self.origin = source(self.path, self.bits, self.group_size)
        self._config, self._lock = None, threading.Lock()

    _quantization = Reader._quantization

    def buffers(self, layer, projection):
        return self.base.buffers(layer, projection)

    def expert(self, layer, projection, expert, out=None, cache=True):
        """One expert's `[weight, scales, biases]` `[1, ...]` at `self.bits`, evaluated here."""
        from . import parallel
        parallel.require(layer, 'requant read')
        buf = self.base.buffers(layer, projection) if out is None else out
        arrays = self.base.expert(layer, projection, expert, buf, cache=cache)
        mode, group_size, bits = self._quantization(layer, projection)
        with self._lock:
            if mode == 'mxfp4' and self.bits == 'mxfp4':
                part = [arrays[0], arrays[1], None]
            else:
                w = arrays[0] if mode is None else dequantize(*arrays, group_size=group_size, bits=bits, mode=mode)
                part = quantize(w, self.bits, self.group_size)
            mx.eval(*[t for t in part if t is not None])
        return part


def lower_model(layers, path, bits, group_size=None, on_layer=None):
    """Lower every MoE layer's `SwitchGLU` in `layers` uniformly to `bits` (integer affine least-squares,
    `'mxfp4'`, or ggml `'q3_k'`, `'q2_k'`, `'iq2_xs'`), streaming from `path` in chunks of 32 experts."""
    from . import demand, fitting, ggml_quants, parallel
    bits_spec = _norm_bits(bits)
    gs_default = fitting.group_size(bits_spec) if group_size is None else int(group_size)
    reader = Reader(path, bits_spec, gs_default)
    moe_layers = [
        (i, arch.switch_of(layer)) for i, layer in enumerate(layers)
        if arch.switch_of(layer) is not None and parallel.own_layer(i)
    ]
    try:
        for idx, (layer_idx, switch) in enumerate(moe_layers):
            n_experts = reader.count(layer_idx, 'gate_proj')
            all_ids = np.arange(n_experts, dtype=np.int64)
            for p in PROJECTIONS:
                w, s, b = reader.experts(layer_idx, p, all_ids)
                if w.ndim == 4:
                    eff_bits = ggml_quants.FORMAT_BITS[bits_spec]
                    eff_gs = 256
                    eff_mode = bits_spec
                else:
                    eff_bits = (4 if bits_spec == 'mxfp4'
                                else ggml_quants.FORMAT_BITS[bits_spec] if bits_spec in GGML_MODES
                                else int(bits_spec))
                    eff_mode = 'mxfp4' if (bits_spec == 'mxfp4' and b is None) else 'affine'
                    in_dims = w.shape[2] * 32 // eff_bits
                    eff_gs = in_dims // s.shape[-1]
                proj = demand.Projection(
                    demand._with_weight_room(w), demand._to_device(s), demand._to_device(b),
                    eff_gs, eff_bits, mode=eff_mode,
                )
                del w, s, b
                mx.eval(proj.parameters())
                setattr(switch, p, proj)
            mx.clear_cache()
            if on_layer is not None:
                on_layer(idx + 1, len(moe_layers))
    finally:
        reader.close()


_NPY_TO_SAFETENSORS = {
    np.dtype(np.uint8): 'U8',
    np.dtype(np.uint16): 'BF16',
    np.dtype(np.uint32): 'U32',
    np.dtype(np.float16): 'F16',
    np.dtype(np.float32): 'F32',
}


class CachedCheckpoint(q2_experts.Checkpoint):
    """A materialized `QuantCache` directory exposed as a `q2_experts.Checkpoint` so `Demand` and
    `HostCopy` stream and hold lowered `[weight, scales, biases]` directly."""

    def __init__(self, reader, moe_indices):
        self.base_path = reader.path
        self.path = reader.cache.dir.resolve()
        self.bits = reader.bits
        self.group_size = reader.group_size
        self.origin = reader.origin
        self.template = reader.base.template
        self.tensors = {}
        self._pool, self._ahead = None, {}
        for layer_idx in moe_indices:
            for proj in PROJECTIONS:
                for part in QuantCache.NAMES:
                    npy_path = reader.cache._path(layer_idx, proj, part)
                    if not npy_path.exists():
                        continue
                    offset, shape, dtype = reader.cache._header(npy_path)
                    st_dtype = _NPY_TO_SAFETENSORS[np.dtype(dtype)]
                    self.tensors[self.name(layer_idx, proj, part)] = (
                        npy_path, st_dtype, tuple(shape), offset,
                    )

    def mode(self, layer, projection):
        w_shape = self.tensors[self.name(layer, projection, 'weight')][2]
        if len(w_shape) == 4:
            return self.bits
        if not self.has(layer, projection, 'biases'):
            return 'mxfp4' if self.bits == 'mxfp4' else super().mode(layer, projection)
        return 'affine'

    def quantization(self, layer, projection, input_dims):
        from . import ggml_quants
        w_shape = self.tensors[self.name(layer, projection, 'weight')][2]
        if len(w_shape) == 4:
            return 256, ggml_quants.FORMAT_BITS[self.bits]
        if self.bits == 'mxfp4':
            eff_bits = 4
        elif self.bits in GGML_MODES:
            eff_bits = ggml_quants.FORMAT_BITS[self.bits]
        else:
            eff_bits = int(self.bits)
        scales_shape = self.tensors[self.name(layer, projection, 'scales')][2]
        return input_dims // scales_shape[-1], eff_bits


def cached_checkpoint(layers, path, bits, group_size=None, on_layer=None):
    """Materialize uniform `bits` expert arrays for `layers` in `QuantCache` and return a
    `CachedCheckpoint` (or `None` when `QUANTEEZE_QUANT_CACHE=off`)."""
    from . import fitting, parallel
    bits_spec = _norm_bits(bits)
    gs_default = fitting.group_size(bits_spec) if group_size is None else int(group_size)
    reader = Reader(path, bits_spec, gs_default)
    if reader.cache is None:
        reader.close()
        return None
    try:
        moe_indices = [
            i for i, layer in enumerate(layers)
            if arch.switch_of(layer) is not None and parallel.own_layer(i)
        ]
        for idx, layer_idx in enumerate(moe_indices):
            n_experts = reader.count(layer_idx, 'gate_proj')
            for p in PROJECTIONS:
                reader.cache.ensure(
                    layer_idx, p, n_experts,
                    lambda ids, l=layer_idx, proj=p: reader._computed(l, proj, ids),
                    chunk=reader.chunk,
                    ahead=lambda ids, l=layer_idx, proj=p: reader._ahead(l, proj, ids),
                )
            mx.clear_cache()
            if on_layer is not None:
                on_layer(idx + 1, len(moe_indices))
        return CachedCheckpoint(reader, moe_indices)
    finally:
        reader.close()
