# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Flash-Next's PLE layer (its layer 2's n-gram embedding mixed into the streams) at decode in two GPU kernels,
bit for bit what it replaces.

A step of one request without a mask whose window is cached (no lengths, no speculation's history), 4-bit
projections, a conv without bias: these kernels; else mlx-vlm's.  None is taken on trust: the first CHECKS
times a shape is met, the streams and the window are compared bit for bit with mlx-vlm's, and the kernels
serve the shape only if both are the same.  With quanteeze.cuda_kernels.FUSE (QUANTEEZE_CUDA_FUSE).
"""
import math
import sys
from functools import lru_cache

import mlx.core as mx

from . import cuda_kernels, cuda_qmv

CHECKS = 3
STATS = dict(served=0, checked=0, mismatched=0)
CHOICE = {}                     # (shape, dtype): whether it matched mlx-vlm's
CHECKED = {}
SAID = []

_PROJECT = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int r = blockIdx.x * 8 + warp.meta_group_rank();
    if (r >= NK + NV) {
      return;
    }
    const bool is_k = r < NK;
    const int row = is_k ? r : r - NK;
    const float sum = is_k ? qz_qmv4_row<K_IN, GS>(x, wk, sk, bk, row) : qz_qmv4_row<K_IN, GS>(x, wv, sv, bv, row);
    if (warp.thread_rank() == 0) {
      (is_k ? k : v)[row] = __float2bfloat16_rn(sum);
    }
"""

_MIX = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int i = blockIdx.x;
    const int tid = block.thread_rank();
    const int lane = warp.thread_rank();
    const int wid = warp.meta_group_rank();
    __shared__ __align__(16) T a[D];
    __shared__ __align__(16) T b[D];
    __shared__ __align__(16) T g[D];
    __shared__ float part[32];
    __shared__ T tpart[32];
    __shared__ float rs_shared;
    __shared__ T sg_shared;
    auto norm = [&](const T* x, const T* w, T* y) {
      if (tid < TG) {
        float acc = 0.0f;
        const T* in = x + tid * 4;
        for (int r = 0; r < D / (TG * 4); ++r) {
          for (int j = 0; j < 4; ++j) {
            float v = static_cast<float>(in[j]);
            acc = __fadd_rn(acc, __fmul_rn(v, v));
          }
          in += TG * 4;
        }
        acc = cg::reduce(warp, acc, QzSum{});
        if (lane == 0) {
          part[wid] = acc;
        }
      }
      block.sync();
      if (wid == 0) {
        float acc = lane < TG / 32 ? part[lane] : 0.0f;
        acc = cg::reduce(warp, acc, QzSum{});
        if (lane == 0) {
          rs_shared = rsqrtf(__fmaf_rn(acc, INV, EPS));
        }
      }
      block.sync();
      const float rs = rs_shared;
      for (int c = tid; c < D; c += NT) {
        float p = __fmul_rn(static_cast<float>(x[c]), rs);
        float q = __fadd_rn(1.0f, static_cast<float>(w[i * D + c]));
        y[c] = static_cast<T>(__fmul_rn(p, q));
      }
      block.sync();
    };
    for (int c = tid; c < D; c += NT) {
      g[c] = keys[i * D + c];
    }
    block.sync();
    norm(g, kw, a);
    norm(hidden + i * D, qw, b);
    if (tid < 64) {
      T acc = static_cast<T>(0.0f);
      for (int r = 0; r < D / 512; ++r) {
        for (int j = 0; j < 8; ++j) {
          const int c = r * 512 + tid * 8 + j;
          acc = acc + QzMul{}(a[c], b[c]);
        }
      }
      acc = cg::reduce(warp, acc, QzSum{});
      if (lane == 0) {
        tpart[wid] = acc;
      }
    }
    block.sync();
    if (wid == 0) {
      T acc = lane < 2 ? tpart[lane] : static_cast<T>(0.0f);
      acc = cg::reduce(warp, acc, QzSum{});
      if (lane == 0) {
        const T gate = Divide{}(acc, __ushort_as_bfloat16(static_cast<unsigned short>(ROOT_BITS)));
        const T mag = Sqrt{}(Maximum{}(Abs{}(gate), __ushort_as_bfloat16(static_cast<unsigned short>(TINY_BITS))));
        sg_shared = Sigmoid{}(Multiply{}(Sign{}(gate), mag));
      }
    }
    block.sync();
    const T sg = sg_shared;
    for (int c = tid; c < D; c += NT) {
      g[c] = Multiply{}(sg, values[c]);
    }
    block.sync();
    norm(g, cw, a);
    for (int c = tid; c < D; c += NT) {
      const int ch = i * D + c;
      float s = 0.0f;
      for (int t = 0; t < TAPS; ++t) {
        const int p = t * DIL;
        const float x = static_cast<float>(p < SL ? state[p * C + ch] : a[c]);
        s = __fmaf_rn(x, static_cast<float>(conv_w[ch * TAPS + t]), s);
      }
      const T conv = static_cast<T>(s);
      const T ple = Add{}(g[c], QzMul{}(conv, Sigmoid{}(conv)));
      out[ch] = Add{}(hidden[ch], ple);
      for (int p = 0; p + 1 < SL; ++p) {
        state_out[p * C + ch] = state[(p + 1) * C + ch];
      }
      state_out[(SL - 1) * C + ch] = a[c];
    }
"""


@lru_cache(maxsize=None)
def _project_kernel():
    return cuda_kernels.kernel('quanteeze_ple_project', ['x', 'wk', 'sk', 'bk', 'wv', 'sv', 'bv'], ['k', 'v'],
                               _PROJECT, cuda_qmv.HEADER)


def _bits(value):
    """The bfloat16 MLX makes of a Python float beside a bfloat16 array, as its bits."""
    return int(mx.array(value, dtype=mx.bfloat16).view(mx.uint16).item())


@lru_cache(maxsize=None)
def _mix_kernel(eps, d):
    inverse = float(mx.array(1.0 / d, dtype=mx.float32).item())
    header = (f'constexpr float EPS = {float(eps)!r}f;\nconstexpr float INV = {inverse!r}f;\n'
              f'constexpr int ROOT_BITS = {_bits(math.sqrt(d))};\nconstexpr int TINY_BITS = {_bits(1e-6)};\n')
    name = f'quanteeze_ple_mix_{eps!r}_{d}'.replace('.', '_').replace('-', 'm')
    return cuda_kernels.kernel(name, ['hidden', 'keys', 'values', 'state', 'kw', 'qw', 'cw', 'conv_w'],
                               ['out', 'state_out'], _MIX, header)


def _quantized(linear, rows, cols):
    return (getattr(linear, 'bits', None) == 4 and getattr(linear, 'mode', 'affine') == 'affine'
            and getattr(linear, 'biases', None) is not None and linear.group_size in (32, 64, 128)
            and linear.weight.dtype == mx.uint32 and tuple(linear.weight.shape) == (rows, cols // 8)
            and linear.scales.dtype == mx.bfloat16 and linear.biases.dtype == mx.bfloat16 and 'bias' not in linear)


def applies(module, hidden, cache, mask):
    """Whether the kernels compute this step's PLE: one token of one request in bfloat16 without a mask, its
    window cached without lengths or speculation's history, Flash-Next's shapes."""
    if not cuda_kernels.FUSE[0] or mask is not None or hidden.dtype != mx.bfloat16 or hidden.ndim != 3:
        return False
    if hidden.shape[:2] != (1, 1) or cache is None or not cuda_kernels.available():
        return False
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm
    hc, d = module.hc_count, module.hidden_size
    c = hc * d
    state, conv = cache[2], module.conv1d
    if (hidden.shape[-1] != c or state is None or state.dtype != hidden.dtype
            or tuple(state.shape) != (1, module.short_conv_state_len, c) or cache.lengths is not None
            or getattr(cache, 'history_capacity', 0)):
        return False
    taps = conv.weight.shape[1]
    if ('bias' in conv or tuple(conv.weight.shape) != (c, taps, 1) or conv.weight.dtype != hidden.dtype
            or module.short_conv_state_len != (taps - 1) * module.conv_dilation or d % 512 or d > 4096):
        return False
    norms = (module.norm_key, module.norm_query, module.norm_conv)
    if not all(type(n) is Qwen4ExpRMSNorm and n.group_size == d and n.weight.shape == (c,)
               and n.weight.dtype == hidden.dtype and n.eps == module.norm_key.eps for n in norms):
        return False
    embed = module.key_proj.weight.shape[-1] * 8
    return (_quantized(module.key_proj, c, embed) and _quantized(module.value_proj, d, embed)
            and module.key_proj.group_size == module.value_proj.group_size and d % (_threads(d) * 4) == 0)


def _threads(d):
    """row_reduce_simple's threads for a float32 row of d (a quarter of its reads' warps, 1 to 32)."""
    reads = -(-d // 4)
    return max(min(-(-reads // 32) // 4, 32), 1) * 32


def mixed(module, hidden, embeddings, state):
    """(hidden + mlx-vlm's _ple of it, the window slid) from the step's embedding and the cached window."""
    hc, d = module.hc_count, module.hidden_size
    c = hc * d
    kp, vp = module.key_proj, module.value_proj
    keys, values = _project_kernel()(
        inputs=[embeddings, kp.weight, kp.scales, kp.biases, vp.weight, vp.scales, vp.biases],
        template=[('GS', int(kp.group_size)), ('K_IN', embeddings.shape[-1]), ('NK', c), ('NV', d)],
        grid=(32 * -(-(c + d) // 8), 8, 1), threadgroup=(32, 8, 1),
        output_shapes=[(1, 1, c), (1, 1, d)], output_dtypes=[hidden.dtype] * 2)
    threads = 640
    out, slid = _mix_kernel(float(module.norm_key.eps), d)(
        inputs=[hidden, keys, values, state, module.norm_key.weight, module.norm_query.weight,
                module.norm_conv.weight, module.conv1d.weight],
        template=[('T', hidden.dtype), ('D', d), ('C', c), ('NT', threads), ('TG', _threads(d)),
                  ('TAPS', module.conv1d.weight.shape[1]), ('DIL', module.conv_dilation),
                  ('SL', module.short_conv_state_len)],
        grid=(threads * hc, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[hidden.shape, state.shape], output_dtypes=[hidden.dtype] * 2)
    return out, slid


def added(forward, module, hidden, input_ids, cache, mask):
    """hidden + forward._ple(module, hidden, input_ids, cache, mask): the kernels' where the step's shape matched."""
    if not applies(module, hidden, cache, mask):
        return hidden + forward._ple(module, hidden, input_ids, cache, mask)
    state = cache[2]
    key = (tuple(hidden.shape), str(hidden.dtype), tuple(state.shape))
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if not choice:
            return hidden + forward._ple(module, hidden, input_ids, cache, mask)
        STATS['served'] += 1
        out, cache[2] = mixed(module, hidden, module.ple_embedding(input_ids, cache), state)
        return out
    STATS['checked'] += 1
    embeddings = module.ple_embedding(input_ids, cache)
    try:
        got = mixed(module, hidden, embeddings, state)
    except Exception as error:
        got, differ = None, [f'{type(error).__name__}: {error}']
    module.__dict__['ple_embedding'] = lambda *args: embeddings
    try:
        want = hidden + forward._ple(module, hidden, input_ids, cache, mask)
    finally:
        module.__dict__.pop('ple_embedding', None)
    if got is not None:
        differ = [name for name, x, y in (('streams', got[0], want), ('window', got[1], cache[2]))
                  if not mx.array_equal(x, y).item()]
    if differ and not SAID:
        SAID.append(True)
        print(f'quanteeze: the PLE layer\'s kernels differ from mlx-vlm\'s in {", ".join(differ)} ({key}); '
              'mlx-vlm\'s serves it', file=sys.stderr, flush=True)
    STATS['mismatched'] += bool(differ)
    CHOICE[key], CHECKED[key] = not differ, (seen + 1 if not differ else CHECKS)
    return want


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
