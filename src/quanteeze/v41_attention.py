# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""DeepSeek V4.1 short-pass full attention built in C++ (`QUANTEEZE_V41_ATTENTION_NATIVE`).

Replaces Python graph construction in `DeepseekV41Attention._attention` for short passes (decode,
distribution-mode verify, and chunks up to `MOST` tokens) on Metal with two C++20 nanobind calls
(`front` and `back` in `vendor/v41_attention/v41_attention.cpp`), while compressor and indexer run
between them. Compares bit for bit against `front_python` and `back_python` on the first `CHECKS`
calls of each shape (or every call when set to `compare`), falling back to Python on any mismatch.
"""
import sys

import mlx.core as mx
import mlx.nn as nn

from . import cuda_kernels, kernel_checks, native_ext, per_load, variables

CHECKS = 3
MOST = 128
_STATE = dict(ext=None, off=False, said=False, checked={}, served=0, compared=0, declined=0)
_SPECS = {}


def _mode():
    return (variables.get('QUANTEEZE_V41_ATTENTION_NATIVE', variables.at_load()) or 'auto').strip().lower()


def _ext():
    mode = _mode()
    if mode == 'off':
        return None
    if _STATE['ext'] is None:
        use = mode in ('on', 'compare') or (mode == 'auto' and (mx.metal.is_available() or cuda_kernels.available()))
        ext = (
            native_ext.extension('v41_attention', with_mlx=True, with_stm=False, quiet=(mode == 'auto'))
            if use else None
        )
        _STATE['ext'] = ext or False
    return None if _STATE['off'] else (_STATE['ext'] or None)


def _lin(ext, module):
    """Wrap Linear/QuantizedLinear/MultiLinear/QuantizedMultiLinear for C++, or return None."""
    from mlx_vlm.models.mla import MultiLinear, QuantizedMultiLinear
    if isinstance(module, nn.QuantizedLinear):
        kind = 1
    elif isinstance(module, QuantizedMultiLinear):
        kind = 3
    elif isinstance(module, MultiLinear):
        kind = 2
    elif isinstance(module, nn.Linear):
        kind = 0
    else:
        return None
    has_bias = 'bias' in module if hasattr(module, '__contains__') else getattr(module, 'bias', None) is not None
    if kind in (2, 3) and has_bias:
        return None
    bias = (module['bias'] if isinstance(module, dict) else module.bias) if has_bias else None
    weight = module['weight'] if isinstance(module, dict) else module.weight
    if kind in (1, 3):
        scales = module['scales'] if isinstance(module, dict) else module.scales
        biases = module.get('biases') if hasattr(module, 'get') else getattr(module, 'biases', None)
        return ext.lin(
            kind, weight, scales, biases, bias,
            int(module.group_size), int(module.bits), str(module.mode),
        )
    return ext.lin(kind, weight, None, None, bias, 0, 0, '')


def _norm(module):
    if type(module) is nn.RMSNorm:
        w = module['weight'] if isinstance(module, dict) else module.weight
        return (w, float(module.eps))
    return None


def _spec(ext, attn):
    """Build or reuse cached `ext.layer` descriptor for `attn`."""
    cached = getattr(attn, '_quanteeze_v41_attn_spec', None)
    if cached is not None and cached[0] is ext:
        return cached[1]
    got = _SPECS.get(id(attn))
    if got is not None and got[0] is attn and got[1] is ext:
        return got[2]
    built = _make_spec(ext, attn)
    try:
        object.__setattr__(attn, '_quanteeze_v41_attn_spec', (ext, built))
    except Exception:
        _SPECS[id(attn)] = (attn, ext, built)
    return built


def _make_spec(ext, attn):
    from mlx_vlm.models.deepseek_v4.language import DeepseekV4RoPE
    from mlx_vlm.models.deepseek_v41 import fakequant
    lins = [_lin(ext, getattr(attn, n)) for n in ('wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b')]
    norms = [_norm(attn.q_norm), _norm(attn.kv_norm)]
    rope = attn.rope
    fp8 = glue = None
    if not mx.metal.is_available():
        from . import cuda_v41
        fp8, glue = cuda_v41.cpp_kernels(ext)
    if (None in lins or None in norms or type(rope) is not DeepseekV4RoPE or rope.freq_scale != 1
            or fakequant.DISABLE or fakequant._fp8_roundtrip_kernel is None
            or (fp8 is None and not mx.metal.is_available())
            or attn.head_dim % 32 or (attn.o_groups * attn.config.o_lora_rank) % 32
            or attn.config.hidden_size % 32 or attn.config.q_lora_rank % 32):
        return None
    layer = ext.layer(
        lins[0], norms[0][0], norms[0][1], lins[1], lins[2], norms[1][0], norms[1][1], lins[3], lins[4],
        rope._get_freqs(attn.head_dim, False), rope._get_freqs(attn.head_dim, True), attn.attn_sink,
        float(attn.scale), int(attn.n_heads), int(attn.head_dim), int(attn.o_groups), int(attn.window_size),
    )
    return layer if mx.metal.is_available() else ext.layer_on_cuda(layer, fp8, glue)


def front_python(attn, x, start_pos, buffer, history, writes):
    """Reference Python implementation of `front` (queries + window keys/ring update)."""
    from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
    batch, seqlen = x.shape[0], x.shape[1]
    qr = attn.q_norm(attn.wq_a(fake_quant_fp8_ue8m0(x)))
    q = attn.wq_b(fake_quant_fp8_ue8m0(qr)).reshape(batch, seqlen, attn.n_heads, attn.head_dim)
    q = attn.rope(q.transpose(0, 2, 1, 3), start_pos)
    win = attn.window_size
    kv = attn.kv_norm(attn.wkv(fake_quant_fp8_ue8m0(x)))
    kv = attn.rope(kv[:, None], start_pos).reshape(batch, seqlen, attn.head_dim)
    kv = fake_quant_fp8_ue8m0(kv)
    if buffer is None:
        buffer = mx.zeros((batch, win, attn.head_dim), dtype=kv.dtype)
    elif buffer.shape[0] < batch:
        buffer = mx.pad(buffer, [(0, batch - buffer.shape[0]), (0, 0), (0, 0)])
    else:
        buffer = buffer[:]
    part = kv if history is None else mx.concatenate([mx.take(buffer[:batch], history, axis=1), kv], axis=1)
    tail = kv[:, -min(seqlen, win):]
    for a, b, c, d in writes:
        buffer[:batch, a:b] = tail[:, c:d]
    return q, qr, part, buffer


def back_python(attn, q, part, valid, pool, idxs, start_pos):
    """Reference Python implementation of `back` (sparse/dense attention + output projections)."""
    from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
    batch, seqlen = q.shape[0], q.shape[2]
    out = None
    if pool is not None and pool.shape[1]:
        out = attn._sparse_attention(q, part, pool, idxs, valid)
    if out is None:
        kv = part[:, None].astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q, kv, kv, scale=attn.scale, mask=valid, sinks=attn.attn_sink.astype(q.dtype),
        )
    out = attn.rope(out, start_pos, inverse=True)
    out = out.reshape(batch, attn.o_groups, -1, seqlen, attn.head_dim)
    out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
    out = attn.wo_a(out)
    out = out.transpose(0, 2, 1, 3).flatten(-2)
    return attn.wo_b(fake_quant_fp8_ue8m0(out))


def _same(got, want):
    return kernel_checks.same_bits(tuple(got), tuple(want))


def _parted(what):
    _STATE['off'] = True
    if not _STATE['said']:
        _STATE['said'] = True
        print(
            f'quanteeze: QUANTEEZE_V41_ATTENTION_NATIVE: C++ attention {what} parted from Python; '
            'using Python from now on',
            file=sys.stderr,
            flush=True,
        )


def attend(attn, cache, x, window_indices):
    """Run DeepSeek V4.1 attention via C++ `_v41_attention`, or return None to use Python."""
    ext = _ext()
    if ext is None or x.ndim != 3 or not 0 < x.shape[1] <= MOST or mx.default_device() != mx.gpu:
        return None
    layer = _spec(ext, attn)
    if layer is None:
        _STATE['declined'] += 1
        return None
    start_pos = cache.offset
    batch, seqlen = x.shape[0], x.shape[1]
    history, writes, valid = window_indices(start_pos, seqlen, attn.window_size)
    buffer = cache.window[attn.layer_idx]
    key = (batch, seqlen, history is None, bool(attn.compress_ratio))
    check = _mode() == 'compare' or _STATE['checked'].get(key, 0) < CHECKS
    q, qr, part, ring = ext.front(layer, x, int(start_pos), buffer, history, writes)
    if check:
        want = front_python(attn, x, start_pos, buffer, history, writes)
        if not _same((q, qr, part, ring), want):
            _parted('front')
            q, qr, part, ring = want
    cache.window[attn.layer_idx] = ring
    pool, idxs = attn._compress_part(x, qr, start_pos, cache) if attn.compress_ratio else (None, None)
    if _STATE['off']:
        return back_python(attn, q, part, valid, pool, idxs, start_pos)
    y = ext.back(layer, q, part, valid, pool, idxs, int(start_pos))
    if check:
        want = back_python(attn, q, part, valid, pool, idxs, start_pos)
        _STATE['compared'] += 1
        _STATE['checked'][key] = _STATE['checked'].get(key, 0) + 1
        if not _same((y,), (want,)):
            _parted('back')
            return want
    _STATE['served'] += 1
    return y


def stats():
    return {k: _STATE[k] for k in ('served', 'compared', 'declined', 'off')}


@per_load.on_let_go
def let_go():
    """Clear per-model module specs and reset extension handle on unload."""
    _SPECS.clear()
    _STATE['ext'] = None
