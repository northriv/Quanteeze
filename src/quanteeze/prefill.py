# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A prompt through the model a layer at a time (GLM-5.3 on demand prefilled at 3 GB/s, every
2,048-token chunk reading again what each layer's bank lacks: 85 GB a chunk, some 12 times for a
24k-token prompt).

mlx-vlm prefills a prompt in chunks, each through every layer before the next.  Installed on a
model's decoder layers (install), a layer given more than `chunk` tokens goes through them
itself, chunk by chunk, on its own cache, before handing them all to the next layer: the
server then gives the model the whole prompt at once (serve: the outer step large), and each
layer runs once for the prompt.

Each chunk meets what it meets in chunk-major order -- the same tokens, the layer's cache after
the chunks before it, and the same masks and positions -- so the output is the same, bit for
bit.  The arguments the model hands a layer for the whole prompt are cut to the chunk (cut): an
array whose second axis is the prompt (hidden states, token ids, top-k positions, a padding mask)
on that axis; an attention mask [..., prompt, keys] made with the cache's offset its chunk's rows
and the keys up to the chunk's end; positions [..., prompt] on their last axis; anything else as
it is.  What a layer returns is written, chunk by chunk, into arrays of the whole prompt's length.
A batch of more than one row, a model whose layers share a single cache or keep a sliding window,
or an argument whose prompt axis cannot be told, goes through whole, as the model would take it
(Unchunked).

enabled() is asked each call: true while the model is served on demand from the SSD.
"""
from contextlib import contextmanager
import threading
import time

import mlx.core as mx
import mlx.nn as nn

_STATE = threading.local()
JOIN_TOKENS = 2048
JOIN_CHECKS = 2
JOINS = {}
SPLITS = {}
_BY_CHUNK = {}
TIMES = {'pre': 0.0, 'post': 0.0, 'layers': 0, 'windows': 0, 'joined': 0}


def times(reset=False):
    """Return a snapshot of `TIMES` (`pre`, `post`, `layers`, `windows`, `joined`); reset counters when `reset=True`."""
    out = dict(TIMES)
    if reset:
        for k in TIMES:
            TIMES[k] = 0.0 if isinstance(TIMES[k], float) else 0
    return out


class Unchunked(Exception):
    """An argument whose prompt axis cannot be told: the layer takes the prompt whole."""


class Settings:
    chunk = 2048
    enabled = staticmethod(lambda: True)


SETTINGS = Settings()


def in_layer():
    """Whether the current thread is executing inside a chunked single-layer prefill pass."""
    return bool(getattr(_STATE, 'in_layer', False))


def window_parts(n):
    """While a joined MoE window of `n` tokens runs: the tuple of its constituent chunk lengths; else `None`."""
    parts = getattr(_STATE, 'window_parts', None)
    return parts if parts is not None and sum(parts) == n else None


def split(cls, pre, post):
    """Register `(pre, post)` on decoder layer `cls` so layer-at-a-time prefill can run `pre` per
    chunk and `post` across joined `QUANTEEZE_PROMPT_MOE_TOKENS` windows."""
    SPLITS[cls] = (pre, post)
    return cls


def _split_of(cls):
    for base in cls.__mro__:
        if base in SPLITS:
            return SPLITS[base]
    return None


def _caches(value):
    """The KV caches in a layer's cache argument (a CacheList's parts)."""
    if value is None or isinstance(value, mx.array):
        return []
    parts = getattr(value, 'caches', None)
    if isinstance(parts, (tuple, list)):
        return [c for part in parts for c in _caches(part)]
    return [value]


def _offset(args, kwargs):
    """The offset of the layer's cache before the prompt, or None.  A cache that keeps a window
    (max_size) masks each chunk against what it still holds, not a cut of the whole prompt's
    mask: Unchunked."""
    found = [c for v in [kwargs.get('cache'), *args] for c in _caches(v) if hasattr(c, 'offset')]
    if any(getattr(c, 'max_size', None) is not None for c in found):
        raise Unchunked('a cache that keeps a window')
    offsets = {c.offset for c in found if isinstance(getattr(c, 'offset', None), int)}
    return offsets.pop() if len(offsets) == 1 else None


def cut(value, n, a, b, offset=None):
    """`value`, an argument given a layer for `n` tokens, cut to tokens a..b."""
    if isinstance(value, (tuple, list)) and any(isinstance(v, mx.array) for v in value):
        return type(value)(cut(v, n, a, b, offset) for v in value)
    if not isinstance(value, mx.array) or value.ndim == 0:
        return value
    shape = value.shape
    if offset is not None and value.ndim >= 2 and shape[-2] == n and shape[-1] == offset + n:
        return value[..., a:b, :offset + b]                       # an attention mask: its rows and keys
    if value.ndim >= 2 and shape[1] == n:
        return value[:, a:b]
    axes = [k for k, d in enumerate(shape) if d == n]
    if not axes:
        return value
    if axes == [value.ndim - 1]:
        return value[..., a:b]                                    # positions [3, 1, n], say
    raise Unchunked(f'an argument of shape {shape} for {n} tokens')


def _whole(out, n, m):
    """Arrays for the whole prompt's n tokens, shaped as a chunk of m tokens' output `out`."""
    if isinstance(out, mx.array) and out.ndim >= 2 and out.shape[1] == m:
        return mx.zeros((out.shape[0], n, *out.shape[2:]), out.dtype)
    if isinstance(out, (tuple, list)):
        return type(out)(_whole(o, n, m) for o in out)
    return None


def _put(whole, out, a, b):
    """A chunk's output written into the whole prompt's arrays at tokens a..b (in place: the
    chunks' outputs are not kept besides the whole); what does not grow with the prompt is the
    last chunk's."""
    if isinstance(whole, mx.array):
        whole[:, a:b] = out
        return whole
    if isinstance(out, (tuple, list)) and isinstance(whole, (tuple, list)):
        return type(out)(_put(w, o, a, b) for w, o in zip(whole, out))
    return out


def _tiling(layer):
    mlp = (getattr(layer, 'mlp', None) or getattr(layer, 'ffn', None)
           or getattr(layer, 'block_sparse_moe', None) or getattr(layer, 'feed_forward', None))
    top_k = (getattr(mlp, 'top_k', None) or getattr(getattr(mlp, 'gate', None), 'top_k', None)
             or getattr(mlp, 'num_experts_per_tok', None) or 1)
    return max(1, -(-64 // max(1, int(top_k))))


def _windows(parts, layer):
    from . import arch, cuda_kernels, variables
    sw = arch.switch_of(layer)
    if sw is None or len(parts) <= 1:
        return [[p] for p in parts]
    raw = (variables.get('QUANTEEZE_PROMPT_MOE_TOKENS') or 'auto').strip().lower()
    if raw in ('off', '0'):
        return [[p] for p in parts]
    if raw == 'auto':
        target = JOIN_TOKENS if (cuda_kernels.available() or getattr(sw, 'demand', None) is not None) else 0
    else:
        target = int(raw)
    if target <= SETTINGS.chunk:
        return [[p] for p in parts]
    min_len = _tiling(layer)
    out = []
    for a, b in parts:
        if b - a < min_len:
            out.append([(a, b)])
        elif out and out[-1][-1][1] - out[-1][-1][0] >= min_len and b - out[-1][0][0] <= target:
            out[-1].append((a, b))
        else:
            out.append([(a, b)])
    return out


def _join(values, lengths):
    first = values[0]
    if isinstance(first, mx.array):
        if first.ndim >= 2 and all(isinstance(v, mx.array) and v.shape[1] == m for v, m in zip(values, lengths)):
            return mx.concatenate(values, axis=1)
        return values[-1]
    if isinstance(first, (tuple, list)):
        return type(first)(_join([v[i] for v in values], lengths) for i in range(len(first)))
    return values[-1]


def _same(a, b):
    if isinstance(a, mx.array) and isinstance(b, mx.array):
        return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b):
        return all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _by_chunk_cls(base_cls):
    if base_cls not in _BY_CHUNK:
        def __call__(self, x, *args, **kwargs):
            n = x.shape[-2] if isinstance(x, mx.array) and x.ndim >= 2 else 0
            parts = window_parts(n)
            if parts is None:
                return base_cls.__call__(self, x, *args, **kwargs)
            at, outs = 0, []
            for m in parts:
                outs.append(base_cls.__call__(self, x[..., at:at + m, :], *args, **kwargs))
                at += m
            return mx.concatenate(outs, axis=-2)

        _BY_CHUNK[base_cls] = type(f'_ByChunk{base_cls.__name__}', (base_cls,), {'__call__': __call__})
    return _BY_CHUNK[base_cls]


def _dense(layer):
    from . import arch
    mlp = (getattr(layer, 'mlp', None) or getattr(layer, 'ffn', None)
           or getattr(layer, 'block_sparse_moe', None) or getattr(layer, 'feed_forward', None))
    if mlp is None:
        return []
    return [m for m in mlp.modules() if isinstance(m, (nn.Linear, nn.QuantizedLinear)) and not arch.is_experts(m)]


@contextmanager
def _by_chunk(layer):
    saved = [(m, type(m)) for m in _dense(layer)]
    for m, cls in saved:
        m.__class__ = _by_chunk_cls(cls)
    try:
        yield
    finally:
        for m, cls in saved:
            m.__class__ = cls


def _joined_post(layer, post, staged_window):
    lengths = tuple(b - a for (a, b), _ in staged_window)
    key = (type(layer), lengths)
    state = JOINS.get(key)

    def apart():
        outs = [post(layer, *vals) for _, vals in staged_window]
        return _join(outs, lengths)

    def together():
        joined_args = [_join([vals[i] for _, vals in staged_window], lengths) for i in range(len(staged_window[0][1]))]
        _STATE.window_parts = lengths
        try:
            with _by_chunk(layer):
                return post(layer, *joined_args)
        finally:
            _STATE.window_parts = None

    if state is not None and not state[0]:
        return apart()
    if state is not None and state[0] and state[1] >= JOIN_CHECKS:
        TIMES['joined'] += 1
        return together()
    want = apart()
    got = together()
    mx.eval(want, got)
    ok = _same(want, got)
    seen = (state[1] if state is not None else 0) + 1
    JOINS[key] = (ok, seen if ok else JOIN_CHECKS)
    return want


def _in_two(layer, x, n, parts, cuts, windows, sp):
    pre, post = sp
    cut_map = dict(zip(parts, cuts))
    whole = None
    TIMES['layers'] += 1
    for window in windows:
        TIMES['windows'] += 1
        t0 = time.perf_counter()
        staged = []
        for a, b in window:
            cargs, ckwargs = cut_map[(a, b)]
            got = pre(layer, x[:, a:b], *cargs, **ckwargs)
            vals = got if isinstance(got, tuple) else (got,)
            arrays = [v for v in vals if isinstance(v, mx.array)]
            if arrays:
                mx.eval(*arrays)
            staged.append(((a, b), vals))
        TIMES['pre'] += time.perf_counter() - t0
        t1 = time.perf_counter()
        w_start, w_end = window[0][0], window[-1][1]
        if len(window) == 1:
            out = post(layer, *staged[0][1])
        else:
            out = _joined_post(layer, post, staged)
        if whole is None:
            whole = _whole(out, n, w_end - w_start)
        whole = _put(whole, out, w_start, w_end)
        mx.eval(whole)
        TIMES['post'] += time.perf_counter() - t1
        del staged, out
    return whole


def _chunked(cls):
    def __call__(self, x, *args, **kwargs):
        n = x.shape[1] if isinstance(x, mx.array) and x.ndim >= 2 else 0
        step = SETTINGS.chunk
        parts = [(a, min(a + step, n)) for a in range(0, n, step)] if n else []
        if (len(parts) <= 1 or x.shape[0] != 1 or getattr(_STATE, 'in_layer', False)
                or not SETTINGS.enabled()):
            return cls.__call__(self, x, *args, **kwargs)
        try:
            offset = _offset(args, kwargs)
            cuts = [([cut(v, n, a, b, offset) for v in args], {k: cut(v, n, a, b, offset) for k, v in kwargs.items()})
                    for a, b in parts]
        except Unchunked:
            return cls.__call__(self, x, *args, **kwargs)
        sp = _split_of(cls)
        windows = _windows(parts, self) if sp is not None else None
        _STATE.in_layer = True
        whole = None
        ok = False
        try:
            if windows is not None and any(len(w) > 1 for w in windows):
                whole = _in_two(self, x, n, parts, cuts, windows, sp)
            else:
                for (a, b), (cargs, ckwargs) in zip(parts, cuts):
                    out = cls.__call__(self, x[:, a:b], *cargs, **ckwargs)
                    if whole is None:
                        whole = _whole(out, n, b - a)
                    whole = _put(whole, out, a, b)
                    mx.eval(whole)
                    del out
            ok = True
        finally:
            _STATE.in_layer = False
            from . import arch
            sw = arch.switch_of(self)
            dem = getattr(sw, 'demand', None) if sw is not None else None
            if dem is not None:
                dem.end_layer_prefill(sw, ok=ok)
        return whole

    return type(f'Chunked{cls.__name__}', (cls,), {'__call__': __call__, '_quanteeze_chunked': True})


def supported(lm):
    """Whether a language model's caches let its layers cut the prompt: one independent cache per
    decoder layer, none keeping a sliding window."""
    try:
        made = lm.make_cache()
        caches = [c for value in made for c in _caches(value)]
    except Exception:
        return False
    layers = getattr(getattr(lm, 'model', None), 'layers', None) or getattr(lm, 'layers', None)
    if layers is None or len(made) != len(layers) or len({id(c) for c in made}) != len(made):
        return False
    return not any(getattr(c, 'max_size', None) is not None for c in caches)


_CLASSES = {}


def install(layers, chunk=None, enabled=None):
    """Each of `layers` (a model's decoder layers) takes a prompt longer than `chunk` tokens a
    chunk at a time; enabled() is asked each call."""
    if chunk is not None:
        SETTINGS.chunk = int(chunk)
    if enabled is not None:
        SETTINGS.enabled = enabled
    for layer in layers:
        cls = type(layer)
        if getattr(cls, '_quanteeze_chunked', False):
            continue
        if cls not in _CLASSES:
            _CLASSES[cls] = _chunked(cls)
        layer.__class__ = _CLASSES[cls]


def remove(layers):
    for layer in layers:
        if getattr(type(layer), '_quanteeze_chunked', False):
            layer.__class__ = type(layer).__base__


def release():
    """The model install() was given for let go: enabled() says no from now."""
    SETTINGS.enabled = lambda: False


def widen(batch_cls, tokens):
    """mlx-vlm's prompt batches (`batch_cls`, its PromptProcessingBatch) of one row step `tokens` at
    a time while enabled(), their own step the layers' chunk: the model is given the prompt, or
    `tokens` of it, at once."""
    step = batch_cls.prompt_step
    if batch_cls.__dict__.get('_quanteeze_widened'):
        return

    def prompt_step(self):
        own = self.prefill_step_size
        # A prompt the prompt cache (APC) took part of comes right-padded, [0] for its one row: no
        # padding, stepped as any other (else every warm prompt went through the model 512 tokens a
        # step, each layer reading the experts its tokens meet, past the GPU's room on demand).
        if (not own or tokens <= own or len(self.uids) != 1
                or any(getattr(self, '_right_pad_per_row', None) or ()) or not SETTINGS.enabled()):
            return step(self)
        SETTINGS.chunk, self.prefill_step_size = own, tokens
        try:
            return step(self)
        finally:
            self.prefill_step_size = own

    batch_cls.prompt_step, batch_cls._quanteeze_widened = prompt_step, True
