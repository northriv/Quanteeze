# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Prompt-cache (APC) block compression for Qwen3.8 / Flash-Next full-attention layers.

When enabled (`QUANTEEZE_CONTEXT_COMPRESS`), each full-attention `QSAKVCache` keeps its start
(`QUANTEEZE_CONTEXT_SINK` tokens), its end (`QUANTEEZE_CONTEXT_RECENT` tokens), and a share
(`QUANTEEZE_CONTEXT_KEEP`, spread over layers by `QUANTEEZE_CONTEXT_LAYERS`: `uniform` or
`pyramid`) of middle blocks (`recent`, `similar`, `h2o`, or `snapkv`) when stored by
`APCManager.store_exact_cache`. Logical vs physical lengths are tracked via per-row `dropped`
counts across `QSAKVCache` / `BatchQSAKVCache`, and active settings are mixed into
`serve._precision()` (`tag()`) so APC caches stay isolated per configuration.
"""
import numpy as np

from . import variables

METHODS = ('off', 'recent', 'similar', 'h2o', 'snapkv')
COUNTED = ('h2o', 'snapkv')
SNAP_POOL = 3
EXTRA = ('votes', 'ends', 'window', 'reviewed')
LAYERS = ('uniform', 'pyramid')
PYRAMID_SPREAD = 0.5
STATS = {'stored': 0, 'compressed': 0, 'tokens_before': 0, 'tokens_after': 0}
_INSTALLED = [False]
_PATCHED = []
_MISSING = object()


def _patch(owner, name, value):
    _PATCHED.append((owner, name, owner.__dict__.get(name, _MISSING)))
    setattr(owner, name, value)


def method():
    got = (variables.get('QUANTEEZE_CONTEXT_COMPRESS', variables.at_load()) or 'off').strip().lower()
    if got not in METHODS:
        raise ValueError(f'QUANTEEZE_CONTEXT_COMPRESS: one of {", ".join(METHODS)}, not {got!r}')
    return got


def settings():
    """Return `(method, layers, keep, recent, sink, min_tokens)`."""
    snap = variables.at_load()
    layers = (variables.get('QUANTEEZE_CONTEXT_LAYERS', snap) or 'uniform').strip().lower()
    if layers not in LAYERS:
        raise ValueError(f'QUANTEEZE_CONTEXT_LAYERS: one of {", ".join(LAYERS)}, not {layers!r}')
    keep = variables.number('QUANTEEZE_CONTEXT_KEEP', snap)
    if not 0.0 <= keep <= 1.0:
        raise ValueError(f'QUANTEEZE_CONTEXT_KEEP: a share from 0 to 1, not {keep}')
    return (
        method(),
        layers,
        keep,
        variables.integer('QUANTEEZE_CONTEXT_RECENT', snap),
        variables.integer('QUANTEEZE_CONTEXT_SINK', snap),
        variables.integer('QUANTEEZE_CONTEXT_MIN_TOKENS', snap),
    )


def tag():
    """Return active compression settings formatted for the APC salt, or '' while off."""
    got = settings()
    return '' if got[0] == 'off' else 'context:' + ':'.join(str(v) for v in got)


def budgets(n, keep, how):
    """Spread `keep` across `n` full-attention layers (`uniform` or clipped `pyramid`)."""
    if n <= 0:
        return []
    if how == 'uniform' or n == 1:
        return [float(keep)] * n
    shares = np.array([keep * (1 + PYRAMID_SPREAD * (1 - 2 * i / (n - 1))) for i in range(n)])
    for _ in range(8):
        clipped = np.clip(shares, 0.0, 1.0)
        short = keep * n - clipped.sum()
        free = (clipped < 1.0) & (clipped > 0.0) if short > 0 else clipped > 0.0
        if abs(short) < 1e-9 or not free.any():
            break
        shares = clipped + short * free / free.sum()
    return [float(v) for v in np.clip(shares, 0.0, 1.0)]


def choose(n_blocks, sink_blocks, recent_blocks, share, scores):
    """Select boolean mask `[n_blocks]` keeping `sink_blocks`, `recent_blocks`, and top `share` middle."""
    keep = np.zeros(n_blocks, bool)
    head = min(sink_blocks, n_blocks)
    tail = max(head, n_blocks - recent_blocks)
    keep[:head] = True
    keep[tail:] = True
    middle = np.arange(head, tail)
    k = int(round(share * middle.size))
    if k > 0:
        order = np.lexsort((-middle, -np.asarray(scores, np.float64)[middle]))
        keep[middle[order[:k]]] = True
    return keep


def recent_scores(n_blocks):
    """Score later blocks higher (`recent`)."""
    return np.arange(n_blocks, dtype=np.float64)


def similar_scores(index_keys, ratio):
    """Score blocks by uniqueness (`1 - max neighbor cosine similarity`)."""
    keys = np.asarray(index_keys, np.float32)
    n = keys.shape[0] // ratio
    if n == 0:
        return np.zeros(0)
    pooled = keys[:n * ratio].reshape(n, ratio, -1).mean(1)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    near = np.full(n, -1.0)
    if n > 1:
        cos = (pooled[1:] * pooled[:-1]).sum(1)
        near[1:] = np.maximum(near[1:], cos)
        near[:-1] = np.maximum(near[:-1], cos)
    return 1.0 - near


def h2o_scores(votes, ends, n_blocks):
    """Score blocks by query selection rate over visible queries (`h2o`)."""
    votes = _widen(np.asarray(votes, np.float64).reshape(-1), n_blocks)
    ends = _widen(np.asarray(ends, np.float64).reshape(-1), n_blocks + 1)
    seen = np.cumsum(ends[::-1])[::-1][1:]
    return votes / np.maximum(seen, 1.0)


def snapkv_scores(window, n_blocks, pool=SNAP_POOL):
    """Score blocks by recent observation window counts max-pooled over `pool` neighbors (`snapkv`)."""
    picks = np.asarray(window, np.int64).reshape(-1)
    picks = picks[(picks >= 0) & (picks < n_blocks)]
    counts = np.bincount(picks, minlength=n_blocks).astype(np.float64)[:n_blocks]
    if pool > 1 and n_blocks:
        half = pool // 2
        padded = np.pad(counts, half, constant_values=0.0)
        counts = np.max(np.stack([padded[i:i + n_blocks] for i in range(pool)]), axis=0)
    return counts


def _widen(a, n):
    return a[:n] if a.size >= n else np.concatenate([a, np.zeros(n - a.size, a.dtype)])


def _extra(cache, name):
    got = getattr(cache, f'_quanteeze_{name}', None)
    return None if got is None else np.array(got)


SCORES = {
    'recent': lambda cache, ratio, n: recent_scores(n),
    'similar': lambda cache, ratio, n: similar_scores(_text(cache.index_keys)[0], ratio),
    'h2o': lambda cache, ratio, n: h2o_scores(
        _zeros_if_none(_extra(cache, 'votes')),
        _zeros_if_none(_extra(cache, 'ends')),
        n,
    ),
    'snapkv': lambda cache, ratio, n: snapkv_scores(_zeros_if_none(_extra(cache, 'window'), -1), n),
}


def _zeros_if_none(a, fill=0):
    return np.full(1, fill) if a is None else a


def _text(a):
    """Convert MLX array to NumPy `float32` (casting `bfloat16` safely)."""
    import mlx.core as mx
    return np.array(a.astype(mx.float32)) if a.dtype == mx.bfloat16 else np.array(a)


def dropped_of(cache):
    """Tokens omitted beyond physical cache length (`0` if never compressed)."""
    return int(getattr(cache, 'dropped', 0) or 0)


def reviewed_of(cache):
    """Tokens already reviewed by earlier compressions so middle blocks are not thinned twice."""
    got = _extra(cache, 'reviewed')
    return 0 if got is None or got.size == 0 else int(got.reshape(-1)[0])


def compact(cache, keep_tokens, ratio=4, reviewed=None):
    """Return a single-row `QSAKVCache` holding only `keep_tokens` and updating `dropped` and extras."""
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache
    from . import apc_disk

    keep = np.asarray(keep_tokens, np.int32)
    idx = mx.array(keep)
    out = QSAKVCache()
    keys, values = cache.state[:2]
    out.keys, out.values = mx.take(keys, idx, axis=2), mx.take(values, idx, axis=2)
    apc_disk.gathered(out.keys, keys, keep, 2)
    apc_disk.gathered(out.values, values, keep, 2)
    out.offset = int(out.keys.shape[2])
    if cache.index_keys is not None:
        index_keys = cache.index_keys[:, :cache.offset]
        out.index_keys = mx.take(index_keys, idx, axis=1)
        positions = cache.index_position_ids[..., :cache.offset]
        out.index_position_ids = mx.take(positions, idx, axis=positions.ndim - 1)
        apc_disk.gathered(out.index_keys, index_keys, keep, 1)
        apc_disk.gathered(out.index_position_ids, positions, keep, positions.ndim - 1)
    out.dropped = dropped_of(cache) + int(cache.offset) - int(idx.size)
    reviewed = reviewed_of(cache) if reviewed is None else int(reviewed)
    if reviewed:
        out._quanteeze_reviewed = mx.array([[reviewed]], dtype=mx.int32)
    blocks = np.unique(np.asarray(keep_tokens, np.int64) // ratio)
    blocks = blocks[np.isin(blocks * ratio + ratio - 1, np.asarray(keep_tokens))]
    votes, ends = _extra(cache, 'votes'), _extra(cache, 'ends')
    if votes is not None and ends is not None:
        n = int(cache.offset) // ratio
        seen = np.cumsum(_widen(ends.reshape(-1).astype(np.float64), n + 1)[::-1])[::-1][1:][blocks]
        out._quanteeze_votes = mx.array(_widen(votes.reshape(-1), n)[blocks][None].astype(np.float32))
        after = np.zeros(blocks.size + 1)
        if blocks.size:
            after[1:-1] = seen[:-1] - seen[1:]
            after[-1] = seen[-1]
        out._quanteeze_ends = mx.array(after[None].astype(np.float32))
    return out


def compress(snapshot, how=None, layers=None, keep=None, recent=None, sink=None, min_tokens=None, ratio=4):
    """Compress full-attention `QSAKVCache` layers in `snapshot` according to settings or overrides."""
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    got = settings()
    how, layers, keep, recent, sink, min_tokens = (
        given if given is not None else default
        for given, default in zip((how, layers, keep, recent, sink, min_tokens), got)
    )
    STATS['stored'] += 1
    if how == 'off' or not snapshot:
        return snapshot
    full = [i for i, c in enumerate(snapshot) if type(c) is QSAKVCache and c.keys is not None]
    if not full or len(full) != sum(1 for c in snapshot if isinstance(c, QSAKVCache)):
        return snapshot
    if min(int(snapshot[i].offset) + dropped_of(snapshot[i]) for i in full) < max(min_tokens, 1):
        return snapshot
    shares = budgets(len(full), keep, layers)
    out = list(snapshot)
    before = after = 0
    for share, i in zip(shares, full):
        c = snapshot[i]
        length = int(c.offset)
        n_blocks = length // ratio
        recent_blocks = -(-recent // ratio)
        head = max(-(-sink // ratio), reviewed_of(c) // ratio)
        tail = n_blocks - recent_blocks
        before += length
        if tail <= head:
            after += length
            continue
        keep_blocks = choose(n_blocks, head, recent_blocks, share, SCORES[how](c, ratio, n_blocks))
        tokens = np.concatenate([
            np.flatnonzero(np.repeat(keep_blocks, ratio)),
            np.arange(n_blocks * ratio, length),
        ])
        after += tokens.size
        if tokens.size < length:
            out[i] = compact(c, tokens, ratio, reviewed=int(keep_blocks[:tail].sum()) * ratio)
            _own(out[i])
    if after == before:
        return snapshot
    STATS['compressed'] += 1
    STATS['tokens_before'] += before
    STATS['tokens_after'] += after
    return out


def summary():
    s = STATS
    kept = s['tokens_after'] / s['tokens_before'] if s['tokens_before'] else 1.0
    return (
        f'{s["compressed"]} of {s["stored"]} stored prompt caches compressed, '
        f'full-attention layers to {kept:.0%} of their tokens'
    )


def _batch_dropped(cache, rows):
    got = getattr(cache, '_quanteeze_dropped', None)
    return list(got) if got is not None and len(got) == rows else [0] * rows


def _rows(cache):
    offset = cache.kv_cache.offset
    return int(offset.size) if hasattr(offset, 'size') else 1


def _fill(name):
    return -1 if name == 'window' else 0


def _blank(name, rows, like):
    import mlx.core as mx
    return mx.full((rows, *like.shape[1:]), _fill(name), dtype=like.dtype)


def _stacked(name, parts):
    import mlx.core as mx
    like = next((a for _, a in parts if a is not None), None)
    if like is None:
        return None
    arrays = [a if a is not None else _blank(name, r, like) for r, a in parts]
    axis = 1 if name == 'window' else -1
    width = max(a.shape[axis] for a in arrays)
    out = []
    for a in arrays:
        short = width - a.shape[axis]
        if short:
            pads = [(0, 0)] * a.ndim
            pads[axis] = (short, 0) if name == 'window' else (0, short)
            a = mx.pad(a, pads, constant_values=_fill(name))
        out.append(a)
    return mx.concatenate(out, axis=0)


def _record(cache, selection, window, ratio):
    import mlx.core as mx
    blocks = selection.selected_blocks
    if blocks is None or blocks.ndim != 3:
        return
    rows, _, k = blocks.shape
    n = max(1, int(selection.key_len) // max(1, int(ratio)))
    votes = getattr(cache, '_quanteeze_votes', None)
    ends = getattr(cache, '_quanteeze_ends', None)
    width = max(n, 0 if votes is None else votes.shape[-1])
    if votes is None or votes.shape[0] != rows:
        votes = mx.zeros((rows, width), mx.float32)
        ends = mx.zeros((rows, width + 1), mx.float32)
    if votes.shape[-1] < width:
        votes = mx.pad(votes, [(0, 0), (0, width - votes.shape[-1])])
        ends = mx.pad(ends, [(0, 0), (0, width + 1 - ends.shape[-1])])
    row = mx.arange(rows, dtype=mx.int32)[:, None, None]
    picked = mx.where(blocks >= 0, blocks, width).astype(mx.int32)
    votes = mx.pad(votes, [(0, 0), (0, 1)]).at[row, picked].add(1.0)[:, :width]
    counts = mx.minimum(selection.complete_counts.astype(mx.int32), width)
    ends = ends.at[mx.arange(rows, dtype=mx.int32)[:, None], counts].add(1.0)
    cache._quanteeze_votes, cache._quanteeze_ends = votes, ends
    kept = [votes, ends]
    if window:
        last = getattr(cache, '_quanteeze_window', None)
        if last is None or last.shape[0] != rows or last.shape[-1] != k:
            last = mx.full((rows, 0, k), -1, dtype=mx.int32)
        last = mx.concatenate([last, blocks.astype(mx.int32)], axis=1)
        cache._quanteeze_window = last[:, -window:]
        kept.append(cache._quanteeze_window)
    mx.async_eval(kept)


def _arrays(cache):
    return [
        a for a in (
            cache.keys,
            cache.values,
            cache.index_keys,
            cache.index_position_ids,
            getattr(cache, '_quanteeze_votes', None),
            getattr(cache, '_quanteeze_ends', None),
            getattr(cache, '_quanteeze_reviewed', None),
        )
        if a is not None
    ]


def _own(cache):
    cache._quanteeze_own = True


def install():
    """Install QSA cache state hooks and `APCManager.store_exact_cache` compression while active."""
    how = method()
    if how == 'off':
        return False
    if _INSTALLED[0]:
        return True
    import mlx.core as mx
    from mlx_vlm import apc, apc_adapters
    from mlx_vlm.models.qwen3_5 import language as q35
    from mlx_vlm.models.qwen4_exp import language as q4
    from . import apc_disk

    single, batch = q4.QSAKVCache, q4.BatchQSAKVCache
    window = variables.integer('QUANTEEZE_CONTEXT_WINDOW') if how == 'snapkv' else 0

    state = single.state

    def get_state(self):
        got = state.fget(self)
        if got[0] is not None:
            for whole, part in ((self.keys, got[0]), (self.values, got[1])):
                gather = apc_disk.gather_of(whole)
                if gather is not None and part is not whole and part.shape == whole.shape:
                    apc_disk.gathered(part, *gather)
        extra = {}
        if dropped_of(self):
            extra['dropped'] = mx.array([dropped_of(self)], dtype=mx.int32)
        for name in ('votes', 'ends', 'reviewed'):
            value = getattr(self, f'_quanteeze_{name}', None)
            if value is not None:
                extra[name] = value
        return got if not extra else (*got, extra)

    def set_state(self, value):
        value = tuple(value)
        extra = {}
        if len(value) == 7:
            value, extra = value[:6], value[6]
            if not isinstance(extra, dict):
                extra = {'dropped': extra}
        state.fset(self, value)
        self.dropped = int(extra['dropped'].item()) if extra.get('dropped') is not None else 0
        for name in ('votes', 'ends', 'reviewed'):
            setattr(self, f'_quanteeze_{name}', extra.get(name))

    _patch(single, 'state', property(get_state, set_state))

    to_batch = single.to_batch

    def single_to_batch(self, left_padding):
        out = to_batch(self, left_padding)
        d = dropped_of(self)
        out._quanteeze_dropped = [d]
        if d and not out.kv_cache.empty():
            out.kv_cache.offset = out.kv_cache.offset + d
        for name in EXTRA:
            setattr(out, f'_quanteeze_{name}', getattr(self, f'_quanteeze_{name}', None))
        return out

    _patch(single, 'to_batch', single_to_batch)

    merge = batch.merge.__func__

    def batch_merge(cls, caches, prefix_lens=None):
        caches = list(caches)
        out = merge(cls, caches, prefix_lens=prefix_lens)
        dropped = [dropped_of(c) for c in caches]
        out._quanteeze_dropped = dropped
        if any(dropped) and not out.kv_cache.empty():
            out.kv_cache.offset = out.kv_cache.offset + mx.array(dropped, dtype=out.kv_cache.offset.dtype)
        for name in EXTRA:
            setattr(
                out,
                f'_quanteeze_{name}',
                _stacked(name, [(1, getattr(c, f'_quanteeze_{name}', None)) for c in caches]),
            )
        return out

    _patch(batch, 'merge', classmethod(batch_merge))

    extract = batch.extract

    def batch_extract(self, idx):
        out = extract(self, idx)
        out.dropped = _batch_dropped(self, _rows(self))[idx]
        for name in EXTRA:
            value = getattr(self, f'_quanteeze_{name}', None)
            setattr(out, f'_quanteeze_{name}', None if value is None else value[idx:idx + 1])
        return out

    _patch(batch, 'extract', batch_extract)

    filter_ = batch.filter

    def batch_filter(self, batch_indices):
        rows = _batch_dropped(self, _rows(self))
        filter_(self, batch_indices)
        picked = np.array(batch_indices.tolist() if hasattr(batch_indices, 'tolist') else batch_indices).reshape(-1)
        self._quanteeze_dropped = [rows[int(i)] for i in picked]
        index = mx.array(picked.astype(np.int32))
        for name in EXTRA:
            value = getattr(self, f'_quanteeze_{name}', None)
            if value is not None:
                setattr(self, f'_quanteeze_{name}', value[index])

    _patch(batch, 'filter', batch_filter)

    extend = batch.extend

    def batch_extend(self, other):
        mine_rows, their_rows = _rows(self), _rows(other)
        mine, theirs = _batch_dropped(self, mine_rows), _batch_dropped(other, their_rows)
        extras = {
            name: _stacked(
                name,
                [
                    (mine_rows, getattr(self, f'_quanteeze_{name}', None)),
                    (their_rows, getattr(other, f'_quanteeze_{name}', None)),
                ],
            )
            for name in EXTRA
        }
        extend(self, other)
        self._quanteeze_dropped = mine + theirs
        for name, value in extras.items():
            setattr(self, f'_quanteeze_{name}', value)

    _patch(batch, 'extend', batch_extend)

    def shared_offset(self):
        idx = self.kv_cache._idx
        rows = _rows(self)
        return idx + (_batch_dropped(self, rows)[0] if rows == 1 else 0)

    _patch(batch, '_offset', property(shared_offset))

    restore = q35._restore_batch_padding_metadata

    def restore_padding(cache_entry, offsets, steps):
        out = restore(cache_entry, offsets, steps)
        dropped = getattr(out, '_quanteeze_dropped', None)
        if offsets is not None and dropped is not None and any(dropped) and hasattr(out, 'left_padding'):
            out.left_padding = out.left_padding + mx.array(dropped, dtype=out.left_padding.dtype)
        return out

    for module in (q35, q4):
        if getattr(module, '_restore_batch_padding_metadata', None) is restore:
            _patch(module, '_restore_batch_padding_metadata', restore_padding)

    if how in COUNTED:
        select = q4.Qwen4ExpQSAIndexer.select_from_projected

        def select_from_projected(self, qk, cache, position_ids):
            selection = select(self, qk, cache, position_ids)
            if selection is not None and cache is not None:
                _record(cache, selection, window, self.compress_ratio)
            return selection

        _patch(q4.Qwen4ExpQSAIndexer, 'select_from_projected', select_from_projected)

    clone = apc_adapters.clone_cache_entry

    def clone_cache_entry(c, *, min_capacity_tokens, eval_targets):
        if getattr(c, '_quanteeze_own', False) and min_capacity_tokens is None:
            c._quanteeze_own = False
            arrays = _arrays(c)
            mx.eval(arrays)
            apc_disk.forget(arrays)
            return c
        return clone(c, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets)

    _patch(apc_adapters, 'clone_cache_entry', clone_cache_entry)

    store = apc.APCManager.store_exact_cache

    def store_exact_cache(self, token_ids, prompt_cache, *args, **kwargs):
        try:
            return store(self, token_ids, compress(list(prompt_cache)), *args, **kwargs)
        finally:
            apc_disk.forget()

    _patch(apc.APCManager, 'store_exact_cache', store_exact_cache)
    _INSTALLED[0] = True
    return True


def uninstall():
    """Restore hooks patched by `install()`."""
    from . import apc_disk
    apc_disk.forget(everyone=True)
    while _PATCHED:
        owner, name, was = _PATCHED.pop()
        if was is _MISSING:
            delattr(owner, name)
        else:
            setattr(owner, name, was)
    _INSTALLED[0] = False
