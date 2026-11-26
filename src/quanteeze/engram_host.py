# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Host-side DeepSeek V4.1 Engram hash computation (`QUANTEEZE_ENGRAM_HASHES_ON_HOST`).

Computes `NgramHashState` multi-head rolling XOR-prime hashes in C++ (`vendor/engram/engram.cpp`)
directly from the pass's input token IDs and a host-side token history tracked alongside each
request's cache row (`_quanteeze_engram_host`), avoiding GPU->host synchronization before Engram
lookups and read-ahead prefetching. Verified bit for bit against the GPU hash output on the first
`CHECKS` passes.
"""
import sys

import numpy as np

from . import native_ext, per_load, variables

CHECKS = 3
DEAD = -1
STATE = dict(checked=0, mismatched=0, off=False, served=0, passes=0, tokens=0, resynced=0, said=False)
_TABLES = {}
_EXT = {}


def wanted():
    """Whether host-side Engram hash computation is enabled and has not parted from GPU reference."""
    return not STATE['off'] and not variables.off('QUANTEEZE_ENGRAM_HASHES_ON_HOST')


def _ext():
    """Load and cache the `_engram` C++20 nanobind extension, or None when disabled/unavailable."""
    if 'ext' not in _EXT:
        _EXT['ext'] = native_ext.extension('engram', with_mlx=False, with_stm=False, quiet=True)
    return _EXT['ext']


def _tables(ngram):
    """Extract and cache contiguous `int64` host tables from `NgramHashState`."""
    cached = getattr(ngram, '_quanteeze_engram_host', None)
    if cached is not None:
        return cached
    got = _TABLES.get(id(ngram))
    if got is not None and got[0] is ngram:
        return got[1]
    contig = lambda a: np.ascontiguousarray(np.array(a), np.int64)  # noqa: E731
    t = dict(
        token_map=contig(ngram._token_map),
        mult=contig(ngram._multipliers),
        primes=contig(ngram._primes),
        offsets=contig(ngram._offsets),
        pad=int(ngram.pad_id),
        n=int(ngram.layout.max_ngram_size),
    )
    try:
        object.__setattr__(ngram, '_quanteeze_engram_host', t)
    except Exception:
        _TABLES[id(ngram)] = (ngram, t)
    return t


def hashes_of(t, history, ids, start, mask=None):
    """Compute `[1, L, layers, cols]` `int64` Engram row hashes for `ids` `[L]` at `start`."""
    ext = _ext()
    if ext is None:
        return None
    layers, cols = int(t['mult'].shape[0]), int(t['offsets'].shape[1])
    heads = cols // (int(t['n']) - 1)
    ids_arr = np.ascontiguousarray(ids, np.int64).reshape(-1)
    m = None if mask is None else np.ascontiguousarray(mask, np.uint8).reshape(-1)
    out = np.empty((1, ids_arr.shape[0], layers, cols), np.int64)
    done = ext.hashes(
        int(t['token_map'].ctypes.data),
        int(t['token_map'].shape[0]),
        int(t['mult'].ctypes.data),
        int(t['primes'].ctypes.data),
        int(t['offsets'].ctypes.data),
        int(t['pad']),
        int(t['n']),
        layers,
        heads,
        int(history.ctypes.data) if history.size else 0,
        int(history.shape[0]),
        int(ids_arr.ctypes.data) if ids_arr.size else 0,
        int(ids_arr.shape[0]),
        int(start),
        0 if m is None or not m.size else int(m.ctypes.data),
        int(out.ctypes.data) if out.size else 0,
    )
    return out if done == 0 else None


def _history(row, before, start, need):
    """Return `row`'s host history array sized to at least `need`, resyncing from `before` if needed."""
    kept = getattr(row, '_quanteeze_engram_host', None)
    if kept is not None and kept[1] is before and kept[0].shape[0] >= start:
        history = kept[0]
    elif start == 0:
        history = np.zeros(0, np.int64)
    elif before is None:
        return None
    else:
        STATE['resynced'] += 1
        history = np.array(before)[0].astype(np.int64)
    if history.shape[0] < need:
        grown = np.zeros(max(need, 2 * history.shape[0], 256), np.int64)
        if history.size:
            grown[:history.shape[0]] = history
        history = grown
    return history


def pass_hashes(ngram, row, input_ids, before, gpu, image_token_id=None):
    """Compute a single-row pass's Engram hashes `[1, L, layers, cols]` on the host, or return None."""
    if not wanted():
        return None
    ids = np.asarray(input_ids).astype(np.int64)
    if ids.ndim != 2 or ids.shape[0] != 1:
        return None
    start = int(row.offset)
    history = _history(row, before, start, start + ids.shape[1])
    if history is None:
        return None
    mask = None if image_token_id is None else ids[0] != int(image_token_id)
    t = _tables(ngram)
    got = hashes_of(t, history, ids[0], start, mask)
    if got is None:
        return None
    row._quanteeze_engram_host = (history, getattr(row, 'engram', None))
    if STATE['checked'] < CHECKS:
        STATE['checked'] += 1
        ref_arr = gpu() if callable(gpu) else gpu
        want = np.asarray(ref_arr)
        if want.dtype != got.dtype or want.shape != got.shape or not np.array_equal(want, got):
            STATE['mismatched'] += 1
            STATE['off'] = True
            if not STATE['said']:
                STATE['said'] = True
                print(
                    'quanteeze: QUANTEEZE_ENGRAM_HASHES_ON_HOST: host Engram hashes differed from GPU '
                    'reference; using GPU hashes from now on',
                    file=sys.stderr,
                    flush=True,
                )
            return None
    STATE['served'] += 1
    STATE['passes'] += 1
    STATE['tokens'] += int(ids.shape[1])
    return got


@per_load.on_let_go
def _let_go_native():
    """Drop cached `_engram` extension handle on unload so QUANTEEZE_NATIVE applies on the next load."""
    _EXT.clear()


@per_load.on_let_go
def _let_go():
    """Release cached NgramHashState tables and reset check count on unload (`STATE['off']` sticks)."""
    _TABLES.clear()
    STATE['checked'] = 0
