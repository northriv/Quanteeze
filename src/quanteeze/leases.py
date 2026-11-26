# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""GPU working-memory request leases backed by the KAME STM work room (`QUANTEEZE_REQUEST_LEASES`).

When multiple concurrent requests execute (`QUANTEEZE_REQUEST_THREADS > 1`), each request leases
its decode pass working memory (held until completion), its prompt pass working memory (released once
prompt processing finishes), and KV-cache room in `CHUNK`-token increments from the KAME STM work
room (`room_put`, `room_take`, `room_give`, `room_free`).
"""
import sys
import threading

import mlx.core as mx
import mlx.nn as nn

from . import per_load, variables

CHUNK = 1024
SLACK_TOKENS = 256
MEASURED_STEPS = 8
WAIT_MS = 50.0

STATS = dict(
    seeded_gb=0.0, leases=0, held_back=0, forced_alone=0, more=0,
    overdrawn=0, overdraw_gb=0.0, working_over=0, kv_over=0, given=0,
    prompt_room=0, prompt_room_gb=0.0,
)
_SIZES = dict(working=0, prompt=0, prompt_tokens=0, kv_token=0.0, kv_fixed=0)
_LOCK = threading.Lock()
_ROOM = [None]
_SEEDED = [None]
_HELD = {}
_AUTO = [None]


def enabled():
    """Whether GPU memory request leases are enabled (`QUANTEEZE_REQUEST_LEASES`, default `on`)."""
    return not variables.off('QUANTEEZE_REQUEST_LEASES', variables.at_load())


def seeded():
    return _ROOM[0] is not None


def seeded_room():
    """Return seeded work-room size in bytes, or None before seeding."""
    return _SEEDED[0]


@per_load.on_let_go
def reset():
    """Reset work-room lease state when unloading or switching models and restore cache limit."""
    with _LOCK:
        was = _ROOM[0]
        _ROOM[0] = _SEEDED[0] = _AUTO[0] = None
        _HELD.clear()
        _SIZES.update(working=0, prompt=0, prompt_tokens=0, kv_token=0.0, kv_fixed=0)
        STATS.update({k: type(v)() for k, v in STATS.items()})
    if was is not None:
        from . import serve
        mx.set_cache_limit(serve._cache_limit())


def _driver():
    from . import demand, stm_core
    for d in list(demand._LIVE):
        stm = getattr(d, '_stm', None)
        if stm is not None and stm.started:
            return stm
    if (variables.get('QUANTEEZE_REQUEST_THREADS_ANY', variables.at_load()) or 'off').lower() == 'on':
        return stm_core.room_driver()
    return None


def _gpu_bytes():
    """Active MLX GPU bytes (excluding any host-level share if configured)."""
    from . import serve
    gpu_lim = getattr(serve, '_GPU_LIMIT', None)
    host = (mx.get_memory_limit() - gpu_lim[0]) if gpu_lim else 0
    return mx.get_active_memory() - max(0, host)


def seed():
    """Seed the KAME STM work room once after model load and before concurrent requests begin."""
    if _ROOM[0] is not None:
        return True
    if not enabled():
        return False
    driver = _driver()
    if driver is None:
        return False
    from . import cuda_ple, guards
    working = guards.recommended_budget() or 0
    room = working - _gpu_bytes()
    found = cuda_ple.gpu_room() if mx.default_device() == mx.gpu else None
    if found is not None:
        room = min(room, found[0] + found[1] + mx.get_cache_memory())
    room = max(0, int(room))
    driver.room_put(room)
    _SEEDED[0] = room
    _ROOM[0] = driver
    STATS['seeded_gb'] = round(room / 1e9, 2)
    extra = '' if found is None else f', driver free {found[0] / 1e9:.2f} GB, pool unused {found[1] / 1e9:.2f} GB'
    print(
        f'quanteeze: leases: work room {room / 1e9:.2f} GB (working set {working / 1e9:.1f} GB, '
        f'MLX active {_gpu_bytes() / 1e9:.1f} GB{extra})',
        file=sys.stderr,
        flush=True,
    )
    cache_limit()
    return True


def _arrays(x):
    if isinstance(x, mx.array):
        return [x]
    if isinstance(x, (list, tuple)):
        return [a for a in x if isinstance(a, mx.array)]
    return []


def kv_sizes(cache):
    """Return `(bytes_per_token, fixed_bytes)` across a request's cache containers."""
    seen, per, fixed, todo = set(), 0.0, 0, [(cache, 0)]
    while todo:
        obj, depth = todo.pop()
        if obj is None or id(obj) in seen or depth > 8 or isinstance(obj, nn.Module):
            continue
        seen.add(id(obj))
        if isinstance(obj, mx.array):
            fixed += obj.nbytes
        elif isinstance(obj, dict):
            todo.extend((v, depth + 1) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            todo.extend((v, depth + 1) for v in obj)
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            fields = vars(obj)
            keys, values = fields.get('keys'), fields.get('values')
            arrays = [a for a in _arrays(keys) + _arrays(values)]
            if arrays and arrays[0].ndim >= 2 and arrays[0].shape[-2] > 0:
                for a in arrays:
                    seen.add(id(a))
                per += sum(a.nbytes for a in arrays) / arrays[0].shape[-2]
            todo.extend((v, depth + 1) for k, v in fields.items() if k not in ('keys', 'values'))
    return per, fixed


def note_kv(cache):
    """Record maximum observed per-token and fixed cache sizes from `cache`."""
    per, fixed = kv_sizes(cache)
    with _LOCK:
        _SIZES['kv_token'] = max(_SIZES['kv_token'], per)
        _SIZES['kv_fixed'] = max(_SIZES['kv_fixed'], int(fixed))


def measure(fn, prompting, tokens, lease, cache_of):
    """Measure working-memory peak of solo step `fn()`, excluding KV cache growth."""
    before_cache = cache_of()
    mx.reset_peak_memory()
    before = mx.get_active_memory()
    out = fn()
    grown = max(0, cache_of() - before_cache)
    used = max(0, int(mx.get_peak_memory() - before - grown))
    with _LOCK:
        if prompting:
            if used > _SIZES['prompt']:
                _SIZES['prompt'], _SIZES['prompt_tokens'] = used, max(1, int(tokens))
            if lease is not None and lease.prompt and used > lease.prompt:
                STATS['working_over'] += 1
        else:
            _SIZES['working'] = max(_SIZES['working'], used)
            if lease is not None and lease.working and used > lease.working:
                STATS['working_over'] += 1
    return out


def _prompt_bytes(tokens, step):
    most, at = _SIZES['prompt'], _SIZES['prompt_tokens']
    if not most:
        return 0
    n = min(int(tokens), int(step or tokens))
    return int(most * max(1.0, n / max(1, at)))


def _kv_bytes(tokens):
    return int(_SIZES['kv_fixed'] + _SIZES['kv_token'] * (int(tokens) + SLACK_TOKENS))


class Lease:
    """Per-request RAII lease of GPU working memory and KV-cache room from the STM work room."""

    def __init__(self, uid, prompt_tokens, max_tokens, step):
        self.uid = uid
        self.working = int(_SIZES['working'])
        self.prompt = _prompt_bytes(prompt_tokens, step)
        self.covered = int(prompt_tokens) + min(int(max_tokens or CHUNK), CHUNK)
        self.kv = _kv_bytes(self.covered)
        self.prompt_tokens = int(prompt_tokens)
        self.max_tokens = int(max_tokens or 0)

    def total(self):
        return self.working + self.prompt + self.kv

    def take(self, alone):
        """Acquire initial lease from work room (or return False if contended/full unless `alone`)."""
        room = _ROOM[0]
        if room is None:
            return True
        if room.room_take(self.total(), overdraw=alone, timeout_ms=WAIT_MS):
            if alone and room.room_free() < 0:
                STATS['forced_alone'] += 1
            with _LOCK:
                _HELD[self.uid] = self
                STATS['leases'] += 1
            cache_limit()
            return True
        return False

    def prompt_done(self):
        """Release prompt pass working-memory reservation back to the work room."""
        room, n = _ROOM[0], self.prompt
        self.prompt = 0
        if room is not None and n:
            room.room_give(n)
            cache_limit()

    def more(self, generated, oldest, cancelled=lambda: False):
        """Lease another `CHUNK` tokens of KV cache room (oldest running request may overdraw)."""
        room = _ROOM[0]
        if room is None or self.prompt_tokens + int(generated) + SLACK_TOKENS // 4 < self.covered:
            return
        covered = self.covered + CHUNK
        want = _kv_bytes(covered) - self.kv
        while want > 0 and not cancelled():
            if room.room_take(want, overdraw=oldest(), timeout_ms=WAIT_MS):
                if room.room_free() < 0:
                    STATS['overdrawn'] += 1
                    STATS['overdraw_gb'] = round(max(STATS['overdraw_gb'], -room.room_free() / 1e9), 2)
                self.kv += want
                break
        self.covered = covered
        STATS['more'] += 1
        cache_limit()

    def held(self, actual):
        """Reconcile actual allocated cache bytes above current KV lease."""
        room = _ROOM[0]
        if room is not None and actual > self.kv:
            STATS['kv_over'] += 1
            room.room_take(actual - self.kv, overdraw=True)
            self.kv = int(actual)

    def give(self):
        """Return all remaining leased bytes to the work room (idempotent)."""
        room = _ROOM[0]
        with _LOCK:
            _HELD.pop(self.uid, None)
            n, self.working, self.prompt, self.kv = self.total(), 0, 0, 0
        if room is not None and n:
            room.room_give(n)
            STATS['given'] += 1
            cache_limit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.give()
        return False


def room_for_prompt(nbytes):
    """Borrow up to `nbytes` from free work room without blocking or overdrawing."""
    room = _ROOM[0]
    if room is None or nbytes <= 0:
        return 0
    n = int(min(int(nbytes), room.room_free()))
    if n <= 0 or not room.room_take(n):
        return 0
    with _LOCK:
        STATS['prompt_room'] += 1
        STATS['prompt_room_gb'] = round(max(STATS['prompt_room_gb'], n / 1e9), 2)
    cache_limit()
    return n


def room_back(nbytes):
    """Return `nbytes` taken via `room_for_prompt()`."""
    room = _ROOM[0]
    if room is not None and nbytes > 0:
        room.room_give(int(nbytes))
        cache_limit()


def cache_limit():
    """Cap MLX's freed-buffer cache limit to available work room plus held working memory."""
    room = _ROOM[0]
    if room is None:
        return
    if _AUTO[0] is None:
        from . import serve
        _AUTO[0] = serve._cache_limit()
    with _LOCK:
        working = sum(l.working + l.prompt for l in _HELD.values())
    mx.set_cache_limit(int(min(_AUTO[0], max(0, room.room_free()) + working)))


def sizes():
    with _LOCK:
        return dict(_SIZES)


def summary():
    s = sizes()
    room = _ROOM[0]
    return (
        (f'room free {room.room_free() / 1e9:.2f} GB, ' if room is not None else 'no room, ')
        + f'working {s["working"] / 1e6:.0f} MB, prompt {s["prompt"] / 1e6:.0f} MB over '
        f'{s["prompt_tokens"]} tokens, caches {s["kv_token"] / 1e3:.1f} kB/tok and '
        f'{s["kv_fixed"] / 1e6:.0f} MB fixed; {STATS}'
    )
