# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for thread-local pass state, per-thread descriptors, and counters (`quanteeze.per_thread`)."""
import threading
import weakref

import mlx.core as mx
import mlx.nn as nn
import pytest

from quanteeze import per_thread


def test_field_and_rows_isolation_across_threads():
    with pytest.raises(ValueError):
        per_thread.Field('does_not_exist')
    with pytest.raises(ValueError):
        per_thread.Rows('does_not_exist')

    f = per_thread.Field('round')
    r = per_thread.Rows('rows')
    with pytest.raises(IndexError):
        _ = f[1]
    with pytest.raises(IndexError):
        f[1] = 3

    barrier = threading.Barrier(2)
    seen = {}

    def worker(tid):
        f[0] = f'val-{tid}'
        r.clear()
        r.append(tid)
        r.append(tid * 10)
        barrier.wait(timeout=5.0)
        seen[tid] = (f[0], f[-1], per_thread.PASS.round, list(r), len(r), r[1])

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert seen[1] == ('val-1', 'val-1', 'val-1', [1, 10], 2, 10)
    assert seen[2] == ('val-2', 'val-2', 'val-2', [2, 20], 2, 20)
    assert f[0] is None
    assert list(r) == []


def test_counts_single_and_multithreaded_accumulation():
    c = per_thread.Counts({'hits': 0, 'lat_ms': 0.0, 'buckets': [0, 0]})
    assert c['hits'] == 0
    c['hits'] += 5
    c['lat_ms'] += 1.5
    c['buckets'][1] += 2
    assert c['hits'] == 5
    assert c['lat_ms'] == pytest.approx(1.5)
    assert list(c['buckets']) == [0, 2]
    assert c.setdefault('extra', 7) == 7
    assert c.get('missing', 'fallback') == 'fallback'

    barrier = threading.Barrier(4)

    def worker():
        barrier.wait(timeout=5.0)
        for _ in range(50):
            c['hits'] += 1
            c['lat_ms'] += 0.25
            c['buckets'][0] += 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    tot = c.total()
    assert tot['hits'] == 5 + 4 * 50
    assert tot['lat_ms'] == pytest.approx(1.5 + 4 * 50 * 0.25)
    assert tot['buckets'] == [4 * 50, 2]
    assert set(c.keys()) == {'hits', 'lat_ms', 'buckets', 'extra'}


def test_own_attributes_and_module_parameters():
    class _Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)
            self._rope_deltas = 'init-delta'
            self._position_ids = None

    mod = _Mod()
    changed = per_thread.own_attributes(mod, ('_rope_deltas', '_position_ids', 'not_present'))
    assert changed == ('_rope_deltas', '_position_ids')
    assert mod._rope_deltas == 'init-delta'

    # Converting an attribute into a thread-local descriptor must not pollute MLX parameters
    param_keys = set(mod.parameters().keys())
    assert param_keys == {'proj'}

    barrier = threading.Barrier(2)
    observed = {}

    def worker(tid):
        mod._rope_deltas = mx.array([tid], dtype=mx.int32)
        mod._position_ids = f'pos-{tid}'
        barrier.wait(timeout=5.0)
        observed[tid] = (int(mod._rope_deltas[0].item()), mod._position_ids)

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert observed[1] == (1, 'pos-1')
    assert observed[2] == (2, 'pos-2')


def test_slots_request_number_serial_passing_and_passmark():
    class _Switch:
        pass

    sw = _Switch()
    per_thread.slots(sw)['k'] = 'main'
    assert per_thread.slots(sw)['k'] == 'main'

    entry1 = _Switch()
    entry1.offset = 17
    entry2 = _Switch()
    r1 = per_thread.request_number(entry1)
    r2 = per_thread.request_number(entry2)
    assert r1 >= 0 and r2 > r1
    assert per_thread.request_number(entry1) == r1
    assert per_thread.request_number(None) == -1
    assert per_thread.request_number(42) == -1  # non-weakrefable

    s0 = per_thread.next_serial(r1)
    s1 = per_thread.next_serial(r1)
    assert s1 == s0 + 1

    per_thread.passing(entry1, verify=True)
    mark = per_thread.PassMark(number=9, request=r1, serial=17, kind='verify')
    assert mark.kind_code == per_thread.KIND_VERIFY

    ref = weakref.ref(sw)
    del sw
    assert ref() is None or per_thread.PASS.memo.get('switch_slots') is not None
