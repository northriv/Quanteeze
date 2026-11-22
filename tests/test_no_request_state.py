# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Verify global pass/decode state replaced by `per_thread` or locks stays isolated across threads."""
import threading
import weakref
from collections import OrderedDict

import mlx.core as mx
import numpy as np

from quanteeze import apc_images, cuda_ple, dspark, ngram_prefetch, per_thread
from quanteeze.arch import deepseek_v41, qwen4_exp


def test_dspark_and_cuda_ple_fields_do_not_bleed_across_threads():
    barrier = threading.Barrier(2)
    results = {}

    def worker(tid, pending_val, drafted_val, row_val, ple_tok, ple_pend):
        dspark.PENDING[0] = pending_val
        dspark.DRAFTED[0] = drafted_val
        dspark.ROWS.clear()
        dspark.ROWS.append(row_val)
        cuda_ple.TOKENS[0] = ple_tok
        cuda_ple.PENDING[0] = ple_pend
        barrier.wait(timeout=5.0)
        results[tid] = (
            dspark.PENDING[0],
            dspark.DRAFTED[0],
            list(dspark.ROWS),
            cuda_ple.TOKENS[0],
            cuda_ple.PENDING[0],
        )

    t1 = threading.Thread(target=worker, args=(1, 'p1', 'd1', 101, 't1', 'pp1'))
    t2 = threading.Thread(target=worker, args=(2, 'p2', 'd2', 202, 't2', 'pp2'))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert results[1] == ('p1', 'd1', [101], 't1', 'pp1')
    assert results[2] == ('p2', 'd2', [202], 't2', 'pp2')
    assert dspark.PENDING[0] is None
    assert cuda_ple.TOKENS[0] is None


def test_deepseek_v41_verifying_windows_and_entry_are_thread_local():
    barrier = threading.Barrier(2)
    seen = {}

    class _DummyLM:
        entry = deepseek_v41.SpecLanguageModel.entry

    dummy = _DummyLM()

    def worker(tid, flag, label):
        class _Entry:
            def __init__(self, name):
                self.name = name

        keep = _Entry(label)
        deepseek_v41.VERIFYING[0] = flag
        per_thread.PASS.entries[id(dummy)] = weakref.ref(keep)
        win = per_thread.PASS.memo.setdefault('v41_windows', {})
        win[(tid,)] = label
        barrier.wait(timeout=5.0)
        got = dummy.entry()
        seen[tid] = (
            deepseek_v41.VERIFYING[0],
            got.name if got is not None else None,
            dict(per_thread.PASS.memo['v41_windows']),
        )

    t1 = threading.Thread(target=worker, args=(1, True, 'req1'))
    t2 = threading.Thread(target=worker, args=(2, False, 'req2'))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert seen[1] == (True, 'req1', {(1,): 'req1'})
    assert seen[2] == (False, 'req2', {(2,): 'req2'})
    assert deepseek_v41.VERIFYING[0] is False


def test_qwen4_exp_compiled_slots_are_thread_local():
    class _Block:
        pass

    block = _Block()
    barrier = threading.Barrier(2)
    ids = {}

    def worker(tid):
        g = per_thread.slots(block).setdefault('_router_and_shared_by_b', {})
        g[tid] = tid * 10
        barrier.wait(timeout=5.0)
        ids[tid] = dict(per_thread.slots(block)['_router_and_shared_by_b'])

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert ids[1] == {1: 10}
    assert ids[2] == {2: 20}
    assert not hasattr(qwen4_exp, '_SHARED_COMPILED')


def test_ngram_prefetch_kept_evaluates_mx_arrays_and_install_isolates_threads():
    lazy = mx.array([1, 2, 3], dtype=mx.int32) + mx.array([4, 5, 6], dtype=mx.int32)
    np_buf = np.array([10, 20], dtype=np.int32)
    out = ngram_prefetch._kept({'arr': lazy, 'np': np_buf})
    assert np_buf.flags.writeable is False
    np.testing.assert_array_equal(np.array(out['arr']), [5, 7, 9])

    class _FakePool:
        min_rows = 4

        def fetch(self, _items):
            return None

    class _FakeEmbedding:
        cache_rows = True

        def __init__(self):
            self._cache = OrderedDict()
            self._arrays = None
            self._interleaved = None
            self._shards = ()
            self.calls = 0

        def _read_rows(self, row_ids):
            self.calls += 1
            return np.asarray(row_ids, dtype=np.int64) * 2

    ngram_prefetch.install(_FakeEmbedding, prefetcher=_FakePool())
    emb = _FakeEmbedding()
    barrier = threading.Barrier(2)
    out_threads = {}

    def worker(tid):
        rids = np.array([tid], dtype=np.int64)
        first = emb._read_rows(rids)
        barrier.wait(timeout=5.0)
        again = emb._read_rows(rids)
        out_threads[tid] = (first.tolist(), again.tolist(), dict(per_thread.slots(emb))['_quanteeze_last'][1].tolist())

    t1 = threading.Thread(target=worker, args=(11,))
    t2 = threading.Thread(target=worker, args=(22,))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert out_threads[11] == ([22], [22], [22])
    assert out_threads[22] == ([44], [44], [44])
    # Each thread missed once then hit its own per_thread slot
    assert emb.calls == 2


def test_apc_images_plans_lock_under_concurrent_updates(monkeypatch):
    monkeypatch.setattr(apc_images, '_PLANS', OrderedDict())
    errors = []

    def worker(tid):
        try:
            for k in range(200):
                full = (tid << 16) + k
                apc_images._note(full, tid, (k,))
                got = apc_images._plan(full)
                if got is not None:
                    assert got[0] == tid
        except Exception as err:
            errors.append(err)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors
    assert len(apc_images._PLANS) <= apc_images.PLANS_KEPT
