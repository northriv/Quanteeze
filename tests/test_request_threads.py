# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for concurrent request worker threads (`quanteeze.request_threads`)."""
import ctypes
from queue import Queue
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import pytest

from quanteeze import cuda_ple, demand, parallel, request_threads


def test_own_random_sampler_isolation():
    logits = mx.array([[0.1, 0.4, 0.9, -0.3, 0.2, 0.7]], dtype=mx.float32)

    def raw_sampler(lp):
        return mx.random.categorical(lp)

    def solo_seq(seed):
        mx.random.seed(seed)
        draws = []
        for _ in range(8):
            x = raw_sampler(logits)
            mx.eval(x)
            draws.append(int(x.item()))
        return draws

    exp1 = solo_seq(101)
    exp2 = solo_seq(202)

    s1 = request_threads._OwnRandom(raw_sampler, 101)
    s2 = request_threads._OwnRandom(raw_sampler, 202)
    barrier = threading.Barrier(2)
    got = {}

    def worker(idx, sampler):
        barrier.wait(timeout=5.0)
        seq = []
        for _ in range(8):
            x = sampler(logits)
            mx.eval(x)
            seq.append(int(x.item()))
        got[idx] = seq

    t1 = threading.Thread(target=worker, args=(1, s1))
    t2 = threading.Thread(target=worker, args=(2, s2))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert got[1] == exp1
    assert got[2] == exp2


def test_while_alone_defers_cuda_ple_free_until_last_thread_finishes(monkeypatch):
    freed = []

    class _FakeCuda:
        def cuCtxPushCurrent_v2(self, _ctx):
            return 0

        def cuCtxSynchronize(self):
            return 0

        def cuMemFreeHost(self, host):
            freed.append(host)
            return 0

        def cuCtxPopCurrent_v2(self, _out):
            return 0

    monkeypatch.setattr(cuda_ple, '_DRIVER', [(_FakeCuda(), ctypes.c_void_p(1))])
    monkeypatch.setattr(cuda_ple, '_LATER', [(0x1000, 64)])
    monkeypatch.setattr(cuda_ple, '_LATER_SAID', [True])
    monkeypatch.setitem(cuda_ple.STATS, 'pinned_bytes', 64)

    entered = threading.Event()
    release = threading.Event()

    def other_worker():
        with request_threads._SERVING_LOCK:
            request_threads._SERVING.add(threading.get_ident())
        try:
            entered.set()
            release.wait(timeout=5.0)
        finally:
            with request_threads._SERVING_LOCK:
                request_threads._SERVING.discard(threading.get_ident())

    t = threading.Thread(target=other_worker)
    t.start()
    assert entered.wait(timeout=5.0)
    try:
        assert request_threads.others_serving() is True
        cuda_ple.free_later()
        assert freed == []
        assert cuda_ple._LATER == [(0x1000, 64)]
    finally:
        release.set()
        t.join(timeout=5.0)

    assert request_threads.others_serving() is False
    cuda_ple.free_later()
    assert freed == [0x1000]
    assert cuda_ple._LATER == []
    assert cuda_ple.STATS['pinned_bytes'] == 0


def test_own_cancellations_leaves_other_requests_untouched():
    cancelled_calls = []

    class _FakeGenerator:
        def __init__(self):
            self._cancel_lock = threading.Lock()
            self._cancelled = {1, 2, 3}

        def _cancel(self, uid):
            cancelled_calls.append(uid)

    class _FakeBatchGen:
        def remove(self, uid):
            return uid == 1

    gen = _FakeGenerator()
    bg = _FakeBatchGen()
    q1 = Queue()
    q2 = Queue()
    active = {1: {'rqueue': q1}, 9: {'rqueue': q2}}

    request_threads._own_cancellations(gen, bg, active)
    assert 1 not in active
    assert 9 in active
    assert q1.get_nowait() is None
    assert gen._cancelled == {2, 3}
    assert cancelled_calls == []


def test_together_ok_flags_and_dispatch_runs_concurrent_workers(monkeypatch):
    monkeypatch.setattr(parallel, 'size', lambda: 1)
    monkeypatch.setattr(demand, '_LIVE', set())

    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS_ANY', 'off')
    assert request_threads.together_ok() is False

    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS_ANY', 'on')
    assert request_threads.together_ok() is True

    monkeypatch.setattr(parallel, 'size', lambda: 2)
    assert request_threads.together_ok() is False
    monkeypatch.setattr(parallel, 'size', lambda: 1)

    class _DummyDemand:
        def __init__(self, started):
            self._stm = SimpleNamespace(started=started)

    dummy_started = _DummyDemand(True)
    dummy_unstarted = _DummyDemand(False)
    monkeypatch.setattr(demand, '_LIVE', {dummy_started})
    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS_ANY', 'off')
    assert request_threads.together_ok() is True
    monkeypatch.setattr(demand, '_LIVE', {dummy_started, dummy_unstarted})
    assert request_threads.together_ok() is False
    monkeypatch.setattr(demand, '_LIVE', set())
    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS_ANY', 'on')

    # End-to-end _dispatch with mocked BatchGenerator running two requests concurrently
    from mlx_vlm.server import generation as G

    monkeypatch.setattr(request_threads, 'STATS', dict(request_threads.STATS))
    request_threads.STATS.update(first_here=0, threads_started=0, together=0, alone=0, most_at_once=0)
    monkeypatch.setattr(request_threads, '_own_request_state', lambda g: None)
    monkeypatch.setattr(request_threads, '_no_cpu_blocks', lambda: None)
    monkeypatch.setattr(request_threads, '_evaluated', lambda g: None)
    monkeypatch.setattr(request_threads, '_one_stream', lambda: None)

    conc_barrier = threading.Barrier(2, timeout=3.0)

    class _MockStreamer:
        def __init__(self, *_a, **_kw):
            pass

        def finalize(self):
            return ''

    class _MockBatchGen:
        def __init__(self, *a, **kw):
            self._prompt_batch = None
            self._unprocessed_sequences = []
            self.uid_count = 0
            self.uids = []
            self.step = 0

        def insert(self, prompts, **_kw):
            uid = int(self.uid_count)
            self.uids.append(uid)
            return (uid,)

        def next(self):
            uid = self.uids[0]
            self.step += 1
            if self.step == 1:
                return ([SimpleNamespace(uid=uid, prompt_tps=100.0, cached_tokens=0)], [])
            # Wait at barrier once both worker threads (req 2 and req 3) have finished prompt and run decode
            if uid in (2, 3):
                conc_barrier.wait()
            res = SimpleNamespace(
                uid=uid,
                token=42,
                token_logprob=0.0,
                finish_reason='stop',
                top_logprobs=None,
            )
            return ([], [res])

        def close(self):
            pass

    monkeypatch.setattr(G, 'BatchGenerator', _MockBatchGen)
    monkeypatch.setattr(G, '_ServerTokenStreamer', _MockStreamer)
    monkeypatch.setattr(G, 'make_streaming_detokenizer', lambda _p: None)
    monkeypatch.setattr(G, 'GenerationContext', lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(G, 'StreamingToken', lambda **kw: SimpleNamespace(**kw))

    class _MockResponseGen:
        def __init__(self):
            self.model = SimpleNamespace(language_model=None)
            self.processor = None
            self.tokenizer = None
            self.stop_tokens = set()
            self.kv_bits = None
            self.kv_group_size = 64
            self.kv_quant_scheme = 'affine'
            self.quantized_kv_start = 0
            self.top_logprobs_k = 0
            self.apc_manager = None
            self.draft_model = None
            self.draft_kind = None
            self._cancel_lock = threading.Lock()
            self._cancelled = set()
            self._ready = threading.Event()
            self._stop = False
            self.requests = Queue()
            self._run_called = False

        def _initialize_model(self):
            pass

        def _effective_prefill_step_size(self):
            return 512

        def _make_sampler(self, _args):
            return None

        def _make_logits_processors(self, _args, _ids):
            return []

        def _gpu_embed(self, raw_inputs, _images, apc_semantic_hash=None):
            return mx.array([[1, 2, 3]], dtype=mx.int32), {}

        def _log_prefill_started(self, *_a, **_kw):
            return {}

        def _log_prefill_progress(self, *_a, **_kw):
            pass

        def _log_prefill_completed(self, *_a, **_kw):
            pass

        def _stream_text(self, _info, tok, _fr):
            return f'tok:{tok}'

        def _log_decode_progress(self, *_a, **_kw):
            return 0.0

        def _cancel(self, _uid):
            pass

    gen = _MockResponseGen()
    reqs = [
        SimpleNamespace(
            rqueue=Queue(),
            args=SimpleNamespace(max_tokens=4, logprobs=0, temperature=0),
            prompt_tokens=3,
            raw_inputs=[1, 2, 3],
            images=None,
            apc_semantic_hash=None,
            thinking_budget_criteria=None,
        )
        for _ in range(3)
    ]

    t = threading.Thread(target=request_threads._dispatch, args=(gen, 2))
    t.start()
    assert gen._ready.wait(timeout=5.0)
    try:
        # Req 1 runs on dispatcher thread first (first_here=1), then req 2 & req 3 run concurrently
        gen.requests.put(reqs[0])
        assert reqs[0].rqueue.get(timeout=5.0).uid == 1
        assert reqs[0].rqueue.get(timeout=5.0).token == 42
        assert reqs[0].rqueue.get(timeout=5.0) is None

        gen.requests.put(reqs[1])
        gen.requests.put(reqs[2])
        for r in reqs[1:]:
            ctx = r.rqueue.get(timeout=5.0)
            tok = r.rqueue.get(timeout=5.0)
            end = r.rqueue.get(timeout=5.0)
            assert ctx.uid in (2, 3)
            assert tok.token == 42
            assert end is None
    finally:
        gen._stop = True
        t.join(timeout=5.0)

    assert request_threads.STATS['first_here'] == 1
    assert request_threads.STATS['threads_started'] == 2
    assert request_threads.STATS['together'] >= 1
    assert request_threads.STATS['most_at_once'] >= 2


def test_mlx_buffers_wired_presets_and_prompt_turns(monkeypatch):
    import quanteeze
    from quanteeze import home, prefill, serve

    assert quanteeze.MLX_BUFFERS['MLX_MAX_OPS_PER_BUFFER'] == '300'

    saved_wired = serve._WIRED[0]
    try:
        monkeypatch.delenv('QUANTEEZE_WIRED_GB', raising=False)
        assert serve._wired_limit() == 0
        monkeypatch.setenv('QUANTEEZE_WIRED_GB', 'off')
        assert serve._wired_limit() == 0
        monkeypatch.setenv('QUANTEEZE_WIRED_GB', '2.5')
        assert serve._wired_limit() == int(2.5 * 1e9)
        monkeypatch.setenv('QUANTEEZE_WIRED_GB', 'auto')
        expected_auto = (
            int(mx.device_info().get('max_recommended_working_set_size', 0))
            if mx.metal.is_available() else 0
        )
        assert serve._wired_limit() == expected_auto

        monkeypatch.setenv('QUANTEEZE_WIRED_GB', '0')
        serve._wire()
        assert serve._WIRED[0] == 0

        monkeypatch.delenv('QUANTEEZE_WIRED_GB', raising=False)
        assert home.presets()['wired']['now'] == 'off'
        monkeypatch.setenv('QUANTEEZE_WIRED_GB', 'auto')
        assert home.presets()['wired']['now'] == 'auto'
        monkeypatch.setenv('QUANTEEZE_WIRED_GB', '12')
        assert home.presets()['wired']['now'] == 'custom'

        monkeypatch.setattr(mx.metal, 'is_available', lambda: False)
        assert 'note' in home.presets()['wired']
    finally:
        serve._WIRED[0] = saved_wired
        if mx.metal.is_available():
            mx.set_wired_limit(saved_wired)

    monkeypatch.setattr(prefill.SETTINGS, 'chunk', 2048)
    monkeypatch.setenv('QUANTEEZE_PROMPT_CONTENDED_TOKENS', '512')
    monkeypatch.setenv('QUANTEEZE_PROMPT_MOE_TOKENS', '2048')
    fake_turns = SimpleNamespace(
        active=lambda: True,
        contended=lambda: True,
        tokens=lambda: 512,
        turn=lambda _layer: SimpleNamespace(__enter__=lambda: None, __exit__=lambda *a: None),
    )
    saved_turns = prefill.TURNS
    try:
        prefill.TURNS = fake_turns
        assert prefill._chunk_step() == 512
        dummy_sw = SimpleNamespace(demand=object())
        dummy_layer = SimpleNamespace(mlp=SimpleNamespace(switch_mlp=dummy_sw, top_k=2))
        parts = [(0, 512), (512, 1024), (1024, 1536)]
        assert prefill._windows(parts, dummy_layer) == [[(0, 512)], [(512, 1024)], [(1024, 1536)]]
    finally:
        prefill.TURNS = saved_turns


def test_whole_model_together_ok_and_wired_while_generating(monkeypatch):
    """`WHOLE=True` enables `together_ok()` without a Demand driver; `_wired_while_generating()` wires Metal once
    when the first request enters and restores the prior limit only when the last request exits."""
    import weakref
    from quanteeze import demand, parallel, serve

    monkeypatch.setattr(demand, '_LIVE', weakref.WeakSet())
    monkeypatch.setattr(parallel, 'size', lambda: 1)
    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS_ANY', 'off')

    monkeypatch.setattr(request_threads, 'WHOLE', [True])
    assert request_threads.together_ok()
    monkeypatch.setattr(request_threads, 'WHOLE', [False])
    assert not request_threads.together_ok()
    monkeypatch.setattr(request_threads, 'WHOLE', [True])
    not_started = SimpleNamespace(_stm=None)
    monkeypatch.setattr(demand, '_LIVE', [not_started])
    assert not request_threads.together_ok()

    # Ref-counted Metal wired limit
    wired_calls = []
    monkeypatch.setattr(mx.metal, 'is_available', lambda: True)
    monkeypatch.setattr(mx, 'device_info', lambda: {'max_recommended_working_set_size': 123456})
    monkeypatch.setattr(mx, 'set_wired_limit', lambda v: wired_calls.append(v) or 999)
    monkeypatch.setattr(request_threads, '_WIRING', dict(generating=0, before=None))
    monkeypatch.setattr(request_threads, 'STATS', dict(request_threads.STATS, wired=0))
    monkeypatch.setattr(serve, '_WIRED', [0])

    with request_threads._wired_while_generating():
        assert request_threads._WIRING['generating'] == 1
        assert wired_calls == [123456]
        with request_threads._wired_while_generating():
            assert request_threads._WIRING['generating'] == 2
            assert wired_calls == [123456]
        assert request_threads._WIRING['generating'] == 1
        assert wired_calls == [123456]
    assert request_threads._WIRING['generating'] == 0
    assert wired_calls == [123456, 999]
    assert request_threads.STATS['wired'] == 1


def test_why_alone_says_what_keeps_requests_apart_and_state_tells_the_page(monkeypatch):
    """why_alone: together_ok's conditions in words for the home page, the first that fails; together_ok is that
    none does.  state() adds a drafter that drafts with its request alone (drafts_alone), for the page alone."""
    from quanteeze import kernel_checks, mtp
    monkeypatch.setattr(parallel, 'size', lambda: 1)
    monkeypatch.setattr(kernel_checks, 'shared', lambda: True)
    monkeypatch.setattr(demand, '_LIVE', set())
    monkeypatch.setattr(request_threads, 'WHOLE', [True])
    monkeypatch.setattr(mtp, 'ACTIVE', [None])
    assert request_threads.why_alone() is None and request_threads.together_ok()
    monkeypatch.setattr(request_threads, 'WHOLE', [False])
    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS_ANY', 'off')
    assert 'GGUF' in request_threads.why_alone() and not request_threads.together_ok()
    class Reader:                                         # (demand._LIVE is a set)
        _stm, _stm_why = None, 'a GGUF tier'
    reader = Reader()
    monkeypatch.setattr(demand, '_LIVE', {reader})
    assert request_threads.why_alone() == 'the C++ STM does not serve the on-demand reads (a GGUF tier)'
    reader._stm = SimpleNamespace(started=False)
    assert request_threads.why_alone() == "the C++ STM starts at the first request's pass"
    reader._stm.started = True
    assert request_threads.why_alone() is None and request_threads.together_ok()
    monkeypatch.setattr(kernel_checks, 'shared', lambda: False)
    assert 'C++ extension' in request_threads.why_alone() and not request_threads.together_ok()
    monkeypatch.setattr(kernel_checks, 'shared', lambda: True)
    monkeypatch.setattr(parallel, 'size', lambda: 2)
    assert 'several GPUs' in request_threads.why_alone() and not request_threads.together_ok()
    monkeypatch.setattr(parallel, 'size', lambda: 1)

    monkeypatch.setattr(request_threads, '_LOADED', dict(threads=3, any=False))   # as the load read it
    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS', '2')
    s = request_threads.state()
    assert s['set'] == 2 and s['loaded'] == 3 and s['why_alone'] is None
    assert {'running', 'together', 'alone', 'most_at_once'} <= set(s)
    monkeypatch.setattr(mtp, 'ACTIVE', [(object(), 'mtp')])
    assert request_threads.state()['why_alone'] == 'a drafter other than DFlash (MTP head) drafts with its request alone'
    monkeypatch.setattr(mtp, 'ACTIVE', [(object(), 'dflash')])
    assert request_threads.state()['why_alone'] is None
    monkeypatch.setattr(request_threads, '_LOADED', dict(threads=1, any=False))   # one loop: nothing to tell
    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS', '1')
    monkeypatch.setattr(parallel, 'size', lambda: 2)
    assert request_threads.state()['why_alone'] is None


def test_the_home_page_gets_requests_at_once(monkeypatch):
    """home.presets: QUANTEEZE_REQUEST_THREADS as set (read at the next load); home.live: state() once a model is
    loaded, nothing before."""
    from quanteeze import control, home
    monkeypatch.delenv('QUANTEEZE_REQUEST_THREADS', raising=False)
    assert home.presets()['request_threads'] == '1'                      # the default
    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS', '3')
    assert home.presets()['request_threads'] == '3'
    monkeypatch.setattr(control, '_LOADED', [None])
    assert 'request_threads' not in home.live()
    monkeypatch.setattr(control, '_LOADED', [object()])
    monkeypatch.setattr(request_threads, 'state', lambda: {'set': 3, 'loaded': 1, 'why_alone': None})
    assert home.live()['request_threads']['loaded'] == 1
