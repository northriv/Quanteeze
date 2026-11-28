# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Regression tests for reload snapshots, C++ check tables, NVRTC PCH lifetime,
multi-lane CPU expert mailboxes, per-thread pass stats, and kernel check hardening."""
import fcntl
import os
import socket
import tempfile
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import quanteeze
from quanteeze import (
    control,
    cpu_experts,
    cuda_attn,
    cuda_engram,
    cuda_kernels,
    cuda_ple,
    cuda_ple_layer,
    demand,
    dspark,
    engram_host,
    expert_qmv,
    kernel_checks,
    leases,
    native_ext,
    ngram_prefetch,
    parallel,
    per_load,
    per_thread,
    q2_experts,
    request_threads,
    serve,
    v41_attention,
    variables,
    vision_ssd,
)


def test_per_load_registration_idempotence_isolation_and_serve_cleanup(monkeypatch):
    registered = set(per_load.registered())
    expected_prefixes = (
        'quanteeze.leases.reset',
        'quanteeze.request_threads.reset',
        'quanteeze.engram_host._let_go_native',
        'quanteeze.engram_host._let_go',
        'quanteeze.cpu_experts._let_go_native',
        'quanteeze.dspark.let_go',
        'quanteeze.v41_attention.let_go',
        'quanteeze.vision_ssd.let_go',
        'quanteeze.parallel.forget_chain',
        'quanteeze.q2_experts.let_go',
        'quanteeze.expert_qmv._let_go',
        'quanteeze.cuda_attn._let_go',
        'quanteeze.kv_bits._let_go',
        'quanteeze.ngram_prefetch.let_go',
    )
    for name in expected_prefixes:
        assert name in registered

    monkeypatch.setattr(per_load, '_LET_GO', list(per_load._LET_GO))
    seen = []

    def _audit_dummy_cb():
        seen.append('first')

    per_load.on_let_go(_audit_dummy_cb)
    count_before = len(per_load._LET_GO)

    def _audit_dummy_cb():  # noqa: F811 -- intentional same qualname for idempotence
        seen.append('second')

    per_load.on_let_go(_audit_dummy_cb)
    assert len(per_load._LET_GO) == count_before

    def _audit_raising_cb():
        seen.append('boom')
        raise RuntimeError('cleanup failure must not stop subsequent callbacks')

    def _audit_after_cb():
        seen.append('after')

    per_load.on_let_go(_audit_raising_cb)
    per_load.on_let_go(_audit_after_cb)
    per_load.let_go()
    assert seen == ['second', 'boom', 'after']

    monkeypatch.setattr(expert_qmv, 'ROWS', [8])
    monkeypatch.setattr(cuda_attn, '_PROVEN', [('dummy', True)])
    request_threads._LOADED['threads'] = 4
    request_threads._LOADED['any'] = True
    engram_host._TABLES[1] = ('dummy', {})
    serve._let_go()
    assert expert_qmv.ROWS[0] is None
    assert cuda_attn._PROVEN[0] == (None, False)
    assert request_threads._LOADED['threads'] is None and request_threads._LOADED['any'] is False
    assert engram_host._TABLES == {}


def test_variables_snapshots_and_control_lifecycle(monkeypatch):
    monkeypatch.setenv('QUANTEEZE_MTP', 'on')
    monkeypatch.setenv('QUANTEEZE_CACHE_GB', '4')
    variables.take_start_snapshot()
    control.loaded()
    assert variables.at_load() is not None
    assert variables.at_start() is not None

    monkeypatch.setenv('QUANTEEZE_MTP', 'off')
    monkeypatch.setenv('QUANTEEZE_CACHE_GB', '9')
    assert variables.get('QUANTEEZE_MTP', variables.at_load()) == 'on'
    assert variables.get('QUANTEEZE_MTP') == 'off'
    assert variables.get('QUANTEEZE_CACHE_GB', variables.at_start()) == '4'
    assert variables.integer_or('QUANTEEZE_ADMIT_AHEAD', default=7, environ={'QUANTEEZE_ADMIT_AHEAD': 'off'}) == 7
    assert variables.integer_or('QUANTEEZE_ADMIT_AHEAD', default=7, environ={'QUANTEEZE_ADMIT_AHEAD': '12'}) == 12

    st = control.state()
    by_name = {row['name']: row for row in st['settings']}
    assert st['loaded'] is True
    assert by_name['QUANTEEZE_MTP']['in_effect'] == 'on'
    assert by_name['QUANTEEZE_MTP']['value'] == 'off'
    assert by_name['QUANTEEZE_MTP']['pending'] is True

    control.unloaded()
    assert variables.at_load() is None
    assert control.state()['loaded'] is False

    snap = {'QUANTEEZE_COMPACT': 'on'}
    assert variables.get('QUANTEEZE_MTP', snap) == 'off'
    assert variables.get('QUANTEEZE_VISION_SSD', snap) == 'on'
    snap_keep = {'QUANTEEZE_COMPACT': 'on', 'QUANTEEZE_COMPACT_KEEP': 'MTP', 'QUANTEEZE_MTP': 'on'}
    assert variables.get('QUANTEEZE_MTP', snap_keep) == 'on'


def test_kernel_checks_atomic_tables_same_bits_and_cap(monkeypatch):
    assert kernel_checks.shared() is True
    with monkeypatch.context() as mp:
        mp.setenv('QUANTEEZE_NATIVE', 'off')
        variables.take_load_snapshot()
        assert kernel_checks.shared() is False
    variables.take_load_snapshot()
    assert kernel_checks.shared() is True

    choice, checked = kernel_checks.tables('test_audit_dict_view')
    assert len(choice) == 0
    assert choice.get('k', 'miss') == 'miss'
    choice['k1'] = 'port_a'
    checked['k1'] = 2
    assert 'k1' in choice and choice['k1'] == 'port_a' and checked['k1'] == 2
    assert choice == {'k1': 'port_a'}
    assert choice.pop('k1') == 'port_a'
    assert 'k1' not in choice
    assert choice.pop('missing', 99) == 99
    choice['k2'] = True
    checked['k2'] = 1
    choice.clear()
    checked.clear()
    assert list(choice.items()) == [] and len(checked) == 0

    c_conc, k_conc = kernel_checks.tables('test_audit_concurrent')
    barrier = threading.Barrier(6)
    errs = []

    def worker(idx):
        try:
            barrier.wait(timeout=5.0)
            for step in range(4):
                ok = not (idx == 0 and step == 0)
                kernel_checks.counted(c_conc, k_conc, 'shape_x', ok, 3)
        except BaseException as exc:
            errs.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert not errs
    assert kernel_checks.read(c_conc, k_conc, 'shape_x') == (False, 3)
    assert kernel_checks.settled(c_conc, k_conc, 'shape_x', 3) is False
    assert kernel_checks.counted(c_conc, k_conc, 'shape_x', True, 3) == (False, 3)

    c_cap, k_cap = kernel_checks.tables('test_audit_cap', reference='fallback')
    monkeypatch.setattr(kernel_checks, 'SHAPES_MOST', 3)
    for i in range(3):
        assert kernel_checks.counted(c_cap, k_cap, ('s', i), 'fast', 2, reference='fallback') == ('fast', 1)
    assert c_cap.table.full(('s', 99)) is True
    assert kernel_checks.read(c_cap, k_cap, ('s', 99)) == ('fallback', kernel_checks._COUNT)
    assert len(c_cap.table.cells) == 3

    pos_zero = mx.array([0.0], dtype=mx.float32)
    neg_zero = mx.array([-0.0], dtype=mx.float32)
    assert not kernel_checks.same_bits(pos_zero, neg_zero)
    assert not kernel_checks.same_bits(pos_zero.astype(mx.bfloat16), neg_zero.astype(mx.bfloat16))
    nan_arr = mx.array([float('nan'), 1.0], dtype=mx.float32)
    assert kernel_checks.same_bits(nan_arr, mx.array([float('nan'), 1.0], dtype=mx.float32))
    assert not kernel_checks.same_bits(pos_zero.astype(mx.bfloat16), pos_zero.astype(mx.float16))
    assert not kernel_checks.same_bits(mx.zeros((1, 2), mx.float32), mx.zeros((2, 1), mx.float32))
    assert kernel_checks.same_bits((pos_zero, [nan_arr]), (mx.array([0.0], mx.float32), [nan_arr]))


def test_per_thread_counts_and_checked():
    counts = per_thread.Counts({'known': 0})
    counts['known'] = counts['known'] + 1
    n_threads = 6
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait(timeout=5.0)
        counts['unseen'] = counts.get('unseen', 0) + 1
        counts['known'] = counts['known'] + 2

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert counts['unseen'] == n_threads
    assert counts['known'] == 1 + 2 * n_threads

    plain_c, plain_k = {}, {}
    assert per_thread.checked(plain_c, plain_k, 'k', True, 2) is True
    assert per_thread.checked(plain_c, plain_k, 'k', False, 2) is False
    assert (plain_c['k'], plain_k['k']) == (False, 2)
    assert per_thread.checked(plain_c, plain_k, 'k', True, 2) is False

    tab_c, tab_k = kernel_checks.tables('test_audit_per_thread', reference='ref')
    assert per_thread.checked(tab_c, tab_k, 'k', 'fast', 2, reference='ref') == 'fast'
    assert per_thread.checked(tab_c, tab_k, 'k', False, 2, reference='ref') == 'ref'
    assert kernel_checks.read(tab_c, tab_k, 'k') == ('ref', 2)


def test_native_ext_cache_trim_room_and_load_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv('QUANTEEZE_NATIVE', 'auto')
    variables.take_load_snapshot()
    assert native_ext.mode() == 'auto'
    monkeypatch.setenv('QUANTEEZE_NATIVE', 'off')
    assert native_ext.mode() == 'auto'
    variables.take_load_snapshot()
    assert native_ext.mode() == 'off'
    assert native_ext.available() is False

    cache = tmp_path / 'native_cache'
    mod_dir = cache / 'mymod'
    newest = mod_dir / '0123456789abcdef'
    oldest = mod_dir / 'fedcba9876543210'
    newest.mkdir(parents=True)
    oldest.mkdir(parents=True)
    (newest / 'ext.so').write_bytes(b'x' * 40)
    (oldest / 'ext.so').write_bytes(b'y' * 40)
    (oldest / 'tmp_leftover').write_bytes(b'z' * 10)
    stale_tmp = cache / 'tmp_root_file'
    stale_tmp.write_bytes(b'w' * 10)

    now = time.time()
    very_old = now - 40 * 86400
    for p in (stale_tmp, oldest / 'tmp_leftover', oldest / 'ext.so', oldest, newest / 'ext.so', newest):
        os.utime(p, (very_old, very_old))
    os.utime(newest, (very_old + 100, very_old + 100))

    monkeypatch.setattr(native_ext, 'CACHE_KEEP_DAYS', 10)
    native_ext._trim(cache, now=now)
    assert newest.is_dir()
    assert not oldest.exists()
    assert not stale_tmp.exists()

    monkeypatch.setattr(native_ext, 'CACHE_MOST_BYTES', 30)
    assert native_ext._held(cache) == 40
    with pytest.raises(OSError, match='at most'):
        native_ext._room(cache)


def test_nvrtc_pch_dir_sweeps_dead_and_preserves_live(tmp_path, monkeypatch):
    if quanteeze.NVRTC_PCH[0] is None:
        assert quanteeze.nvrtc_pch_counts() is None
        assert quanteeze.nvrtc_pch_heap() is None
    else:
        counts = quanteeze.nvrtc_pch_counts()
        assert isinstance(counts, tuple) and len(counts) == 3 and all(isinstance(v, int) for v in counts)
        heap = quanteeze.nvrtc_pch_heap()
        assert isinstance(heap, tuple) and len(heap) == 3 and all(isinstance(v, int) for v in heap)

    class _CountsLib:
        def qz_nvrtc_pch_counts(self, arr):
            arr[0], arr[1], arr[2] = 5, 4, 2

        def qz_nvrtc_pch_heap(self, arr):
            arr[0], arr[1], arr[2] = 1, 402653184, 402653184

    with monkeypatch.context() as mp:
        mp.setattr(quanteeze, 'NVRTC_PCH', [_CountsLib()])
        assert quanteeze.nvrtc_pch_counts() == (5, 4, 2)
        assert quanteeze.nvrtc_pch_heap() == (1, 402653184, 402653184)

    host = socket.gethostname().split('.')[0]
    prefix = f'{quanteeze._PCH_PREFIX}{host}-'
    dead_dir = tmp_path / f'{prefix}dead-111'
    dead_dir.mkdir()
    (dead_dir / quanteeze._PCH_ALIVE).write_bytes(b'')
    (dead_dir / 'stale.pch').write_bytes(b'123')

    live_dir = tmp_path / f'{prefix}live-222'
    live_dir.mkdir()
    live_alive = live_dir / quanteeze._PCH_ALIVE
    live_fd = os.open(live_alive, os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(live_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(quanteeze, '_PCH_HELD', [])
    monkeypatch.setattr(quanteeze, 'NVRTC_PCH_DIR', [None])
    bound = []

    class _FakeLib:
        def qz_nvrtc_pch_dir(self, path_bytes):
            bound.append(path_bytes.decode())
            return 1

    try:
        made = quanteeze._pch_dir(_FakeLib())
        assert not dead_dir.exists()
        assert live_dir.is_dir()
        assert len(bound) == 1 and bound[0] == made
        assert os.path.isdir(made)
    finally:
        for fd in quanteeze._PCH_HELD:
            try:
                os.close(fd)
            except OSError:
                pass
        os.close(live_fd)


def test_cpu_experts_multi_lane_and_demand_cleanup():
    if cpu_experts.isa() is None:
        pytest.skip('no C++17 compiler here')
    rng = np.random.default_rng(7)
    w = mx.array(rng.normal(0, 0.05, (8, 32, 64)).astype(np.float32)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    mx.eval(q, s, b)
    arrs = []
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        for part, t in (('weight', q), ('scales', s), ('biases', b)):
            arr = np.array(mx.view(t, mx.uint16) if t.dtype == mx.bfloat16 else t)
            arrs.append((proj, part, np.ascontiguousarray(arr)))
    spec = cpu_experts.layer_spec(arrs, {'gate_proj': 64, 'up_proj': 64, 'down_proj': 64}, 'affine', 4)
    pool = cpu_experts.Pool(threads=1)
    server = cpu_experts.Server(pool, {0: spec}, 64, pinned=False, lanes=3)
    try:
        barrier = threading.Barrier(3)
        got_lanes = []
        lock = threading.Lock()

        def worker():
            lane = server.lane()
            with lock:
                got_lanes.append(lane)
            barrier.wait(timeout=5.0)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert sorted(got_lanes) == [0, 1, 2]

        reused = []
        t_reuse = threading.Thread(target=lambda: reused.append(server.lane()))
        t_reuse.start()
        t_reuse.join(timeout=5.0)
        assert reused[0] in (0, 1, 2)
        for lane in range(3):
            server._addr[(lane, 0)] = 1000 + lane
        assert 0 in server
        assert server.addr[0] in (1000, 1001, 1002)
    finally:
        server.close()
        pool.close()

    dem = object.__new__(demand.Demand)
    sw = SimpleNamespace(demand=dem)
    dem.switches = {0: sw}
    dem.workers = []
    dem._stm = None
    dem._cpu_status_lock = threading.Lock()
    dem._cpu_eval_tid = None
    dem._cpu = None
    dem._cpu_load = None
    dem._prefill_stage = {}
    dem._prefill_dev = []
    dem._stage_inflight = []
    dem._stage_ring = []
    dem._advised_views = []
    dem.base = SimpleNamespace(close=lambda: None)
    dem._stream_lock = threading.Lock()
    dem._stream_threads = set()
    dem._pool = SimpleNamespace(shutdown=lambda **kw: None)
    dem._native = None
    done_calls = []
    dem.thread_done = lambda: done_calls.append('thread_done')
    dem.close()
    assert sw.demand is None
    assert dem.switches == {}
    dem.pass_failed()
    assert done_calls == ['thread_done']


def test_request_threads_leases_and_parallel_forget(monkeypatch):
    monkeypatch.setattr(demand, '_LIVE', set())
    monkeypatch.setattr(request_threads, 'WHOLE', [False])
    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS_ANY', 'off')
    variables.take_load_snapshot()
    assert request_threads.together_ok() is False
    monkeypatch.setenv('QUANTEEZE_REQUEST_THREADS_ANY', 'on')
    variables.take_load_snapshot()
    assert request_threads.together_ok() is True

    with monkeypatch.context() as mp:
        mp.setattr(kernel_checks, 'shared', lambda: False)
        assert request_threads.together_ok() is False

    seen_calls = []

    class _FakeDemand:
        _stm = None

        def pass_failed(self):
            seen_calls.append('demand_failed')

    fake_demand = _FakeDemand()
    monkeypatch.setattr(demand, '_LIVE', {fake_demand})
    assert request_threads.together_ok() is False

    monkeypatch.setattr(request_threads, '_passes_done', lambda: seen_calls.append('done'))
    request_threads.passes_failed()
    assert seen_calls == ['demand_failed']

    with request_threads._SERVING_LOCK:
        request_threads._SERVING.add(threading.get_ident())
    with request_threads._RUNNING_LOCK:
        request_threads._RUNNING['u1'] = 'active'
    broken_batch = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError('close error')))
    request_threads._served_ends('u1', broken_batch, {'u1'}, False, False)
    with request_threads._SERVING_LOCK:
        assert threading.get_ident() not in request_threads._SERVING
    with request_threads._RUNNING_LOCK:
        assert 'u1' not in request_threads._RUNNING
    assert seen_calls == ['demand_failed', 'done']
    per_thread.PASS.drafted = 'dirty'
    per_thread.PASS.__init__()
    assert per_thread.PASS.drafted is None

    leases._ROOM[0] = 123
    leases._SEEDED[0] = 456
    assert leases.seeded_room() == 456
    leases.reset()
    assert leases.seeded_room() is None
    assert leases._ROOM[0] is None

    parallel._LAST[0] = 'chain_sentinel'
    parallel.forget_chain()
    assert parallel._LAST[0] is None

    parallel._LAST[0] = 'chain_sentinel'

    def _boom():
        raise ValueError('pass exploded')

    with pytest.raises(ValueError, match='pass exploded'):
        parallel._forgetting(_boom)()
    assert parallel._LAST[0] is None


def test_dspark_log_cap_ple_put_off_r5_keys_and_double_install(tmp_path, monkeypatch):
    log_path = tmp_path / 'dspark.log'
    monkeypatch.setattr(dspark, 'LOG_KEPT_BYTES', 32)
    monkeypatch.setattr(dspark, 'LOG_WRITTEN_BYTES', 32)
    monkeypatch.setitem(dspark.LOG_BYTES, 'file', 0)
    monkeypatch.setitem(dspark.LOG_BYTES, 'written', 0)
    monkeypatch.setitem(dspark.LOG_BYTES, 'said', False)
    f = dspark._open_log(log_path)
    assert f is not False
    dspark.LOG[0] = f
    assert dspark._log_full(16) is False
    assert dspark._log_full(64) is True
    assert dspark.LOG[0] is False

    before_put_off = cuda_ple_layer.STATS['put_off']
    monkeypatch.setattr(cuda_ple_layer, 'applies', lambda m, h, c, mask: True)
    monkeypatch.setattr(cuda_ple, '_while_alone', lambda fn: False)
    fake_mod = SimpleNamespace(
        key_proj=SimpleNamespace(bits=4, group_size=32),
        up_proj=None,
    )
    fake_fwd = SimpleNamespace(_ple=lambda m, h, ids, c, mask: h * 2)
    out = cuda_ple_layer.added(fake_fwd, fake_mod, mx.ones((1, 1, 4), mx.bfloat16), mx.array([[1]]), [None, None, mx.zeros((1, 1, 4))], None)
    assert cuda_ple_layer.STATS['put_off'] == before_put_off + 1
    assert out.shape == (1, 1, 4)

    # R5 key invariance: cuda_ple.lookup & cuda_engram.lookup ignore row_ids.size / ids.size
    ple_c, ple_k = kernel_checks.tables('test_r5_ple')
    eng_c, eng_k = kernel_checks.tables('test_r5_engram')
    monkeypatch.setattr(cuda_ple, 'CHOICE', ple_c)
    monkeypatch.setattr(cuda_ple, 'CHECKED', ple_k)
    monkeypatch.setattr(cuda_engram, 'CHOICE', eng_c)
    monkeypatch.setattr(cuda_engram, 'CHECKED', eng_k)
    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
    monkeypatch.setattr(cuda_ple, 'ON', [True])

    fake_table = cuda_ple.Table(starts=None, pointers=None, shards=2, group_size=32, strides=None)
    monkeypatch.setattr(cuda_ple, '_table', lambda emb: fake_table)
    monkeypatch.setattr(cuda_ple, 'rows', lambda emb, tbl, rids: mx.ones((rids.size, 8), mx.bfloat16))
    emb = SimpleNamespace(row_width=8)
    own_ple = lambda e, rids: mx.ones((rids.size, 8), mx.bfloat16)
    for _ in range(cuda_ple.CHECKS):
        cuda_ple.lookup(emb, mx.array([0, 1]), own_ple)
    served_before = cuda_ple.STATS['served']
    got_ple = cuda_ple.lookup(emb, mx.array([0, 1, 2, 3, 4]), own_ple)
    assert got_ple is not None and cuda_ple.STATS['served'] == served_before + 1
    assert list(ple_c.keys()) == [(2, 8, 32)]

    monkeypatch.setattr(cuda_engram, '_table', lambda emb: ('engram', 8))
    monkeypatch.setattr(cuda_engram, 'rows', lambda tbl, ids: mx.ones((ids.size, 8), mx.bfloat16))
    own_eng = lambda e, ids: mx.ones((ids.size, 8), mx.bfloat16)
    for _ in range(cuda_engram.CHECKS):
        cuda_engram.lookup(object(), mx.array([0, 1]), own_eng)
    eng_before = cuda_engram.STATS['engram']
    got_eng = cuda_engram.lookup(object(), mx.array([0, 1, 2, 3]), own_eng)
    assert got_eng is not None and cuda_engram.STATS['engram'] == eng_before + 1
    assert list(eng_c.keys()) == [('engram', 8)]

    class DummyMMap:
        def __call__(self, row_ids):
            return row_ids

    class DummyMod:
        def __call__(self, input_ids, cache):
            return input_ids

    cuda_ple.install(DummyMMap, DummyMod)
    first_call = DummyMod.__call__
    first_own = DummyMod._quanteeze_cuda_ple_own
    cuda_ple.install(DummyMMap, DummyMod)
    assert DummyMod.__call__ is first_call
    assert DummyMod._quanteeze_cuda_ple_own is first_own


def test_same_bits_each_batch_comparison():
    pos = mx.array([0.0, 1.5], dtype=mx.float32)
    neg = mx.array([-0.0, 1.5], dtype=mx.float32)
    nan_a = mx.array([float('nan'), 2.0], dtype=mx.bfloat16)
    nan_b = mx.array([float('nan'), 2.0], dtype=mx.bfloat16)
    assert kernel_checks.same_bits_each([]) == []
    got = kernel_checks.same_bits_each([
        (pos, mx.array([0.0, 1.5], dtype=mx.float32)),
        (pos, neg),
        (nan_a, nan_b),
        (pos, pos.astype(mx.float16)),
        (mx.zeros((1, 2), mx.int32), mx.zeros((2, 1), mx.int32)),
    ])
    assert got == [True, False, True, False, False]


def test_tables_room_split_reads_host_room_once_and_keeper_bound(tmp_path, monkeypatch):
    from quanteeze import fitting, home

    ckpt = tmp_path / 'fake_ckpt'
    ckpt.mkdir()
    reads = []

    def fake_host_room(*_args):
        reads.append(1)
        # Return 50 GB on first call and 100 GB if accidentally called a second time
        return int(50e9) if len(reads) == 1 else int(100e9)

    monkeypatch.setattr(fitting, 'host_room', fake_host_room)
    monkeypatch.setattr(parallel, 'runs_here', lambda: [range(0, 2), range(2, 4)])
    monkeypatch.setattr(parallel, 'owned', lambda: range(0, 2))
    monkeypatch.setattr(home, 'layer_experts_bytes', lambda _p: {0: int(4e9), 1: int(4e9), 2: int(5e9), 3: int(5e9)})
    monkeypatch.setattr(serve, '_KEEPER', [None])

    room = serve._tables_room(ckpt)
    assert len(reads) == 1
    # most = (5 + 5 + 2*5 + 3) GB = 23 GB; table = 2 * (50 - 23) = 54 GB
    assert room == int(54e9)

    # Now bound this process with QUANTEEZE_KEEP_GB leaving 30 GB (< share=50 GB)
    reads.clear()
    fake_keeper = SimpleNamespace(limit=int(40e9))
    monkeypatch.setattr(serve, '_KEEPER', [fake_keeper])
    monkeypatch.setattr(serve.guards.Readings, 'footprint', lambda self: int(10e9))
    bounded = serve._tables_room(ckpt)
    assert len(reads) == 1
    # mine = (4 + 4 + 2*4 + 3) GB = 19 GB; left - mine = 30 - 19 = 11 GB
    assert bounded == int(11e9)


def test_dsml_turn_starts_prompt_checkpoints_and_cancelled_prefill_kept(monkeypatch):
    from quanteeze import dsml

    class FakeTok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, tok):
            return {dsml.USER_TOKEN: 777, dsml.IMAGE_TOKEN: 0}.get(tok, 0)

    class FakeProc:
        tokenizer = FakeTok()

    lm = SimpleNamespace()
    model = SimpleNamespace(config=SimpleNamespace(model_type='deepseek_v41'), language_model=lm)
    dsml.use_model(model, FakeProc())
    assert serve.turn_starts(model) == frozenset({777})
    assert serve.turn_starts(lm) == frozenset({777})
    serve.mark(model, serve.PROMPT_STEP, 64)
    assert serve._attribute(lm, serve.PROMPT_STEP) == 64

    class FakeCoord:
        def __init__(self, mdl):
            self.model = mdl
            self.manager = SimpleNamespace(
                checkpoint_interval_tokens=16,
                block_size=8,
                _exact_cache_max=4,
                disk=True,
                exact_cache_min_tokens=8,
            )

        def checkpoint_len(self, token_ids, _media):
            return len(token_ids) - 1

    serve._checkpoints_a_prompt(FakeCoord)
    # 150 tokens -> final = 149; last interval = 144; PROMPT_STEP multiples <= 133: 64, 128;
    # user turns at 20, 50, 95, 140 -> last two in [16, 133] are 95 and 50
    toks = [10] * 150
    for pos in (20, 50, 95, 140):
        toks[pos] = 777
    got_lengths = FakeCoord(lm).checkpoint_lengths(toks, frozenset())
    assert got_lengths == [50, 64, 95, 128, 144, 149]

    stored_calls = []

    class FakePrefillBatch:
        def __init__(self):
            self.uids = ['u0']
            self.model = SimpleNamespace(config=None)
            self.prompt_cache = ['dummy_cache']
            self._apc_manager = SimpleNamespace(exact_cache_min_tokens=8)
            self._apc_coordinator = SimpleNamespace(
                store_checkpoint=lambda prefix, cache, batch_idx=0, extra_hash=0: (
                    stored_calls.append(( tuple(prefix), batch_idx, extra_hash )) or True
                )
            )
            self._apc_meta = [{
                'full_input_ids': list(range(100)),
                'checkpoint_lengths': [64, 99],
                'prefix_len': 16,
                'checkpoint_stored': 0,
                'extra_hash': 5,
            }]
            self._right_pad_per_row = None
            self._processed_prompt_columns = 40
            self._suffix_lens = [84]

        def _apc_uses_checkpoints(self):
            return True

        def _row_real_tokens_processed(self, index):
            assert index == 0
            return 48

        def remove(self, uid):
            return f'removed:{uid}'

    serve._cut_prefills_kept(FakePrefillBatch)
    fb = FakePrefillBatch()
    assert fb.remove('u0') == 'removed:u0'
    assert stored_calls == [(tuple(range(48)), 0, 5)]
    assert fb._apc_meta[0]['checkpoint_stored'] == 48


def test_mtp_structured_output_undrafted_and_held_during_drafting(monkeypatch):
    from quanteeze import mtp, stream

    class FakeGen:
        def _make_logits_processors(self, args, input_ids=None):
            return list(args.logits_processors) if args.logits_processors is not None else None

        def generate(self, prompt, images=None, audio=None, args=None, videos=None):
            return SimpleNamespace(uid='u0'), iter(['tok'])

    mtp.install(FakeGen)
    fg = FakeGen()
    monkeypatch.setattr(mtp, 'ACTIVE', [('fake_drafter', 'mtp')])

    args = SimpleNamespace(logits_processors=['json_schema_proc'], thinking_budget=128)
    _ctx, toks = fg.generate('hello', args=args)
    assert list(toks) == ['tok']
    assert args.logits_processors is None
    assert args.thinking_budget is None
    procs = fg._make_logits_processors(args)
    assert isinstance(procs, mtp.Structured) and list(procs) == ['json_schema_proc']

    seen_states = []

    class FakeBatchGen:
        def __init__(self):
            self.draft_model = 'drafter_obj'
            self.draft_kind = 'mtp'
            self._prompt_batch = None
            self._generation_batch = []
            self.prefill_batch_size = 1
            self._unprocessed_sequences = []

        def _next(self, **kwargs):
            seen_states.append((
                self.draft_model,
                [s[0] for s in self._unprocessed_sequences],
            ))
            return 'stepped'

    mtp.install_batches(FakeBatchGen)
    bg = FakeBatchGen()
    s_row = ('u_struct', [1, 2, 3], 16, {}, procs)
    plain_row = ('u_plain', [1, 2], 16, {}, None)

    # 1. Batch starting with a structured row runs undrafted
    bg._unprocessed_sequences = [s_row]
    bg._next()
    assert seen_states[-1] == (None, ['u_struct'])

    # 2. While a drafting batch is active, structured rows are held back and restored afterward
    bg.draft_model, bg.draft_kind = 'drafter_obj', 'mtp'
    bg._generation_batch = ['active_draft_row']
    bg._unprocessed_sequences = [s_row, plain_row]
    bg._next()
    assert seen_states[-1] == ('drafter_obj', ['u_plain'])
    assert [s[0] for s in bg._unprocessed_sequences] == ['u_plain', 'u_struct']


def test_cpp_drafts_counters_across_threads_and_v41_attention_cuda_bindings(monkeypatch):
    import copy
    from mlx_vlm.speculative import mtp as spec_mtp
    from quanteeze import cuda_v41, mtp

    class PlainDrafter:
        def __init__(self):
            self.accept_lens = []
            self.draft_lens = []
            self.speculative_total_rounds = 0
            self.speculative_total_drafted = 0
            self.speculative_total_accepted = 0.0

    base_drafter = PlainDrafter()
    monkeypatch.setattr(mtp, 'ACTIVE', [None])
    mtp.enable(base_drafter, 'mtp')
    try:
        assert mtp._drafts() is not None
        assert mtp.counted(base_drafter) == (0, 0, 0.0)

        def run_copy():
            clone = copy.copy(base_drafter)
            clone.accept_lens, clone.draft_lens = [], []
            spec_mtp._record_speculative_round(clone, 2.5, 4)

        threads = [threading.Thread(target=run_copy) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Base drafter's own attributes stayed 0 because each thread mutated a shallow copy,
        # while C++ `_drafts` accumulated all 4 rounds across threads!
        assert base_drafter.speculative_total_rounds == 0
        assert mtp.counted(base_drafter) == (4, 16, 10.0)
        s = mtp.summary(base_drafter)
        assert s['rounds'] == 4 and s['drafted'] == 16 and s['accepted'] == 10
        assert s['acceptance'] == pytest.approx(0.625)

        # Drafter marked `_quanteeze_counts_itself` is not double-counted by `_record_speculative_round`
        class SelfCountingDrafter(PlainDrafter):
            _quanteeze_counts_itself = True

        spec_mtp._record_speculative_round(SelfCountingDrafter(), 3.0, 5)
        assert mtp.counted(base_drafter) == (4, 16, 10.0)

        # Re-enabling resets C++ counters to 0
        mtp.enable(base_drafter, 'mtp')
        assert mtp.counted(base_drafter) == (0, 0, 0.0)

        # Fallback under QUANTEEZE_NATIVE=off reads drafter attributes directly
        with monkeypatch.context() as mp:
            mp.setenv('QUANTEEZE_NATIVE', 'off')
            variables.take_load_snapshot()
            base_drafter.speculative_total_rounds = 2
            base_drafter.speculative_total_drafted = 6
            base_drafter.speculative_total_accepted = 4.0
            assert mtp.counted(base_drafter) == (2, 6, 4.0)
    finally:
        mtp.disable()

    ext = native_ext.extension('v41_attention', with_mlx=True, with_stm=False, force=True)
    assert ext is not None
    for sym in ('CudaKernel', 'cuda_kernel_spec', 'layer_on_cuda', 'fq8', 'front', 'back'):
        assert hasattr(ext, sym), sym
    kspec = ext.cuda_kernel_spec(
        'quanteeze_v41_fp8_roundtrip', ['x'], ['out'], cuda_kernels.HEADER, cuda_v41.FP8_SOURCE, 0
    )
    assert type(kspec).__name__ == 'CudaKernel'
    assert hasattr(cuda_v41, 'cpp_kernels') and hasattr(cuda_v41, 'install_attention')


def test_apc_none_kept_caps_reserve_unless_user_set(monkeypatch):
    class Manager:
        def __init__(self):
            self.memory_max_bytes = 4 << 30
            self.memory_reserve_bytes = 11 << 30

    serve._apc_seen(Manager)
    monkeypatch.setitem(serve._APC, 'given', False)
    monkeypatch.setitem(serve._APC, 'off_gpu', False)
    monkeypatch.setenv('APC_MEMORY_MAX_GB', '4')
    monkeypatch.delenv('APC_MEMORY_RESERVE_GB', raising=False)
    monkeypatch.setattr(serve.mx, 'default_device', lambda: serve.mx.gpu)
    before = Manager()
    assert serve._apc_off_the_gpu()
    after = Manager()
    for m in (before, after):
        assert m.memory_max_bytes == 0
        assert m.memory_reserve_bytes == serve.APC_RESERVE_NONE_KEPT == 1 << 30
    monkeypatch.setenv('APC_MEMORY_RESERVE_GB', '11')
    assert Manager().memory_reserve_bytes == 11 << 30
    small = Manager()
    small.memory_reserve_bytes = 1 << 29
    monkeypatch.delenv('APC_MEMORY_RESERVE_GB')
    serve._apc_none_kept(small)
    assert small.memory_reserve_bytes == 1 << 29


def _generation_loops():
    import inspect
    from mlx_vlm.server import generation
    run = generation.ResponseGenerator._run_impl
    original = inspect.getclosurevars(run).nonlocals.get('original', run)
    return [('mlx-vlm', original), ('request threads', lambda self: request_threads._dispatch(self, 2))]


@pytest.mark.parametrize('loop', range(2))
def test_load_refused_reaches_wait_until_ready_in_both_generation_loops(monkeypatch, loop):
    from mlx_vlm.server import generation
    from quanteeze import progress

    def broken(*args, **kwargs):
        raise SystemExit('quanteeze: QUANTEEZE_BITS=9: refused')

    monkeypatch.setattr(serve, '_load_serving', broken)
    name, run = _generation_loops()[loop]
    gen = type('Gen', (), {})()
    gen._ready, gen._load_error = threading.Event(), None
    gen._initialize_model = lambda: serve._load_hooked('/m')
    t = threading.Thread(target=run, args=(gen,), daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), name
    with pytest.raises(serve.LoadRefused, match='QUANTEEZE_BITS=9: refused') as caught:
        generation.ResponseGenerator.wait_until_ready(gen, timeout=1)
    assert isinstance(caught.value.__cause__, SystemExit)
    assert 'refused' in progress.state()['error']


def test_reask_continues_premature_stop_inside_thinking():
    import copy
    from quanteeze import phases, reask

    calls = []

    class Chunk:
        def __init__(self, text, finish_reason=None, token_count=1):
            self.text = text
            self.finish_reason = finish_reason
            self.token_count = token_count

    class FakeGen:
        processor = None

        def generate(self, prompt, images=None, audio=None, args=None, videos=None):
            calls.append((prompt, images, copy.copy(args)))
            if len(calls) == 1:
                return SimpleNamespace(uid='r1'), iter([
                    Chunk('Let me inspect the file. '),
                    Chunk('Wait, ', finish_reason='stop', token_count=2),
                ])
            return SimpleNamespace(uid='r2'), iter([
                Chunk('found the bug.</think>\nFixed it.', finish_reason='stop', token_count=4),
            ])

    reask.install(FakeGen)
    gen = FakeGen()
    args = SimpleNamespace(temperature=0.6, top_p=0.9, seed=10, max_tokens=100, thinking_budget=None)
    _, stream_iter = gen.generate('<|im_start|>assistant\n<think>\n', images=['img0'], args=args)
    items = list(stream_iter)

    assert len(calls) == 2
    assert calls[1][0] == '<|im_start|>assistant\n<think>\nLet me inspect the file. Wait, '
    assert calls[1][1] == ['img0']
    assert calls[1][2].temperature == reask.RETRY_TEMPERATURE
    assert calls[1][2].seed == 10 + reask.SEED_STEP
    assert calls[1][2].max_tokens == 97
    assert getattr(calls[1][2], '_quanteeze_in_thinking', False) is True
    assert phases._opens_thinking(gen, SimpleNamespace(args=calls[1][2], raw_inputs={})) is True
    assert ''.join(i.text for i in items) == 'Let me inspect the file. Wait, found the bug.</think>\nFixed it.'
    assert [i.finish_reason for i in items] == [None, None, 'stop']

    # `finish_reason == 'length'` inside `<think>` must NOT trigger continuation
    class LengthGen:
        processor = None

        def generate(self, prompt, images=None, audio=None, args=None, videos=None):
            return SimpleNamespace(uid='r_len'), iter([Chunk('still thinking', finish_reason='length')])

    reask.install(LengthGen)
    _, len_iter = LengthGen().generate('<|im_start|>assistant\n<think>\n', args=SimpleNamespace(max_tokens=10))
    len_items = list(len_iter)
    assert len(len_items) == 1 and len_items[0].finish_reason == 'length'

    # Complete `</tool_call>` inside unclosed `<think>` must NOT trigger continuation
    tool_calls = []

    class ToolInThinkGen:
        processor = None

        def generate(self, prompt, images=None, audio=None, args=None, videos=None):
            tool_calls.append(prompt)
            return SimpleNamespace(uid='r_tool'), iter([
                Chunk('considering tool\n'),
                Chunk('<tool_call>\n{"name": "shell"}\n</tool_call>', finish_reason='stop', token_count=4),
            ])

    reask.install(ToolInThinkGen)
    _, tool_iter = ToolInThinkGen().generate('<|im_start|>assistant\n<think>\n', args=SimpleNamespace(max_tokens=100))
    tool_items = list(tool_iter)
    assert len(tool_calls) == 1
    assert [i.finish_reason for i in tool_items] == [None, 'stop']

    # Repeated end-in-thinking on retry #2 closes `</think>\n\n` and leaves thinking
    force_calls = []

    class TwoStepThinkGen:
        processor = None

        def generate(self, prompt, images=None, audio=None, args=None, videos=None):
            force_calls.append((prompt, copy.copy(args)))
            if len(force_calls) == 1:
                return SimpleNamespace(uid='r_fc1'), iter([Chunk('thought 1.', finish_reason='stop')])
            if len(force_calls) == 2:
                return SimpleNamespace(uid='r_fc2'), iter([Chunk('still stuck', finish_reason='stop')])
            return SimpleNamespace(uid='r_fc3'), iter([Chunk('Final answer.', finish_reason='stop')])

    reask.install(TwoStepThinkGen)
    _, fc_iter = TwoStepThinkGen().generate('<|im_start|>assistant\n<think>\n', args=SimpleNamespace(max_tokens=100))
    fc_items = list(fc_iter)
    assert len(force_calls) == 3
    assert getattr(force_calls[1][1], '_quanteeze_in_thinking', False) is True
    assert getattr(force_calls[2][1], '_quanteeze_in_thinking', False) is False
    assert force_calls[2][0].endswith('thought 1.still stuck\n</think>\n\n')
    assert ''.join(i.text for i in fc_items) == 'thought 1.still stuck\n</think>\n\nFinal answer.'
    assert [i.finish_reason for i in fc_items] == [None, None, None, 'stop']

    # `phases.wrote` transitions `'thinking'` -> `'tool'` when `<tool_call>` appears without `</think>`
    entry = phases.begin('reask_tool_phase', 100, thinking=True)
    phases.prompt_read(entry, 0)
    assert entry['phase'] == 'thinking'
    phases.wrote(entry, '<tool_call>\n{"name": "shell"}\n</tool_call>', 5, 'stop')
    assert entry['phase'] == 'tool'
    ended = next(r for r in phases.state()['ended'] if r['id'] == 'reask_tool_phase')
    assert ended['how'] == 'tool'


def test_multimodal_cancelled_prefill_and_long_prompt_checkpoints(monkeypatch):
    from quanteeze import apc_images

    # 1. Cancelled multimodal prefill before a trailing image keeps already-read prefix when apc_images has a plan
    apc_images._note(77, 11, (101,))
    monkeypatch.setattr(apc_images, 'INSTALLED', [True])
    stored = []

    full_ids = [10] * 40000
    for idx in range(35000, 35100):
        full_ids[idx] = 999  # trailing image span near end of prompt

    class FakeMultimodalBatch:
        def __init__(self, read_pos):
            self._read_pos = read_pos
            self.uids = ['u_mm']
            self.model = SimpleNamespace(config=SimpleNamespace(image_token_id=999))
            self.prompt_cache = ['kv']
            self._apc_manager = SimpleNamespace(exact_cache_min_tokens=16)
            self._apc_coordinator = SimpleNamespace(
                store_checkpoint=lambda prefix, cache, batch_idx=0, extra_hash=0: (
                    stored.append((len(prefix), extra_hash)) or True
                )
            )
            self._apc_meta = [{
                'full_input_ids': full_ids,
                'checkpoint_lengths': [38912, 39999],
                'prefix_len': 0,
                'checkpoint_stored': 0,
                'extra_hash': 77,
            }]
            self._right_pad_per_row = None
            self._processed_prompt_columns = read_pos
            self._suffix_lens = [len(full_ids)]

        def _apc_uses_checkpoints(self):
            return True

        def _row_real_tokens_processed(self, index):
            return self._read_pos

        def remove(self, uid):
            return uid

    serve._cut_prefills_kept(FakeMultimodalBatch)
    # Cancelled at 16,384 tokens (before trailing image at 35,000): stored!
    b_ok = FakeMultimodalBatch(16384)
    b_ok.remove('u_mm')
    assert stored == [(16384, 77)]

    # Cancelled at 35,050 (strictly inside image span 35,000..35,100): rejected so image span is never split
    stored.clear()
    b_mid = FakeMultimodalBatch(35050)
    b_mid.remove('u_mm')
    assert stored == []

    # 2. Long in-memory prompt schedules 16,384-step checkpoints + 2-token ChatML `<|im_start|>user` boundaries
    class ChatMLTok:
        unk_token_id = 0

        def encode(self, text, add_special_tokens=False):
            if text == '<|im_start|>user':
                return [151644, 872]
            return [0]

    lm = SimpleNamespace()
    mdl = SimpleNamespace(language_model=lm)
    serve._mark_turn_starts(mdl, SimpleNamespace(tokenizer=ChatMLTok()))
    assert (151644, 872) in serve.turn_starts(lm)

    class Coord:
        def __init__(self):
            self.model = lm
            self.manager = SimpleNamespace(
                checkpoint_interval_tokens=2048,
                block_size=16,
                _exact_cache_max=32,
                disk=True,
                exact_cache_min_tokens=16,
            )

        def checkpoint_len(self, token_ids, _media):
            return len(token_ids) - 1

    serve._checkpoints_a_prompt(Coord)
    # Place user turn `<|im_start|>user` at 25000 even with trailing image at 35000..35100
    full_ids[25000], full_ids[25001] = 151644, 872
    got = Coord().checkpoint_lengths(full_ids, {999})
    assert 16384 in got
    assert 25000 in got
    assert 32768 in got
    assert got[-1] == 39999


def test_mtp_live_fields_and_ui_wording(monkeypatch):
    from pathlib import Path
    from quanteeze import home, mtp

    class BundledHead:
        speculative_total_rounds = 10
        speculative_total_drafted = 20
        speculative_total_accepted = 9.0

        def named_modules(self):
            return [('proj', SimpleNamespace(bits=4))]

        def parameters(self):
            return {}

    class Eagle3Draft:
        pass

    assert mtp.name(BundledHead()) == 'MTP head'
    assert mtp.name(Eagle3Draft()) == 'EAGLE-3'

    monkeypatch.setattr(mtp, 'ACTIVE', [(BundledHead(), 'mtp')])
    monkeypatch.setattr(mtp, '_drafts', lambda: None)
    live = home.live()
    assert live['mtp']['drafter'] == 'MTP head'
    assert live['mtp']['kind'] == 'mtp'
    assert live['mtp']['bits'] == '4'

    pkg_dir = Path(home.__file__).resolve().parent
    home_html = (pkg_dir / 'home.html').read_text(encoding='utf-8')
    control_html = (pkg_dir / 'control.html').read_text(encoding='utf-8')
    assert 'Speculative decoding (${kind})' in home_html
    assert '均一' not in home_html
    assert 'uniform precision' not in home_html.lower()
    assert 'uniform expert' not in control_html.lower()


def test_cache_limit_cuda_floor_and_metal_share(monkeypatch):
    """`serve._cache_limit()` enforces `CACHE_LEAST_ON_CUDA` (1 GB) on CUDA and `CACHE_SHARE` on Metal."""
    from quanteeze import guards, serve

    monkeypatch.setattr(serve.mx, 'default_device', lambda: serve.mx.gpu)
    monkeypatch.delenv('QUANTEEZE_CACHE_GB', raising=False)

    monkeypatch.setattr(serve.cuda_kernels, 'available', lambda: True)
    monkeypatch.setattr(guards, 'recommended_budget', lambda: 15.5e9)
    assert serve._cache_limit() == serve.CACHE_LEAST_ON_CUDA

    monkeypatch.setattr(guards, 'recommended_budget', lambda: 40e9)
    assert serve._cache_limit() == int(serve.CACHE_SHARE * 40e9)

    monkeypatch.setattr(guards, 'recommended_budget', lambda: 80e9)
    assert serve._cache_limit() == int(2e9)

    monkeypatch.setattr(serve.cuda_kernels, 'available', lambda: False)
    monkeypatch.setattr(guards, 'recommended_budget', lambda: 15.5e9)
    assert serve._cache_limit() == int(serve.CACHE_SHARE * 15.5e9)

    monkeypatch.setenv('QUANTEEZE_CACHE_GB', '4')
    assert serve._cache_limit() == int(4e9)
