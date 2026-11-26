# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
import os
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from quanteeze import (
    admission,
    apc_disk,
    context_compress,
    fitting,
    guards,
    home,
    leases,
    parallel,
    serve,
    stm_core,
)


def _restore_env(saved):
    for key, prev in saved.items():
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def test_strain_and_keeper_pressure_and_intake_hold():
    saved = {
        'QUANTEEZE_KEEP_PRESSURE': os.environ.get('QUANTEEZE_KEEP_PRESSURE'),
        'QUANTEEZE_KEEP_STRAIN': os.environ.get('QUANTEEZE_KEEP_STRAIN'),
        'QUANTEEZE_KEEP_CUT_GB': os.environ.get('QUANTEEZE_KEEP_CUT_GB'),
    }
    ticks = [0.0]
    states = iter([
        {'memory_psi': 5.0, 'io_psi': 0.0, 'available_gb': 10.0, 'pswpout': 0, 'allocstall_normal': 0, 'allocstall_movable': 0},
        {'memory_psi': 15.0, 'io_psi': 0.0, 'available_gb': 10.0, 'pswpout': 1000, 'allocstall_normal': 0, 'allocstall_movable': 0},
        {'memory_psi': 20.0, 'io_psi': 0.0, 'available_gb': 10.0, 'pswpout': 1000, 'allocstall_normal': 0, 'allocstall_movable': 0},
        {'memory_psi': 0.0, 'io_psi': 0.0, 'available_gb': 12.0, 'pswpout': 1000, 'allocstall_normal': 0, 'allocstall_movable': 0},
        {'memory_psi': 0.0, 'io_psi': 0.0, 'available_gb': 12.0, 'pswpout': 1000, 'allocstall_normal': 0, 'allocstall_movable': 0},
        {'memory_psi': 0.0, 'io_psi': 0.0, 'available_gb': 1.0, 'pswpout': 1000, 'allocstall_normal': 0, 'allocstall_movable': 0},
    ])

    def clock():
        ticks[0] += 2.0
        return ticks[0]

    try:
        strain = guards.Strain(
            marks={'memory_psi': 10.0},
            polls=2,
            calm=2,
            floor_gb=2.0,
            state=lambda: next(states),
            clock=clock,
        )
        strained, r1 = strain.sample()
        assert strained is False and r1['became'] is False
        strained, r2 = strain.sample()
        assert strained is False and r2['became'] is False
        strained, r3 = strain.sample()
        assert strained is True and r3['became'] is True and strain.times == 1
        strained, _ = strain.sample()
        assert strained is True
        strained, _ = strain.sample()
        assert strained is False and strain.seconds == pytest.approx(4.0)

        # Available-memory floor triggers after `polls` readings too.
        strain2 = guards.Strain(
            marks={'memory_psi': 10.0},
            polls=1,
            calm=1,
            floor_gb=2.0,
            state=lambda: {'memory_psi': 0.0, 'io_psi': 0.0, 'available_gb': 1.5},
            clock=clock,
        )
        s_floor, r_floor = strain2.sample()
        assert s_floor is True and r_floor['became'] is True

        pressure_level = [1]
        logs = []
        sheds = []
        relieved = []

        class FakeReadings:
            def mlx_held(self):
                return 0.5e9

            def swap_used(self):
                return 0

            def swap_disk_free(self):
                return 50e9

            def swap_free(self):
                return 100e9

            def footprint(self):
                return 1e9

            def pressure(self):
                return pressure_level[0]

            def others(self, floor=0):
                return [(4e9, 4321, 'other-proc')]

            def memory_available(self):
                return 20e9

        class FakeStrain:
            def __init__(self):
                self.results = iter([(True, {'became': True, 'memory_psi': 25.0}), (False, {'became': False, 'memory_psi': 1.0})])

            def sample(self):
                return next(self.results)

            def summary(self):
                return 'the host strained 1 times, 2 s in all'

        keeper = guards.Keeper(
            limit=100e9,
            readings=FakeReadings(),
            shed=lambda: sheds.append(True),
            log=logs.append,
            system=True,
            strain=FakeStrain(),
            relieve=relieved.append,
        )

        # Critical pressure warns, sheds buffers, reports top other process, and holds intake.
        pressure_level[0] = 4
        keeper._system(1e9)
        assert guards.HOLD_INTAKE[0] is True
        assert len(sheds) == 1
        assert any('critical' in msg and 'other-proc' in msg for msg in logs)

        adm = admission.Admission(limit=None, most=4, ahead=0)
        assert adm.rows(pending=3, free_rows=4, step=128, busy=True, lengths=[32, 32, 32]) == 0
        assert adm.rows(pending=3, free_rows=4, step=128, busy=False, lengths=[32, 32, 32]) == 1

        # Normal pressure lifts intake hold.
        pressure_level[0] = 1
        keeper._system(1e9)
        assert guards.HOLD_INTAKE[0] is False
        assert adm.rows(pending=3, free_rows=4, step=128, busy=True, lengths=[32, 32, 32]) == 3

        # Strain warnings and relief hook transitions.
        keeper._strained(1e9)
        assert keeper._was_strained is True
        assert relieved[-1]['strained'] is True and relieved[-1]['memory_psi'] == 25.0
        keeper._strained(1e9)
        assert keeper._was_strained is False
        assert any('eased' in msg for msg in logs)
    finally:
        guards.HOLD_INTAKE[0] = False
        _restore_env(saved)


def test_apc_disk_write_cap_and_lazy_gather_streaming(tmp_path):
    saved_env = {'QUANTEEZE_APC_DISK_WRITE_GB': os.environ.get('QUANTEEZE_APC_DISK_WRITE_GB')}
    saved_wrote = dict(apc_disk.WROTE)
    try:
        with apc_disk._WROTE_LOCK:
            apc_disk.WROTE.update({'bytes': 0, 'refused': 0, 'said': False})
        os.environ['QUANTEEZE_APC_DISK_WRITE_GB'] = '0.000001'  # 1 KB
        assert apc_disk.write_cap() == pytest.approx(1000.0)
        assert apc_disk.over_cap() is False

        payload_file = tmp_path / 'shard.bin'
        payload_file.write_bytes(b'x' * 2048)
        assert apc_disk._record_wrote(payload_file) == 2048
        assert apc_disk.over_cap() is True
        assert apc_disk.WROTE['refused'] == 1
        assert apc_disk.WROTE['said'] is True

        # Lazy gather registration streams identical C-order bytes without full materialization.
        src = mx.arange(24, dtype=mx.float32).reshape(2, 6, 2)
        indices = np.array([1, 3, 5], dtype=np.int32)
        view = mx.take(src, mx.array(indices), axis=1)
        mx.eval(src, view)
        apc_disk.gathered(view, src, indices, axis=1)
        assert apc_disk.gather_of(view) is not None
        streamed = b''.join(apc_disk._pieces(view, limit=8))
        assert streamed == apc_disk._bytes(view)
        apc_disk.forget([view])
        assert apc_disk.gather_of(view) is None
    finally:
        apc_disk.forget(everyone=True)
        with apc_disk._WROTE_LOCK:
            apc_disk.WROTE.update(saved_wrote)
        _restore_env(saved_env)


def test_context_compress_algorithms_and_qsa_cache(monkeypatch):
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    saved_env = {
        k: os.environ.get(k)
        for k in (
            'QUANTEEZE_CONTEXT_COMPRESS',
            'QUANTEEZE_CONTEXT_LAYERS',
            'QUANTEEZE_CONTEXT_KEEP',
            'QUANTEEZE_CONTEXT_RECENT',
            'QUANTEEZE_CONTEXT_SINK',
            'QUANTEEZE_CONTEXT_MIN_TOKENS',
            'QUANTEEZE_CONTEXT_WINDOW',
        )
    }
    saved_stats = dict(context_compress.STATS)
    try:
        uniform = context_compress.budgets(4, 0.5, 'uniform')
        assert uniform == [pytest.approx(0.5)] * 4
        pyramid = context_compress.budgets(4, 0.5, 'pyramid')
        assert pyramid[0] > pyramid[1] > pyramid[2] > pyramid[3]
        assert sum(pyramid) / 4 == pytest.approx(0.5)

        scores = np.array([0.1, 9.0, 0.2, 8.0, 0.3, 0.4, 7.0, 0.1, 0.1, 0.1])
        picked = context_compress.choose(n_blocks=10, sink_blocks=1, recent_blocks=2, share=0.5, scores=scores)
        assert picked[0] and picked[8] and picked[9]
        assert picked[1] and picked[3] and picked[6]
        assert int(picked.sum()) >= 5

        assert list(context_compress.recent_scores(4)) == [0.0, 1.0, 2.0, 3.0]
        votes = np.array([[12.0, 1.0, 5.0, 1.0]], dtype=np.float32)
        ends = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        h2o = context_compress.h2o_scores(votes, ends, 4)
        assert h2o[0] > h2o[2] > h2o[3] > h2o[1]

        win = np.array([[1, 1, 1]], dtype=np.int32)
        snap = context_compress.snapkv_scores(win, 4, pool=3)
        assert snap[0] == snap[1] == snap[2] == pytest.approx(3.0) and snap[3] == 0.0

        # Build a synthetic QSAKVCache and exercise compact + compress across modes.
        def make_qsa(tokens=16):
            c = QSAKVCache()
            c.keys = mx.arange(2 * tokens * 4, dtype=mx.float16).reshape(1, 2, tokens, 4)
            c.values = mx.arange(2 * tokens * 4, dtype=mx.float16).reshape(1, 2, tokens, 4)
            ik = mx.ones((1, tokens, 4), dtype=mx.float16)
            ik[:, 8:12, :] = -1.0  # distinct block for similarity check
            c.index_keys = ik
            c.index_position_ids = mx.arange(tokens, dtype=mx.int32).reshape(1, tokens)
            c.offset = tokens
            c._quanteeze_votes = mx.array([[5.0, 1.0, 9.0, 2.0]], dtype=mx.float32)
            c._quanteeze_ends = mx.array([[0.0, 0.0, 0.0, 0.0, 2.0]], dtype=mx.float32)
            c._quanteeze_window = mx.array([[0, 1, 1, 2]], dtype=mx.int32)
            mx.eval(c.keys, c.values, c.index_keys, c.index_position_ids, c._quanteeze_votes, c._quanteeze_ends, c._quanteeze_window)
            return c

        for mode in ('recent', 'similar', 'h2o', 'snapkv'):
            os.environ['QUANTEEZE_CONTEXT_COMPRESS'] = mode
            os.environ['QUANTEEZE_CONTEXT_KEEP'] = '0.5'
            assert context_compress.tag().startswith(f'context:{mode}:')
            cache = make_qsa(16)
            out = context_compress.compress([cache], how=mode, layers='uniform', keep=0.5, recent=4, sink=4, min_tokens=8, ratio=4)
            assert isinstance(out[0], QSAKVCache)
            assert out[0].offset == 12
            assert out[0].dropped == 4
            assert context_compress.reviewed_of(out[0]) > 0

        # Ensure install / uninstall cleanly patch and restore QSAKVCache.
        orig_state = QSAKVCache.state
        assert context_compress.install() is True
        assert QSAKVCache.state is not orig_state
        context_compress.uninstall()
        assert QSAKVCache.state is orig_state
    finally:
        if context_compress._INSTALLED[0]:
            context_compress.uninstall()
        context_compress.STATS.update(saved_stats)
        _restore_env(saved_env)


def test_stm_room_primitives_and_leases():
    ext = stm_core.extension()
    if ext is None:
        pytest.skip('_stm_core C++ extension unavailable')
    drv = stm_core.room_driver()
    assert drv is not None
    drv.room_put(1000)
    assert drv.room_free() == 1000
    assert drv.room_take(400) is True
    assert drv.room_free() == 600
    assert drv.room_take(700, overdraw=False, timeout_ms=20.0) is False
    assert drv.room_free() == 600
    assert drv.room_take(700, overdraw=True) is True
    assert drv.room_free() == -100
    drv.room_give(200)
    assert drv.room_free() == 100

    # Blocked waiter wakes cleanly once room_give restores enough room.
    woke = []

    def waiter():
        woke.append(drv.room_take(500, overdraw=False, timeout_ms=2000.0))

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.03)
    drv.room_give(400)
    t.join(timeout=2.0)
    assert woke == [True]
    assert drv.room_free() == 0

    # leases.kv_sizes accounts for per-token K/V arrays and fixed buffers.
    c = SimpleNamespace(
        keys=mx.zeros((1, 2, 8, 4), dtype=mx.float16),      # 128 bytes / 8 tokens
        values=mx.zeros((1, 2, 8, 4), dtype=mx.float16),    # 128 bytes / 8 tokens
        extra=mx.zeros((4, 4), dtype=mx.float32),           # 64 fixed bytes
    )
    per_tok, fixed = leases.kv_sizes([c])
    assert per_tok == pytest.approx(32.0)
    assert fixed == 64

    # leases.Lease RAII & overdraw checks against a seeded RoomDriver.
    saved_env = {'QUANTEEZE_REQUEST_LEASES': os.environ.get('QUANTEEZE_REQUEST_LEASES')}
    saved_sizes = dict(leases._SIZES)
    try:
        os.environ['QUANTEEZE_REQUEST_LEASES'] = 'on'
        leases.reset()
        room = stm_core.room_driver()
        room.room_put(10_000_000)
        leases._ROOM[0] = room
        leases._SIZES.update({'working': 1_000_000, 'peak_token': 0.0, 'kv_token': 1000.0, 'kv_fixed': 0})

        with leases.Lease('req-1', prompt_tokens=100, max_tokens=100, step=100) as l1:
            assert l1.take(alone=False) is True
            held_after_take = room.room_free()
            assert held_after_take < 10_000_000
            l1.prompt_done()
            assert room.room_free() >= held_after_take
            assert leases.room_for_prompt(500_000) == 500_000
            leases.room_back(500_000)
        assert room.room_free() == 10_000_000

        # Contended take fails when room is insufficient, succeeds when alone=True overdraws.
        leases._SIZES['working'] = 20_000_000
        l2 = leases.Lease('req-2', prompt_tokens=10, max_tokens=10, step=10)
        assert l2.take(alone=False) is False
        assert l2.take(alone=True) is True
        assert leases.STATS['forced_alone'] >= 1
        l2.give()
        assert room.room_free() == 10_000_000
    finally:
        leases.reset()
        leases._SIZES.update(saved_sizes)
        _restore_env(saved_env)


def test_parallel_crossing_guards_and_tables_room_split(monkeypatch, tmp_path):
    saved_env = {'QUANTEEZE_PARALLEL_CROSSING': os.environ.get('QUANTEEZE_PARALLEL_CROSSING')}
    saved_group = parallel.GROUP[0]
    saved_owned = parallel._OWNED[0]
    saved_cut = parallel._CUT[0]
    saved_hosts = parallel._HOSTS[0]
    saved_on_host = parallel._ON_HOST[0]
    saved_crossed = list(parallel._CROSSED)
    try:
        parallel.GROUP[0] = SimpleNamespace(rank=lambda: 1, size=lambda: 2)
        parallel._OWNED[0] = range(2, 4)
        parallel._CUT[0] = [0, 2, 4]
        parallel._HOSTS[0] = [111, 111]
        parallel._ON_HOST[0] = 2
        parallel._CROSSED.clear()

        assert parallel.own_layer(2) is True
        assert parallel.own_layer(0) is False
        assert parallel.runs_here() == [range(0, 2), range(2, 4)]
        assert parallel.on_this_host() == 2

        os.environ['QUANTEEZE_PARALLEL_CROSSING'] = 'refuse'
        assert parallel.keep(2, 'ahead') is True
        assert parallel.keep(0, 'ahead') is False
        assert parallel.crossings() == {'ahead': 1}
        with pytest.raises(parallel.Crossing):
            parallel.require(0, 'expert read')

        os.environ['QUANTEEZE_PARALLEL_CROSSING'] = 'raise'
        with pytest.raises(parallel.Crossing):
            parallel.keep(0, 'ahead')

        # serve._tables_room_split shares remaining host room fairly across ranks on this host.
        monkeypatch.setattr(home, 'layer_experts_bytes', lambda p: {0: 10, 1: 20, 2: 30, 3: 40})
        monkeypatch.setattr(fitting, 'host_room', lambda *a, **kw: 1000)
        monkeypatch.setattr(serve, 'HELD_AFTER', 100)
        runs = [range(0, 2), range(2, 4)]
        # run 0 need = 30 + 40 + 100 = 170; run 1 need = 70 + 80 + 100 = 250 -> max = 250
        # share = 1000 -> 2 * (1000 - 250) = 1500
        assert serve._tables_room_split(tmp_path, runs, room=1000) == 1500
        # Keeper-clamped room below share bounds table by (room - mine) = 500 - 250 = 250
        assert serve._tables_room_split(tmp_path, runs, room=500) == 250
    finally:
        parallel.GROUP[0] = saved_group
        parallel._OWNED[0] = saved_owned
        parallel._CUT[0] = saved_cut
        parallel._HOSTS[0] = saved_hosts
        parallel._ON_HOST[0] = saved_on_host
        parallel._CROSSED.clear()
        parallel._CROSSED.extend(saved_crossed)
        _restore_env(saved_env)


def test_parallel_no_delay_and_per_thread_collective_and_crossed():
    """`parallel.no_delay` sets `TCP_NODELAY` on both socket ends, `_LAST` isolates per thread, and
    `refuse`/`crossings` sum thread-local dicts cleanly."""
    import socket
    import threading
    from multiprocessing.connection import Client, Listener
    from quanteeze import parallel

    with Listener(('127.0.0.1', 0), authkey=b'k') as listener:
        got = {}
        taker = threading.Thread(target=lambda: got.update(conn=parallel.no_delay(listener.accept())))
        taker.start()
        out = parallel.no_delay(Client(listener.address, authkey=b'k'))
        taker.join()
    for conn in (out, got['conn']):
        sock = socket.socket(fileno=os.dup(conn.fileno()))
        try:
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        finally:
            sock.close()
    out.send_bytes(b'x' * 65536)
    assert got['conn'].recv_bytes() == b'x' * 65536
    out.close()
    got['conn'].close()

    # Thread isolation of _LAST
    seen = {}
    parallel._LAST[0] = 'main'

    def worker(tag):
        parallel._LAST[0] = tag
        seen[tag] = parallel._LAST[0]

    threads = [threading.Thread(target=worker, args=(f't{i}',)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert parallel._LAST[0] == 'main'
    assert seen == {'t0': 't0', 't1': 't1', 't2': 't2', 't3': 't3'}
