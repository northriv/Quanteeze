# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for `quanteeze.cpu_experts`: C++ projections (AVX-512, AVX2, plain) and references against
MLX's quantized matmul, worker pool execution, work-stealing on preempted workers, runtime sanity
checks, core affinity, and CPU load hysteresis."""
import time

import mlx.core as mx
import numpy as np
import pytest

from quanteeze import cpu_experts as ce

pytestmark = pytest.mark.skipif(ce.isa() is None, reason='no C++17 compiler here')


def _affine(rows, cols, bits, group, seed=0):
    w = mx.array(np.random.default_rng(seed).normal(size=(rows, cols)).astype(np.float32)).astype(mx.bfloat16)
    q, scales, biases = mx.quantize(w, group_size=group, bits=bits)
    p = ce.projection(np.array(q), np.array(scales.view(mx.uint16)), np.array(biases.view(mx.uint16)), bits, group)
    return p, lambda x: np.array(
        mx.quantized_matmul(
            mx.array(x)[None], q, scales.astype(mx.float32), biases.astype(mx.float32),
            transpose=True, group_size=group, bits=bits,
        )
    )[0]


def _mxfp4(rows, cols, seed=0):
    w = mx.array(np.random.default_rng(seed).normal(size=(rows, cols)).astype(np.float32)).astype(mx.bfloat16)
    q, scales = mx.quantize(w, group_size=32, bits=4, mode='mxfp4')
    p = ce.projection(np.array(q), np.array(scales), None, 4, 32, kind='mxfp4')
    return p, lambda x: np.array(
        mx.quantized_matmul(
            mx.array(x)[None], q, scales, transpose=True, group_size=32, bits=4, mode='mxfp4',
        )
    )[0]


def _close(got, want, tol=1e-4):
    return float(np.abs(got - want).max()) <= tol * max(float(np.abs(want).max()), 1e-6)


@pytest.mark.parametrize('bits,group', [(4, 64), (4, 32), (8, 64), (2, 128), (4, 128)])
def test_affine_projection_matches_mlx_on_every_isa(bits, group):
    p, mlx = _affine(48, 256, bits, group, seed=bits)
    x = np.random.default_rng(1).normal(size=256).astype(np.float32)
    want = mlx(x)
    assert _close(ce.project_reference(p, x), want)
    for level in range(ce.isa() + 1):
        assert _close(ce.project(p, x, level), want), ce.ISA_NAMES[level]


def test_mxfp4_projection_matches_mlx_on_every_isa():
    p, mlx = _mxfp4(40, 256)
    x = np.random.default_rng(2).normal(size=256).astype(np.float32)
    want = mlx(x)
    assert _close(ce.project_reference(p, x), want)
    for level in range(ce.isa() + 1):
        assert _close(ce.project(p, x, level), want), ce.ISA_NAMES[level]


def test_unsupported_projection_shape_is_refused():
    p, _ = _affine(8, 96, 4, 32)
    p = dict(p, bits=3)
    with pytest.raises(ValueError):
        ce.project_rows(p, np.zeros(96, np.float32), np.zeros(3, np.float32), np.zeros(8, np.float32), 0, 8, 0)


def test_expert_pool_checks_against_reference(monkeypatch):
    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    D, I = 256, 128
    expert = {
        'gate_proj': _affine(I, D, 4, 64, 1)[0],
        'up_proj': _affine(I, D, 4, 64, 2)[0],
        'down_proj': _affine(D, I, 4, 64, 3)[0],
    }
    x = np.random.default_rng(3).normal(size=D).astype(np.float32)
    pool = ce.Pool(3)
    try:
        for limit in (0.0, 1.0):
            got = pool.expert(expert, x, limit)
            assert got.shape == (D,) and _close(got, ce.expert_reference(expert, x, limit), 1e-3)
        assert ce.verified() == (2, None)
        monkeypatch.setattr(ce, 'TOLERANCE', -1.0)
        assert pool.expert(expert, x) is None and ce.verified()[1] is not None
        assert pool.expert(expert, x) is None
    finally:
        pool.close()


def test_no_worker_runs_when_none_allowed():
    expert = {
        'gate_proj': _affine(64, 64, 4, 64)[0],
        'up_proj': _affine(64, 64, 4, 64)[0],
        'down_proj': _affine(64, 64, 4, 64)[0],
    }
    pool = ce.Pool(2, allowed=lambda: 0)
    try:
        assert pool.expert(expert, np.ones(64, np.float32)) is None
    finally:
        pool.close()


def test_load_yields_and_recovers_with_hysteresis():
    clock = [0.0]
    readings = []
    busy, total, own = [0], [0], [0]
    own_step = 50

    def read():
        share, pressure = readings.pop(0)
        busy[0] += int(share * 100) + own_step
        total[0] += 100
        own[0] += own_step
        return (busy[0], total[0]), own[0], pressure

    load = ce.Load(threads=8, yield_at=25, cores=16, clock=lambda: clock[0], read=read, least=4)
    readings[:] = [(0.0, 0.0)] * 30
    allowed = []
    for _ in range(14):
        clock[0] += 0.5
        allowed.append(load.sample())
    assert allowed[0] == 0 and allowed == sorted(allowed)
    assert set(allowed) - {0} and min(set(allowed) - {0}) == 4
    assert all(b - a <= 1 for a, b in zip(allowed, allowed[1:]) if a)

    readings[:] = [(0.30, 0.0)]
    clock[0] += 0.25
    assert load.sample() == 0 and load.others == pytest.approx(30.0)

    readings[:] = [(0.0, 8.0)]
    clock[0] += 0.25
    assert load.sample() == 0

    readings[:] = [(0.10, 0.0)] * 40
    for _ in range(40):
        clock[0] += 3.0
        top = load.sample()
    assert top == 8

    small = ce.Load(threads=32, yield_at=25, cores=16, clock=lambda: clock[0], read=read, least=4)
    readings[:] = [(0.10, 0.0)] * 60
    for _ in range(60):
        clock[0] += 3.0
        top = small.sample()
    assert top == 16 - ce.RESERVE_CORES - 2

    few = ce.Load(threads=8, yield_at=90, cores=6, clock=lambda: clock[0], read=read, least=4)
    readings[:] = [(0.30, 0.0)] * 20
    for _ in range(20):
        clock[0] += 3.0
        assert few.sample() == 0


def test_proc_stat_parsing(tmp_path):
    total, own, _ = ce._read_proc()
    assert total is None or (total[1] >= total[0] >= 0)
    assert own >= 0
    stat = tmp_path / 'stat'
    stat.write_text('cpu  100 0 50 800 50 0 0 0 30 0\n')
    got, _, _ = ce._read_proc(stat=str(stat), own=str(tmp_path / 'none'), psi=str(tmp_path / 'none'))
    assert got == (150, 1000)


@pytest.mark.parametrize('level', [0, 1, 2])
def test_mxfp4_expert_through_pool_on_each_isa(level, monkeypatch):
    if level > ce.isa():
        pytest.skip('ISA unsupported on this CPU')
    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    D, I = 256, 128
    expert = {
        'gate_proj': _mxfp4(I, D, 1)[0],
        'up_proj': _mxfp4(I, D, 2)[0],
        'down_proj': _mxfp4(D, I, 3)[0],
    }
    x = np.random.default_rng(4).normal(size=D).astype(np.float32)
    pool = ce.Pool(4, level)
    try:
        got = pool.expert(expert, x, 10.0)
        assert _close(got, ce.expert_reference(expert, x, 10.0), 1e-3) and ce.verified() == (1, None)
    finally:
        pool.close()


def test_workers_pinned_last_physical_cores_first(tmp_path, monkeypatch):
    for cpu in range(16):
        topo = tmp_path / f'cpu{cpu}' / 'topology'
        topo.mkdir(parents=True)
        (topo / 'core_id').write_text(str(cpu % 8))
        (topo / 'physical_package_id').write_text('0')
    monkeypatch.setattr(ce.os, 'sched_getaffinity', lambda pid: set(range(16)), raising=False)
    assert ce.cores(4, root=tmp_path) == [7, 6, 5, 4]
    assert ce.cores(6, root=tmp_path) == [7, 6, 5, 4, 3, 2]
    assert ce.cores(8, root=tmp_path) == [7, 6, 5, 4, 3, 2, 15, 14]
    assert ce.cores(16, root=tmp_path, siblings=False) == [7, 6, 5, 4, 3, 2]
    assert ce.cores(4, root=tmp_path / 'none') == []


def test_pool_counts_and_determinism(monkeypatch):
    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    D, I = 512, 384
    expert = {
        'gate_proj': _affine(I, D, 4, 64, 1)[0],
        'up_proj': _affine(I, D, 4, 64, 2)[0],
        'down_proj': _affine(D, I, 4, 64, 3)[0],
    }
    x = np.random.default_rng(5).normal(size=D).astype(np.float32)
    pool = ce.Pool(4, spin_us=0)
    try:
        first = pool.expert(expert, x)
        assert first is not None
        for _ in range(200):
            assert np.array_equal(pool.expert(expert, x), first)
        until = time.monotonic() + 10
        while pool.busy() and time.monotonic() < until:
            time.sleep(0.01)
        s = pool.stats()
        chunks = 201 * (2 * I // 32 + D // 32)
        assert s['by_workers'] + s['by_caller'] == chunks and s['stolen'] >= 0
        assert 0 <= s['woke'] <= s['rung'] == 201 * 2 * 4
        assert s['wake_ns'] >= 0 and s['wake_ns_most'] * max(s['woke'], 1) >= s['wake_ns']
        assert s['caller_ns'] > 0 and s['moved'] == 0
        if hasattr(ce.os, 'sched_getaffinity'):
            allowed = ce.os.sched_getaffinity(0)
            assert s['caller_cpus'] and s['caller_cpus'] <= allowed and s['worker_cpus'] <= set(pool.cpus or allowed)
    finally:
        pool.close()
    assert pool.stats() is None and pool.busy() == 0


def test_stalled_worker_in_chunk_is_stolen(monkeypatch):
    monkeypatch.setattr(ce, '_STATE', {'checked': ce.CHECKS, 'failed': None})
    D, I = 512, 384
    expert = {
        'gate_proj': _affine(I, D, 4, 64, 1)[0],
        'up_proj': _affine(I, D, 4, 64, 2)[0],
        'down_proj': _affine(D, I, 4, 64, 3)[0],
    }
    x = np.random.default_rng(6).normal(size=D).astype(np.float32)
    pool = ce.Pool(2, spin_us=200)
    try:
        want = pool.expert(expert, x)
        assert _close(want, ce.expert_reference(expert, x), 1e-3)
        ce.lib().qz_pool_stall(ce.ctypes.c_void_p(pool._handle), 300_000_000)
        outs = []
        t = time.monotonic()
        while pool.stats()['stolen'] == 0 and time.monotonic() - t < 20:
            outs.append(pool.expert(expert, x))
        assert pool.stats()['stolen'] >= 1, 'no worker took a chunk in 20 s'
        assert pool.busy() >= 1
        for _ in range(20):
            outs.append(pool.expert(expert, x))
        until = time.monotonic() + 10
        while pool.busy() and time.monotonic() < until:
            time.sleep(0.01)
        assert pool.busy() == 0
        assert all(np.array_equal(o, want) for o in outs)
    finally:
        pool.close()


@pytest.mark.parametrize('kind', ['affine2', 'affine4', 'affine8', 'mxfp4'])
def test_cpp_reference_matches_numpy(kind, monkeypatch):
    D, I = 256, 96
    if kind == 'mxfp4':
        expert = {
            'gate_proj': _mxfp4(I, D, 1)[0],
            'up_proj': _mxfp4(I, D, 2)[0],
            'down_proj': _mxfp4(D, I, 3)[0],
        }
    else:
        bits = int(kind[-1])
        expert = {
            'gate_proj': _affine(I, D, bits, 32, 1)[0],
            'up_proj': _affine(I, D, bits, 64, 2)[0],
            'down_proj': _affine(D, I, bits, 32, 3)[0],
        }
    x = np.random.default_rng(8).normal(size=D).astype(np.float32)
    p = expert['gate_proj']
    y = np.empty(p['rows'], np.float32)
    ce.lib().qz_reference(ce.ctypes.byref(ce._proj(p)), ce._fptr(x), ce._fptr(y))
    assert _close(y, ce.project_reference(p, x), 1e-6)
    for limit in (0.0, 1.0):
        assert _close(ce.native_reference(expert, x, limit), ce.expert_reference(expert, x, limit), 1e-5)
    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    monkeypatch.setattr(ce, 'expert_reference', None)
    pool = ce.Pool(2)
    try:
        assert pool.expert(expert, x) is not None and ce.verified() == (1, None)
    finally:
        pool.close()


def test_deepseek_v41_roundings_match_mlx_vlm():
    fakequant = pytest.importorskip('mlx_vlm.models.deepseek_v41.fakequant')
    rng = np.random.default_rng(9)
    a = (rng.normal(size=4096) * np.exp(rng.normal(size=4096) * 4)).astype(np.float32)
    a[:32], a[32:64], a[64:96] = 0.0, 1e-7, 3e4
    assert np.array_equal(ce.fp8_roundtrip(a), np.array(fakequant.fake_quant_fp8_ue8m0(mx.array(a))))
    assert np.array_equal(ce.to_bf16(a), np.array(mx.array(a).astype(mx.bfloat16).astype(mx.float32)))
    gate = rng.uniform(17, 60, 2048).astype(np.float32)
    up = a[:2048].copy()
    for limit, weight in ((0.0, 1.0), (30.0, 0.37)):
        want = ce.activate(gate, up, limit, weight, 'deepseek_v41')
        got = np.empty(2048, np.float32)
        ce.lib().qz_activate(ce._fptr(gate), ce._fptr(up), ce._fptr(got), 2048, limit, weight, 1)
        assert np.array_equal(got, want)


def test_deepseek_v41_expert_matches_mlx_vlm(monkeypatch):
    language = pytest.importorskip('mlx_vlm.models.deepseek_v41.language')
    fakequant = pytest.importorskip('mlx_vlm.models.deepseek_v41.fakequant')
    D, I = 512, 256
    projections = [_mxfp4(I, D, 1)[0], _mxfp4(I, D, 2)[0], _mxfp4(D, I, 3)[0]]
    expert = dict(zip(('gate_proj', 'up_proj', 'down_proj'), projections))
    qs = []
    for rows, cols, seed in ((I, D, 1), (I, D, 2), (D, I, 3)):
        w = mx.array(np.random.default_rng(seed).normal(size=(rows, cols)).astype(np.float32)).astype(mx.bfloat16)
        qs.append(mx.quantize(w, group_size=32, bits=4, mode='mxfp4'))

    def gpu(x, q):
        return mx.quantized_matmul(
            x.astype(mx.float32), q[0], q[1], transpose=True, group_size=32, bits=4, mode='mxfp4',
        ).astype(mx.bfloat16)

    pool = ce.Pool(2)
    try:
        for k, (limit, weight) in enumerate(((0.0, 1.0), (10.0, 0.37), (1.0, 2.5))):
            x = mx.array(np.random.default_rng(20 + k).normal(size=(1, D)).astype(np.float32)).astype(mx.bfloat16)
            x = fakequant.fake_quant_fp8_ue8m0(x)
            act = language.DeepseekV41SwiGLU(limit)(gpu(x, qs[1]), gpu(x, qs[0]), mx.array(weight, mx.float32))
            want = np.array(gpu(act, qs[2]).astype(mx.float32))[0]
            xf = np.array(x.astype(mx.float32))[0]
            monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
            checked = pool.expert(expert, xf, limit, weight, 'deepseek_v41')
            assert ce.verified() == (1, None)
            got = pool.expert(expert, xf, limit, weight, 'deepseek_v41')
            assert np.array_equal(checked, got)
            assert _close(got, want, 1e-2) and np.mean(got == want) >= 0.98
            ref = ce.expert_reference(expert, xf, limit, weight, 'deepseek_v41')
            assert np.mean(ref == want) >= 0.98
            assert _close(ce.native_reference(expert, xf, limit, weight, 'deepseek_v41'), ref, 1e-2)
    finally:
        pool.close()


def test_worker_preempted_before_taking_chunk_is_stolen(monkeypatch):
    monkeypatch.setattr(ce, '_STATE', {'checked': ce.CHECKS, 'failed': None})
    D, I = 512, 384
    expert = {
        'gate_proj': _affine(I, D, 4, 64, 1)[0],
        'up_proj': _affine(I, D, 4, 64, 2)[0],
        'down_proj': _affine(D, I, 4, 64, 3)[0],
    }
    x = np.random.default_rng(7).normal(size=D).astype(np.float32)
    pool = ce.Pool(2, spin_us=200)
    try:
        want = pool.expert(expert, x)
        ce.lib().qz_pool_stall_before(ce.ctypes.c_void_p(pool._handle), 300_000_000)
        slowest, outs = 0.0, []
        t = time.monotonic()
        while pool.stats()['stolen'] == 0 and time.monotonic() - t < 20:
            t0 = time.monotonic()
            outs.append(pool.expert(expert, x))
            slowest = max(slowest, time.monotonic() - t0)
        assert pool.stats()['stolen'] >= 1, 'no worker took a chunk in 20 s'
        assert slowest < 0.15, slowest
        until = time.monotonic() + 10
        while pool.busy() and time.monotonic() < until:
            time.sleep(0.01)
        outs += [pool.expert(expert, x) for _ in range(10)]
        assert all(np.array_equal(o, want) for o in outs)
    finally:
        pool.close()


def test_serving_thread_excludes_worker_cores():
    top = ce.topology()
    if len({core for core in top.values()}) < 3:
        pytest.skip('too few physical cores')
    pool = ce.Pool(2)
    try:
        free = ce.free_cpus(pool.cpus)
        assert free and not ({top[c] for c in free} & {top[c] for c in pool.cpus})
    finally:
        pool.close()
