# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for the portable C++20 KAME STM core (`_stm_core`, `native_ext`, `stm_core`, `Demand`)."""
import gc
import threading
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from quanteeze import allocator, demand, native_ext, native_reads, stm_core


def _require_stm():
    tc, why = native_ext._toolchain(with_mlx=True)
    if tc is None:
        pytest.skip(why)
    ext = native_ext.extension('stm_core', with_mlx=True, with_stm=True, force=True, quiet=False)
    assert ext is not None, '_stm_core failed to compile despite C++20/nanobind/MLX toolchain being available'
    return ext


def test_selftest_concurrent_writers_and_readers():
    ext = _require_stm()
    count, total_sum, torn, shots = ext.selftest(writers=4, readers=2, commits=300)
    assert count == 1200
    assert total_sum == 300 * (1 + 2 + 3 + 4)
    assert torn == 0
    assert shots > 0


def test_model_placement_tables_hints_and_mismatch():
    ext = _require_stm()
    m = ext.Model(2)
    assert m.placement(0) is None
    assert m.tables(0) is None

    route = np.array([0, 1, -1, -1], dtype=np.int64)
    absent = np.array([False, False, True, True], dtype=bool)
    experts = np.array([0, 1], dtype=np.int64)
    v1 = m.put_placement(0, route, absent, experts)
    assert v1 == 1

    hints = m.hints
    assert int(hints[0]) == 1
    assert int(hints[1]) == 0

    ver, r_out, ab_out, exp_out = m.placement(0)
    assert ver == 1
    np.testing.assert_array_equal(r_out, route)
    np.testing.assert_array_equal(ab_out, absent)
    np.testing.assert_array_equal(exp_out, experts)

    tver, tr, tab, texps, slot_in = m.tables(0)
    assert tver == 1
    assert int(slot_in[1]) == 1
    assert tab[2] == 1
    np.testing.assert_array_equal(texps, experts)

    assert m.check(0, route, absent, experts) == 0
    assert m.check(0, np.array([1, 0, -1, -1], dtype=np.int64), absent, experts) == 1
    assert m.check(0, route, np.array([False, True, True, True], dtype=bool), experts) == 2
    assert m.check(0, route, absent, np.array([1, 0], dtype=np.int64)) == 4

    # Dropping Python Model reference keeps hints view valid via reference_internal
    del m
    gc.collect()
    assert int(hints[0]) == 1


def test_pass_leases_tokens_and_release_modes():
    ext = _require_stm()
    m = ext.Model(3)
    m.put_placement(0, np.array([0, -1], dtype=np.int64), np.array([False, True], dtype=bool), np.array([0], dtype=np.int64))
    m.put_placement(1, np.array([0, -1], dtype=np.int64), np.array([False, True], dtype=bool), np.array([0], dtype=np.int64))

    assert m.begin_layer(0, 10) == 1
    assert m.begin_layer(1, 10) == 1
    assert m.active_passes(0) == [10]
    assert m.active_passes(1) == [10]

    m.release_layer(0, 10)
    assert m.active_passes(0) == []
    assert m.is_completed(10) is False  # layer 1 still held

    m.release(10, keep_k=1)
    assert m.active_passes(1) == [10]
    m.release_all()
    assert m.active_passes(1) == []
    assert m.is_completed(10) is True

    # Token lifetime holds completion until MLX destroys token after eval + gc
    m.begin_layer(0, 42)
    tok = m.pass_token(42)
    y = mx.depends(mx.array([2.0], dtype=mx.float32), tok)[0]
    assert m.is_completed(42) is False
    del tok
    mx.eval(y)
    mx.synchronize()
    del y
    gc.collect()
    m.completed()
    assert m.is_completed(42) is True
    assert m.active_passes(0) == []


def test_room_node_budget_and_drop_quant():
    ext = _require_stm()
    m = ext.Model(2)
    w = mx.ones((2, 4), dtype=mx.float32)
    mx.eval(w)
    m.register_quant(0, 0, w, None, None, 0, 0, 'raw', (-1, -1, -1))
    arrs = m.arrays_of(0)
    assert arrs[0] is not None
    m.drop_quant(0)
    assert m.arrays_of(0)[0] is None

    m.room_put(100)
    assert m.room_free() == 100
    assert m.room_take(120) is False
    assert m.room_take(60) is True
    assert m.room_free() == 40
    m.room_give(60)
    assert m.room_free() == 100


@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 compiler here')
def test_lander_reads_via_c_readers_and_waits_for_pass_lease(tmp_path):
    ext = _require_stm()
    blob = tmp_path / 'experts.bin'
    expert_rows = np.arange(4 * 16, dtype=np.uint8).reshape(4, 16)
    blob.write_bytes(expert_rows.tobytes())

    bank = mx.zeros((2, 16), dtype=mx.uint8)
    bank[0] = mx.array(expert_rows[0])
    bank[1] = mx.array(expert_rows[1])
    mx.eval(bank)

    rd = native_reads.Readers(workers=2)
    m = ext.Model(1)
    try:
        m.put_placement(
            0,
            np.array([0, 1, -1, -1], dtype=np.int64),
            np.array([False, False, True, True], dtype=bool),
            np.array([0, 1], dtype=np.int64),
        )
        alloc_addr, free_addr = allocator.host_alloc_addrs()
        if alloc_addr and free_addr:
            m.attach_host(alloc_addr, free_addr, 0, 0, 0)
        handle, sub, st, rel, tm, bl, wk = rd.c_entrypoints()
        m.attach_readers(handle, sub, st, rel, tm, bl, wk)
        m.register_quant(0, 0, bank, None, None, 0, 0, 'raw', (0, -1, -1))
        fid = rd.file_id(blob)
        files = np.full(4, fid, dtype=np.int32)
        offsets = np.arange(4, dtype=np.int64) * 16
        m.register_tensor(0, 0, files, offsets, 16)

        m.begin_layer(0, 99)
        m.start_lander(wait_timeout_ms=2000.0)
        assert m.submit_landing(0, [1], [3], priority=0) is True

        # Wait briefly while pass 99 holds layer 0: placement must NOT flip underneath pass 99
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline and m.held_back() == 0:
            time.sleep(0.005)
        assert int(m.hints[0]) == 1

        # Releasing pass 99 lets the lander copy expert 3 into slot 1 and commit version 2
        m.release_layer(0, 99)
        done_by = time.monotonic() + 3.0
        while time.monotonic() < done_by and int(m.hints[0]) < 2:
            time.sleep(0.01)

        assert int(m.hints[0]) == 2
        landed, not_fit, failed, _ = m.landings()
        assert landed == 1 and not_fit == 0 and failed == 0
        np.testing.assert_array_equal(np.array(bank[1]), expert_rows[3])
        ver, route, absent, experts = m.placement(0)
        assert ver == 2
        assert int(route[3]) == 1
        assert int(route[1]) == -1
        assert bool(absent[1]) is True
        assert list(experts) == [0, 3]
    finally:
        m.stop_lander()
        rd.close()


def _make_safetensors_moe(tmp_path, seed=17, n_exp=8, n_layers=1):
    rng = np.random.default_rng(seed)
    hidden, inner, gs = 128, 128, 64
    sw0 = SwitchGLU(input_dims=hidden, hidden_dims=inner, num_experts=n_exp, bias=False)
    for pname in ('gate_proj', 'up_proj', 'down_proj'):
        setattr(sw0, pname, getattr(sw0, pname).to_quantized(group_size=gs, bits=4))
    tensors = {}
    layers = []
    for li in range(n_layers):
        sw = SwitchGLU(input_dims=hidden, hidden_dims=inner, num_experts=n_exp, bias=False)
        for pname in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(sw, pname, getattr(sw, pname).to_quantized(group_size=gs, bits=4))
            w = mx.array(rng.normal(0, 0.05, (n_exp, hidden, inner)).astype(np.float32)).astype(mx.bfloat16)
            q, s, b = mx.quantize(w, group_size=gs, bits=4)
            mx.eval(q, s, b)
            tensors[f'layers.{li}.mlp.switch_mlp.{pname}.weight'] = q
            tensors[f'layers.{li}.mlp.switch_mlp.{pname}.scales'] = s.astype(mx.bfloat16)
            tensors[f'layers.{li}.mlp.switch_mlp.{pname}.biases'] = b.astype(mx.bfloat16)
        mlp = nn.Module()
        mlp.switch_mlp = sw
        layer = nn.Module()
        layer.mlp = mlp
        layers.append(layer)
    (tmp_path / 'config.json').write_text('{}', encoding='utf-8')
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)
    model = nn.Module()
    model.layers = layers
    from quanteeze import q2_experts
    return model, q2_experts.Checkpoint(tmp_path), rng


@pytest.mark.skipif(native_ext._toolchain(with_mlx=True)[0] is None, reason='no C++20 STM toolchain')
@pytest.mark.parametrize('stm_mode', ['shadow', 'on'])
def test_demand_stm_shadow_and_on_integration(tmp_path, monkeypatch, stm_mode):
    _require_stm()
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')
    monkeypatch.setenv('QUANTEEZE_DEMAND_READ_NATIVE', 'off')
    monkeypatch.setenv('QUANTEEZE_DEMAND_STM', stm_mode)

    model, ckpt, rng = _make_safetensors_moe(tmp_path)
    dem = demand.Demand(model, ckpt, budget_bytes=1, workers=1)
    try:
        assert dem._stm is not None
        assert dem._stm.mode == stm_mode
        assert dem._stm.started is (stm_mode == 'on')
        assert dem in demand._LIVE

        x = mx.array(rng.normal(0, 0.2, (1, 1, 128)).astype(np.float32)).astype(mx.bfloat16)
        # Route a mix of initially resident and SSD-only experts to force place_many + pass token flow
        idx = mx.array([[[0, 7]]], dtype=mx.uint32)
        mx.eval(x, idx)
        y = model.layers[0].mlp.switch_mlp(x, idx)
        mx.eval(y)

        s = dem.summary()
        assert 'stm' in s
        assert s['stm']['commits'] >= 1
        if stm_mode == 'shadow':
            assert s['stm']['checks'] >= 1
            assert s['stm']['mismatched'] == 0
        else:
            assert s['stm']['started'] is True
            dem.thread_done()
    finally:
        dem.close()
        assert dem not in demand._LIVE


@pytest.mark.skipif(native_ext._toolchain(with_mlx=True)[0] is None, reason='no C++20 STM toolchain')
@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 native readers')
def test_demand_cpp_step_and_land_now_bit_exact(tmp_path, monkeypatch):
    _require_stm()
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')
    monkeypatch.setenv('QUANTEEZE_DEMAND_READ_NATIVE', 'on')

    rng = np.random.default_rng(2026)
    x_pre = mx.array(rng.normal(0, 0.25, (1, 24, 128)).astype(np.float32)).astype(mx.bfloat16)
    idx_pre = mx.array(rng.integers(0, 16, (1, 24, 2)).astype(np.uint32))
    sc_pre = mx.array(rng.uniform(0.1, 0.9, (1, 24, 2)).astype(np.float32)).astype(mx.bfloat16)
    decode_steps = [
        (
            mx.array(rng.normal(0, 0.25, (1, 1, 128)).astype(np.float32)).astype(mx.bfloat16),
            mx.array([[exps]], dtype=mx.uint32),
            mx.array([[weights]], dtype=mx.float32).astype(mx.bfloat16),
        )
        for exps, weights in [
            ([0, 8], [0.6, 0.4]),
            ([1, 9], [0.7, 0.3]),
            ([2, 10], [0.5, 0.5]),
            ([3, 11], [0.8, 0.2]),   # past initial CHECKS=3: executes via C++ demand_step + land_now
            ([11, 12], [0.4, 0.6]),  # mixed hit + miss in C++
            ([12, 13], [0.9, 0.1]),  # second C++ landing after prior pass token retires
            ([12, 13], [0.3, 0.7]),  # pure C++ cache hit
        ]
    ]
    mx.eval(x_pre, idx_pre, sc_pre)
    for xd, idxd, scd in decode_steps:
        mx.eval(xd, idxd, scd)

    def _run(stm_mode):
        monkeypatch.setenv('QUANTEEZE_DEMAND_STM', stm_mode)
        monkeypatch.setattr(native_reads, '_STATE', {'checked': 0, 'failed': None})
        sub = tmp_path / stm_mode
        sub.mkdir()
        model, ckpt, _ = _make_safetensors_moe(sub, seed=31, n_exp=16)
        dem = demand.Demand(model, ckpt, budget_bytes=1, workers=2)
        try:
            assert dem.slots == 8
            y_pre = model.layers[0].mlp.switch_mlp(x_pre, idx_pre, sc_pre)
            mx.eval(y_pre)
            ys = [y_pre]
            for xd, idxd, scd in decode_steps:
                y = model.layers[0].mlp.switch_mlp(xd, idxd, scd)
                mx.eval(y)
                ys.append(y)
            if stm_mode == 'on':
                dem.thread_done()
            return ys, dem.switches[0].route.copy(), dem.weight[0].copy(), dem.summary()
        finally:
            dem.close()

    ys_off, route_off, weight_off, sum_off = _run('off')
    ys_on, route_on, weight_on, sum_on = _run('on')

    assert 'stm' not in sum_off
    assert sum_on['stm']['serving'] is True
    assert sum_on['stm']['why'] is None
    assert sum_on['stm']['landed'] > 0
    assert sum_on['stm']['reads'] > 0
    assert sum_on['hits'] == sum_off['hits']
    assert sum_on['misses'] == sum_off['misses']
    assert sum_on['native_reads'] == sum_off['native_reads']
    np.testing.assert_array_equal(route_on, route_off)
    np.testing.assert_allclose(weight_on, weight_off, rtol=0, atol=0)
    for y_on, y_off in zip(ys_on, ys_off):
        assert mx.array_equal(y_on, y_off).item()


def test_home_html_renders_cpp_stm_labels():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / 'src' / 'quanteeze' / 'home.html').read_text(encoding='utf-8')
    assert 'On-demand SSD (C++)' in html
    assert 'オンデマンド SSD (C++)' in html
    assert 'On-demand SSD (Python)' in html
    assert 'オンデマンド SSD (Python)' in html
    assert "id=\"stm-warn\"" in html
    assert 'C++ STM' in html


@pytest.mark.skipif(native_ext._toolchain(with_mlx=True)[0] is None, reason='no C++20 STM toolchain')
@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 native readers')
def test_demand_concurrent_threads_cpp_stm(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    _require_stm()
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')
    monkeypatch.setenv('QUANTEEZE_DEMAND_READ_NATIVE', 'on')
    monkeypatch.setenv('QUANTEEZE_DEMAND_STM', 'on')
    monkeypatch.setattr(native_reads, '_STATE', {'checked': 0, 'failed': None})

    model, ckpt, rng = _make_safetensors_moe(tmp_path, seed=42, n_exp=16, n_layers=4)
    dem = demand.Demand(model, ckpt, budget_bytes=1, workers=2)
    try:
        assert dem.slots == 8
        barrier = threading.Barrier(2, timeout=5.0)
        xs = [
            mx.array(rng.normal(0, 0.25, (1, 1, 128)).astype(np.float32)).astype(mx.bfloat16)
            for _ in range(2)
        ]
        mx.eval(*xs)

        def _worker(tid):
            h = xs[tid]
            outs = []
            try:
                barrier.wait()
                for step in range(10):
                    for li in range(4):
                        e0 = (tid * 4 + step + li) % 16
                        e1 = (e0 + 8) % 16
                        idx = mx.array([[[e0, e1]]], dtype=mx.uint32)
                        sc = mx.array([[[0.6, 0.4]]], dtype=mx.float32).astype(mx.bfloat16)
                        y = model.layers[li].mlp.switch_mlp(h, idx, sc)
                        h = (y * sc[..., None]).sum(axis=-2)
                    mx.eval(h)
                    outs.append(h.shape)
            finally:
                mx.synchronize()
                dem.thread_done()
                clear = getattr(mx, 'clear_streams', None)
                if clear is not None:
                    clear()
            return outs

        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(_worker, tid) for tid in range(2)]
            for fut in futs:
                assert fut.result(timeout=30.0) == [(1, 1, 128)] * 10

        assert native_reads._STATE['failed'] is None
        s = dem.summary()
        assert s['stm']['serving'] is True
        assert s['stm']['why'] is None
        assert s['stm']['landed'] > 0
        assert s['stm']['not_fit'] == 0
        assert s['stm']['failed'] == 0
        assert s['stm']['write_failed'] == 0
    finally:
        dem.close()


@pytest.mark.skipif(native_ext._toolchain(with_mlx=True)[0] is None, reason='no C++20 STM toolchain')
@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 native readers')
def test_demand_stm_auto_default_and_quiet_fallback(tmp_path, monkeypatch, capsys):
    _require_stm()
    monkeypatch.delenv('QUANTEEZE_DEMAND_STM', raising=False)
    monkeypatch.setattr(stm_core, '_SAID', set())
    assert stm_core.mode() == 'auto'

    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')
    monkeypatch.setenv('QUANTEEZE_DEMAND_READ_NATIVE', 'on')
    monkeypatch.setattr(native_reads, '_STATE', {'checked': 0, 'failed': None})
    (tmp_path / 'on').mkdir()
    model, ckpt, _ = _make_safetensors_moe(tmp_path / 'on', seed=51, n_exp=8)
    dem = demand.Demand(model, ckpt, budget_bytes=1, workers=1)
    try:
        s = dem.summary()
        assert s['stm']['asked'] == 'auto'
        assert s['stm']['mode'] == 'auto'
        assert s['stm']['serving'] is True
    finally:
        dem.close()

    monkeypatch.setenv('QUANTEEZE_DEMAND_READ_NATIVE', 'off')
    (tmp_path / 'off').mkdir(exist_ok=True)
    model2, ckpt2, _ = _make_safetensors_moe(tmp_path / 'off', seed=52, n_exp=8)
    capsys.readouterr()
    dem2 = demand.Demand(model2, ckpt2, budget_bytes=1, workers=1)
    try:
        s2 = dem2.summary()
        assert s2['stm']['asked'] == 'auto'
        assert s2['stm']['serving'] is False
    finally:
        dem2.close()
    captured = capsys.readouterr()
    assert 'WARNING' not in captured.err and 'warning' not in captured.err.lower()


def test_turn_node_arbitration_and_timeout():
    ext = _require_stm()
    m = ext.Model(1)

    t0 = ext.Model.now_ns()
    assert m.turn_try(stm_core.TURN_DECODE, t0) is True
    assert m.turn_state() == (False, False, 1, 0)

    # Concurrent second decode does not block
    dec2_ok = []

    def _dec2():
        assert m.turn_try(stm_core.TURN_DECODE, t0 + 1) is True
        dec2_ok.append(m.turn_state())
        m.turn_give()

    th_d2 = threading.Thread(target=_dec2)
    th_d2.start()
    th_d2.join(timeout=2.0)
    assert dec2_ok == [(False, False, 2, 0)]

    # Prompt on another thread cannot acquire while first decode holds, and unblocks on turn_give()
    prompt_try = []
    prompt_res = []

    def _prompt_waiter():
        prompt_try.append(m.turn_try(stm_core.TURN_PROMPT, t0 + 2))
        took = m.turn_take(stm_core.TURN_PROMPT, ext.Model.now_ns(), 2000.0)
        prompt_res.append(took)
        m.turn_give()

    th_p = threading.Thread(target=_prompt_waiter)
    th_p.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not m.turn_state()[1]:
        time.sleep(0.005)
    assert prompt_try == [False]
    assert m.turn_state()[1] is True
    m.turn_give()
    th_p.join(timeout=2.0)
    assert len(prompt_res) == 1 and prompt_res[0] >= 0

    # Older prompt (birth=100) beats newer decode (birth=200) when waiting behind held decode
    dec_held = threading.Event()
    dec_release = threading.Event()

    def _older_dec_holder():
        assert m.turn_try(stm_core.TURN_DECODE, 10) is True
        dec_held.set()
        assert dec_release.wait(timeout=2.0)
        m.turn_give()

    th_hold = threading.Thread(target=_older_dec_holder)
    th_hold.start()
    assert dec_held.wait(timeout=2.0)

    order = []

    def _older_prompt():
        took = m.turn_take(stm_core.TURN_PROMPT, 100, 2000.0)
        order.append(('prompt', took >= 0))
        time.sleep(0.02)
        m.turn_give()

    def _newer_decode():
        took = m.turn_take(stm_core.TURN_DECODE, 200, 2000.0)
        order.append(('decode', took >= 0))
        m.turn_give()

    th_op = threading.Thread(target=_older_prompt)
    th_op.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not m.turn_state()[1]:
        time.sleep(0.002)
    th_nd = threading.Thread(target=_newer_decode)
    th_nd.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and m.turn_state()[3] < 1:
        time.sleep(0.002)
    dec_release.set()
    th_hold.join(timeout=2.0)
    th_op.join(timeout=2.0)
    th_nd.join(timeout=2.0)
    assert order == [('prompt', True), ('decode', True)]

    # Older waiting decode (birth=300) beats newer prompt (birth=400) when waiting behind held prompt
    p_held = threading.Event()
    p_release = threading.Event()

    def _prompt_holder():
        assert m.turn_try(stm_core.TURN_PROMPT, 250) is True
        p_held.set()
        assert p_release.wait(timeout=2.0)
        m.turn_give()

    th_ph = threading.Thread(target=_prompt_holder)
    th_ph.start()
    assert p_held.wait(timeout=2.0)

    order2 = []

    def _older_decode():
        took = m.turn_take(stm_core.TURN_DECODE, 300, 2000.0)
        order2.append(('decode', took >= 0))
        time.sleep(0.02)
        m.turn_give()

    def _newer_prompt():
        took = m.turn_take(stm_core.TURN_PROMPT, 400, 2000.0)
        order2.append(('prompt', took >= 0))
        m.turn_give()

    th_od = threading.Thread(target=_older_decode)
    th_od.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and m.turn_state()[3] < 1:
        time.sleep(0.002)
    th_np = threading.Thread(target=_newer_prompt)
    th_np.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not m.turn_state()[1]:
        time.sleep(0.002)
    p_release.set()
    th_ph.join(timeout=2.0)
    th_od.join(timeout=2.0)
    th_np.join(timeout=2.0)
    assert order2 == [('decode', True), ('prompt', True)]

    # Forced timeout returns -1 and increments turn_timeouts (turn_stats()[5])
    hold_done = threading.Event()
    prompt_locked = threading.Event()

    def _hold_prompt():
        assert m.turn_try(stm_core.TURN_PROMPT, 1000) is True
        prompt_locked.set()
        hold_done.wait(timeout=2.0)
        m.turn_give()

    th_hp = threading.Thread(target=_hold_prompt)
    th_hp.start()
    assert prompt_locked.wait(timeout=2.0)
    before_timeouts = m.turn_stats()[5]
    timed_out = m.turn_take(stm_core.TURN_DECODE, 1001, 15.0)
    m.turn_give()
    hold_done.set()
    th_hp.join(timeout=2.0)
    assert timed_out == -1
    assert m.turn_stats()[5] == before_timeouts + 1


def test_cuda_qmm_room_and_weight_room(monkeypatch):
    from quanteeze import cuda_kernels
    ext = _require_stm()
    monkeypatch.setattr(mx.metal, 'is_available', lambda: False)

    x_small = mx.arange(32, dtype=mx.bfloat16).reshape(1, 2, 16)
    mx.eval(x_small)
    xr = cuda_kernels.qmm_room(x_small)
    mx.eval(xr)
    assert xr.shape == x_small.shape
    assert xr.dtype == x_small.dtype
    assert mx.array_equal(xr, x_small).item()

    x_large = mx.arange(16 * 16, dtype=mx.bfloat16).reshape(16, 16)
    mx.eval(x_large)
    assert cuda_kernels.qmm_room(x_large) is x_large

    m = ext.Model(1)
    for dt, vals in ((mx.uint8, np.arange(32, dtype=np.uint8).reshape(2, 16)),
                     (mx.bfloat16, np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(2, 16))):
        src = mx.array(vals).astype(dt)
        mx.eval(src)
        out = demand._with_weight_room(src)
        assert out.shape == src.shape
        assert out.dtype == src.dtype
        assert mx.array_equal(out, src).item()
        # Verify row_contiguous in C++ via register_quant
        m.register_quant(0, 0, out, None, None, 0, 0, 'raw', (-1, -1, -1))
