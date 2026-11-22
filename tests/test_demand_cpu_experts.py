# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for `quanteeze.cpu_experts` mailbox `Server`, CUDA `cpu_ask`/`cpu_wait`
handshake, and `quanteeze.demand.Demand` CPU expert offload (`QUANTEEZE_CPU_EXPERTS`)."""
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from quanteeze import cpu_experts as ce
from quanteeze import cuda_kernels, cuda_ple, demand, expert_qmv, q2_experts, requant

pytestmark = pytest.mark.skipif(ce.isa() is None, reason='no C++17 compiler here')

E, HIDDEN, INNER, GS, K = 16, 256, 128, 64, 4
LIMIT = 10.0


def _make_arrays_and_tensors(rng, mode, num_layers=1, num_experts=E, k=K, hidden=HIDDEN, inner=INNER):
    """Build quantized weights + checkpoint tensor dict for `mode` in (`'affine'`, `'mxfp4'`)."""
    tensors = {}
    specs = {}
    for layer in range(num_layers):
        layer_arrays = []
        projs = (('gate_proj', (inner, hidden)), ('up_proj', (inner, hidden)), ('down_proj', (hidden, inner)))
        for p, (n_out, n_in) in projs:
            w = mx.array(rng.normal(0, 0.05, (num_experts, n_out, n_in)).astype(np.float32)).astype(mx.bfloat16)
            if mode == 'affine':
                q, s, b = mx.quantize(w, group_size=GS, bits=4)
                parts = (('weight', q), ('scales', s.astype(mx.bfloat16)), ('biases', b.astype(mx.bfloat16)))
            else:
                q, s = mx.quantize(w, group_size=32, bits=4, mode='mxfp4')
                parts = (('weight', q), ('scales', s))
            mx.eval(*[a for _, a in parts])
            for name, a in parts:
                tensors[f'layers.{layer}.mlp.switch_mlp.{p}.{name}'] = a
                host = np.array(mx.view(a, mx.uint16) if a.dtype == mx.bfloat16 else a)
                layer_arrays.append((p, name, np.ascontiguousarray(host)))
        groups = {p: GS if mode == 'affine' else 32 for p, _ in projs}
        specs[layer] = ce.layer_spec(layer_arrays, groups, mode, 4)
    return tensors, specs


class _WeighedSwiGLU(nn.Module):
    def __init__(self, limit=LIMIT):
        super().__init__()
        self.limit = limit

    def __call__(self, up, gate, weights):
        from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
        g = mx.minimum(gate.astype(mx.float32), self.limit) if self.limit > 0 else gate.astype(mx.float32)
        u = mx.clip(up.astype(mx.float32), -self.limit, self.limit) if self.limit > 0 else up.astype(mx.float32)
        return fake_quant_fp8_ue8m0((nn.silu(g) * u * weights).astype(up.dtype))


class _WeighedSwitch(SwitchGLU):
    def __call__(self, x, indices, scores):
        shape = indices.shape
        x_exp = mx.expand_dims(x, (-2, -3))
        up = self.up_proj(x_exp, indices)
        gate = self.gate_proj(x_exp, indices)
        act = self.activation(up, gate, scores[..., None, None])
        return self.down_proj(act, indices).squeeze(-2).reshape(*shape, -1)


def _build_model(tensors, mode, num_layers=1, num_experts=E, hidden=HIDDEN, inner=INNER, limit=0.0):
    m = nn.Module()
    m.layers = []
    for layer in range(num_layers):
        if mode == 'mxfp4':
            sw = _WeighedSwitch(input_dims=hidden, hidden_dims=inner, num_experts=num_experts, bias=False,
                                activation=_WeighedSwiGLU(limit or LIMIT))
            nn.quantize(sw, group_size=32, bits=4, mode='mxfp4')
        else:
            from mlx_vlm.models.deepseek_v4.language import LimitedSwiGLU
            act = LimitedSwiGLU(limit) if limit > 0 else nn.SiLU()  # patched cleanly below
            sw = SwitchGLU(input_dims=hidden, hidden_dims=inner, num_experts=num_experts, bias=False)
            if limit > 0:
                sw.activation = act
            nn.quantize(sw, group_size=GS, bits=4)
        for p in ('gate_proj', 'up_proj', 'down_proj'):
            proj = getattr(sw, p)
            for part in ('weight', 'scales', 'biases'):
                key = f'layers.{layer}.mlp.switch_mlp.{p}.{part}'
                if key in tensors:
                    setattr(proj, part, tensors[key])
        mlp = nn.Module()
        mlp.switch_mlp = sw
        layer_mod = nn.Module()
        layer_mod.mlp = mlp
        m.layers.append(layer_mod)
    mx.eval(*tensors.values())
    return m


def _poll_answer(ans_row, stamp, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if int(ans_row[0]) == stamp:
            return True
        time.sleep(0.0005)
    return False


@pytest.mark.parametrize('mode', ['affine', 'mxfp4'])
def test_cpp_server_answers_match_reference(mode, monkeypatch):
    if mode == 'mxfp4':
        pytest.importorskip('mlx_vlm.models.deepseek_v41.fakequant')
    monkeypatch.setattr(ce, '_STATE', {'checked': ce.CHECKS, 'failed': None})
    rng = np.random.default_rng(11 if mode == 'affine' else 12)
    _, specs = _make_arrays_and_tensors(rng, mode, num_layers=1)
    spec = specs[0]
    x = rng.normal(0, 1, HIDDEN).astype(np.float32)
    slots = np.array([1, 5, 9], np.int32)
    weights = np.array([0.2, 0.35, 0.45], np.float32)
    pool = ce.Pool(2, priority='normal')
    server = ce.Server(pool, [3], HIDDEN, pinned=False)
    try:
        act_style = 'deepseek_v41' if mode == 'mxfp4' else 'plain'
        lim = LIMIT if mode == 'mxfp4' else 0.0
        server.add(3, spec, lim, act_style)
        server.start()

        b = server.box[3]
        b[ce.SERVE_HEAD:ce.SERVE_HEAD + HIDDEN].view(np.float32)[:] = x
        b[1] = len(slots)
        b[2:2 + len(slots)] = slots
        b[2 + ce.SERVE_MMAX:2 + ce.SERVE_MMAX + len(slots)].view(np.float32)[:] = weights
        stamp = int(b[0]) + 1
        b[0] = stamp
        assert _poll_answer(server.ans[3], stamp)
        assert int(server.ans[3, 1]) == len(slots)
        assert server.ans[3, 2:2 + len(slots)].tolist() == [1, 1, 1]
        ans_out = server.ans[3, ce.SERVE_HEAD:].view(np.float32).reshape(ce.SERVE_MMAX, HIDDEN)[:len(slots)].copy()
        want = np.stack([
            ce.native_reference(
                ce.layer_expert(spec, int(slot)), x, lim, float(w), act_style,
                q8=pool.q8 and pool.isa >= 1,
            )
            for slot, w in zip(slots, weights)
        ], axis=0)
        assert np.abs(want).max() > 0
        assert np.abs(ans_out - want).max() <= 1e-2 * np.abs(want).max()
        counts = server.counts()
        assert counts['requests'] == 1 and counts['experts'] == 3 and counts['failed'] == 0
        assert server.quiet()
    finally:
        server.close()
        pool.close()


@pytest.mark.parametrize('mode', ['affine', 'mxfp4'])
def test_demand_cpu_offload_direct_pool_matches_stock_and_preserves_bank(tmp_path, monkeypatch, mode):
    if mode == 'mxfp4':
        pytest.importorskip('mlx_vlm.models.deepseek_v41.fakequant')
    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'on')
    monkeypatch.setenv('QUANTEEZE_CPU_THREADS', '2')
    monkeypatch.setenv('QUANTEEZE_CPU_PRIORITY', 'normal')
    monkeypatch.setattr(cuda_kernels, 'available', lambda: False)

    rng = np.random.default_rng(101 if mode == 'affine' else 102)
    tensors, _ = _make_arrays_and_tensors(rng, mode, num_layers=1)
    (tmp_path / 'config.json').write_text('{}', encoding='utf-8')
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)

    limit = LIMIT if mode == 'mxfp4' else 2.5
    m_ref = _build_model(tensors, mode, num_layers=1, limit=limit)
    expert_qmv.install_switch(m_ref.layers[0].mlp.switch_mlp)
    m_dem = _build_model(tensors, mode, num_layers=1, limit=limit)

    ckpt = q2_experts.Checkpoint(tmp_path)
    held = q2_experts.hold(ckpt, budget=10**9, workers=2, by_rows=True)
    dem = None
    try:
        held.wait()
        assert held.holds_layer(0)
        dem = demand.Demand(m_dem, ckpt, budget_bytes=1, workers=2)
        assert dem.slots == 8
        initial_bank = dem.switches[0].experts.copy()

        # Route mixes 2 GPU-resident hits (0, 1) and 2 host-resident misses (10, 14).
        x = mx.array(rng.normal(0, 0.5, (1, 1, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
        inds = mx.array([[[0, 10, 1, 14]]], dtype=mx.uint32)
        scores = mx.array([[[0.2, 0.3, 0.15, 0.35]]], dtype=mx.float32) if mode == 'mxfp4' else None
        mx.eval(x, inds)

        want = m_ref.layers[0].mlp.switch_mlp(x, inds, scores) if mode == 'mxfp4' else m_ref.layers[0].mlp.switch_mlp(x, inds)
        got = m_dem.layers[0].mlp.switch_mlp(x, inds, scores)
        mx.eval(want, got)

        a = np.array(got.astype(mx.float32))[0, 0]
        b = np.array(want.astype(mx.float32))[0, 0]
        # Resident slots (0, 2) came from the same GPU decode path; CPU slots (1, 3) match within rounding.
        assert np.array_equal(a[[0, 2]], b[[0, 2]])
        scale = max(float(np.abs(b[[1, 3]]).max()), 1e-6)
        assert float(np.abs(a[[1, 3]] - b[[1, 3]]).max()) <= 0.05 * scale

        # CPU offload preserved the GPU bank without evicting resident experts.
        assert np.array_equal(dem.switches[0].experts, initial_bank)
        assert dem.cpu_served == 2
        assert dem.host_served == 2
        assert dem.summary()['cpu_served'] == 2
    finally:
        if dem is not None:
            dem.close()
        q2_experts.let_go(ckpt.path)


def test_demand_partial_host_and_ssd_miss_with_mock_cuda_server(tmp_path, monkeypatch):
    """Exercise the `Server` + `CPU_ASK`/`CPU_WAIT` + `Load` hysteresis + mixed GPU/CPU/SSD miss
    codepaths synchronously via a mapped host mailbox shim."""
    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'on')
    monkeypatch.setenv('QUANTEEZE_CPU_THREADS', '2')
    monkeypatch.setenv('QUANTEEZE_CPU_PRIORITY', 'normal')

    rng = np.random.default_rng(303)
    tensors, _ = _make_arrays_and_tensors(rng, 'affine', num_layers=1)
    (tmp_path / 'config.json').write_text('{}', encoding='utf-8')
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)

    host_boxes = {}

    def fake_pinned(shape, dtype, what='', otherwise=''):
        arr = np.zeros(shape, dtype)
        host_boxes[arr.ctypes.data] = arr
        return arr

    def fake_device(arr):
        return int(arr.__array_interface__['data'][0])

    def make_ask():
        def call(inputs, template, grid, threadgroup, output_shapes, output_dtypes):
            rows_mx, inv_mx, weights_mx, x_mx, addr_mx = inputs
            tmap = dict(template)
            b_n, m_n, k_n, d_n = tmap.get('B', 1), tmap['M'], tmap['K'], tmap['D']
            head, mmax, weighs = tmap['HEAD'], tmap['MMAX'], tmap['WEIGHS']
            box_ptr = int(np.array(addr_mx)[0])
            raw_box = host_boxes_lookup(box_ptr)
            box = raw_box[:head + b_n * d_n]
            rows_np = np.array(rows_mx, dtype=np.int32).reshape(-1)
            inv_np = np.array(inv_mx, dtype=np.int32).reshape(-1)
            w_np = np.array(weights_mx, dtype=np.float32).reshape(-1)
            x_np = np.array(x_mx.astype(mx.float32)).reshape(-1)
            if not np.any(rows_np >= 0):
                return [mx.array([0, 0, 0, 0], dtype=mx.int32)]
            stamp = int(box[0]) + 1
            box[head:head + b_n * d_n].view(np.float32)[:] = x_np
            box[1] = m_n
            box[2 + 3 * mmax] = b_n
            k_per = (k_n // b_n) if b_n > 1 else k_n
            for m in range(m_n):
                box[2 + m] = int(rows_np[m])
                wt = 1.0
                xr = 0
                for j in range(k_n):
                    if int(inv_np[j]) == m:
                        if weighs:
                            wt = float(w_np[j])
                        xr = j // k_per if k_per > 0 else 0
                        break
                box[2 + mmax + m:2 + mmax + m + 1].view(np.float32)[0] = wt
                box[2 + 2 * mmax + m] = xr
            box[0] = stamp
            return [mx.array([stamp, 0, 0, 0], dtype=mx.int32)]
        return call

    def make_wait():
        def call(inputs, template, grid, threadgroup, output_shapes, output_dtypes):
            token_mx, y_mx, inv_mx, addr_mx = inputs
            tmap = dict(template)
            m_n, k_n, d_n, head, timeout_ns = tmap['M'], tmap['K'], tmap['D'], tmap['HEAD'], tmap['TIMEOUT_NS']
            ans_ptr = int(np.array(addr_mx)[1])
            ans = host_boxes_lookup(ans_ptr)
            stamp = int(np.array(token_mx)[0])
            deadline = time.monotonic() + timeout_ns / 1e9
            got = 0
            while stamp != 0 and time.monotonic() < deadline:
                if int(ans[0]) == stamp:
                    got = 1
                    break
                time.sleep(0.0005)
            status = mx.array([got, int(stamp != 0 and not got), int(stamp == 0), 5], dtype=mx.int32)
            out_np = np.array(y_mx.astype(mx.float32)).reshape(k_n, d_n).copy()
            inv_np = np.array(inv_mx, dtype=np.int32).reshape(-1)
            ansf = ans[head:].view(np.float32).reshape(ce.SERVE_MMAX, d_n)
            for j in range(k_n):
                m = int(inv_np[j])
                if got and 0 <= m < m_n and int(ans[2 + m]) != 0:
                    out_np[j] = ansf[m]
            return [mx.array(out_np).astype(y_mx.dtype).reshape(-1), status]
        return call

    def host_boxes_lookup(ptr):
        for base, arr in host_boxes.items():
            end = base + arr.nbytes
            if base <= ptr < end:
                off = (ptr - base) // 4
                return arr.reshape(-1)[off:]
        raise KeyError(ptr)

    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
    monkeypatch.setattr(demand.mx.metal, 'is_available', lambda: False)
    monkeypatch.setattr(cuda_ple, 'pinned', fake_pinned)
    monkeypatch.setattr(cuda_ple, '_device', fake_device)
    monkeypatch.setattr(ce, '_cpu_ask_kernel', make_ask)
    monkeypatch.setattr(ce, '_cpu_wait_kernel', make_wait)

    m_ref = _build_model(tensors, 'affine', num_layers=1)
    expert_qmv.install_switch(m_ref.layers[0].mlp.switch_mlp)
    m_dem = _build_model(tensors, 'affine', num_layers=1)

    # Hold only 12 of 16 experts in HostCopy (one expert size * 12) so experts 12..15 remain SSD-only!
    ckpt = q2_experts.Checkpoint(tmp_path)
    one_exp = sum(ckpt.nbytes(ckpt.name(0, p, part)) for p in q2_experts.PROJECTIONS for part in q2_experts.NAMES) // E
    held = q2_experts.hold(ckpt, budget=one_exp * 12 + 128, workers=2, by_rows=True)
    dem = None
    try:
        held.wait()
        assert held.expert_in[0].tolist() == list(range(12))
        dem = demand.Demand(m_dem, ckpt, budget_bytes=1, workers=2)
        assert dem.slots == 8

        # Expert 0 in GPU bank, experts 8 & 10 in HostCopy, expert 14 on SSD only!
        x = mx.array(rng.normal(0, 0.5, (1, 1, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
        inds = mx.array([[[0, 8, 14, 10]]], dtype=mx.uint32)
        mx.eval(x, inds)

        want = m_ref.layers[0].mlp.switch_mlp(x, inds)
        got = m_dem.layers[0].mlp.switch_mlp(x, inds)
        mx.eval(want, got)

        a = np.array(got.astype(mx.float32))[0, 0]
        b = np.array(want.astype(mx.float32))[0, 0]
        scale = max(float(np.abs(b).max()), 1e-6)
        assert float(np.abs(a - b).max()) <= 0.02 * scale

        summary = dem.summary()
        assert dem._cpu is not None and dem._cpu.server is not None
        assert 14 in dem.switches[0].experts
        assert 8 not in dem.switches[0].experts and 10 not in dem.switches[0].experts
        assert summary['cpu_served'] == 2
        assert summary['host_served'] == 2
        assert summary['cpu_requests'] == 1
        assert summary['cpu_experts'] == 2
        assert summary['cpu_failed'] == 0
        assert summary['cpu_late'] == 0

        # Also test direct cpu_ask / cpu_wait no-request and timeout cases.
        addr = dem._cpu.server.addr[0]
        y_init = mx.array(rng.normal(0, 1, (1, 1, K, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
        inv_empty = mx.full((K,), -1, dtype=mx.int32)
        tok_empty = ce.cpu_ask(addr, x, mx.full((3,), -1, dtype=mx.int32), inv_empty)
        got_empty, st_empty = ce.cpu_wait(addr, tok_empty, y_init, inv_empty, most=3)
        assert np.array(st_empty).tolist()[:3] == [0, 0, 1]
        assert mx.array_equal(got_empty, y_init).item()

        # Timeout test when nobody answers stamp.
        monkeypatch.setattr(ce, 'CPU_TIMEOUT_NS', 10_000_000)
        dem._cpu.server.close()
        inv_late = mx.array([0, 1, -1, -1], dtype=mx.int32)
        tok_late = ce.cpu_ask(addr, x, mx.array([0, 1], dtype=mx.int32), inv_late)
        got_late, st_late = ce.cpu_wait(addr, tok_late, y_init, inv_late, most=2)
        assert np.array(st_late).tolist()[:2] == [0, 1]
        assert mx.array_equal(got_late, y_init).item()
    finally:
        if dem is not None:
            dem.close()
        q2_experts.let_go(ckpt.path)


def test_demand_auto_yields_to_gpu_staging_when_load_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'auto')
    monkeypatch.setattr(cuda_kernels, 'available', lambda: False)

    rng = np.random.default_rng(404)
    tensors, _ = _make_arrays_and_tensors(rng, 'affine', num_layers=1)
    (tmp_path / 'config.json').write_text('{}', encoding='utf-8')
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)

    m_dem = _build_model(tensors, 'affine', num_layers=1)
    ckpt = q2_experts.Checkpoint(tmp_path)
    held = q2_experts.hold(ckpt, budget=10**9, workers=2, by_rows=True)
    dem = None
    try:
        held.wait()
        dem = demand.Demand(m_dem, ckpt, budget_bytes=1, workers=2)
        dem._cpu_load.workers = 0
        x = mx.array(rng.normal(0, 0.5, (1, 1, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
        inds = mx.array([[[0, 9, 10, 11]]], dtype=mx.uint32)
        mx.eval(x, inds)
        y = m_dem.layers[0].mlp.switch_mlp(x, inds)
        mx.eval(y)
        assert dem.cpu_served == 0
        assert dem.host_served == 3
    finally:
        if dem is not None:
            dem.close()
        q2_experts.let_go(ckpt.path)


def test_openblas_one_thread_returns_bool():
    assert isinstance(ce.openblas_one_thread(), bool)


def test_demand_cpu_offload_byte_packed_i8_mxfp4_safetensors(tmp_path, monkeypatch):
    """Verify DeepSeek V4.1 native `'I8'` byte-packed FP4 + `'F8_E8M0'` safetensors checkpoints work
    end-to-end with `HostCopy(by_rows=True)` and `QUANTEEZE_CPU_EXPERTS=on`."""
    import json
    import struct
    pytest.importorskip('mlx_vlm.models.deepseek_v41.fakequant')
    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'on')
    monkeypatch.setenv('QUANTEEZE_CPU_THREADS', '2')
    monkeypatch.setenv('QUANTEEZE_CPU_PRIORITY', 'normal')
    monkeypatch.setattr(cuda_kernels, 'available', lambda: False)

    rng = np.random.default_rng(505)
    tensors_u32, _ = _make_arrays_and_tensors(rng, 'mxfp4', num_layers=1)

    # Serialize one safetensors shard with DeepSeek V4.1's raw `I8` and `F8_E8M0` dtypes.
    header = {}
    payload = bytearray()
    for key, arr in tensors_u32.items():
        np_arr = np.ascontiguousarray(np.array(arr))
        if key.endswith('.weight'):
            st_dtype = 'I8'
            raw = np_arr.view(np.uint8)
            shape = list(raw.shape)
        else:
            st_dtype = 'F8_E8M0'
            raw = np_arr.view(np.uint8)
            shape = list(raw.shape)
        start = len(payload)
        payload.extend(raw.tobytes())
        header[key] = {'dtype': st_dtype, 'shape': shape, 'data_offsets': [start, len(payload)]}
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    (tmp_path / 'config.json').write_text('{}', encoding='utf-8')
    with open(tmp_path / 'model.safetensors', 'wb') as f:
        f.write(struct.pack('<Q', len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)

    m_ref = _build_model(tensors_u32, 'mxfp4', num_layers=1)
    expert_qmv.install_switch(m_ref.layers[0].mlp.switch_mlp)
    m_dem = _build_model(tensors_u32, 'mxfp4', num_layers=1)

    ckpt = q2_experts.Checkpoint(tmp_path)
    assert ckpt.mode(0, 'gate_proj') == 'mxfp4'
    assert ckpt.tensors[ckpt.name(0, 'gate_proj', 'weight')][1] == 'I8'
    held = q2_experts.hold(ckpt, budget=10**9, workers=2, by_rows=True)
    dem = None
    try:
        held.wait()
        dem = demand.Demand(m_dem, ckpt, budget_bytes=1, workers=2)
        initial_bank = dem.switches[0].experts.copy()
        x = mx.array(rng.normal(0, 0.5, (1, 1, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
        inds = mx.array([[[0, 9, 2, 13]]], dtype=mx.uint32)
        scores = mx.array([[[0.25, 0.25, 0.25, 0.25]]], dtype=mx.float32)
        mx.eval(x, inds, scores)

        want = m_ref.layers[0].mlp.switch_mlp(x, inds, scores)
        got = m_dem.layers[0].mlp.switch_mlp(x, inds, scores)
        mx.eval(want, got)

        a = np.array(got.astype(mx.float32))[0, 0]
        b = np.array(want.astype(mx.float32))[0, 0]
        assert np.array_equal(a[[0, 2]], b[[0, 2]])
        scale = max(float(np.abs(b[[1, 3]]).max()), 1e-6)
        assert float(np.abs(a[[1, 3]] - b[[1, 3]]).max()) <= 0.05 * scale
        assert np.array_equal(dem.switches[0].experts, initial_bank)
        assert dem.cpu_served == 2
        assert dem.host_served == 2
    finally:
        if dem is not None:
            dem.close()
        q2_experts.let_go(ckpt.path)
