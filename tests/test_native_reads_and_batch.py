# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for lock-free C++ SSD readers (`native_reads`), `QUANTEEZE_VISION_SSD=auto` + macOS
mount detection (`vision_ssd`, `q2_experts`), and concurrent batch decode `B=2..8` with CPU offload."""
import gc
import json
import struct
import sys
import threading
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from quanteeze import cpu_experts as ce
from quanteeze import (
    cuda_kernels,
    cuda_ple,
    demand,
    expert_qmv,
    gguf_experts,
    gguf_qmv,
    iq_experts,
    native_reads,
    q2_experts,
    requant,
    vision_ssd,
)


def _write_gguf(path, tensors, alignment=32):
    def pack_str(s):
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b

    buf = bytearray()
    buf += b'GGUF'
    buf += struct.pack('<IQQ', 3, len(tensors), 1)
    buf += pack_str('general.alignment') + struct.pack('<II', 4, alignment)
    offset = 0
    blobs = []
    for name, dims, kind, data in tensors:
        buf += pack_str(name)
        buf += struct.pack('<I', len(dims))
        buf += struct.pack(f'<{len(dims)}Q', *dims)
        buf += struct.pack('<IQ', kind, offset)
        pad = (-len(data)) % alignment
        blobs.append(data + b'\x00' * pad)
        offset += len(data) + pad
    buf += b'\x00' * ((-len(buf)) % alignment)
    with open(path, 'wb') as f:
        f.write(buf)
        for blob in blobs:
            f.write(blob)


# ---------------------------------------------------------------- 1. Lock-free C++ native_reads


@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 compiler here')
def test_native_reads_multi_segment_slab_growth_cancel_and_refusal(tmp_path):
    f1 = tmp_path / 'a.bin'
    f2 = tmp_path / 'b.bin'
    data1 = np.arange(8192, dtype=np.uint8)
    data2 = (np.arange(8192, dtype=np.uint8) * 3 + 7).astype(np.uint8)
    f1.write_bytes(data1.tobytes())
    f2.write_bytes(data2.tobytes())

    rd = native_reads.Readers(workers=2)
    try:
        # Multi-segment read across two files, uncached and cached
        dst1 = np.zeros(512, dtype=np.uint8)
        dst2 = np.zeros(256, dtype=np.uint8)
        fut = rd.read(
            [(f1, 100, 512, dst1.ctypes.data), (f2, 300, 256, dst2.ctypes.data)],
            finish=lambda: (dst1.copy(), dst2.copy()),
            cache=False,
        )
        got1, got2 = fut.result(timeout=5.0)
        np.testing.assert_array_equal(got1, data1[100:612])
        np.testing.assert_array_equal(got2, data2[300:556])

        # Ticket recycling beyond first slab (5000 completed sequential reads)
        tiny = np.zeros(64, dtype=np.uint8)
        for step in range(0, 5000, 32):
            futs = [
                rd.read([(f1, (step + k) % 4096, 64, tiny.ctypes.data)], cache=True, priority=k % 2)
                for k in range(32)
            ]
            for f in futs:
                assert f is not None
                f.result(timeout=5.0)

        counts = rd.counts()
        assert counts['completed'] >= 5000
        assert counts['failed'] == 0

        # Short read past EOF raises OSError
        bad = np.zeros(128, dtype=np.uint8)
        f_err = rd.read([(f1, 8150, 128, bad.ctypes.data)], cache=True)
        with pytest.raises(OSError):
            f_err.result(timeout=5.0)
    finally:
        rd.close()

    # Deterministic slab expansion across all 4 slabs (16,384 tickets), refusal on 16,385th, and cancel
    q = native_reads.Readers(workers=0)
    try:
        held = []
        dst = np.zeros(64, dtype=np.uint8)
        for i in range(4096 * 4):
            fut = q.read([(f1, 0, 64, dst.ctypes.data)], priority=i % 2)
            assert fut is not None, f'failed allocating ticket {i}'
            held.append(fut)
        assert q.read([(f1, 0, 64, dst.ctypes.data)]) is None
        assert q.counts()['refused'] == 1

        # Cancelling one queued ticket frees its ticket cleanly after both refs drop
        assert held[0].cancel() is True
        assert held[0].cancelled() is True
        assert q.counts()['cancelled'] == 1
        with pytest.raises(RuntimeError):
            held[0].result(timeout=0.1)

        # Replacement ticket reuses the freed slot with incremented generation
        rep = q.read([(f1, 0, 64, dst.ctypes.data)])
        assert rep is not None
        rep.cancel()
    finally:
        q.close()


def _make_moe_checkpoint(tmp_path, container, seed=55, n_exp=16, hidden=256, inner=256, gs=64):
    rng = np.random.default_rng(seed)
    m = nn.Module()
    if container.startswith('gguf'):
        sw = SwitchGLU(input_dims=hidden, hidden_dims=inner, num_experts=n_exp, bias=False)
        gguf_kind = {
            'gate_proj': iq_experts.Q4_K,
            'up_proj': iq_experts.IQ2_XXS,
            'down_proj': iq_experts.Q2_K,
        }
        gguf_name = {'gate_proj': 'ffn_gate', 'up_proj': 'ffn_up', 'down_proj': 'ffn_down'}
        entries = []
        f16_offsets = {
            iq_experts.Q4_K: (0, 2),
            iq_experts.IQ2_XXS: (0,),
            iq_experts.Q2_K: (80, 82),
        }
        scale_bytes = np.frombuffer(np.float16(0.05).tobytes(), dtype=np.uint8)
        for p, kind in gguf_kind.items():
            out_d, in_d = (hidden, inner) if p == 'down_proj' else (inner, hidden)
            kb = in_d // iq_experts.per_block(kind)
            bb = iq_experts.BLOCK[kind]
            raw = rng.integers(0, 256, size=(n_exp, out_d, kb, bb), dtype=np.uint8)
            for off in f16_offsets[kind]:
                raw[:, :, :, off:off + 2] = scale_bytes
            if container == 'gguf_stacked':
                entries.append((f'blk.0.{gguf_name[p]}_exps.weight', (in_d, out_d, n_exp), kind, raw.tobytes()))
            else:
                for e in range(n_exp):
                    entries.append((f'blk.0.{gguf_name[p]}.{e}.weight', (in_d, out_d), kind, raw[e].tobytes()))
        ckpt_path = tmp_path / 'experts.gguf'
        _write_gguf(ckpt_path, entries)
        src = gguf_experts.ExpertGGUF(ckpt_path)
        mlp = nn.Module()
        mlp.switch_mlp = sw
        layer = nn.Module()
        layer.mlp = mlp
        m.layers = [layer]
        return m, src, rng

    sw = SwitchGLU(input_dims=hidden, hidden_dims=inner, num_experts=n_exp, bias=False)
    sw.gate_proj = sw.gate_proj.to_quantized(group_size=gs, bits=4)
    sw.up_proj = sw.up_proj.to_quantized(group_size=gs, bits=4)
    sw.down_proj = sw.down_proj.to_quantized(group_size=gs, bits=4)
    tensors = {}
    for p, short in (('gate_proj', 'w1'), ('up_proj', 'w3'), ('down_proj', 'w2')):
        out_d, in_d = (hidden, inner) if p == 'down_proj' else (inner, hidden)
        w = mx.array(rng.normal(0, 0.05, (n_exp, out_d, in_d)).astype(np.float32)).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=gs, bits=4)
        mx.eval(q, s, b)
        for part, arr in (('weight', q), ('scales', s.astype(mx.bfloat16)), ('biases', b.astype(mx.bfloat16))):
            mx.eval(arr)
            if container in ('safetensors_stacked', 'quant_cache'):
                tensors[f'layers.0.mlp.switch_mlp.{p}.{part}'] = arr
            else:
                for e in range(n_exp):
                    tensors[f'layers.0.ffn.experts.{e}.{short}.{part}'] = arr[e]
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)
    (tmp_path / 'config.json').write_text(
        json.dumps({'quantization': {'group_size': gs, 'bits': 4}}),
        encoding='utf-8',
    )
    mlp = nn.Module()
    mlp.switch_mlp = sw
    layer = nn.Module()
    layer.mlp = mlp
    m.layers = [layer]
    if container == 'quant_cache':
        src = requant.cached_checkpoint(m.layers, tmp_path, bits=3, group_size=gs)
    else:
        src = q2_experts.Checkpoint(tmp_path)
    return m, src, rng


@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 compiler here')
@pytest.mark.parametrize(
    'container',
    ['safetensors_stacked', 'safetensors_per_expert', 'quant_cache', 'gguf_stacked', 'gguf_per_expert'],
)
def test_demand_native_reads_matches_python_across_containers(tmp_path, monkeypatch, container):
    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')

    sub_off = tmp_path / 'off'
    sub_off.mkdir()
    monkeypatch.setenv('QUANTEEZE_QUANT_CACHE', str(sub_off / 'qcache'))
    m_off, src_off, rng = _make_moe_checkpoint(sub_off, container, seed=81)

    sub_on = tmp_path / 'on'
    sub_on.mkdir()
    monkeypatch.setenv('QUANTEEZE_QUANT_CACHE', str(sub_on / 'qcache'))
    m_on, src_on, _ = _make_moe_checkpoint(sub_on, container, seed=81)

    x_pre = mx.array(rng.normal(0, 0.3, (1, 32, 256)).astype(np.float32)).astype(mx.bfloat16)
    idx_pre = mx.array(rng.integers(0, 16, (1, 32, 4)).astype(np.uint32))
    x_dec = mx.array(rng.normal(0, 0.3, (1, 1, 256)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(x_pre, idx_pre, x_dec)

    monkeypatch.setenv('QUANTEEZE_DEMAND_READ_NATIVE', 'off')
    monkeypatch.setattr(gguf_qmv, 'CHOICE', {})
    monkeypatch.setattr(gguf_qmv, 'CHECKED', {})
    dem_off = demand.Demand(m_off, src_off, budget_bytes=1, workers=2)
    try:
        y_pre_off = m_off.layers[0].mlp.switch_mlp(x_pre, idx_pre)
        mx.eval(y_pre_off)
        missing_after_pre = [e for e in range(16) if dem_off.switches[0].route[e] < 0]
        assert len(missing_after_pre) >= 4
        idx_dec = mx.array([[missing_after_pre[:4]]], dtype=mx.uint32)
        mx.eval(idx_dec)
        y_dec_off = m_off.layers[0].mlp.switch_mlp(x_dec, idx_dec)
        mx.eval(y_dec_off)
        assert dem_off.native_reads == 0
        off_route = dem_off.switches[0].route.copy()
        def _raw(a):
            return np.array(mx.view(a, mx.uint16) if a.dtype == mx.bfloat16 else a)

        off_weights = {
            (p, part): _raw(getattr(getattr(dem_off.switches[0], p), part))
            for p in q2_experts.PROJECTIONS
            for part in q2_experts.NAMES
            if getattr(getattr(dem_off.switches[0], p), part, None) is not None
        }
    finally:
        dem_off.close()

    monkeypatch.setattr(native_reads, '_STATE', {'checked': 0, 'failed': None})
    monkeypatch.setattr(gguf_qmv, 'CHOICE', {})
    monkeypatch.setattr(gguf_qmv, 'CHECKED', {})
    monkeypatch.setenv('QUANTEEZE_DEMAND_READ_NATIVE', 'on')
    dem_on = demand.Demand(m_on, src_on, budget_bytes=1, workers=2)
    try:
        assert dem_on._native is not None
        y_pre_on = m_on.layers[0].mlp.switch_mlp(x_pre, idx_pre)
        mx.eval(y_pre_on)
        pre_reads = dem_on.summary()['native_reads']
        y_dec_on = m_on.layers[0].mlp.switch_mlp(x_dec, idx_dec)
        mx.eval(y_dec_on)
        assert dem_on.summary()['native_reads'] > pre_reads
        assert np.array_equal(dem_on.switches[0].route, off_route)
        for (p, part), want_arr in off_weights.items():
            got_arr = _raw(getattr(getattr(dem_on.switches[0], p), part))
            assert np.array_equal(got_arr, want_arr)
        assert mx.array_equal(y_pre_on, y_pre_off).item()
        assert mx.array_equal(y_dec_on, y_dec_off).item()
    finally:
        dem_on.close()


@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 compiler here')
def test_demand_native_reads_verification_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')
    monkeypatch.setenv('QUANTEEZE_DEMAND_READ_NATIVE', 'auto')
    monkeypatch.setattr(native_reads, '_STATE', {'checked': 0, 'failed': None})

    m, src, rng = _make_moe_checkpoint(tmp_path, 'safetensors_stacked', seed=99)
    dem = demand.Demand(m, src, budget_bytes=1, workers=2)
    try:
        orig_from_file = src._from_file

        def tampered_ref(name, rows=None, cache=False, out=None):
            arr = orig_from_file(name, rows=rows, cache=cache, out=out)
            return (arr ^ np.uint8(1)) if arr.dtype == np.uint8 else (arr ^ 1)

        monkeypatch.setattr(src, '_from_file', tampered_ref)
        x = mx.array(rng.normal(0, 0.3, (1, 1, 256)).astype(np.float32)).astype(mx.bfloat16)
        idx = mx.array([[[12, 13, 14, 15]]], dtype=mx.uint32)
        mx.eval(x, idx)
        _ = m.layers[0].mlp.switch_mlp(x, idx)
        assert native_reads._STATE['failed'] is not None
        assert not native_reads.enabled()
    finally:
        dem.close()


# ---------------------------------------------------------------- 2. QUANTEEZE_VISION_SSD=auto & macOS mount


def test_macos_mount_network_fs_detection(monkeypatch):
    mount_out = (
        '/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n'
        '//user@nas.local/models on /Volumes/Models (smbfs, nodev, nosuid, mounted by user)\n'
        'nas.local:/exports/lab on /Volumes/Lab Share (nfs, nodev, nosuid)\n'
    )
    monkeypatch.setattr(sys, 'platform', 'darwin')
    read_text = q2_experts.Path.read_text

    def no_proc(self, *args, **kwargs):                # macOS has no /proc/mounts; hide Linux's when run there
        if str(self) == '/proc/mounts':
            raise FileNotFoundError(str(self))
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(q2_experts.Path, 'read_text', no_proc)
    monkeypatch.setattr(
        q2_experts.subprocess,
        'run',
        lambda *a, **kw: type('P', (), {'stdout': mount_out})(),
    )
    assert q2_experts.file_system('/Volumes/Models/DeepSeek-V4.1') == 'smbfs'
    assert q2_experts.on_network('/Volumes/Models/DeepSeek-V4.1') is True
    assert q2_experts.file_system('/Volumes/Lab Share/Qwen3.8') == 'nfs'
    assert q2_experts.on_network('/Volumes/Lab Share/Qwen3.8') is True
    assert q2_experts.file_system('/Users/example/ckpt') == 'apfs'
    assert q2_experts.on_network('/Users/example/ckpt') is False


def test_vision_ssd_auto_holds_when_roomy_streams_when_tight_and_holds_on_network(tmp_path, monkeypatch):
    from quanteeze import guards
    from quanteeze.arch import deepseek_v41 as q_v41

    class VisionBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.proj = nn.Linear(dim, dim, bias=False)

        def __call__(self, x):
            return self.proj(x)

    class VisionTower(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.blocks = [VisionBlock(dim), VisionBlock(dim)]
            self.post = nn.Linear(dim, dim, bias=False)

        def __call__(self, x):
            for b in self.blocks:
                x = b(x)
            return self.post(x)

    def build_model():
        mx.random.seed(123)
        m = nn.Module()
        m.vision_tower = VisionTower(64)
        for k, a in tree_flatten(m.vision_tower.parameters()):
            vision_ssd.set_param(m.vision_tower, k, a.astype(mx.bfloat16))
        sw = SwitchGLU(input_dims=128, hidden_dims=64, num_experts=8, bias=False)
        sw.gate_proj = sw.gate_proj.to_quantized(group_size=64, bits=4)
        sw.up_proj = sw.up_proj.to_quantized(group_size=64, bits=4)
        sw.down_proj = sw.down_proj.to_quantized(group_size=64, bits=4)
        mlp = nn.Module()
        mlp.switch_mlp = sw
        layer = nn.Module()
        layer.mlp = mlp
        lm = nn.Module()
        lm.layers = [layer]
        m.language_model = lm
        mx.eval(m.parameters())
        return m

    model_ref = build_model()
    tensors = {}
    for k, a in tree_flatten(model_ref.vision_tower.parameters()):
        tensors[f'vision_tower.{k}'] = a
    for k, a in tree_flatten(model_ref.language_model.parameters()):
        tensors[f'language_model.{k}'] = a
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp'}), encoding='utf-8')

    x_img = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)
    want_img = model_ref.vision_tower(x_img)
    mx.eval(want_img)

    monkeypatch.delenv('QUANTEEZE_VISION_SSD', raising=False)
    monkeypatch.delenv('QUANTEEZE_COMPACT', raising=False)
    monkeypatch.delenv('QUANTEEZE_LOAD_WHOLE', raising=False)
    monkeypatch.delenv('QUANTEEZE_MEMORY_RESERVE', raising=False)
    monkeypatch.setattr(q2_experts, 'file_system', lambda _p: 'apfs')

    # 1) Roomy memory budget -> held (freed == 0)
    vision_ssd._AUTO.clear()
    vision_ssd.IMAGES.clear()
    m_roomy = build_model()
    monkeypatch.setattr(guards, 'recommended_budget', lambda: 1 << 30)
    assert vision_ssd.install(m_roomy, tmp_path) == 0
    assert sum(a.nbytes for _, a in tree_flatten(m_roomy.vision_tower.parameters())) > 0
    assert q_v41.images_mode(tmp_path, model=m_roomy) == 'on'

    # 2) Tight memory budget on local SSD -> released & streamed bit-for-bit
    vision_ssd._AUTO.clear()
    vision_ssd.IMAGES.clear()
    m_tight = build_model()
    monkeypatch.setattr(guards, 'recommended_budget', lambda: 100_000)
    freed = vision_ssd.install(m_tight, tmp_path)
    assert freed > 0
    assert sum(a.nbytes for _, a in tree_flatten(m_tight.vision_tower.parameters())) == 0
    got_img = m_tight.vision_tower(x_img)
    mx.eval(got_img)
    assert mx.array_equal(want_img, got_img).item()
    assert q_v41.images_mode(tmp_path, model=m_tight) == 'ssd'

    # 3) Same tight budget on network FS -> kept in memory (held)
    vision_ssd._AUTO.clear()
    m_net = build_model()
    monkeypatch.setattr(q2_experts, 'file_system', lambda _p: 'smbfs')
    assert vision_ssd.install(m_net, tmp_path) == 0
    assert sum(a.nbytes for _, a in tree_flatten(m_net.vision_tower.parameters())) > 0
    assert q_v41.images_mode(tmp_path, model=m_net) == 'on'
    vision_ssd.IMAGES.clear()
    vision_ssd._AUTO.clear()


# ---------------------------------------------------------------- 3. Concurrent batch decode B=2..8 & CPU offload

_CPU_E, _CPU_HIDDEN, _CPU_INNER, _CPU_GS, _CPU_K = 16, 256, 128, 64, 4


def _make_cpu_arrays_and_tensors(rng, mode, num_layers=1, num_experts=_CPU_E, hidden=_CPU_HIDDEN, inner=_CPU_INNER):
    tensors = {}
    specs = {}
    for layer in range(num_layers):
        layer_arrays = []
        projs = (('gate_proj', (inner, hidden)), ('up_proj', (inner, hidden)), ('down_proj', (hidden, inner)))
        for p, (n_out, n_in) in projs:
            w = mx.array(rng.normal(0, 0.05, (num_experts, n_out, n_in)).astype(np.float32)).astype(mx.bfloat16)
            if mode == 'affine':
                q, s, b = mx.quantize(w, group_size=_CPU_GS, bits=4)
                parts = (('weight', q), ('scales', s.astype(mx.bfloat16)), ('biases', b.astype(mx.bfloat16)))
            else:
                q, s = mx.quantize(w, group_size=32, bits=4, mode='mxfp4')
                parts = (('weight', q), ('scales', s))
            mx.eval(*[a for _, a in parts])
            for name, a in parts:
                tensors[f'layers.{layer}.mlp.switch_mlp.{p}.{name}'] = a
                host = np.array(mx.view(a, mx.uint16) if a.dtype == mx.bfloat16 else a)
                layer_arrays.append((p, name, np.ascontiguousarray(host)))
        groups = {p: _CPU_GS if mode == 'affine' else 32 for p, _ in projs}
        specs[layer] = ce.layer_spec(layer_arrays, groups, mode, 4)
    return tensors, specs


def _build_cpu_model(tensors, num_layers=1, num_experts=_CPU_E, hidden=_CPU_HIDDEN, inner=_CPU_INNER):
    m = nn.Module()
    m.layers = []
    for layer in range(num_layers):
        sw = SwitchGLU(input_dims=hidden, hidden_dims=inner, num_experts=num_experts, bias=False)
        nn.quantize(sw, group_size=_CPU_GS, bits=4)
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


@pytest.mark.skipif(ce.isa() is None, reason='no C++17 compiler here')
@pytest.mark.parametrize('mode', ['affine', 'mxfp4'])
@pytest.mark.parametrize('b_rows', [3, 8])
def test_cpu_experts_server_batch_mailbox_matches_reference(mode, b_rows, monkeypatch):
    if mode == 'mxfp4':
        pytest.importorskip('mlx_vlm.models.deepseek_v41.fakequant')

    monkeypatch.setattr(ce, '_STATE', {'checked': ce.CHECKS, 'failed': None})
    rng = np.random.default_rng(70 + b_rows + (0 if mode == 'affine' else 100))
    _, specs = _make_cpu_arrays_and_tensors(rng, mode, num_layers=1)
    spec = specs[0]
    x_batch = rng.normal(0, 1, (b_rows, 256)).astype(np.float32)

    # Mix slots across batch rows `xr` up to `SERVE_MMAX=16`
    n_req = min(b_rows * 2, ce.SERVE_MMAX)
    slots = [int(rng.integers(0, 16)) for _ in range(n_req)]
    xrows = [i // 2 for i in range(n_req)]
    weights = rng.uniform(0.1, 0.9, size=n_req).astype(np.float32)
    act_style = 'deepseek_v41' if mode == 'mxfp4' else 'plain'
    lim = 10.0 if mode == 'mxfp4' else 0.0

    pool = ce.Pool(2, priority='normal')
    server = ce.Server(pool, [0], 256, pinned=False)
    try:
        server.add(0, spec, lim, act_style)
        server.start()
        b = server.box[0]
        b[ce.SERVE_HEAD:ce.SERVE_HEAD + b_rows * 256].view(np.float32)[:] = x_batch.reshape(-1)
        b[1] = n_req
        b[2 + 3 * ce.SERVE_MMAX] = b_rows
        for m in range(n_req):
            b[2 + m] = slots[m]
            b[2 + ce.SERVE_MMAX + m:2 + ce.SERVE_MMAX + m + 1].view(np.float32)[0] = weights[m]
            b[2 + 2 * ce.SERVE_MMAX + m] = xrows[m]
        b[0] = 1

        a = server.ans[0]
        import time
        deadline = time.monotonic() + 5.0
        while int(a[0]) != 1 and time.monotonic() < deadline:
            time.sleep(0.0005)
        assert int(a[0]) == 1
        assert a[2:2 + n_req].tolist() == [1] * n_req
        out = a[ce.SERVE_HEAD:ce.SERVE_HEAD + n_req * 256].view(np.float32).reshape(n_req, 256)
        for m in range(n_req):
            ref = ce.native_reference(
                ce.layer_expert(spec, slots[m]),
                x_batch[xrows[m]],
                lim,
                float(weights[m]),
                act_style,
                q8=pool.q8 and pool.isa >= 1,
            )
            np.testing.assert_allclose(out[m], ref, rtol=1e-2, atol=1e-2)
    finally:
        server.close()
        pool.close()


@pytest.mark.skipif(ce.isa() is None, reason='no C++17 compiler here')
@pytest.mark.parametrize('b_rows', [3, 8])
@pytest.mark.parametrize('use_server', [False, True])
def test_demand_batch_decode_cpu_offload_and_overflow(tmp_path, b_rows, use_server, monkeypatch):
    monkeypatch.setattr(ce, '_STATE', {'checked': ce.CHECKS, 'failed': None})
    monkeypatch.setattr(ce, '_SAID', [])
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'on')
    monkeypatch.setenv('QUANTEEZE_CPU_THREADS', '2')
    monkeypatch.setenv('QUANTEEZE_CPU_PRIORITY', 'normal')

    rng = np.random.default_rng(202 + b_rows + (10 if use_server else 0))
    tensors, _ = _make_cpu_arrays_and_tensors(rng, 'affine', num_layers=1)
    (tmp_path / 'config.json').write_text('{}', encoding='utf-8')
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)

    if use_server:
        # Install the B=1..8 mock CUDA mailbox kernels from test_demand_cpu_experts
        host_boxes = {}

        def fake_pinned(shape, dtype, *_a, **_kw):
            arr = np.zeros(shape, dtype)
            host_boxes[int(arr.__array_interface__['data'][0])] = arr
            return arr

        def fake_device(arr):
            return int(arr.__array_interface__['data'][0])

        def lookup(ptr):
            for base, arr in host_boxes.items():
                if base <= ptr < base + arr.nbytes:
                    return arr.reshape(-1)[(ptr - base) // 4:]
            raise KeyError(ptr)

        def make_ask():
            def call(inputs, template, grid, threadgroup, output_shapes, output_dtypes):
                rows_mx, inv_mx, weights_mx, x_mx, addr_mx = inputs
                tmap = dict(template)
                b_n, m_n, k_n, d_n = tmap['B'], tmap['M'], tmap['K'], tmap['D']
                head, mmax, weighs = tmap['HEAD'], tmap['MMAX'], tmap['WEIGHS']
                box = lookup(int(np.array(addr_mx)[0]))[:head + b_n * d_n]
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
                k_per = k_n // b_n if b_n > 1 else k_n
                for m in range(m_n):
                    box[2 + m] = int(rows_np[m])
                    wt = 1.0
                    xr = 0
                    for j in range(k_n):
                        if int(inv_np[j]) == m:
                            if weighs:
                                wt = float(w_np[j])
                            xr = j // k_per
                            break
                    box[2 + mmax + m:2 + mmax + m + 1].view(np.float32)[0] = wt
                    box[2 + 2 * mmax + m] = xr
                box[0] = stamp
                return [mx.array([stamp, 0, 0, 0], dtype=mx.int32)]
            return call

        def make_wait():
            import time

            def call(inputs, template, grid, threadgroup, output_shapes, output_dtypes):
                token_mx, y_mx, inv_mx, addr_mx = inputs
                tmap = dict(template)
                m_n, k_n, d_n, head, timeout_ns = tmap['M'], tmap['K'], tmap['D'], tmap['HEAD'], tmap['TIMEOUT_NS']
                ans = lookup(int(np.array(addr_mx)[1]))
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

        monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
        monkeypatch.setattr(demand.mx.metal, 'is_available', lambda: False)
        monkeypatch.setattr(cuda_ple, 'pinned', fake_pinned)
        monkeypatch.setattr(cuda_ple, '_device', fake_device)
        monkeypatch.setattr(ce, '_cpu_ask_kernel', make_ask)
        monkeypatch.setattr(ce, '_cpu_wait_kernel', make_wait)
    else:
        monkeypatch.setattr(cuda_kernels, 'available', lambda: False)

    m_ref = _build_cpu_model(tensors, num_layers=1)
    expert_qmv.install_switch(m_ref.layers[0].mlp.switch_mlp)
    m_dem = _build_cpu_model(tensors, num_layers=1)

    ckpt = q2_experts.Checkpoint(tmp_path)
    one_exp = (
        sum(ckpt.nbytes(ckpt.name(0, p, part)) for p in q2_experts.PROJECTIONS for part in q2_experts.NAMES)
        // _CPU_E
    )
    # Hold 12 of 16 experts so experts 12..15 remain SSD-only, and at B=8 also exercise SERVE_MMAX=16 overflow
    held = q2_experts.hold(ckpt, budget=one_exp * 12 + 128, workers=2, by_rows=True)
    dem = None
    try:
        held.wait()
        dem = demand.Demand(m_dem, ckpt, budget_bytes=1, workers=2)
        assert dem.slots == 8
        # Construct B rows mixing GPU-bank hits (0..3), HostCopy CPU misses (8..11), and SSD-only miss (14)
        rows_idx = []
        for r in range(b_rows):
            rows_idx.append([[0, 8 + (r % 4), 14 if r == 0 else 9, 10]])
        x = mx.array(rng.normal(0, 0.5, (b_rows, 1, _CPU_HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
        inds = mx.array(rows_idx, dtype=mx.uint32)
        mx.eval(x, inds)

        want = m_ref.layers[0].mlp.switch_mlp(x, inds)
        got = m_dem.layers[0].mlp.switch_mlp(x, inds)
        mx.eval(want, got)
        assert got.shape == want.shape == (b_rows, 1, _CPU_K, _CPU_HIDDEN)

        a = np.array(got.astype(mx.float32))
        b = np.array(want.astype(mx.float32))
        scale = max(float(np.abs(b).max()), 1e-6)
        assert float(np.abs(a - b).max()) <= 0.02 * scale

        summary = dem.summary()
        assert summary['cpu_served'] > 0
        assert summary['cpu_served'] <= ce.SERVE_MMAX
        if use_server:
            assert summary['cpu_requests'] == 1
            assert summary['cpu_failed'] == 0
    finally:
        if dem is not None:
            dem.close()
        q2_experts.let_go(ckpt.path)


def test_expert_qmv_and_qwen4_exp_batch_decode_compile():
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpSparseMoeBlock, TextConfig
    from quanteeze.arch import qwen4_exp

    mx.random.seed(77)
    sw = SwitchGLU(input_dims=128, hidden_dims=64, num_experts=8, bias=False)
    sw.apply(lambda a: a.astype(mx.bfloat16))
    nn.quantize(sw, group_size=64, bits=4)
    mx.eval(sw.parameters())

    for b_rows in (3, 8):
        x = mx.random.normal((b_rows, 1, 128)).astype(mx.bfloat16)
        indices = mx.array([[[(r + k) % 8 for k in range(4)]] for r in range(b_rows)], dtype=mx.uint32)
        mx.eval(x, indices)
        assert expert_qmv.eligible(sw, x, indices) == mx.metal.is_available()
        if not mx.metal.is_available():                # expert_qmv's own kernels are Metal's; elsewhere the stock
            continue                                   # product serves, and only the compiled forward is checked
        want = expert_qmv.stock(sw, x, indices)
        got_own = expert_qmv.own(sw, x, indices)
        got_dec = expert_qmv.decode(sw, x, indices)
        mx.eval(want, got_own, got_dec)
        assert mx.array_equal(want, got_own).item()
        assert mx.array_equal(want, got_dec).item()

    cfg = TextConfig(
        model_type='qwen4_exp',
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    block = Qwen4ExpSparseMoeBlock(cfg)
    block.apply(lambda a: a.astype(mx.bfloat16))
    mx.eval(block.parameters())
    block._layer_idx = 0
    fwd = qwen4_exp._BatchInvariantForward()
    for b_rows in (1, 3, 8):
        x = mx.random.normal((b_rows, 1, 128)).astype(mx.bfloat16)
        mx.eval(x)
        # First pass initializes compiled graph for batch size `b_rows`, second executes compiled graph
        ref = super(qwen4_exp._BatchInvariantForward, fwd)._feed_forward(block, x)
        got0 = fwd._feed_forward(block, x)
        got1 = fwd._feed_forward(block, x)
        mx.eval(ref, got0, got1)
        assert got0.shape == ref.shape == (b_rows, 1, 128)
        assert mx.array_equal(got0, ref).item()
        assert mx.array_equal(got1, ref).item()
    from quanteeze import per_thread
    assert set(per_thread.slots(block)['_router_and_shared_by_b']) == {(1, 1), (3, 1), (8, 1)}


# ---------------------------------------------------------------- 4. Concurrency & safety regressions (Q-1..Q-5)


@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 compiler here')
def test_native_reads_concurrent_close_while_waiting_no_uaf(tmp_path):
    f = tmp_path / 'blob.bin'
    f.write_bytes(b'x' * 4096)

    # 1) Threads blocked in qz_rd_wait when Readers.close() fires must wake safely via intrusive live_refs
    for _ in range(10):
        rd = native_reads.Readers(workers=0)
        dsts = [np.zeros(256, dtype=np.uint8) for _ in range(4)]
        futs = [rd.read([(f, 0, 256, d.ctypes.data)]) for d in dsts]
        assert all(fut is not None for fut in futs)
        barrier = threading.Barrier(len(futs) + 1)
        errors = []

        def waiter(fut):
            barrier.wait()
            try:
                fut.result(timeout=5.0)
            except RuntimeError as exc:
                if 'cancelled' not in str(exc):
                    errors.append(exc)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=waiter, args=(fut,)) for fut in futs]
        for th in threads:
            th.start()
        barrier.wait()
        time.sleep(0.002)
        rd.close()
        for th in threads:
            th.join(timeout=5.0)
            assert not th.is_alive()
        assert not errors
        assert all(fut.cancelled() for fut in futs)

    # 2) Unawaited futures closed / GC'd AFTER Readers.close() returned must not UAF
    rd2 = native_reads.Readers(workers=2)
    dst2 = np.zeros(512, dtype=np.uint8)
    lingering = [rd2.read([(f, 0, 512, dst2.ctypes.data)]) for _ in range(8)]
    rd2.close()
    assert rd2.read([(f, 0, 512, dst2.ctypes.data)]) is None
    for fut in lingering:
        fut.close()
    lingering.clear()
    gc.collect()


@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 compiler here')
def test_native_reads_concurrent_slab_expansion_and_generation_status(tmp_path):
    f = tmp_path / 'slab.bin'
    payload = np.arange(8192, dtype=np.uint8)
    f.write_bytes(payload.tobytes())

    # Q-2: 8 threads concurrently expanding slabs 1 and 2 (9,600 simultaneous tickets) without false exhaustion
    q = native_reads.Readers(workers=0)
    try:
        n_threads = 8
        per_thread = 1200
        dst = np.zeros(64, dtype=np.uint8)
        barrier = threading.Barrier(n_threads)
        buckets = [[] for _ in range(n_threads)]
        errs = []

        def grab(tid):
            try:
                barrier.wait()
                local = []
                for i in range(per_thread):
                    fut = q.read([(f, i % 1024, 64, dst.ctypes.data)], priority=i % 2)
                    assert fut is not None, f'thread {tid} failed at allocation {i}'
                    local.append(fut)
                buckets[tid] = local
            except Exception as exc:
                errs.append(exc)

        threads = [threading.Thread(target=grab, args=(i,)) for i in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=10.0)
        assert not errs
        assert q.counts()['refused'] == 0
        assert sum(len(b) for b in buckets) == n_threads * per_thread
        for b in buckets:
            for fut in b:
                fut.cancel()
    finally:
        q.close()

    # Q-3a: 3,200 concurrent wakeups across 8 threads must complete rapidly (no 200ms StoreLoad lost-wake stall)
    rd = native_reads.Readers(workers=4)
    try:
        barrier = threading.Barrier(8)
        errs = []

        def roundtrip(tid):
            buf = np.zeros(128, dtype=np.uint8)
            try:
                barrier.wait()
                for i in range(400):
                    fut = rd.read([(f, (tid * 64 + i) % 4096, 128, buf.ctypes.data)], priority=i % 2)
                    assert fut is not None
                    fut.result(timeout=5.0)
            except Exception as exc:
                errs.append(exc)

        t0 = time.monotonic()
        threads = [threading.Thread(target=roundtrip, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=10.0)
        assert not errs
        assert (time.monotonic() - t0) < 5.0
    finally:
        rd.close()

    # Q-3b: qz_rd_status on an expired generation token must return CANCELLED even after slot reuse in QUEUED state
    q2 = native_reads.Readers(workers=0)
    try:
        dst = np.zeros(64, dtype=np.uint8)
        held = [q2.read([(f, 0, 64, dst.ctypes.data)]) for _ in range(4096 * 4)]
        assert all(x is not None for x in held)
        fut_old = held[0]
        old_tok = fut_old._tok
        assert fut_old.cancel() is True
        fut_new = q2.read([(f, 0, 64, dst.ctypes.data)])
        assert fut_new is not None
        assert (fut_new._tok & 0xFFFFFFFF) == (old_tok & 0xFFFFFFFF)
        assert (fut_new._tok >> 32) > (old_tok >> 32)
        assert int(q2._py.qz_rd_status(q2._raw_handle, old_tok)) == native_reads._CANCELLED
        assert not fut_new.done()
        fut_new.cancel()
        for fut in held[1:]:
            fut.cancel()
    finally:
        q2.close()


@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 compiler here')
def test_native_reads_timeout_and_taken_close_drains_inflight_io(tmp_path, monkeypatch):
    # Q-5a: 1) TimeoutError followed by fut.close() waits for any TAKEN I/O before releasing dst
    f = tmp_path / 'big.bin'
    data = np.tile(np.arange(256, dtype=np.uint8), 32768)  # 8 MiB
    f.write_bytes(data.tobytes())

    q0 = native_reads.Readers(workers=0)
    try:
        dst0 = np.zeros(1024, dtype=np.uint8)
        fut0 = q0.read([(f, 0, 1024, dst0.ctypes.data)])
        with pytest.raises(TimeoutError):
            fut0.result(timeout=0.0)
        assert not fut0.cancelled()
        fut0.close()
        assert fut0.cancelled()
    finally:
        q0.close()

    rd = native_reads.Readers(workers=1)
    try:
        dst = np.zeros(len(data), dtype=np.uint8)
        # Force a real pread likely to be TAKEN when fut.close() arrives, or already DONE
        segs = [(f, i * (len(data) // 8), len(data) // 8, dst[i * (len(data) // 8):].ctypes.data) for i in range(8)]
        fut = rd.read(segs, cache=False)
        fut.close()
        # Once fut.close() returns, no background ::pread may still write into `dst`
        dst.fill(0xAA)
        time.sleep(0.02)
        assert np.all(dst == 0xAA)
        counts = rd.counts()
        assert counts['completed'] + counts['cancelled'] == counts['submitted'] == 1
    finally:
        rd.close()

    # 2) Demand._native_prefill_targets failure on 2nd batch closes 1st batch future before Python fallback
    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')
    monkeypatch.setenv('QUANTEEZE_DEMAND_READ_NATIVE', 'on')
    monkeypatch.setattr(native_reads, '_STATE', {'checked': native_reads.CHECKS, 'failed': None})

    sub = tmp_path / 'ckpt'
    sub.mkdir()
    m, src, rng = _make_moe_checkpoint(sub, 'safetensors_stacked', seed=61)
    dem = demand.Demand(m, src, budget_bytes=1, workers=2)
    try:
        orig_read = dem._native.read
        calls = [0]

        def fail_second_batch(*a, **kw):
            calls[0] += 1
            if calls[0] >= 2:
                return None
            return orig_read(*a, **kw)

        # Build > MAX_SEGMENTS (12) non-contiguous targets so _native_prefill_targets splits into 2 batches
        targets = []
        row_bytes = None
        for p in q2_experts.PROJECTIONS:
            for part in q2_experts.NAMES:
                t_name = src.name(0, p, part)
                _, st_dtype, full_shape, _ = src.tensors[t_name]
                np_dt = np.dtype(q2_experts._SAFETENSORS[st_dtype])
                arr = np.zeros((2,) + tuple(full_shape[1:]), dtype=np_dt)
                targets.append((t_name, arr))
                row_bytes = arr.nbytes
        rows = np.array([1, 5], dtype=np.int64)  # non-contiguous -> 2 segments per tensor = 18 > 12

        monkeypatch.setattr(dem._native, 'read', fail_second_batch)
        assert len(targets) == 9
        assert row_bytes > 0
        assert dem._native_prefill_targets(targets, rows) is False
        # Verify batch 1 was cleanly drained/cancelled (completed + cancelled == 1)
        counts = dem._native.counts()
        assert counts['completed'] + counts['cancelled'] == counts['submitted']
    finally:
        dem.close()


@pytest.mark.skipif(native_reads.libs() is None, reason='no C++17 compiler here')
def test_demand_summary_from_http_thread_and_hostcopy_locking(tmp_path, monkeypatch):
    from quanteeze import serve

    monkeypatch.setattr(ce, '_STATE', {'checked': 0, 'failed': None})
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')
    sub = tmp_path / 'ckpt'
    sub.mkdir()
    m, src, _ = _make_moe_checkpoint(sub, 'safetensors_stacked', seed=42)
    dem = demand.Demand(m, src, budget_bytes=1, workers=1)
    try:
        monkeypatch.setitem(serve._SERVING, 'demand', dem)
        monkeypatch.setitem(serve._SERVING, 'info', {'mode': 'on_demand'})
        dem._cpu_eval_tid = threading.get_ident()
        dem.cpu_served = 4

        stop_ev = threading.Event()
        http_errs = []
        snapshots = []
        http_tid = [None]
        orig_eval = demand.mx.eval

        def guarded_eval(*a, **kw):
            if http_tid[0] is not None and threading.get_ident() == http_tid[0]:
                raise AssertionError('mx.eval called from HTTP thread')
            return orig_eval(*a, **kw)

        monkeypatch.setattr(demand.mx, 'eval', guarded_eval)

        def http_poller():
            http_tid[0] = threading.get_ident()
            while not stop_ev.is_set():
                try:
                    info = serve.serving_info()
                    snapshots.append(info['demand'])
                    # Direct call from non-inference thread must also skip _flush_cpu_counts / mx.eval
                    dem.summary()
                except Exception as exc:
                    http_errs.append(exc)
                    return

        th = threading.Thread(target=http_poller)
        th.start()
        try:
            for i in range(60):
                st = mx.array([1, 1 if i % 10 == 0 else 0, 0, 5], dtype=mx.int32)
                with dem._cpu_status_lock:
                    dem._cpu_status.append(st)
                if i % 5 == 0:
                    dem._step_cpu()
                time.sleep(0.0005)
        finally:
            stop_ev.set()
            th.join(timeout=5.0)

        assert not http_errs
        assert len(snapshots) > 0
        final = dem.summary()
        assert final['cpu_late'] == 6
        assert final['cpu_wait_ms'] == pytest.approx(0.3, abs=1e-2)
    finally:
        dem.close()

    # Q-5b: Concurrent HostCopy.has / held_in / holds_expert / reading / read with close()
    layer_bytes = sum(
        src.nbytes(src.name(0, p, part)) for p in q2_experts.PROJECTIONS for part in q2_experts.NAMES
    )
    held = q2_experts.hold(src, budget=layer_bytes // 2, workers=2, by_rows=True)
    held.wait()
    hc_errs = []
    hc_stop = threading.Event()

    def reader():
        q_exps = np.array([0, 3, 9, 15], dtype=np.int64)
        t_name = src.name(0, 'gate_proj', 'weight')
        try:
            while not hc_stop.is_set():
                held.held_in(0, q_exps)
                held.holds_expert(0, t_name, 0)
                held.has(0, 0)
                held.reading()
                held.read()
        except Exception as exc:
            hc_errs.append(exc)

    r_threads = [threading.Thread(target=reader) for _ in range(3)]
    for th in r_threads:
        th.start()
    time.sleep(0.02)
    held.close()
    time.sleep(0.005)
    hc_stop.set()
    for th in r_threads:
        th.join(timeout=5.0)
    q2_experts.let_go(src.path)
    assert not hc_errs
    assert held.has(0, 0) is False
    assert held.held_in(0, [0]).size == 0
    assert held.holds_expert(0, src.name(0, 'gate_proj', 'weight'), 0) is False
