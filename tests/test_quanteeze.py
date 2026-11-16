# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for the Quanteeze package: module imports, variable table consistency, uniform
least-squares quantization (`requant`), single-group expert Metal kernels (`expert_qmv`),
and Browser UI routes (`control`, `home`)."""
import importlib
import pkgutil
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

import quanteeze
from quanteeze import control, expert_qmv, fitting, home, requant, variables

ROOT = Path(quanteeze.__file__).resolve().parents[2]
SRC = Path(quanteeze.__file__).resolve().parent


def test_all_modules_import():
    """Every module in `quanteeze` and `quanteeze.arch` imports without error."""
    imported = []
    for info in pkgutil.walk_packages([str(SRC)], prefix='quanteeze.'):
        mod = importlib.import_module(info.name)
        imported.append(mod.__name__)
    assert 'quanteeze.serve' in imported
    assert 'quanteeze.requant' in imported
    assert 'quanteeze.expert_qmv' in imported
    assert 'quanteeze.demand' in imported
    assert 'quanteeze.control' in imported
    assert 'quanteeze.home' in imported
    for mod_name in (
        'quanteeze.allocator',
        'quanteeze.cuda_attn',
        'quanteeze.cuda_head',
        'quanteeze.cuda_moe',
        'quanteeze.cuda_ple',
        'quanteeze.cuda_ple_layer',
        'quanteeze.cuda_v41',
        'quanteeze.embed_ssd',
        'quanteeze.ggml_quants',
        'quanteeze.gguf_experts',
        'quanteeze.gguf_qmv',
        'quanteeze.iq_experts',
        'quanteeze.iq_grids',
        'quanteeze.kquant_qmv',
        'quanteeze.reask',
        'quanteeze.unindexed',
        'quanteeze.vision_ssd',
        'quanteeze.arch.deepseek_v4',
        'quanteeze.arch.qwen3_next',
    ):
        assert mod_name in imported


def test_dsml_and_reask_validation():
    """`dsml` and `reask` parse valid DSML tool calls and flag malformed or duplicate ones."""
    from quanteeze import dsml, reask

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'read_file',
                'parameters': {
                    'type': 'object',
                    'properties': {'path': {'type': 'string'}},
                    'required': ['path'],
                    'additionalProperties': False,
                },
            },
        }
    ]
    body = {'tools': tools}
    opening, closing = dsml.V4.calls
    good = (
        f'{opening}\n'
        '<｜DSML｜invoke name="read_file">\n'
        '<｜DSML｜parameter name="path" string="true">README.md</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        f'{closing}'
    )
    assert reask.problems(good, opening, closing, dsml.parser, body) == []

    # Duplicate string="true" attribute on parameter tag is repaired by dsml.parse_tool_call
    dup_attr_inner = (
        '<｜DSML｜invoke name="read_file">\n'
        '<｜DSML｜parameter name="path" string="true" string="true">README.md</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>'
    )
    parsed = dsml.parse_tool_call(dup_attr_inner)
    assert parsed['name'] == 'read_file'
    assert parsed['arguments'] == {'path': 'README.md'}

    bad_tool = (
        f'{opening}\n'
        '<｜DSML｜invoke name="unknown_tool">\n'
        '</｜DSML｜invoke>\n'
        f'{closing}'
    )
    assert reask.problems(bad_tool, opening, closing, dsml.parser, body) != []


def test_variables_table_complete():
    """Every `QUANTEEZE_*` name referenced in `src/quanteeze` is defined in `variables.SETTINGS`."""
    pattern = re.compile(r'\bQUANTEEZE_[A-Z0-9_]*[A-Z0-9]\b')
    referenced = set()
    for path in SRC.rglob('*.py'):
        referenced |= set(pattern.findall(path.read_text(encoding='utf-8')))
    missing = referenced - set(variables.SETTINGS)
    assert not missing, f'Unregistered QUANTEEZE_* variables: {sorted(missing)}'


@pytest.mark.parametrize('bits', [1, 2, 3, 4, 8])
def test_requant_fit_pack_roundtrip(bits):
    """MLX least-squares `fit` + `pack` + `dequantize` matches GPU `requant.quantize`."""
    rng = np.random.default_rng(42 + bits)
    group_size = fitting.group_size(bits)
    w = rng.standard_normal((2, 16, 128)).astype(np.float32) * 0.1
    w_mx = mx.array(w, dtype=mx.bfloat16)
    q, s, b = requant.fit(w_mx, bits, group_size)
    packed = requant.pack(q, bits)
    unpacked = requant.unpack(packed, bits)
    np.testing.assert_array_equal(np.array(q), np.array(unpacked))

    wq_gpu, sq_gpu, bq_gpu = requant.quantize(w_mx, bits, group_size)
    mx.eval(wq_gpu, sq_gpu, bq_gpu)

    deq_cpu = requant.dequantize(packed, s, b, group_size, bits).astype(mx.float32)
    deq_gpu = requant.dequantize(wq_gpu, sq_gpu, bq_gpu, group_size, bits).astype(mx.float32)
    np.testing.assert_allclose(np.array(deq_gpu), np.array(deq_cpu), rtol=2e-2, atol=2e-2)


def test_expert_qmv_matches_stock_switch_glu():
    """`expert_qmv.decode` and `expert_qmv.install_switch` match stock `SwitchGLU` bit for bit."""
    mx.random.seed(7)
    sw = SwitchGLU(input_dims=128, hidden_dims=64, num_experts=8, bias=False)
    sw.gate_proj = sw.gate_proj.to_quantized(group_size=64, bits=4)
    sw.up_proj = sw.up_proj.to_quantized(group_size=64, bits=4)
    sw.down_proj = sw.down_proj.to_quantized(group_size=64, bits=4)
    assert isinstance(sw.gate_proj, QuantizedSwitchLinear)

    x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
    expected = sw(x, indices)
    mx.eval(expected)

    expert_qmv.install_switch(sw)
    actual = sw(x, indices)
    mx.eval(actual)
    assert mx.array_equal(expected, actual)


def test_control_and_home_pages_and_presets(tmp_path, monkeypatch):
    """`control` and `home` manage settings, presets, and HTML pages cleanly."""
    monkeypatch.setenv('QUANTEEZE_SETTINGS', str(tmp_path / 'settings.json'))
    st = control.state()
    assert st['groups'] == list(variables.GROUPS)
    assert any(s['name'] == 'QUANTEEZE_BITS' for s in st['settings'])

    res = control.change({'QUANTEEZE_BITS': '2'})
    assert 'QUANTEEZE_BITS' in res['applied']
    assert not res['rejected']
    assert control._uniform_bits() == 2

    res_ggml = control.change({'QUANTEEZE_BITS': 'q2_k'})
    assert 'QUANTEEZE_BITS' in res_ggml['applied']
    assert not res_ggml['rejected']
    assert control._uniform_bits() == 'q2_k'

    bad = control.change({'QUANTEEZE_BITS': '5'})
    assert 'QUANTEEZE_BITS' in bad['rejected']

    pre = home.choose_preset('memory', 'roomy')
    assert pre['memory']['now'] == 'roomy'
    assert variables.get('QUANTEEZE_MEMORY_RESERVE') == '0.45'

    assert control.PAGE.is_file()
    assert control.HOME.is_file()
    assert 'quanteeze settings' in control.PAGE.read_text(encoding='utf-8')
    assert 'quanteeze' in control.HOME.read_text(encoding='utf-8')


def test_phases_and_live_status():
    """`phases` tracks reading, thinking, answering, tok/s, TTFT, and request counts; `home.live` returns them."""
    import time
    from quanteeze import phases

    entry = phases.begin('req-test-1', prompt=120, thinking=True, effort='medium')
    phases.read(entry, 60)
    st_reading = phases.state()
    assert any(r['id'] == 'req-test-1' and r['phase'] == 'reading' and r['read'] == 60 for r in st_reading['now'])

    phases.prompt_read(entry, cached=40)
    phases.wrote(entry, 'reasoning ', 1, None)
    phases.wrote(entry, '</think>', 1, None)
    time.sleep(0.02)
    phases.wrote(entry, 'answer', 3, 'stop')

    st_done = phases.state()
    assert not any(r['id'] == 'req-test-1' for r in st_done['now'])
    ended = next(r for r in st_done['ended'] if r['id'] == 'req-test-1')
    assert ended['how'] == 'answered'
    assert ended['cached'] == 40
    assert ended['thought'] == 2
    assert ended['written'] == 3
    assert ended['ttft_s'] is not None
    assert ended['decode_tps'] is not None and ended['decode_tps'] > 0
    assert st_done['counts']['requests'] >= 1

    live_info = home.live()
    assert 'memory' in live_info and 'keeper' in live_info['memory']
    assert 'allocator' in live_info['memory']
    assert 'requests' in live_info and 'counts' in live_info['requests']


def test_thermal_notify_token_cached_and_closed():
    """`guards.Readings` registers macOS thermal notify token once per lib and releases on `close`."""
    import ctypes
    from quanteeze import guards

    calls = {'register': 0, 'get_state': 0, 'cancel': []}

    class FakeLib:
        def notify_register_check(self, key, token_ptr):
            assert key == guards.THERMAL_KEY
            calls['register'] += 1
            ctypes.cast(token_ptr, ctypes.POINTER(ctypes.c_int)).contents.value = 77
            return 0

        def notify_get_state(self, token, state_ptr):
            calls['get_state'] += 1
            assert int(token.value if hasattr(token, 'value') else token) == 77
            ctypes.cast(state_ptr, ctypes.POINTER(ctypes.c_uint64)).contents.value = 2
            return 0

        def notify_cancel(self, token):
            calls['cancel'].append(int(token.value if hasattr(token, 'value') else token))
            return 0

    lib = FakeLib()
    r1 = guards.Readings(lib=lib)
    r2 = guards.Readings(lib=lib)
    try:
        assert r1.thermal() == 2
        assert r1.thermal() == 2
        assert r2.thermal() == 2
        assert calls['register'] == 1
        assert calls['get_state'] == 3
    finally:
        r1.close()
    assert calls['cancel'] == [77]
    assert id(lib) not in guards._THERMAL_TOKENS


def test_default_port_8765(monkeypatch):
    """Default port across `control` and `home.connect` is 8765."""
    import sys

    assert control.DEFAULT_PORT == '8765'
    assert control.argument(['--host', '127.0.0.1'], '--port', control.DEFAULT_PORT) == '8765'
    monkeypatch.setattr(sys, 'argv', ['quanteeze'])
    conn = home.connect()
    assert conn['anthropic'] == 'http://127.0.0.1:8765'
    assert conn['openai'] == 'http://127.0.0.1:8765/v1'


def test_demand_batched_prefill_matches_unbatched(tmp_path):
    """Bounded-memory `Demand` prefill batching (`prompt_room`) matches single-batch bit for bit."""
    import mlx.nn as nn
    from quanteeze import demand, q2_experts

    mx.random.seed(11)
    sw_base = SwitchGLU(input_dims=64, hidden_dims=64, num_experts=16, bias=False)
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        setattr(sw_base, proj, getattr(sw_base, proj).to_quantized(group_size=64, bits=4))
    mx.eval(sw_base.parameters())

    tensors = {}
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        p = getattr(sw_base, proj)
        tensors[f'layers.0.mlp.switch_mlp.{proj}.weight'] = p.weight
        tensors[f'layers.0.mlp.switch_mlp.{proj}.scales'] = p.scales
        tensors[f'layers.0.mlp.switch_mlp.{proj}.biases'] = p.biases
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)

    def make_model():
        sw = SwitchGLU(input_dims=64, hidden_dims=64, num_experts=16, bias=False)
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(sw, proj, getattr(sw, proj).to_quantized(group_size=64, bits=4))
        mlp = nn.Module()
        mlp.switch_mlp = sw
        layer = nn.Module()
        layer.mlp = mlp
        m = nn.Module()
        m.layers = [layer]
        return m

    m1, m2 = make_model(), make_model()
    c1, c2 = q2_experts.Checkpoint(tmp_path), q2_experts.Checkpoint(tmp_path)
    d1 = demand.Demand(m1, c1, budget_bytes=1, workers=2, prompt_room=None)
    d2 = demand.Demand(m2, c2, budget_bytes=1, workers=2, prompt_room=1)
    try:
        assert d1.slots == 8 and d2.slots == 8
        x = mx.random.normal((1, 16, 64)).astype(mx.bfloat16)
        idx_np = (np.arange(16 * 4, dtype=np.uint32).reshape(1, 16, 4) * 3) % 16
        indices = mx.array(idx_np)
        scores = mx.ones((1, 16, 4), dtype=mx.bfloat16) * 0.25
        mx.eval(x, indices, scores)

        y1 = m1.layers[0].mlp.switch_mlp(x, indices, scores)
        y2 = m2.layers[0].mlp.switch_mlp(x, indices, scores)
        mx.eval(y1, y2)

        assert d1.prefill_layers == 1 and d2.prefill_layers == 1
        assert mx.array_equal(y1, y2)
        np.testing.assert_array_equal(d1.switches[0].experts, d2.switches[0].experts)
    finally:
        d1.close()
        d2.close()


def test_bf16_head_keep_matches_full():
    """`BF16Head(x, keep=1)` evaluates vocabulary chunks and slices without altering logits."""
    import mlx.nn as nn
    from quanteeze.arch import deepseek_v41

    mx.random.seed(19)
    vocab = deepseek_v41.HEAD_CHUNK + 256
    dim = 256
    head = deepseek_v41.BF16Head.__new__(deepseek_v41.BF16Head)
    nn.Module.__init__(head)
    head.weight = mx.random.normal((vocab, dim)).astype(mx.bfloat16)
    x = mx.random.normal((1, deepseek_v41.HEAD_ROWS + 4, dim)).astype(mx.bfloat16)
    mx.eval(head.weight, x)

    full = head(x)
    kept = head(x, keep=1)
    mx.eval(full, kept)
    assert kept.shape == (1, 1, vocab)
    assert mx.array_equal(full[..., -1:, :], kept)


@pytest.mark.parametrize('mode', ['q2_k', 'q3_k', 'iq2_xs'])
def test_ggml_quants_matches_iq_experts_decode(mode):
    """`ggml_quants` pack + `dequantize` matches `iq_experts.decode` bit for bit in float32."""
    from quanteeze import ggml_quants, iq_experts

    kinds = {'q2_k': iq_experts.Q2_K, 'q3_k': iq_experts.Q3_K, 'iq2_xs': iq_experts.IQ2_XS}
    rng = np.random.default_rng(101)
    w = mx.array((rng.standard_normal((2, 8, 512)) * 0.25).astype(np.float32), dtype=mx.bfloat16)
    body, d, dmin, values = ggml_quants.FORMATS[mode](w)
    mx.eval(body, d, values, *([dmin] if dmin is not None else []))

    raw = ggml_quants.blocks(body, d, dmin, mode=mode)
    decoded_iq = iq_experts.decode(raw, kinds[mode]).reshape(w.shape)
    decoded_qz = ggml_quants.dequantize(body, d, dmin, mode)
    mx.eval(decoded_iq, decoded_qz)

    np.testing.assert_array_equal(np.array(decoded_qz), np.array(values))
    np.testing.assert_array_equal(np.array(decoded_qz), np.array(decoded_iq))


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal-only K-quant kernels')
@pytest.mark.parametrize('mode', ['q2_k', 'q3_k'])
def test_kquant_qmv_metal_decode_and_prefill(mode):
    """Metal `kquant_qmv` decode (`gather`) and tiled prefill (`gather_rows`) match unpacked matmul."""
    import mlx.nn as nn
    from quanteeze import ggml_quants, kquant_qmv

    rng = np.random.default_rng(202)
    w = mx.array((rng.standard_normal((8, 64, 512)) * 0.2).astype(np.float32), dtype=mx.bfloat16)
    body, d, dmin, _ = ggml_quants.FORMATS[mode](w)
    proj = nn.Module()
    proj.weight = body
    proj.scales = d
    if dmin is not None:
        proj.biases = dmin
    proj.mode = mode
    proj.group_size = 256
    proj.bits = ggml_quants.FORMAT_BITS[mode]
    mx.eval(proj.parameters())

    def ref_fn(x_in, idx_in):
        w_unp = ggml_quants.dequantize(proj.weight, proj.scales, dmin, mode).astype(x_in.dtype)
        return mx.gather_mm(x_in, w_unp.swapaxes(-1, -2), rhs_indices=idx_in)

    x_dec = mx.array(rng.standard_normal((1, 1, 1, 512)).astype(np.float32), dtype=mx.bfloat16)
    idx_dec = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
    mx.eval(x_dec, idx_dec)
    want_dec = ref_fn(x_dec, idx_dec)
    mx.eval(want_dec)
    assert kquant_qmv.takes(proj, x_dec, idx_dec)
    for _ in range(kquant_qmv.CHECKS + 2):
        got_dec = kquant_qmv.gather(proj, x_dec, idx_dec, lambda: ref_fn(x_dec, idx_dec))
        mx.eval(got_dec)
        assert float(mx.abs(got_dec.astype(mx.float32) - want_dec.astype(mx.float32)).max().item()) <= (
            kquant_qmv.TOLERANCE * max(float(mx.abs(want_dec.astype(mx.float32)).max().item()), 1e-30)
        )

    x_pre = mx.array(rng.standard_normal((20, 1, 512)).astype(np.float32), dtype=mx.bfloat16)
    idx_pre = mx.array((np.arange(20, dtype=np.uint32) * 3) % 8, dtype=mx.uint32)
    mx.eval(x_pre, idx_pre)
    want_pre = ref_fn(x_pre, idx_pre)
    mx.eval(want_pre)
    assert kquant_qmv.takes_rows(proj, x_pre, idx_pre)
    for _ in range(kquant_qmv.CHECKS + 2):
        got_pre = kquant_qmv.gather_rows(proj, x_pre, idx_pre, lambda: ref_fn(x_pre, idx_pre))
        mx.eval(got_pre)
        assert float(mx.abs(got_pre.astype(mx.float32) - want_pre.astype(mx.float32)).max().item()) <= (
            kquant_qmv.TOLERANCE * max(float(mx.abs(want_pre.astype(mx.float32)).max().item()), 1e-30)
        )


@pytest.mark.parametrize('mode', ['q2_k', 'q3_k', 'iq2_xs'])
def test_uniform_ggml_lower_model_and_demand(tmp_path, monkeypatch, mode):
    """Uniform `requant.lower_model`, `QuantCache`, and `Demand` work end-to-end for ggml modes."""
    import mlx.nn as nn
    from quanteeze import demand

    monkeypatch.setenv('QUANTEEZE_QUANT_CACHE', str(tmp_path / 'qcache'))
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')
    mx.random.seed(303)
    sw_stock = SwitchGLU(input_dims=256, hidden_dims=256, num_experts=8, bias=False)
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        setattr(sw_stock, proj, getattr(sw_stock, proj).to_quantized(group_size=64, bits=4))
    mx.eval(sw_stock.parameters())

    tensors = {}
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        p = getattr(sw_stock, proj)
        tensors[f'layers.0.mlp.switch_mlp.{proj}.weight'] = p.weight
        tensors[f'layers.0.mlp.switch_mlp.{proj}.scales'] = p.scales
        tensors[f'layers.0.mlp.switch_mlp.{proj}.biases'] = p.biases
    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir()
    (ckpt_dir / 'config.json').write_text(
        '{"quantization": {"group_size": 64, "bits": 4}}',
        encoding='utf-8',
    )
    mx.save_safetensors(str(ckpt_dir / 'model.safetensors'), tensors)

    def build_model():
        sw = SwitchGLU(input_dims=256, hidden_dims=256, num_experts=8, bias=False)
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(sw, proj, getattr(sw, proj).to_quantized(group_size=64, bits=4))
        mlp = nn.Module()
        mlp.switch_mlp = sw
        layer = nn.Module()
        layer.mlp = mlp
        m = nn.Module()
        m.layers = [layer]
        return m

    m_mem = build_model()
    requant.lower_model(m_mem.layers, ckpt_dir, mode, 256)
    expert_qmv.install_switch(m_mem.layers[0].mlp.switch_mlp)

    m_ssd = build_model()
    flat = requant.Flat(ckpt_dir, mode, 256)
    dem = demand.Demand(m_ssd, flat, budget_bytes=1, workers=2)
    try:
        for x_shape, idx_arr in [
            ((1, 1, 256), mx.array([[[0, 2, 5, 7]]], dtype=mx.uint32)),
            ((1, 6, 256), mx.array((np.arange(24, dtype=np.uint32).reshape(1, 6, 4)) % 8)),
        ]:
            x = mx.random.normal(x_shape).astype(mx.bfloat16)
            scores = mx.ones(idx_arr.shape, dtype=mx.bfloat16) * 0.25
            mx.eval(x, idx_arr, scores)
            y_mem = m_mem.layers[0].mlp.switch_mlp(x, idx_arr)
            y_ssd = m_ssd.layers[0].mlp.switch_mlp(x, idx_arr, scores)
            mx.eval(y_mem, y_ssd)
            np.testing.assert_allclose(
                np.array(y_mem.astype(mx.float32)),
                np.array(y_ssd.astype(mx.float32)),
                rtol=2e-2,
                atol=2e-2,
            )
    finally:
        dem.close()


def test_vision_ssd_embed_ssd_and_compact(tmp_path, monkeypatch):
    """`vision_ssd` and `embed_ssd` stream from safetensors bit-for-bit and release resident weights."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from quanteeze import embed_ssd, vision_ssd

    monkeypatch.setenv('QUANTEEZE_COMPACT', 'on')
    monkeypatch.setenv('QUANTEEZE_COMPACT_KEEP', 'MTP')
    assert variables.compact() is True
    assert variables.get('QUANTEEZE_VISION_SSD') == 'on'
    assert variables.get('QUANTEEZE_EMBED_SSD') == 'on'
    assert variables.get('QUANTEEZE_MTP') == 'auto'

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

    mx.random.seed(404)
    model = nn.Module()
    model.vision_tower = VisionTower(32)
    for k, a in tree_flatten(model.vision_tower.parameters()):
        vision_ssd.set_param(model.vision_tower, k, a.astype(mx.bfloat16))

    lm = nn.Module()
    lm.args = type('Args', (), {'tie_word_embeddings': False})()
    inner = nn.Module()
    emb = nn.Embedding(64, 32).to_quantized(group_size=32, bits=4)
    inner.embed_tokens = emb
    lm.model = inner
    model.language_model = lm
    mx.eval(model.parameters())

    x_img = mx.random.normal((2, 8, 32)).astype(mx.bfloat16)
    tok_ids = mx.array([[0, 5, 19, 63, 5]], dtype=mx.int32)
    want_img = model.vision_tower(x_img)
    want_emb = model.language_model.model.embed_tokens(tok_ids)
    mx.eval(want_img, want_emb)

    ckpt_tensors = {}
    for k, a in tree_flatten(model.vision_tower.parameters()):
        ckpt_tensors[f'vision_tower.{k}'] = a
    for k in ('weight', 'scales', 'biases'):
        ckpt_tensors[f'language_model.model.embed_tokens.{k}'] = inner.embed_tokens[k]
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), ckpt_tensors)

    vision_ssd.IMAGES.clear()
    reused_before = vision_ssd.IMAGE_STATS['reused']
    freed_vis = vision_ssd.install(model, tmp_path)
    freed_emb = embed_ssd.install(model, tmp_path)
    assert freed_vis > 0
    assert freed_emb > 0
    assert sum(a.nbytes for _, a in tree_flatten(model.vision_tower.parameters())) == 0
    assert isinstance(model.language_model.model.embed_tokens, embed_ssd.SSDTable)

    got_img = model.vision_tower(x_img)
    got_img2 = model.vision_tower(x_img)
    got_emb = model.language_model.model.embed_tokens(tok_ids)
    mx.eval(got_img, got_img2, got_emb)

    assert mx.array_equal(want_img, got_img)
    assert mx.array_equal(want_img, got_img2)
    assert vision_ssd.IMAGE_STATS['reused'] == reused_before + 1
    assert sum(a.nbytes for _, a in tree_flatten(model.vision_tower.parameters())) == 0
    assert mx.array_equal(want_emb, got_emb)
    vision_ssd.IMAGES.clear()


def test_ngram_prefetch_advise_ahead_and_repeat_cache(tmp_path):
    """`ngram_prefetch` advises small lookups via `madvise`, runs `ahead()`, and reuses identical lookups."""
    from quanteeze import ngram_prefetch

    raw_path = tmp_path / 'ngram.bin'
    arr_init = np.arange(128 * 32, dtype=np.uint16).reshape(128, 32)
    arr_init.tofile(raw_path)
    mm = np.memmap(raw_path, mode='r', dtype=np.uint16, shape=(128, 32))

    rows = np.array([1, 7, 33, 99], dtype=np.int64)
    assert ngram_prefetch._advise([(mm, rows)]) is True
    pf = ngram_prefetch.Prefetcher(threads=2, min_rows=64, advise=True)
    assert pf.fetch([(mm, rows)]) == 4
    pf.ahead([(mm, rows)])

    calls = []

    class DummyTable:
        def __init__(self, table):
            self._interleaved = table

        def _read_rows(self, flat_ids):
            calls.append(list(flat_ids))
            return self._interleaved[np.asarray(flat_ids, dtype=np.int64)]

    ngram_prefetch.install(DummyTable, prefetcher=pf)
    tbl = DummyTable(mm)
    again_before = ngram_prefetch.STATS['again']
    r1 = tbl._read_rows(np.array([3, 12, 45], dtype=np.int64))
    r2 = tbl._read_rows(np.array([3, 12, 45], dtype=np.int64))
    np.testing.assert_array_equal(r1, r2)
    assert len(calls) == 1
    assert ngram_prefetch.STATS['again'] == again_before + 1


def test_arch_registrations_reserve_and_prefill_window_join(monkeypatch):
    """`deepseek_v4` and `qwen3_next` are registered, CUDA V4.1 reserve relaxes to 0.13, and prefill joins windows."""
    import mlx.nn as nn
    from quanteeze import arch, prefill

    assert arch.get('deepseek_v4') is not None
    assert arch.get('qwen3_next') is not None

    with monkeypatch.context() as mp:
        mp.delenv('QUANTEEZE_MEMORY_RESERVE', raising=False)
        fitting.MODEL[0] = 'deepseek_v41'
        try:
            mp.setattr(mx.metal, 'is_available', lambda: False)
            mp.setattr(mx, 'default_device', lambda: mx.gpu)
            assert fitting.reserve() == pytest.approx(0.13)
            mp.setenv('QUANTEEZE_MEMORY_RESERVE', '0.25')
            assert fitting.reserve() == pytest.approx(0.25)
        finally:
            fitting.MODEL[0] = None

    class DummyCache:
        def __init__(self):
            self.offset = 0
            self.seen = []

        def step(self, x):
            self.offset += x.shape[1]
            self.seen.append(self.offset)

    class SplitLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = nn.Linear(dim, dim, bias=False)
            mlp = nn.Module()
            mlp.top_k = 8
            mlp.switch_mlp = SwitchGLU(input_dims=dim, hidden_dims=dim, num_experts=4, bias=False)
            mlp.shared = nn.Linear(dim, dim, bias=False)
            self.mlp = mlp

        def _pre(self, x, mask=None, cache=None):
            if cache is not None:
                cache.step(x)
            return x + self.attn(x)

        def _post(self, h):
            idx = mx.zeros((h.shape[0], h.shape[1], 2), dtype=mx.uint32)
            routed = self.mlp.switch_mlp(h, idx).mean(axis=-2)
            return h + routed + self.mlp.shared(h)

        def __call__(self, x, mask=None, cache=None):
            return self._post(self._pre(x, mask=mask, cache=cache))

    prefill.split(SplitLayer, SplitLayer._pre, SplitLayer._post)
    mx.random.seed(505)
    l_ref = SplitLayer(16)
    l_join = SplitLayer(16)
    l_join.update(l_ref.parameters())
    mx.eval(l_ref.parameters(), l_join.parameters())

    monkeypatch.setenv('QUANTEEZE_PROMPT_MOE_TOKENS', '16')
    prefill.times(reset=True)
    prefill.JOINS.clear()
    prefill.install([l_join], chunk=8, enabled=lambda: True)
    try:
        for _ in range(prefill.JOIN_CHECKS + 2):
            x = mx.random.normal((1, 24, 16)).astype(mx.bfloat16)
            mx.eval(x)
            c_ref, c_join = DummyCache(), DummyCache()
            y_ref = l_ref(x, cache=c_ref)
            y_join = l_join(x, cache=c_join)
            mx.eval(y_ref, y_join)
            assert mx.array_equal(y_ref, y_join)
            assert c_join.seen == [8, 16, 24]
        assert prefill.times()['joined'] >= 1
    finally:
        prefill.release()


def test_apc_disk_convergent_encryption_exact_and_hybrid(tmp_path, monkeypatch):
    """`apc_disk` seals exact & hybrid APC checkpoints with convergent AES-256-GCM, isolating precision & rejecting tampering."""
    from mlx_vlm import apc as _apc
    from mlx_vlm.models import cache as lm_cache
    from quanteeze import apc_disk, control

    assert 'QUANTEEZE_APC_ENCRYPT' in variables.SETTINGS
    assert variables.SETTINGS['QUANTEEZE_APC_ENCRYPT'].default == 'on'
    assert control.CHOICES['QUANTEEZE_APC_ENCRYPT'] == ('on', 'off')

    apc_disk.install()
    monkeypatch.setattr(apc_disk, 'ON', [True])
    monkeypatch.setattr(apc_disk, 'PRECISION', [None])
    monkeypatch.setenv('QUANTEEZE_BITS', '4')
    monkeypatch.setenv('APC_DISK_MIN_FREE_RAM_GB', '0')

    # 1. Hybrid checkpoint (KVCache + ArraysCache) -- non-trimmable (`exact_v1`)
    hybrid_dir = tmp_path / 'hybrid'
    store1 = _apc.DiskBlockStore(hybrid_dir, namespace='ns_hybrid', num_workers=1)
    mgr1 = _apc.APCManager(
        num_blocks=4,
        block_size=8,
        disk=store1,
        overrides={'memory_max_gb': 0, 'checkpoint_entries': 0},
    )
    mgr1.exact_cache_min_tokens = 4

    secret_tokens = tuple(9100 + i * 7 for i in range(18))
    kv = lm_cache.KVCache()
    kv.keys = mx.arange(1 * 2 * 18 * 4, dtype=mx.float16).reshape(1, 2, 18, 4)
    kv.values = (mx.arange(1 * 2 * 18 * 4, dtype=mx.float16).reshape(1, 2, 18, 4) + 3)
    kv.offset = 18
    ssm = lm_cache.ArraysCache(size=2)
    ssm.cache = [mx.ones((1, 4, 8), dtype=mx.bfloat16) * 2.5, None]
    mx.eval(kv.keys, kv.values, ssm.cache[0])

    assert mgr1.store_exact_cache(secret_tokens, [kv, ssm], extra_hash=42) is True
    store1.flush()
    store1.close()

    files = list((hybrid_dir / _apc._safe_namespace('ns_hybrid')).glob('exact_*.safetensors'))
    assert len(files) == 1
    sealed_path = files[0]
    assert (sealed_path.stat().st_mode & 0o777) == 0o600
    raw = sealed_path.read_bytes()
    assert raw.startswith(apc_disk.MAGIC)
    csv_tokens = ','.join(str(t) for t in secret_tokens).encode('ascii')
    bin_tokens = np.asarray(secret_tokens, dtype='<i4').tobytes()
    assert csv_tokens not in raw
    assert bin_tokens not in raw
    assert b'__metadata__' not in raw
    assert b'exact_cache_v1' not in raw

    # Cold restart: wipe all ephemeral keys from RAM, open fresh DiskBlockStore
    apc_disk.clear_runtime_keys()
    store2 = _apc.DiskBlockStore(hybrid_dir, namespace='ns_hybrid', num_workers=1)
    mgr2 = _apc.APCManager(
        num_blocks=4,
        block_size=8,
        disk=store2,
        overrides={'memory_max_gb': 0, 'checkpoint_entries': 0},
    )
    mgr2.exact_cache_min_tokens = 4

    try:
        # Wrong prompt cannot decrypt
        wrong_query = tuple(5000 + i for i in range(22))
        assert mgr2.lookup_exact_cache(wrong_query, extra_hash=42) == (None, 0)

        # Different QUANTEEZE_BITS must miss cleanly (never misread stale precision cache)
        monkeypatch.setenv('QUANTEEZE_BITS', 'q2_k')
        query_tokens = secret_tokens + (9991, 9992, 9993)
        assert mgr2.lookup_exact_cache(query_tokens, extra_hash=42) == (None, 0)

        # Matching QUANTEEZE_BITS + matching prompt decrypts bit-for-bit
        monkeypatch.setenv('QUANTEEZE_BITS', '4')
        restored, matched_len = mgr2.lookup_exact_cache(query_tokens, extra_hash=42)
        assert matched_len == 18
        assert restored is not None and len(restored) == 2
        assert mx.array_equal(restored[0].keys[..., :18, :], kv.keys)
        assert mx.array_equal(restored[0].values[..., :18, :], kv.values)
        assert mx.array_equal(restored[1].cache[0], ssm.cache[0])
        assert restored[1].cache[1] is None
        assert mgr2.stats.disk_hits == 1

        # Tamper 1 byte in ciphertext -> rejected cleanly by AES-GCM without raising
        tampered = bytearray(sealed_path.read_bytes())
        tampered[-10] ^= 0x5A
        sealed_path.write_bytes(bytes(tampered))
        rejected_before = apc_disk.STATS['rejected']
        assert mgr2.lookup_exact_cache(query_tokens, extra_hash=42) == (None, 0)
        assert apc_disk.STATS['rejected'] == rejected_before + 1
        assert not sealed_path.exists()
    finally:
        store2.close()

    # 2. Trimmable dense checkpoint (`exact_trim_v1`) supports block-aligned prefix restore without knowing tail tokens
    dense_dir = tmp_path / 'dense'
    store3 = _apc.DiskBlockStore(dense_dir, namespace='ns_dense', num_workers=1)
    mgr3 = _apc.APCManager(
        num_blocks=4,
        block_size=8,
        disk=store3,
        overrides={'memory_max_gb': 0, 'checkpoint_entries': 0},
    )
    mgr3.exact_cache_min_tokens = 4

    dense_tokens = tuple(3000 + i for i in range(20))  # 2 full blocks (16) + 4-token tail
    dkv = lm_cache.KVCache()
    dkv.keys = mx.arange(1 * 2 * 20 * 4, dtype=mx.float32).reshape(1, 2, 20, 4)
    dkv.values = mx.arange(1 * 2 * 20 * 4, dtype=mx.float32).reshape(1, 2, 20, 4) * 0.5
    dkv.offset = 20
    mx.eval(dkv.keys, dkv.values)
    assert mgr3.store_exact_cache(dense_tokens, [dkv], extra_hash=7) is True
    store3.flush()
    store3.close()

    apc_disk.clear_runtime_keys()
    store4 = _apc.DiskBlockStore(dense_dir, namespace='ns_dense', num_workers=1)
    mgr4 = _apc.APCManager(
        num_blocks=4,
        block_size=8,
        disk=store4,
        overrides={'memory_max_gb': 0, 'checkpoint_entries': 0},
    )
    mgr4.exact_cache_min_tokens = 4
    try:
        # Query shares only the first 16 tokens (2 blocks) and diverges afterwards
        diverged_query = dense_tokens[:16] + (8881, 8882, 8883, 8884, 8885)
        trimmed_cache, trimmed_len = mgr4.lookup_exact_cache(diverged_query, extra_hash=7)
        assert trimmed_len == 16
        assert trimmed_cache is not None
        assert mx.array_equal(trimmed_cache[0].keys[..., :16, :], dkv.keys[..., :16, :])
        assert mx.array_equal(trimmed_cache[0].values[..., :16, :], dkv.values[..., :16, :])

        # Full 20-token prefix match also restores tail
        full_query = dense_tokens + (7771, 7772)
        full_cache, full_len = mgr4.lookup_exact_cache(full_query, extra_hash=7)
        assert full_len == 20
        assert mx.array_equal(full_cache[0].keys[..., :20, :], dkv.keys)
        assert mx.array_equal(full_cache[0].values[..., :20, :], dkv.values)
    finally:
        store4.close()


def test_apc_disk_block_shards_and_legacy_purge(tmp_path, monkeypatch):
    """`apc_disk` seals per-block shards (`blocks_v1`) for partial prefix reuse and purges unsealed legacy files."""
    from mlx_vlm import apc as _apc
    from quanteeze import apc_disk

    apc_disk.install()
    monkeypatch.setattr(apc_disk, 'ON', [True])
    monkeypatch.setattr(apc_disk, 'PRECISION', [None])
    monkeypatch.setenv('QUANTEEZE_BITS', '3')
    monkeypatch.setenv('APC_DISK_MIN_FREE_RAM_GB', '0')

    shard_root = tmp_path / 'shards'
    ns_dir = shard_root / _apc._safe_namespace('ns_blocks')
    ns_dir.mkdir(parents=True, exist_ok=True)
    # Plant a legacy plaintext safetensors shard in the directory before DiskBlockStore starts
    legacy_path = ns_dir / ('shard_' + 'ab' * 16 + '.safetensors')
    mx.save_safetensors(
        str(legacy_path),
        {'k0': mx.zeros((1, 2, 8, 4), dtype=mx.float16)},
        metadata={'block_hashes': '123', 'b0_meta': '{"token_ids":[1,2,3]}'},
    )
    assert legacy_path.exists()

    purged_before = apc_disk.STATS['purged_plaintext']
    store1 = _apc.DiskBlockStore(shard_root, namespace='ns_blocks', num_workers=1)
    assert not legacy_path.exists()
    assert apc_disk.STATS['purged_plaintext'] == purged_before + 1

    mgr1 = _apc.APCManager(
        num_blocks=2,
        block_size=8,
        disk=store1,
        overrides={'memory_max_gb': 0, 'checkpoint_entries': 0},
    )
    tokens_32 = tuple(4100 + i for i in range(32))  # 4 blocks of 8 tokens
    k0 = mx.arange(1 * 2 * 32 * 4, dtype=mx.float16).reshape(1, 2, 32, 4)
    v0 = (mx.arange(1 * 2 * 32 * 4, dtype=mx.float16).reshape(1, 2, 32, 4) + 1)
    mx.eval(k0, v0)

    blocks = mgr1.store_kv_blocks(tokens_32, [k0], [v0], extra_hash=11)
    mgr1.release(blocks)
    store1.flush()
    store1.close()

    shard_files = list(ns_dir.glob('shard_*.safetensors'))
    assert len(shard_files) == 1
    raw = shard_files[0].read_bytes()
    assert raw.startswith(apc_disk.MAGIC)
    assert (shard_files[0].stat().st_mode & 0o777) == 0o600
    assert b'4100,4101' not in raw
    assert b'layer_major_v2' not in raw

    # Cold restart: clear RAM keys, query a prompt sharing only the first 2 blocks (16 tokens)
    apc_disk.clear_runtime_keys()
    store2 = _apc.DiskBlockStore(shard_root, namespace='ns_blocks', num_workers=1)
    mgr2 = _apc.APCManager(
        num_blocks=2,
        block_size=8,
        disk=store2,
        overrides={'memory_max_gb': 0, 'checkpoint_entries': 0},
    )
    try:
        partial_query = tokens_32[:16] + tuple(7000 + i for i in range(16))
        warm_cache, matched = mgr2.lookup_prefix_disk_cache(partial_query, extra_hash=11)
        assert matched == 16
        assert warm_cache is not None and len(warm_cache) == 1
        assert mx.array_equal(warm_cache[0].keys[..., :16, :], k0[..., :16, :])
        assert mx.array_equal(warm_cache[0].values[..., :16, :], v0[..., :16, :])

        # Changing precision salt invalidates block lookup cleanly
        monkeypatch.setenv('QUANTEEZE_BITS', '2')
        assert mgr2.lookup_prefix_disk_cache(partial_query, extra_hash=11) == (None, 0)
    finally:
        store2.close()


def test_kamepoolalloc_build_and_preload(tmp_path, monkeypatch):
    """`allocator` compiles bundled `kamepoolalloc` (`rss-zero-loss`), caches it, and intercepts `malloc` under preload."""
    import json
    import subprocess
    import sys
    from quanteeze import allocator

    monkeypatch.setenv('QUANTEEZE_SETTINGS', 'off')
    monkeypatch.setenv('QUANTEEZE_ALLOCATOR', 'off')
    assert allocator.ensure_active([]) is None

    monkeypatch.setenv('QUANTEEZE_ALLOCATOR', 'invalid_mode')
    with pytest.raises(SystemExit):
        allocator.ensure_active([])
    monkeypatch.setenv('QUANTEEZE_ALLOCATOR', 'auto')

    if not allocator.supported() or allocator.find_compiler() is None:
        pytest.skip('kamepoolalloc preload test requires macOS/Linux and a C++17 compiler')

    built_msgs = []
    lib1 = allocator.ensure_library(cache_dir=tmp_path, notify=built_msgs.append)
    assert lib1.is_file() and lib1.stat().st_size > 0
    assert len(built_msgs) == 1

    # Second call hits the SHA-256 cached shared library without invoking the compiler
    cached_msgs = []
    lib2 = allocator.ensure_library(cache_dir=tmp_path, notify=cached_msgs.append)
    assert lib2 == lib1
    assert cached_msgs == []

    probe_code = """
import ctypes, json, mlx.core as mx, numpy as np
from quanteeze import allocator

libc = ctypes.CDLL(None)
libc.malloc.argtypes = [ctypes.c_size_t]
libc.malloc.restype = ctypes.c_void_p
libc.free.argtypes = [ctypes.c_void_p]
libc.free.restype = None

p_small = libc.malloc(96)
p_large = libc.malloc(1 << 20)
u_small = allocator.usable_size(p_small)
u_large = allocator.usable_size(p_large)
libc.free(p_small)
libc.free(p_large)

arr_np = np.ones((256, 256), dtype=np.float32)
arr_mx = mx.array(arr_np) + 1.0
mx.eval(arr_mx)

info = {
    "active": allocator.active(),
    "u_small": u_small,
    "u_large": u_large,
    "reserved": allocator.reserved_bytes(),
    "cap": allocator.large_cache_cap(),
    "trimmed": allocator.trim(),
    "mx_sum": float(mx.sum(arr_mx).item()),
}
print(json.dumps(info))
"""
    env = allocator.preload_env(lib1)
    env['PYTHONPATH'] = str(ROOT / 'src')
    proc = subprocess.run([sys.executable, '-c', probe_code], env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f'preload subprocess failed:\nstdout={proc.stdout}\nstderr={proc.stderr}'
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data['active'] is True
    assert data['u_small'] >= 96
    assert data['u_large'] >= (1 << 20)
    assert data['reserved'] > 0
    assert data['cap'] > 0
    assert data['trimmed'] is True
    assert data['mx_sum'] == float(256 * 256 * 2.0)

    # Verify full `serve.main()` automatic `os.execve` preload re-exec path (runs `--help` and exits 0)
    import os
    import shutil
    fake_home = tmp_path / 'home'
    kpa_cache = fake_home / '.quanteeze' / 'kamepoolalloc'
    kpa_cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lib1, kpa_cache / lib1.name)

    reexec_env = {
        k: v for k, v in os.environ.items()
        if k not in ('DYLD_INSERT_LIBRARIES', 'LD_PRELOAD', allocator.PRELOAD_SENTINEL)
    }
    reexec_env.update(
        HOME=str(fake_home),
        PYTHONPATH=str(ROOT / 'src'),
        QUANTEEZE_ALLOCATOR='on',
        QUANTEEZE_SETTINGS='off',
        QUANTEEZE_CONTROL_TOKEN='test-token-1234',
        QUANTEEZE_PARALLEL='off',
    )
    reexec = subprocess.run(
        [sys.executable, '-c', 'from quanteeze.serve import main; main()', '--help'],
        env=reexec_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert reexec.returncode == 0, f'serve.main --help re-exec failed:\nstdout={reexec.stdout}\nstderr={reexec.stderr}'
    banner = 'quanteeze: host allocator: kamepoolalloc (rss-zero-loss,'
    assert reexec.stderr.count(banner) == 1, f'expected single allocator banner in stderr:\n{reexec.stderr}'


def _assert_gguf_close(got, want, rel=2e-2):
    """Compare GGUF outputs relative to peak output magnitude, matching `gguf_qmv._checked`."""
    g = np.asarray(got.astype(mx.float32) if hasattr(got, 'astype') else got, dtype=np.float32)
    w = np.asarray(want.astype(mx.float32) if hasattr(want, 'astype') else want, dtype=np.float32)
    assert g.shape == w.shape
    scale = max(float(np.abs(w).max()), 1.0)
    np.testing.assert_allclose(g, w, rtol=rel, atol=rel * scale)


def _random_ggml_blocks(rng, kind, n_blocks):
    """Synthesize finite `uint8 [n_blocks, BLOCK[kind]]` ggml blocks for any supported type."""
    from quanteeze import iq_experts

    bb = iq_experts.BLOCK[kind]
    raw = rng.integers(0, 256, size=(n_blocks, bb), dtype=np.uint8)

    def f16_bytes(count):
        return rng.uniform(0.02, 0.2, size=(n_blocks, count)).astype(np.float16).view(np.uint8)

    if kind == iq_experts.F16:
        return rng.uniform(-0.2, 0.2, size=(n_blocks, 32)).astype(np.float16).view(np.uint8)
    if kind == iq_experts.BF16:
        words = (rng.uniform(-0.2, 0.2, size=(n_blocks, 32)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        return words.view(np.uint8)
    if kind in (
        iq_experts.Q4_0, iq_experts.Q5_0, iq_experts.Q8_0, iq_experts.Q2_0, iq_experts.IQ4_NL,
        iq_experts.IQ4_XS, iq_experts.IQ2_XXS, iq_experts.IQ2_XS, iq_experts.IQ2_S,
        iq_experts.IQ1_S, iq_experts.IQ3_XXS,
    ):
        raw[:, 0:2] = f16_bytes(1)
    elif kind in (iq_experts.Q4_1, iq_experts.Q5_1, iq_experts.Q4_K, iq_experts.Q5_K):
        raw[:, 0:4] = f16_bytes(2)
    elif kind == iq_experts.Q2_K:
        raw[:, 80:84] = f16_bytes(2)
    elif kind == iq_experts.Q3_K:
        raw[:, 108:110] = f16_bytes(1)
    elif kind == iq_experts.Q6_K:
        raw[:, 208:210] = f16_bytes(1)
    elif kind == iq_experts.MXFP4:
        raw[:, 0] = rng.integers(120, 132, size=n_blocks, dtype=np.uint8)
    elif kind == iq_experts.IQ1_M:
        # Top 4 bits of sc[0..3] (at bytes 48..55) pack fp16 super-scale d_m = 0x2e66 (~0.1).
        for byte_idx, nib in ((49, 0x6), (51, 0x6), (53, 0xe), (55, 0x2)):
            raw[:, byte_idx] = (raw[:, byte_idx] & 0x0f) | (nib << 4)
    return raw


def _write_gguf(path, tensors, alignment=32):
    """Write a minimal GGUF v3 file with `(name, ggml_dims, kind, data_bytes)` entries."""
    import struct

    def pack_str(s):
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b

    buf = bytearray()
    buf += b'GGUF'
    buf += struct.pack('<IQQ', 3, len(tensors), 3)
    # KV 1: general.alignment (uint32, type 4)
    buf += pack_str('general.alignment') + struct.pack('<II', 4, alignment)
    # KV 2: general.name (string, type 8)
    buf += pack_str('general.name') + struct.pack('<I', 8) + pack_str('quanteeze-test')
    # KV 3: uint32 array (type 9, elem uint32 type 4) to exercise GGUFFile array skipping
    buf += pack_str('test.array') + struct.pack('<IIQII', 9, 4, 2, 7, 11)

    offset = 0
    aligned_blobs = []
    for name, dims, kind, data in tensors:
        buf += pack_str(name)
        buf += struct.pack('<I', len(dims))
        buf += struct.pack(f'<{len(dims)}Q', *dims)
        buf += struct.pack('<IQ', kind, offset)
        pad = (-len(data)) % alignment
        aligned_blobs.append(data + b'\x00' * pad)
        offset += len(data) + pad

    header_pad = (-len(buf)) % alignment
    buf += b'\x00' * header_pad
    with open(path, 'wb') as f:
        f.write(buf)
        for blob in aligned_blobs:
            f.write(blob)


@pytest.mark.parametrize('kind,name', sorted(__import__('quanteeze.iq_experts', fromlist=['TYPES']).TYPES.items()))
def test_iq_experts_and_gguf_qmv_all_21_types(kind, name):
    """All 22 ggml types (including MXFP4) decode identically via `requant.dequantize` and match `gguf_qmv` on GPU."""
    import mlx.nn as nn
    from quanteeze import cuda_kernels, gguf_qmv, iq_experts

    rng = np.random.default_rng(1000 + kind)
    e, n, pb = 6, 36, iq_experts.per_block(kind)
    k = max(64, 2 * pb)
    kb = k // pb
    raw_np = _random_ggml_blocks(rng, kind, e * n * kb).reshape(e, n, kb, iq_experts.BLOCK[kind])
    raw_mx = mx.array(raw_np)

    mode = f'gguf:{name.lower()}'
    dec_direct = iq_experts.decode(raw_mx.reshape(-1, raw_mx.shape[-1]), kind).reshape(e, n, k)
    dec_req = requant.dequantize(raw_mx, None, None, pb, max(1, (8 * iq_experts.BLOCK[kind]) // pb), mode=mode)
    mx.eval(dec_direct, dec_req)
    np.testing.assert_allclose(np.array(dec_req), np.array(dec_direct), rtol=0, atol=0)

    if not (mx.metal.is_available() or cuda_kernels.available()) or mx.default_device() != mx.gpu:
        return

    proj = nn.Module()
    proj.weight = raw_mx
    proj.mode = mode
    proj.group_size = pb
    proj.bits = max(1, (8 * iq_experts.BLOCK[kind]) // pb)
    mx.eval(proj.parameters())

    def ref_fn(x_in, idx_in):
        w_unp = requant.dequantize(proj.weight, None, None, pb, proj.bits, mode=mode).astype(x_in.dtype)
        return mx.gather_mm(x_in, w_unp.swapaxes(-1, -2), rhs_indices=idx_in)

    # Decode path (`gather`)
    x_dec = mx.array(rng.standard_normal((2, 1, 1, k)).astype(np.float32), dtype=mx.bfloat16)
    idx_dec = mx.array([[[0, 2, 5]], [[4, 1, 3]]], dtype=mx.uint32)
    mx.eval(x_dec, idx_dec)
    want_dec = ref_fn(x_dec, idx_dec)
    mx.eval(want_dec)
    assert gguf_qmv.takes(proj, x_dec, idx_dec)
    prev_served = gguf_qmv.STATS['served']
    for _ in range(gguf_qmv.CHECKS + 2):
        got_dec = gguf_qmv.gather(proj, x_dec, idx_dec, lambda: ref_fn(x_dec, idx_dec))
        mx.eval(got_dec)
        assert float(mx.abs(got_dec.astype(mx.float32) - want_dec.astype(mx.float32)).max().item()) <= (
            gguf_qmv.TOLERANCE * max(float(mx.abs(want_dec.astype(mx.float32)).max().item()), 1e-30)
        )
    assert gguf_qmv.STATS['mismatched'] == 0
    assert gguf_qmv.STATS['served'] >= prev_served + 2

    # Prefill path (`gather_rows`) with unsorted repeated indices in float32 & bfloat16
    for dtype in (mx.float32, mx.bfloat16):
        x_pre = mx.array(rng.standard_normal((40, 1, k)).astype(np.float32), dtype=dtype)
        idx_pre = mx.array((np.arange(40, dtype=np.uint32) * 5 + 1) % e, dtype=mx.uint32)
        mx.eval(x_pre, idx_pre)
        want_pre = ref_fn(x_pre, idx_pre)
        mx.eval(want_pre)
        assert gguf_qmv.takes_rows(proj, x_pre, idx_pre)
        prev_rows_served = gguf_qmv.STATS['served']
        for _ in range(gguf_qmv.CHECKS + 2):
            got_pre = gguf_qmv.gather_rows(proj, x_pre, idx_pre, lambda: ref_fn(x_pre, idx_pre))
            mx.eval(got_pre)
            assert float(mx.abs(got_pre.astype(mx.float32) - want_pre.astype(mx.float32)).max().item()) <= (
                gguf_qmv.TOLERANCE * max(float(mx.abs(want_pre.astype(mx.float32)).max().item()), 1e-30)
            )
        assert gguf_qmv.STATS['mismatched'] == 0
        assert gguf_qmv.STATS['served'] >= prev_rows_served + 2


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal-only K-quant kernels')
def test_kquant_qmv_iq2_xs():
    """Metal `kquant_qmv` supports uniform `iq2_xs` for both decode (`M=6`) and prefill (`M=300`)."""
    from quanteeze import demand, ggml_quants, kquant_qmv

    rng = np.random.default_rng(4242)
    w = mx.array((rng.standard_normal((6, 64, 512)) * 0.2).astype(np.float32), dtype=mx.bfloat16)
    body, d, _, _ = ggml_quants.FORMATS['iq2_xs'](w)
    raw_blocks = ggml_quants.blocks(body, d, mode='iq2_xs').reshape(6, 64, 2, 74)
    # Note: kquant_qmv uses body = raw[:, :, :, 2:74] and scales = d (float16)
    proj = demand.Projection(body, d, None, 256, 2, mode='iq2_xs')
    mx.eval(proj.parameters())
    assert raw_blocks.shape[-1] == 74

    def ref_fn(x_in, idx_in):
        w_unp = ggml_quants.dequantize(proj.weight, proj.scales, None, 'iq2_xs').astype(x_in.dtype)
        return mx.gather_mm(x_in, w_unp.swapaxes(-1, -2), rhs_indices=idx_in)

    x_dec = mx.array(rng.standard_normal((2, 1, 1, 512)).astype(np.float32), dtype=mx.bfloat16)
    idx_dec = mx.array([[[0, 2, 4]], [[1, 3, 5]]], dtype=mx.uint32)
    mx.eval(x_dec, idx_dec)
    want_dec = ref_fn(x_dec, idx_dec)
    mx.eval(want_dec)
    assert kquant_qmv.takes(proj, x_dec, idx_dec)
    prev_served = kquant_qmv.STATS['served']
    for _ in range(kquant_qmv.CHECKS + 2):
        got_dec = kquant_qmv.gather(proj, x_dec, idx_dec, lambda: ref_fn(x_dec, idx_dec))
        mx.eval(got_dec)
        assert float(mx.abs(got_dec.astype(mx.float32) - want_dec.astype(mx.float32)).max().item()) <= (
            kquant_qmv.TOLERANCE * max(float(mx.abs(want_dec.astype(mx.float32)).max().item()), 1e-30)
        )
    assert kquant_qmv.STATS['mismatched'] == 0
    assert kquant_qmv.STATS['served'] >= prev_served + 2

    x_pre = mx.array(rng.standard_normal((300, 1, 512)).astype(np.float32), dtype=mx.bfloat16)
    idx_pre = mx.array((np.arange(300, dtype=np.uint32) * 5 + 2) % 6, dtype=mx.uint32)
    mx.eval(x_pre, idx_pre)
    want_pre = ref_fn(x_pre, idx_pre)
    mx.eval(want_pre)
    assert kquant_qmv.takes_rows(proj, x_pre, idx_pre)
    prev_rows = kquant_qmv.STATS['served']
    for _ in range(kquant_qmv.CHECKS + 2):
        got_pre = kquant_qmv.gather_rows(proj, x_pre, idx_pre, lambda: ref_fn(x_pre, idx_pre))
        mx.eval(got_pre)
        assert float(mx.abs(got_pre.astype(mx.float32) - want_pre.astype(mx.float32)).max().item()) <= (
            kquant_qmv.TOLERANCE * max(float(mx.abs(want_pre.astype(mx.float32)).max().item()), 1e-30)
        )
    assert kquant_qmv.STATS['mismatched'] == 0
    assert kquant_qmv.STATS['served'] >= prev_rows + 2


def test_gguf_experts_dynamic_split_shards_in_memory_and_demand(tmp_path):
    """Split dynamic GGUF (`UD-*` stacked + per-expert 2D) works whole in memory and via `Demand`."""
    import mlx.nn as nn
    from quanteeze import demand, gguf_experts, iq_experts

    rng = np.random.default_rng(777)
    n_exp, d_in, d_hid = 12, 256, 64
    layer_specs = [
        # Layer 0: stacked 3D mixed types
        ('stacked', {'gate_proj': iq_experts.IQ2_XXS, 'up_proj': iq_experts.IQ1_S, 'down_proj': iq_experts.Q4_0}),
        # Layer 1: per-expert 2D mixed types
        ('per_expert', {'gate_proj': iq_experts.IQ2_XS, 'up_proj': iq_experts.Q3_K, 'down_proj': iq_experts.IQ4_NL}),
    ]
    gguf_name = {'gate_proj': 'ffn_gate', 'up_proj': 'ffn_up', 'down_proj': 'ffn_down'}

    raw_arrays = {}
    shard_tensors = [[], []]
    for l_idx, (layout, projs) in enumerate(layer_specs):
        for p, kind in projs.items():
            out_d, in_d = (d_in, d_hid) if p == 'down_proj' else (d_hid, d_in)
            pb = iq_experts.per_block(kind)
            kb = in_d // pb
            arr = _random_ggml_blocks(rng, kind, n_exp * out_d * kb).reshape(n_exp, out_d, kb, iq_experts.BLOCK[kind])
            raw_arrays[(l_idx, p)] = (arr, kind)
            if layout == 'stacked':
                shard_tensors[l_idx].append(
                    (f'blk.{l_idx}.{gguf_name[p]}_exps.weight', (in_d, out_d, n_exp), kind, arr.tobytes())
                )
            else:
                for e in range(n_exp):
                    shard_tensors[l_idx].append(
                        (f'blk.{l_idx}.{gguf_name[p]}.{e}.weight', (in_d, out_d), kind, arr[e].tobytes())
                    )

    gdir = tmp_path / 'gguf_shards'
    gdir.mkdir()
    shard1 = gdir / 'ud-model-00001-of-00002.gguf'
    shard2 = gdir / 'ud-model-00002-of-00002.gguf'
    _write_gguf(shard1, shard_tensors[0])
    _write_gguf(shard2, shard_tensors[1])

    src = gguf_experts.ExpertGGUF(shard1)
    assert src.files == [shard1.resolve(), shard2.resolve()]
    assert gguf_experts.ExpertGGUF(gdir).files == src.files
    assert 'gguf' in src.format_label([0, 1])

    def build_model():
        m = nn.Module()
        m.layers = []
        for _ in range(2):
            mlp = nn.Module()
            mlp.switch_mlp = SwitchGLU(input_dims=d_in, hidden_dims=d_hid, num_experts=n_exp, bias=False)
            layer = nn.Module()
            layer.mlp = mlp
            m.layers.append(layer)
        return m

    def ref_switch(l_idx, x_in, idx_in):
        decoded = {}
        for p in ('gate_proj', 'up_proj', 'down_proj'):
            arr, kind = raw_arrays[(l_idx, p)]
            in_d = d_hid if p == 'down_proj' else d_in
            out_d = d_in if p == 'down_proj' else d_hid
            w = iq_experts.decode(mx.array(arr).reshape(-1, arr.shape[-1]), kind)
            decoded[p] = w.reshape(n_exp, out_d, in_d).astype(x_in.dtype)
        x_exp = mx.expand_dims(x_in, (-2, -3))
        gate = mx.gather_mm(x_exp, decoded['gate_proj'].swapaxes(-1, -2), rhs_indices=idx_in)
        up = mx.gather_mm(x_exp, decoded['up_proj'].swapaxes(-1, -2), rhs_indices=idx_in)
        hidden = nn.silu(gate) * up
        return mx.gather_mm(hidden, decoded['down_proj'].swapaxes(-1, -2), rhs_indices=idx_in).squeeze(-2)

    m_mem = build_model()
    gguf_experts.load_model(m_mem.layers, shard1)
    assert abs(fitting.weights(m_mem.layers[0].mlp.switch_mlp) - (d_in * d_hid * 3 * n_exp)) < 1e-6

    m_ssd = build_model()
    dem = demand.Demand(m_ssd, gguf_experts.ExpertGGUF(gdir), budget_bytes=1, workers=2)
    assert dem.slots == 8                               # forces eviction across 12 experts
    try:
        batches = [
            (mx.array(rng.standard_normal((1, 1, d_in)).astype(np.float32), dtype=mx.bfloat16),
             mx.array([[[0, 3, 7, 11]]], dtype=mx.uint32)),
            (mx.array(rng.standard_normal((1, 1, d_in)).astype(np.float32), dtype=mx.bfloat16),
             mx.array([[[1, 4, 8, 10]]], dtype=mx.uint32)),
            (mx.array(rng.standard_normal((1, 20, d_in)).astype(np.float32), dtype=mx.bfloat16),
             mx.array((np.arange(80, dtype=np.uint32).reshape(1, 20, 4)) % n_exp)),
        ]
        for l_idx in (0, 1):
            for x_in, idx_in in batches:
                mx.eval(x_in, idx_in)
                y_ref = ref_switch(l_idx, x_in, idx_in)
                y_mem = m_mem.layers[l_idx].mlp.switch_mlp(x_in, idx_in)
                y_ssd = m_ssd.layers[l_idx].mlp.switch_mlp(x_in, idx_in)
                mx.eval(y_ref, y_mem, y_ssd)
                _assert_gguf_close(y_mem, y_ref, rel=2e-2)
                _assert_gguf_close(y_ssd, y_ref, rel=2e-2)
        assert dem.misses >= 4
        assert dem.prefill_layers == 2
    finally:
        dem.close()


def test_cli_experts_gguf_flag(tmp_path, monkeypatch):
    """`serve._extract_cli_flags` strips `--experts-gguf` cleanly and rejects invalid combinations."""
    from quanteeze import iq_experts, serve

    gpath = tmp_path / 'experts.gguf'
    arr = _random_ggml_blocks(np.random.default_rng(1), iq_experts.IQ2_XXS, 2)
    _write_gguf(gpath, [('blk.0.ffn_gate_exps.weight', (256, 1, 2), iq_experts.IQ2_XXS, arr.tobytes())])

    monkeypatch.delenv('QUANTEEZE_EXPERTS_GGUF', raising=False)
    monkeypatch.delenv('QUANTEEZE_BITS', raising=False)

    rest = serve._extract_cli_flags(['quanteeze', '--model', 'm', '--experts-gguf', str(gpath), '--port', '8765'])
    assert rest == ['quanteeze', '--model', 'm', '--port', '8765']
    assert serve.variables.get('QUANTEEZE_EXPERTS_GGUF') == str(gpath)
    assert serve._experts_gguf(None) is not None

    monkeypatch.delenv('QUANTEEZE_EXPERTS_GGUF', raising=False)
    rest2 = serve._extract_cli_flags(['quanteeze', f'--experts-gguf={gpath}', '--enable-thinking'])
    assert rest2 == ['quanteeze', '--enable-thinking']
    assert serve.variables.get('QUANTEEZE_EXPERTS_GGUF') == str(gpath)

    with pytest.raises(SystemExit):
        serve._extract_cli_flags(['quanteeze', '--experts-gguf'])

    with pytest.raises(SystemExit):
        serve._experts_gguf((2, 64))


def test_host_copy_whole_and_by_rows(tmp_path, monkeypatch):
    """`HostCopy` holds whole tensors (`by_rows=False`) or equal per-layer row slices (`by_rows=True`)."""
    import mlx.nn as nn
    from quanteeze import demand, q2_experts

    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')

    mx.random.seed(13)
    tensors = {}
    for layer in range(2):
        sw = SwitchGLU(input_dims=64, hidden_dims=64, num_experts=16, bias=False)
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(sw, proj, getattr(sw, proj).to_quantized(group_size=64, bits=4))
        mx.eval(sw.parameters())
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            p = getattr(sw, proj)
            tensors[f'layers.{layer}.mlp.switch_mlp.{proj}.weight'] = p.weight
            tensors[f'layers.{layer}.mlp.switch_mlp.{proj}.scales'] = p.scales
            tensors[f'layers.{layer}.mlp.switch_mlp.{proj}.biases'] = p.biases
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)

    ckpt = q2_experts.Checkpoint(tmp_path)
    layer0_bytes = sum(ckpt.nbytes(n) for n in ckpt.routed() if n.startswith('layers.0.'))
    sample_name = 'layers.0.mlp.switch_mlp.gate_proj.weight'
    layer1_name = 'layers.1.mlp.switch_mlp.gate_proj.weight'

    def make_model():
        m = nn.Module()
        m.layers = []
        for _ in range(2):
            sw = SwitchGLU(input_dims=64, hidden_dims=64, num_experts=16, bias=False)
            for proj in ('gate_proj', 'up_proj', 'down_proj'):
                setattr(sw, proj, getattr(sw, proj).to_quantized(group_size=64, bits=4))
            mlp = nn.Module()
            mlp.switch_mlp = sw
            layer = nn.Module()
            layer.mlp = mlp
            m.layers.append(layer)
        return m

    try:
        whole = q2_experts.hold(tmp_path, budget=layer0_bytes, workers=1, by_rows=False)
        whole.wait()
        assert not whole.by_rows
        assert sample_name in whole.arrays and layer1_name not in whole.arrays
        assert np.array_equal(whole.rows(sample_name), np.asarray(tensors[sample_name]))
        assert whole.rows(layer1_name) is None
        d_whole = demand.Demand(make_model(), ckpt, budget_bytes=1, workers=1)
        assert d_whole._host() is None
        d_whole.close()
    finally:
        q2_experts.let_go(tmp_path)

    half_bytes = sum(ckpt.nbytes(n) for n in ckpt.routed()) // 2
    try:
        by_rows = q2_experts.hold(tmp_path, budget=half_bytes, workers=1, by_rows=True)
        by_rows.wait()
        assert by_rows.by_rows
        assert list(by_rows.expert_in[0]) == list(range(8))
        assert list(by_rows.expert_in[1]) == list(range(8))
        assert by_rows.has(0, 0) and by_rows.has(1, 7) and not by_rows.has(0, 8)
        assert np.array_equal(by_rows.rows(sample_name, [0, 1]), np.asarray(tensors[sample_name])[:2])
        assert by_rows.rows(sample_name, [8]) is None
        d_rows = demand.Demand(make_model(), ckpt, budget_bytes=1, workers=1)
        assert d_rows._host() is by_rows
        d_rows.close()
    finally:
        q2_experts.let_go(tmp_path)


def test_uniform_bf16_on_demand_cached_checkpoint_and_host_copy(tmp_path, monkeypatch):
    """Uniform bf16 on-demand materializes `QuantCache`, holds quantized experts in `HostCopy`, and
    serves them (`host_served > 0`) matching `requant.lower_model` bit-for-bit."""
    import mlx.nn as nn
    from quanteeze import demand, q2_experts

    monkeypatch.setenv('QUANTEEZE_QUANT_CACHE', str(tmp_path / 'qcache'))
    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')
    mx.random.seed(404)
    tensors = {}
    for layer in range(2):
        sw = SwitchGLU(input_dims=256, hidden_dims=256, num_experts=16, bias=False)
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            p = getattr(sw, proj)
            p.weight = p.weight.astype(mx.bfloat16)
            tensors[f'layers.{layer}.mlp.switch_mlp.{proj}.weight'] = p.weight
    mx.eval(*tensors.values())
    ckpt_dir = tmp_path / 'bf16_ckpt'
    ckpt_dir.mkdir()
    (ckpt_dir / 'config.json').write_text('{}', encoding='utf-8')
    mx.save_safetensors(str(ckpt_dir / 'model.safetensors'), tensors)
    bf16_bytes = sum(a.nbytes for a in tensors.values())

    def make_bf16_model():
        m = nn.Module()
        m.layers = []
        for _ in range(2):
            sw = SwitchGLU(input_dims=256, hidden_dims=256, num_experts=16, bias=False)
            for proj in ('gate_proj', 'up_proj', 'down_proj'):
                p = getattr(sw, proj)
                p.weight = p.weight.astype(mx.bfloat16)
            mlp = nn.Module()
            mlp.switch_mlp = sw
            layer = nn.Module()
            layer.mlp = mlp
            m.layers.append(layer)
        return m

    m_ref = make_bf16_model()
    cached = requant.cached_checkpoint(m_ref.layers, ckpt_dir, 4, 64)
    assert isinstance(cached, requant.CachedCheckpoint)
    for layer in range(2):
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            done_path = cached.path / f'{layer}.{proj}.done.npy'
            assert done_path.exists()
            assert bool(np.load(done_path).all())

    calls = [0]
    orig_quantize = requant.quantize

    def counted_quantize(*args, **kwargs):
        calls[0] += 1
        return orig_quantize(*args, **kwargs)

    monkeypatch.setattr(requant, 'quantize', counted_quantize)
    cached2 = requant.cached_checkpoint(m_ref.layers, ckpt_dir, 4, 64)
    assert isinstance(cached2, requant.CachedCheckpoint)
    assert calls[0] == 0

    requant.lower_model(m_ref.layers, ckpt_dir, 4, 64)
    for layer in m_ref.layers:
        expert_qmv.install_switch(layer.mlp.switch_mlp)

    cached_bytes = sum(cached.nbytes(n) for n in cached.routed())
    budget = int(cached_bytes * 12 // 16 + 1024)
    m_ssd = make_bf16_model()
    held = q2_experts.hold(cached, budget=budget, workers=2, by_rows=True)
    dem = None
    try:
        held.wait()
        assert held.by_rows
        assert held.total < bf16_bytes // 2
        assert list(held.expert_in[0]) == list(range(12))
        dem = demand.Demand(m_ssd, cached, budget_bytes=1, workers=2)
        assert dem._host() is held

        x_host = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
        idx_host = mx.array([[[8, 9, 10, 11]]], dtype=mx.uint32)
        mx.eval(x_host, idx_host)
        for layer_idx in range(2):
            y_ref = m_ref.layers[layer_idx].mlp.switch_mlp(x_host, idx_host)
            y_ssd = m_ssd.layers[layer_idx].mlp.switch_mlp(x_host, idx_host)
            mx.eval(y_ref, y_ssd)
            assert np.array_equal(
                np.array(y_ssd.astype(mx.float32)),
                np.array(y_ref.astype(mx.float32)),
            )
        assert dem.host_served == 8

        x_disk = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
        idx_disk = mx.array([[[12, 13, 14, 15]]], dtype=mx.uint32)
        mx.eval(x_disk, idx_disk)
        for layer_idx in range(2):
            y_ref = m_ref.layers[layer_idx].mlp.switch_mlp(x_disk, idx_disk)
            y_ssd = m_ssd.layers[layer_idx].mlp.switch_mlp(x_disk, idx_disk)
            mx.eval(y_ref, y_ssd)
            assert np.array_equal(
                np.array(y_ssd.astype(mx.float32)),
                np.array(y_ref.astype(mx.float32)),
            )
        assert dem.host_served == 8
        assert dem.misses == 16
    finally:
        if dem is not None:
            dem.close()
        q2_experts.let_go(cached.path)


def test_idle_false_during_host_copy_and_clean_shutdown_helpers(tmp_path, monkeypatch):
    """`serve._idle()` stays False during active `HostCopy`, `Keeper.close()` joins promptly, and
    `serve._terminated()` uses exit code 0 after graceful shutdown signalling."""
    import concurrent.futures
    import signal
    import time
    from quanteeze import guards, q2_experts, serve, shutdown

    fut = concurrent.futures.Future()
    fake_held = type('FakeHeld', (), {'reading': lambda self: not fut.done(), 'close': lambda self: None})()
    monkeypatch.setitem(q2_experts._HELD, tmp_path / 'fake', fake_held)
    monkeypatch.setitem(serve.control._SWITCH, 'inflight', 0)
    monkeypatch.setitem(serve.control._SWITCH, 'busy', False)
    monkeypatch.setattr(serve.progress, 'state', lambda: {'loading': False})
    monkeypatch.setattr(serve.ngram_prefetch, 'unread', lambda: 0)

    assert q2_experts.reading()
    assert not serve._idle()
    fut.set_result(None)
    assert not q2_experts.reading()
    assert serve._idle()

    keeper = guards.Keeper(limit=100e9, poll=60.0, log=lambda _msg: None).start()
    try:
        assert keeper._thread.is_alive()
    finally:
        t0 = time.monotonic()
        keeper.close()
        elapsed = time.monotonic() - t0
    assert not keeper._thread.is_alive()
    assert elapsed < 2.0

    monkeypatch.setitem(shutdown._STATE, 'signalled', True)
    with pytest.raises(SystemExit) as exc_graceful:
        serve._terminated(signal.SIGTERM, None)
    assert exc_graceful.value.code == 0

    monkeypatch.setitem(shutdown._STATE, 'signalled', False)
    with pytest.raises(SystemExit) as exc_ungraceful:
        serve._terminated(signal.SIGTERM, None)
    assert exc_ungraceful.value.code == 128 + signal.SIGTERM


def test_demand_batched_place_many_and_ring_staging(tmp_path, monkeypatch):
    """`Demand` updates all missed slots in one `place_many` batch, updates `_local` in place, wraps
    `_STAGE_RING` cleanly over many decode steps, and matches stock `SwitchGLU` bit-for-bit."""
    import mlx.nn as nn
    from quanteeze import demand, q2_experts

    monkeypatch.setenv('QUANTEEZE_CPU_EXPERTS', 'off')

    mx.random.seed(505)
    tensors = {}
    for layer in range(2):
        sw = SwitchGLU(input_dims=256, hidden_dims=256, num_experts=16, bias=False)
        nn.quantize(sw, group_size=64, bits=4)
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            p = getattr(sw, proj)
            for part in ('weight', 'scales', 'biases'):
                arr = getattr(p, part)
                tensors[f'layers.{layer}.mlp.switch_mlp.{proj}.{part}'] = arr
    mx.eval(*tensors.values())
    (tmp_path / 'config.json').write_text('{}', encoding='utf-8')
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)

    def make_q_model():
        m = nn.Module()
        m.layers = []
        for layer in range(2):
            sw = SwitchGLU(input_dims=256, hidden_dims=256, num_experts=16, bias=False)
            nn.quantize(sw, group_size=64, bits=4)
            for proj in ('gate_proj', 'up_proj', 'down_proj'):
                p = getattr(sw, proj)
                for part in ('weight', 'scales', 'biases'):
                    setattr(p, part, tensors[f'layers.{layer}.mlp.switch_mlp.{proj}.{part}'])
            mlp = nn.Module()
            mlp.switch_mlp = sw
            layer_mod = nn.Module()
            layer_mod.mlp = mlp
            m.layers.append(layer_mod)
        return m

    m_ref = make_q_model()
    for layer_mod in m_ref.layers:
        expert_qmv.install_switch(layer_mod.mlp.switch_mlp)
    m_ssd = make_q_model()
    ckpt = q2_experts.Checkpoint(tmp_path)
    held = q2_experts.hold(ckpt, budget=10**9, workers=2, by_rows=True)
    dem = None
    try:
        held.wait()
        dem = demand.Demand(m_ssd, ckpt, budget_bytes=1, workers=2)
        assert dem.slots == 8
        step_indices = [
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 10, 12, 14],
            [1, 3, 9, 11],
            [5, 7, 13, 15],
        ]
        # Verify lazy pipelining across > 2 * _STAGE_RING layer calls before evaluating outputs.
        lazy_pairs = []
        for exps in step_indices:
            x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
            idx = mx.array([[exps]], dtype=mx.uint32)
            mx.eval(x, idx)
            for layer_idx in range(2):
                y_ref = m_ref.layers[layer_idx].mlp.switch_mlp(x, idx)
                y_ssd = m_ssd.layers[layer_idx].mlp.switch_mlp(x, idx)
                lazy_pairs.append((y_ref, y_ssd))
        for y_ref, y_ssd in lazy_pairs:
            mx.eval(y_ref, y_ssd)
            assert np.array_equal(
                np.array(y_ssd.astype(mx.float32)),
                np.array(y_ref.astype(mx.float32)),
            )
        assert dem.misses > 0
        assert dem.host_served == dem.misses
        for sw in dem.switches.values():
            local_np = np.array(sw._local)
            for e in range(16):
                slot = int(sw.route[e])
                assert int(local_np[e]) == (slot if slot >= 0 else 0)
    finally:
        if dem is not None:
            dem.close()
        q2_experts.let_go(ckpt.path)


def test_advise_host_staging_and_cuda_experts_projection(monkeypatch):
    """`cuda_ple.advise_host_staging` issues `cuMemAdvise` CPU/device hints on >=4KB views, and
    `cuda_experts._projection` accepts `demand.Projection` 4-bit affine modules."""
    from quanteeze import cuda_experts, cuda_kernels, cuda_ple, demand

    assert cuda_ple.advise_host_staging(np.zeros((64,), dtype=np.uint8), enable=True) is False

    calls = []

    class FakeCuda:
        def cuCtxPushCurrent_v2(self, ctx):
            return 0

        def cuCtxPopCurrent_v2(self, out_ctx):
            return 0

        def cuDeviceGet(self, out_dev, ordinal):
            out_dev._obj.value = 3
            return 0

        def cuMemAdvise(self, ptr, size, advice, device):
            sz = size.value if hasattr(size, 'value') else int(size)
            adv = advice.value if hasattr(advice, 'value') else int(advice)
            dev = device.value if hasattr(device, 'value') else int(device)
            calls.append((int(sz), int(adv), int(dev)))
            return 0

    monkeypatch.setattr(cuda_ple.mx.metal, 'is_available', lambda: False)
    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
    monkeypatch.setattr(cuda_ple, '_DRIVER', [(FakeCuda(), object())])

    buf = np.zeros((4096,), dtype=np.uint8)
    assert cuda_ple.advise_host_staging(buf, enable=True) is True
    assert cuda_ple.advise_host_staging(buf, enable=False) is True
    assert calls == [
        (4096, 3, -1),
        (4096, 5, 3),
        (4096, 6, 3),
        (4096, 4, -1),
    ]

    proj = demand.Projection(
        mx.zeros((8, 64, 16), dtype=mx.uint32),
        mx.ones((8, 64, 2), dtype=mx.bfloat16),
        mx.zeros((8, 64, 2), dtype=mx.bfloat16),
        group_size=64,
        bits=4,
        mode='affine',
    )
    assert cuda_experts._projection(proj, 128, mx.bfloat16) is True


def test_prefill_layer_reuse_no_ahead_leak_and_cuda_device_donate(tmp_path, monkeypatch):
    """Layer-at-a-time prefill loads each MoE layer once across windows, frees `_ahead` / staging
    buffers after the last layer, and separates `prefill_read_s` from decode `read_s`."""
    import mlx.nn as nn
    from quanteeze import demand, prefill, q2_experts

    class DummyCache:
        def __init__(self):
            self.offset = 0
            self.max_size = None

    class SplitMoELayer(nn.Module):
        def __init__(self, sw):
            super().__init__()
            mlp = nn.Module()
            mlp.switch_mlp = sw
            mlp.top_k = 4
            self.mlp = mlp

        def _pre(self, x, idx, cache=None):
            if cache is not None:
                cache.offset += x.shape[1]
            return x, idx

        def _post(self, x, idx):
            return self.mlp.switch_mlp(x, idx).sum(axis=-2)

        def __call__(self, x, idx, cache=None):
            h, i = self._pre(x, idx, cache=cache)
            return self._post(h, i)

    class ToyLM(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

        def make_cache(self):
            return [DummyCache() for _ in self.layers]

    prefill.split(SplitMoELayer, SplitMoELayer._pre, SplitMoELayer._post)

    mx.random.seed(606)
    tensors = {}
    for layer_idx in range(2):
        sw = SwitchGLU(input_dims=256, hidden_dims=256, num_experts=16, bias=False)
        nn.quantize(sw, group_size=64, bits=4)
        for p_name in ('gate_proj', 'up_proj', 'down_proj'):
            p = getattr(sw, p_name)
            for part in ('weight', 'scales', 'biases'):
                tensors[f'layers.{layer_idx}.mlp.switch_mlp.{p_name}.{part}'] = getattr(p, part)
    mx.eval(*tensors.values())
    (tmp_path / 'config.json').write_text('{}', encoding='utf-8')
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), tensors)

    def make_model():
        layers = []
        for layer_idx in range(2):
            sw = SwitchGLU(input_dims=256, hidden_dims=256, num_experts=16, bias=False)
            nn.quantize(sw, group_size=64, bits=4)
            for p_name in ('gate_proj', 'up_proj', 'down_proj'):
                p = getattr(sw, p_name)
                for part in ('weight', 'scales', 'biases'):
                    setattr(p, part, tensors[f'layers.{layer_idx}.mlp.switch_mlp.{p_name}.{part}'])
            layers.append(SplitMoELayer(sw))
        return ToyLM(layers)

    m_ref = make_model()
    for layer_mod in m_ref.layers:
        expert_qmv.install_switch(layer_mod.mlp.switch_mlp)

    m_ssd = make_model()
    ckpt = q2_experts.Checkpoint(tmp_path)
    held = q2_experts.hold(ckpt, budget=10**9, workers=2, by_rows=True)
    dem = None
    prefill.JOINS.clear()
    prefill.times(reset=True)
    monkeypatch.setenv('QUANTEEZE_PROMPT_MOE_TOKENS', '32')
    monkeypatch.setattr(demand.mx.metal, 'is_available', lambda: False)
    try:
        held.wait()
        assert held.holds_layer(0) and held.holds_layer(1)

        sample_key = ckpt.name(0, 'gate_proj', 'weight')
        full_buf = np.empty_like(held.arrays[sample_key])
        assert held.copy_layer(0, None, [(sample_key, full_buf)]) is True
        assert np.array_equal(full_buf, np.asarray(tensors[sample_key]))
        sub_buf = np.empty((3,) + held.arrays[sample_key].shape[1:], dtype=held.arrays[sample_key].dtype)
        assert held.copy_layer(0, [2, 5, 11], [(sample_key, sub_buf)]) is True
        assert np.array_equal(sub_buf, np.asarray(tensors[sample_key])[[2, 5, 11]])

        dem = demand.Demand(m_ssd, ckpt, budget_bytes=1, workers=2)
        assert dem.slots == 8
        prefill.install(m_ssd.layers, chunk=16, enabled=lambda: True)

        prompt_len = 68
        x = mx.random.normal((1, prompt_len, 256)).astype(mx.bfloat16)
        idx = mx.array((np.arange(prompt_len * 4, dtype=np.uint32).reshape(1, prompt_len, 4)) % 16)
        mx.eval(x, idx)

        c_ref = m_ref.make_cache()
        c_ref_parts = m_ref.make_cache()
        c_ssd = m_ssd.make_cache()
        h_ref, h_ref_parts, h_ssd = x, x, x
        parts = [(0, 16), (16, 32), (32, 48), (48, 64), (64, 68)]
        for l_idx in range(2):
            h_ref = m_ref.layers[l_idx](h_ref, idx, cache=c_ref[l_idx])
            h_ref_parts = mx.concatenate(
                [
                    m_ref.layers[l_idx](h_ref_parts[:, a:b], idx[:, a:b], cache=c_ref_parts[l_idx])
                    for a, b in parts
                ],
                axis=1,
            )
            h_ssd = m_ssd.layers[l_idx](h_ssd, idx, cache=c_ssd[l_idx])
        mx.eval(h_ref, h_ref_parts, h_ssd)
        np.testing.assert_allclose(
            np.array(h_ssd.astype(mx.float32)),
            np.array(h_ref.astype(mx.float32)),
            rtol=2e-2,
            atol=2e-4,
        )
        assert np.array_equal(
            np.array(h_ssd.astype(mx.float32)),
            np.array(h_ref_parts.astype(mx.float32)),
        )

        assert dem.prefill_layers == 2
        assert dem.hits == 0 and dem.misses == 0
        assert dem.read_s == 0.0
        assert dem.prefill_read_s > 0.0
        assert 'prefill_read_s' in dem.summary()
        assert len(ckpt._ahead) == 0
        assert len(dem._prefill_stage) == 0
        assert len(dem._prefill_dev) == 0
        assert dem._prefill_active is None

        # Decode step on uncached experts records decode read_s and misses separately.
        x_dec = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
        uncached_exps = sorted(set(range(16)) - set(int(e) for e in dem.switches[0].experts))[:4]
        idx_dec = mx.array([[uncached_exps]], dtype=mx.uint32)
        mx.eval(x_dec, idx_dec)
        y_dec_ref = m_ref.layers[0].mlp.switch_mlp(x_dec, idx_dec)
        y_dec_ssd = m_ssd.layers[0].mlp.switch_mlp(x_dec, idx_dec)
        mx.eval(y_dec_ref, y_dec_ssd)
        assert np.array_equal(
            np.array(y_dec_ssd.astype(mx.float32)),
            np.array(y_dec_ref.astype(mx.float32)),
        )
        assert dem.misses == 4
        assert dem.host_served == 4
        assert dem.read_s > 0.0
    finally:
        prefill.release()
        if dem is not None:
            dem.close()
        q2_experts.let_go(ckpt.path)


def test_efforts_qwen36_preopened_response_template_and_chat_template_kwargs():
    """`efforts.install` propagates `chat_template_kwargs.enable_thinking` and supplies the
    pre-opened `<think>\\n` prefix to `response_template` streaming and non-streaming parsers."""
    import importlib
    from mlx_vlm.server import generation
    from mlx_vlm.server.schemas import ChatRequest
    from transformers.utils.chat_parsing.response_parser import ResponseParser
    from quanteeze import efforts

    efforts.install(generation.GenerationArguments)
    efforts.install(generation.GenerationArguments)

    app_mod = importlib.import_module('mlx_vlm.server.app')
    openai_mod = importlib.import_module('mlx_vlm.server.openai')
    resp_state = importlib.import_module('mlx_vlm.server.responses_state')

    req_on = ChatRequest(
        model='qwen3.6',
        messages=[{'role': 'user', 'content': 'solve'}],
        chat_template_kwargs={'enable_thinking': True},
    )
    assert app_mod._build_gen_args(req_on).enable_thinking is True

    req_explicit_off = ChatRequest(
        model='qwen3.6',
        messages=[{'role': 'user', 'content': 'solve'}],
        enable_thinking=False,
        chat_template_kwargs={'enable_thinking': True},
    )
    assert app_mod._build_gen_args(req_explicit_off).enable_thinking is False

    qwen36_template = {
        'version': 1,
        'start_anchor': '<|im_start|>assistant\n',
        'defaults': {'content': None, 'reasoning_content': None},
        'fields': {
            'reasoning_content': {
                'open': '<think>\n',
                'close': '</think>\n\n',
                'optional': True,
            },
            'content': {},
        },
    }

    class FakeTokenizer:
        response_template = qwen36_template

        def get_response_parser(self, prefix=None):
            return ResponseParser(self.response_template, prefix=prefix)

        def parse_response(self, text, prefix=None):
            parser = self.get_response_parser(prefix=prefix)
            parser.feed(text)
            parsed, _ = parser.finalize()
            return parsed

    processor = type('FakeProcessor', (), {'tokenizer': FakeTokenizer()})()

    state = openai_mod.make_response_stream_state(processor, enable_thinking=True)
    reasoning_parts, content_parts = [], []
    closed = False
    chunks = ['Let us ', 'compute 5+5=10.\n', '</think>', '\n\n', '10']
    for i, tok in enumerate(chunks):
        delta = state.feed(tok, last=(i == len(chunks) - 1))
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)
        if delta.thinking_closed:
            closed = True
    assert ''.join(reasoning_parts) == 'Let us compute 5+5=10.\n'
    assert closed is True
    assert ''.join(content_parts) == '10'

    state_off = openai_mod.make_response_stream_state(processor, enable_thinking=False)
    delta_off = state_off.feed('10', last=True)
    assert delta_off.reasoning is None
    assert delta_off.content == '10'

    full_text = 'Let us compute 5+5=10.\n</think>\n\n10'
    assert app_mod._split_thinking(full_text, starts_in_thinking=True, processor=processor) == (
        'Let us compute 5+5=10.', '10'
    )
    assert resp_state._split_thinking(full_text, starts_in_thinking=True, processor=processor) == (
        'Let us compute 5+5=10.', '10'
    )
    assert app_mod._split_thinking(full_text, starts_in_thinking=False, processor=processor) == (
        'Let us compute 5+5=10.', '10'
    )


def test_gguf_qmv_cuda_sources_and_dispatch(tmp_path, monkeypatch):
    """Compile and execute `_CUDA_HEADER` (`qz_decode_octad` aligned/unaligned and `qz_load_bytes`)
    across all 22 ggml formats in C++17, type-check `_CUDA_SOURCE`, `_CUDA_GATE_UP`, and `_CUDA_TILED`,
    and verify CUDA decode/fused-switch/prefill dispatch and padding."""
    import ctypes
    import shutil
    import subprocess
    import mlx.nn as nn
    from quanteeze import cuda_kernels, expert_qmv, gguf_qmv, iq_experts

    cxx = shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')
    if cxx is not None:
        cases = '\n'.join(
            f'      case {k}: if (aligned4) {{ qz_decode_octad<{k}, true>(blk, grid, oct, bo + oct * 8u); }} '
            f'else {{ qz_decode_octad<{k}, false>(blk, grid, oct, bo + oct * 8u); }} break;'
            for k in sorted(iq_experts.TYPES)
        )
        load_cases = '\n'.join(
            f'      case {bb}: qz_load_bytes<{bb}>(dst, src, static_cast<uint32_t>(nbytes), lane); break;'
            for bb in sorted(set(iq_experts.BLOCK.values()))
        )
        cpp = f"""
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>

#define __device__
#define __forceinline__ inline
#define __shared__
#define __align__(x) alignas(x)

struct alignas(8) uint2 {{ uint32_t x, y; }};
struct alignas(16) uint4 {{ uint32_t x, y, z, w; }};
struct float4 {{ float x, y, z, w; }};
struct dim3_stub {{ uint32_t x, y, z; }};
static thread_local dim3_stub blockIdx{{0, 0, 0}}, threadIdx{{0, 0, 0}};
inline void __syncthreads() {{}}
inline float __uint_as_float(uint32_t u) {{ float f; std::memcpy(&f, &u, sizeof(f)); return f; }}
inline uint16_t __ushort_as_half(uint16_t u) {{ return u; }}
inline int __popc(uint32_t x) {{ return __builtin_popcount(x); }}
inline float __half2float(uint16_t h) {{
  uint32_t s = static_cast<uint32_t>(h & 0x8000u) << 16;
  int e = (h >> 10) & 0x1f;
  uint32_t m = h & 0x03ffu;
  if (e == 0) {{
    if (m == 0u) return __uint_as_float(s);
    e = 1;
    while ((m & 0x0400u) == 0u) {{ m <<= 1; e--; }}
    m &= 0x03ffu;
  }} else if (e == 31) {{
    return __uint_as_float(s | 0x7f800000u | (m << 13));
  }}
  return __uint_as_float(s | (static_cast<uint32_t>(e + 112) << 23) | (m << 13));
}}
struct QzSum {{ template <typename U> U operator()(U a, U b) const {{ return a + b; }} }};
struct Sigmoid {{
  template <typename U> U operator()(U x) const {{
    return static_cast<U>(1.0f / (1.0f + std::exp(-static_cast<float>(x))));
  }}
}};
struct Multiply {{
  template <typename U> U operator()(U a, U b) const {{
    return static_cast<U>(static_cast<float>(a) * static_cast<float>(b));
  }}
}};
struct StubBF16 {{
  uint16_t v;
  StubBF16() = default;
  explicit StubBF16(float f) {{ uint32_t u; std::memcpy(&u, &f, 4); v = static_cast<uint16_t>(u >> 16); }}
  explicit operator float() const {{ return __uint_as_float(static_cast<uint32_t>(v) << 16); }}
}};
namespace cg {{
struct Block {{}};
inline Block this_thread_block() {{ return {{}}; }}
template <int N> struct Tile {{
  uint32_t thread_rank() const {{ return 0; }}
  uint32_t meta_group_rank() const {{ return 0; }}
}};
template <int N> Tile<N> tiled_partition(Block) {{ return {{}}; }}
template <int N, typename U, typename Op> U reduce(Tile<N>, U val, Op) {{ return val; }}
}}

{gguf_qmv._CUDA_HEADER}

template <typename T, int K_SIZE, int K_BLOCKS, int N_SIZE, int KIND, int PB, int BB, int ROWS, int WARPS, int X_DIV>
void instantiate_qmv(const T* x, const uint8_t* w, const uint16_t* grid, const int* idx, T* y) {{
{gguf_qmv._CUDA_SOURCE}
}}

template <typename T, int K_SIZE, int N_SIZE, int X_DIV,
          int KIND_G, int PB_G, int BB_G, int KB_G,
          int KIND_U, int PB_U, int BB_U, int KB_U, int ROWS>
void instantiate_gate_up(const T* x, const uint8_t* wg, const uint16_t* grid_g,
                         const uint8_t* wu, const uint16_t* grid_u, const int* idx, T* y) {{
{gguf_qmv._CUDA_GATE_UP}
}}

template <typename T, int K_SIZE, int K_BLOCKS, int N_SIZE, int KIND, int PB, int BB>
void instantiate_mm(const T* x, const uint8_t* w, const uint16_t* grid, const int* tiles, T* y) {{
{gguf_qmv._CUDA_TILED}
}}

template void instantiate_qmv<float, 256, 1, 36, 16, 256, 66, 4, 4, 1>(
    const float*, const uint8_t*, const uint16_t*, const int*, float*);
template void instantiate_qmv<StubBF16, 256, 1, 36, 16, 256, 66, 4, 4, 8>(
    const StubBF16*, const uint8_t*, const uint16_t*, const int*, StubBF16*);
template void instantiate_gate_up<float, 256, 36, 8, 13, 256, 176, 1, 12, 256, 144, 1, 4>(
    const float*, const uint8_t*, const uint16_t*, const uint8_t*, const uint16_t*, const int*, float*);
template void instantiate_gate_up<StubBF16, 256, 36, 8, 13, 256, 176, 1, 12, 256, 144, 1, 4>(
    const StubBF16*, const uint8_t*, const uint16_t*, const uint8_t*, const uint16_t*, const int*, StubBF16*);
template void instantiate_gate_up<float, 256, 36, 1, 14, 256, 210, 1, 18, 256, 98, 1, 4>(
    const float*, const uint8_t*, const uint16_t*, const uint8_t*, const uint16_t*, const int*, float*);
template void instantiate_gate_up<StubBF16, 256, 36, 1, 14, 256, 210, 1, 18, 256, 98, 1, 4>(
    const StubBF16*, const uint8_t*, const uint16_t*, const uint8_t*, const uint16_t*, const int*, StubBF16*);
template void instantiate_mm<float, 256, 1, 36, 16, 256, 66>(
    const float*, const uint8_t*, const uint16_t*, const int*, float*);
template void instantiate_mm<StubBF16, 256, 1, 36, 16, 256, 66>(
    const StubBF16*, const uint8_t*, const uint16_t*, const int*, StubBF16*);

extern "C" void decode_blocks(int kind, const uint8_t* w, const uint16_t* grid, int n_blocks, int pb, int bb,
                              int aligned4, float* out) {{
  alignas(16) uint8_t ablk[256];
  for (int b = 0; b < n_blocks; ++b) {{
    const uint8_t* blk = w + b * bb;
    if (aligned4) {{
      std::memcpy(ablk, blk, static_cast<size_t>(bb));
      blk = ablk;
    }}
    float* bo = out + b * pb;
    for (uint32_t oct = 0; oct < static_cast<uint32_t>(pb / 8); ++oct) {{
      switch (kind) {{
{cases}
      }}
    }}
  }}
}}

extern "C" void check_load_bytes(int bb, const uint8_t* src, int nbytes, uint8_t* dst) {{
  for (uint32_t lane = 0; lane < 32u; ++lane) {{
    switch (bb) {{
{load_cases}
    }}
  }}
}}
"""
        src_path = tmp_path / 'check_gguf_cuda.cpp'
        so_path = tmp_path / 'check_gguf_cuda.so'
        src_path.write_text(cpp, encoding='utf-8')
        subprocess.run([cxx, '-std=c++17', '-O2', '-shared', '-fPIC', str(src_path), '-o', str(so_path)], check=True)
        lib = ctypes.CDLL(str(so_path))
        lib.decode_blocks.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        lib.check_load_bytes.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        grids = iq_experts.compact_grids()
        dummy_grid = np.zeros((8,), dtype=np.uint16)
        rng = np.random.default_rng(2026)
        for kind in sorted(iq_experts.TYPES):
            pb, bb = iq_experts.per_block(kind), iq_experts.BLOCK[kind]
            n_blocks = 4
            raw = _random_ggml_blocks(rng, kind, n_blocks)
            grid_np = np.array(grids[kind], dtype=np.uint16) if kind in grids else dummy_grid
            want = np.array(iq_experts.decode(mx.array(raw), kind))
            for aligned4 in ([0, 1] if (pb == 256 or bb % 4 == 0) else [0]):
                got = np.empty((n_blocks, pb), dtype=np.float32)
                lib.decode_blocks(
                    kind, raw.ctypes.data, grid_np.ctypes.data, n_blocks, pb, bb, aligned4, got.ctypes.data
                )
                np.testing.assert_allclose(got, want, rtol=0, atol=0)

        for kind, bb in sorted({(k, iq_experts.BLOCK[k]) for k in iq_experts.TYPES}):
            pb = iq_experts.per_block(kind)
            for nbytes in sorted({bb, (256 // pb) * bb}):
                src_raw = np.empty(1024, dtype=np.uint8)
                dst_raw = np.empty(1024, dtype=np.uint8)
                off_src = (-src_raw.ctypes.data) % 16
                off_dst = (-dst_raw.ctypes.data) % 16
                src_al = src_raw[off_src:off_src + nbytes]
                dst_al = dst_raw[off_dst:off_dst + nbytes]
                src_al[:] = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
                dst_al.fill(0)
                lib.check_load_bytes(bb, src_al.ctypes.data, nbytes, dst_al.ctypes.data)
                np.testing.assert_array_equal(dst_al, src_al)
                if bb % 2 == 0 and bb % 4 != 0:
                    src_mis = src_raw[off_src + 2:off_src + 2 + nbytes]
                    src_mis[:] = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
                    dst_al.fill(0)
                    lib.check_load_bytes(bb, src_mis.ctypes.data, nbytes, dst_al.ctypes.data)
                    np.testing.assert_array_equal(dst_al, src_mis)

    # Verify Python CUDA dispatch, padding for < 4 rows, fused gate_up+SwiGLU, and single-sync prefill plan
    captured = {}
    dispatch_log = []

    class FakeCudaKernel:
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
            t = dict(template)
            entry = {
                'inputs': inputs, 'template': t,
                'grid': grid, 'threadgroup': threadgroup,
                'output_shapes': output_shapes, 'output_dtypes': output_dtypes,
            }
            captured[self.tag] = entry
            dispatch_log.append(self.tag)
            if self.tag == 'qmv':
                xs, w, _grid, idx = inputs
                w_unp = iq_experts.decode(w.reshape(-1, w.shape[-1]), t['KIND']).reshape(
                    w.shape[0], t['N_SIZE'], t['K_SIZE']
                ).astype(t['T'])
                x_div = t['X_DIV']
                slots = xs.shape[0] * x_div
                rows_used = idx[:slots].astype(mx.uint32)
                x_rep = xs[:, None, :] if x_div == 1 else xs[mx.arange(slots, dtype=mx.int32) // x_div, None, :]
                ref = mx.gather_mm(x_rep, w_unp.swapaxes(-1, -2), rhs_indices=rows_used).reshape(-1)
                pad = output_shapes[0][0] - ref.size
                out = mx.pad(ref, (0, pad)) if pad > 0 else ref
                return [out]
            if self.tag == 'gate_up':
                xs, wg, _gg, wu, _gu, idx = inputs
                x_div = t['X_DIV']
                slots = xs.shape[0] * x_div
                rows_used = idx[:slots].astype(mx.uint32)
                x_rep = xs[:, None, :] if x_div == 1 else xs[mx.arange(slots, dtype=mx.int32) // x_div, None, :]
                wg_unp = iq_experts.decode(wg.reshape(-1, wg.shape[-1]), t['KIND_G']).reshape(
                    wg.shape[0], t['N_SIZE'], t['K_SIZE']
                ).astype(t['T'])
                wu_unp = iq_experts.decode(wu.reshape(-1, wu.shape[-1]), t['KIND_U']).reshape(
                    wu.shape[0], t['N_SIZE'], t['K_SIZE']
                ).astype(t['T'])
                gv = mx.gather_mm(x_rep, wg_unp.swapaxes(-1, -2), rhs_indices=rows_used).reshape(slots, t['N_SIZE'])
                uv = mx.gather_mm(x_rep, wu_unp.swapaxes(-1, -2), rhs_indices=rows_used).reshape(slots, t['N_SIZE'])
                ref = (nn.silu(gv) * uv).reshape(-1)
                pad = output_shapes[0][0] - ref.size
                out = mx.pad(ref, (0, pad)) if pad > 0 else ref
                return [out]
            xs, w, _grid, tiles = inputs
            w_np = np.array(
                iq_experts.decode(w.reshape(-1, w.shape[-1]), t['KIND']).reshape(w.shape[0], t['N_SIZE'], t['K_SIZE']),
                dtype=np.float32,
            )
            xs_np = np.array(xs.astype(mx.float32), dtype=np.float32)
            raw_tiles = np.array(tiles).reshape(-1)
            tiles_np = raw_tiles[: (raw_tiles.size // 3) * 3].reshape(-1, 3)
            pieces = []
            for e_id, r0, cnt in tiles_np:
                if cnt <= 0:
                    continue
                pieces.append(xs_np[int(r0):int(r0 + cnt)] @ w_np[int(e_id)].T)
            ref = mx.array(np.concatenate(pieces, axis=0).reshape(-1), dtype=t['T'])
            pad = output_shapes[0][0] - ref.size
            out = mx.pad(ref, (0, pad)) if pad > 0 else ref
            return [out]

    monkeypatch.setattr(gguf_qmv.mx.metal, 'is_available', lambda: False)
    monkeypatch.setattr(gguf_qmv, 'mx', gguf_qmv.mx)
    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
    monkeypatch.setattr(gguf_qmv.mx, 'default_device', lambda: mx.gpu)
    monkeypatch.setattr(gguf_qmv, '_cuda_kernel', lambda: FakeCudaKernel('qmv'))
    monkeypatch.setattr(gguf_qmv, '_cuda_gate_up', lambda: FakeCudaKernel('gate_up'))
    monkeypatch.setattr(gguf_qmv, '_cuda_tiled', lambda: FakeCudaKernel('mm'))
    monkeypatch.setattr(gguf_qmv, 'CHOICE', {})
    monkeypatch.setattr(gguf_qmv, 'CHECKED', {})
    monkeypatch.setattr(gguf_qmv, 'STATS', dict(served=0, checked=0, mismatched=0))

    rng = np.random.default_rng(909)
    raw = _random_ggml_blocks(rng, iq_experts.Q4_K, 4 * 20 * 1).reshape(4, 20, 1, iq_experts.BLOCK[iq_experts.Q4_K])
    proj = nn.Module()
    proj.weight = mx.array(raw)
    proj.mode = 'gguf:q4_k'
    proj.group_size = 256
    proj.bits = 4
    mx.eval(proj.parameters())

    w_dec = iq_experts.decode(proj.weight.reshape(-1, proj.weight.shape[-1]), iq_experts.Q4_K).reshape(4, 20, 256)

    # Single-row decode (`rows == 1`) pads `idx` to 4 int32s (16 bytes) and sets CUDA WARPS + X_DIV
    x_one = mx.array(rng.standard_normal((1, 1, 256)).astype(np.float32), dtype=mx.bfloat16)
    idx_one = mx.array([[2]], dtype=mx.uint32)
    assert gguf_qmv.takes(proj, x_one, idx_one)
    y_one = gguf_qmv.run(proj, x_one, idx_one)
    want_one = mx.gather_mm(x_one, w_dec.astype(mx.bfloat16).swapaxes(-1, -2), rhs_indices=idx_one)
    mx.eval(y_one, want_one)
    assert y_one.shape == want_one.shape == (1, 1, 1, 20)
    _assert_gguf_close(y_one, want_one, rel=1e-2)
    assert captured['qmv']['inputs'][3].shape == (4,)
    assert captured['qmv']['template']['WARPS'] == gguf_qmv.WARPS
    assert captured['qmv']['template']['X_DIV'] == 1
    assert captured['qmv']['threadgroup'] == (32 * gguf_qmv.WARPS, 1, 1)
    assert captured['qmv']['grid'] == (32 * gguf_qmv.WARPS * 2, 1, 1)

    # Single-tile prefill (`len(tiles) == 1`) pads `flat_tiles` from 3 to 4 int32s (16 bytes)
    x_t1 = mx.array(rng.standard_normal((2, 1, 256)).astype(np.float32), dtype=mx.float32)
    idx_t1 = mx.array([2, 2], dtype=mx.uint32)
    y_t1 = gguf_qmv.run_rows(proj, x_t1, idx_t1)
    mx.eval(y_t1)
    assert captured['mm']['inputs'][3].shape == (4,)

    # Unsorted prefill rows restore original ordering after sorted tiled dispatch
    x_pre = mx.array(rng.standard_normal((10, 1, 256)).astype(np.float32), dtype=mx.float32)
    idx_pre = mx.array([3, 0, 2, 1, 3, 2, 0, 1, 2, 3], dtype=mx.uint32)
    assert gguf_qmv.takes_rows(proj, x_pre, idx_pre)
    y_pre = gguf_qmv.run_rows(proj, x_pre, idx_pre)
    want_pre = mx.gather_mm(x_pre, w_dec.swapaxes(-1, -2), rhs_indices=idx_pre)
    mx.eval(y_pre, want_pre)
    want_fp32 = np.einsum(
        'mk,mnk->mn',
        np.array(x_pre[:, 0], dtype=np.float32),
        np.array(w_dec, dtype=np.float32)[np.array(idx_pre)],
    )[:, None, :]
    assert captured['mm']['threadgroup'] == (128, 1, 1)
    np.testing.assert_allclose(np.array(y_pre), want_fp32, rtol=1e-5, atol=1e-4)
    assert float(mx.abs(y_pre - want_pre).max().item()) <= (
        gguf_qmv.TOLERANCE * max(float(mx.abs(want_pre).max().item()), 1e-30)
    )

    # Fused GGUF SwitchGLU decode (`expert_qmv.decode` -> `gguf_qmv.decode_switch`) on mixed types (Q5_K/Q4_K/Q6_K)
    from mlx_vlm.models.qwen3_next.language import SwitchGLU

    sw_mix = SwitchGLU(input_dims=256, hidden_dims=256, num_experts=8, bias=False)
    for p_name, k_kind, bits in (
        ('gate_proj', iq_experts.Q5_K, 5),
        ('up_proj', iq_experts.Q4_K, 4),
        ('down_proj', iq_experts.Q6_K, 6),
    ):
        pr = nn.Module()
        pr.weight = mx.array(
            _random_ggml_blocks(rng, k_kind, 8 * 256).reshape(8, 256, 1, iq_experts.BLOCK[k_kind])
        )
        pr.mode = f'gguf:{iq_experts.TYPES[k_kind].lower()}'
        pr.group_size = 256
        pr.bits = bits
        setattr(sw_mix, p_name, pr)
    mx.eval(sw_mix.parameters())

    for b_sz in (1, 4):
        x_dec = mx.array(rng.standard_normal((b_sz, 1, 256)).astype(np.float32), dtype=mx.bfloat16)
        idx_dec = mx.array(((np.arange(b_sz * 4, dtype=np.uint32) * 3 + 1) % 8).reshape(b_sz, 1, 4))
        mx.eval(x_dec, idx_dec)
        want_dec = expert_qmv.stock(sw_mix, x_dec, idx_dec)
        mx.eval(want_dec)
        prev_served = gguf_qmv.STATS['served']
        for _ in range(gguf_qmv.CHECKS + 2):
            dispatch_log.clear()
            got_dec = expert_qmv.decode(sw_mix, x_dec, idx_dec)
            mx.eval(got_dec)
            assert float(mx.abs(got_dec.astype(mx.float32) - want_dec.astype(mx.float32)).max().item()) <= (
                gguf_qmv.TOLERANCE * max(float(mx.abs(want_dec.astype(mx.float32)).max().item()), 1e-30)
            )
        assert dispatch_log == ['gate_up', 'qmv']
        assert captured['gate_up']['template']['X_DIV'] == 4
        assert captured['gate_up']['inputs'][0].shape == (b_sz, 256)
        assert captured['qmv']['template']['X_DIV'] == 1
        assert gguf_qmv.STATS['served'] >= prev_served + 2
        assert gguf_qmv.STATS['mismatched'] == 0

    # GGUF prefill `expert_qmv.stock` with M = 128 routes to `_cuda_tiled` and builds `prepare_rows` once per layer
    orig_prepare = gguf_qmv.prepare_rows
    prepare_calls = []
    monkeypatch.setattr(gguf_qmv, 'prepare_rows', lambda ind: (prepare_calls.append(1), orig_prepare(ind))[1])
    for sorted_flag in (False, True):
        x_pre16 = mx.array(rng.standard_normal((1, 16, 256)).astype(np.float32), dtype=mx.bfloat16)
        if sorted_flag:
            idx_np16 = np.sort((np.arange(128, dtype=np.uint32) * 5) % 8).reshape(1, 16, 8)
        else:
            idx_np16 = ((np.arange(128, dtype=np.uint32) * 5 + 3) % 8).reshape(1, 16, 8)
        idx_pre16 = mx.array(idx_np16, dtype=mx.uint32)
        mx.eval(x_pre16, idx_pre16)
        for run_i in range(gguf_qmv.CHECKS + 1):
            prepare_calls.clear()
            dispatch_log.clear()
            y_stock = expert_qmv.stock(sw_mix, x_pre16, idx_pre16)
            mx.eval(y_stock)
            if run_i == gguf_qmv.CHECKS:
                assert len(prepare_calls) == 1
                assert dispatch_log == ['mm', 'mm', 'mm']
        x_u = mx.expand_dims(x_pre16, (-2, -3))

        def _ref_mm(proj_m, inp):
            w_u = iq_experts.decode(proj_m.weight.reshape(-1, proj_m.weight.shape[-1]), gguf_qmv.kind_of(proj_m)).reshape(
                proj_m.weight.shape[0], proj_m.weight.shape[1], 256
            ).astype(inp.dtype)
            return mx.gather_mm(inp, w_u.swapaxes(-1, -2), rhs_indices=idx_pre16)

        ref_hid = sw_mix.activation(_ref_mm(sw_mix.up_proj, x_u), _ref_mm(sw_mix.gate_proj, x_u))
        ref_y = _ref_mm(sw_mix.down_proj, ref_hid).squeeze(-2)
        mx.eval(ref_y)
        assert float(mx.abs(y_stock.astype(mx.float32) - ref_y.astype(mx.float32)).max().item()) <= (
            gguf_qmv.TOLERANCE * max(float(mx.abs(ref_y.astype(mx.float32)).max().item()), 1e-30)
        )
    assert gguf_qmv.STATS['mismatched'] == 0


def test_gguf_experts_host_copy_and_mixed_precision_prefill(tmp_path, monkeypatch):
    """`ExpertGGUF` integrates with `q2_experts.HostCopy` (`by_rows=False` and `True`), `serve._hold_experts`,
    and `Demand` mixed-precision chunked prefill + lazy ring-buffer decode staging."""
    import mlx.nn as nn
    from quanteeze import demand, gguf_experts, iq_experts, prefill, q2_experts, serve

    class DummyCache:
        def __init__(self):
            self.offset = 0
            self.max_size = None

    class SplitMoELayer(nn.Module):
        def __init__(self, sw):
            super().__init__()
            mlp = nn.Module()
            mlp.switch_mlp = sw
            mlp.top_k = 4
            self.mlp = mlp

        def _pre(self, x, idx, cache=None):
            if cache is not None:
                cache.offset += x.shape[1]
            return x, idx

        def _post(self, x, idx):
            return self.mlp.switch_mlp(x, idx).sum(axis=-2)

        def __call__(self, x, idx, cache=None):
            h, i = self._pre(x, idx, cache=cache)
            return self._post(h, i)

    class ToyLM(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

        def make_cache(self):
            return [DummyCache() for _ in self.layers]

    prefill.split(SplitMoELayer, SplitMoELayer._pre, SplitMoELayer._post)

    rng = np.random.default_rng(1707)
    n_exp, d_in, d_hid = 16, 256, 256
    layer_specs = [
        ('stacked', {'gate_proj': iq_experts.IQ2_XXS, 'up_proj': iq_experts.IQ1_M, 'down_proj': iq_experts.Q4_K}),
        ('per_expert', {'gate_proj': iq_experts.IQ2_XS, 'up_proj': iq_experts.Q3_K, 'down_proj': iq_experts.IQ4_NL}),
    ]
    gguf_name = {'gate_proj': 'ffn_gate', 'up_proj': 'ffn_up', 'down_proj': 'ffn_down'}

    raw_arrays = {}
    shard_tensors = [[], []]
    for l_idx, (layout, projs) in enumerate(layer_specs):
        for p, kind in projs.items():
            out_d, in_d = (d_in, d_hid) if p == 'down_proj' else (d_hid, d_in)
            kb = in_d // iq_experts.per_block(kind)
            arr = _random_ggml_blocks(rng, kind, n_exp * out_d * kb).reshape(n_exp, out_d, kb, iq_experts.BLOCK[kind])
            raw_arrays[(l_idx, p)] = (arr, kind)
            if layout == 'stacked':
                shard_tensors[l_idx].append(
                    (f'blk.{l_idx}.{gguf_name[p]}_exps.weight', (in_d, out_d, n_exp), kind, arr.tobytes())
                )
            else:
                for e in range(n_exp):
                    shard_tensors[l_idx].append(
                        (f'blk.{l_idx}.{gguf_name[p]}.{e}.weight', (in_d, out_d), kind, arr[e].tobytes())
                    )

    gdir = tmp_path / 'ud_shards'
    gdir.mkdir()
    shard1 = gdir / 'ud-00001-of-00002.gguf'
    shard2 = gdir / 'ud-00002-of-00002.gguf'
    _write_gguf(shard1, shard_tensors[0])
    _write_gguf(shard2, shard_tensors[1])

    src = gguf_experts.ExpertGGUF(shard1)
    assert src.routed() == [
        'blk.0.ffn_gate_exps.weight', 'blk.0.ffn_up_exps.weight', 'blk.0.ffn_down_exps.weight',
        'blk.1.ffn_gate_exps.weight', 'blk.1.ffn_up_exps.weight', 'blk.1.ffn_down_exps.weight',
    ]

    # 1) Whole-tensor HostCopy (`by_rows=False`) invoked via shard file path
    layer0_bytes = sum(src.nbytes(src.name(0, p, 'weight')) for p in ('gate_proj', 'up_proj', 'down_proj'))
    try:
        whole = q2_experts.hold(shard1, budget=layer0_bytes, workers=1, by_rows=False)
        whole.wait()
        assert not whole.by_rows
        assert np.array_equal(whole.rows('blk.0.ffn_up_exps.weight', [3, 9]), raw_arrays[(0, 'up_proj')][0][[3, 9]])
        assert whole.rows('blk.1.ffn_gate_exps.weight') is None
    finally:
        q2_experts.let_go(shard1)

    # 2) Row-wise HostCopy (`by_rows=True`) through `serve._hold_experts` + `Demand` prefill & decode
    def make_model():
        layers = []
        for _ in range(2):
            sw = SwitchGLU(input_dims=d_in, hidden_dims=d_hid, num_experts=n_exp, bias=False)
            layers.append(SplitMoELayer(sw))
        return ToyLM(layers)

    m_mem = make_model()
    gguf_experts.load_model(m_mem.layers, src)
    m_ssd = make_model()

    monkeypatch.setenv('QUANTEEZE_HOST_EXPERTS_GB', '1')
    monkeypatch.setenv('QUANTEEZE_HOST_EXPERTS_BY', 'rows')
    held = serve._hold_experts(src)
    assert held is not None
    dem = None
    prefill.JOINS.clear()
    prefill.times(reset=True)
    monkeypatch.setenv('QUANTEEZE_PROMPT_MOE_TOKENS', '32')
    monkeypatch.setattr(demand.mx.metal, 'is_available', lambda: False)
    try:
        held.wait()
        assert held.by_rows
        assert held.holds_layer(0) and held.holds_layer(1)

        k0 = src.name(0, 'up_proj', 'weight')
        sub_buf = np.empty((3,) + raw_arrays[(0, 'up_proj')][0].shape[1:], dtype=np.uint8)
        assert held.copy_layer(0, [2, 7, 13], [(k0, sub_buf)]) is True
        assert np.array_equal(sub_buf, raw_arrays[(0, 'up_proj')][0][[2, 7, 13]])

        k1 = src.name(1, 'gate_proj', 'weight')
        full_buf = np.empty_like(raw_arrays[(1, 'gate_proj')][0])
        assert held.copy_layer(1, None, [(k1, full_buf)]) is True
        assert np.array_equal(full_buf, raw_arrays[(1, 'gate_proj')][0])

        dem = demand.Demand(m_ssd, src, budget_bytes=1, workers=2)
        assert dem.slots == 8
        assert len(dem._stage_ring) == demand._STAGE_RING
        prefill.install(m_ssd.layers, chunk=16, enabled=lambda: True)

        # Prompt prefill (> 1 chunk) reuses flat `(p, part)` staging/device buffers across mixed-precision layers
        prompt_len = 48
        x_pre = mx.array(rng.standard_normal((1, prompt_len, d_in)).astype(np.float32), dtype=mx.bfloat16)
        idx_pre = mx.array((np.arange(prompt_len * 4, dtype=np.uint32).reshape(1, prompt_len, 4)) % n_exp)
        mx.eval(x_pre, idx_pre)

        seen_dev_keys = []
        seen_stage_keys = []
        orig_load_pre = dem._load_prefill_projs

        def _spy_load_pre(layer, switch, expert_ids, cap_rows):
            res = orig_load_pre(layer, switch, expert_ids, cap_rows)
            seen_stage_keys.append(set(dem._prefill_stage.keys()))
            seen_dev_keys.append(set(dem._prefill_dev.keys()))
            assert all(v[0].dtype == mx.uint32 for v in dem._prefill_stage.values())
            assert all(v.dtype == mx.uint32 for v in dem._prefill_dev.values())
            return res

        dem._load_prefill_projs = _spy_load_pre

        c_mem, c_ssd = m_mem.make_cache(), m_ssd.make_cache()
        h_mem, h_ssd = x_pre, x_pre
        for l_idx in range(2):
            h_mem = m_mem.layers[l_idx](h_mem, idx_pre, cache=c_mem[l_idx])
            h_ssd = m_ssd.layers[l_idx](h_ssd, idx_pre, cache=c_ssd[l_idx])
        mx.eval(h_mem, h_ssd)
        _assert_gguf_close(h_ssd, h_mem, rel=2e-2)
        expected_slot_keys = {('gate_proj', 'weight'), ('up_proj', 'weight'), ('down_proj', 'weight')}
        assert len(seen_stage_keys) == 2 and all(k == expected_slot_keys for k in seen_stage_keys)
        assert len(seen_dev_keys) == 2 and all(k == expected_slot_keys for k in seen_dev_keys)
        assert dem.prefill_layers == 2
        assert len(src._ahead) == 0
        assert len(dem._prefill_stage) == 0
        assert len(dem._prefill_dev) == 0

        # Lazy pipelined decode steps (> 2 * _STAGE_RING) exercise host-cached GGUF staging & ring wrap-around
        lazy_pairs = []
        for step in range(2 * demand._STAGE_RING + 2):
            base = (step * 4) % n_exp
            exps = [(base + offset) % n_exp for offset in range(4)]
            x_dec = mx.array(rng.standard_normal((1, 1, d_in)).astype(np.float32), dtype=mx.bfloat16)
            idx_dec = mx.array([[exps]], dtype=mx.uint32)
            mx.eval(x_dec, idx_dec)
            for l_idx in range(2):
                lazy_pairs.append((
                    m_mem.layers[l_idx].mlp.switch_mlp(x_dec, idx_dec),
                    m_ssd.layers[l_idx].mlp.switch_mlp(x_dec, idx_dec),
                ))
        for y_mem, y_ssd in lazy_pairs:
            mx.eval(y_mem, y_ssd)
            _assert_gguf_close(y_ssd, y_mem, rel=2e-2)
        assert all(set(ring_map.keys()) <= expected_slot_keys for ring_map in dem._stage_ring)
        assert all(v[0].dtype == mx.uint32 for ring_map in dem._stage_ring for v in ring_map.values())
        assert dem.misses > 0
        assert dem.host_served == dem.misses

        # Direct in-place donation & byte-exactness check on `DemandSwitchGLU.place_many` with Q6_K uint8 bank
        bank_np = rng.integers(0, 256, size=(16, 64, 2, 210), dtype=np.uint8)
        upd_np = rng.integers(0, 256, size=(3, 64, 2, 210), dtype=np.uint8)
        sw_bank = demand.DemandSwitchGLU(
            0,
            demand.Projection(mx.array(bank_np), None, None, 256, 6, mode='gguf:q6_k'),
            demand.Projection(mx.array(bank_np), None, None, 256, 6, mode='gguf:q6_k'),
            demand.Projection(mx.array(bank_np), None, None, 256, 6, mode='gguf:q6_k'),
            np.arange(16, dtype=np.int64),
            32,
            nn.silu,
            None,
        )
        upd_mx = {
            p: [mx.array(upd_np), None, None]
            for p in ('gate_proj', 'up_proj', 'down_proj')
        }
        mx.eval(sw_bank.parameters(), [v[0] for v in upd_mx.values()])
        mem_before = mx.get_active_memory()
        written = sw_bank.place_many([1, 5, 9], [20, 21, 22], upd_mx)
        mx.eval(written)
        del written, upd_mx
        mem_after = mx.get_active_memory()
        assert mem_after <= mem_before + 3 * upd_np.nbytes
        expected_bank = bank_np.copy()
        expected_bank[[1, 5, 9]] = upd_np
        for p in ('gate_proj', 'up_proj', 'down_proj'):
            np.testing.assert_array_equal(np.array(getattr(sw_bank, p).weight), expected_bank)
    finally:
        prefill.release()
        if dem is not None:
            dem.close()
        q2_experts.let_go(src)


def test_demand_short_prefill_skips_resident_experts_and_parallel_host_copy(tmp_path):
    """Short uncached prefill skips re-loading GPU-resident experts, `_update_bank_after_prefill`
    reads missing promotions from `HostCopy` before SSD while keeping `host_served == misses`,
    and parallel `HostCopy.copy_layer` matches single-worker output on >= 512 KiB slices."""
    import mlx.nn as nn
    from quanteeze import demand, q2_experts

    rng = np.random.default_rng(23)
    host_dir = tmp_path / 'big'
    big_tensors = {}
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        big_tensors[f'layers.0.mlp.switch_mlp.{proj}.weight'] = mx.array(
            rng.integers(0, 1 << 31, size=(8, 256, 128), dtype=np.uint32)
        )
        big_tensors[f'layers.0.mlp.switch_mlp.{proj}.scales'] = mx.ones((8, 256, 4), dtype=mx.bfloat16)
        big_tensors[f'layers.0.mlp.switch_mlp.{proj}.biases'] = mx.zeros((8, 256, 4), dtype=mx.bfloat16)
    mx.eval(list(big_tensors.values()))
    host_dir.mkdir()
    mx.save_safetensors(str(host_dir / 'model.safetensors'), big_tensors)

    ckpt_big = q2_experts.Checkpoint(host_dir)
    hc4 = q2_experts.HostCopy(ckpt_big, budget=1 << 28, workers=4, by_rows=True)
    hc1 = q2_experts.HostCopy(ckpt_big, budget=1 << 28, workers=1, by_rows=True)
    try:
        hc4.wait()
        hc1.wait()
        targets4, targets1 = [], []
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            for part in ('weight', 'scales', 'biases'):
                name = ckpt_big.name(0, proj, part)
                _, st_dtype, full_shape, _ = ckpt_big.tensors[name]
                np_dt = np.dtype(q2_experts._SAFETENSORS[st_dtype])
                targets4.append((name, np.empty(full_shape, np_dt)))
                targets1.append((name, np.empty(full_shape, np_dt)))
        assert hc4.copy_layer(0, None, targets4) is True
        assert hc1.copy_layer(0, None, targets1) is True
        for (_, a4), (_, a1) in zip(targets4, targets1):
            np.testing.assert_array_equal(a4, a1)

        subset = np.array([1, 3, 5, 7], dtype=np.int64)
        sub4, sub1 = [], []
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            for part in ('weight', 'scales', 'biases'):
                name = ckpt_big.name(0, proj, part)
                _, st_dtype, full_shape, _ = ckpt_big.tensors[name]
                np_dt = np.dtype(q2_experts._SAFETENSORS[st_dtype])
                sub4.append((name, np.empty((subset.size,) + tuple(full_shape[1:]), np_dt)))
                sub1.append((name, np.empty((subset.size,) + tuple(full_shape[1:]), np_dt)))
        assert hc4.copy_layer(0, subset, sub4) is True
        assert hc1.copy_layer(0, subset, sub1) is True
        for (_, a4), (_, a1) in zip(sub4, sub1):
            np.testing.assert_array_equal(a4, a1)
    finally:
        hc4.close()
        hc1.close()

    model_dir = tmp_path / 'moe'
    model_dir.mkdir()
    mx.random.seed(29)
    sw_base = SwitchGLU(input_dims=64, hidden_dims=64, num_experts=16, bias=False)
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        setattr(sw_base, proj, getattr(sw_base, proj).to_quantized(group_size=64, bits=4))
    mx.eval(sw_base.parameters())
    moe_tensors = {}
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        p = getattr(sw_base, proj)
        moe_tensors[f'layers.0.mlp.switch_mlp.{proj}.weight'] = p.weight
        moe_tensors[f'layers.0.mlp.switch_mlp.{proj}.scales'] = p.scales
        moe_tensors[f'layers.0.mlp.switch_mlp.{proj}.biases'] = p.biases
    mx.save_safetensors(str(model_dir / 'model.safetensors'), moe_tensors)

    def make_model():
        sw = SwitchGLU(input_dims=64, hidden_dims=64, num_experts=16, bias=False)
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(sw, proj, getattr(sw, proj).to_quantized(group_size=64, bits=4))
        mlp = nn.Module()
        mlp.switch_mlp = sw
        layer = nn.Module()
        layer.mlp = mlp
        m = nn.Module()
        m.layers = [layer]
        return m

    held = q2_experts.hold(model_dir, budget=1 << 28, workers=4, by_rows=True)
    held.wait()
    d_short = None
    d_batched = None
    try:
        m_short, m_batched = make_model(), make_model()
        d_short = demand.Demand(m_short, q2_experts.Checkpoint(model_dir), budget_bytes=1, workers=2, prompt_room=None)
        d_batched = demand.Demand(m_batched, q2_experts.Checkpoint(model_dir), budget_bytes=1, workers=2, prompt_room=1)
        assert d_short.slots == 8 and d_batched.slots == 8

        recorded_loads = []
        orig_load = d_short._load_prefill_projs

        def spy_load(layer, switch, expert_ids, cap_rows):
            recorded_loads.append(np.asarray(expert_ids, dtype=np.int64).tolist())
            return orig_load(layer, switch, expert_ids, cap_rows)

        d_short._load_prefill_projs = spy_load

        x_short = mx.random.normal((1, 16, 64)).astype(mx.bfloat16)
        idx_short = mx.array((np.arange(16 * 4, dtype=np.uint32).reshape(1, 16, 4)) % 12)
        scores_short = mx.ones((1, 16, 4), dtype=mx.bfloat16) * 0.25
        mx.eval(x_short, idx_short, scores_short)

        y_ref_short = expert_qmv.stock(sw_base, x_short, idx_short)
        y_short = m_short.layers[0].mlp.switch_mlp(x_short, idx_short, scores_short)
        mx.eval(y_ref_short, y_short)

        assert recorded_loads == [[8, 9, 10, 11]]
        assert mx.array_equal(y_short, y_ref_short)

        copy_calls = []
        orig_copy = held.copy_layer

        def spy_copy(layer, experts, targets):
            copy_calls.append(None if experts is None else np.asarray(experts, dtype=np.int64).tolist())
            return orig_copy(layer, experts, targets)

        held.copy_layer = spy_copy

        x_all = mx.random.normal((1, 20, 64)).astype(mx.bfloat16)
        idx_all_np = (np.arange(20 * 4, dtype=np.uint32).reshape(1, 20, 4)) % 16
        scores_all_np = np.full((1, 20, 4), 0.1, dtype=np.float32)
        scores_all_np[idx_all_np == 8] = 5.0
        idx_all = mx.array(idx_all_np)
        scores_all = mx.array(scores_all_np).astype(mx.bfloat16)
        mx.eval(x_all, idx_all, scores_all)

        y_ref_all = expert_qmv.stock(sw_base, x_all, idx_all)
        y_batched = m_batched.layers[0].mlp.switch_mlp(x_all, idx_all, scores_all)
        mx.eval(y_ref_all, y_batched)

        assert mx.array_equal(y_batched, y_ref_all)
        assert [8] in copy_calls
        assert d_batched.host_served == d_batched.misses == 0
        assert 8 in d_batched.switches[0].experts.tolist()
    finally:
        if d_short is not None:
            d_short.close()
        if d_batched is not None:
            d_batched.close()
        q2_experts.let_go(model_dir)


def test_demand_prefill_cap_reclaims_mlx_cache_for_in_layer_mode(tmp_path, monkeypatch):
    """`_prefill_cap` and `end_layer_prefill` reclaim MLX's cached CUDA pool bytes so
    uniform 4-bit stays in `in_layer_mode` across all layers instead of reloading per chunk."""
    import mlx.nn as nn
    from quanteeze import cuda_ple, demand, prefill, q2_experts

    class DummyCache:
        def __init__(self):
            self.offset = 0
            self.max_size = None

    cache_state = {'bytes': 0, 'layer_bytes': 0}

    class CachedSplitLayer(nn.Module):
        def __init__(self, sw):
            super().__init__()
            mlp = nn.Module()
            mlp.switch_mlp = sw
            mlp.top_k = 4
            self.mlp = mlp

        def _pre(self, x, idx, cache=None):
            if cache is not None:
                if cache.offset == 0:
                    # Simulate earlier attention buffers lingering in MLX BufferCache on layer entry
                    cache_state['bytes'] = cache_state['layer_bytes']
                cache.offset += x.shape[1]
            return x, idx

        def _post(self, x, idx):
            return self.mlp.switch_mlp(x, idx).sum(axis=-2)

        def __call__(self, x, idx, cache=None):
            h, i = self._pre(x, idx, cache=cache)
            return self._post(h, i)

    class ToyLM(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

        def make_cache(self):
            return [DummyCache() for _ in self.layers]

    prefill.split(CachedSplitLayer, CachedSplitLayer._pre, CachedSplitLayer._post)

    mx.random.seed(41)
    n_exp, d_in, d_hid = 16, 64, 64
    tensors = {}
    ref_layers = []
    for l_idx in range(2):
        sw = SwitchGLU(input_dims=d_in, hidden_dims=d_hid, num_experts=n_exp, bias=False)
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            q = getattr(sw, proj).to_quantized(group_size=64, bits=4)
            setattr(sw, proj, q)
            tensors[f'layers.{l_idx}.mlp.switch_mlp.{proj}.weight'] = q.weight
            tensors[f'layers.{l_idx}.mlp.switch_mlp.{proj}.scales'] = q.scales
            tensors[f'layers.{l_idx}.mlp.switch_mlp.{proj}.biases'] = q.biases
        mx.eval(sw.parameters())
        ref_layers.append(CachedSplitLayer(sw))
    m_ref = ToyLM(ref_layers)

    model_dir = tmp_path / 'u4_moe'
    model_dir.mkdir()
    mx.save_safetensors(str(model_dir / 'model.safetensors'), tensors)

    m_ssd = ToyLM([
        CachedSplitLayer(SwitchGLU(input_dims=d_in, hidden_dims=d_hid, num_experts=n_exp, bias=False))
        for _ in range(2)
    ])
    for layer in m_ssd.layers:
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(
                layer.mlp.switch_mlp,
                proj,
                getattr(layer.mlp.switch_mlp, proj).to_quantized(group_size=64, bits=4),
            )

    dem = None
    clear_calls = []
    try:
        dem = demand.Demand(m_ssd, q2_experts.Checkpoint(model_dir), budget_bytes=1, workers=2, prompt_room=None)
        assert dem.slots == 8

        full_layer = dem._layer_expert_bytes(0) * n_exp
        reserve = max(int(dem._free_least or 0), 256 << 20)
        base_free = reserve + full_layer + (1 << 12)
        cache_state['layer_bytes'] = full_layer

        orig_clear = demand.mx.clear_cache

        def fake_clear_cache():
            clear_calls.append((len(dem._prefill_dev), cache_state['bytes']))
            cache_state['bytes'] = 0
            orig_clear()

        def fake_gpu_room():
            held_dev = sum(int(a.nbytes) for a in dem._prefill_dev.values())
            return (max(0, base_free - cache_state['bytes'] - held_dev), 0)

        monkeypatch.setattr(demand.mx.metal, 'is_available', lambda: False)
        monkeypatch.setattr(cuda_ple, 'gpu_room', fake_gpu_room)
        monkeypatch.setattr(demand.mx, 'get_cache_memory', lambda: cache_state['bytes'])
        monkeypatch.setattr(demand.mx, 'clear_cache', fake_clear_cache)

        recorded_loads = []
        orig_load_pre = dem._load_prefill_projs

        def spy_load_pre(layer, switch, expert_ids, cap_rows):
            recorded_loads.append((layer, int(cap_rows), -1 if expert_ids is None else len(expert_ids)))
            return orig_load_pre(layer, switch, expert_ids, cap_rows)

        dem._load_prefill_projs = spy_load_pre

        prefill.install(m_ssd.layers, chunk=16, enabled=lambda: True)
        prompt_len = 48
        x_pre = mx.random.normal((1, prompt_len, d_in)).astype(mx.bfloat16)
        idx_pre = mx.array((np.arange(prompt_len * 4, dtype=np.uint32).reshape(1, prompt_len, 4)) % n_exp)
        mx.eval(x_pre, idx_pre)

        c_ref, c_ssd = m_ref.make_cache(), m_ssd.make_cache()
        h_ref, h_ssd = x_pre, x_pre
        for l_idx in range(2):
            h_ref = m_ref.layers[l_idx](h_ref, idx_pre, cache=c_ref[l_idx])
            h_ssd = m_ssd.layers[l_idx](h_ssd, idx_pre, cache=c_ssd[l_idx])
        mx.eval(h_ref, h_ssd)

        assert mx.array_equal(h_ssd, h_ref)
        assert dem.prefill_layers == 2
        assert recorded_loads == [(0, n_exp, n_exp), (1, n_exp, n_exp)]
        assert any(dev_len > 0 and prev_cache == 0 for dev_len, prev_cache in clear_calls)
        assert any(dev_len > 0 and prev_cache == full_layer for dev_len, prev_cache in clear_calls)
        assert len(dem._prefill_stage) == 0
        assert len(dem._prefill_dev) == 0
    finally:
        prefill.release()
        if dem is not None:
            dem.close()
        q2_experts.let_go(model_dir)


def test_host_copy_layer_bypasses_pool_while_background_reading_active(tmp_path):
    """`HostCopy.copy_layer` copies directly on the caller thread while background SSD reads
    occupy `HostCopy._pool`, preventing head-of-line stalls right after server startup, and
    resumes parallel `_pool` copying once `reading()` completes."""
    import threading
    import time
    from quanteeze import q2_experts

    rng = np.random.default_rng(73)
    tensors = {}
    for l_idx in range(2):
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            tensors[f'layers.{l_idx}.mlp.switch_mlp.{proj}.weight'] = mx.array(
                rng.integers(0, 1 << 31, size=(8, 256, 128), dtype=np.uint32)
            )
            tensors[f'layers.{l_idx}.mlp.switch_mlp.{proj}.scales'] = mx.ones((8, 256, 4), dtype=mx.bfloat16)
            tensors[f'layers.{l_idx}.mlp.switch_mlp.{proj}.biases'] = mx.zeros((8, 256, 4), dtype=mx.bfloat16)
    mx.eval(list(tensors.values()))
    ckpt_dir = tmp_path / 'hc_stall'
    ckpt_dir.mkdir()
    mx.save_safetensors(str(ckpt_dir / 'model.safetensors'), tensors)

    ckpt = q2_experts.Checkpoint(ckpt_dir)
    gate = threading.Event()
    orig_from_file = ckpt._from_file

    def gated_from_file(name, rows=None, cache=False, out=None):
        if name.startswith('layers.1.'):
            gate.wait(timeout=5.0)
        return orig_from_file(name, rows, cache, out)

    ckpt._from_file = gated_from_file
    hc = q2_experts.HostCopy(ckpt, budget=1 << 28, workers=2, by_rows=True)
    try:
        deadline = time.monotonic() + 2.0
        while not hc.holds_layer(0) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert hc.holds_layer(0)
        assert hc.reading()

        layer0_names = [
            ckpt.name(0, proj, part)
            for proj in ('gate_proj', 'up_proj', 'down_proj')
            for part in ('weight', 'scales', 'biases')
        ]
        ref_np = {
            name: np.asarray(tensors[name].view(mx.uint16) if tensors[name].dtype == mx.bfloat16 else tensors[name])
            for name in layer0_names
        }

        def _alloc_targets(n_rows):
            out = []
            for name in layer0_names:
                _, st_dtype, full_shape, _ = ckpt.tensors[name]
                np_dt = np.dtype(q2_experts._SAFETENSORS[st_dtype])
                out.append((name, np.empty((n_rows,) + tuple(full_shape[1:]), np_dt)))
            return out

        submitted = []
        orig_submit = hc._pool.submit

        def spy_submit(fn, *args, **kwargs):
            submitted.append(fn)
            return orig_submit(fn, *args, **kwargs)

        hc._pool.submit = spy_submit

        subset = np.array([1, 3, 5, 7], dtype=np.int64)
        targets = _alloc_targets(8)
        sub_targets = _alloc_targets(subset.size)
        t0 = time.monotonic()
        assert hc.copy_layer(0, None, targets) is True
        assert hc.copy_layer(0, subset, sub_targets) is True
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0
        assert submitted == []
        for (name, full_arr), (_, sub_arr) in zip(targets, sub_targets):
            np.testing.assert_array_equal(full_arr, ref_np[name])
            np.testing.assert_array_equal(sub_arr, ref_np[name][subset])

        gate.set()
        hc.wait()
        assert not hc.reading()
        assert hc._reading_done is True

        targets_after = _alloc_targets(8)
        assert hc.copy_layer(0, None, targets_after) is True
        assert len(submitted) >= 2
        for (_, before_arr), (_, after_arr) in zip(targets, targets_after):
            np.testing.assert_array_equal(after_arr, before_arr)
    finally:
        gate.set()
        hc.close()


def test_admission_short_lengths_wanted_and_on_demand_coalesce(monkeypatch):
    """Per-row `lengths`/`wanted` budgeting admits all short concurrent prompts (`rows == 4`),
    `admission.install` forwards `(lengths, wanted)` from `_unprocessed_sequences`, and
    `serve._coalesce_on_demand` enables `ON_DEMAND_COALESCE_S` only when idle, batched, and serving on demand."""
    from quanteeze import admission, serve

    adm = admission.Admission(
        limit=16 * 1024**3,
        fixed=600 * 1024**2,
        batch_fixed=450 * 1024**2,
        most=8,
        ahead=4096,
    )
    adm.per_token = float(2 * 1024**2)
    adm.kv_token = float(512 * 1024)
    held = int(13.8 * 1024**3)
    assert adm.rows(4, 4, 512, False, held=held, lengths=[48, 52, 44, 50], wanted=[256, 256, 256, 256]) == 4
    assert adm.rows(4, 4, 512, False, held=held) == 1

    seen_calls = []
    orig_rows = adm.rows

    def spy_rows(pending, free_rows, step, busy, held=None, remaining=(), lengths=None, wanted=None):
        seen_calls.append({'lengths': lengths, 'wanted': wanted})
        return orig_rows(pending, free_rows, step, busy, held=held, remaining=remaining,
                         lengths=lengths, wanted=wanted)

    adm.rows = spy_rows

    class DummyGen:
        def __init__(self):
            self.prefill_batch_size = 4
            self.completion_batch_size = 4
            self.prefill_step_size = 512
            self._unprocessed_sequences = [(1, [0] * 40, 128), (2, [0] * 50, 256)]
            self._generation_batch = []
            self._prompt_batch = None

        def _next(self, **kwargs):
            return 'ok'

    admission.install(DummyGen, adm)
    assert DummyGen()._next() == 'ok'
    assert seen_calls == [{'lengths': [40, 50], 'wanted': [128, 256]}]

    class DummyRespGen:
        def _collect_pending_requests(self, *, active, capacity=None, idle_timeout=0.1, coalesce_s=0.0):
            return coalesce_s

    serve._coalesce_on_demand(DummyRespGen)
    rg = DummyRespGen()

    monkeypatch.setitem(serve._SERVING, 'demand', object())
    assert rg._collect_pending_requests(active=False, capacity=4, coalesce_s=0.0) == serve.ON_DEMAND_COALESCE_S
    assert rg._collect_pending_requests(active=True, capacity=4, coalesce_s=0.0) == 0.0
    assert rg._collect_pending_requests(active=False, capacity=1, coalesce_s=0.0) == 0.0

    monkeypatch.setitem(serve._SERVING, 'demand', None)
    assert rg._collect_pending_requests(active=False, capacity=4, coalesce_s=0.0) == 0.0


def test_multirow_decode_dispatch_and_qwen3_5_moe_left_padded_prefill(tmp_path, monkeypatch):
    """Verify multi-row CUDA dispatch predicates (`expert_qmv`, `cuda_router`, `cuda_experts`) and
    end-to-end left-padded `Qwen3_5MoeModel` layer-major batch prefill + decode parity."""
    from mlx.utils import tree_flatten, tree_map_with_path
    from mlx_vlm.generate.ar import _make_cache
    from mlx_vlm.models.qwen3_5_moe.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.qwen3_5_moe.language import LanguageModel
    from quanteeze import arch, cuda_experts, cuda_kernels, cuda_router, demand, q2_experts

    mx.random.seed(31)
    sw_cuda = SwitchGLU(input_dims=128, hidden_dims=128, num_experts=16, bias=False)
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        p = getattr(sw_cuda, proj)
        p.weight = p.weight.astype(mx.bfloat16)
        setattr(sw_cuda, proj, p.to_quantized(group_size=64, bits=4))
    mx.eval(sw_cuda.parameters())
    expert_qmv.install_switch(sw_cuda)

    with monkeypatch.context() as mp:
        decode_shapes = []

        def fake_decode(switch, x, indices, weights=None):
            decode_shapes.append(x.shape)
            return mx.zeros((x.shape[0], x.shape[1], indices.shape[-1], x.shape[-1]), dtype=x.dtype)

        mp.setattr(mx.metal, 'is_available', lambda: False)
        mp.setattr(expert_qmv, 'decode', fake_decode)
        sw_cuda(mx.zeros((4, 1, 128), dtype=mx.bfloat16), mx.zeros((4, 1, 4), dtype=mx.uint32))
        assert decode_shapes == [(4, 1, 128)]

        mp.setattr(cuda_kernels, 'available', lambda: True)
        assert cuda_router.takes(mx.zeros((4, 1, 16), dtype=mx.bfloat16), 4)
        assert not cuda_router.takes(mx.zeros((65, 1, 16), dtype=mx.bfloat16), 4)
        assert cuda_experts.takes(sw_cuda, mx.zeros((4, 1, 128), dtype=mx.bfloat16), mx.zeros((4, 1, 4), dtype=mx.uint32))
        assert not cuda_experts.takes(
            sw_cuda, mx.zeros((9, 1, 128), dtype=mx.bfloat16), mx.zeros((9, 1, 4), dtype=mx.uint32)
        )

    cfg = TextConfig(
        model_type='qwen3_5_moe',
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=128,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        full_attention_interval=2,
        norm_topk_prob=True,
        rope_parameters={
            'type': 'default',
            'mrope_section': [3, 3, 2],
            'rope_theta': 10000.0,
            'partial_rotary_factor': 0.25,
        },
    )
    model_cfg = ModelConfig(text_config=cfg, vision_config=VisionConfig(), model_type='qwen3_5_moe')

    def build_model():
        m = LanguageModel(cfg, config=model_cfg)
        casted = tree_map_with_path(
            lambda path, val: (val.astype(mx.bfloat16) if (val.dtype == mx.float32 and 'A_log' not in path) else val),
            m.parameters(),
        )
        m.update(casted)
        for layer in m.model.layers:
            sw = layer.mlp.switch_mlp
            for proj in ('gate_proj', 'up_proj', 'down_proj'):
                setattr(sw, proj, getattr(sw, proj).to_quantized(group_size=64, bits=4))
        return m

    mx.random.seed(37)
    m_ref = build_model()
    mx.eval(m_ref.parameters())

    m_fast = build_model()
    m_fast.load_weights(tree_flatten(m_ref.parameters()), strict=True)
    mx.eval(m_fast.parameters())

    ckpt_tensors = {}
    for layer_idx, layer in enumerate(m_ref.model.layers):
        sw = layer.mlp.switch_mlp
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            p = getattr(sw, proj)
            ckpt_tensors[f'layers.{layer_idx}.mlp.switch_mlp.{proj}.weight'] = p.weight
            ckpt_tensors[f'layers.{layer_idx}.mlp.switch_mlp.{proj}.scales'] = p.scales
            ckpt_tensors[f'layers.{layer_idx}.mlp.switch_mlp.{proj}.biases'] = p.biases
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), ckpt_tensors)

    arch.of(m_fast).install(m_fast)
    d_fast = demand.Demand(m_fast, q2_experts.Checkpoint(tmp_path), budget_bytes=1, workers=2)
    try:
        left_padding = [4, 0, 2]
        inputs = mx.array([
            [0, 0, 0, 0] + list(range(1, 17)),
            list(range(20, 40)),
            [0, 0] + list(range(50, 68)),
        ], dtype=mx.int32)
        cache_ref = _make_cache(m_ref, left_padding)
        cache_fast = _make_cache(m_fast, left_padding)

        out_ref = m_ref(inputs, cache=cache_ref)
        out_fast = m_fast(inputs, cache=cache_fast)
        logits_ref = out_ref.logits if hasattr(out_ref, 'logits') else out_ref
        logits_fast = out_fast.logits if hasattr(out_fast, 'logits') else out_fast
        mx.eval(logits_ref, logits_fast)

        assert d_fast.prefill_layers == 4
        assert mx.allclose(
            logits_fast[:, -1, :].astype(mx.float32),
            logits_ref[:, -1, :].astype(mx.float32),
            atol=5e-2, rtol=2e-2,
        )

        next_tok = mx.array([[8], [17], [25]], dtype=mx.int32)
        dec_ref = m_ref(next_tok, cache=cache_ref)
        dec_fast = m_fast(next_tok, cache=cache_fast)
        dec_logits_ref = dec_ref.logits if hasattr(dec_ref, 'logits') else dec_ref
        dec_logits_fast = dec_fast.logits if hasattr(dec_fast, 'logits') else dec_fast
        mx.eval(dec_logits_ref, dec_logits_fast)

        assert mx.allclose(
            dec_logits_fast.astype(mx.float32),
            dec_logits_ref.astype(mx.float32),
            atol=5e-2, rtol=2e-2,
        )
    finally:
        d_fast.close()


def test_mxfp4_uniform_quant_gguf_and_cuda_experts(tmp_path, monkeypatch):
    """Verify full MXFP4 support: C++17 `cuda_experts.MXFP4_DECODE_HEADER` bit-for-bit bfloat16 parity
    against `mx.dequantize`, CUDA `cuda_experts` (unstaged & staged) dispatch + `cuda_moe` opt-out,
    `QUANTEEZE_BITS=mxfp4` (`lower_model`, `Flat`, `QuantCache`/`CachedCheckpoint`, `Demand`), and
    GGUF `MXFP4` (kind 39) in-memory and `HostCopy` + `Demand` execution."""
    import ctypes
    import shutil
    import subprocess
    import mlx.nn as nn
    from quanteeze import control, cuda_experts, cuda_kernels, cuda_moe, demand, fitting, gguf_experts, home, iq_experts, prefill, q2_experts, requant, serve

    # 1. C++17 bit-for-bit BF16 parity of `qz_e8m0_bf162` & `qz_deq_mxfp4` vs `mx.dequantize`
    cxx = shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')
    if cxx is not None:
        cpp = f"""
#include <cstdint>
#include <cstring>

#define __device__
#define __forceinline__ inline

struct alignas(4) __nv_bfloat162 {{
  uint16_t lo, hi;
}};

inline float stub_bf16_to_f32(uint16_t b) {{
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}}

inline uint16_t stub_f32_to_bf16_rn(float f) {{
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x0040u);
  uint32_t bias = 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + bias) >> 16);
}}

inline __nv_bfloat162 __hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {{
  return {{
      stub_f32_to_bf16_rn(stub_bf16_to_f32(a.lo) * stub_bf16_to_f32(b.lo)),
      stub_f32_to_bf16_rn(stub_bf16_to_f32(a.hi) * stub_bf16_to_f32(b.hi)),
  }};
}}

{cuda_experts.MXFP4_DECODE_HEADER}

extern "C" void deq_mxfp4_words(const uint32_t* w, const uint8_t* scales, int n_groups, uint16_t* out_bf16) {{
  for (int g = 0; g < n_groups; ++g) {{
    const __nv_bfloat162 s2 = qz_e8m0_bf162(scales[g]);
    for (int k = 0; k < 4; ++k) {{
      const uint32_t word = w[g * 4 + k];
      for (int t = 0; t < 4; ++t) {{
        const unsigned pair = qz_deq_mxfp4(word, t, s2);
        out_bf16[g * 32 + k * 8 + 2 * t] = static_cast<uint16_t>(pair & 0xffffu);
        out_bf16[g * 32 + k * 8 + 2 * t + 1] = static_cast<uint16_t>(pair >> 16);
      }}
    }}
  }}
}}
"""
        src = tmp_path / 'check_mxfp4_cuda.cpp'
        so = tmp_path / 'check_mxfp4_cuda.so'
        src.write_text(cpp, encoding='utf-8')
        subprocess.run([cxx, '-std=c++17', '-O2', '-shared', '-fPIC', str(src), '-o', str(so)], check=True)
        lib = ctypes.CDLL(str(so))
        lib.deq_mxfp4_words.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

        exps = np.arange(2, 253, dtype=np.uint8)
        n_groups = exps.size
        rng = np.random.default_rng(771)
        w_np = rng.integers(0, 1 << 32, size=(n_groups, 4), dtype=np.uint32)
        # Cover all 16 E2M1 nibbles explicitly in groups 0..15 across every byte slot
        for g in range(16):
            nib_word = sum(((g + i) & 0xf) << (4 * i) for i in range(8))
            w_np[g, :] = nib_word
        got_u16 = np.empty((n_groups, 32), dtype=np.uint16)
        lib.deq_mxfp4_words(w_np.ctypes.data, exps.ctypes.data, n_groups, got_u16.ctypes.data)
        want_bf16 = mx.dequantize(
            mx.array(w_np), scales=mx.array(exps[:, None]), group_size=32, bits=4, mode='mxfp4', dtype=mx.bfloat16
        )
        mx.eval(want_bf16)
        np.testing.assert_array_equal(got_u16, np.array(want_bf16.view(mx.uint16)))

    # 2. `cuda_experts` (unstaged & staged) dispatch + `cuda_moe` opt-out for `mxfp4`
    def make_mxfp4_switch(k_dim, n_dim, num_experts=16):
        sw = SwitchGLU(input_dims=k_dim, hidden_dims=n_dim, num_experts=num_experts, bias=False)
        for p_name in ('gate_proj', 'up_proj', 'down_proj'):
            p = getattr(sw, p_name)
            p.weight = p.weight.astype(mx.bfloat16)
            setattr(sw, p_name, p.to_quantized(group_size=32, bits=4, mode='mxfp4'))
        mx.eval(sw.parameters())
        return sw

    mx.random.seed(772)
    sw_small = make_mxfp4_switch(128, 128)
    sw_staged = make_mxfp4_switch(512, 128)
    assert cuda_experts._staged_smem(512, 32, mxfp4=True) < cuda_experts._staged_smem(512, 32, mxfp4=False)
    assert cuda_experts.staged(128, 512, 32, mxfp4=True)

    with monkeypatch.context() as mp:
        calls = []

        class FakeExpertsKernel:
            def __init__(self, kind_tag):
                self.kind_tag = kind_tag

            def __call__(self, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
                t = dict(template)
                calls.append((self.kind_tag, t, inputs))
                assert t['MXFP4'] is True
                x_in, w, s, b_arg, idx, w2, s2, b2_arg = inputs
                # biases slots carry `scales` uint8 when MXFP4=True
                assert b_arg.dtype == mx.uint8 and b2_arg.dtype == mx.uint8
                slots, n_out = output_shapes[0]
                k_in = t['K']
                gs = t['GS']
                if self.kind_tag == 'staged' or t.get('PAIR'):
                    x_mat = x_in.reshape(-1, k_in)
                    rows_in = x_mat.shape[0]
                    top_k = max(1, slots // rows_in)
                    x_4d = x_mat.reshape(rows_in, 1, 1, 1, k_in)
                    rhs = idx.reshape(rows_in, 1, top_k)
                    g_val = mx.gather_qmm(x_4d, w, scales=s, transpose=True, group_size=gs, bits=4, mode='mxfp4',
                                          rhs_indices=rhs)
                    u_val = mx.gather_qmm(x_4d, w2, scales=s2, transpose=True, group_size=gs, bits=4, mode='mxfp4',
                                          rhs_indices=rhs)
                    return [sw_small.activation(u_val, g_val).reshape(slots, n_out)]
                rhs = idx.reshape(2, 1, slots // 2)
                return [mx.gather_qmm(x_in.reshape(2, 1, slots // 2, 1, k_in), w, scales=s, transpose=True,
                                      group_size=gs, bits=4, mode='mxfp4', rhs_indices=rhs).reshape(slots, n_out)]

        mp.setattr(cuda_experts.mx.metal, 'is_available', lambda: False)
        mp.setattr(cuda_experts.mx, 'default_device', lambda: mx.gpu)
        mp.setattr(cuda_moe.mx.metal, 'is_available', lambda: False)
        mp.setattr(cuda_moe.mx, 'default_device', lambda: mx.gpu)
        mp.setattr(cuda_kernels, 'available', lambda: True)
        mp.setattr(cuda_experts, '_kernel', lambda: FakeExpertsKernel('plain'))
        mp.setattr(cuda_experts, '_staged_kernel', lambda k, gs, mxfp4=False: FakeExpertsKernel('staged'))
        mp.setattr(cuda_experts, 'ON', [True])
        mp.setattr(cuda_experts, 'CHOICE', {})
        mp.setattr(cuda_experts, 'CHECKED', {})
        mp.setattr(cuda_experts, 'STATS', dict(served=0, checked=0, mismatched=0))

        for k_dim, sw_cur, exp_tag in ((128, sw_small, 'plain'), (512, sw_staged, 'staged')):
            x_tok = mx.random.normal((2, 1, k_dim)).astype(mx.bfloat16)
            ind_tok = mx.array([[[0, 2, 5, 7]], [[1, 3, 6, 9]]], dtype=mx.uint32)
            mx.eval(x_tok, ind_tok)
            assert cuda_experts.takes(sw_cur, x_tok, ind_tok)
            for _ in range(cuda_experts.CHECKS + 1):
                got = cuda_experts.decode(sw_cur, x_tok, ind_tok, lambda s=sw_cur, x=x_tok, i=ind_tok: expert_qmv.stock(s, x, i))
                want = expert_qmv.stock(sw_cur, x_tok, ind_tok)
                mx.eval(got, want)
                assert mx.array_equal(got, want)
            assert any(tag == exp_tag for tag, _, _ in calls)
        assert cuda_experts.STATS['mismatched'] == 0
        assert cuda_experts.STATS['served'] >= 2

        # `cuda_moe.takes` must opt out when `down_proj.mode == 'mxfp4'`
        dummy_block = nn.Module()
        dummy_block.top_k = 8
        dummy_block.gate = nn.Linear(128, 16, bias=False).to_quantized(group_size=64, bits=8)
        dummy_block.shared_expert_gate = nn.Linear(128, 1, bias=False).to_quantized(group_size=64, bits=8)
        sh = nn.Module()
        for pn, (in_d, out_d) in (('gate_proj', (128, 64)), ('up_proj', (128, 64)), ('down_proj', (64, 128))):
            setattr(sh, pn, nn.Linear(in_d, out_d, bias=False).to_quantized(group_size=64, bits=4))
        dummy_block.shared_expert = sh
        dummy_block.switch_mlp = sw_small
        mx.eval(dummy_block.parameters())
        assert not cuda_moe.takes(
            dummy_block, mx.zeros((1, 1, 128), dtype=mx.bfloat16),
            mx.zeros((1, 1, 128), dtype=mx.bfloat16), mx.ones((1, 1, 1), dtype=mx.bfloat16),
        )

    # 3. `QUANTEEZE_BITS=mxfp4`: fitting, serve, control, home, requant, QuantCache, HostCopy, Demand
    assert fitting.group_size('mxfp4') == 32
    assert fitting.quantized_bytes(32, 'mxfp4') == 17.0
    assert fitting.quantized_bytes(128, 'mxfp4') == 68.0
    with monkeypatch.context() as mp:
        mp.setenv('QUANTEEZE_SETTINGS', str(tmp_path / 'settings.json'))
        mp.setenv('QUANTEEZE_BITS', 'mxfp4')
        assert serve._uniform_bits() == ('mxfp4', 32)
        assert serve._uniform_label(('mxfp4', 32)) == 'uniform MXFP4 (group size 32)'
        assert control._uniform_bits() == 'mxfp4'
        mp.setenv('QUANTEEZE_BITS', '')
        res_mxfp4 = control.change({'QUANTEEZE_BITS': 'mxfp4'})
        assert 'QUANTEEZE_BITS' in res_mxfp4['applied'] and not res_mxfp4['rejected']
        assert control._uniform_bits() == 'mxfp4'

    class ToyMoELayer(nn.Module):
        def __init__(self, sw):
            super().__init__()
            mlp = nn.Module()
            mlp.switch_mlp = sw
            self.mlp = mlp

    class ToyLM(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

    ckpt_dir = tmp_path / 'bf16_ckpt'
    ckpt_dir.mkdir()
    bf16_tensors = {}
    mx.random.seed(773)
    for layer_idx in range(2):
        sw = SwitchGLU(input_dims=128, hidden_dims=128, num_experts=12, bias=False)
        for p_name in ('gate_proj', 'up_proj', 'down_proj'):
            w_bf16 = getattr(sw, p_name).weight.astype(mx.bfloat16)
            getattr(sw, p_name).weight = w_bf16
            bf16_tensors[f'layers.{layer_idx}.mlp.switch_mlp.{p_name}.weight'] = w_bf16
    mx.eval(*bf16_tensors.values())
    (ckpt_dir / 'config.json').write_text('{"num_hidden_layers": 2}', encoding='utf-8')
    mx.save_safetensors(str(ckpt_dir / 'model.safetensors'), bf16_tensors)

    ub_map, stock_bits = home.uniform_bytes(ckpt_dir)
    assert stock_bits == 16 and 'mxfp4' in ub_map
    assert ub_map['mxfp4'] == int(fitting.quantized_bytes(2 * 3 * 12 * 128 * 128, 'mxfp4'))

    def make_bf16_model():
        layers = []
        for layer_idx in range(2):
            sw = SwitchGLU(input_dims=128, hidden_dims=128, num_experts=12, bias=False)
            for p_name in ('gate_proj', 'up_proj', 'down_proj'):
                getattr(sw, p_name).weight = bf16_tensors[f'layers.{layer_idx}.mlp.switch_mlp.{p_name}.weight']
            layers.append(ToyMoELayer(sw))
        mx.eval([l.parameters() for l in layers])
        return ToyLM(layers)

    m_lowered = make_bf16_model()
    with monkeypatch.context() as mp:
        cache_dir = tmp_path / 'qcache'
        mp.setenv('QUANTEEZE_QUANT_CACHE', str(cache_dir))
        requant.lower_model(m_lowered.layers, ckpt_dir, 'mxfp4')
        for layer in m_lowered.layers:
            for p_name in ('gate_proj', 'up_proj', 'down_proj'):
                proj = getattr(layer.mlp.switch_mlp, p_name)
                assert proj.mode == 'mxfp4'
                assert proj.bits == 4
                assert proj.group_size == 32
                assert proj.get('biases') is None
                assert proj.scales.dtype == mx.uint8

        cached = requant.cached_checkpoint(m_lowered.layers, ckpt_dir, 'mxfp4')
        assert isinstance(cached, requant.CachedCheckpoint)
        assert cached.mode(0, 'gate_proj') == 'mxfp4'
        assert cached.quantization(0, 'gate_proj', 128) == (32, 4)

        m_flat = make_bf16_model()
        m_cached = make_bf16_model()
        d_flat = demand.Demand(m_flat, requant.Flat(ckpt_dir, 'mxfp4'), budget_bytes=1, workers=2)
        held = q2_experts.hold(cached, budget=1 << 24, workers=2, by_rows=True)
        d_cached = None
        try:
            held.wait()
            d_cached = demand.Demand(m_cached, cached, budget_bytes=1, workers=2)
            assert d_flat.switches[0].gate_proj.mode == 'mxfp4'
            assert d_cached.switches[0].gate_proj.mode == 'mxfp4'
            x_test = mx.random.normal((2, 3, 128)).astype(mx.bfloat16)
            idx_test = mx.array([[[0, 3, 7, 11]] * 3, [[1, 4, 8, 10]] * 3], dtype=mx.uint32)
            x_all = mx.random.normal((1, 3, 128)).astype(mx.bfloat16)
            idx_all = mx.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]], dtype=mx.uint32)
            mx.eval(x_test, idx_test, x_all, idx_all)
            for l_idx in range(2):
                y_ref = m_lowered.layers[l_idx].mlp.switch_mlp(x_test, idx_test)
                y_fl = m_flat.layers[l_idx].mlp.switch_mlp(x_test, idx_test)
                y_ca = m_cached.layers[l_idx].mlp.switch_mlp(x_test, idx_test)
                yp_ref = m_lowered.layers[l_idx].mlp.switch_mlp(x_all, idx_all)
                yp_fl = m_flat.layers[l_idx].mlp.switch_mlp(x_all, idx_all)
                yp_ca = m_cached.layers[l_idx].mlp.switch_mlp(x_all, idx_all)
                mx.eval(y_ref, y_fl, y_ca, yp_ref, yp_fl, yp_ca)
                assert mx.array_equal(y_fl, y_ref)
                assert mx.array_equal(y_ca, y_ref)
                assert mx.array_equal(yp_fl, yp_ref)
                assert mx.array_equal(yp_ca, yp_ref)
        finally:
            if d_cached is not None:
                d_cached.close()
            d_flat.close()
            q2_experts.let_go(cached.path)

    # 4. GGUF `MXFP4` (`kind == 39`): in-memory `load_model` and `HostCopy` + `Demand`
    gguf_path = tmp_path / 'experts-mxfp4.gguf'
    gguf_rng = np.random.default_rng(774)
    n_exp, d_in, d_hid = 16, 128, 128
    tensors_spec = []
    for l_idx in range(2):
        for gguf_name, rows, cols in (
            ('ffn_gate_exps', d_hid, d_in),
            ('ffn_up_exps', d_hid, d_in),
            ('ffn_down_exps', d_in, d_hid),
        ):
            kb = cols // iq_experts.per_block(iq_experts.MXFP4)
            arr = _random_ggml_blocks(gguf_rng, iq_experts.MXFP4, n_exp * rows * kb).reshape(
                n_exp, rows, kb, iq_experts.BLOCK[iq_experts.MXFP4]
            )
            tensors_spec.append((f'blk.{l_idx}.{gguf_name}.weight', (cols, rows, n_exp), iq_experts.MXFP4, arr.tobytes()))
    _write_gguf(gguf_path, tensors_spec)

    src_mem = gguf_experts.ExpertGGUF(gguf_path)
    src_dem = gguf_experts.ExpertGGUF(gguf_path)
    m_mem = make_bf16_model()
    for l_idx in range(2):
        m_mem.layers[l_idx].mlp.switch_mlp = SwitchGLU(input_dims=d_in, hidden_dims=d_hid, num_experts=n_exp, bias=False)
    m_dem = make_bf16_model()
    for l_idx in range(2):
        m_dem.layers[l_idx].mlp.switch_mlp = SwitchGLU(input_dims=d_in, hidden_dims=d_hid, num_experts=n_exp, bias=False)
    assert 'MXFP4' in src_mem.format_label([0, 1])

    gguf_experts.load_model(m_mem.layers, src_mem)
    assert m_mem.layers[0].mlp.switch_mlp.gate_proj.mode == 'gguf:mxfp4'
    held_gguf = q2_experts.hold(src_dem, budget=1 << 24, workers=2, by_rows=True)
    dem_gguf = None
    try:
        held_gguf.wait()
        dem_gguf = demand.Demand(m_dem, src_dem, budget_bytes=1, workers=2)
        x_dec = mx.random.normal((1, 1, d_in)).astype(mx.bfloat16)
        idx_dec = mx.array([[[8, 9, 10, 11]]], dtype=mx.uint32)
        mx.eval(x_dec, idx_dec)
        for l_idx in range(2):
            ym = m_mem.layers[l_idx].mlp.switch_mlp(x_dec, idx_dec)
            yd = m_dem.layers[l_idx].mlp.switch_mlp(x_dec, idx_dec)
            mx.eval(ym, yd)
            _assert_gguf_close(yd, ym, rel=1e-2)
        assert dem_gguf.host_served == 8
    finally:
        if dem_gguf is not None:
            dem_gguf.close()
        q2_experts.let_go(src_dem)
        src_mem.close()



