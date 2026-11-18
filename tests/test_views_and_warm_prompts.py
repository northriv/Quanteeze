# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A checkpoint's n-gram (PLE) view found only where this server reads its manifest (`ple_store.readable`,
`home.ple_view`, `home._make_view`, `home.ngram_source`); a prompt the prompt cache took part of widened
as any other (`prefill.widen`); and a gated delta layer's projections held once beside their fused
decode copy (`linear_attn.share_fused_linears`)."""
import gc
import json
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from quanteeze import home, linear_attn, ple_store, prefill


def _checkpoint(root, name='Model'):
    path = root / name
    path.mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp', 'text_config': {}}))
    return path


def _view(path, source, manifest):
    """A view at `path` of the checkpoint at `source` whose config names `manifest`."""
    path.mkdir(parents=True)
    (path / 'ple-store.json').write_text(json.dumps(manifest))
    (path / 'config.json').write_text(json.dumps(
        {'model_type': 'qwen4_exp', 'text_config': {'ple_storage': {'manifest': 'ple-store.json', 'cache_rows': 0}}}))
    (path / 'EXTERNAL_PLE.json').write_text(json.dumps(
        {'format': 'quanteeze_external_ple_v1', 'source_model': str(source), 'external_ple_manifest': 'ple-store.json'}))
    return path


OURS = {'format': ple_store.FORMAT, 'source_root': '../Model', 'row_width': 160, 'quantization': {}}
MLX_VLM = {'version': ple_store.MLX_VLM_VERSION, 'source_root': '../Model', 'row_width': 160, 'quantization': {}}
OTHER = {'format': 'other_tool_ple_v1', 'source_root': '../Model', 'row_width': 160, 'quantization': {}}


def test_readable_takes_ours_and_mlx_vlms_only(tmp_path):
    src = _checkpoint(tmp_path)
    assert ple_store.readable(_view(tmp_path / 'ours', src, OURS))
    assert ple_store.readable(_view(tmp_path / 'mlx-vlm', src, MLX_VLM))
    assert not ple_store.readable(_view(tmp_path / 'other', src, OTHER))
    assert not ple_store.readable(_view(tmp_path / 'old', src, {**MLX_VLM, 'version': MLX_VLM['version'] + 1}))
    assert not ple_store.readable(_view(tmp_path / 'short', src, {k: v for k, v in MLX_VLM.items() if k != 'version'}))
    assert not ple_store.readable(tmp_path / 'missing')
    assert not ple_store.readable(src)                               # a checkpoint names no manifest


def test_ple_view_passes_over_another_tools_view(tmp_path, monkeypatch):
    monkeypatch.setattr(home, 'PLE_VIEWS', tmp_path / 'views')
    src = _checkpoint(tmp_path)
    other = _view(src.with_name('Model.ple-view'), src, OTHER)
    assert home.ple_view(src) is None
    ours = _view(src.with_name('Model.quanteeze.ple-view'), src, OURS)
    assert home.ple_view(src) == ours
    assert json.loads((other / 'ple-store.json').read_text()) == OTHER


def test_make_view_beside_another_tools_view(tmp_path, monkeypatch):
    """With the view's own place elsewhere failing (another volume) and its place beside the
    checkpoint another tool's, the view is made at `<name>.quanteeze.ple-view`, the other left as it is."""
    views = tmp_path / 'views'
    monkeypatch.setattr(home, 'PLE_VIEWS', views)
    src = _checkpoint(tmp_path)
    other = _view(src.with_name('Model.ple-view'), src, OTHER)
    before = sorted(p.name for p in other.iterdir())
    calls = []

    def prepare(source, target):
        calls.append(target)
        if views in target.parents:
            raise ValueError('on different volumes: no hard links')
        _view(target, source, OURS)

    made = home._make_view(src, prepare)
    assert made == src.with_name('Model.quanteeze.ple-view')
    assert home.ple_view(src) == made
    assert all(c.name != 'Model.ple-view' and not c.name.startswith('Model.ple-view.part') for c in calls)
    assert sorted(p.name for p in other.iterdir()) == before
    assert json.loads((other / 'ple-store.json').read_text()) == OTHER


def test_ngram_source_serves_the_checkpoint_of_another_tools_view(tmp_path, monkeypatch):
    monkeypatch.setattr(home, 'PLE_VIEWS', tmp_path / 'views')
    monkeypatch.setenv('QUANTEEZE_NGRAM_SSD', 'on')
    src = _checkpoint(tmp_path)
    other = _view(src.with_name('Model.ple-view'), src, OTHER)
    ours = _view(src.with_name('Model.quanteeze.ple-view'), src, OURS)
    monkeypatch.setattr(home, 'ple_info', lambda p: {'gb': 1.0, 'view': None, 'is_view': False})
    view, why = home.ngram_source(other)
    assert view == ours and 'read from the SSD' in why
    view, _ = home.ngram_source(ours)                                # a view of ours: served as named
    assert view == ours


class _Batch:
    def __init__(self, uids, pad, step=512):
        self.uids, self._right_pad_per_row, self.prefill_step_size = uids, pad, step
        self.seen = None

    def prompt_step(self):
        self.seen = (self.prefill_step_size, prefill.SETTINGS.chunk)
        return 1


@pytest.mark.parametrize('uids, pad, widened', [
    ([1], None, True),
    ([1], [0], True),                                                # a prompt the APC took part of
    ([1], [3], False),
    ([1, 2], [0, 3], False),
    ([1, 2], None, False),
])
def test_widen_takes_a_warm_prompt_of_one_row(monkeypatch, uids, pad, widened):
    monkeypatch.setattr(prefill.SETTINGS, 'chunk', prefill.SETTINGS.chunk)
    monkeypatch.setattr(prefill.SETTINGS, 'enabled', lambda: True)
    cls = type('Batch', (_Batch,), {})
    prefill.widen(cls, 16384)
    batch = cls(uids, pad)
    batch.prompt_step()
    assert batch.seen == ((16384, 512) if widened else (512, prefill.SETTINGS.chunk))
    assert batch.prefill_step_size == 512


def _projections(outputs, inputs=512, bits=4, group_size=64, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n in outputs:
        q = nn.QuantizedLinear.from_linear(nn.Linear(inputs, n, bias=False), group_size=group_size, bits=bits)
        q.weight = mx.array(rng.integers(0, 2 ** 32, q.weight.shape, dtype=np.uint32))
        q.scales = mx.array(rng.standard_normal(q.scales.shape).astype(np.float32) * 0.01).astype(mx.bfloat16)
        q.biases = mx.array(rng.standard_normal(q.biases.shape).astype(np.float32) * 0.01).astype(mx.bfloat16)
        mx.eval(q.parameters())
        out.append(q)
    return out


def test_share_fused_linears_holds_one_copy_bit_for_bit():
    from mlx_vlm.speculative.ops import linear as real
    ops = types.SimpleNamespace(_decode_quantized_linears_fused=real._decode_quantized_linears_fused)
    linear_attn.share_fused_linears(ops)
    linear_attn.share_fused_linears(ops)                             # once
    assert ops._decode_quantized_linears_fused._quanteeze_shared
    outputs = (1024, 512, 32, 32)
    x1 = mx.random.normal((1, 1, 512), key=mx.random.key(1)).astype(mx.bfloat16)
    xn = mx.random.normal((1, 5, 512), key=mx.random.key(2)).astype(mx.bfloat16)

    ref = _projections(outputs)
    want1 = real._decode_quantized_linears_fused(ref, x1)
    wantn = tuple(lin(xn) for lin in ref)
    mx.eval(want1, wantn)
    del ref
    gc.collect()

    got = _projections(outputs)
    fused_bytes = sum(lin.weight.nbytes + lin.scales.nbytes + lin.biases.nbytes for lin in got)
    before, count = mx.get_active_memory(), linear_attn.SHARED['fused']
    got1 = ops._decode_quantized_linears_fused(got, x1)
    mx.eval(got1)
    gc.collect()
    held = mx.get_active_memory() - before
    assert linear_attn.SHARED['fused'] == count + 1
    assert held < fused_bytes / 2                                    # mlx-vlm's own kept all of it twice
    again = ops._decode_quantized_linears_fused(got, x1)             # its key still holds: not fused again
    gotn = tuple(lin(xn) for lin in got)
    mx.eval(again, gotn)
    assert linear_attn.SHARED['fused'] == count + 1
    for want, *have in zip(want1, got1, again):
        assert all(mx.array_equal(want, h).item() for h in have)
    assert all(mx.array_equal(w, g).item() for w, g in zip(wantn, gotn))
    key, weights, _, _, splits = got[0]._fused_decode_linears
    assert key == tuple((id(lin.weight), id(lin.scales), id(lin.biases)) for lin in got)
    bounds = [0, *splits, weights.shape[0]]
    assert [lin.weight.shape[0] for lin in got] == [b - a for a, b in zip(bounds, bounds[1:])]
