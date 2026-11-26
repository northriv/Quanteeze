# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from quanteeze import control, cuda_attn, cuda_experts, cuda_kernels, cuda_moe, cuda_qmv, hyper, linear_attn, mtp, variables
from quanteeze.arch import qwen4_exp


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal backend required')
@pytest.mark.parametrize('bits', [4, 8])
def test_hyper_project_rows_metal_matches_per_token(bits):
    mx.random.seed(0)
    down = nn.QuantizedLinear(512, 8, bias=False, group_size=64, bits=bits)
    inject = nn.QuantizedLinear(512, 4, bias=False, group_size=64, bits=bits)
    for mod in (down, inject):
        mod.scales = mod.scales.astype(mx.bfloat16)
        mod.biases = mod.biases.astype(mx.bfloat16)
    x = mx.random.normal((1, 3, 512)).astype(mx.bfloat16)
    mx.eval(down.parameters(), inject.parameters(), x)

    module = SimpleNamespace(
        hc_count=4,
        hidden_size=128,
        hc_norm=SimpleNamespace(weight=mx.ones((128,), mx.bfloat16), group_size=128),
        input_mix_weight_down=down,
        block_inject_weight=inject,
    )
    assert hyper.rows_metal(module, x)

    low_rows, inj_rows = hyper.project_rows(down, inject, x, 4)
    each = [hyper.project(down, inject, x[:, i:i + 1], 4) for i in range(3)]
    low_want = mx.concatenate([e[0] for e in each], axis=1)
    inj_want = mx.concatenate([e[1] for e in each], axis=1)
    mx.eval(low_rows, inj_rows, low_want, inj_want)
    assert mx.array_equal(low_rows, low_want).item()
    assert mx.array_equal(inj_rows, inj_want).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal backend required')
def test_linear_attn_conv_rows_metal_matches_per_token():
    mx.random.seed(1)
    c, taps = 384, 4
    conv1d = nn.Conv1d(in_channels=c, out_channels=c, kernel_size=taps, groups=c, bias=False)
    conv1d.weight = mx.random.normal(conv1d.weight.shape).astype(mx.bfloat16)
    layer = SimpleNamespace(
        head_k_dim=128,
        head_v_dim=128,
        num_k_heads=1,
        conv_kernel_size=taps,
        conv1d=conv1d,
        norm=SimpleNamespace(activation='sigmoid', weight=mx.ones((128,), mx.bfloat16)),
        training=False,
    )
    state = mx.random.normal((1, taps - 1, c)).astype(mx.bfloat16)
    x = mx.random.normal((1, 3, c)).astype(mx.bfloat16)
    mx.eval(conv1d.weight, state, x)

    assert linear_attn.applies_rows(layer, x, None, [state])
    got = linear_attn.conv_rows(layer, state, x)
    win = state
    outs = []
    for i in range(3):
        xi = x[:, i:i + 1]
        outs.append(linear_attn.conv(layer, win, xi))
        win = mx.concatenate([win[:, 1:], xi], axis=1)
    want = mx.concatenate(outs, axis=1)
    mx.eval(got, want)
    assert mx.array_equal(got, want).item()


def test_qwen4_exp_verifying_and_padding_on_host():
    seen = []

    def base_verify(self, inputs, cache, sampler=None):
        seen.append(qwen4_exp.VERIFYING[0])
        return 'ok'

    wrapped = qwen4_exp._verifying(base_verify)
    assert qwen4_exp.VERIFYING[0] is False
    assert wrapped(None, None, None) == 'ok'
    assert seen == [True]
    assert qwen4_exp.VERIFYING[0] is False

    from mlx_vlm.models import cache as caches
    from mlx_vlm.models.qwen3_5 import language as qwen35

    assert qwen4_exp.padding_on_host()
    c = caches.ArraysCache(size=2, left_padding=[5, 7])
    c.advance(2)
    pads, max_pad = qwen35._qwen3_5_left_padding_info(c)
    assert pads == (3, 5)
    assert max_pad == 5


def test_cuda_qmv_attn_moe_experts_predicates(monkeypatch):
    monkeypatch.setattr(cuda_qmv, 'ONE_LEAST', [1 << 20])
    x1 = mx.zeros((1, 1, 512), mx.bfloat16)
    x3 = mx.zeros((1, 3, 512), mx.bfloat16)
    w4 = mx.zeros((256, 64), mx.uint32)
    s = mx.zeros((256, 8), mx.bfloat16)
    b = mx.zeros((256, 8), mx.bfloat16)
    assert cuda_qmv.takes(x1, w4, s, b, True, 64, 4, 'affine')
    assert cuda_qmv.takes_rows(x3, w4, s, b, True, 64, 4, 'affine')

    w8_small = mx.zeros((256, 128), mx.uint32)
    assert cuda_qmv.one_row(x1, w8_small, s, b, True, 64, 8, 'affine')
    assert not cuda_qmv.takes8(x1, w8_small, s, b, True, 64, 8, 'affine')
    w8_big = mx.zeros((4096, 128), mx.uint32)
    s_big = mx.zeros((4096, 8), mx.bfloat16)
    b_big = mx.zeros((4096, 8), mx.bfloat16)
    assert cuda_qmv.takes8(x1, w8_big, s_big, b_big, True, 64, 8, 'affine')

    assert cuda_attn._causal('causal', 3, 10)
    mask = mx.tril(mx.ones((1, 1, 3, 5), mx.bool_), k=2)
    assert cuda_attn._causal(mask, 3, 2)
    bad = mx.ones((1, 1, 3, 5), mx.bool_)
    assert not cuda_attn._causal(bad, 3, 2)
    assert cuda_attn.softmax_block(256) == 32

    assert 'qz_qmv8_row<' in cuda_moe.SHARED_IN_SOURCE
    assert 'QzMul{}' in cuda_moe.SHARED_OUT_SOURCE
    assert cuda_moe.summary()['shared'] == cuda_moe.STATS['shared']

    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)

    class Projection:
        def __init__(self, shape, sshape):
            self.mode = 'affine'
            self.bits = 4
            self.group_size = 64
            self.weight = mx.zeros(shape, mx.uint32)
            self.scales = mx.zeros(sshape, mx.bfloat16)
            self.biases = mx.zeros(sshape, mx.bfloat16)

        def get(self, name, default=None):
            return getattr(self, name, default)

        def __contains__(self, name):
            return hasattr(self, name)

    class SwiGLU:
        pass

    sw = SimpleNamespace(
        gate_proj=Projection((16, 128, 32), (16, 128, 4)),
        up_proj=Projection((16, 128, 32), (16, 128, 4)),
        down_proj=Projection((16, 256, 16), (16, 256, 2)),
        activation=SwiGLU(),
    )
    assert cuda_experts.takes(sw, mx.zeros((1, 3, 256), mx.bfloat16), mx.zeros((1, 3, 4), mx.uint32))


def test_mtp_pays_in_memory_and_settings(monkeypatch):
    monkeypatch.setattr(mx.metal, 'is_available', lambda: True)
    assert mtp.pays_in_memory('qwen4_exp')
    monkeypatch.setattr(mx.metal, 'is_available', lambda: False)
    assert not mtp.pays_in_memory('qwen4_exp')
    assert mtp.pays_in_memory('glm5_next')

    for name in ('QUANTEEZE_CUDA_ATTN_ROWS', 'QUANTEEZE_CUDA_QMV_ROWS'):
        assert name in variables.SETTINGS
        assert control.CHOICES[name] == ('on', 'off')


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal backend required')
@pytest.mark.parametrize('shape', [(1, 2, 256), (1, 3, 256), (1, 4, 256), (2, 1, 256)])
def test_expert_qmv_verify_fused_matches_stock_and_toggle(shape, monkeypatch):
    """Multi-token verify (`1, T, D`) and batched decode (`B, 1, D`) run via `fused()` when `VERIFY=True`
    and match stock `SwitchGLU` bit-for-bit; `VERIFY=False` falls back to `own()` (`served`)."""
    from mlx_lm.models.switch_layers import SwitchGLU
    from quanteeze import expert_qmv

    mx.random.seed(19)
    sw = SwitchGLU(input_dims=256, hidden_dims=128, num_experts=8, bias=False)
    sw.set_dtype(mx.bfloat16)
    for proj_name in ('gate_proj', 'up_proj', 'down_proj'):
        q = getattr(sw, proj_name).to_quantized(group_size=64, bits=4, mode='affine')
        setattr(sw, proj_name, q)
    mx.eval(sw.parameters())

    def stock(x_in, idx_in):
        return sw(x_in, idx_in)

    monkeypatch.setattr(expert_qmv, 'CHOICE', {})
    monkeypatch.setattr(expert_qmv, 'CHECKED', {})
    monkeypatch.setattr(expert_qmv, 'STATS', dict(served=0, fused=0, checked=0, mismatched=0))
    monkeypatch.setattr(expert_qmv, 'ON', [True])
    monkeypatch.setattr(expert_qmv, 'FUSED', [True])
    monkeypatch.setattr(expert_qmv, 'VERIFY', [True])

    b0, t0, d = shape
    for step in range(expert_qmv.CHECKS + 2):
        mx.random.seed(100 + step)
        x = (mx.random.normal((b0, t0, d)) * 0.25).astype(mx.bfloat16)
        idx = mx.random.randint(0, 8, (b0, t0, 3), dtype=mx.uint32)
        mx.eval(x, idx)
        want = stock(x, idx)
        got = expert_qmv.decode(sw, x, idx)
        mx.eval(want, got)
        assert mx.array_equal(got, want).item()
    assert expert_qmv.STATS['mismatched'] == 0
    assert expert_qmv.STATS['fused'] == 2

    if t0 > 1:
        monkeypatch.setattr(expert_qmv, 'VERIFY', [False])
        assert not expert_qmv.fusable(sw, x)
        served_before = expert_qmv.STATS['served']
        for _ in range(expert_qmv.CHECKS + 1):
            got = expert_qmv.decode(sw, x, idx)
            mx.eval(got)
        assert expert_qmv.STATS['served'] == served_before + 1

    assert 'QUANTEEZE_EXPERT_QMV_VERIFY_FUSED' in variables.SETTINGS
    assert control.CHOICES['QUANTEEZE_EXPERT_QMV_VERIFY_FUSED'] == ('on', 'off')
