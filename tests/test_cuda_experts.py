# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for uniform 2-bit & 4-bit CUDA expert dispatch and `qz_expert_rows2` in `quanteeze.cuda_experts`."""
import mlx.core as mx
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from quanteeze import cuda_experts, cuda_kernels


def _make_switch(hidden=512, inner=256, num_experts=12, bits=2, gs=64, mode='affine',
                 gate_bits=None, up_bits=None, down_bits=None):
    mx.random.seed(11 + bits + gs)
    sw = SwitchGLU(input_dims=hidden, hidden_dims=inner, num_experts=num_experts, bias=False)
    sw.set_dtype(mx.bfloat16)
    g_bits = bits if gate_bits is None else gate_bits
    u_bits = bits if up_bits is None else up_bits
    d_bits = bits if down_bits is None else down_bits
    sw.gate_proj = sw.gate_proj.to_quantized(group_size=gs, bits=g_bits, mode=mode)
    sw.up_proj = sw.up_proj.to_quantized(group_size=gs, bits=u_bits, mode=mode)
    sw.down_proj = sw.down_proj.to_quantized(group_size=gs, bits=d_bits, mode=mode)
    mx.eval(sw.parameters())
    return sw


def test_source_and_predicates_for_2bit_4bit_and_mxfp4(monkeypatch):
    """`HEADER`/`SOURCE` expose `qz_expert_rows2` and `BITS`, and `_projection`/`takes` accept 2-bit affine."""
    assert 'qz_expert_rows2' in cuda_experts.HEADER
    assert 'qz_expert_rows<N, K, GS, PAIR, MXFP4, BITS>' in cuda_experts.SOURCE

    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
    x = mx.zeros((1, 1, 512), dtype=mx.bfloat16)
    idx = mx.array([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=mx.uint32)

    sw2 = _make_switch(bits=2, gs=64)
    assert cuda_experts._projection(sw2.gate_proj, 512, mx.bfloat16)
    assert cuda_experts._projection(sw2.down_proj, 256, mx.bfloat16)
    assert cuda_experts.takes(sw2, x, idx)

    sw4 = _make_switch(bits=4, gs=64)
    assert cuda_experts.takes(sw4, x, idx)

    sw_mxfp4 = _make_switch(bits=4, gs=32, mode='mxfp4')
    assert cuda_experts._projection(sw_mxfp4.gate_proj, 512, mx.bfloat16)
    assert cuda_experts.takes(sw_mxfp4, x, idx)

    sw3 = _make_switch(bits=3, gs=64)
    assert not cuda_experts._projection(sw3.gate_proj, 512, mx.bfloat16)
    assert not cuda_experts.takes(sw3, x, idx)

    sw_mismatch = _make_switch(gate_bits=4, up_bits=2, down_bits=4, gs=64)
    assert not cuda_experts.takes(sw_mismatch, x, idx)


def test_2bit_gate_up_routes_through_run_not_staged_and_separates_choice_keys(monkeypatch):
    """2-bit `gate_up` bypasses `_staged_kernel` via `_run`, and `decode()` keys 2-bit separately from 4-bit."""
    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
    monkeypatch.setattr(cuda_experts, 'ON', [True])
    monkeypatch.setattr(cuda_experts, 'CHOICE', {})
    monkeypatch.setattr(cuda_experts, 'CHECKED', {})
    monkeypatch.setattr(cuda_experts, 'STATS', dict(served=0, checked=0, mismatched=0))
    monkeypatch.setattr(cuda_experts, 'CHECKS', 1)

    run_calls = []
    monkeypatch.setattr(
        cuda_experts,
        '_run',
        lambda x, proj, idx, x_div, pair=None: run_calls.append(
            (getattr(proj, 'bits', None), x_div, pair is not None)
        ) or mx.zeros((idx.size, proj.weight.shape[1]), dtype=x.dtype),
    )
    monkeypatch.setattr(
        cuda_experts,
        '_staged_kernel',
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('2-bit must not enter _staged_kernel')),
    )

    sw2 = _make_switch(hidden=512, inner=512, bits=2, gs=64)
    x = mx.ones((1, 1, 512), dtype=mx.bfloat16)
    idx = mx.array([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=mx.uint32)
    h = cuda_experts.gate_up(sw2, x.reshape(1, -1), idx.reshape(-1))
    assert h.shape == (8, 512)
    assert run_calls == [(2, 0, True)]

    zero_out = mx.zeros((1, 1, 8, 512), dtype=mx.bfloat16)
    cuda_experts.decode(sw2, x, idx, lambda: zero_out)
    sw4 = _make_switch(hidden=512, inner=512, bits=4, gs=64)
    cuda_experts.decode(sw4, x, idx, lambda: zero_out)
    assert (8, 512, 512, 64, 2, 2) in cuda_experts.CHOICE
    assert (8, 512, 512, 64) in cuda_experts.CHOICE


@pytest.mark.skipif(not cuda_kernels.available(), reason="MLX's CUDA backend")
@pytest.mark.parametrize('bits', [2, 4])
@pytest.mark.parametrize('gs', [32, 64, 128])
@pytest.mark.parametrize('batch', [1, 3])
def test_cuda_experts_2bit_and_4bit_match_stock_bit_for_bit(bits, gs, batch, monkeypatch):
    """On CUDA, `cuda_experts.decode` matches `SwitchGLU` bit for bit for both 2-bit and 4-bit weights."""
    for name in ('CHOICE', 'CHECKED'):
        monkeypatch.setattr(cuda_experts, name, {})
    monkeypatch.setattr(cuda_experts, 'STATS', dict(served=0, checked=0, mismatched=0))
    monkeypatch.setattr(cuda_experts, 'ON', [True])

    sw = _make_switch(hidden=512, inner=256, num_experts=12, bits=bits, gs=gs)
    for step in range(cuda_experts.CHECKS + 2):
        mx.random.seed(100 + step)
        x = (mx.random.normal((batch, 1, 512)) * 0.5).astype(mx.bfloat16)
        idx = mx.random.randint(0, 12, (batch, 1, 8), dtype=mx.uint32)
        want = sw(x, idx)
        got = cuda_experts.decode(sw, x, idx, lambda: sw(x, idx))
        mx.eval(want, got)
        assert mx.array_equal(got, want).item()
    assert cuda_experts.STATS['mismatched'] == 0
    assert cuda_experts.STATS['served'] == 2


def test_cuda_head_ple_8bit_sm120_and_v41_gpu_guard(monkeypatch):
    """`cuda_head` & `cuda_ple_layer` accept 4- and 8-bit affine projections, `cuda_v41._gpu()` rejects CPU-rank
    execution, and `cuda_qmv.qmv_fp8()` delegates cleanly by `cuda_ple.compute_major()`."""
    from types import SimpleNamespace
    from quanteeze import cuda_head, cuda_ple, cuda_ple_layer, cuda_qmv, cuda_v41

    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)

    class Projection:
        def __init__(self, shape, sshape, bits=8, gs=64):
            self.mode = 'affine'
            self.bits = bits
            self.group_size = gs
            self.weight = mx.zeros(shape, mx.uint32)
            self.scales = mx.zeros(sshape, mx.bfloat16)
            self.biases = mx.zeros(sshape, mx.bfloat16)

        def get(self, name, default=None):
            return getattr(self, name, default)

        def __contains__(self, name):
            return hasattr(self, name)

    # rename type name to QuantizedLinear for cuda_head.takes check
    QuantizedLinear = type('QuantizedLinear', (Projection,), {})
    x = mx.zeros((1, 1, 512), mx.bfloat16)
    head8 = QuantizedLinear((128, 128), (128, 8), bits=8, gs=64)
    head2 = QuantizedLinear((128, 32), (128, 8), bits=2, gs=64)
    assert cuda_head.takes(head8, x)
    assert not cuda_head.takes(head2, x)

    assert 'qz_qmv8_row<' in cuda_head.ROWS_SOURCE
    assert 'qz_qmv8_row<' in cuda_ple_layer._PROJECT
    assert cuda_ple_layer._quantized(head8, 128, 512)
    assert not cuda_ple_layer._quantized(head2, 128, 512)

    assert isinstance(cuda_ple.compute_major(), int)
    assert cuda_v41._gpu() is (mx.default_device() == mx.gpu)
    monkeypatch.setattr(mx, 'default_device', lambda: mx.cpu)
    assert not cuda_v41._gpu()
    assert not cuda_v41.available()

    # qmv_fp8 predicate & dispatch path
    w = mx.zeros((64, 512 // 4), mx.uint32)
    s = mx.zeros((64, 512 // 32), mx.uint8)
    assert cuda_qmv.takes_fp8(x.reshape(1, -1), w, s, None, True, 32, 8, 'mxfp8')
    monkeypatch.setattr(cuda_ple, 'compute_major', lambda: 8)
    sentinel = mx.ones((1, 64), mx.bfloat16)
    monkeypatch.setattr(cuda_qmv, '_ORIGINAL', [lambda *a, **k: sentinel])
    assert cuda_qmv.qmv_fp8(x.reshape(1, -1), w, s) is sentinel
