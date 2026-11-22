# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for multi-request CUDA batch decode linear attention in `quanteeze.linear_attn` and `quanteeze.arch.qwen4_exp`."""
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.language import (
    Qwen4ExpBatchInvariantForward,
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpRMSNormGated,
)

from quanteeze import cuda_kernels, linear_attn
from quanteeze.arch import qwen4_exp

D, HK, HV = 128, 2, 4


def _make_layer(seed=0):
    mx.random.seed(seed)
    cfg = SimpleNamespace(
        model_type='qwen4_exp',
        hidden_size=256,
        rms_norm_eps=1e-6,
        hidden_act='silu',
        output_gate_type='sigmoid',
        linear_num_value_heads=HV,
        linear_num_key_heads=HK,
        linear_key_head_dim=D,
        linear_value_head_dim=D,
        linear_conv_kernel_dim=4,
    )
    layer = Qwen4ExpGatedDeltaNet(cfg)
    layer.set_dtype(mx.bfloat16)
    layer.A_log = mx.random.normal((HV,)).astype(mx.float32)
    layer.dt_bias = (mx.random.normal((HV,)) * 0.5).astype(mx.bfloat16)
    layer.eval()
    mx.eval(layer.parameters())
    return layer


def _make_cache(batch):
    cache = ArraysCache(size=2)
    cache[0] = (mx.random.normal((batch, 3, 2 * HK * D + HV * D)) * 0.5).astype(mx.bfloat16)
    cache[1] = mx.random.normal((batch, HV, D, D)).astype(mx.float32)
    mx.eval(cache[0], cache[1])
    return cache


def test_batch_kernel_sources_and_replaced_helper():
    """Batch CUDA shader sources inject per-request offsets cleanly and `_replaced` enforces exact match counts."""
    assert 'blockIdx.y' in linear_attn._CONV_BATCH_CUDA
    assert '/*STATE*/' not in linear_attn._CONV_BATCH_CUDA
    assert 'state[sb + k * CH + c]' in linear_attn._CONV_BATCH_CUDA
    assert 'state_out[sb + (TAPS - 2) * CH + c] = x[xb + c]' in linear_attn._STATE_BATCH
    assert 'BATCH * HV * DV' in linear_attn._DELTA_BATCH_CUDA
    assert 'bi = hf / HV' in linear_attn._DELTA_BATCH_CUDA
    assert 'g[hf]' in linear_attn._GATES_BATCH and 'beta[hf]' in linear_attn._GATES_BATCH
    assert 'a[hf]' in linear_attn._GATES_BATCH_COMPUTED and 'b[hf]' in linear_attn._GATES_BATCH_COMPUTED

    with pytest.raises(ValueError, match='met 0 times, not 1'):
        linear_attn._replaced('abc', [('missing', 'x', 1)])


def test_applies_batch_deltable_batch_and_forward_dispatch(monkeypatch):
    """`applies_batch` / `deltable_batch` gate B=2..8 on CUDA, and `_BatchInvariantForward` dispatches to `decode_batch`."""
    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
    monkeypatch.setattr(linear_attn, 'ON', [True])
    layer = _make_layer(1)

    for b in (2, 3, 8):
        x = mx.zeros((b, 1, 256), dtype=mx.bfloat16)
        cache = _make_cache(b)
        assert linear_attn.applies_batch(layer, x, None, cache)
        q = mx.zeros((b, 1, HK, D), dtype=mx.bfloat16)
        k = mx.zeros((b, 1, HK, D), dtype=mx.bfloat16)
        v = mx.zeros((b, 1, HV, D), dtype=mx.bfloat16)
        assert linear_attn.deltable_batch(q, k, v, cache[1], layer)

    for b in (1, 9):
        x = mx.zeros((b, 1, 256), dtype=mx.bfloat16)
        assert not linear_attn.applies_batch(layer, x, None, _make_cache(b))

    x3 = mx.zeros((3, 1, 256), dtype=mx.bfloat16)
    c3 = _make_cache(3)
    assert not linear_attn.applies_batch(layer, x3.astype(mx.float16), None, c3)
    assert not linear_attn.applies_batch(layer, x3, mx.zeros((1,)), c3)

    monkeypatch.setattr(linear_attn, 'ON', [False])
    assert not linear_attn.applies_batch(layer, x3, None, c3)
    monkeypatch.setattr(linear_attn, 'ON', [True])

    sentinel = object()
    called = []
    monkeypatch.setattr(
        linear_attn,
        'decode_batch',
        lambda fwd, lyr, inp, ch: called.append((fwd, lyr, inp.shape[0])) or sentinel,
    )
    fwd = qwen4_exp._BatchInvariantForward.__new__(qwen4_exp._BatchInvariantForward)
    assert fwd._gated_delta(layer, x3, None, c3) is sentinel
    assert called == [(fwd, layer, 3)]


@pytest.mark.skipif(not cuda_kernels.available(), reason="MLX's CUDA backend")
@pytest.mark.parametrize('n', [2, 3, 8])
def test_a_batchs_conv_is_each_requests_decode_bit_for_bit(n):
    """`conv_batch` matches mlx-vlm batch ops and single-request `conv` bit for bit, sliding windows cleanly."""
    layer = _make_layer(20 + n)
    c = layer.conv_dim
    mx.random.seed(20 + n)
    state = (mx.random.normal((n, 3, c)) * 1.5).astype(mx.bfloat16)
    x = (mx.random.normal((n, 1, c)) * 1.5).astype(mx.bfloat16)
    window = mx.concatenate([state, x], axis=1)
    want = linear_attn._conv_original_batch(Qwen4ExpBatchInvariantForward(), layer, window, n)
    got = linear_attn._qkv_batch(layer, linear_attn.conv_batch(layer, state, x), n)
    assert all(mx.array_equal(a, b).item() for a, b in zip(got, want))
    y, slid = linear_attn.conv_batch(layer, state, x, sliding=True)
    assert all(mx.array_equal(a, b).item() for a, b in zip(linear_attn._qkv_batch(layer, y, n), want))
    assert mx.array_equal(slid, window[:, 1:]).item()
    for i in range(n):
        one = linear_attn._qkv(layer, linear_attn.conv(layer, state[i : i + 1], x[i : i + 1]))
        assert all(mx.array_equal(a[i : i + 1], b).item() for a, b in zip(got, one))


@pytest.mark.skipif(not cuda_kernels.available(), reason="MLX's CUDA backend")
@pytest.mark.parametrize('n', [2, 3, 8])
def test_a_batchs_delta_step_and_norm_are_each_requests_bit_for_bit(n):
    """`delta_batch` (precomputed and fused gating) and `gated_norm` match mlx-vlm and single-request kernels."""
    from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta, gated_delta_update

    mx.random.seed(40 + n)
    q, k = (mx.random.normal((n, 1, HK, D)) for _ in range(2))
    q = (q / mx.linalg.norm(q, axis=-1, keepdims=True) * D ** -0.5).astype(mx.bfloat16)
    k = (k / mx.linalg.norm(k, axis=-1, keepdims=True)).astype(mx.bfloat16)
    v = mx.random.normal((n, 1, HV, D)).astype(mx.bfloat16)
    a, b = (mx.random.normal((n, 1, HV)).astype(mx.bfloat16) for _ in range(2))
    a_log = mx.random.normal((HV,)).astype(mx.float32)
    state = mx.random.normal((n, HV, D, D)).astype(mx.float32)
    for bias in ((mx.random.normal((HV,)) * 0.5).astype(mx.bfloat16), mx.random.normal((HV,)) * 0.5):
        layer = SimpleNamespace(A_log=a_log, dt_bias=bias)
        want = gated_delta_update(q, k, v, a, b, a_log, bias, state=state)
        for got in (
            linear_attn.delta_batch(q, k, v, _compute_g_beta(a_log, a, b, bias), state),
            linear_attn.delta_batch(q, k, v, (a, b), state, layer),
        ):
            assert all(mx.array_equal(x, y).item() for x, y in zip(got, want))
            for i in range(n):
                one = linear_attn.delta(
                    q[i : i + 1],
                    k[i : i + 1],
                    v[i : i + 1],
                    *_compute_g_beta(a_log, a[i : i + 1], b[i : i + 1], bias),
                    state[i : i + 1],
                )
                assert mx.array_equal(got[0][i : i + 1], one[0]).item()
                assert mx.array_equal(got[1][i : i + 1], one[1]).item()
    norm = Qwen4ExpRMSNormGated(D, eps=1e-6, activation='sigmoid')
    norm.weight = (mx.random.normal((D,)) * 0.3 + 1).astype(mx.bfloat16)
    z = (mx.random.normal((n, 1, HV, D)) * 3).astype(mx.bfloat16)
    assert mx.array_equal(linear_attn.gated_norm(norm, v, z), norm(v, z)).item()


@pytest.mark.skipif(not cuda_kernels.available(), reason="MLX's CUDA backend")
def test_decode_batch_matches_stock_forward_over_checks_and_serves(monkeypatch):
    """Multi-step `decode_batch` on `Qwen4ExpGatedDeltaNet` matches `Qwen4ExpBatchInvariantForward` bit for bit."""
    for name in ('CHOICE', 'CHECKED'):
        monkeypatch.setattr(linear_attn, name, {})
    monkeypatch.setattr(linear_attn, 'STATS', {k: 0 for k in linear_attn.STATS})
    monkeypatch.setattr(linear_attn, 'ON', [True])
    monkeypatch.setattr(linear_attn, 'DELTA', [True])

    layer = _make_layer(77)
    ref_fwd = Qwen4ExpBatchInvariantForward()
    test_fwd = qwen4_exp._BatchInvariantForward.__new__(qwen4_exp._BatchInvariantForward)
    cache_want = _make_cache(3)
    cache_got = ArraysCache(size=2)
    cache_got[0], cache_got[1] = mx.array(cache_want[0]), mx.array(cache_want[1])
    mx.eval(cache_got[0], cache_got[1])

    before = dict(linear_attn.STATS)
    for step in range(linear_attn.CHECKS + 2):
        mx.random.seed(300 + step)
        x = (mx.random.normal((3, 1, 256)) * 0.5).astype(mx.bfloat16)
        want = ref_fwd._gated_delta(layer, x, None, cache_want)
        got = test_fwd._gated_delta(layer, x, None, cache_got)
        mx.eval(want, got, cache_want[0], cache_want[1], cache_got[0], cache_got[1])
        assert mx.array_equal(got, want).item()
        assert mx.array_equal(cache_got[0], cache_want[0]).item()
        assert mx.array_equal(cache_got[1], cache_want[1]).item()

    assert linear_attn.STATS['mismatched'] == before['mismatched']
    assert linear_attn.STATS['slide_batch'] + linear_attn.STATS['conv_batch'] > 0
    assert linear_attn.STATS['gating_batch'] + linear_attn.STATS['delta_batch'] > 0
    assert linear_attn.STATS['norm_batch'] > 0
