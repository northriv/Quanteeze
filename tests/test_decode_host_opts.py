# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for decode host/kernel optimizations (`expert_qmv._plan`, `linear_attn._in_proj`/Metal `SLIDE`,
`QUANTEEZE_EVAL_LAYERS=auto`, and tiny `qwen4_exp` / `qwen3_5_moe` decode parity with `metal_router`)."""
import gc
from types import SimpleNamespace
import weakref

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.config import (
    ModelConfig as Q4ModelConfig,
    TextConfig as Q4TextConfig,
    VisionConfig as Q4VisionConfig,
)
from mlx_vlm.models.qwen4_exp.language import (
    LanguageModel as Q4LanguageModel,
    Qwen4ExpBatchInvariantForward,
    Qwen4ExpGatedDeltaNet,
)
import numpy as np
import pytest

from quanteeze import control, cuda_kernels, expert_qmv, kernel_checks, linear_attn, metal_router, per_thread, variables
from quanteeze.arch import qwen3_5_moe, qwen4_exp


def _make_quantized_switch(d=512, i=640, num_experts=16, bits=4, gs=64):
    mx.random.seed(11)
    sw = SwitchGLU(input_dims=d, hidden_dims=i, num_experts=num_experts, bias=False)
    sw.set_dtype(mx.bfloat16)
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        setattr(sw, name, getattr(sw, name).to_quantized(group_size=gs, bits=bits, mode='affine'))
    mx.eval(sw.parameters())
    return sw


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal backend required')
def test_expert_qmv_plan_caches_weakref_gc_and_invalidation(monkeypatch):
    monkeypatch.setattr(expert_qmv, 'ON', [True])
    monkeypatch.setattr(expert_qmv, 'FUSED', [True])
    monkeypatch.setattr(expert_qmv, 'VERIFY', [True])

    sw = _make_quantized_switch()
    rng = np.random.default_rng(11)
    x = (mx.random.normal((1, 1, 512)) * 0.25).astype(mx.bfloat16)
    indices = mx.array(rng.integers(0, 16, size=(1, 1, 4)), dtype=mx.uint32)
    mx.eval(x, indices)

    plan = expert_qmv._plan(sw, x)
    assert plan is not None and plan[0] == 640 and len(plan[1]) == 4
    assert expert_qmv._plan(sw, x) is plan
    assert mx.array_equal(expert_qmv.fused(sw, x, indices), expert_qmv.stock(sw, x, indices)).item()

    # Projection replacement: weakref must allow old projection to be collected and recompute plan.
    old_gp_ref = weakref.ref(sw.gate_proj)
    sw.gate_proj = _make_quantized_switch().gate_proj
    gc.collect()
    assert old_gp_ref() is None
    plan_after_proj = expert_qmv._plan(sw, x)
    assert plan_after_proj is not plan and plan_after_proj == plan
    assert mx.array_equal(expert_qmv.fused(sw, x, indices), expert_qmv.stock(sw, x, indices)).item()

    # Weight array replacement on an existing projection invalidates plan.
    new_up = _make_quantized_switch().up_proj
    sw.up_proj.weight = mx.contiguous(new_up.weight * 1)
    mx.eval(sw.up_proj.weight)
    plan_after_w = expert_qmv._plan(sw, x)
    assert plan_after_w is not plan_after_proj

    # Shape, dtype, and VERIFY toggles invalidate plan cleanly.
    x_verify = (mx.random.normal((1, 2, 512)) * 0.25).astype(mx.bfloat16)
    indices_v = mx.array(rng.integers(0, 16, size=(1, 2, 4)), dtype=mx.uint32)
    mx.eval(x_verify, indices_v)
    plan_v = expert_qmv._plan(sw, x_verify)
    assert plan_v is not None and plan_v is not plan_after_w
    assert mx.array_equal(expert_qmv.fused(sw, x_verify, indices_v), expert_qmv.stock(sw, x_verify, indices_v)).item()

    monkeypatch.setattr(expert_qmv, 'VERIFY', [False])
    assert expert_qmv._plan(sw, x_verify) is None
    assert expert_qmv._plan(sw, x.astype(mx.float16)) is None


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal backend required')
def test_linear_attn_in_proj_cache_and_invalidation():
    linear_attn.share_fused_linears()
    mx.random.seed(1)
    cfg = SimpleNamespace(
        model_type='qwen4_exp',
        hidden_size=256,
        rms_norm_eps=1e-6,
        hidden_act='silu',
        output_gate_type='sigmoid',
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    layer = Qwen4ExpGatedDeltaNet(cfg)
    layer.set_dtype(mx.bfloat16)
    layer.A_log = mx.random.normal((4,)).astype(mx.float32)
    layer.dt_bias = (mx.random.normal((4,)) * 0.5).astype(mx.bfloat16)
    nn.quantize(layer, group_size=32, bits=4, class_predicate=lambda p, m: isinstance(m, nn.Linear) and 'in_proj_' in p)
    layer.eval()
    mx.eval(layer.parameters())

    forward = Qwen4ExpBatchInvariantForward()
    x = (mx.random.normal((1, 1, 256)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    want = forward._linears(tuple(layer[n] for n in linear_attn.IN_PROJ), x)
    first = linear_attn._in_proj(forward, layer, x)
    assert '_quanteeze_in_proj' in layer.__dict__
    again = linear_attn._in_proj(forward, layer, x)
    for a, b, c in zip(want, first, again):
        assert a.shape == b.shape == c.shape
        assert mx.array_equal(a, b).item() and mx.array_equal(a, c).item()

    other = nn.QuantizedLinear.from_linear(
        nn.Linear(256, layer.in_proj_z.weight.shape[0], bias=False), group_size=32, bits=4
    )
    other.set_dtype(mx.bfloat16)
    mx.eval(other.parameters())
    layer.in_proj_z = other

    got = linear_attn._in_proj(forward, layer, x)
    want_new = forward._linears(tuple(layer[n] for n in linear_attn.IN_PROJ), x)
    assert all(mx.array_equal(a, b).item() for a, b in zip(got, want_new))
    assert not mx.array_equal(got[1], first[1]).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal backend required')
def test_linear_attn_metal_conv_slide_bit_for_bit(monkeypatch):
    la_choice, la_checked = kernel_checks.tables('test_linear_attn_slide')
    monkeypatch.setattr(linear_attn, 'CHOICE', la_choice)
    monkeypatch.setattr(linear_attn, 'CHECKED', la_checked)
    monkeypatch.setattr(linear_attn, 'STATS', {k: 0 for k in linear_attn.STATS})
    monkeypatch.setattr(linear_attn, 'ON', [True])
    monkeypatch.setattr(linear_attn, 'SLIDE', [True])

    mx.random.seed(3)
    cfg = SimpleNamespace(
        model_type='qwen4_exp',
        hidden_size=256,
        rms_norm_eps=1e-6,
        hidden_act='silu',
        output_gate_type='sigmoid',
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    layer = Qwen4ExpGatedDeltaNet(cfg)
    layer.set_dtype(mx.bfloat16)
    layer.A_log = mx.random.normal((4,)).astype(mx.float32)
    layer.dt_bias = (mx.random.normal((4,)) * 0.5).astype(mx.bfloat16)
    layer.eval()
    mx.eval(layer.parameters())

    ch = 2 * 2 * 128 + 4 * 128
    state = (mx.random.normal((1, 3, ch)) * 0.5).astype(mx.bfloat16)
    x_conv = (mx.random.normal((1, 1, ch)) * 0.5).astype(mx.bfloat16)
    forward = Qwen4ExpBatchInvariantForward()

    joined = mx.concatenate([state, x_conv], axis=1)
    want_qkv = linear_attn._conv_original(forward, layer, joined)
    want_slid = mx.contiguous(joined[:, -3:])
    y, got_slid = linear_attn.conv(layer, state, x_conv, sliding=True)
    got_qkv = linear_attn._qkv(layer, y)
    assert all(kernel_checks.same_bits(a, b) for a, b in zip(got_qkv, want_qkv))
    assert kernel_checks.same_bits(got_slid, want_slid)

    cache_mine = ArraysCache(size=2)
    cache_ref = ArraysCache(size=2)
    cache_mine[0], cache_mine[1] = mx.array(state), mx.zeros((1, 4, 128, 128), mx.float32)
    cache_ref[0], cache_ref[1] = mx.array(state), mx.zeros((1, 4, 128, 128), mx.float32)
    for step in range(linear_attn.CHECKS + 3):
        mx.random.seed(50 + step)
        inp = (mx.random.normal((1, 1, 256)) * 0.5).astype(mx.bfloat16)
        got = linear_attn.decode(forward, layer, inp, cache_mine)
        want = Qwen4ExpBatchInvariantForward._gated_delta(forward, layer, inp, None, cache_ref)
        assert kernel_checks.same_bits(got, want)
        assert kernel_checks.same_bits(cache_mine[0], cache_ref[0])
        assert kernel_checks.same_bits(cache_mine[1], cache_ref[1])
    assert linear_attn.STATS['slide'] == 3
    assert linear_attn.STATS['mismatched'] == 0


def test_eval_layers_auto_policy_and_settings(monkeypatch):
    assert variables.SETTINGS['QUANTEEZE_EVAL_LAYERS'].default == 'auto'
    assert variables.SETTINGS['QUANTEEZE_METAL_ROUTER'].default == 'on'
    assert control.CHOICES['QUANTEEZE_METAL_ROUTER'] == ('on', 'off')

    assert control.check('QUANTEEZE_EVAL_LAYERS', 'auto') is None
    assert control.check('QUANTEEZE_EVAL_LAYERS', '4') is None
    assert control.check('QUANTEEZE_EVAL_LAYERS', '12') is None
    assert control.check('QUANTEEZE_EVAL_LAYERS', '0') is not None
    assert control.check('QUANTEEZE_EVAL_LAYERS', '-1') is not None
    assert control.check('QUANTEEZE_EVAL_LAYERS', 'bogus') is not None

    monkeypatch.setattr(qwen4_exp, 'EVAL_LAYERS', list(qwen4_exp.EVAL_LAYERS))
    monkeypatch.setattr(qwen4_exp, 'COMPILE_STATS', dict(compiled=0, hyper=0, refused=0))
    eval_fn = qwen4_exp._BatchInvariantForward._eval_layers
    one_tok = mx.zeros((1, 1, 8))
    many_tok = mx.zeros((1, 4, 8))

    qwen4_exp.EVAL_LAYERS[0] = 0
    monkeypatch.setattr(cuda_kernels, 'available', lambda: False)
    assert eval_fn(one_tok) == qwen4_exp.EVAL_LAYERS_ON_METAL == 4
    assert eval_fn(many_tok) == 1

    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
    assert eval_fn(one_tok) == qwen4_exp.EVAL_LAYERS_ON_CUDA == 12
    assert eval_fn(many_tok) == 1

    qwen4_exp.EVAL_LAYERS[0] = 6
    assert eval_fn(one_tok) == 6
    assert eval_fn(many_tok) == 1


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal backend required')
def test_tiny_qwen4_exp_and_qwen3_5_moe_decode_parity_with_metal_router(monkeypatch):
    from mlx_vlm.models.qwen3_5_moe.config import (
        ModelConfig as Q35ModelConfig,
        TextConfig as Q35TextConfig,
        VisionConfig as Q35VisionConfig,
    )
    from mlx_vlm.models.qwen3_5_moe.language import LanguageModel as Q35LanguageModel
    from quanteeze.arch.qwen4_exp import _weighted_graph  # ensure compiled graph exists

    assert callable(_weighted_graph)

    def _make_qwen4_text_config(num_layers=4):
        return Q4TextConfig(
            model_type='qwen4_exp',
            hidden_size=128,
            num_hidden_layers=num_layers,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            linear_num_value_heads=4,
            linear_num_key_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            num_experts=16,
            num_experts_per_tok=4,
            shared_expert_intermediate_size=128,
            moe_intermediate_size=128,
            rms_norm_eps=1e-6,
            vocab_size=512,
            max_position_embeddings=256,
            hc_count=4,
            hc_lowrank=64,
            full_attention_interval=3,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=64,
            indexer_budget=64,
            indexer_compress_ratio=4,
            rope_parameters={
                'type': 'default',
                'mrope_section': [11, 11, 10],
                'rope_theta': 10_000,
                'partial_rotary_factor': 0.5,
            },
        )

    # Isolate kernel check tables for clean stats observation.
    m_choice, m_checked = kernel_checks.tables('test_e2e_metal_router')
    w_choice, w_checked = kernel_checks.tables('test_e2e_metal_weighted')
    eq_choice, eq_checked = kernel_checks.tables('test_e2e_expert_qmv')
    la_choice, la_checked = kernel_checks.tables('test_e2e_linear_attn')
    monkeypatch.setattr(metal_router, 'CHOICE', m_choice)
    monkeypatch.setattr(metal_router, 'CHECKED', m_checked)
    monkeypatch.setattr(metal_router, 'STATS', dict(served=0, checked=0, mismatched=0))
    monkeypatch.setattr(metal_router._Weighed, 'CHOICE', w_choice)
    monkeypatch.setattr(metal_router._Weighed, 'CHECKED', w_checked)
    monkeypatch.setattr(metal_router._Weighed, 'STATS', dict(served=0, checked=0, mismatched=0))
    monkeypatch.setattr(expert_qmv, 'CHOICE', eq_choice)
    monkeypatch.setattr(expert_qmv, 'CHECKED', eq_checked)
    monkeypatch.setattr(expert_qmv, 'STATS', dict(served=0, fused=0, checked=0, mismatched=0))
    monkeypatch.setattr(linear_attn, 'CHOICE', la_choice)
    monkeypatch.setattr(linear_attn, 'CHECKED', la_checked)
    monkeypatch.setattr(linear_attn, 'STATS', {k: 0 for k in linear_attn.STATS})
    monkeypatch.setattr(qwen4_exp, 'EVAL_LAYERS', list(qwen4_exp.EVAL_LAYERS))
    monkeypatch.setattr(qwen4_exp, 'COMPILE_STATS', dict(compiled=0, hyper=0, refused=0))

    # 1. Tiny qwen4_exp model
    mx.random.seed(21)
    q4_cfg = _make_qwen4_text_config(num_layers=4)
    q4_cfg.num_experts = 16
    q4_cfg.num_experts_per_tok = 4
    q4_cfg.moe_intermediate_size = 128
    q4_cfg.shared_expert_intermediate_size = 128
    q4_lm = Q4LanguageModel(
        q4_cfg, Q4ModelConfig(text_config=q4_cfg, vision_config=Q4VisionConfig(), model_type=q4_cfg.model_type)
    )
    q4_lm.set_dtype(mx.bfloat16)
    for layer in q4_lm.model.layers:
        sw = layer.mlp.switch_mlp
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(sw, proj, getattr(sw, proj).to_quantized(group_size=64, bits=4))
    q4_lm.eval()
    mx.eval(q4_lm.parameters())
    qwen4_exp.install(q4_lm)

    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    steps = [mx.array([[4 + i]], dtype=mx.int32) for i in range(6)]

    def run_q4(router_on, eval_layers):
        per_thread.PASS.memo.clear()
        qwen4_exp.COMPILE_STATS.update(compiled=0, hyper=0, refused=0)
        monkeypatch.setattr(metal_router, 'ON', [router_on])
        qwen4_exp.EVAL_LAYERS[0] = eval_layers
        cache = q4_lm.make_cache()
        mx.eval(q4_lm(prompt, cache=cache).logits)
        outs = []
        for tok in steps:
            logits = q4_lm(tok, cache=cache).logits
            mx.eval(logits)
            outs.append(logits)
        return outs

    ref_q4 = run_q4(False, 12)
    got_q4 = run_q4(True, 0)
    for step_idx, (got, want) in enumerate(zip(got_q4, ref_q4)):
        assert kernel_checks.same_bits(got, want), f'qwen4_exp step {step_idx} diverged'
    assert metal_router.STATS['served'] > 0
    assert metal_router._Weighed.STATS['served'] > 0
    assert metal_router.STATS['mismatched'] == 0
    assert metal_router._Weighed.STATS['mismatched'] == 0
    assert qwen4_exp.COMPILE_STATS['compiled'] > 0
    assert qwen4_exp.COMPILE_STATS['refused'] == 0

    # 2. Tiny qwen3_5_moe model
    monkeypatch.setattr(metal_router, 'STATS', dict(served=0, checked=0, mismatched=0))
    monkeypatch.setattr(metal_router._Weighed, 'STATS', dict(served=0, checked=0, mismatched=0))
    mx.random.seed(22)
    q35_cfg = Q35TextConfig(
        model_type='qwen3_5_moe',
        hidden_size=128,
        num_hidden_layers=2,
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
        rope_parameters={'type': 'default', 'mrope_section': [3, 3, 2], 'rope_theta': 10000.0, 'partial_rotary_factor': 0.25},
    )
    q35_lm = Q35LanguageModel(
        q35_cfg, config=Q35ModelConfig(text_config=q35_cfg, vision_config=Q35VisionConfig(), model_type='qwen3_5_moe')
    )
    q35_lm.update(
        tree_map_with_path(
            lambda path, val: (val.astype(mx.bfloat16) if (val.dtype == mx.float32 and 'A_log' not in path) else val),
            q35_lm.parameters(),
        )
    )
    for layer in q35_lm.model.layers:
        sw = layer.mlp.switch_mlp
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(sw, proj, getattr(sw, proj).to_quantized(group_size=64, bits=4))
    q35_lm.eval()
    mx.eval(q35_lm.parameters())
    qwen3_5_moe.install(q35_lm)

    def run_q35(router_on):
        monkeypatch.setattr(metal_router, 'ON', [router_on])
        cache = q35_lm.make_cache()
        mx.eval(q35_lm(prompt, cache=cache).logits)
        outs = []
        for tok in steps:
            logits = q35_lm(tok, cache=cache).logits
            mx.eval(logits)
            outs.append(logits)
        return outs

    ref_q35 = run_q35(False)
    got_q35 = run_q35(True)
    for step_idx, (got, want) in enumerate(zip(got_q35, ref_q35)):
        assert kernel_checks.same_bits(got, want), f'qwen3_5_moe step {step_idx} diverged'
    assert metal_router.STATS['served'] > 0
    assert metal_router._Weighed.STATS['served'] > 0
    assert metal_router.STATS['mismatched'] == 0
    assert metal_router._Weighed.STATS['mismatched'] == 0
