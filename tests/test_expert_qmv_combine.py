# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Regression tests for `expert_qmv.install_switch` preserving `SwitchGLU._combine`."""
import copy

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel
from mlx_vlm.models.switch_layers import SwitchGLU
from mlx_vlm.sample_utils import make_sampler
from mlx_vlm.speculative.drafters.glm5_next_mtp.config import Glm5NextMTPConfig
from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel
from mlx_vlm.speculative.mtp import _mtp_rounds_batch

from quanteeze import expert_qmv, mtp, requant, verify_qmv


def _quantized_switch(bits=4, group_size=64, input_dims=128, hidden_dims=64, num_experts=8):
    sw = SwitchGLU(input_dims=input_dims, hidden_dims=hidden_dims, num_experts=num_experts, bias=False)
    for proj_name in ('gate_proj', 'up_proj', 'down_proj'):
        proj = getattr(sw, proj_name).to_quantized(group_size=group_size, bits=bits)
        proj.scales = proj.scales.astype(mx.bfloat16)
        if getattr(proj, 'biases', None) is not None:
            proj.biases = proj.biases.astype(mx.bfloat16)
        setattr(sw, proj_name, proj)
    mx.eval(sw.parameters())
    return sw


def test_install_switch_preserves_combine_weights_shared_and_residual(monkeypatch):
    """Installed `SwitchGLU` applies `_combine(routed, weights, shared, residual)` at T=1, T=2, and fallback."""
    mx.random.seed(11)
    ref = _quantized_switch(bits=4)
    wrapped = copy.deepcopy(ref)
    expert_qmv.install_switch(wrapped)

    for t in (1, 2):
        x = mx.random.normal((1, t, 128)).astype(mx.bfloat16)
        indices = mx.broadcast_to(mx.array([[[0, 2, 5, 7]]], dtype=mx.uint32), (1, t, 4))
        weights = mx.softmax(mx.random.normal((1, t, 4)), axis=-1).astype(mx.bfloat16)
        shared = mx.random.normal((1, t, 128)).astype(mx.bfloat16)

        exp_pos = ref(x, indices, weights, shared)
        act_pos = wrapped(x, indices, weights, shared)
        act_kw = wrapped(x, indices, weights=weights, shared=shared, residual=None)
        act_mix = wrapped(x, indices, weights, shared=shared)
        mx.eval(exp_pos, act_pos, act_kw, act_mix)

        assert act_pos.shape == (1, t, 128)
        assert mx.array_equal(exp_pos, act_pos)
        assert mx.array_equal(exp_pos, act_kw)
        assert mx.array_equal(exp_pos, act_mix)

    seen_residuals = []
    orig_combine = SwitchGLU._combine

    def tracking_combine(routed, weights=None, shared=None, residual=None):
        seen_residuals.append(residual)
        out = orig_combine(routed, weights, shared, residual=None)
        if residual is not None:
            out = out + residual[0]
        return out

    monkeypatch.setattr(type(wrapped), '_combine', staticmethod(tracking_combine))
    x1 = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    idx1 = mx.array([[[1, 3, 4, 6]]], dtype=mx.uint32)
    w1 = mx.softmax(mx.random.normal((1, 1, 4)), axis=-1).astype(mx.bfloat16)
    sh1 = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    res_bias = mx.ones((1, 1, 128), dtype=mx.bfloat16)

    act_res = wrapped(x1, idx1, w1, sh1, residual=(res_bias,))
    exp_res = tracking_combine(wrapped(x1, idx1), w1, sh1, residual=(res_bias,))
    mx.eval(act_res, exp_res)
    assert len(seen_residuals) >= 1 and seen_residuals[0] is not None
    assert mx.array_equal(act_res, exp_res)


def test_install_switch_stock_fallback_preserves_combine(monkeypatch):
    """Non-native quantized `SwitchGLU` fallback (`stock`) also preserves `_combine`."""
    mx.random.seed(23)
    ref = _quantized_switch(bits=2)
    wrapped = copy.deepcopy(ref)
    expert_qmv.install_switch(wrapped)

    t = verify_qmv.MAX_TOKENS + 2
    x = mx.random.normal((1, t, 128)).astype(mx.bfloat16)
    indices = mx.broadcast_to(mx.array([[[0, 1, 2, 3]]], dtype=mx.uint32), (1, t, 4))
    weights = mx.softmax(mx.random.normal((1, t, 4)), axis=-1).astype(mx.bfloat16)
    shared = mx.random.normal((1, t, 128)).astype(mx.bfloat16)

    exp = ref(x, indices, weights, shared)
    monkeypatch.setattr(requant, 'native', lambda bits, mode='affine': False)
    act = wrapped(x, indices, weights, shared)
    mx.eval(exp, act)
    assert act.shape == (1, t, 128)
    assert mx.array_equal(exp, act)


def test_glm5_next_mtp_installed_switches_return_2d_draft_tokens():
    """Tiny quantized `Glm5NextMTPDraftModel` produces 2D drafts and runs `_mtp_rounds_batch`."""
    mx.random.seed(31)
    tc = TextConfig(
        model_type='glm5_next_text',
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=0,
        v_head_dim=64,
        index_n_heads=2,
        index_head_dim=64,
        index_topk=8,
        index_kpool=4,
        linear_num_heads=4,
        linear_head_dim=64,
        linear_conv_kernel_dim=4,
        layer_types=['deepseek_sparse_attention', 'deepseek_sparse_attention'],
        mlp_layer_types=['sparse', 'sparse'],
        indexer_types=['full', 'full'],
    )
    lm = LanguageModel(tc)
    nn.quantize(
        lm,
        group_size=64,
        bits=4,
        class_predicate=lambda p, m: hasattr(m, 'to_quantized') and lm.quant_predicate(p, m),
    )
    dm = Glm5NextMTPDraftModel(Glm5NextMTPConfig(text_config=tc))
    nn.quantize(dm, group_size=64, bits=4, class_predicate=lambda _p, m: hasattr(m, 'to_quantized'))
    mtp.lower(dm, 2)
    for sw in mtp.switches(dm):
        expert_qmv.install_switch(sw)
    mx.eval(lm.parameters(), dm.parameters())

    cache = lm.make_cache()
    pt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    h = lm.model(pt, cache=cache)
    fb = mx.argmax(lm.lm_head(h)[:, -1, :], axis=-1)
    sampler = make_sampler(temp=0.0)

    dm.reset(lm, left_padding=[0])
    dm.prefill_from_target_hidden(pt, h, fb, sampler, mx.int32, greedy=True, left_padding=[0])
    dt = dm.draft_block(fb, h[:, -1:, :], None, 2, sampler, mx.int32, greedy=True)
    mx.eval(dt)
    assert dt.ndim == 2
    assert dt.shape == (1, 1)

    cache2 = lm.make_cache()
    h2 = lm.model(pt, cache=cache2)
    fb2 = mx.argmax(lm.lm_head(h2)[:, -1, :], axis=-1)
    rounds = _mtp_rounds_batch(
        lm,
        dm,
        cache2,
        h2,
        {},
        prompt_tokens=pt,
        first_bonus=fb2,
        max_tokens=5,
        sampler=sampler,
        draft_block_size=2,
        greedy_sampling=True,
    )
    step1 = next(rounds)
    step2 = next(rounds)
    rounds.close()
    assert len(step1[0]) == 1 and isinstance(step1[0][0], int)
    assert len(step2[0]) == 1 and isinstance(step2[0][0], int)
