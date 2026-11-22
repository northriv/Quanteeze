# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from quanteeze import control, home, mtp, variables
from quanteeze.arch import qwen4_exp


def _make_qwen4_text_config(num_layers=6):
    from mlx_vlm.models.qwen4_exp.config import TextConfig

    return TextConfig(
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
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
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


def _save_dflash2_checkpoint(dest, target_layer_ids=(1, 3, 5), num_target_layers=6):
    from mlx_vlm.speculative.drafters.dflash2 import Model as DFlash2DraftModel
    from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config

    dest.mkdir(parents=True, exist_ok=True)
    raw_cfg = {
        'architectures': ['DFlash2DraftModel'],
        'model_type': 'qwen3',
        'hidden_size': 128,
        'intermediate_size': 256,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'num_key_value_heads': 2,
        'head_dim': 64,
        'vocab_size': 512,
        'max_position_embeddings': 256,
        'num_target_layers': num_target_layers,
        'layer_types': ['sliding_attention', 'sliding_attention'],
        'sliding_window': 64,
        'rms_norm_eps': 1e-6,
        'tie_word_embeddings': False,
        'dflash_config': {
            'block_size': 4,
            'mask_token_id': 511,
            'target_layer_ids': list(target_layer_ids),
            'conv_kernel_size': 2,
            'conv_group_size': 16,
            'selector_rank': 64,
            'selector_top_k': 8,
        },
    }
    (dest / 'config.json').write_text(json.dumps(raw_cfg), encoding='utf-8')
    cfg = DFlash2Config.from_dict(raw_cfg)
    draft = DFlash2DraftModel(cfg)
    draft.apply(lambda a: a.astype(mx.bfloat16))
    mx.eval(draft.parameters())
    weights = {f'model.{k}': v for k, v in dict(nn.utils.tree_flatten(draft.parameters())).items()}
    mx.save_safetensors(str(dest / 'model.safetensors'), weights)
    return dest


def test_mtp_block_and_presets_and_drafter_filter(tmp_path):
    assert 'qwen4_exp' not in mtp.BLOCKS
    assert mtp.block_for('auto', 'qwen4_exp') is None
    assert 'dflash2' in control.CHOICES['QUANTEEZE_MTP']
    assert 'QUANTEEZE_DRAFTER' in variables.SETTINGS
    assert 'dflash2' in home.PRESETS['mtp']
    assert home.PRESETS['mtp']['dflash2']['values']['QUANTEEZE_MTP'] == 'dflash2'

    saved_override = mtp.OVERRIDE_BLOCK[0]
    try:
        auto_drafter = SimpleNamespace(config=SimpleNamespace(block_size=2, model_type='qwen4_exp_mtp'))
        assert mtp.set_block(auto_drafter, 4) == 4
        assert mtp.OVERRIDE_BLOCK[0] == 4
        assert auto_drafter.config.runtime_block_size is None
        mtp.disable()
        assert mtp.OVERRIDE_BLOCK[0] is None

        dflash2_drafter = SimpleNamespace(
            config=SimpleNamespace(block_size=8, model_type='dflash2', target_layer_ids=[5, 15, 25, 35, 45]),
            candidate_selector=object(),
        )
        assert mtp.set_block(dflash2_drafter, 5) == 5
        assert dflash2_drafter.config.runtime_block_size == 5
        assert mtp.OVERRIDE_BLOCK[0] is None
        assert mtp.set_block(dflash2_drafter, 16) == 8
        assert dflash2_drafter.config.runtime_block_size == 8
    finally:
        mtp.OVERRIDE_BLOCK[0] = saved_override

    d2_dir = tmp_path / 'dflash2_only'
    d2_dir.mkdir()
    (d2_dir / 'config.json').write_text(json.dumps({
        'architectures': ['DFlash2DraftModel'],
        'model_type': 'qwen3',
        'dflash_config': {'block_size': 8},
    }))
    assert home._drafter(d2_dir) is True

    target_dir = tmp_path / 'target_only'
    target_dir.mkdir()
    (target_dir / 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp'}))
    assert home._drafter(target_dir) is False


def test_qwen4_exp_dflash2_hc_mean_capture_verify_and_quant(tmp_path):
    from mlx_vlm.models.qwen4_exp.config import ModelConfig, VisionConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    from mlx_vlm.speculative import dflash as spec_dflash

    mx.random.seed(0)
    cfg = _make_qwen4_text_config(num_layers=6)
    model_cfg = ModelConfig(text_config=cfg, vision_config=VisionConfig(), model_type='qwen4_exp', vocab_size=cfg.vocab_size)
    lm = LanguageModel(cfg, config=model_cfg)
    lm.apply(lambda a: a.astype(mx.bfloat16))
    mx.eval(lm.parameters())
    qwen4_exp.install(lm)

    target_layers = [1, 3, 5]
    prompt = mx.array([[3, 17, 42, 99]], dtype=mx.int32)

    # 1. Verify capture_layer_ids produces unweighted 4-stream mean at completed target layers.
    c_prefill = lm.make_cache()
    out_prefill = lm(prompt, cache=c_prefill, capture_layer_ids=target_layers)
    mx.eval(out_prefill.logits, *out_prefill.hidden_states)
    assert len(out_prefill.hidden_states) == 3
    for h in out_prefill.hidden_states:
        assert h.shape == (1, 4, 128)

    # Reference manual pass computing raw HC mean vs mixer output.
    c_ref = lm.make_cache()
    h_raw = mx.tile(lm.model.embed_tokens(prompt), (1, 1, cfg.hc_count))
    fa_mask = 'causal'
    ssm_mask = None
    ref_pos, _ = lm.get_rope_index(prompt)
    ref_means = []
    ref_mixed = []
    for idx, (layer, lc) in enumerate(zip(lm.model.layers, c_ref)):
        h_raw = layer(
            h_raw,
            prompt,
            mask=(ssm_mask if layer.is_linear else fa_mask),
            cache=lc,
            position_ids=ref_pos,
        )
        if idx in target_layers:
            ref_means.append(qwen4_exp._hc_mean(h_raw, cfg.hc_count, cfg.hidden_size))
            ref_mixed.append(lm.model.hyper_connection_mixer(h_raw))
    mx.eval(*ref_means, *ref_mixed)
    for got_h, exp_mean, mixed_h in zip(out_prefill.hidden_states, ref_means, ref_mixed):
        assert mx.allclose(got_h, exp_mean, atol=1e-2, rtol=1e-2).item()
        assert not mx.allclose(got_h, mixed_h, atol=1e-2, rtol=1e-2).item()

    # 2. Verify 1-token APC prefix-hit prefill still captures 3 taps cleanly.
    tok1 = mx.array([[11]], dtype=mx.int32)
    out_one = lm(tok1, cache=c_prefill, capture_layer_ids=target_layers)
    mx.eval(out_one.logits, *out_one.hidden_states)
    assert len(out_one.hidden_states) == 3
    for h in out_one.hidden_states:
        assert h.shape == (1, 1, 128)

    # 3. Verify speculative_verify_dflash_hidden and speculative_verify=True match sequential decode.
    verify_ids = mx.array([[21, 32, 43, 54]], dtype=mx.int32)

    def _clone_cache(src_cache):
        import copy
        cloned = copy.deepcopy(src_cache)
        mx.eval([c.state for c in cloned])
        return cloned

    c_seq = _clone_cache(c_prefill)
    c_greedy = _clone_cache(c_prefill)
    c_sampled = _clone_cache(c_prefill)

    seq_logits = []
    for t in range(verify_ids.shape[1]):
        o_t = lm(verify_ids[:, t:t + 1], cache=c_seq)
        mx.eval(o_t.logits)
        seq_logits.append(o_t.logits)
    seq_all = mx.concatenate(seq_logits, axis=1)

    g_taps, g_final, g_tx = lm.speculative_verify_dflash_hidden(verify_ids, c_greedy, target_layers)
    g_tokens = lm.speculative_argmax_from_hidden(g_final)
    mx.eval(*g_taps, g_final, g_tokens)
    g_tx.abort()
    assert len(g_taps) == 3
    for h in g_taps:
        assert h.shape == (1, 4, 128)
    assert mx.array_equal(g_tokens, mx.argmax(seq_all, axis=-1)).item()

    s_out = lm(
        verify_ids,
        cache=c_sampled,
        capture_layer_ids=target_layers,
        speculative_verify=True,
    )
    mx.eval(s_out.logits, *s_out.hidden_states)
    s_out.gdn_states.abort()
    assert len(s_out.hidden_states) == 3
    for gh, sh in zip(g_taps, s_out.hidden_states):
        assert mx.allclose(gh, sh, atol=1e-2, rtol=1e-2).item()
    assert mx.allclose(s_out.logits, seq_all, atol=1e-2, rtol=1e-2).item()

    # 4. Load external DFlash2 checkpoint: quantize to 4 bits (not 2!), bind selector on greedy, run rounds.
    ckpt_dir = _save_dflash2_checkpoint(tmp_path / 'dflash2_ckpt', target_layer_ids=target_layers, num_target_layers=6)
    drafter, kind = mtp.load_external(ckpt_dir, bits=2, target_model=lm)
    assert kind == 'dflash'
    assert mtp.label_of(drafter) == 'DFlash2'
    assert mtp.bits_of(drafter) == '4'
    assert drafter.draft_block_greedy.__func__ is drafter.draft_block.__func__

    # Verify CandidateSelector.select is invoked on greedy round loop and tokens emit cleanly.
    selector_calls = []
    orig_select = drafter.candidate_selector.select

    def tracked_select(*args, **kwargs):
        selector_calls.append(True)
        return orig_select(*args, **kwargs)

    drafter.candidate_selector.select = tracked_select

    c_run_g = lm.make_cache()
    pref_g = lm(prompt, cache=c_run_g, capture_layer_ids=target_layers)
    mx.eval(pref_g.logits, *pref_g.hidden_states)
    hidden_g = mx.concatenate(pref_g.hidden_states, axis=-1)[:, -1:, :]
    bonus_g = int(mx.argmax(pref_g.logits[:, -1, :], axis=-1).item())
    emitted_g = list(
        spec_dflash._dflash_rounds(
            lm,
            drafter,
            c_run_g,
            hidden_g,
            first_bonus=bonus_g,
            max_tokens=6,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            greedy_sampling=True,
        )
    )
    assert len(emitted_g) == 5
    assert selector_calls

    c_run_s = lm.make_cache()
    pref_s = lm(prompt, cache=c_run_s, capture_layer_ids=target_layers)
    mx.eval(pref_s.logits, *pref_s.hidden_states)
    hidden_s = mx.concatenate(pref_s.hidden_states, axis=-1)[:, -1:, :]
    bonus_s = int(mx.argmax(pref_s.logits[:, -1, :], axis=-1).item())
    emitted_s = list(
        spec_dflash._dflash_rounds(
            lm,
            drafter,
            c_run_s,
            hidden_s,
            first_bonus=bonus_s,
            max_tokens=5,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            greedy_sampling=False,
        )
    )
    assert len(emitted_s) == 4
