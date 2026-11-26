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
    assert 'dflash2' not in home.PRESETS['mtp']
    assert 'dflash2' in home.PRESETS['drafter']
    assert home.PRESETS['drafter']['dflash2']['values']['QUANTEEZE_DRAFTER'] == 'dflash2'

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
    c_skip = _clone_cache(c_prefill)

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
    assert g_final.shape == (1, 4, cfg.hc_count * cfg.hidden_size)
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

    s_skip = lm(
        verify_ids,
        cache=c_skip,
        capture_layer_ids=target_layers,
        speculative_verify=True,
        return_hidden=True,
        skip_logits=True,
    )
    assert s_skip.logits is None
    assert len(s_skip.hidden_states) == 4
    assert s_skip.hidden_states[-1].shape == (1, 4, cfg.hc_count * cfg.hidden_size)
    mx.eval(*s_skip.hidden_states)
    s_skip.gdn_states.abort()
    assert mx.allclose(s_skip.hidden_states[-1], g_final, atol=1e-2, rtol=1e-2).item()

    c_lazy = _clone_cache(c_prefill)
    c_lazy_seq = _clone_cache(c_prefill)
    step_tok = mx.array([[15]], dtype=mx.int32)
    _ = lm(step_tok, cache=c_lazy)
    o_lazy_seq = lm(step_tok, cache=c_lazy_seq)
    mx.eval(o_lazy_seq.logits)
    s_lazy = lm(
        verify_ids,
        cache=c_lazy,
        capture_layer_ids=target_layers,
        speculative_verify=True,
        return_hidden=True,
        skip_logits=True,
    )
    mx.eval(*s_lazy.hidden_states)
    s_lazy.gdn_states.abort()
    seq_after_step = []
    for t in range(verify_ids.shape[1]):
        o_t = lm(verify_ids[:, t:t + 1], cache=c_lazy_seq)
        mx.eval(o_t.logits)
        seq_after_step.append(o_t.logits)
    lazy_seq_logits = mx.concatenate(seq_after_step, axis=1)
    assert s_lazy.logits is None
    assert mx.array_equal(
        lm.speculative_argmax_from_hidden(s_lazy.hidden_states[-1]),
        mx.argmax(lazy_seq_logits, axis=-1),
    ).item()

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


def test_one_token_capture_uses_batch_invariant_forward_and_drafter_bits_and_coalescing(tmp_path, monkeypatch):
    """1-token hidden capture stays on `_BatchInvariantForward`, `drafter_bits` follows `AUTO_BITS`,
    sequential requests avoid the 2s coalescing delay, and missing external drafters fall back."""
    import gc
    import time
    from mlx_vlm.models.qwen4_exp import language as _q4_lang
    from quanteeze import serve

    # 1. 1-token capture routes through _BatchInvariantForward without disabling _supports_batch_invariant_decode
    from mlx_vlm.models.qwen4_exp.config import ModelConfig, VisionConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    mx.random.seed(1)
    cfg = _make_qwen4_text_config(num_layers=6)
    model_cfg = ModelConfig(text_config=cfg, vision_config=VisionConfig(), model_type='qwen4_exp', vocab_size=cfg.vocab_size)
    lm = LanguageModel(cfg, config=model_cfg)
    lm.apply(lambda a: a.astype(mx.bfloat16))
    mx.eval(lm.parameters())
    qwen4_exp.install(lm)
    target_layers = [1, 3, 5]
    seen_bi = []
    bi_obj = _q4_lang._QWEN4_BATCH_INVARIANT_FORWARD
    orig_bi_call = type(bi_obj).__call__

    def tracked_bi_call(self_bi, language_model, inputs, **kwargs):
        seen_bi.append((tuple(inputs.shape), kwargs.get('capture_layer_ids'), qwen4_exp.CAPTURE[0]))
        return orig_bi_call(self_bi, language_model, inputs, **kwargs)

    monkeypatch.setattr(type(bi_obj), '__call__', tracked_bi_call)
    cache = lm.make_cache()
    prefill = lm(mx.array([[1, 2, 3]]), cache=cache, capture_layer_ids=target_layers)
    mx.eval(prefill.logits, *prefill.hidden_states)
    out = lm(mx.array([[4]]), cache=cache, capture_layer_ids=target_layers)
    mx.eval(out.logits, *out.hidden_states)
    assert len(seen_bi) == 1 and seen_bi[0][0] == (1, 1) and seen_bi[0][2] == target_layers
    assert len(out.hidden_states) == len(target_layers)
    assert qwen4_exp.CAPTURE[0] is None

    # 2. QUANTEEZE_MTP_BITS=auto -> 4 for qwen4_exp, 2 otherwise; explicit int & off respected
    monkeypatch.delenv('QUANTEEZE_MTP_BITS', raising=False)
    assert mtp.drafter_bits('qwen4_exp') == 4
    assert mtp.drafter_bits('glm5_next') == 2
    monkeypatch.setenv('QUANTEEZE_MTP_BITS', '3')
    assert mtp.drafter_bits('qwen4_exp') == 3
    monkeypatch.setenv('QUANTEEZE_MTP_BITS', 'off')
    assert mtp.drafter_bits('qwen4_exp') is None

    # 3. _Flying coalescing: sequential completed responses never wait 2s; overlapping in-flight does
    class Gen:
        def generate(self, prompt, images=None, audio=None, args=None, videos=None):
            return 'ctx', (t for t in ('a', 'b'))

    mtp.install(Gen)
    from mlx_vlm.server import generation
    monkeypatch.delenv('MLX_VLM_SPEC_BATCH_COALESCE_MS', raising=False)
    monkeypatch.setattr(mtp, '_COMING', {'flying': set(), 'burst': None})
    clock = [1000.0]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    gen = Gen()

    def answered():
        list(gen.generate('p', args=SimpleNamespace(thinking_budget=None))[1])

    answered()
    assert generation.get_speculative_batch_coalesce_s() == 0.0
    for _ in range(3):
        clock[0] += 0.5
        answered()
    assert generation.get_speculative_batch_coalesce_s() == 0.0
    first = gen.generate('p', args=SimpleNamespace(thinking_budget=None))[1]
    next(first)
    clock[0] += 0.5
    answered()
    assert generation.get_speculative_batch_coalesce_s() == mtp.GATHER_MS / 1000
    list(first)
    assert not mtp._COMING['flying']
    clock[0] += mtp.BURST_S + 1
    assert generation.get_speculative_batch_coalesce_s() == 0.0
    closed = gen.generate('p', args=SimpleNamespace(thinking_budget=None))[1]
    closed.close()
    dropped = gen.generate('p', args=SimpleNamespace(thinking_budget=None))[1]
    del dropped
    gc.collect()
    assert not mtp._COMING['flying']
    assert generation.get_speculative_batch_coalesce_s() == 0.0

    # 4. External drafter catalog detection & fallback to bundled MTP head when missing
    df2_dir = _save_dflash2_checkpoint(tmp_path / 'df2', target_layer_ids=target_layers)
    assert mtp.drafter_checkpoint(df2_dir)
    assert not mtp.drafter_checkpoint(tmp_path / 'missing')
    loaded_df2, _ = mtp.load_external(df2_dir, bits=4, target_model=lm)
    assert mtp.name(loaded_df2) == 'DFlash2'
    assert mtp.name(SimpleNamespace()) == 'MTP head'
    monkeypatch.setattr(home, 'model_roots', lambda: [tmp_path])
    monkeypatch.setattr(home, 'hub', lambda: tmp_path / 'hub')
    monkeypatch.setenv('QUANTEEZE_DRAFTER', 'dflash2')
    assert mtp.known_drafter('qwen4_exp') and not mtp.known_drafter('glm5_next')
    assert mtp.drafters_for('qwen4_exp')[0]['path'] is None
    monkeypatch.setattr(mtp, 'auto', lambda p, k: (True, 'ok'))
    fake_head = SimpleNamespace(config=SimpleNamespace(runtime_block_size=None))
    sentinel = (fake_head, 'mtp')
    monkeypatch.setattr(mtp, 'load', lambda p, b: sentinel)
    monkeypatch.setattr(mtp, 'nbytes', lambda d: 0)
    monkeypatch.setattr(mtp, 'bits_of', lambda d: '2')
    assert serve._drafter(str(tmp_path / 'tgt'), 'qwen4_exp') is sentinel

    # QUANTEEZE_COMPACT=on with QUANTEEZE_COMPACT_KEEP=MTP keeps external drafters active
    monkeypatch.setenv('QUANTEEZE_COMPACT', 'on')
    monkeypatch.setenv('QUANTEEZE_COMPACT_KEEP', 'MTP')
    monkeypatch.setenv('QUANTEEZE_DRAFTER', str(df2_dir))
    compact_drafter, compact_kind = serve._drafter(str(tmp_path / 'tgt'), 'qwen4_exp', model=lm)
    assert compact_kind == 'dflash' and mtp.name(compact_drafter) == 'DFlash2'
    monkeypatch.delenv('QUANTEEZE_COMPACT', raising=False)
    monkeypatch.delenv('QUANTEEZE_COMPACT_KEEP', raising=False)

    # 5. Home drafter preset, legacy QUANTEEZE_MTP=dflash2 display/reset, & kind='drafter' download validation
    monkeypatch.setenv('QUANTEEZE_SETTINGS', str(tmp_path / 'settings.json'))
    monkeypatch.delenv('QUANTEEZE_DRAFTER', raising=False)
    monkeypatch.setenv('QUANTEEZE_MTP', 'dflash2')
    assert home.presets()['drafter']['now'] == 'dflash2'
    home.choose_preset('drafter', 'head')
    assert variables.value('QUANTEEZE_MTP') is None and variables.value('QUANTEEZE_DRAFTER') is None
    assert home.presets()['drafter']['now'] == 'head'
    assert home.PRESETS['drafter']['dflash2']['values']['QUANTEEZE_DRAFTER'] == 'dflash2'
    monkeypatch.setattr(home, '_DOWNLOAD', {
        'repo': None, 'state': 'idle', 'error': None, 'bytes': 0, 'total': None, 'path': None, 'kind': 'model',
    })
    with pytest.raises(ValueError):
        home.download('org/name', kind='judge')
    dl_out = home.download('org/df2', fetch=lambda repo_id: str(tmp_path), kind='drafter')
    assert dl_out['kind'] == 'drafter'


def test_verify_pass_async_evals_each_layer(monkeypatch):
    """Multi-token verify passes schedule per-layer async_eval instead of batching by EVAL_LAYERS."""
    from mlx_vlm.models.qwen4_exp.config import ModelConfig, VisionConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    cfg = _make_qwen4_text_config(num_layers=6)
    lm = LanguageModel(
        cfg,
        ModelConfig(text_config=cfg, vision_config=VisionConfig(), model_type=cfg.model_type),
    )
    lm.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters())
    qwen4_exp.install(lm)
    monkeypatch.setattr(qwen4_exp, 'EVAL_LAYERS', [4])

    cache = lm.make_cache()
    prefill = lm(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
    mx.eval(prefill.logits)

    orig_async_eval = mx.async_eval
    seen = []
    hc_width = cfg.hc_count * cfg.hidden_size

    def tracked_async_eval(*args, **kwargs):
        if (
            len(args) == 1
            and isinstance(args[0], mx.array)
            and args[0].ndim == 3
            and args[0].shape[-1] == hc_width
        ):
            seen.append(tuple(args[0].shape))
        return orig_async_eval(*args, **kwargs)

    monkeypatch.setattr(mx, 'async_eval', tracked_async_eval)

    dec_out = lm(mx.array([[4]], dtype=mx.int32), cache=cache)
    mx.eval(dec_out.logits)
    assert seen == [(1, 1, hc_width), (1, 1, hc_width)]

    seen.clear()
    out = lm(
        mx.array([[5, 6, 7, 8]], dtype=mx.int32),
        cache=cache,
        capture_layer_ids=[1, 3, 5],
        speculative_verify=True,
    )
    mx.eval(out.logits, *out.hidden_states)
    out.gdn_states.abort()
    assert seen == [(1, 4, hc_width)] * cfg.num_hidden_layers


def test_qsa_cache_preallocates_indexer_and_preserves_blocks_on_trim():
    """`BatchQSAKVCache` preallocates indexer buffers, handles promotion, preserves valid block summaries on trim,
    and matches cleared-block sparse verify bit for bit."""
    from mlx_vlm.generate.ar import _make_cache
    from mlx_vlm.models.qwen4_exp import language as _language
    from mlx_vlm.models.qwen4_exp.config import ModelConfig, VisionConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    from quanteeze import kv_bits

    qwen4_exp._install_qsa_cache()
    _, kv_batch_cls = kv_bits.classes()

    # 1. BatchQSAKVCache and kv_bits.Batch preallocate in 256-token chunks and promote int32->int64 and 2D->3D.
    for batch_cls in (_language.BatchQSAKVCache, kv_batch_cls):
        c = batch_cls([0])
        k0, p0 = c.update_indexer(mx.ones((1, 10, 16), dtype=mx.bfloat16), mx.arange(10, dtype=mx.int32)[None])
        assert c.index_keys.shape == (1, 256, 16)
        assert c.index_position_ids.shape == (1, 256)
        assert c.index_offset == 10 and k0.shape == (1, 10, 16) and p0.shape == (1, 10)

        k1, p1 = c.update_indexer(mx.full((1, 2, 16), 2.0, dtype=mx.bfloat16), mx.array([[10, 11]], dtype=mx.int64))
        assert c.index_keys.shape[1] == 256
        assert c.index_position_ids.dtype == mx.int64
        assert c.index_offset == 12 and k1.shape == (1, 12, 16) and p1.shape == (1, 12)
        assert p1[0].tolist() == list(range(12))

        pos3 = mx.broadcast_to(mx.array([[12, 13]], dtype=mx.int64)[None], (3, 1, 2))
        k2, p2 = c.update_indexer(mx.full((1, 2, 16), 3.0, dtype=mx.bfloat16), pos3)
        assert c.index_position_ids.shape == (3, 1, 256)
        assert c.index_offset == 14 and k2.shape == (1, 14, 16) and p2.shape == (3, 1, 14)

        assert c.state[1].shape == (1, 14, 16) and c.state[2].shape == (3, 1, 14)
        extracted = c.extract(0)
        assert extracted.index_keys.shape == (1, 14, 16) and extracted.index_position_ids.shape == (3, 1, 14)

        k3, p3 = c.update_indexer(
            mx.ones((1, 250, 16), dtype=mx.bfloat16),
            mx.broadcast_to(mx.arange(14, 264, dtype=mx.int64)[None, None], (3, 1, 250)),
        )
        assert c.index_keys.shape == (1, 270, 16) and c.index_position_ids.shape == (3, 1, 270)
        assert c.index_offset == 264 and k3.shape == (1, 264, 16) and p3.shape == (3, 1, 264)
        k4, p4 = c.update_indexer(
            mx.ones((1, 2, 16), dtype=mx.bfloat16),
            mx.broadcast_to(mx.arange(264, 266, dtype=mx.int64)[None, None], (3, 1, 2)),
        )
        assert c.index_keys.shape == (1, 270, 16) and c.index_position_ids.shape == (3, 1, 270)
        assert c.index_offset == 266 and k4.shape == (1, 266, 16) and p4.shape == (3, 1, 266)

    # 2. trim() preserves complete block summaries before the trimmed length across all QSA cache types.
    for cache_cls, single in (
        (_language.QSAKVCache, True),
        (_language.QSAQuantizedKVCache, True),
        (_language.BatchQSAKVCache, False),
        (kv_batch_cls, False),
    ):
        c = cache_cls() if single else cache_cls([0])
        c.update_and_fetch(mx.zeros((1, 1, 14, 64), dtype=mx.bfloat16), mx.zeros((1, 1, 14, 64), dtype=mx.bfloat16))
        c.update_indexer(mx.ones((1, 14, 16), dtype=mx.bfloat16), mx.arange(14, dtype=mx.int32)[None])
        blocks = mx.arange(3 * 16, dtype=mx.bfloat16).reshape(1, 1, 3, 16)
        c.index_block_keys = blocks
        c.index_block_ratio = 4

        assert c.trim(2) == 2
        assert c.index_block_keys.shape == (1, 1, 3, 16) and c.index_block_ratio == 4
        if single:
            assert c.offset == 12 and c.index_keys.shape[1] == 12 and c.index_position_ids.shape[-1] == 12
        else:
            assert c.index_offset == 12

        assert c.trim(1) == 1
        assert c.index_block_keys.shape == (1, 1, 2, 16) and c.index_block_ratio == 4
        assert mx.array_equal(c.index_block_keys, blocks[:, :, :2]).item()

        assert c.trim(8) == 8
        assert c.index_block_keys is None and c.index_block_ratio is None

    # 3. End-to-end sparse verify with preserved blocks matches cleared-blocks reference bit for bit.
    mx.random.seed(12)
    cfg = _make_qwen4_text_config(num_layers=6)
    lm = LanguageModel(
        cfg,
        ModelConfig(text_config=cfg, vision_config=VisionConfig(), model_type=cfg.model_type),
    )
    lm.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters())
    qwen4_exp.install(lm)

    full_idx = next(i for i, layer in enumerate(lm.layers) if not layer.is_linear)
    ids = mx.random.randint(2, cfg.vocab_size, (1, 86), dtype=mx.int32)

    def run_verify(clear_blocks):
        caches = _make_cache(lm, [0])
        mx.eval(lm(ids[:, :80], cache=caches).logits)
        first = lm(ids[:, 80:84], cache=caches, capture_layer_ids=[1, 3, 5], speculative_verify=True)
        mx.eval(first.logits, *first.hidden_states)
        first.gdn_states.commit([2])
        fc = caches[full_idx]
        assert fc.index_offset == 82
        assert fc.index_block_keys is not None and fc.index_block_keys.shape[2] == 20
        assert fc.index_keys.shape[1] == 256
        if clear_blocks:
            for c in caches:
                if hasattr(c, 'clear_index_blocks'):
                    c.clear_index_blocks()
        second = lm(ids[:, 82:86], cache=caches, capture_layer_ids=[1, 3, 5], speculative_verify=True)
        mx.eval(second.logits, *second.hidden_states)
        second.gdn_states.abort()
        return second.logits

    assert mx.array_equal(run_verify(clear_blocks=False), run_verify(clear_blocks=True)).item()
