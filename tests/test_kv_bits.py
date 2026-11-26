# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Flash-Next's full-attention keys and values at 8 bits (`quanteeze.kv_bits`), on a tiny random Qwen4-Exp model."""
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models import base
from mlx_vlm.models.qwen4_exp import language as _language
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import LanguageModel

from quanteeze import kv_bits, per_load, serve

ARGS = dict(
    model_type='qwen4_exp', hidden_size=64, num_hidden_layers=4, num_attention_heads=4, head_dim=64,
    num_key_value_heads=2, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
    linear_value_head_dim=16, linear_conv_kernel_dim=4, num_experts=16, num_experts_per_tok=4,
    shared_expert_intermediate_size=32, moe_intermediate_size=32, rms_norm_eps=1e-6, vocab_size=128,
    max_position_embeddings=512, hc_count=4, hc_lowrank=8, ple_layer_ids=[2], ngram_size=3,
    heads_per_ngram=2, ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=4, eos_token_id=1, indexer_n_heads=2, indexer_head_dim=16, indexer_budget=64,
    indexer_compress_ratio=4,
    rope_parameters={'type': 'default', 'mrope_section': [1, 1, 0], 'rope_theta': 10000.0,
                     'partial_rotary_factor': 0.25},
)
VISION = SimpleNamespace(
    vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=1000,
    video_token_id=1001, vision_start_token_id=1002,
)
ON = {'QUANTEEZE_KV_BITS': '8'}
# On Metal MLX computes float32 matmuls via tf32 (~1.4e-3 difference vs ~1e-4 on CUDA/CPU).
FLOAT32_CLOSE = 3e-3 if mx.metal.is_available() else 1e-4


def _model():
    mx.random.seed(0)
    lm = LanguageModel(TextConfig(**ARGS), VISION)
    mx.eval(lm.parameters())
    return lm


def _off(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.linalg.norm(b)).item()


@pytest.fixture
def fresh(monkeypatch):
    """Reset `kv_bits` check tables, counters, and `mlx-vlm` hooks around each test."""
    monkeypatch.setattr(base, 'quantized_scaled_dot_product_attention', base.quantized_scaled_dot_product_attention)
    monkeypatch.setattr(kv_bits, 'CHOICE', {})
    monkeypatch.setattr(kv_bits, 'CHECKED', {})
    monkeypatch.setattr(kv_bits, 'STATS', {name: 0 for name in kv_bits.STATS})
    monkeypatch.setattr(_language, 'dispatch_qsa_attention', _language.dispatch_qsa_attention)
    monkeypatch.setattr(kv_bits, 'SELECTED', [True])
    monkeypatch.setattr(kv_bits, 'UNPACKED', [False])
    monkeypatch.setattr(kv_bits, 'SAID', [])
    monkeypatch.delenv('KV_BITS', raising=False)
    monkeypatch.setattr(kv_bits, '_UNKNOWN', [])
    monkeypatch.setattr(kv_bits, 'SERVED', [False])
    monkeypatch.setattr(kv_bits, '_CLASSES', [])
    for name in ('to_batch', 'merge', 'prefix_cache_merge', 'prefix_cache_snapshot'):
        if name in vars(_language.QSAQuantizedKVCache):
            monkeypatch.setattr(_language.QSAQuantizedKVCache, name, vars(_language.QSAQuantizedKVCache)[name])
        else:
            monkeypatch.setattr(_language.QSAQuantizedKVCache, name, None, raising=False)
            monkeypatch.delattr(_language.QSAQuantizedKVCache, name)


@pytest.fixture
def lm(fresh):
    model = _model()
    assert kv_bits.install(model, ON)
    return model


def _quantized(shape, dtype):
    x = mx.random.normal(shape).astype(dtype)
    return x, mx.quantize(x, group_size=kv_bits.GROUP, bits=kv_bits.BITS)


@pytest.mark.parametrize('rows,length,mask', [(1, 1, None), (1, 2, 'causal'), (2, 1, None), (1, 3, 'array')])
@pytest.mark.parametrize('dtype', [mx.float32, mx.bfloat16])
def test_the_attention_a_head_at_a_time_is_the_spread_out_products(fresh, rows, length, mask, dtype):
    mx.random.seed(3)
    keys, values = _quantized((rows, 2, 40, 64), dtype), _quantized((rows, 2, 40, 64), dtype)
    queries = mx.random.normal((rows, 4, length, 64)).astype(dtype)
    if mask == 'array':
        mask = mx.arange(40)[None, None, None, :] < (38 + mx.arange(length))[None, None, :, None]
    for n in range(kv_bits.CHECKS + 2):
        got = kv_bits.attention(queries, keys[1], values[1], 0.125, mask)
        want = mx.fast.scaled_dot_product_attention(queries, keys[0], values[0], scale=0.125, mask=mask)
        assert got.shape == want.shape and got.dtype == dtype
        assert _off(got, want) < 0.05, n
    assert (kv_bits.STATS['served'], kv_bits.STATS['checked']) == (2, kv_bits.CHECKS)
    assert kv_bits.STATS['differed'] == kv_bits.STATS['spread'] == 0
    scaled = (queries * 0.125).reshape(rows, 2, 2, length, 64)
    by_head = kv_bits._by_head(scaled, keys[1], values[1], mask, kv_bits.GROUP, kv_bits.BITS)
    spread = kv_bits._spread(scaled, keys[1], values[1], mask, kv_bits.GROUP, kv_bits.BITS)
    assert _off(by_head, spread) < (FLOAT32_CLOSE if dtype == mx.float32 else 2e-2)


def test_a_product_that_is_not_the_spread_out_ones_is_not_served(fresh, monkeypatch, capsys):
    mx.random.seed(4)
    keys, values = _quantized((1, 2, 24, 64), mx.float32), _quantized((1, 2, 24, 64), mx.float32)
    queries = mx.random.normal((1, 4, 1, 64))
    right = kv_bits._by_head
    monkeypatch.setattr(kv_bits, '_by_head', lambda *a: right(*a) * 2.0)
    for _ in range(kv_bits.CHECKS + 3):
        got = kv_bits.attention(queries, keys[1], values[1], 0.125, None)
        scaled = (queries * 0.125).reshape(1, 2, 2, 1, 64)
        want = kv_bits._spread(scaled, keys[1], values[1], None, kv_bits.GROUP, kv_bits.BITS).reshape(got.shape)
        assert mx.array_equal(got, want).item()
    assert kv_bits.STATS['differed'] >= 1 and kv_bits.STATS['served'] == 0 and kv_bits.STATS['spread'] >= 3
    assert capsys.readouterr().err.count('is not the spread-out product') == 1


def test_unset_the_model_keeps_its_float_caches(fresh, capsys):
    model = _model()
    made = model.make_cache
    assert not kv_bits.install(model, {}) and not kv_bits.install(model, {'QUANTEEZE_KV_BITS': 'off'})
    assert model.make_cache == made
    assert not getattr(base.quantized_scaled_dot_product_attention, '_quanteeze_kv_bits', False)
    assert not kv_bits.install(model, {'QUANTEEZE_KV_BITS': '4'})
    assert 'neither 8 nor off' in capsys.readouterr().err and model.make_cache == made


def test_mlx_vlms_own_setting_beside_it_is_refused(fresh, monkeypatch, capsys):
    monkeypatch.setenv('KV_BITS', '8')
    model = _model()
    assert not kv_bits.install(model, ON) and type(model.make_cache()[3]) is _language.QSAKVCache
    assert 'KV_BITS is set beside it' in capsys.readouterr().err


def test_a_head_that_is_not_groups_of_64_is_left_float(fresh, capsys):
    mx.random.seed(0)
    small = LanguageModel(TextConfig(**dict(ARGS, head_dim=16)), VISION)
    assert not kv_bits.install(small, ON)
    assert 'not applied' in capsys.readouterr().err
    assert type(small.make_cache()[3]) is _language.QSAKVCache


def test_the_caches_made_are_8_bit_for_the_full_attention_layers_alone(lm):
    single, batch = kv_bits.classes()
    linear = [layer.is_linear for layer in lm.layers]
    assert [type(c) is single for c in lm.make_cache()] == [not flat for flat in linear]
    caches = _make_cache(lm, [0])
    assert [type(c) is batch for c in caches] == [not flat for flat in linear]
    assert all(c.bits == 8 and c.group_size == 64 for c in caches if type(c) is batch)
    assert kv_bits.install(lm, ON) and len([c for c in lm.make_cache() if type(c) is single]) == linear.count(False)


@pytest.mark.parametrize('tokens', [20, 100])
def test_a_batchs_row_and_one_request_alone_decode_the_same_over_8_bit_keys(lm, tokens):
    mx.random.seed(5)
    ids = mx.random.randint(2, ARGS['vocab_size'], (1, tokens + 3))
    alone, batch = lm.make_cache(), _make_cache(lm, [0])
    outs = []
    for caches in (alone, batch):
        got = [lm(ids[:, :tokens], cache=caches).logits[:, -1:]]
        got += [lm(ids[:, tokens + n: tokens + n + 1], cache=caches).logits for n in range(3)]
        mx.eval(got)
        outs.append(mx.concatenate(got, axis=1))
    assert mx.isfinite(outs[1]).all().item()
    assert _off(outs[1], outs[0]) < 1e-3
    assert kv_bits.STATS['differed'] == 0 and kv_bits.STATS['spread'] == 0
    full = next(i for i, layer in enumerate(lm.layers) if not layer.is_linear)
    a, b = alone[full], batch[full]
    assert a.offset == tokens + 3 and b.offset.tolist() == [tokens + 3]
    for mine, theirs in zip(b.state[0][0] + b.state[0][1], a.state[0] + a.state[1]):
        assert mine.shape == theirs.shape and mine.dtype == theirs.dtype
    assert mx.array_equal(b.state[0][0][0], a.state[0][0]).item()
    assert mx.array_equal(b.index_keys[:, :b.index_offset], a.index_keys).item()


def test_a_layers_output_over_8_bit_keys_is_near_the_one_over_float_keys(lm):
    mx.random.seed(6)
    plain = _model()
    ids = mx.random.randint(2, ARGS['vocab_size'], (1, 41))
    full = next(i for i, layer in enumerate(lm.layers) if not layer.is_linear)
    seen = {}
    for name, model in (('float', plain), ('bits', lm)):
        attention = model.layers[full].self_attn
        calls = []
        original = type(attention).__call__
        caches = _make_cache(model, [0])
        model(ids[:, :40], cache=caches)
        forward = _language._QWEN4_BATCH_INVARIANT_FORWARD
        was = forward._qsa_attention

        def spy(attn, hidden, cache, position_ids, mask, was=was, calls=calls):
            out = was(attn, hidden, cache, position_ids, mask)
            calls.append((hidden, out))
            return out
        forward._qsa_attention = spy
        try:
            mx.eval(model(ids[:, 40:], cache=caches).logits)
        finally:
            del forward._qsa_attention
        assert calls and original is type(attention).__call__
        seen[name] = calls[0]
        seen[name + ' bytes'] = caches[full].kv_cache.nbytes
    assert _off(seen['bits'][1], seen['float'][1]) < 0.1
    held = 256 * 2 * 2
    assert seen['float bytes'] == held * 64 * 4
    assert seen['bits bytes'] == held * (64 + 2 * 4)


def test_a_batch_is_merged_extended_filtered_extracted_and_trimmed(lm):
    single, batch = kv_bits.classes()
    mx.random.seed(7)
    ids = mx.random.randint(2, ARGS['vocab_size'], (2, 30))
    full = next(i for i, layer in enumerate(lm.layers) if not layer.is_linear)
    rows = []
    for r, length in ((0, 30), (1, 22)):
        caches = _make_cache(lm, [0])
        mx.eval(lm(ids[r:r + 1, :length], cache=caches).logits)
        rows.append(caches[full])
    first = rows[0].extract(0)
    assert type(first) is single and first.offset == 30 and first.keys[0].shape[2] == 30
    assert mx.array_equal(first.index_keys, rows[0].index_keys[:, :30]).item()
    again = first.to_batch([0])
    assert type(again) is batch and again.offset.tolist() == [30] and again.kv_cache._idx == 30
    for a, b in zip(again.kv_cache.state[0] + again.kv_cache.state[1], rows[0].kv_cache.state[0] + rows[0].kv_cache.state[1]):
        assert mx.array_equal(a, b).item()
    merged = batch.merge([rows[0].extract(0), rows[1].extract(0)])
    assert merged.offset.tolist() == [30, 22] and merged.left_padding.tolist() == [0, 8]
    assert merged.index_keys.shape[:2] == (2, 30) and merged.kv_cache.keys[0].shape[:3] == (2, 2, 30)
    assert mx.array_equal(merged.kv_cache.keys[0][1, :, 8:], rows[1].kv_cache.state[0][0][0]).item()
    rows[0].extend(rows[1])
    assert rows[0].offset.tolist() == [30, 22] and rows[0].left_padding.tolist() == [0, 8]
    second = rows[0].extract(1)
    assert second.offset == 22 and mx.array_equal(second.keys[0], merged.extract(1).keys[0]).item()
    rows[0].filter(mx.array([1]))
    assert rows[0].offset.tolist() == [22] and rows[0].left_padding.tolist() == [0] and rows[0].index_offset == 22
    assert rows[0].trim(2) == 2 and rows[0].offset.tolist() == [20] and rows[0].index_offset == 20
    state = rows[0].state
    back = batch.from_state(state, '')
    assert type(back.kv_cache) is type(rows[0].kv_cache) and back.kv_cache._idx == 20 and back.index_offset == 20
    with pytest.raises(TypeError):
        rows[0].extend(_language.BatchQSAKVCache([0]))
    with pytest.raises(TypeError):
        batch.merge([_language.QSAKVCache()])


def _sparse_steps(model, ids, tokens, steps, caches):
    out = [model(ids[:, :tokens], cache=caches).logits[:, -1:]]
    at = tokens
    for step in steps:
        out.append(model(ids[:, at:at + step], cache=caches).logits)
        at += step
    mx.eval(out)
    return mx.concatenate(out, axis=1)


@pytest.mark.parametrize('batch', [False, True])
def test_a_sparse_decode_over_the_selected_keys_alone_is_the_masked_ones(lm, monkeypatch, batch):
    mx.random.seed(9)
    ids = mx.random.randint(2, ARGS['vocab_size'], (1, 200))
    steps = [1] * 6 + [2, 2, 1]
    monkeypatch.setattr(kv_bits, 'LONGER', 2)
    monkeypatch.setattr(kv_bits, 'SELECTED', [False])
    masked_caches = _make_cache(lm, [0]) if batch else lm.make_cache()
    masked = _sparse_steps(lm, ids, 150, steps, masked_caches)
    assert kv_bits.STATS['selected'] == kv_bits.STATS['selected_checked'] == 0
    monkeypatch.setattr(kv_bits, 'SELECTED', [True])
    caches = _make_cache(lm, [0]) if batch else lm.make_cache()
    got = _sparse_steps(lm, ids, 150, steps, caches)
    assert kv_bits.STATS['selected_differed'] == 0 and kv_bits.STATS['selected_checked'] >= kv_bits.CHECKS
    assert kv_bits.STATS['selected'] >= 1
    assert _off(got, masked) < 1e-3
    full = next(i for i, layer in enumerate(lm.layers) if not layer.is_linear)
    a, b = caches[full], masked_caches[full]
    for mine, theirs in zip(a.state[0] if not batch else a.state[0][0], b.state[0] if not batch else b.state[0][0]):
        assert _off(mine.astype(mx.float32), theirs.astype(mx.float32)) < 1e-2


def test_the_selected_keys_are_mlx_vlms_masks(fresh):
    mx.random.seed(10)
    model = _model()
    indexer = next(layer for layer in model.layers if not layer.is_linear).self_attn.indexer
    top, ratio = indexer.block_topk, indexer.compress_ratio
    for size, length in ((203, 1), (202, 2), (top * ratio + ratio, 2), (top * ratio + ratio + 1, 1)):
        keys, values = _quantized((1, 2, size, 64), mx.float32), _quantized((1, 2, size, 64), mx.float32)
        queries = mx.random.normal((1, 4, length, 64))
        ends = (size - length + 1 + mx.arange(length, dtype=mx.int32))[None]
        whole = ends // ratio
        scores = mx.random.normal((1, length, size // ratio))
        scores = mx.where(mx.arange(size // ratio)[None, None] < whole[..., None], scores, -mx.inf)
        blocks = mx.argpartition(scores, kth=-top, axis=-1)[..., -top:].astype(mx.int32)
        blocks = mx.where(blocks < whole[..., None], blocks, -1)
        selection = SimpleNamespace(
            selected_blocks=blocks, query_ends=ends, complete_counts=whole,
            left_padding=mx.zeros((1,), dtype=mx.int32), key_len=size, zero_padding=True,
        )
        mask = indexer.build_mask(selection)
        scaled = (queries * 0.125).reshape(1, 2, 2, length, 64)
        want = kv_bits._spread(scaled, keys[1], values[1], mask, kv_bits.GROUP, kv_bits.BITS).reshape(queries.shape)
        got = kv_bits._selected(queries, keys[1], values[1], blocks, ends, 0.125, ratio, kv_bits.GROUP, kv_bits.BITS)
        assert _off(got, want) < FLOAT32_CLOSE, (size, length)


def test_the_prompt_cache_takes_a_row_and_gives_it_back(lm):
    from mlx_vlm import apc, apc_adapters
    single, batch = kv_bits.classes()
    assert apc._resolve_checkpoint_class(single.__module__, single.__qualname__) is single
    for c in lm.make_cache():
        assert apc.classify_layer_for_apc(c).status != 'unsupported', type(c).__name__
    mx.random.seed(11)
    ids = mx.random.randint(2, ARGS['vocab_size'], (1, 31))
    caches = _make_cache(lm, [0])
    mx.eval(lm(ids[:, :30], cache=caches).logits)
    want = lm(ids[:, 30:], cache=caches).logits
    mx.eval(want)
    caches = _make_cache(lm, [0])
    mx.eval(lm(ids[:, :30], cache=caches).logits)
    rows = apc.extract_prompt_cache_from_batch(caches, 0)
    targets, kept = [], []
    for row in rows:
        copy = apc_adapters.clone_cache_entry(row, min_capacity_tokens=None, eval_targets=targets)
        assert copy is not None and type(copy) is type(row)
        kept.append(copy)
    mx.eval(targets)
    full = next(i for i, layer in enumerate(lm.layers) if not layer.is_linear)
    assert type(kept[full]) is single and kept[full].offset == 30 and kept[full].index_keys.shape[1] == 30
    arrays, metadata = {}, {}
    store = apc.DiskBlockStore.__new__(apc.DiskBlockStore)
    assert store._snapshot_exact_cache_entry(kept[full], 'c', arrays, metadata)
    assert metadata['c_kind'] == 'checkpoint' and metadata['c_qualname'] == 'QSAQuantizedKVCache'
    merged = [apc_adapters.merge_cache_entries([row], [30]) for row in kept]
    assert type(merged[full]) is batch and merged[full].offset.tolist() == [30]
    got = lm(ids[:, 30:], cache=merged).logits
    assert _off(got, want) < 1e-3


def test_the_prompt_caches_salt_tells_the_bits_while_such_a_model_is_served(fresh, monkeypatch):
    monkeypatch.setattr(serve, '_SERVING', {})
    assert kv_bits.tag() == ''
    assert not kv_bits.install(_model(), {}) and kv_bits.tag() == ''
    assert kv_bits.install(_model(), ON)
    assert kv_bits.tag() == 'keys and values at 8 bits a group of 64'
    assert 'keys and values at 8 bits a group of 64' in serve._precision()
    per_load.let_go()
    assert kv_bits.tag() == ''


def test_unpacked_is_auto_on_metal_and_said_on_or_off(fresh):
    assert kv_bits.unpacked({}) == mx.metal.is_available()
    assert kv_bits.unpacked({'QUANTEEZE_KV_BITS_UNPACKED': 'off'}) is False
    assert kv_bits.unpacked({'QUANTEEZE_KV_BITS_UNPACKED': 'on'}) is True


@pytest.mark.parametrize('length', [1, 3])
@pytest.mark.parametrize('mask', [None, 'causal', 'array'])
def test_the_attention_over_the_keys_spread_out_is_the_spread_out_products(fresh, monkeypatch, mask, length):
    monkeypatch.setattr(kv_bits, 'UNPACKED', [True])
    mx.random.seed(4)
    keys, values = _quantized((2, 2, 40, 64), mx.bfloat16), _quantized((2, 2, 40, 64), mx.bfloat16)
    queries = mx.random.normal((2, 4, length, 64)).astype(mx.bfloat16)
    if mask == 'array':
        mask = mx.arange(40)[None, None, None, :] < (38 + mx.arange(length))[None, None, :, None]
    if mask == 'causal' and length == 1:
        mask = None
    scaled = (queries * 0.125).reshape(2, 2, 2, length, 64)
    want = kv_bits._spread(scaled, keys[1], values[1], mask, kv_bits.GROUP, kv_bits.BITS).reshape(queries.shape)
    for n in range(kv_bits.CHECKS + 2):
        got = kv_bits.attention(queries, keys[1], values[1], 0.125, mask)
        assert got.shape == queries.shape and got.dtype == mx.bfloat16
        assert _off(got, want) < 2e-2, n
    assert kv_bits.STATS['checked'] == kv_bits.CHECKS
    assert kv_bits.STATS['quantized' if length <= kv_bits.ROWS_MOST else 'unpacked'] == 2
    assert kv_bits.STATS['differed'] == kv_bits.STATS['spread'] == 0


def test_a_prompt_read_over_the_keys_spread_out_is_the_one_a_head_at_a_time(lm, monkeypatch):
    mx.random.seed(12)
    ids = mx.random.randint(2, ARGS['vocab_size'], (1, 160))
    outs, kept = [], []
    for on in (False, True):
        monkeypatch.setattr(kv_bits, 'UNPACKED', [on])
        caches = _make_cache(lm, [0])
        got = [lm(ids[:, n:n + 16], cache=caches).logits[:, -1:] for n in range(0, 144, 16)]
        got += [lm(ids[:, n:n + 1], cache=caches).logits for n in range(144, 148)]
        mx.eval(got)
        outs.append(mx.concatenate(got, axis=1))
        kept.append(caches)
    assert mx.isfinite(outs[1]).all().item()
    assert _off(outs[1], outs[0]) < 1e-3
    assert kv_bits.STATS['unpacked_checked'] == kv_bits.CHECKS and kv_bits.STATS['unpacked_differed'] == 0
    assert kv_bits.STATS['unpacked'] >= 1 and kv_bits.STATS['differed'] == 0
    full = next(i for i, layer in enumerate(lm.layers) if not layer.is_linear)
    assert mx.array_equal(kept[1][full].state[0][0][0], kept[0][full].state[0][0][0]).item()


def test_home_kv_preset_and_model_info(tmp_path, monkeypatch):
    import json
    from quanteeze import home

    monkeypatch.setenv('QUANTEEZE_SETTINGS', str(tmp_path / 'settings.json'))
    monkeypatch.delenv('QUANTEEZE_KV_BITS', raising=False)
    monkeypatch.setattr(home, '_MODELS', {})

    qwen_dir = tmp_path / 'qwen4'
    qwen_dir.mkdir()
    (qwen_dir / 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp', 'num_hidden_layers': 4}))

    plain_dir = tmp_path / 'qwen35'
    plain_dir.mkdir()
    (plain_dir / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5', 'num_hidden_layers': 4}))

    other_dir = tmp_path / 'other'
    other_dir.mkdir()
    (other_dir / 'config.json').write_text(json.dumps({'model_type': 'llama', 'num_hidden_layers': 4}))

    assert home.model_info(qwen_dir)['kv_bits'] == 'flash'
    assert home.model_info(plain_dir)['kv_bits'] == 'plain'
    assert home.model_info(other_dir)['kv_bits'] is False

    state = home.presets()
    assert state['kv']['now'] == 'off'
    assert all('plain' in c for c in state['kv']['choices'])
    state = home.choose_preset('kv', '8')
    assert state['kv']['now'] == '8'
    state = home.choose_preset('kv', 'off')
    assert state['kv']['now'] == 'off'
    monkeypatch.setenv('QUANTEEZE_KV_BITS', 'off')
    assert home.presets()['kv']['now'] == 'off'
