# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""quanteeze.apc_plain: a prompt read back from the prefix cache for a request alone, served on the model's plain
caches as a fresh request is, on a tiny random Qwen3.5 text model (Qwen3.8-27B's module).  No download, a few
seconds.  What is asked: the decode over plain()'s caches gives the logits mlx-vlm's single-row batch caches give, bit
for bit, without taking a row apart at each step; rows plain() cannot make so are left as they are; install() gives
PromptProcessingBatch plain()'s caches only where mlx-vlm gives a fresh request the model's plain ones.
"""
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models import cache
from mlx_vlm.models.qwen3_5 import language as qwen35
from mlx_vlm.models.qwen3_5.config import TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel

from quanteeze import apc_plain, control, variables

ARGS = dict(model_type='qwen3_5_text', hidden_size=64, intermediate_size=128, linear_num_value_heads=4,
            linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
            num_hidden_layers=8, num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=2,
            max_position_embeddings=512, head_dim=64,
            rope_parameters={'type': 'default', 'mrope_section': [1, 1, 0], 'rope_theta': 10000.0,
                             'partial_rotary_factor': 0.25})
VISION = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=1000,
                         video_token_id=1001, vision_start_token_id=1002)


def _model():
    mx.random.seed(0)
    lm = LanguageModel(TextConfig(**ARGS), VISION)
    mx.eval(lm.parameters())
    return lm


def _warm(lm, ids):
    """Batch caches of one row holding the prompt, as the prefix cache's read back gives them."""
    caches = _make_cache(lm, [0])
    mx.eval(lm(ids, cache=caches).logits)
    return caches


def _decode(lm, caches, ids):
    got = [lm(ids[:, n:n + 1], cache=caches).logits for n in range(ids.shape[1])]
    mx.eval(got)
    return mx.concatenate(got, axis=1)


@pytest.fixture
def counted(monkeypatch):
    seen = {'extracted': 0}
    original = qwen35._extract_row_cache

    def extract(entry, row):
        seen['extracted'] += 1
        return original(entry, row)

    monkeypatch.setattr(qwen35, '_extract_row_cache', extract)
    return seen


def test_setting_and_control_choices_registered():
    assert 'QUANTEEZE_APC_READ_BACK_PLAIN' in variables.SETTINGS
    assert variables.SETTINGS['QUANTEEZE_APC_READ_BACK_PLAIN'].default == 'on'
    assert control.CHOICES['QUANTEEZE_APC_READ_BACK_PLAIN'] == ('on', 'off')


def test_plain_caches_decode_as_the_batch_caches_do_bit_for_bit_without_taking_rows_apart(counted):
    lm = _model()
    mx.random.seed(5)
    prompt = mx.random.randint(2, ARGS['vocab_size'], (1, 37))
    after = mx.random.randint(2, ARGS['vocab_size'], (1, 6))
    batch = _warm(lm, prompt)
    counted['extracted'] = 0
    want = _decode(lm, batch, after)
    assert counted['extracted'] > 0                                   # mlx-vlm's: a row taken apart each step
    made = apc_plain.plain(lm, _warm(lm, prompt))
    assert [type(c).__name__ for c in made] == [type(c).__name__ for c in lm.make_cache()]
    counted['extracted'] = 0
    got = _decode(lm, made, after)
    assert counted['extracted'] == 0
    assert mx.array_equal(got, want).item()
    fresh = lm.make_cache()                                          # a fresh request's: the prompt, then the same
    mx.eval(lm(prompt, cache=fresh).logits)
    assert mx.array_equal(_decode(lm, fresh, after), want).item()


def test_rows_with_padding_or_of_other_kinds_are_left_as_they_are():
    lm = _model()
    mx.random.seed(5)
    prompt = mx.random.randint(2, ARGS['vocab_size'], (1, 20))
    padded = _warm(lm, prompt)                         # (the model's single-row pass merges a row back unpadded:
    padded[3].left_padding = mx.array([2])             # padding set as a batch of rows might leave it)
    assert apc_plain.plain(lm, padded) is None
    padded = _warm(lm, prompt)
    padded[0].left_padding = mx.array([2])
    assert apc_plain.plain(lm, padded) is None
    two = _make_cache(lm, [0, 0])
    mx.eval(lm(mx.concatenate([prompt, prompt]), cache=two).logits)
    assert apc_plain.plain(lm, two) is None
    quantized = _make_cache(lm, [0], kv_bits=8, kv_group_size=64)
    mx.eval(lm(prompt, cache=quantized).logits)
    assert apc_plain.plain(lm, quantized) is None
    assert apc_plain.plain(lm, _warm(lm, prompt)[:-1]) is None


def test_install_gives_a_warm_request_alone_the_plain_caches_where_a_fresh_one_would_have_them(monkeypatch):
    class Batch:                                                     # PromptProcessingBatch's arguments that count
        def __init__(self, model, uids, input_ids, max_tokens, kv_bits=None, warm_cache=None,
                     right_pad_per_row=None, draft_model=None, draft_kind=None):
            self.warm_cache = warm_cache

    monkeypatch.setattr(apc_plain, 'STATS', {'plain': 0, 'kept': 0})
    assert apc_plain.install(Batch) and not apc_plain.install(Batch)
    lm = _model()
    prompt = mx.array([[5, 6, 7, 8, 9, 10, 11]])
    warm = _warm(lm, prompt)
    made = Batch(lm, [1], [[12]], [4], warm_cache=warm).warm_cache
    assert type(made[3]) is cache.KVCache and made[3].offset == 7 and made[3].keys is warm[3].keys
    assert type(made[0]) is cache.ArraysCache and made[0].left_padding is None
    assert Batch(lm, [1], [[12]], [4], kv_bits=8, warm_cache=warm).warm_cache is warm
    assert Batch(lm, [1, 2], [[12], [13]], [4, 4], warm_cache=warm).warm_cache is warm
    assert Batch(lm, [1], [[12]], [4], warm_cache=warm, right_pad_per_row=[2]).warm_cache is warm
    assert type(Batch(lm, [1], [[12]], [4], warm_cache=warm,                 # mlx-vlm's read back: one row's [0]
                      right_pad_per_row=[0]).warm_cache[3]) is cache.KVCache
    assert Batch(lm, [1], [[12]], [4], warm_cache=warm, draft_model=object(), draft_kind='mtp').warm_cache is warm
    assert Batch(lm, [1], [[12]], [4]).warm_cache is None
    assert apc_plain.STATS == {'plain': 2, 'kept': 0}
