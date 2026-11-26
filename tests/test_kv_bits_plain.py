# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""QUANTEEZE_KV_BITS=8 on a model whose full-attention layers keep a plain KVCache (quanteeze.kv_bits.install_plain:
e.g. Qwen3.8-27B), tested on a tiny random Qwen3.5 text model."""
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models import base, cache
from mlx_vlm.models.qwen3_5.config import TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel
from mlx_vlm.server.runtime import runtime

from quanteeze import kv_bits

ARGS = dict(
    model_type='qwen3_5_text', hidden_size=64, intermediate_size=128, linear_num_value_heads=4,
    linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    num_hidden_layers=8, num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=2,
    max_position_embeddings=512, head_dim=64,
    rope_parameters={'type': 'default', 'mrope_section': [1, 1, 0], 'rope_theta': 10000.0,
                     'partial_rotary_factor': 0.25},
)
VISION = SimpleNamespace(
    vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=1000,
    video_token_id=1001, vision_start_token_id=1002,
)
ON = {'QUANTEEZE_KV_BITS': '8'}


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
    """Reset check tables, stats, and mlx-vlm runtime config around each test."""
    monkeypatch.setattr(base, 'quantized_scaled_dot_product_attention', base.quantized_scaled_dot_product_attention)
    for name, value in (
        ('CHOICE', {}), ('CHECKED', {}), ('UNPACKED', [False]), ('SERVED', [False]),
        ('PLAIN', []), ('SAID', []), ('_UNKNOWN', []),
    ):
        monkeypatch.setattr(kv_bits, name, value)
    monkeypatch.setattr(kv_bits, 'STATS', {name: 0 for name in kv_bits.STATS})
    for name in ('kv_bits', 'kv_group_size', 'kv_quant_scheme'):
        monkeypatch.setattr(runtime.config, name, getattr(runtime.config, name))
    monkeypatch.delenv('KV_BITS', raising=False)


def test_install_plain_sets_mlx_vlms_kv_bits_and_gives_the_old_back(fresh):
    lm = _model()
    was = (runtime.config.kv_bits, runtime.config.kv_group_size, runtime.config.kv_quant_scheme)
    assert not kv_bits.install_plain(lm, {})
    assert kv_bits.install_plain(lm, ON)
    assert (runtime.config.kv_bits, runtime.config.kv_group_size, runtime.config.kv_quant_scheme) == (8, 64, 'uniform')
    assert kv_bits.SERVED[0] and kv_bits.tag()
    assert getattr(base.quantized_scaled_dot_product_attention, '_quanteeze_kv_bits', False)
    kv_bits._let_go()
    assert (runtime.config.kv_bits, runtime.config.kv_group_size, runtime.config.kv_quant_scheme) == was
    assert not kv_bits.SERVED[0] and not kv_bits.PLAIN


def test_caches_that_are_not_plain_and_mlx_vlms_own_setting_are_refused(fresh, monkeypatch, capsys):
    lm = _model()
    other = SimpleNamespace(make_cache=lambda: [cache.CacheList(cache.KVCache(), cache.KVCache())], layers=[])
    assert not kv_bits.install_plain(other, ON)
    assert 'not plain' in capsys.readouterr().err
    monkeypatch.setenv('KV_BITS', '8')
    assert not kv_bits.install_plain(lm, ON)
    assert 'KV_BITS is set beside it' in capsys.readouterr().err
    assert runtime.config.kv_bits is None or runtime.config.kv_bits != 8 or kv_bits.PLAIN == []


def _steps(lm, ids, caches):
    got = [lm(ids[:, n:n + 8], cache=caches).logits[:, -1:] for n in range(0, 48, 8)]
    got += [lm(ids[:, n:n + 1], cache=caches).logits for n in range(48, 52)]
    mx.eval(got)
    return mx.concatenate(got, axis=1)


def test_the_logits_over_mlx_vlms_8_bit_cache_a_head_at_a_time_and_spread_out(fresh, monkeypatch):
    """mlx-vlm's 8-bit batch cache (final full-attention layer left float) yields logits close to float KV cache
    and identical across per-head and Metal unpacked-prefill / 1-call quantized-decode paths."""
    lm = _model()
    assert kv_bits.install_plain(lm, ON)
    mx.random.seed(3)
    ids = mx.random.randint(2, ARGS['vocab_size'], (1, 52))
    floats = _steps(lm, ids, _make_cache(lm, [0]))
    outs = []
    for on in (False, True):
        monkeypatch.setattr(kv_bits, 'UNPACKED', [on])
        caches = _make_cache(lm, [0], kv_bits=8, kv_group_size=64)
        assert type(caches[3]).__name__ == 'BatchQuantizedKVCache' and type(caches[7]).__name__ == 'BatchKVCache'
        outs.append(_steps(lm, ids, caches))
    assert mx.isfinite(outs[1]).all().item()
    assert _off(outs[1], outs[0]) < 1e-3
    assert _off(outs[0], floats) < 0.1
    assert kv_bits.STATS['checked'] >= kv_bits.CHECKS
    if mx.metal.is_available():
        assert kv_bits.STATS['differed'] == kv_bits.STATS['spread'] == 0
        assert kv_bits.STATS['unpacked'] >= 1 and kv_bits.STATS['quantized'] >= 1
    else:
        # MLX CUDA's quantized_matmul has no implementation for mlx-vlm's 8-bit attention in one call: the decode's
        # is found out once and the spread-out product serves it (the logits above are still the per-head path's)
        assert kv_bits.STATS['differed'] == 1 and kv_bits.STATS['quantized'] == 0
        assert kv_bits.STATS['spread'] >= 1 and kv_bits.STATS['unpacked'] >= 1
