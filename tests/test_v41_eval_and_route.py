# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""DeepSeek V4.1: a short pass handed to the GPU layer by layer as it is built (PLAIN_EVAL, QUANTEEZE_EVAL_LAYERS on
Metal) computes what the whole pass did; the CUDA routing kernel takes an image mask's correction bias a row
(cuda_v41.takes_bias), only once that matched the ops at install.  A tiny random model, no download."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_vlm.models.deepseek_v41.config import ModelConfig
from mlx_vlm.models.deepseek_v41.engram import NgramHashState
from mlx_vlm.models.deepseek_v41.language import LanguageModel

from quanteeze import cuda_v41
from quanteeze.arch import deepseek_v41

ARGS = dict(vocab_size=128, hidden_size=64, moe_intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
            head_dim=32, qk_rope_head_dim=8, q_lora_rank=32, o_lora_rank=32, o_groups=2, n_routed_experts=16,
            num_experts_per_tok=4, sliding_window=8, compress_ratios=[0, 2, 2, 2, 1, 1], kv_source_layer_ids=[1, 4],
            index_source_layer_ids=[1, 3, 4], index_n_heads=2, index_head_dim=32, index_topk=4,
            candidate_source_layer_id=4, candidate_topk_blocks=2, candidate_block_size=2, hc_mult=2,
            hc_sinkhorn_iters=3, engram_layer_ids=[1, 3], engram_num_embeddings=[300, 300], engram_max_ngram_size=3,
            engram_vocab_size=20, engram_n_heads=2, engram_head_dim=16, engram_pad_token_id=2,
            engram_compressed_vocab_size=128, max_position_embeddings=512)


def _model():
    mx.random.seed(0)
    lm = LanguageModel(ModelConfig(**ARGS))
    lm.engram_hash = NgramHashState(lm.config, lm.layout, token_map=list(range(ARGS['vocab_size'])))
    lm.head.weight = mx.random.normal(lm.head.weight.shape) * 0.02     # mlx-vlm starts it at zeros
    mx.eval(lm.parameters())
    deepseek_v41.speculative(lm)
    return lm


def _run(lm, tokens):
    cache, out = lm.make_cache(), []
    out.append(lm(tokens[:, :8], cache=cache).logits)
    for i in range(8, tokens.shape[1]):
        out.append(lm(tokens[:, i:i + 1], cache=cache).logits)
    mx.eval(out)
    return out


def test_a_short_pass_handed_to_the_gpu_by_layers_computes_what_the_whole_pass_did(monkeypatch):
    lm = _model()
    base = tree_flatten(lm.parameters())
    tokens = mx.array(np.random.default_rng(1).integers(3, ARGS['vocab_size'], (1, 12)))
    monkeypatch.setattr(deepseek_v41.mx.metal, 'is_available', lambda: True)    # (PLAIN_EVAL is Metal's)
    handed = []
    real = mx.async_eval
    monkeypatch.setattr(deepseek_v41.mx, 'async_eval', lambda *a: (handed.append(1), real(*a))[1])
    runs, counts = {}, {}
    for n in (0, 1, 4):
        monkeypatch.setattr(deepseek_v41, 'PLAIN_EVAL', [n])
        lm.load_weights(base)
        handed.clear()
        runs[n] = _run(lm, tokens)
        counts[n] = len(handed)
    for n in (1, 4):
        assert all(mx.array_equal(a, b).item() for a, b in zip(runs[n], runs[0]))
    assert counts[1] > counts[4] > counts[0]                 # layers handed on: every one, every 4th, none


def test_the_routing_kernel_takes_a_bias_a_row_only_once_matched(monkeypatch):
    logits = mx.zeros((1, 3, 16), dtype=mx.float32)
    monkeypatch.setattr(cuda_v41, 'ROUTE_ROWS_BIAS', [False])
    assert cuda_v41.takes_bias(logits, mx.zeros((16,)))
    assert not cuda_v41.takes_bias(logits, mx.zeros((15,)))
    assert not cuda_v41.takes_bias(logits, mx.zeros((1, 3, 16)))          # a mask's rows: not until matched
    monkeypatch.setattr(cuda_v41, 'ROUTE_ROWS_BIAS', [True])
    assert cuda_v41.takes_bias(logits, mx.zeros((1, 3, 16)))
    assert not cuda_v41.takes_bias(logits, mx.zeros((1, 2, 16)))
    assert 'bias[(bias_ndim > 1 ? row * E : 0) + e]' in cuda_v41.ROUTE_SOURCE


def test_install_route_leaves_the_masked_bias_off_where_it_cannot_check(monkeypatch):
    monkeypatch.setattr(cuda_v41, '_gpu', lambda: False)
    monkeypatch.setattr(cuda_v41, 'ROUTE_ROWS_BIAS', [True])
    assert not cuda_v41.install_route([object()])
    assert cuda_v41.ROUTE_ROWS_BIAS == [False] and cuda_v41.ROUTE == [False]
