# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Inference hooks for mlx-vlm's Qwen3.5 and Qwen3.6 MoE blocks (`model_type: qwen3_5_moe`)."""
from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeDecoderLayer, Qwen3_5MoeSparseMoeBlock

from .. import expert_qmv, prefill


def _before_moe(layer, x, mask=None, cache=None, position_ids=None, position_embeddings=None):
    if layer.is_linear:
        r = layer.linear_attn(layer.input_layernorm(x), mask, cache)
    else:
        r = layer.self_attn(layer.input_layernorm(x), mask=mask, cache=cache,
                            position_ids=position_ids, position_embeddings=position_embeddings)
    h = x + r
    return layer.post_attention_layernorm(h), h


def _from_moe(layer, normed, h):
    return h + layer.mlp(normed)


prefill.split(Qwen3_5MoeDecoderLayer, _before_moe, _from_moe)


def install(model):
    """Install `expert_qmv` switch hooks on each MoE block of `model`; returns `{layer_idx: block}`."""
    blocks = {}
    for name, module in model.named_modules():
        parts = name.split('.')
        if type(module) is Qwen3_5MoeSparseMoeBlock and 'layers' in parts and 'mtp' not in parts:
            layer = int(parts[parts.index('layers') + 1])
            object.__setattr__(module, '_layer_idx', layer)
            expert_qmv.install_switch(module.switch_mlp)
            blocks[layer] = module
    return blocks
