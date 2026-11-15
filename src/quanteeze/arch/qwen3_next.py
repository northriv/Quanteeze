# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Inference hooks for mlx-lm's Qwen3-Next MoE blocks (`model_type: qwen3_next`)."""
from mlx_lm.models.qwen3_next import Qwen3NextDecoderLayer, Qwen3NextSparseMoeBlock

from .. import expert_qmv, prefill


def _before_moe(layer, x, mask=None, cache=None):
    if layer.is_linear:
        r = layer.linear_attn(layer.input_layernorm(x), mask, cache)
    else:
        r = layer.self_attn(layer.input_layernorm(x), mask, cache)
    h = x + r
    return layer.post_attention_layernorm(h), h


def _from_moe(layer, normed, h):
    return h + layer.mlp(normed)


prefill.split(Qwen3NextDecoderLayer, _before_moe, _from_moe)


def install(model):
    """Install `expert_qmv` switch hooks on each MoE block of `model`; returns `{layer_idx: block}`."""
    blocks = {}
    for name, module in model.named_modules():
        parts = name.split('.')
        if isinstance(module, Qwen3NextSparseMoeBlock) and 'layers' in parts and 'mtp' not in parts:
            layer = int(parts[parts.index('layers') + 1])
            object.__setattr__(module, '_layer_idx', layer)
            expert_qmv.install_switch(module.switch_mlp)
            blocks[layer] = module
    return blocks
