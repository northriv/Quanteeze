# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Inference hooks for mlx-vlm's DeepSeek V4 MoE blocks (`model_type: deepseek_v4`)."""
from mlx_vlm.models.deepseek_v4.language import DeepseekV4MoE

from .. import expert_qmv


def install(model):
    """Install `expert_qmv` switch hooks on each MoE block of `model`; returns `{layer_idx: block}`."""
    blocks = {}
    for name, module in model.named_modules():
        parts = name.split('.')
        if type(module) is DeepseekV4MoE and 'layers' in parts and 'mtp' not in parts:
            layer = int(parts[parts.index('layers') + 1])
            object.__setattr__(module, '_layer_idx', layer)
            expert_qmv.install_switch(module.switch_mlp)
            blocks[layer] = module
    return blocks
