# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Inference hooks for mlx-vlm's GLM-5.3 MoE blocks (`model_type: glm5_next`).

Installs the fused Metal kernels for sigmoid MoE routing (`quanteeze.sigmoid_router`), GatedDeltaNet
linear attention (`quanteeze.glm_linear_attn`), hyper-connections (`quanteeze.glm_hyper`), DSA
sparse-attention indexer (`quanteeze.glm_indexer`), and routed-expert QMV (`quanteeze.expert_qmv`).
"""
from mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection
from mlx_vlm.models.glm5_next.language import Glm5NextIndexer, Glm5NextLinearAttention, Glm5NextMoE, MoEGate

from .. import expert_qmv, glm_hyper, glm_indexer, glm_linear_attn, sigmoid_router


class Gate(MoEGate):
    """MoEGate with a decode token's logits and route in `quanteeze.sigmoid_router`'s kernels, where
    their shapes matched the stock ones bit for bit."""

    def __call__(self, x):
        return sigmoid_router.gate_route(self, sigmoid_router.gate_logits(self, x))


class LinearAttention(Glm5NextLinearAttention):
    """Glm5NextLinearAttention with a decode step of one request in `quanteeze.glm_linear_attn`'s
    kernels, where its shape matched the stock one bit for bit."""

    def __call__(self, x, mask=None, cache=None):
        if not glm_linear_attn.applies(self, x, mask, cache):
            return super().__call__(x, mask=mask, cache=cache)
        return glm_linear_attn.step(
            self, x, cache, lambda: super(LinearAttention, self).__call__(x, mask=mask, cache=cache)
        )


class Hyper(HyperConnection):
    """HyperConnection with a step of one token in `quanteeze.glm_hyper`'s kernels, where its shapes
    matched the stock ones bit for bit."""

    def apply_branch(self, x, norm, branch, *args, **kwargs):
        if not glm_hyper.applies(self, x):
            return super().apply_branch(x, norm, branch, *args, **kwargs)
        return glm_hyper.apply_branch(self, x, norm, branch, *args, **kwargs)


class Indexer(Glm5NextIndexer):
    """Glm5NextIndexer with a decode step of one request choosing its positions in
    `quanteeze.glm_indexer`'s kernel, where that path matched the stock one bit for bit."""

    def __call__(self, x, q_resid, padding_mask=None, cache=None, pool_cache=None, offset=0):
        if not glm_indexer.applies(self, x, padding_mask, cache, pool_cache):
            return super().__call__(x, q_resid, padding_mask, cache, pool_cache, offset)
        return glm_indexer.step(
            self, x, q_resid, cache, pool_cache, offset,
            lambda: super(Indexer, self).__call__(x, q_resid, padding_mask, cache, pool_cache, offset),
        )


def install(model):
    """Give each MoE block of `model` `expert_qmv.install_switch`, each gate `Gate`, each
    linear-attention layer `LinearAttention`, each hyper-connection `Hyper` and each indexer
    `Indexer`; returns `{layer index: block}`.  An MTP head's blocks are left alone, its gates and
    the rest not."""
    blocks = {}
    for name, module in model.named_modules():
        parts = name.split('.')
        if type(module) is MoEGate:
            module.__class__ = Gate
        if type(module) is Glm5NextLinearAttention:
            module.__class__ = LinearAttention
        if type(module) is HyperConnection:
            module.__class__ = Hyper
        if type(module) is Glm5NextIndexer:
            module.__class__ = Indexer
        if type(module) is Glm5NextMoE and 'layers' in parts and 'mtp' not in parts:
            layer = int(parts[parts.index('layers') + 1])
            object.__setattr__(module, '_layer_idx', layer)
            expert_qmv.install_switch(module.switch_mlp)
            blocks[layer] = module
    return blocks
