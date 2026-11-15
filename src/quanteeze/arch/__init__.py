# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Per-architecture MoE inference hooks, and model inspection helpers (`quanteeze.arch`)."""
from dataclasses import dataclass, field
import importlib

from .. import expert_qmv

PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
USUAL = (('mlp', 'switch_mlp'), ('ffn', 'switch_mlp'), ('block_sparse_moe', 'switch_mlp'),
         ('feed_forward', 'switch_mlp'), ('mlp', 'experts'))


@dataclass(frozen=True)
class Arch:
    model_type: str
    hooks: str = ''
    efforts: dict = field(default_factory=dict)
    not_held: str = ''

    def install(self, model):
        """Install architecture-specific inference hooks and `expert_qmv` switch hooks on `model`."""
        blocks = {}
        if self.hooks:
            blocks = importlib.import_module(self.hooks).install(model) or {}
        for i, layer in enumerate(decoder_layers(model)):
            sw = switch_of(layer)
            if sw is not None:
                expert_qmv.install_switch(sw)
                blocks.setdefault(i, layer)
        return blocks

    def prepare(self, model, path):
        """Make the loaded model's routed experts and projections match what `path` stores (e.g.
        DeepSeek V4.1's FP4 experts and FP8 projections)."""
        if not self.hooks:
            return False
        found = getattr(importlib.import_module(self.hooks), 'prepare', None)
        return bool(found(model, path)) if found is not None else False


ARCHS = {}


def register(arch):
    ARCHS[arch.model_type] = arch
    return arch


register(Arch('qwen4_exp', 'quanteeze.arch.qwen4_exp', {'high': 'xhigh', 'minimal': 'low'}))
register(Arch('qwen3_5_moe', 'quanteeze.arch.qwen3_5_moe'))
register(Arch('qwen3_next', 'quanteeze.arch.qwen3_next'))
register(Arch('deepseek_v4', 'quanteeze.arch.deepseek_v4'))
register(Arch('deepseek_v41', 'quanteeze.arch.deepseek_v41',
              not_held=r'^(mtp|vision|aligner)\.|^embed\.weight$'))
register(Arch('glm5_next', 'quanteeze.arch.glm5_next'))


def get(model_type):
    """The `Arch` of `model_type`, or `None`."""
    return ARCHS.get(model_type)


def of(model):
    """The `Arch` of a loaded `model` by its config's `model_type`, or `None`."""
    config = getattr(model, 'config', None) or getattr(model, 'args', None)
    return get(getattr(config, 'model_type', None) if config is not None else None)


def resolve(model):
    """The `Arch` of a loaded `model`, or a generic `Arch` when `model` has routed-expert `SwitchGLU`
    layers of another `model_type`, or `None` for a dense model."""
    known = of(model)
    if known is not None:
        return known
    try:
        layers = decoder_layers(model)
    except AttributeError:
        return None
    if not any(switch_of(layer) is not None for layer in layers):
        return None
    config = getattr(model, 'config', None) or getattr(model, 'args', None)
    return Arch(str(getattr(config, 'model_type', None) or 'generic'))


def load(path, **kwargs):
    """`mlx_vlm.load(path, **kwargs)` followed by `Arch.prepare(model, path)`."""
    from mlx_vlm import load as mlx_vlm_load
    model, processor = mlx_vlm_load(path, **kwargs)
    known = resolve(model)
    if known is not None:
        known.prepare(model, path)
    return model, processor


ROW_BY_ROW = '.engram.embed.'   # tables read from the checkpoint row by row (OffloadedEngramEmbedding)


def in_memory(model):
    """`model`'s parameters `(key, array)` excluding tables read from the checkpoint row by row."""
    from mlx.utils import tree_flatten
    return [(k, a) for k, a in tree_flatten(model.parameters()) if ROW_BY_ROW not in k]


def language_model(model):
    lm = getattr(model, 'language_model', None)
    return lm if lm is not None else model


def decoder_layers(model):
    lm = language_model(model)
    inner = getattr(lm, 'model', None)
    return inner.layers if inner is not None and hasattr(inner, 'layers') else lm.layers


def is_experts(module):
    """Whether `module` is a routed-expert `SwitchGLU` or `DemandSwitchGLU`."""
    if type(module).__name__ == 'DemandSwitchGLU':
        return True
    projections = [getattr(module, p, None) for p in PROJECTIONS]
    return all(getattr(getattr(p, 'weight', None), 'ndim', 0) in (3, 4) for p in projections)


def _walk(obj, path):
    for part in path:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def experts_slot(layer):
    """`(parent, attribute)` of the routed experts a decoder layer holds, or `None` for a dense layer."""
    for parent_path, attr in USUAL:
        parent = getattr(layer, parent_path, None)
        if parent is not None and is_experts(getattr(parent, attr, None)):
            return parent, attr
    named = getattr(layer, 'named_modules', None)
    if named is None:
        return None
    for name, module in sorted(named(), key=lambda nm: nm[0].count('.')):
        if name and is_experts(module):
            path = name.split('.')
            return _walk(layer, path[:-1]), path[-1]
    return None


def switch_of(layer):
    """The routed experts a decoder layer holds, or `None` for a dense layer."""
    slot = experts_slot(layer)
    return getattr(*slot) if slot else None


def set_switch(layer, module):
    """Replace the routed experts of `layer` with `module`."""
    slot = experts_slot(layer)
    if slot is None:
        raise ValueError('the layer holds no routed experts')
    setattr(slot[0], slot[1], module)
