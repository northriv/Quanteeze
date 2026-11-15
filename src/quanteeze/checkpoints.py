# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Checkpoints mlx-vlm's loader takes as they are named and stored, whoever made them.

A checkpoint may carry its multi-token-prediction head beside the model's weights.  mlx-vlm
loads that head as a drafter of its own (quanteeze.mtp, load_drafter: under `mtp.`,
`language_model.mtp.` or `model.mtp.`), and a family's sanitize leaves it out of the model --
under the name the family's converter writes: Qwen4-Exp's drops `mtp.` only, so TensorFold's
8-bit Flash-Next, whose head is `language_model.mtp.`, could not be loaded at all ('Received 76
parameters not in model').  install() wraps mlx_vlm.utils.sanitize_weights: after the
model's own sanitize, a weight with an `mtp` component in its path is left out when the model
holds no such module.

A norm may scale by (1 + w), its weight stored about 0 -- Qwen4-Exp's RMSNorm, Gemma's -- and a
converter may fold the 1 into the weight: TensorFold's 8-bit Flash-Next stores every such norm
of the text model at +1.000 against mlx-community's 4-bit (hc_norm, q/k norms, the indexer's,
PLE's, the MTP head's; the gated norm and the vision tower's, which scale by w, as they are), and
served so its streams came out twice as large from the first layer on, its text garbled.
No config says which; so the norms that add the 1 are found by what they do -- a
norm given a zero weight that still gives something back -- and when their weights in the
checkpoint average above 1/2 (folded about 1, not about 0) the 1 is taken off.

quanteeze installs it as mlx_vlm.utils is imported (quanteeze/__init__.py).
"""
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

FOLDED = 0.5         # the mean weight of the norms that add 1 above which the checkpoint folded it in


def _mtp(key):
    return 'mtp' in key.split('.')


def install(utils):
    """utils (mlx_vlm.utils): its sanitize_weights then leaves out an MTP head the model does not hold,
    and takes a folded 1 off the norms that add it (unfold)."""
    if getattr(utils.sanitize_weights, '_quanteeze', False):
        return
    own = utils.sanitize_weights

    def sanitize_weights(model_obj, weights, config=None):
        weights = own(model_obj, weights, config)
        if config is not None or not hasattr(model_obj, 'parameters'):
            return weights
        if any(_mtp(name) for name, _ in tree_flatten(model_obj.parameters())):
            return unfold(model_obj, weights)
        left = [k for k in weights if _mtp(k)]
        if left:
            print(f'quanteeze: {len(left)} weights of an MTP head the model does not hold left out '
                  f'(the drafter reads them on its own): {left[0]}, ...', file=sys.stderr, flush=True)
        return unfold(model_obj, {k: v for k, v in weights.items() if not _mtp(k)})

    sanitize_weights._quanteeze = True
    utils.sanitize_weights = sanitize_weights


def offset_norms(model):
    """Names of `model`'s norms that scale by (1 + weight): each norm module with a weight of one
    axis, given a zero weight for a moment, still gives back something of an input."""
    out = []
    for name, module in model.named_modules():
        weight = module.get('weight') if isinstance(module, nn.Module) else None
        if 'norm' not in type(module).__name__.lower() or not isinstance(weight, mx.array) or weight.ndim != 1:
            continue
        module['weight'] = mx.zeros(weight.shape, mx.float32)
        try:
            y = module(mx.ones((1, weight.shape[0]), mx.float32))
            if isinstance(y, mx.array) and float(mx.abs(y).max()) > 0:
                out.append(name)
        except Exception:                          # a norm that takes more than its input (gated)
            pass
        finally:
            module['weight'] = weight
    return out


def unfold(model, weights):
    """weights with the 1 taken off the norms of `model` that add it, when the checkpoint folded
    it in (their weights average above FOLDED)."""
    keys = [f'{n}.weight' for n in offset_norms(model) if f'{n}.weight' in weights]
    if not keys:
        return weights
    mean = float(np.mean([float(weights[k].astype(mx.float32).mean()) for k in keys]))
    if mean <= FOLDED:
        return weights
    print(f'quanteeze: the checkpoint folded the 1 into {len(keys)} norms that add it (their weights average '
          f'{mean:.3f}): taken off', file=sys.stderr, flush=True)
    keys = set(keys)
    return {k: (v - 1).astype(v.dtype) if k in keys else v for k, v in weights.items()}
