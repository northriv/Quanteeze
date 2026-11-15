# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Inference hooks for mlx-vlm's Qwen4-Exp MoE blocks (Qwen3.8 Flash-Next).

Hooks the batch-invariant decode/verify forward so one-token decode and speculative verification use:
- fused GatedDeltaNet linear-attention kernels (`quanteeze.linear_attn`),
- fused dense QSA attention kernels (`quanteeze.cuda_attn`),
- `mx.compile` graphs for the router, shared expert, and weighting (`QUANTEEZE_COMPILE_DECODE`),
- fused hyper-connection and PLE kernels (`quanteeze.hyper`, `quanteeze.cuda_ple_layer`),
- custom Metal/CUDA routed-expert and MoE kernels (`quanteeze.expert_qmv`, `quanteeze.cuda_experts`,
  `quanteeze.cuda_router`, `quanteeze.cuda_moe`).
"""
import sys
import weakref

import mlx.core as mx
from mlx_vlm.models.qwen4_exp import language as _language
from mlx_vlm.models.qwen4_exp.language import (Qwen4ExpBatchInvariantForward, Qwen4ExpDecoderLayer,
                                               Qwen4ExpSparseMoeBlock)

from .. import expert_qmv, linear_attn, prefill, requant

COMPILE = [True]                        # quanteeze.serve: QUANTEEZE_COMPILE_DECODE
EVAL_LAYERS = [12]                      # quanteeze.serve: QUANTEEZE_EVAL_LAYERS
COMPILE_STATS = {'compiled': 0, 'hyper': 0, 'refused': 0}


def refused(what, error):
    """A graph that would not compile: said once, counted each time."""
    if not COMPILE_STATS['refused']:
        print(f'quanteeze: {what} would not compile ({type(error).__name__}: {error}); built op by op',
              file=sys.stderr, flush=True)
    COMPILE_STATS['refused'] += 1


def _router_and_shared(forward, block, x):
    from .. import cuda_router
    indices, scores = cuda_router.routed(forward._linear(block.gate, x), block.top_k)
    shared = forward._feed_forward(block.shared_expert, x)
    return indices, scores, mx.sigmoid(forward._linear(block.shared_expert_gate, x)) * shared


@mx.compile
def _weighted(output, scores, shared):
    return (output * scores[..., None]).sum(axis=-2) + shared


class _BatchInvariantForward(Qwen4ExpBatchInvariantForward):
    """Batch-invariant decode and verify forward with fused linear attention, dense QSA attention,
    compiled decode routing/shared-expert graphs, and custom Metal/CUDA expert kernels."""

    def _model(self, model, inputs, cache, inputs_embeds, position_ids):
        """mlx-vlm's _model, a one-token step's layers handed to the GPU EVAL_LAYERS at a time."""
        hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        hidden = mx.tile(hidden, (1, 1, model.args.hc_count))
        if cache is None:
            cache = [None] * len(model.layers)
        fa_mask = _language._create_qwen3_5_attention_mask(hidden, cache[model.fa_idx])
        ssm_mask = _language._create_qwen3_5_ssm_mask(hidden, cache[model.ssm_idx])
        every = max(1, EVAL_LAYERS[0]) if hidden.shape[1] == 1 else 1
        last = len(model.layers) - 1
        for i, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            hidden = self._layer(layer, hidden, inputs, layer_mask, layer_cache, position_ids)
            if (i + 1) % every == 0 or i == last:
                mx.async_eval(hidden)
        return hidden

    def _gated_delta(self, layer, inputs, mask, cache):
        if linear_attn.ON[0] and linear_attn.applies(layer, inputs, mask, cache):
            return linear_attn.decode(self, layer, inputs, cache)
        return super()._gated_delta(layer, inputs, mask, cache)

    def _qsa_attention(self, attention, hidden_states, cache, position_ids, mask):
        from .. import cuda_attn
        if cuda_attn.applies(attention, hidden_states, cache, position_ids, mask):
            return cuda_attn.attend(self, attention, hidden_states, cache, position_ids)
        return super()._qsa_attention(attention, hidden_states, cache, position_ids, mask)

    def _switch_glu(self, switch_mlp, x, indices):
        if getattr(switch_mlp, 'demand', None) is not None:
            return switch_mlp(x, indices)
        bits = getattr(switch_mlp.gate_proj, 'bits', None)
        mode = getattr(switch_mlp.gate_proj, 'mode', 'affine')
        if (expert_qmv.ON[0] and mx.metal.is_available()) or (bits is not None and not requant.native(bits, mode)):
            return switch_mlp(x, indices)
        return super()._switch_glu(switch_mlp, x, indices)

    def _feed_forward(self, feed_forward, x):
        if isinstance(feed_forward, Qwen4ExpSparseMoeBlock) and COMPILE[0] and x.shape[:2] == (1, 1):
            return self._compiled(feed_forward, x)
        return super()._feed_forward(feed_forward, x)

    def _feed_forward_into(self, feed_forward, x, hyper_input, injection):
        if (isinstance(feed_forward, Qwen4ExpSparseMoeBlock) and x.shape[:2] == (1, 1)
                and getattr(feed_forward.switch_mlp, 'demand', None) is None):
            from .. import cuda_moe
            return cuda_moe.feed_forward(self, feed_forward, x, hyper_input, injection)
        return self._inject(self._feed_forward(feed_forward, x), hyper_input, injection)

    def _compiled(self, block, x):
        graph = getattr(block, '_router_and_shared', None)
        if graph is None:
            ref = weakref.ref(block)
            object.__setattr__(block, '_router_and_shared', mx.compile(lambda h: _router_and_shared(self, ref(), h)))
            indices, scores, shared = _router_and_shared(self, block, x)
            return _weighted(self._experts(block, x, indices), scores, shared)
        if not graph:
            return super()._feed_forward(block, x)
        try:
            indices, scores, shared = graph(x)
        except Exception as error:
            object.__setattr__(block, '_router_and_shared', False)
            refused(f'layer {getattr(block, "_layer_idx", "?")}\'s router and shared expert', error)
            return super()._feed_forward(block, x)
        COMPILE_STATS['compiled'] += 1
        return _weighted(self._experts(block, x, indices), scores, shared)

    def _experts(self, block, x, indices):
        from .. import cuda_experts
        return cuda_experts.switch_glu(self, block.switch_mlp, x, indices)


def _before_moe(layer, hidden_states, input_ids, mask=None, cache=None, position_ids=None):
    if 'ple' in layer:
        hidden_states = hidden_states + layer.ple(hidden_states, input_ids, cache, mask)
    mixed, hyper_input, injection_weights = layer.attn_hyper_connection(hidden_states)
    if layer.is_linear:
        branch = layer.linear_attn(mixed, mask=mask, cache=cache)
    else:
        branch = layer.self_attn(mixed, mask=mask, cache=cache, position_ids=position_ids)
    injection = branch[..., None, :] * injection_weights[..., None]
    hidden_states = hyper_input + injection.reshape(*hyper_input.shape)
    return layer.mlp_hyper_connection(hidden_states)


def _from_moe(layer, mixed, hyper_input, injection_weights):
    branch = layer.mlp(mixed)
    injection = branch[..., None, :] * injection_weights[..., None]
    return hyper_input + injection.reshape(*hyper_input.shape)


prefill.split(Qwen4ExpDecoderLayer, _before_moe, _from_moe)


def install(model):
    """Install `_BatchInvariantForward` and `expert_qmv` switch hooks on `model`."""
    blocks = {}
    for name, module in model.named_modules():
        parts = name.split('.')
        if isinstance(module, Qwen4ExpSparseMoeBlock) and 'layers' in parts and 'mtp' not in parts:
            layer = int(parts[parts.index('layers') + 1])
            object.__setattr__(module, '_layer_idx', layer)
            expert_qmv.install_switch(module.switch_mlp)
            blocks[layer] = module
    forward = _language._QWEN4_BATCH_INVARIANT_FORWARD
    if type(forward) is Qwen4ExpBatchInvariantForward:
        forward.__class__ = _BatchInvariantForward
    return blocks
