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
import threading
import weakref

import mlx.core as mx
from mlx_vlm.models.qwen4_exp import language as _language
from mlx_vlm.models.qwen4_exp.language import (Qwen4ExpBatchInvariantForward, Qwen4ExpDecoderLayer,
                                               Qwen4ExpSparseMoeBlock)

from .. import expert_qmv, linear_attn, per_thread, prefill, requant

COMPILE = [True]                        # quanteeze.serve: QUANTEEZE_COMPILE_DECODE
EVAL_LAYERS = [12]                      # quanteeze.serve: QUANTEEZE_EVAL_LAYERS
COMPILE_STATS = {'compiled': 0, 'hyper': 0, 'refused': 0}
VERIFY_ROWS_MOST = 8
VERIFYING = per_thread.Field('verify')


class _LocalFlag(threading.local):
    def __init__(self):
        self.on = False


_CAPTURE_STATE = _LocalFlag()


class _Capturing:
    def __getitem__(self, _i):
        return _CAPTURE_STATE.on

    def __setitem__(self, _i, val):
        _CAPTURE_STATE.on = bool(val)


CAPTURING = _Capturing()


def _hc_mean(hidden, hc_count, hidden_size):
    """Average `hidden` across the hyper-connection streams `[..., hc_count * hidden_size]` -> `[..., hidden_size]`."""
    return mx.mean(hidden.reshape(*hidden.shape[:-1], hc_count, hidden_size), axis=-2)


def refused(what, error):
    """A graph that would not compile: said once, counted each time."""
    if not COMPILE_STATS['refused']:
        print(f'quanteeze: {what} would not compile ({type(error).__name__}: {error}); built op by op',
              file=sys.stderr, flush=True)
    COMPILE_STATS['refused'] += 1


def _router_and_shared(forward, block, x):
    from .. import cuda_moe, cuda_router
    indices, scores = cuda_router.routed(forward._linear(block.gate, x), block.top_k)
    return indices, scores, cuda_moe.gated_shared(forward, block, x)


@mx.compile
def _weighted(output, scores, shared):
    return (output * scores[..., None]).sum(axis=-2) + shared


class _BatchInvariantForward(Qwen4ExpBatchInvariantForward):
    """Batch-invariant decode and verify forward with fused linear attention, dense QSA attention,
    compiled decode routing/shared-expert graphs, and custom Metal/CUDA expert kernels."""

    def verify(self, language_model, inputs, *, cache=None, **kwargs):
        was, VERIFYING[0] = VERIFYING[0], True
        try:
            return super().verify(language_model, inputs, cache=cache, **kwargs)
        finally:
            VERIFYING[0] = was

    def _model(self, model, inputs, cache, inputs_embeds, position_ids, capture_layer_ids=None, hidden_sink=None):
        """mlx-vlm's _model, a one-token step's layers handed to the GPU EVAL_LAYERS at a time."""
        hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        hidden = mx.tile(hidden, (1, 1, model.args.hc_count))
        if cache is None:
            cache = [None] * len(model.layers)
        fa_mask = _language._create_qwen3_5_attention_mask(hidden, cache[model.fa_idx])
        ssm_mask = _language._create_qwen3_5_ssm_mask(hidden, cache[model.ssm_idx])
        every = max(1, EVAL_LAYERS[0]) if hidden.shape[1] == 1 else 1
        last = len(model.layers) - 1
        capture = set(capture_layer_ids or ())
        for i, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            hidden = self._layer(layer, hidden, inputs, layer_mask, layer_cache, position_ids)
            if hidden_sink is not None and i in capture:
                hidden_sink.append(_hc_mean(hidden, model.args.hc_count, model.args.hidden_size))
            if (i + 1) % every == 0 or i == last:
                mx.async_eval(hidden)
        if hidden_sink is not None and capture_layer_ids == []:
            hidden_sink.append(hidden)
        return hidden

    def __call__(
        self,
        language_model,
        inputs,
        *,
        cache=None,
        inputs_embeds=None,
        position_ids=None,
        capture_layer_ids=None,
        return_hidden=False,
        return_shared_kv=False,
        skip_logits=False,
    ):
        if capture_layer_ids is None:
            return super().__call__(
                language_model,
                inputs,
                cache=cache,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                skip_logits=skip_logits,
            )
        hidden_sink = []
        hidden = self._model(
            language_model.model,
            inputs,
            cache,
            inputs_embeds,
            position_ids,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
        )
        logits_hidden = self._hyper_connection(language_model.model.hyper_connection_mixer, hidden)
        if return_hidden and capture_layer_ids != []:
            hidden_sink.append(logits_hidden)
        if skip_logits:
            logits = None
        elif language_model.args.tie_word_embeddings:
            logits = self._embedding_as_linear(language_model.model.embed_tokens, logits_hidden)
        else:
            logits = self._linear(language_model.lm_head, logits_hidden)
        return _language.LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            shared_kv_states={} if return_shared_kv else None,
        )

    def _gated_delta(self, layer, inputs, mask, cache):
        if linear_attn.ON[0] and linear_attn.applies(layer, inputs, mask, cache):
            return linear_attn.decode(self, layer, inputs, cache)
        if linear_attn.applies_rows(layer, inputs, mask, cache):
            return linear_attn.decode_rows(self, layer, inputs, cache)
        if linear_attn.applies_batch(layer, inputs, mask, cache):
            return linear_attn.decode_batch(self, layer, inputs, cache)
        return super()._gated_delta(layer, inputs, mask, cache)

    def _qsa_attention(self, attention, hidden_states, cache, position_ids, mask):
        from .. import cuda_attn
        if cuda_attn.applies(attention, hidden_states, cache, position_ids, mask):
            return cuda_attn.attend(self, attention, hidden_states, cache, position_ids)
        return super()._qsa_attention(attention, hidden_states, cache, position_ids, mask)

    def _switch_glu(self, switch_mlp, x, indices):
        if getattr(switch_mlp, 'demand', None) is not None:
            return switch_mlp(x, indices)
        non_native = any(
            getattr(p, 'bits', None) is not None and not requant.native(p.bits, getattr(p, 'mode', 'affine'))
            for p in (switch_mlp.gate_proj, switch_mlp.up_proj, switch_mlp.down_proj)
        )
        if (expert_qmv.ON[0] and mx.metal.is_available()) or non_native:
            return switch_mlp(x, indices)
        return super()._switch_glu(switch_mlp, x, indices)

    def _feed_forward(self, feed_forward, x):
        if (isinstance(feed_forward, Qwen4ExpSparseMoeBlock) and COMPILE[0] and x.ndim == 3
                and ((x.shape[1] == 1 and 1 <= x.shape[0] <= linear_attn.ROWS_MOST)
                     or (VERIFYING[0] and x.shape[0] == 1 and 1 < x.shape[1] <= VERIFY_ROWS_MOST))):
            return self._compiled(feed_forward, x)
        return super()._feed_forward(feed_forward, x)

    def _feed_forward_into(self, feed_forward, x, hyper_input, injection):
        if (isinstance(feed_forward, Qwen4ExpSparseMoeBlock) and x.shape[:2] == (1, 1)
                and getattr(feed_forward.switch_mlp, 'demand', None) is None):
            from .. import cuda_moe
            return cuda_moe.feed_forward(self, feed_forward, x, hyper_input, injection)
        return self._inject(self._feed_forward(feed_forward, x), hyper_input, injection)

    def _compiled(self, block, x):
        from .. import verify_qmv
        graphs = per_thread.slots(block).setdefault('_router_and_shared_by_b', {})
        key = (int(x.shape[0]), int(x.shape[1]))
        graph = graphs.get(key)
        if graph is False:
            return super()._feed_forward(block, x)
        if x.shape[1] > 1 and (graph is None or isinstance(graph, int)):
            seen = graph or 0
            if seen < verify_qmv.CHECKS:
                graphs[key] = seen + 1
                indices, scores, shared = _router_and_shared(self, block, x)
                return _weighted(self._experts(block, x, indices), scores, shared)
            ref = weakref.ref(block)
            graphs[key] = graph = mx.compile(lambda h: _router_and_shared(self, ref(), h))
        elif graph is None:
            ref = weakref.ref(block)
            graphs[key] = mx.compile(lambda h: _router_and_shared(self, ref(), h))
            indices, scores, shared = _router_and_shared(self, block, x)
            return _weighted(self._experts(block, x, indices), scores, shared)
        try:
            indices, scores, shared = graph(x)
        except Exception as error:
            graphs[key] = False
            refused(f'layer {getattr(block, "_layer_idx", "?")}\'s router and shared expert', error)
            return super()._feed_forward(block, x)
        COMPILE_STATS['compiled'] += 1
        return _weighted(self._experts(block, x, indices), scores, shared)

    def _experts(self, block, x, indices):
        if getattr(block.switch_mlp, 'demand', None) is not None:
            return block.switch_mlp(x, indices)
        from .. import cuda_experts
        return cuda_experts.switch_glu(self, block.switch_mlp, x, indices)


def _verifying(original):
    """Wrap `LanguageModel._speculative_verify` so verify steps set `VERIFYING[0]`."""
    def verify(self, inputs, cache, sampler=None):
        was, VERIFYING[0] = VERIFYING[0], True
        try:
            return original(self, inputs, cache, sampler)
        finally:
            VERIFYING[0] = was

    verify._quanteeze_verifying = True
    return verify


def padding_on_host():
    """Read `ArraysCache.left_padding` on the host when backed by `_left_padding` and `_left_padding_advance`."""
    from mlx_vlm.models import cache as caches
    from mlx_vlm.models.qwen3_5 import language as qwen35
    current = getattr(qwen35, '_qwen3_5_left_padding_info', None)
    if current is None or not hasattr(caches, 'ArraysCache'):
        return False
    if getattr(current, '_quanteeze_original', None) is not None:
        info = current
    else:
        original, prop = current, caches.ArraysCache.__dict__.get('left_padding')

        def info(cache):
            base = getattr(cache, '_left_padding', None)
            advance = getattr(cache, '_left_padding_advance', None)
            if (prop is None or type(advance) is not int or not isinstance(base, mx.array) or base.ndim == 0
                    or base.size == 0 or getattr(type(cache), 'left_padding', None) is not prop):
                return original(cache)
            qwen35._qwen3_5_refresh_metadata(cache)
            pads = tuple(int(p) - advance for p in base.tolist())
            return pads, max(pads)

        info._quanteeze_original = original
    for module in (qwen35, _language):
        if hasattr(module, '_qwen3_5_left_padding_info'):
            module._qwen3_5_left_padding_info = info
    return True


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


def _verify_position_ids(lm, inputs, cache):
    """Build `[B, L]` (or `[3, B, L]`) RoPE position IDs advancing past `cache` for speculative verify."""
    batch, length = inputs.shape
    cache_entry = cache[lm.model.fa_idx]
    cache_offset = getattr(cache_entry, 'offset', 0)
    if isinstance(cache_offset, mx.array) and cache_offset.ndim > 0:
        offsets = cache_offset[:batch].astype(mx.int64)
    else:
        offsets = mx.full((batch,), int(cache_offset), dtype=mx.int64)
    rope_deltas = getattr(lm, '_rope_deltas', None)
    if rope_deltas is not None:
        offsets = offsets + rope_deltas[:batch].reshape(-1).astype(mx.int64)
    position_ids = offsets[:, None] + mx.arange(length, dtype=mx.int64)[None]
    pos_ids = getattr(lm, '_position_ids', None)
    if pos_ids is not None and pos_ids.ndim == 3:
        position_ids = mx.broadcast_to(position_ids[None], (3, batch, length))
    return position_ids


def _install_dflash():
    """Patch `Qwen4ExpModel` & `LanguageModel` for DFlash/DFlash2 unweighted HC-mean capture and verify."""
    inner_cls = _language.Qwen4ExpModel
    if not getattr(inner_cls, '_quanteeze_dflash', False):
        orig_inner_call = inner_cls.__call__

        def inner_call(
            self,
            inputs,
            inputs_embeds=None,
            mask=None,
            cache=None,
            position_ids=None,
            capture_layer_ids=None,
            hidden_sink=None,
            **kwargs,
        ):
            if hidden_sink is None or not capture_layer_ids:
                return orig_inner_call(
                    self,
                    inputs,
                    inputs_embeds=inputs_embeds,
                    mask=mask,
                    cache=cache,
                    position_ids=position_ids,
                    capture_layer_ids=capture_layer_ids,
                    hidden_sink=hidden_sink,
                    **kwargs,
                )
            hidden_states = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
            if cache is None:
                cache = [None] * len(self.layers)
            hidden_states = mx.tile(hidden_states, (1, 1, self.args.hc_count))
            fa_mask = _language._create_qwen4_exp_attention_mask(hidden_states, cache[self.fa_idx])
            ssm_mask = _language._create_qwen3_5_ssm_mask(hidden_states, cache[self.ssm_idx])
            if mask is not None and isinstance(mask, mx.array) and mask.ndim == 2:
                ssm_mask = mask
            capture = set(capture_layer_ids)
            for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
                layer_mask = ssm_mask if layer.is_linear else fa_mask
                hidden_states = layer(
                    hidden_states,
                    inputs,
                    mask=layer_mask,
                    cache=layer_cache,
                    position_ids=position_ids,
                )
                if index in capture:
                    hidden_sink.append(_hc_mean(hidden_states, self.args.hc_count, self.args.hidden_size))
            return self.hyper_connection_mixer(hidden_states)

        inner_cls.__call__ = inner_call
        inner_cls._quanteeze_dflash = True

    lm_cls = _language.LanguageModel
    if not getattr(lm_cls, '_quanteeze_dflash', False):
        orig_lm_call = lm_cls.__call__
        orig_supports = lm_cls._supports_batch_invariant_decode

        def speculative_verify_dflash_hidden(self, inputs, cache, target_layer_ids):
            mx.async_eval([entry.state for entry in cache])
            position_ids = _verify_position_ids(self, inputs, cache)
            out = _language._QWEN4_BATCH_INVARIANT_FORWARD.verify(
                self,
                inputs,
                cache=cache,
                position_ids=position_ids,
                capture_layer_ids=list(target_layer_ids),
                return_hidden=True,
                skip_logits=True,
            )
            return out.hidden_states[:-1], out.hidden_states[-1], out.gdn_states

        def lm_call(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
            if kwargs.get('speculative_verify'):
                kwargs.pop('speculative_verify', None)
                position_ids = kwargs.pop('position_ids', None)
                if position_ids is None and cache is not None:
                    position_ids = _verify_position_ids(self, inputs, cache)
                return _language._QWEN4_BATCH_INVARIANT_FORWARD.verify(
                    self,
                    inputs,
                    cache=cache,
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    capture_layer_ids=kwargs.get('capture_layer_ids'),
                    return_hidden=bool(kwargs.get('return_hidden', False)),
                    return_shared_kv=bool(kwargs.get('return_shared_kv', False)),
                    skip_logits=bool(kwargs.get('skip_logits', False)),
                )
            if kwargs.get('capture_layer_ids') is not None:
                was, CAPTURING[0] = CAPTURING[0], True
                try:
                    return orig_lm_call(self, inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache, **kwargs)
                finally:
                    CAPTURING[0] = was
            return orig_lm_call(self, inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache, **kwargs)

        def supports_batch_invariant_decode(self):
            if CAPTURING[0]:
                return False
            return orig_supports(self)

        lm_cls.speculative_verify_dflash_hidden = speculative_verify_dflash_hidden
        lm_cls.__call__ = lm_call
        lm_cls._supports_batch_invariant_decode = supports_batch_invariant_decode
        lm_cls._quanteeze_dflash = True


def install(model):
    """Install `_BatchInvariantForward`, DFlash2 capture/verify, and `expert_qmv` switch hooks on `model`."""
    _install_dflash()
    lm = getattr(_language, 'LanguageModel', None)
    if lm is not None and hasattr(lm, '_speculative_verify') and not getattr(
            lm._speculative_verify, '_quanteeze_verifying', False):
        lm._speculative_verify = _verifying(lm._speculative_verify)
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
    padding_on_host()
    return blocks
