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
import functools
import sys
import weakref

import mlx.core as mx
from mlx_vlm.models.qwen4_exp import language as _language
from mlx_vlm.models.qwen4_exp.language import (Qwen4ExpBatchInvariantForward, Qwen4ExpDecoderLayer,
                                               Qwen4ExpSparseMoeBlock)

from .. import expert_qmv, linear_attn, per_thread, prefill, requant

COMPILE = [True]                        # quanteeze.serve: QUANTEEZE_COMPILE_DECODE
EVAL_LAYERS = [12]                      # quanteeze.serve: QUANTEEZE_EVAL_LAYERS
VERIFY_LEAD = [0]                       # quanteeze.serve: QUANTEEZE_VERIFY_LEAD_LAYERS (-1: auto, _verify_lead)
VERIFY_LEAD_ON_CUDA = 4                 # auto default on a single CUDA GPU
PACED = dict(waits=0)                   # layers of verify passes the host waited before building
COMPILE_STATS = {'compiled': 0, 'hyper': 0, 'refused': 0}
VERIFY_ROWS_MOST = 8
VERIFYING = per_thread.Field('verify')
CAPTURE = per_thread.Field('capture')


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

    @staticmethod
    def _verify_lead(hidden):
        """Layers the host may build ahead of the GPU during a single-request MTP verify pass (0: unbounded).

        With `QUANTEEZE_VERIFY_LEAD_LAYERS`, before building a layer of a verify pass (`VERIFYING` and
        `hidden.shape[0] == 1` and `hidden.shape[1] >= 2`), the host waits for the input to the layer that many
        steps behind. `auto` (`-1`) enables `VERIFY_LEAD_ON_CUDA` (4) on a single CUDA GPU so freed transient
        buffers do not pile up in the driver pool ahead of execution and evict managed-memory experts; `0` on
        Apple Silicon or multi-GPU runs.
        """
        if not VERIFYING[0] or hidden.shape[0] != 1 or hidden.shape[1] < 2:
            return 0
        lead = VERIFY_LEAD[0]
        if lead >= 0:
            return lead
        from .. import cuda_kernels, parallel
        return VERIFY_LEAD_ON_CUDA if cuda_kernels.available() and parallel.size() == 1 else 0

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
        lead, behind = self._verify_lead(hidden), []
        for i, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
            if lead:
                behind.append(hidden)
                if len(behind) > lead:
                    mx.eval(behind.pop(0))
                    PACED['waits'] += 1
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
        if capture_layer_ids is None and CAPTURE[0] is not None:
            capture_layer_ids = CAPTURE[0]
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
        if return_hidden and capture_layer_ids != []:
            hidden_sink.append(hidden)
        if skip_logits:
            logits = None
        else:
            logits_hidden = self._hyper_connection(language_model.model.hyper_connection_mixer, hidden)
            if language_model.args.tie_word_embeddings:
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

        def _eval_cache_states(cache):
            if cache is None:
                return
            states = [entry.state for entry in cache if entry is not None and hasattr(entry, 'state')]
            if states:
                mx.async_eval(states)

        def speculative_verify_dflash_hidden(self, inputs, cache, target_layer_ids):
            _eval_cache_states(cache)
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
                _eval_cache_states(cache)
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
            capture_ids = kwargs.get('capture_layer_ids')
            if capture_ids is not None:
                was, CAPTURE[0] = CAPTURE[0], list(capture_ids)
                try:
                    return orig_lm_call(self, inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache, **kwargs)
                finally:
                    CAPTURE[0] = was
            return orig_lm_call(self, inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache, **kwargs)

        lm_cls.speculative_verify_dflash_hidden = speculative_verify_dflash_hidden
        lm_cls.__call__ = lm_call
        lm_cls._quanteeze_dflash = True


def _trim_index_blocks(cache, new_len):
    """Preserve already-summarized QSA blocks strictly before `new_len` across speculative rollback."""
    blocks = getattr(cache, 'index_block_keys', None)
    if blocks is None:
        return
    ratio = getattr(cache, 'index_block_ratio', None)
    if not ratio or ratio <= 0:
        cache.clear_index_blocks()
        return
    keep = int(new_len) // int(ratio)
    if keep <= 0:
        cache.clear_index_blocks()
    elif blocks.shape[2] > keep:
        cache.index_block_keys = blocks[:, :, :keep]


def _install_qsa_cache():
    """Patch QSA KV caches to preallocate indexer storage and retain valid block summaries on trim."""
    qsa_cls = _language.QSAKVCache
    if not getattr(qsa_cls, '_quanteeze_qsa_cache', False):
        orig_qsa_trim = qsa_cls.trim

        @functools.wraps(orig_qsa_trim)
        def qsa_trim(self, n):
            n = min(self.offset, n)
            super(_language.QSAKVCache, self).trim(n)
            if self.index_keys is not None:
                self.index_keys = self.index_keys[:, : self.offset]
                self.index_position_ids = self.index_position_ids[..., : self.offset]
            _trim_index_blocks(self, self.offset)
            return n

        qsa_cls.trim = qsa_trim
        qsa_cls._quanteeze_qsa_cache = True

    qsa_quant_cls = _language.QSAQuantizedKVCache
    if not getattr(qsa_quant_cls, '_quanteeze_qsa_cache', False):
        orig_quant_trim = qsa_quant_cls.trim

        @functools.wraps(orig_quant_trim)
        def qsa_quant_trim(self, n):
            n = min(self.offset, n)
            super(_language.QSAQuantizedKVCache, self).trim(n)
            if self.index_keys is not None:
                self.index_keys = self.index_keys[:, : self.offset]
                self.index_position_ids = self.index_position_ids[..., : self.offset]
            _trim_index_blocks(self, self.offset)
            return n

        qsa_quant_cls.trim = qsa_quant_trim
        qsa_quant_cls._quanteeze_qsa_cache = True

    batch_cls = _language.BatchQSAKVCache
    if not getattr(batch_cls, '_quanteeze_qsa_cache', False):
        orig_batch_trim = batch_cls.trim
        orig_batch_update = batch_cls.update_indexer

        @functools.wraps(orig_batch_trim)
        def batch_trim(self, n):
            trimmed = self.kv_cache.trim(n)
            self.index_offset = max(0, self.index_offset - trimmed)
            _trim_index_blocks(self, self.index_offset)
            return trimmed

        @functools.wraps(orig_batch_update)
        def batch_update_indexer(self, keys, position_ids):
            prev = int(self.index_offset)
            incoming = keys.shape[1]
            step = int(getattr(self, 'step', 256) or 256)
            add = ((incoming + step - 1) // step) * step if incoming > 0 else step
            if self.index_keys is None:
                self.index_keys = mx.zeros((keys.shape[0], add, keys.shape[2]), dtype=keys.dtype)
                self.index_position_ids = mx.zeros((*position_ids.shape[:-1], add), dtype=position_ids.dtype)
                if incoming:
                    self.index_keys[:, :incoming] = keys
                    self.index_position_ids[..., :incoming] = position_ids
                self.index_offset = incoming
                return self.index_keys[:, :incoming], self.index_position_ids[..., :incoming]

            if self.index_position_ids.ndim == 3 and position_ids.ndim == 2:
                position_ids = mx.broadcast_to(position_ids[None], (self.index_position_ids.shape[0], *position_ids.shape))
            elif self.index_position_ids.ndim == 2 and position_ids.ndim == 3:
                self.index_position_ids = mx.contiguous(
                    mx.broadcast_to(self.index_position_ids[None], (position_ids.shape[0], *self.index_position_ids.shape))
                )
            elif self.index_position_ids.ndim != position_ids.ndim:
                raise ValueError(
                    'QSA position IDs must be 2-D text positions or 3-D MRoPE positions, '
                    f'got cached={self.index_position_ids.shape} and current={position_ids.shape}.'
                )

            key_dtype = mx.result_type(self.index_keys, keys)
            if key_dtype != self.index_keys.dtype:
                self.index_keys = self.index_keys.astype(key_dtype)
            if key_dtype != keys.dtype:
                keys = keys.astype(key_dtype)

            pos_dtype = mx.result_type(self.index_position_ids, position_ids)
            if pos_dtype != self.index_position_ids.dtype:
                self.index_position_ids = self.index_position_ids.astype(pos_dtype)
            if pos_dtype != position_ids.dtype:
                position_ids = position_ids.astype(pos_dtype)

            end = prev + incoming
            if end > self.index_keys.shape[1]:
                cur = self.index_keys if prev == self.index_keys.shape[1] else self.index_keys[:, :prev]
                self.index_keys = mx.concatenate([cur, mx.zeros((keys.shape[0], add, keys.shape[2]), dtype=key_dtype)], axis=1)
            if end > self.index_position_ids.shape[-1]:
                cur = (
                    self.index_position_ids
                    if prev == self.index_position_ids.shape[-1]
                    else self.index_position_ids[..., :prev]
                )
                self.index_position_ids = mx.concatenate(
                    [cur, mx.zeros((*position_ids.shape[:-1], add), dtype=pos_dtype)], axis=-1
                )

            if incoming:
                self.index_keys[:, prev:end] = keys
                self.index_position_ids[..., prev:end] = position_ids
            self.index_offset = end
            return self.index_keys[:, :end], self.index_position_ids[..., :end]

        batch_cls.trim = batch_trim
        batch_cls.update_indexer = batch_update_indexer
        batch_cls._quanteeze_qsa_cache = True


def install(model):
    """Install `_BatchInvariantForward`, DFlash2 capture/verify, QSA indexer cache hooks, and `expert_qmv` switches."""
    _install_dflash()
    _install_qsa_cache()
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
