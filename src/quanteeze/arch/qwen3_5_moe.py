# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Inference hooks for mlx-vlm's Qwen3.5 and Qwen3.6 MoE blocks (`model_type: qwen3_5_moe`)."""
import mlx.core as mx
from mlx_vlm.models.qwen3_5_moe.language import (Qwen3_5MoeDecoderLayer, Qwen3_5MoeModel,
                                                  Qwen3_5MoeSparseMoeBlock)

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


def _install_block(block):
    cls = type(block)
    if getattr(cls, '_quanteeze_moe_fast', False):
        return block
    orig_call = cls.__call__

    class _FastMoeBlock(cls):
        _quanteeze_moe_fast = True

        def __call__(self, x):
            from .. import cuda_router, metal_router
            if self.norm_topk_prob and x.ndim == 3 and (
                x.shape[1] == 1 or (x.shape[0] == 1 and 1 < x.shape[1] <= metal_router.ROWS_MOST)
            ):
                logits = self.gate(x)
                inds, scores = metal_router.routed(logits, self.top_k, lambda: cuda_router.routed(logits, self.top_k))
                y = self.switch_mlp(x, inds)
                shared_y = self._shared_expert_scale(x) * self.shared_expert(x)
                return metal_router.weighted(
                    y,
                    scores,
                    shared_y,
                    lambda: (y * scores[..., None]).sum(axis=-2) + shared_y,
                )
            return orig_call(self, x)

    _FastMoeBlock.__name__ = cls.__name__
    block.__class__ = _FastMoeBlock
    return block


def _batch_prefill(model_obj, orig_call, inputs, inputs_embeds=None, mask=None, cache=None,
                   position_ids=None, capture_layer_ids=None, hidden_sink=None):
    from mlx_vlm.models.qwen3_5 import language as _qlang

    if hidden_sink is not None or cache is None:
        return orig_call(model_obj, inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache,
                         position_ids=position_ids, capture_layer_ids=capture_layer_ids,
                         hidden_sink=hidden_sink)

    h = model_obj.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
    if h.ndim < 3 or h.shape[0] <= 1 or h.shape[1] <= 1:
        return orig_call(model_obj, inputs, inputs_embeds=h, mask=mask, cache=cache,
                         position_ids=position_ids, capture_layer_ids=capture_layer_ids,
                         hidden_sink=hidden_sink)

    fa_cache = cache[model_obj.fa_idx]
    if not (fa_cache is not None
            and hasattr(fa_cache, 'extract')
            and hasattr(fa_cache.__class__, 'merge')
            and isinstance(getattr(fa_cache, 'offset', None), mx.array)
            and fa_cache.offset.ndim > 0):
        return orig_call(model_obj, inputs, inputs_embeds=h, mask=mask, cache=cache,
                         position_ids=position_ids, capture_layer_ids=capture_layer_ids,
                         hidden_sink=hidden_sink)

    if not any(getattr(getattr(getattr(layer, 'mlp', None), 'switch_mlp', None), 'demand', None) is not None
               for layer in model_obj.layers):
        return orig_call(model_obj, inputs, inputs_embeds=h, mask=mask, cache=cache,
                         position_ids=position_ids, capture_layer_ids=capture_layer_ids,
                         hidden_sink=hidden_sink)

    query_left_padding = mx.minimum(mx.maximum(-fa_cache.offset, 0), h.shape[1])
    cache_left_padding = getattr(fa_cache, 'left_padding', None)
    has_left_padding = (
        isinstance(cache_left_padding, mx.array)
        and cache_left_padding.ndim > 0
        and int(cache_left_padding.max().item()) > 0
    )
    if not (has_left_padding or int(query_left_padding.max().item()) > 0):
        return orig_call(model_obj, inputs, inputs_embeds=h, mask=mask, cache=cache,
                         position_ids=position_ids, capture_layer_ids=capture_layer_ids,
                         hidden_sink=hidden_sink)

    pads = [min(max(int(p), 0), h.shape[1]) for p in query_left_padding.tolist()]
    active_rows = [row for row, pad in enumerate(pads) if pad < h.shape[1]]
    if len(active_rows) < 2:
        return orig_call(model_obj, inputs, inputs_embeds=h, mask=mask, cache=cache,
                         position_ids=position_ids, capture_layer_ids=capture_layer_ids,
                         hidden_sink=hidden_sink)

    batch_offsets = []
    for cache_entry in cache:
        offsets = getattr(cache_entry, 'offset', None)
        if isinstance(offsets, mx.array) and offsets.ndim > 0 and offsets.size >= h.shape[0]:
            batch_offsets.append(offsets[:h.shape[0]])
        else:
            batch_offsets.append(None)

    row_caches = [[] for _ in cache]
    row_outputs = [None] * h.shape[0]
    active_states = []
    for row, pad in enumerate(pads):
        current_cache = [
            None if cache_entry is None else _qlang._extract_row_cache(cache_entry, row)
            for cache_entry in cache
        ]
        for i, ce in enumerate(current_cache):
            row_caches[i].append(ce)
        if pad == h.shape[1]:
            row_outputs[row] = mx.zeros_like(h[row:row + 1])
            continue
        row_h = h[row:row + 1, pad:]
        row_pos_ids = None
        if position_ids is not None:
            row_pos_ids = (position_ids[row:row + 1, pad:] if position_ids.ndim == 2
                           else position_ids[:, row:row + 1, pad:])
        fa_mask = _qlang._create_qwen3_5_attention_mask(row_h, current_cache[model_obj.fa_idx])
        ssm_mask = _qlang._create_qwen3_5_ssm_mask(row_h, current_cache[model_obj.ssm_idx])
        decode_lp = (getattr(current_cache[model_obj.fa_idx], '_qwen3_5_decode_left_padding', None)
                     if isinstance(fa_mask, str) and fa_mask == 'left_padded_decode' else None)
        _qlang._set_qwen3_5_decode_left_padding(current_cache, model_obj.layers, decode_lp)
        pos_emb = None
        if row_pos_ids is not None:
            for lyr in model_obj.layers:
                if not lyr.is_linear:
                    if not lyr.self_attn.rotary_emb.fused_apply:
                        pos_emb = lyr.self_attn.rotary_emb(row_h, row_pos_ids)
                    break
        active_states.append({
            'row': row, 'pad': pad, 'len': row_h.shape[1], 'h': row_h,
            'cache': current_cache, 'pos_ids': row_pos_ids, 'pos_emb': pos_emb,
            'fa_mask': fa_mask, 'ssm_mask': ssm_mask,
        })

    for i, layer in enumerate(model_obj.layers):
        mlp = layer.mlp
        has_switch = hasattr(mlp, 'switch_mlp')
        per_row = []
        for st in active_states:
            layer_mask = st['ssm_mask'] if layer.is_linear else st['fa_mask']
            normed_r, resid_r = _before_moe(
                layer, st['h'], mask=layer_mask, cache=st['cache'][i],
                position_ids=st['pos_ids'], position_embeddings=st['pos_emb'],
            )
            if not has_switch:
                st['h'] = resid_r + mlp(normed_r)
                continue
            gates = mx.softmax(mlp.gate(normed_r), axis=-1, precise=True)
            k = mlp.top_k
            inds_r = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
            scores_r = mx.take_along_axis(gates, inds_r, axis=-1)
            if mlp.norm_topk_prob:
                scores_r = scores_r / scores_r.sum(axis=-1, keepdims=True)
            shared_r = mlp._shared_expert_scale(normed_r) * mlp.shared_expert(normed_r)
            per_row.append((st, resid_r, normed_r, inds_r, scores_r, shared_r))
        if has_switch:
            cat_normed = mx.concatenate([item[2] for item in per_row], axis=1)
            cat_inds = mx.concatenate([item[3] for item in per_row], axis=1)
            cat_y = mlp.switch_mlp(cat_normed, cat_inds)
            at = 0
            for st, resid_r, _, _, scores_r, shared_r in per_row:
                n_tok = st['len']
                y_r = cat_y[:, at:at + n_tok]
                at += n_tok
                st['h'] = resid_r + (y_r * scores_r[..., None]).sum(axis=-2) + shared_r
        mx.eval(*[st['h'] for st in active_states])

    for st in active_states:
        row_out = model_obj.norm(st['h'])
        if st['pad'] > 0:
            row_out = _qlang._pad_row_time(row_out, st['pad'], h.shape[1])
        row_outputs[st['row']] = row_out

    for i, entries in enumerate(row_caches):
        if cache[i] is None:
            continue
        if hasattr(cache[i].__class__, 'merge'):
            cache[i] = _qlang._restore_batch_padding_metadata(
                cache[i].__class__.merge(entries),
                batch_offsets[i],
                h.shape[1],
            )
    return mx.concatenate(row_outputs, axis=0)


def _install_model(qwen_model):
    cls = type(qwen_model)
    if getattr(cls, '_quanteeze_batch_prefill', False):
        return qwen_model
    orig_call = cls.__call__

    class _BatchPrefillModel(cls):
        _quanteeze_batch_prefill = True

        def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None,
                     position_ids=None, capture_layer_ids=None, hidden_sink=None):
            return _batch_prefill(
                self, orig_call, inputs, inputs_embeds=inputs_embeds, mask=mask,
                cache=cache, position_ids=position_ids,
                capture_layer_ids=capture_layer_ids, hidden_sink=hidden_sink,
            )

    _BatchPrefillModel.__name__ = cls.__name__
    qwen_model.__class__ = _BatchPrefillModel
    return qwen_model


def install(model):
    """Install `expert_qmv`, router, and layer-major batch prefill hooks on `model`; returns `{layer_idx: block}`."""
    blocks = {}
    for name, module in model.named_modules():
        parts = name.split('.')
        if isinstance(module, Qwen3_5MoeModel) and 'mtp' not in parts:
            _install_model(module)
        if isinstance(module, Qwen3_5MoeSparseMoeBlock) and 'layers' in parts and 'mtp' not in parts:
            layer = int(parts[parts.index('layers') + 1])
            object.__setattr__(module, '_layer_idx', layer)
            expert_qmv.install_switch(module.switch_mlp)
            _install_block(module)
            blocks[layer] = module
    return blocks
