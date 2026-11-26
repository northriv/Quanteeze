# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Flash-Next's full attention's keys and values kept at 8 bits (QUANTEEZE_KV_BITS=8) -- and those of models whose
full attention layers keep a plain KVCache (e.g. Qwen3.8-27B: install_plain, below).

A full attention layer of Qwen3.8 Flash-Next keeps, per token, its 2 KV heads' keys and values of 256 bfloat16 each:
2,048 bytes per layer, 24.6 kB over its 12 such layers -- 1.7 GB at 70k tokens, 6.4 GB at the model's 262,144, beside
the experts on the GPU.  At 8 bits, MLX's affine quantization (mx.quantize: a byte per value, and a bfloat16 scale and
bias per group of 64) takes 1,088 bytes per layer and token, 0.53 of bfloat16.  The indexer's keys (256 bytes per
layer and token) and the linear layers' states stay as they are.

mlx-vlm has the format and the caches of one request served alone (QSAQuantizedKVCache, its KV_BITS), and its
attention reads them without spreading them out (quantized_scaled_dot_product_attention: the scores and the output
MLX's quantized_matmul, the keys and the values its weights).  Four things this module adds:

- the batch's cache.  A server's requests run as batches (continuous batching, a batch of one too), and mlx-vlm
  refuses KV_BITS there for a cache with state of its own as QSA's (NotImplementedError).  Batch here is its
  BatchQSAKVCache -- the indexer's keys and positions as it keeps them -- over its BatchQuantizedKVCache; one
  request's is its QSAQuantizedKVCache itself (Single), given how to become and to be merged into that batch and
  how to be stored empty (the prompt cache takes back only classes of mlx-vlm's own modules).  install(lm) has
  the model's make_cache give Single for its full attention layers, and mlx-vlm's own KV_BITS stays unset;
- the attention where MLX's quantized_matmul is wrong.  On MLX's CUDA backend, the product with weights of more than
  two dimensions -- mlx-vlm's: [batch, KV heads, 1, keys, .] -- is not the product of each head's (a relative error
  near 1 of the scores and of the output; a head at a time it matches the spread-out product to ~1e-5).  attention()
  runs mlx-vlm's function one (row, KV head) at a time with two-dimensional weights, put in mlx_vlm.models.base's
  place.  The first CHECKS calls of a shape are computed from the dequantized keys and values too (mx.dequantize,
  MLX's matmul) and compared: a relative difference over TOLERANCE and that shape is served from the spread-out ones
  from there on, logged once;
- a decode token's sparse attention over the keys it selected alone (QUANTEEZE_KV_BITS_SELECTED, on).  Past the
  indexer's budget (2,048 tokens) QSA attends to the 512 blocks of 4 keys its indexer picks and the few after the
  last whole block, whereas mlx-vlm computes scores of every key under a mask when its Metal bfloat16 indexed kernel
  does not apply.  dispatch(), in the place of mlx-vlm's dispatch_qsa_attention, gathers those selected keys' and
  values' bytes, scales and biases, spreads them out and attends to them with MLX's matmul: one or two tokens of one
  request (a decode, an MTP verify pair), its first CHECKS calls of a shape compared with the masked attention over
  every key, which serves the shape if one differs by more than TOLERANCE;
- on Metal, unpacking cached 8-bit keys and values to `queries.dtype` on prefill while running decode/verify in one
  call through mlx-vlm's quantized attention (QUANTEEZE_KV_BITS_UNPACKED, auto: enabled on Metal).  mlx-vlm's QSA
  prefill kernel (`qsa_sparse_attention`: the selected blocks alone) accepts bfloat16 keys only, and over 8-bit tuples
  it otherwise materializes dense masks and scores across every key (~3 GB per 2,048-token prefill step at 32k
  context).  Unpacked per layer, 32k keys and values take 67 MB (~0.3 ms) and run through `qsa_sparse_attention` on
  QSA prefill or `mx.fast.scaled_dot_product_attention` on plain prefill (`length > ROWS_MOST`), whereas decode and
  2-token MTP verify steps (`length <= ROWS_MOST`) invoke mlx-vlm's `quantized_scaled_dot_product_attention` directly
  in one call without writing/reading dequantized keys every token, verified against the dequantized reference on the
  first CHECKS calls per shape.

A model whose layers' caches are mlx-vlm's plain `KVCache` (and `ArraysCache` for linear layers) -- e.g. Qwen3.8-27B,
whose 16 full attention layers keep 65.5 kB/token in bfloat16 -- uses mlx-vlm's own `BatchQuantizedKVCache` (and APC
snapshots) when `runtime.config.kv_bits` is set.  `install_plain(lm)` sets `runtime.config` (`kv_bits=8`,
`kv_group_size=64`, `kv_quant_scheme='uniform'`, restored on unload) and installs `attention()` without QSA dispatch
(mlx-vlm's `should_quantize_kv_layer` leaves the final layer unquantized: 36.7 kB/token, 0.56 of bfloat16).

Not bit-identical to bfloat16 keys and values (~0.7% norm difference per 64-element group).  Kernels that read
bfloat16 keys directly (quanteeze.cuda_attn, QUANTEEZE_CUDA_ATTN) decline these caches so mlx-vlm's ops serve them.
"""
import sys

import mlx.core as mx

from . import kernel_checks, per_load, per_thread, variables

BITS = 8
GROUP = 64                      # mx.quantize's group: a scale and a bias each (mlx-vlm's default, and QuantizedLinear's)
CHECKS = 3
TOLERANCE = 0.02                # of the output's norm: the spread-out product's rounding is ~1e-5 of it
SELECTED = [True]               # QUANTEEZE_KV_BITS_SELECTED: sparse decode over the selected keys alone
UNPACKED = [False]              # QUANTEEZE_KV_BITS_UNPACKED: unpack on prefill / 1-call quantized decode on Metal
LONGER = 4                      # dispatch() gathers when cached keys exceed LONGER * selected keys
ROWS_MOST = 2                   # query tokens at most that dispatch() gathers / Metal runs quantized in one call
STATS = dict(
    served=0, checked=0, differed=0, spread=0,
    selected=0, selected_checked=0, selected_differed=0,
    unpacked=0, quantized=0, unpacked_checked=0, unpacked_differed=0,
)
CHOICE, CHECKED = kernel_checks.tables('kv_bits')
SAID = []
PLAIN = []                      # mlx-vlm runtime.config (kv_bits, kv_group_size, kv_quant_scheme) saved by install_plain
SERVED = [False]                # whether the model being served keeps its keys and values at 8 bits (install; tag())
_UNKNOWN = []                   # invalid QUANTEEZE_KV_BITS values already warned about
_CLASSES = []                   # (Single, Batch) once constructed


def bits(environ=None):
    """QUANTEEZE_KV_BITS as an integer bit width, or None where keys and values stay bfloat16."""
    value = variables.get('QUANTEEZE_KV_BITS', environ)
    if value is None or value.strip().lower() in ('', 'off', '16'):
        return None
    if value.strip() != str(BITS):                           # logged, not raised, so an invalid value falls back safely
        if value not in _UNKNOWN:
            _UNKNOWN.append(value)
            print(f'quanteeze: QUANTEEZE_KV_BITS={value} is neither {BITS} nor off: the keys and values stay '
                  f'bfloat16', file=sys.stderr, flush=True)
        return None
    return BITS


@per_load.on_let_go
def _let_go():
    SERVED[0] = False
    if PLAIN:
        from mlx_vlm.server.runtime import runtime
        runtime.config.kv_bits, runtime.config.kv_group_size, runtime.config.kv_quant_scheme = PLAIN.pop()


def tag():
    """Precision salt component for APC (`serve._precision`): isolates 8-bit KV checkpoints from bfloat16 ones."""
    return f'keys and values at {BITS} bits a group of {GROUP}' if SERVED[0] else ''


def classes():
    """Return `(Single, Batch)` 8-bit full-attention QSA KV cache classes."""
    if _CLASSES:
        return _CLASSES[0]
    from mlx_vlm.models.cache import BatchQuantizedKVCache
    from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, QSAQuantizedKVCache

    # Single is mlx-vlm's QSAQuantizedKVCache itself so APC's _resolve_checkpoint_class accepts its module/qualname.
    Single = QSAQuantizedKVCache

    def to_batch(self, left_padding):
        padding = mx.array(left_padding)
        if self.keys is None and self.index_keys is None:
            return Batch(left_padding)
        if padding.size != 1 or int(padding.item()):
            raise ValueError(f'a warm 8-bit QSA cache seeds one row without padding, got {padding.tolist()}')
        return Batch.merge([self])

    def merge(cls, caches, prefix_lens=None):
        return Batch.merge(caches, prefix_lens=prefix_lens)

    def prefix_cache_merge(self, rows, prefix_lens):
        return Batch.merge(rows, prefix_lens=prefix_lens)

    def prefix_cache_snapshot(self):
        return {'state': self.state, 'meta_state': self.meta_state}

    Single.to_batch = to_batch
    Single.merge = classmethod(merge)
    Single.prefix_cache_merge = prefix_cache_merge
    Single.prefix_cache_snapshot = prefix_cache_snapshot

    class Rows(BatchQuantizedKVCache):
        """`BatchQuantizedKVCache` exposing `.nbytes` for admission and QSA byte accounting."""

        def __init__(self, left_padding):
            super().__init__(left_padding, group_size=GROUP, bits=BITS)

        @property
        def nbytes(self):
            return 0 if self.keys is None else sum(part.nbytes for part in self.keys + self.values)

    class Batch(BatchQSAKVCache):
        """`BatchQSAKVCache` backed by `BatchQuantizedKVCache` (`Rows`) storing `(weight, scales, biases)` tuples."""

        def __init__(self, left_padding):
            super().__init__(left_padding)
            self.kv_cache = Rows(left_padding)

        @property
        def bits(self):
            return self.kv_cache.bits

        @property
        def group_size(self):
            return self.kv_cache.group_size

        @property
        def state(self):
            return BatchQSAKVCache.state.fget(self)

        @state.setter
        def state(self, value):
            if len(value) == 3:
                value = (*value, None, None)
            kv_state, self.index_keys, self.index_position_ids, self.index_block_keys, self.index_block_ratio = value
            padding = [0] if kv_state is None or len(kv_state) < 4 or kv_state[3] is None else kv_state[3]
            self.kv_cache = Rows([0])
            self.kv_cache.left_padding = mx.array(padding)
            self.kv_cache.offset = -self.kv_cache.left_padding
            if kv_state is not None and kv_state[0] is not None:
                self.kv_cache.state = kv_state
            elif kv_state is not None and kv_state[2] is not None:
                self.kv_cache.offset = kv_state[2]
            self.index_offset = 0 if self.index_keys is None else self.index_keys.shape[1]

        def extend(self, other):
            if type(other) is not Batch:
                raise TypeError(f'an 8-bit QSA batch is extended with another, not {type(other).__name__}')
            super().extend(other)

        def extract(self, idx):
            got = super().extract(idx)
            out = Single(group_size=GROUP, bits=BITS)
            for name in ('keys', 'values', 'offset', 'index_keys', 'index_position_ids', 'index_block_keys',
                         'index_block_ratio'):
                setattr(out, name, getattr(got, name))
            return out

        @classmethod
        def merge(cls, caches, prefix_lens=None):
            from types import SimpleNamespace
            caches = list(caches)
            out = cls([0] * len(caches))
            if not caches:
                return out
            if prefix_lens is not None and len(prefix_lens) != len(caches):
                raise ValueError('prefix_lens must have one entry per QSA cache')
            if not all(isinstance(c, QSAQuantizedKVCache) and (c.group_size, c.bits) == (GROUP, BITS) for c in caches):
                raise TypeError('an 8-bit QSA batch is merged from 8-bit QSA caches, not '
                                + ', '.join(type(c).__name__ for c in caches))
            target = max(c.offset for c in caches)
            held = [c for c in caches if c.keys is not None and c.offset]
            if held:
                def rows(name):
                    parts = []
                    for n in range(3):
                        sample = getattr(held[0], name)[n]
                        each = []
                        for c in caches:
                            got = (getattr(c, name)[n][..., :c.offset, :] if c.keys is not None and c.offset
                                   else mx.zeros((1, sample.shape[1], 0, sample.shape[3]), dtype=sample.dtype))
                            left = target - got.shape[2]
                            each.append(mx.pad(got, [(0, 0), (0, 0), (left, 0), (0, 0)]) if left else got)
                        parts.append(each[0] if len(each) == 1 else mx.concatenate(each, axis=0))
                    return tuple(parts)
                kv = out.kv_cache
                kv.keys, kv.values = rows('keys'), rows('values')
                kv.offset = mx.array([c.offset for c in caches])
                kv.left_padding = mx.array([target - c.offset for c in caches])
                kv._idx = target
            sample_keys, sample_positions = cls._samples(*caches)
            if sample_keys is None:
                return out
            index = [cls._pad_index(SimpleNamespace(index_keys=c.index_keys, index_position_ids=c.index_position_ids,
                                                    index_offset=c.offset, offset=mx.array([c.offset])),
                                    target, sample_keys, sample_positions) for c in caches]
            out.index_keys = mx.concatenate([row[0] for row in index], axis=0)
            out.index_position_ids = mx.concatenate([row[1] for row in index],
                                                    axis=1 if sample_positions.ndim == 3 else 0)
            out.index_offset = target
            summaries, ratios = [c.index_block_keys for c in caches], [c.index_block_ratio for c in caches]
            if (all(s is not None for s in summaries) and len(set(ratios)) == 1
                    and len({c.offset for c in caches}) == 1 and len({s.shape[2] for s in summaries}) == 1):
                out.index_block_keys = mx.concatenate(summaries, axis=0)
                out.index_block_ratio = ratios[0]
            return out

    _CLASSES.append((Single, Batch))
    return _CLASSES[0]


def install(lm, environ=None):
    """Configure `lm` (Qwen4-Exp language model) to use 8-bit full-attention QSA KV caches when
    `QUANTEEZE_KV_BITS=8`, and patch quantized attention & sparse QSA dispatch."""
    if bits(environ) is None:
        return False
    import os
    if os.environ.get('KV_BITS'):
        print(f'quanteeze: QUANTEEZE_KV_BITS={BITS} not applied: mlx-vlm\'s KV_BITS is set beside it, with which '
              f'it refuses a QSA cache in a batch (unset KV_BITS)', file=sys.stderr, flush=True)
        return False
    full = [layer for layer in lm.layers if not layer.is_linear]
    widths = {layer.self_attn.head_dim for layer in full}
    if not full or any(width % GROUP for width in widths):
        print(f'quanteeze: QUANTEEZE_KV_BITS={BITS} not applied: a head of {sorted(widths)} values is not groups '
              f'of {GROUP}', file=sys.stderr, flush=True)
        return False
    single, _ = classes()
    made = lm.make_cache
    if getattr(made, '_quanteeze_kv_bits', False):
        SELECTED[0] = not variables.off('QUANTEEZE_KV_BITS_SELECTED', environ)
        UNPACKED[0] = unpacked(environ)
        SERVED[0] = True
        return True

    def make_cache():
        caches = made()
        for i, layer in enumerate(lm.layers):
            if not layer.is_linear and type(caches[i]).__name__ == 'QSAKVCache':
                caches[i] = single(group_size=GROUP, bits=BITS)
        return caches
    make_cache._quanteeze_kv_bits = True
    lm.make_cache = make_cache
    SELECTED[0] = not variables.off('QUANTEEZE_KV_BITS_SELECTED', environ)
    UNPACKED[0] = unpacked(environ)
    SERVED[0] = True
    install_attention()
    return True


def unpacked(environ=None):
    """Return whether `QUANTEEZE_KV_BITS_UNPACKED` unpacks 8-bit KV caches for MLX/mlx-vlm kernels (`auto` -> Metal)."""
    value = (variables.get('QUANTEEZE_KV_BITS_UNPACKED', environ) or 'auto').strip().lower()
    if value in ('auto', ''):
        return bool(mx.metal.is_available())
    return value not in ('off', '0', 'false', 'no')


def install_plain(lm, environ=None):
    """Configure a model whose full-attention layers use plain `mlx_vlm.models.cache.KVCache` (with optional
    `ArraysCache` linear layers, e.g. Qwen3.8-27B) to use mlx-vlm's own 8-bit batch cache (`runtime.config.kv_bits`)
    when `QUANTEEZE_KV_BITS=8`, and patch `quantized_scaled_dot_product_attention`."""
    if bits(environ) is None:
        return False
    import os
    if os.environ.get('KV_BITS'):
        print(f'quanteeze: QUANTEEZE_KV_BITS={BITS} not applied: mlx-vlm\'s KV_BITS is set beside it and serves '
              f'(unset one of them)', file=sys.stderr, flush=True)
        return False
    from mlx_vlm.models import cache
    if not hasattr(lm, 'make_cache'):
        print(f'quanteeze: QUANTEEZE_KV_BITS={BITS} not applied: the model makes no caches of its own',
              file=sys.stderr, flush=True)
        return False
    made = lm.make_cache()
    kinds = {type(c) for c in made}
    others = sorted(t.__name__ for t in kinds - {cache.KVCache, cache.ArraysCache})
    if cache.KVCache not in kinds or others:
        print(f'quanteeze: QUANTEEZE_KV_BITS={BITS} not applied: its layers\' caches are '
              f'{", ".join(others) or "none of them plain KVCache"}, not plain ones', file=sys.stderr, flush=True)
        return False
    layers = getattr(getattr(lm, 'model', None), 'layers', None) or getattr(lm, 'layers', [])
    widths = {getattr(getattr(layers[i], 'self_attn', None), 'head_dim', None) for i, c in enumerate(made)
              if type(c) is cache.KVCache and i < len(layers)}
    if not widths or any(not isinstance(w, int) or w % GROUP for w in widths):
        print(f'quanteeze: QUANTEEZE_KV_BITS={BITS} not applied: a head of {sorted(map(str, widths))} values is not '
              f'groups of {GROUP}', file=sys.stderr, flush=True)
        return False
    from mlx_vlm.server.runtime import runtime
    if not PLAIN:
        PLAIN.append((runtime.config.kv_bits, runtime.config.kv_group_size, runtime.config.kv_quant_scheme))
    runtime.config.kv_bits, runtime.config.kv_group_size, runtime.config.kv_quant_scheme = BITS, GROUP, 'uniform'
    UNPACKED[0] = unpacked(environ)
    SERVED[0] = True
    install_attention(qsa=False)
    return True


def install_attention(qsa=True):
    """Patch `mlx_vlm.models.base.quantized_scaled_dot_product_attention`, and when `qsa` is true also patch
    `mlx_vlm.models.qwen4_exp.language.dispatch_qsa_attention`, once each."""
    from mlx_vlm.models import base
    if not getattr(base.quantized_scaled_dot_product_attention, '_quanteeze_kv_bits', False):
        attention._theirs = base.quantized_scaled_dot_product_attention
        base.quantized_scaled_dot_product_attention = attention
    if not qsa:
        return
    from mlx_vlm.models.qwen4_exp import language
    if getattr(language.dispatch_qsa_attention, '_quanteeze_kv_bits', False):
        return
    dispatch._theirs = language.dispatch_qsa_attention
    language.dispatch_qsa_attention = dispatch


def _by_head(queries, q_keys, q_values, mask, group_size, bits):
    """Quantized attention of scaled queries `[B, KV heads, repeats, L, D]`, invoking `mx.quantized_matmul`
    one `(row, KV head)` at a time so weights are 2D."""
    rows, heads = queries.shape[0], queries.shape[1]

    def each(x, parts, transpose):
        return mx.stack([mx.stack([mx.quantized_matmul(x[b, h], *(p[b, h] for p in parts), transpose=transpose,
                                                       group_size=group_size, bits=bits) for h in range(heads)])
                         for b in range(rows)])
    scores = _masked(each(queries, q_keys, True), mask)
    return each(mx.softmax(scores, axis=-1, precise=True), q_values, False)


def _spread(queries, q_keys, q_values, mask, group_size, bits):
    """Reference attention computed via `mx.dequantize` and standard matmul."""
    keys = mx.expand_dims(mx.dequantize(*q_keys, group_size=group_size, bits=bits), 2)
    values = mx.expand_dims(mx.dequantize(*q_values, group_size=group_size, bits=bits), 2)
    scores = _masked(queries @ keys.swapaxes(-1, -2), mask)
    return mx.softmax(scores, axis=-1, precise=True) @ values


def _unpack(parts, dtype, group_size, bits):
    return mx.dequantize(*parts, group_size=group_size, bits=bits).astype(dtype)


def _unpacked(queries, q_keys, q_values, scale, mask, group_size, bits):
    """`queries` `[B, heads, L, D]` over dequantized keys and values via `mx.fast.scaled_dot_product_attention`."""
    return mx.fast.scaled_dot_product_attention(
        queries,
        _unpack(q_keys, queries.dtype, group_size, bits),
        _unpack(q_values, queries.dtype, group_size, bits),
        scale=scale,
        mask=mask,
    )


def _mlx_vlms():
    """Return mlx-vlm's original `quantized_scaled_dot_product_attention` (never `attention()` itself)."""
    theirs = getattr(attention, '_theirs', None)
    if theirs is None:
        from mlx_vlm.models import base
        theirs = base.quantized_scaled_dot_product_attention
        if getattr(theirs, '_quanteeze_kv_bits', False):
            raise RuntimeError("mlx-vlm's quantized attention is not known: attention() stands in its place")
    return theirs


def _masked(scores, mask):
    """Apply `mask` to attention scores `[B, KV heads, repeats, L, keys]`."""
    if mask is None:
        return scores
    from mlx_vlm.models.base import align_attention_mask_to_scores
    if isinstance(mask, str):
        rows, keys = scores.shape[-2:]
        mask = mx.arange(keys - rows, keys)[:, None] >= mx.arange(keys)[None]
    mask = align_attention_mask_to_scores(mask, scores)
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask


def attention(queries, q_keys, q_values, scale, mask, group_size=GROUP, bits=BITS):
    """Per-(row, KV head), 1-call Metal quantized (`length <= ROWS_MOST`), or unpacked (`length > ROWS_MOST`)
    `quantized_scaled_dot_product_attention`, checking the first `CHECKS` calls per shape against `_spread` and
    falling back to `_spread` if relative difference exceeds `TOLERANCE`."""
    rows, heads, length, width = queries.shape
    kv_heads = q_keys[0].shape[-3]
    scaled = (queries * scale).reshape(rows, kv_heads, heads // kv_heads, length, width)
    key = (heads, kv_heads, width, int(group_size), int(bits), min(length, 3), str(queries.dtype),
           'none' if mask is None else mask if isinstance(mask, str) else str(mask.dtype))
    whole = UNPACKED[0] and length <= ROWS_MOST
    if whole:
        key = ('quantized',) + key

        def mine():
            # Pass pre-scaled queries with scale=1.0 because mlx-vlm mutates `queries *= scale` in place.
            return _mlx_vlms()(queries * scale, q_keys, q_values, 1.0, mask, group_size=group_size, bits=bits)
    elif UNPACKED[0]:
        key = ('unpacked',) + key

        def mine():
            return _unpacked(queries, q_keys, q_values, scale, mask, group_size, bits)
    else:
        def mine():
            return _by_head(scaled, q_keys, q_values, mask, group_size, bits).reshape(rows, heads, length, width)

    def spread():
        return _spread(scaled, q_keys, q_values, mask, group_size, bits).reshape(rows, heads, length, width)
    choice, seen = kernel_checks.read(CHOICE, CHECKED, key)
    if choice is not None and seen >= CHECKS:
        STATS['served' if choice else 'spread'] += 1
        STATS['quantized' if whole else 'unpacked'] += bool(choice and UNPACKED[0])
        return (mine if choice else spread)()
    STATS['checked'] += 1
    want = spread()
    try:
        got = mine()
        a, b = got.astype(mx.float32), want.astype(mx.float32)
        off = (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-30)).item()
        why = None if off <= TOLERANCE else f'a relative difference of {off:.3g}'
    except Exception as error:
        got, why = want, f'{type(error).__name__}: {error}'
    if why and not SAID:
        SAID.append(True)
        how = ("mlx-vlm's in one call" if whole else "over the keys spread out (MLX's fast attention)" if UNPACKED[0]
               else 'a head at a time')
        print(f'quanteeze: the 8-bit keys\' attention {how} is not the spread-out product\'s ({why}; '
              f'{key}); that shape is served from the keys and values spread out by MLX\'s matmul', file=sys.stderr,
              flush=True)
    STATS['differed'] += bool(why)
    per_thread.checked(CHOICE, CHECKED, key, not why, CHECKS)
    return want if why else got


attention._quanteeze_kv_bits = True


def _selected(queries, q_keys, q_values, blocks, ends, scale, ratio, group_size, bits):
    """QSA sparse attention of `queries` `[1, heads, L, D]` over only the indexer-selected blocks and tail keys."""
    _, heads, length, width = queries.shape
    kv_heads, size = q_keys[0].shape[1], q_keys[0].shape[2]
    top = blocks.shape[-1]
    most = top * ratio + ratio - 1
    blocks = mx.sort(blocks[0].astype(mx.int32), axis=-1)
    ends = ends[0].astype(mx.int32)
    whole = ends // ratio
    within = mx.arange(ratio, dtype=mx.int32)
    tail = (whole * ratio)[:, None] + within[None, :ratio - 1]
    chosen = mx.concatenate([(blocks[..., None] * ratio + within).reshape(length, top * ratio), tail], axis=1)
    chosen_ok = mx.concatenate([mx.repeat(blocks >= 0, ratio, axis=1), tail < ends[:, None]], axis=1)
    first = mx.broadcast_to(mx.arange(most, dtype=mx.int32)[None], (length, most))
    sparse = (whole > top)[:, None]
    at = mx.clip(mx.where(sparse, chosen, first), 0, size - 1)
    ok = mx.where(sparse, chosen_ok, first < ends[:, None])

    def spread(parts):
        return mx.dequantize(*(mx.take(part, at, axis=2)[0] for part in parts), group_size=group_size, bits=bits)
    keys, values = spread(q_keys), spread(q_values)                             # [KV heads, L, most, D]
    q = (queries * scale).reshape(kv_heads, heads // kv_heads, length, width).transpose(0, 2, 1, 3)
    scores = q @ keys.swapaxes(-1, -2)                                          # [KV heads, L, repeats, most]
    scores = mx.where(ok[None, :, None, :], scores, mx.finfo(scores.dtype).min)
    out = mx.softmax(scores, axis=-1, precise=True) @ values
    return out.transpose(0, 2, 1, 3).reshape(1, heads, length, width)


def dispatch(queries, keys, values, block_indices, query_ends, *, cache, scale, block_size, causal, mask,
             mask_factory, allow_sparse_decode=False):
    """Replacement for `dispatch_qsa_attention` gathering selected 8-bit keys on 1..`ROWS_MOST` decode/verify tokens
    once cached length exceeds `LONGER` times selected keys, and unpacking prompt steps on Metal (`UNPACKED`)."""
    def theirs():
        return dispatch._theirs(queries, keys, values, block_indices, query_ends, cache=cache, scale=scale,
                                block_size=block_size, causal=causal, mask=mask, mask_factory=mask_factory,
                                allow_sparse_decode=allow_sparse_decode)
    if (UNPACKED[0] and isinstance(keys, tuple) and getattr(cache, 'bits', None) is not None
            and queries.ndim == 4 and queries.shape[2] > ROWS_MOST):
        return _prefill(queries, keys, values, block_indices, query_ends, cache=cache, scale=scale,
                        block_size=block_size, causal=causal, mask=mask, mask_factory=mask_factory,
                        allow_sparse_decode=allow_sparse_decode)
    if not (SELECTED[0] and isinstance(keys, tuple) and getattr(cache, 'bits', None) is not None and causal
            and mask is None and isinstance(block_indices, mx.array) and isinstance(query_ends, mx.array)
            and queries.ndim == 4 and queries.shape[0] == 1 and 1 <= queries.shape[2] <= ROWS_MOST
            and block_indices.shape[:2] == (1, queries.shape[2]) and query_ends.shape == (1, queries.shape[2])
            and block_size > 1
            and keys[0].shape[2] > LONGER * (block_indices.shape[-1] * block_size + block_size)):
        return theirs()
    heads, length, width = queries.shape[1:]
    key = ('selected', heads, keys[0].shape[1], width, int(cache.group_size), int(cache.bits), length,
           int(block_indices.shape[-1]), int(block_size), str(queries.dtype))
    choice, seen = kernel_checks.read(CHOICE, CHECKED, key)
    if choice is not None and seen >= CHECKS:
        if not choice:
            return theirs()
        STATS['selected'] += 1
        return _selected(queries, keys, values, block_indices, query_ends, scale, block_size, cache.group_size,
                         cache.bits)
    STATS['selected_checked'] += 1
    want = theirs()
    try:
        got = _selected(queries, keys, values, block_indices, query_ends, scale, block_size, cache.group_size,
                        cache.bits)
        a, b = got.astype(mx.float32), want.astype(mx.float32)
        off = (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-30)).item()
        why = None if off <= TOLERANCE else f'a relative difference of {off:.3g}'
    except Exception as error:
        got, why = want, f'{type(error).__name__}: {error}'
    if why and len(SAID) < 2:
        SAID.append(True)
        print(f'quanteeze: the 8-bit keys\' attention over the selected keys alone is not the masked one\'s ({why}; '
              f'{key}); that shape is served over every key under the mask', file=sys.stderr, flush=True)
    STATS['selected_differed'] += bool(why)
    per_thread.checked(CHOICE, CHECKED, key, not why, CHECKS)
    return want if why else got


dispatch._quanteeze_kv_bits = True


def _prefill(queries, keys, values, block_indices, query_ends, *, cache, scale, block_size, causal, mask, mask_factory,
             allow_sparse_decode):
    """Prompt steps on Metal (`UNPACKED`): unpack 8-bit keys and values to `queries.dtype` and pass `cache=None` to
    `mlx-vlm`'s `dispatch_qsa_attention` so its `qsa_sparse_attention` kernel runs over selected blocks alone,
    checking the first `CHECKS` calls per shape against `_spread`."""
    group_size, bits = int(cache.group_size), int(cache.bits)

    def mine():
        return dispatch._theirs(
            queries,
            _unpack(keys, queries.dtype, group_size, bits),
            _unpack(values, queries.dtype, group_size, bits),
            block_indices, query_ends,
            cache=None, scale=scale, block_size=block_size, causal=causal, mask=mask,
            mask_factory=mask_factory, allow_sparse_decode=allow_sparse_decode,
        )
    rows, heads, length, width = queries.shape
    kv_heads = keys[0].shape[1]
    key = ('unpacked prefill', heads, kv_heads, width, group_size, bits, int(block_indices.shape[-1]),
           int(block_size), bool(causal), mask is None, str(queries.dtype))
    choice, seen = kernel_checks.read(CHOICE, CHECKED, key)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS['unpacked'] += 1
            return mine()
        return dispatch._theirs(queries, keys, values, block_indices, query_ends, cache=cache, scale=scale,
                                block_size=block_size, causal=causal, mask=mask, mask_factory=mask_factory,
                                allow_sparse_decode=allow_sparse_decode)
    STATS['unpacked_checked'] += 1
    scaled = (queries * scale).reshape(rows, kv_heads, heads // kv_heads, length, width)
    want = _spread(scaled, keys, values, mask if mask is not None or not causal else mask_factory(), group_size,
                   bits).reshape(queries.shape)
    try:
        got = mine()
        a, b = got.astype(mx.float32), want.astype(mx.float32)
        off = (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-30)).item()
        why = None if off <= TOLERANCE else f'a relative difference of {off:.3g}'
    except Exception as error:
        got, why = want, f'{type(error).__name__}: {error}'
    if why and len(SAID) < 3:
        SAID.append(True)
        print(f'quanteeze: the 8-bit keys\' prompt attention over the keys spread out is not the masked one\'s '
              f'({why}; {key}); that shape is served as mlx-vlm serves 8-bit keys', file=sys.stderr, flush=True)
    STATS['unpacked_differed'] += bool(why)
    per_thread.checked(CHOICE, CHECKED, key, not why, CHECKS)
    return want if why else got


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
