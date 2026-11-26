# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A prompt read back from the prefix cache (APC) for a request alone, served on the caches a fresh one has.

mlx-vlm gives a request alone -- one row, no right padding, no KV bits, no draft model -- the model's own plain
caches (PromptProcessingBatch: cache.make_prompt_cache), and a request whose prompt it read back from the prefix
cache the batch caches its APC built (BatchKVCache, ArraysCache with a left padding), one row of them.  Qwen3.5's
module (Qwen3.8-27B; mlx_vlm.models.qwen3_5) takes a single-row batch cache apart at every decode step: the row
extracted to a KVCache (the keys and values up to the step copied out), the step's token added (copied again, into
room for 256 more), and the batch cache merged back (copied a third time) -- some 2 GB of copies a token at 32k
tokens.  Qwen3.8-27B decoded the 32k table at 45.5 tokens a second and at 24.3 after its prompt was read back,
in memory or from the disk; at 8 bits (BatchQuantizedKVCache, which that path leaves alone) at 38 both.

install() has such a request's warm caches made into the model's plain ones, holding the same arrays (plain()): a
BatchKVCache's keys and values and their length as a KVCache's, an ArraysCache's states in one with no padding.
The model then sees what a fresh request gives it, the same values bit for bit.  A read-back row comes with a
right padding of its suffix's (right_pad_per_row [0] for one row: none), which counts as none.  A row with padding,
a cache of another kind (a quantized one, Flash-Next's QSA caches, a rotating one), a request with KV bits or a draft
model: as mlx-vlm made it.  QUANTEEZE_APC_READ_BACK_PLAIN=off: as mlx-vlm made it throughout.
"""
import functools
import inspect

STATS = {'plain': 0, 'kept': 0}


def plain(model, warm):
    """The model's plain caches (cache.make_prompt_cache) holding `warm`'s arrays -- a batch of one row and no padding
    -- or None where a cache is not one this can make so: a KVCache for a BatchKVCache, an ArraysCache for an
    ArraysCache."""
    from mlx_vlm.models import cache as caches
    fresh = caches.make_prompt_cache(model)
    if len(fresh) != len(warm):
        return None
    for new, old in zip(fresh, warm):
        if type(new) is caches.KVCache and type(old) is caches.BatchKVCache:
            if old._right_padding is not None or int(old.left_padding.size) != 1 or int(old.left_padding.max().item()):
                return None
            if old.keys is not None:
                if old.keys.shape[0] != 1:
                    return None
                new.keys, new.values, new.offset = old.keys, old.values, int(old._idx)
        elif type(new) is caches.ArraysCache and type(old) is caches.ArraysCache:
            pads, lengths = old.left_padding, old.lengths
            if (len(new.cache) != len(old.cache) or old.is_speculating or lengths is not None
                    or (pads is not None and (int(pads.size) != 1 or int(pads.max().item())))
                    or any(s is not None and s.shape[0] != 1 for s in old.cache)):
                return None
            new.cache = list(old.cache)
        else:
            return None
    return fresh


def install(batch_cls=None):
    """PromptProcessingBatch (or `batch_cls`, the same arguments) given plain()'s caches for a warm request alone,
    where mlx-vlm gives a fresh one the model's plain caches.  Installed once; whether it was now."""
    if batch_cls is None:
        from mlx_vlm.generate.ar import PromptProcessingBatch as batch_cls
    if getattr(batch_cls.__init__, '_quanteeze_plain', False):
        return False
    original = batch_cls.__init__
    signature = inspect.signature(original)

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        given = bound.arguments
        warm, model = given.get('warm_cache'), given.get('model')
        if (warm is not None and len(given.get('input_ids') or ()) == 1
                and not any(given.get('right_pad_per_row') or ()) and given.get('kv_bits') is None
                and (given.get('draft_model') is None or given.get('draft_kind') is None)
                and hasattr(model, 'make_cache')):
            made = plain(model, warm)
            STATS['plain' if made is not None else 'kept'] += 1
            if made is not None:
                given['warm_cache'] = made
                return original(*bound.args, **bound.kwargs)
        return original(self, *args, **kwargs)

    __init__._quanteeze_plain = True
    batch_cls.__init__ = __init__
    return True
