# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Reuse a cached prefix when a new image follows it.

mlx-vlm's prefix cache (APC) salts every snapshot of a request with one hash
of all the request's images.  When a conversation gains a plot, the salt of
every prefix changes, and the next request is prefilled from the first token:
in a KAME agent run, prompts of 62k and 67k tokens, about 70 s each.  The
cache also wants a restored prefix to cover every image, so that the suffix
is text only.

Through the server, a request carries its whole prompt's embeddings, with the
images merged in, and its multimodal (M-RoPE) positions.  BatchGenerator
resumes a cached prefix by slicing both at the prefix length.  So a prefix can
end before an image, as long as its salt says which images it holds.
install():
- hashes each image of a request apart (pixel_values split by
  image_grid_thw), in ResponseGenerator._preprocess_request;
- notes, under the salt the server computes for the request, the salt
  without images and the images' hashes in prompt order;
- salts each stored snapshot with the images inside it, and looks each
  stored prefix up by the images inside the same prefix;
- lets such a request resume before an image.
A request without images, or whose images cannot be told apart, is salted and
looked up as before.  use_model(model) gives the model's image token ids.

use_precision(precision) mixes what the served weights compute a prompt with
into every request's salt, the disk tier's too: a prefix cached under other
settings is computed again, not read.
"""
import hashlib
import threading
from collections import OrderedDict

from mlx_vlm import apc as _apc

PLANS_KEPT = 256
_MEDIA_IDS = set()
_PLANS = OrderedDict()          # the request's salt -> (salt without images, image hashes)
_LOCAL = threading.local()      # .images: this request's image hashes; .relax: a lookup that may end before an image
PRECISION = [None]              # what the served weights compute a prompt with, as text (use_precision)


def use_model(model):
    """The image and video token ids of `model`'s prompts."""
    _MEDIA_IDS.clear()
    _MEDIA_IDS.update(_apc.multimodal_token_ids_from_config(model.config))


def use_precision(precision):
    """Mix `precision()`, text, into each request's salt from now on ('' or None: nothing mixed)."""
    PRECISION[0] = precision


def _precise(salt, tag):
    if not tag:
        return salt
    return _combine(salt, int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], 'little', signed=True))


def _combine(*values):
    h = hashlib.sha256()
    for value in values:
        h.update(int(value & ((1 << 64) - 1)).to_bytes(8, 'little'))
    return int.from_bytes(h.digest()[:8], 'little', signed=True)


def image_hashes(raw):
    """Each image's content hash, in prompt order, or None when they cannot be told apart."""
    pixels, grid = raw.get('pixel_values'), raw.get('image_grid_thw')
    if isinstance(pixels, (list, tuple)):
        return _image_inputs(pixels)
    if pixels is None or grid is None or raw.get('pixel_values_videos') is not None:
        return None
    grid = [tuple(int(v) for v in row) for row in grid.tolist()]
    sizes = [t * h * w for t, h, w in grid]
    if not sizes or sum(sizes) != pixels.shape[0]:
        return None
    hashes, start = [], 0
    for size, thw in zip(sizes, grid):
        hashes.append(_combine(_apc.hash_image_payload(pixel_values=pixels[start:start + size]), *thw))
        start += size
    return tuple(hashes)


def _image_inputs(rows):
    """DeepSeek V4.1's: its processor's pixel_values a list of rows, each of ImageInputs (an
    image's ViT patches and layout) in prompt order."""
    images = [image for row in rows for image in (row if isinstance(row, (list, tuple)) else [row])]
    if not images or not all(hasattr(i, 'patches') and hasattr(i, 'types') for i in images):
        return None
    return tuple(_combine(_apc.hash_image_payload(pixel_values=i.patches), int(i.n_vit_h), int(i.n_vit_w),
                          len(i.types)) for i in images)


def _salt(plan, images):
    base, hashes = plan
    return base if images == 0 else _combine(base, *hashes[:images])


def prefix_salt(plan, token_ids):
    """The salt of a stored prefix: the images inside it, or None if it holds more than the plan."""
    images = len(_apc.media_token_spans(token_ids, _MEDIA_IDS))
    return _salt(plan, images) if images <= len(plan[1]) else None


def segments(plan, token_ids):
    """(salt, lo, hi) for each count of images, the most first: prefixes of length lo < L <= hi.

    None when the prompt's image spans do not match the plan's images.
    """
    starts = [start for start, _ in _apc.media_token_spans(token_ids, _MEDIA_IDS)]
    if len(starts) != len(plan[1]):
        return None
    bounds = [0] + starts + [len(token_ids)]
    return [(_salt(plan, m), bounds[m], bounds[m + 1]) for m in range(len(starts), -1, -1)]


def _note(full, base, hashes):
    _PLANS[full] = (base, hashes)
    _PLANS.move_to_end(full)
    while len(_PLANS) > PLANS_KEPT:
        _PLANS.popitem(last=False)


def install(generator_cls, batch_cls, manager_cls):
    """Wrap ResponseGenerator, BatchGenerator and APCManager (mlx-vlm's) for image-aware prefixes."""
    preprocess = generator_cls._preprocess_request
    semantic = _apc.semantic_extra_hash
    lookup, store = manager_cls.lookup_exact_cache, manager_cls.store_exact_cache
    pick, safe_min, text_only = (batch_cls._apc_pick_for, batch_cls._apc_safe_prefix_lookup_min,
                                 batch_cls._apc_suffix_is_text_only)

    def _preprocess_request(self, *args, **kwargs):
        raw = preprocess(self, *args, **kwargs)
        _LOCAL.images = image_hashes(raw) if isinstance(raw, dict) else None
        return raw

    def semantic_extra_hash(*, tenant=None, image_hash=0, media=None, model=None, processor=None):
        tag = PRECISION[0]() if PRECISION[0] is not None and model is not None else None
        full = _precise(semantic(tenant=tenant, image_hash=image_hash, media=media, model=model,
                                 processor=processor), tag)
        hashes, _LOCAL.images = getattr(_LOCAL, 'images', None), None
        if hashes and image_hash and not any(v is not None for v in (media or {}).values()):
            base = _precise(semantic(tenant=tenant, image_hash=0, media=media, model=model, processor=processor), tag)
            _note(full, base, hashes)
        return full

    def lookup_exact_cache(self, token_ids, extra_hash=0, max_prefix_tokens=None, min_prefix_tokens=0):
        plan = _PLANS.get(extra_hash)
        parts = segments(plan, token_ids) if plan is not None else None
        if parts is None:
            return lookup(self, token_ids, extra_hash=extra_hash, max_prefix_tokens=max_prefix_tokens,
                          min_prefix_tokens=min_prefix_tokens)
        for salt, lo, hi in parts:
            if max_prefix_tokens is not None and max_prefix_tokens > 0:
                hi = min(hi, max_prefix_tokens)
            if hi <= max(lo, min_prefix_tokens):
                continue
            cache, length = lookup(self, token_ids, extra_hash=salt, max_prefix_tokens=hi,
                                   min_prefix_tokens=max(lo, min_prefix_tokens))
            if cache is not None and length > 0:
                return cache, length
        return None, 0

    def store_exact_cache(self, token_ids, prompt_cache, *, extra_hash=0):
        plan = _PLANS.get(extra_hash)
        salt = prefix_salt(plan, token_ids) if plan is not None else None
        return store(self, token_ids, prompt_cache, extra_hash=extra_hash if salt is None else salt)

    def _apc_pick_for(self, sequence):
        ids, kwargs = sequence[1], sequence[3] or {}
        embeds = kwargs.get('inputs_embeds')
        relax = (self.apc_manager is not None and embeds is not None and embeds.ndim == 3
                 and embeds.shape[1] == len(ids) and self._apc_extra_hash(kwargs) in _PLANS)
        before, _LOCAL.relax = getattr(_LOCAL, 'relax', False), relax
        try:
            return pick(self, sequence)
        finally:
            _LOCAL.relax = before

    def _apc_safe_prefix_lookup_min(self, ids_list):
        return 0 if getattr(_LOCAL, 'relax', False) else safe_min(self, ids_list)

    def _apc_suffix_is_text_only(self, ids_list, prefix_len):
        return True if getattr(_LOCAL, 'relax', False) else text_only(self, ids_list, prefix_len)

    generator_cls._preprocess_request = _preprocess_request
    _apc.semantic_extra_hash = semantic_extra_hash
    manager_cls.lookup_exact_cache = lookup_exact_cache
    manager_cls.store_exact_cache = store_exact_cache
    batch_cls._apc_pick_for = _apc_pick_for
    batch_cls._apc_safe_prefix_lookup_min = _apc_safe_prefix_lookup_min
    batch_cls._apc_suffix_is_text_only = _apc_suffix_is_text_only
