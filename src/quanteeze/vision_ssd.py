# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""On-demand SSD checkpoint loading for vision towers (`QUANTEEZE_VISION_SSD`).

Each vision-tower BF16 weight whose checkpoint bytes match its in-memory values bit-for-bit is
released to an empty placeholder and streamed via numpy `memmap` one block at a time when encoding
images, then immediately released after `mx.eval`.  Encoded image embeddings are cached in an LRU
(`IMAGES`, up to `IMAGES_KEPT_GB`) across multi-turn conversations."""
import hashlib
import json
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import variables

TOWERS = ('vision_tower',)
PREFIXES = ('vision_tower.', 'model.visual.', 'visual.', 'model.vision_tower.')
_AUTO = {}                      # {(path, working_set, bits, gguf, reserve, whole, fs): bool}


def vision_whole(path, model=None):
    """Whether holding the vision tower still leaves room for all routed experts in memory."""
    from mlx.utils import tree_flatten
    from . import arch, fitting, gguf_experts, guards, parallel

    if (variables.get('QUANTEEZE_LOAD_WHOLE') or '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return True
    working_set = guards.recommended_budget()
    if not working_set:
        return True
    if model is not None:
        try:
            layers = arch.decoder_layers(model)
        except AttributeError:
            layers = []
        moe = [(i, arch.switch_of(l)) for i, l in enumerate(layers) if arch.switch_of(l) is not None]
        if not moe:
            resident = sum(a.nbytes for _, a in arch.in_memory(model))
            return resident <= working_set * (1 - fitting.reserve())
        switches = [sw for _, sw in moe]
        sw_bytes = sum(a.nbytes for s in switches for _, a in tree_flatten(s.parameters()))
        resident = sum(a.nbytes for _, a in arch.in_memory(model)) - sw_bytes
        space = fitting.room(resident, working_set)
        gguf_path = parallel._experts_gguf_path()
        target_bits = None if gguf_path is not None else parallel._bits_target()
        if gguf_path is not None:
            experts_bytes = gguf_experts.ExpertGGUF(gguf_path).total_bytes([i for i, _ in moe])
        elif target_bits is not None:
            experts_bytes = sum(fitting.quantized_bytes(fitting.weights(s), target_bits) for s in switches)
        else:
            experts_bytes = sw_bytes
        return space is None or experts_bytes <= space

    layer_bytes, rest = parallel.sizes(path, include_vision=True)
    meta = parallel._config_meta(path)
    text_cfg = meta.get('text_config') or {}
    model_type = meta.get('model_type') or text_cfg.get('model_type')
    if (not variables.is_set('QUANTEEZE_MEMORY_RESERVE') and model_type in fitting.RESERVE_ON_CUDA
            and not mx.metal.is_available() and mx.default_device() == mx.gpu):
        res = fitting.RESERVE_ON_CUDA[model_type]
    else:
        res = fitting.reserve()
    return (sum(layer_bytes.values()) + rest) <= working_set * (1 - res)


def reads(path, model=None):
    """Decision for `QUANTEEZE_VISION_SSD=auto` on `path`: stream on demand if holding the
    vision tower keeps routed experts from fitting whole, unless `path` is on a network FS."""
    from . import guards, q2_experts
    key = str(Path(path).expanduser().resolve())
    fs = q2_experts.file_system(key)
    cache_key = (
        key,
        guards.recommended_budget(),
        variables.get('QUANTEEZE_BITS'),
        variables.get('QUANTEEZE_EXPERTS_GGUF'),
        variables.get('QUANTEEZE_MEMORY_RESERVE'),
        variables.get('QUANTEEZE_LOAD_WHOLE'),
        variables.get('QUANTEEZE_V41_READS_IMAGES'),
        fs,
    )
    if cache_key not in _AUTO:
        if fs in q2_experts.NETWORK:
            _AUTO[cache_key], why = False, f'held: checkpoint lies on a network file system ({fs})'
        elif vision_whole(key, model=model):
            _AUTO[cache_key], why = False, 'held: routed experts fit whole beside it'
        else:
            _AUTO[cache_key], why = True, (
                'streamed from checkpoint on demand: holding it leaves routed experts less room than they take'
            )
        print(f'quanteeze: QUANTEEZE_VISION_SSD=auto: vision tower {why}', file=sys.stderr, flush=True)
    return _AUTO[cache_key]


def enabled(path=None, model=None):
    """`QUANTEEZE_VISION_SSD`: `auto` (default), `on`, or `off`."""
    value = (variables.get('QUANTEEZE_VISION_SSD') or 'auto').strip().lower()
    if value not in ('auto', 'on', 'off'):
        raise ValueError(f'QUANTEEZE_VISION_SSD: auto, on, or off, not {value}')
    if value != 'auto':
        return value == 'on'
    return path is not None and reads(path, model=model)


def weight_map(path):
    """Return `{tensor_name: shard_filename}` for sharded or single-file safetensors at `path`."""
    p = Path(path)
    idx_file = p / 'model.safetensors.index.json'
    if idx_file.is_file():
        return json.loads(idx_file.read_text())['weight_map']
    single = p / 'model.safetensors'
    if single.is_file():
        with open(single, 'rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(length))
        return {k: 'model.safetensors' for k in header if k != '__metadata__'}
    return None


class VisionWeights:
    """A subset of vision-tower weights streamed from `maps` on `load()` and freed to empty arrays on `release()`."""

    def __init__(self, maps, names):
        self.maps, self.names = maps, names

    def load(self):
        for name, (root, key, dtype) in self.names.items():
            a = mx.array(np.asarray(self.maps[name])).view(mx.bfloat16)
            set_param(root, key, a if dtype == mx.bfloat16 else a.astype(dtype))

    def release(self):
        for name, (root, key, dtype) in self.names.items():
            set_param(root, key, mx.zeros((0,), dtype))


def set_param(root, key, value):
    """Set `root`'s parameter at dotted `key` (indexing into `list` containers by integer part) to `value`."""
    *path, last = key.split('.')
    m = root
    for part in path:
        m = m[int(part)] if isinstance(m, list) else m[part]
    m[last] = value


def bf16_maps(path, names, index=None):
    """`{name: uint16 memmap view}` for checkpoint BF16 tensors in `names` at `path` (one map per file)."""
    from . import mapped
    if index is None:
        index = weight_map(path)
    if index is None:
        return {}
    headers, files, out = {}, {}, {}
    for name in names:
        shard = index.get(name)
        if shard is None:
            continue
        if shard not in headers:
            with open(Path(path) / shard, 'rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                headers[shard] = (length, json.loads(f.read(length)))
        length, header = headers[shard]
        tensor = header.get(name)
        if not isinstance(tensor, dict) or tensor.get('dtype') != 'BF16':
            continue
        start, _ = tensor['data_offsets']
        out[name] = mapped.view(
            files, Path(path) / shard, np.uint16, 8 + length + start, tuple(tensor['shape'])
        )
    return out


def matching(path, params):
    """Filter `params` `[(candidate_names, module, key, array)]` to those whose checkpoint BF16 bytes
    equal `array` bit-for-bit: returns `({ckpt_name: (module, key, dtype)}, maps, freed_bytes)`."""
    try:
        index = weight_map(path)
    except (OSError, ValueError, KeyError):
        return {}, {}, 0
    if index is None:
        return {}, {}, 0
    chosen = {}
    for names, root, key, a in params:
        name = next((n for n in names if n in index), None)
        if name is not None and a.size:
            chosen[name] = (root, key, a)
    try:
        maps = bf16_maps(path, list(chosen), index=index)
    except (OSError, ValueError, KeyError):
        return {}, {}, 0
    taken, freed = {}, 0
    for name, (root, key, a) in chosen.items():
        got = maps.get(name)
        if got is None or tuple(got.shape) != tuple(a.shape):
            continue
        on_disk = mx.array(np.asarray(got)).view(mx.bfloat16)
        if not mx.array_equal(on_disk if a.dtype == mx.bfloat16 else on_disk.astype(a.dtype), a).item():
            continue
        taken[name] = (root, key, a.dtype)
        freed += a.nbytes
    return taken, maps, freed


def _evaluated(call):
    def __call__(self, *args, **kwargs):
        ssd = self.__dict__.get('_quanteeze_from_ssd')
        if ssd is None:
            y = call(self, *args, **kwargs)
            if self.__dict__.get('_quanteeze_blocks'):
                mx.eval(y)
            return y
        ssd.load()
        try:
            y = call(self, *args, **kwargs)
            mx.eval(y)
        finally:
            ssd.release()
        return y

    __call__._quanteeze_vision_ssd = call
    return __call__


def _hook(cls, wrap):
    if not getattr(cls.__call__, '_quanteeze_vision_ssd', None):
        cls.__call__ = wrap(cls.__call__)


IMAGES_KEPT_GB = 1.0
IMAGES = OrderedDict()
IMAGE_STATS = dict(encoded=0, reused=0)


def _key(args, kwargs):
    parts = []
    for a in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if isinstance(a, mx.array):
            parts.append(
                np.asarray(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).tobytes() + str(a.shape).encode()
            )
        elif isinstance(a, (int, float, bool, type(None))):
            parts.append(repr(a).encode())
        else:
            return None
    return hashlib.blake2b(b'|'.join(parts), digest_size=16).digest()


def _tower(call):
    def __call__(self, *args, **kwargs):
        key = _key(args, kwargs) if self.__dict__.get('_quanteeze_keep') else None
        if key is not None and key in IMAGES:
            IMAGES.move_to_end(key)
            IMAGE_STATS['reused'] += 1
            return IMAGES[key]
        rest = self.__dict__.get('_quanteeze_vision_rest')
        if rest is not None:
            rest.load()
        try:
            y = call(self, *args, **kwargs)
            mx.eval(y)
        finally:
            if rest is not None:
                rest.release()
        IMAGE_STATS['encoded'] += 1
        if key is not None:
            from mlx.utils import tree_flatten
            IMAGES[key] = y
            while IMAGES and sum(a.nbytes for v in IMAGES.values() for _, a in tree_flatten(v)) > IMAGES_KEPT_GB * 1e9:
                IMAGES.popitem(last=False)
        return y

    __call__._quanteeze_vision_ssd = call
    return __call__


def tower_of(model):
    for attr in TOWERS:
        tower = model.get(attr) if isinstance(model, dict) else None
        if isinstance(tower, nn.Module):
            return tower
    return None


def install(model, path):
    """Hook `model`'s vision tower for block-by-block evaluation and embedding caching, and when
    `QUANTEEZE_VISION_SSD` enables it (`on`, or `auto` when memory is tight), release matching BF16
    weights to stream from `path` on demand."""
    from mlx.utils import tree_flatten
    tower = tower_of(model)
    if tower is None:
        return 0
    blocks = getattr(tower, 'blocks', None) or []
    for block in blocks:
        object.__setattr__(block, '_quanteeze_blocks', True)
    if blocks:
        _hook(type(blocks[0]), _evaluated)
    object.__setattr__(tower, '_quanteeze_keep', True)
    _hook(type(tower), _tower)
    if not enabled(path, model=model):
        return 0
    params = [([p + k for p in PREFIXES], tower, k, a) for k, a in tree_flatten(tower.parameters())]
    taken, maps, freed = matching(path, params)
    if not taken:
        return 0
    rest = dict(taken)
    for b, block in enumerate(blocks):
        mine = {name: v for name, v in taken.items() if v[1].startswith(f'blocks.{b}.')}
        for name in mine:
            del rest[name]
        if mine:
            object.__setattr__(block, '_quanteeze_from_ssd', VisionWeights(maps, mine))
    object.__setattr__(tower, '_quanteeze_vision_rest', VisionWeights(maps, rest) if rest else None)
    VisionWeights(maps, taken).release()
    kept = sum(a.nbytes for _, a in tree_flatten(tower.parameters()))
    print(f'quanteeze: vision tower streamed from checkpoint on demand (QUANTEEZE_VISION_SSD): '
          f'{freed / 1e9:.2f} GB released ({len(taken)} weights), {kept / 1e9:.2f} GB kept in memory',
          file=sys.stderr, flush=True)
    return freed
