# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A language model's token embedding read from the checkpoint a row at a time (`QUANTEEZE_EMBED_SSD`),
for models not already served by DeepSeek V4.1's `SSDEmbedding`.  Reads rows directly through numpy's
`memmap` of the checkpoint safetensors (`nn.QuantizedEmbedding` weight/scales/biases or plain BF16/F16/F32
weight) and verifies bit-exact equality on first, last, and random rows before releasing the held arrays."""
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import variables

NAMES = ('embed_tokens',)
DTYPES = {'BF16': np.uint16, 'U32': np.uint32, 'F16': np.float16, 'F32': np.float32}


def enabled():
    """`QUANTEEZE_EMBED_SSD`: `on`, read token embeddings from the checkpoint a row at a time; `off` (default)."""
    value = (variables.get('QUANTEEZE_EMBED_SSD') or 'off').strip().lower()
    if value not in ('on', 'off'):
        raise ValueError(f'QUANTEEZE_EMBED_SSD: on or off, not {value}')
    return value == 'on'


def _weight_map(path):
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


def _maps(path, names, index=None):
    """`{name: (numpy memmap view, safetensors dtype)}` of `names` at checkpoint `path` (one map per file)."""
    from . import mapped
    if index is None:
        index = _weight_map(path)
    if index is None:
        return None
    headers, files, out = {}, {}, {}
    for name in names:
        shard = index.get(name)
        if shard is None:
            return None
        if shard not in headers:
            with open(Path(path) / shard, 'rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                headers[shard] = (length, json.loads(f.read(length)))
        length, header = headers[shard]
        t = header.get(name)
        if not isinstance(t, dict) or t.get('dtype') not in DTYPES:
            return None
        start, _ = t['data_offsets']
        out[name] = (
            mapped.view(files, Path(path) / shard, DTYPES[t['dtype']], 8 + length + start, tuple(t['shape'])),
            t['dtype'],
        )
    return out


def _array(rows, dtype):
    a = mx.array(np.ascontiguousarray(rows))
    return a.view(mx.bfloat16) if dtype == 'BF16' else a


class SSDTable(nn.Module):
    """A token embedding read from the checkpoint a row at a time: `parts` `{'weight', 'scales', 'biases'}`
    (quantized, with `quant=(group_size, bits, mode)`) or `{'weight'}` (plain), each `(memmap, dtype)`."""

    def __init__(self, parts, quant=None, shape=None):
        super().__init__()
        object.__setattr__(self, '_parts', parts)
        object.__setattr__(self, '_quant', quant)
        object.__setattr__(self, '_shape', shape)

    def __call__(self, ids):
        host = np.asarray(ids).astype(np.int64)
        unique, inverse = np.unique(host.ravel(), return_inverse=True)
        got = {k: _array(m[unique], d) for k, (m, d) in self._parts.items()}
        at = mx.array(inverse.reshape(host.shape).astype(np.int32))
        if self._quant is None:
            return got['weight'][at]
        group_size, bits, mode = self._quant
        return mx.dequantize(
            got['weight'][at],
            scales=got['scales'][at],
            biases=got['biases'][at] if 'biases' in got else None,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )


def install(model, path):
    """Replace `model`'s untied token embedding (`NAMES` under `language_model.model`) with an `SSDTable`
    when `QUANTEEZE_EMBED_SSD=on` and the checkpoint matches bit-for-bit; returns bytes released (`0` otherwise)."""
    if not enabled():
        return 0
    lm = getattr(model, 'language_model', model)
    inner = getattr(lm, 'model', None)
    if inner is None or getattr(getattr(lm, 'args', None), 'tie_word_embeddings', False):
        return 0
    for name in NAMES:
        table = inner.get(name) if isinstance(inner, dict) else None
        if isinstance(table, (nn.Embedding, nn.QuantizedEmbedding)) and type(table) in (
            nn.Embedding, nn.QuantizedEmbedding
        ):
            break
    else:
        return 0
    quantized = isinstance(table, nn.QuantizedEmbedding)
    keys = ['weight', 'scales'] + (['biases'] if quantized and table.get('biases') is not None else [])
    keys = keys if quantized else ['weight']
    try:
        index = _weight_map(path)
    except (OSError, ValueError):
        return 0
    if index is None:
        return 0
    base = next(
        (f'{p}{n}' for n in (name, 'embed') for p in ('language_model.model.', 'model.language_model.', 'model.', '')
         if f'{p}{n}.weight' in index),
        None,
    )
    if base is None:
        return 0
    try:
        maps = _maps(path, [f'{base}.{k}' for k in keys], index=index)
    except (OSError, ValueError, KeyError):
        return 0
    if maps is None:
        return 0
    parts = {k: maps[f'{base}.{k}'] for k in keys}
    if any(tuple(m.shape) != tuple(table[k].shape) for k, (m, _) in parts.items()):
        return 0
    ssd = SSDTable(parts, (table.group_size, table.bits, table.mode) if quantized else None, table.weight.shape)
    n = table.weight.shape[0]
    ids = mx.array(np.unique(np.r_[0, n - 1, np.random.default_rng(0).integers(0, n, min(64, n))])[None])
    want = table(ids)
    if not mx.array_equal(ssd(ids), want).item():
        return 0
    held = sum(table[k].nbytes for k in keys)
    inner[name] = ssd
    print(f'quanteeze: token embedding read from checkpoint on demand (QUANTEEZE_EMBED_SSD): '
          f'{held / 1e9:.2f} GB released', file=sys.stderr, flush=True)
    return held
