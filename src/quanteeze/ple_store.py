# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Flash-Next's n-gram (PLE) table read from the SSD row by row, whatever its checkpoint stores.

A Qwen4-Exp checkpoint keeps its n-gram embedding table as shards of a quantized embedding: 32
GB in mlx-community's 4-bit build, 57.6 GB in TensorFold's 8-bit one, too much to hold beside the
experts.  mlx-vlm reads it from the SSD (qwen4_exp.ple_storage: a view of the checkpoint whose
config names a manifest of where each shard's arrays lie), but only at 4 bits in groups of 32 or
nvfp4, and only shards named `ngram_embedding.shards.<n>`: the 8-bit build, which names them
`shard_<n>`, could not be served.  Here, for any layout (as many
models as can be):

- prepare(source, target, rows=False): the view.  The table's shards are found by either name;
  each tensor's dtype and shape come from the safetensors headers, its quantization from the
  config (bits, group size, mode: affine, mxfp4/mxfp8, nvfp4; or none, the table in bf16, f16
  or f32); every other file is hard-linked or copied, and the config points the model to the
  manifest.  rows=True also writes the table one record a row (ple.rows beside the view), so a
  row's weights, scales and biases lie on one page.
- The manifest says `format: quanteeze_ple_v1`.  mlx-vlm's QuantizedMMapNGramEmbedding takes it
  once installed (install(): its __init__, _read_rows and __call__ hand such a manifest to the
  code here and anything else to its own), so the model, which builds that class from its
  config, and quanteeze.ngram_prefetch, which wraps its _read_rows, need nothing else.  A
  lookup dequantizes the rows it read as mx.dequantize does the whole shard, bit for bit.
  quanteeze installs it when mlx-vlm's ple_storage is first
  imported (quanteeze/__init__.py): before any model is built, in the server and every script.

mlx-community's 4-bit view keeps mlx-vlm's own manifest and code.
"""
import json
import mmap
import os
import re
import shutil
import struct
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np

FORMAT = 'quanteeze_ple_v1'
SHARD = re.compile(r'^(?P<table>.+\.ple\.ple_embedding\.ngram_embedding)\.shards?[._](?P<n>\d+)'
                   r'\.(?P<part>weight|scales|biases)$')
PARTS = ('weight', 'scales', 'biases')
DTYPES = {'U32': np.dtype('<u4'), 'U16': np.dtype('<u2'), 'BF16': np.dtype('<u2'), 'F16': np.dtype('<f2'),
          'F32': np.dtype('<f4'), 'U8': np.dtype('u1')}


def _headers(root, files):
    """{file: (where its payload starts, its header)} of safetensors files."""
    out = {}
    for name in files:
        with open(root / name, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            out[name] = (8 + n, json.loads(f.read(n)))
    return out


def _quantization(config, table, n):
    """The table's quantization from the config: a shard's own entry under either name, else the
    model's; None when the config quantizes nothing."""
    q = config.get('quantization_config') or config.get('quantization') or {}
    for key in (f'{table}.shards.{n}', f'{table}.shard_{n}'):
        own = q.get(key) or q.get(key.replace('language_model.model.', 'language_model.'))
        if isinstance(own, dict):
            q = own
            break
    if not q.get('bits'):
        return None
    return {'bits': int(q['bits']), 'group_size': int(q['group_size']), 'mode': q.get('mode') or 'affine'}


def manifest(source, root_for):
    """The table's manifest: its shards' arrays in the checkpoint at `source`, in order of shard,
    with paths relative to `root_for` (where the manifest will lie)."""
    source = Path(source).resolve()
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    config = json.loads((source / 'config.json').read_text())
    found = {}
    for key, file in index['weight_map'].items():
        m = SHARD.match(key)
        if m:
            found.setdefault(m['table'], {}).setdefault(int(m['n']), {})[m['part']] = (key, file)
    if len(found) != 1:
        raise ValueError(f'{source}: {len(found)} n-gram tables (one is served): {sorted(found)}')
    (table, shards), = found.items()
    headers = _headers(source, {file for parts in shards.values() for _, file in parts.values()})
    quantization, tensors, out, start = None, None, [], 0
    for n in sorted(shards):
        if n != len(out):
            raise ValueError(f'{table}: shard {len(out)} is missing')
        parts = shards[n]
        q = _quantization(config, table, n) if 'scales' in parts else None
        names = [p for p in PARTS if p in parts]
        entry = {}
        for p in names:
            key, file = parts[p]
            begin, header = headers[file]
            info = header[key]
            if info['dtype'] not in DTYPES or len(info['shape']) != 2:
                raise ValueError(f'{key}: {info["dtype"]} {info["shape"]} is not a table this reads')
            entry[p] = {'file': file, 'offset': begin + int(info['data_offsets'][0]), 'dtype': info['dtype'],
                        'shape': [int(d) for d in info['shape']]}
        layout = [(p, entry[p]['dtype'], entry[p]['shape'][1]) for p in names]
        if out and (q != quantization or layout != tensors):
            raise ValueError(f'{table}: shard {n} is stored otherwise than shard 0')
        quantization, tensors = q, layout
        count = entry['weight']['shape'][0]
        if any(entry[p]['shape'][0] != count for p in names):
            raise ValueError(f'{table}: shard {n}\'s arrays have different row counts')
        out.append({'row_start': start, 'row_count': count, **entry})
        start += count
    cols = tensors[0][2]
    width = cols * 32 // quantization['bits'] if quantization else cols
    if quantization and (tensors[0][1] != 'U32' or width % quantization['group_size']
                         or any(c != width // quantization['group_size'] for _, _, c in tensors[1:])):
        raise ValueError(f'{table}: arrays {tensors} do not fit {quantization}')
    return {'format': FORMAT, 'table': table, 'source_root': os.path.relpath(source, Path(root_for).resolve()),
            'layout': 'safetensors_ranges', 'row_width': width, 'row_count': start, 'quantization': quantization,
            'tensors': [list(t) for t in tensors], 'cache_rows': 0, 'shards': out}


def write_rows(view, chunk_rows=131_072, data_file='ple.rows'):
    """The table of the view at `view` rewritten one record a row, beside its manifest; the
    manifest then points there."""
    view = Path(view)
    path = view / json.loads((view / 'config.json').read_text())['text_config']['ple_storage']['manifest']
    m = json.loads(path.read_text())
    source = (path.parent / m['source_root']).resolve()
    widths = [c * DTYPES[d].itemsize for _, d, c in m['tensors']]
    stride, total = sum(widths), m['row_count'] * sum(widths)
    fd, partial = tempfile.mkstemp(prefix=f'.{data_file}.', suffix='.partial', dir=path.parent)
    os.close(fd)
    partial = Path(partial)
    try:
        with partial.open('wb') as f:
            f.truncate(total)
        for shard in m['shards']:
            arrays = [np.memmap(source / shard[p]['file'], dtype=DTYPES[d], mode='r', offset=shard[p]['offset'],
                                shape=tuple(shard[p]['shape'])) for p, d, _ in m['tensors']]
            for a in range(0, shard['row_count'], chunk_rows):
                b = min(a + chunk_rows, shard['row_count'])
                out = np.memmap(partial, dtype=np.uint8, mode='r+', offset=(shard['row_start'] + a) * stride,
                                shape=(b - a, stride))
                at = 0
                for array, w in zip(arrays, widths):
                    out[:, at:at + w] = np.asarray(array[a:b]).view(np.uint8).reshape(b - a, w)
                    at += w
                out.flush()
                del out
        os.link(partial, path.parent / data_file)
    finally:
        partial.unlink(missing_ok=True)
    m.update(layout='interleaved', data_file=data_file, row_stride=stride)
    del m['shards']
    path.write_text(json.dumps(m, indent=1) + '\n')
    return m


MLX_VLM_VERSION = 2      # mlx-vlm's QuantizedMMapNGramEmbedding.MANIFEST_VERSION, where it cannot be imported
MLX_VLM_FIELDS = ('version', 'source_root', 'row_width', 'quantization')


def readable(view):
    """Whether the view at `view` names a manifest the model's table can be read from: this module's
    (`format: quanteeze_ple_v1`), or mlx-vlm's own at the version it reads.  A view another tool made,
    in a format of its own, is not: a load of it would fail on its manifest."""
    view = Path(view)
    try:
        config = json.loads((view / 'config.json').read_text())
        name = ((config.get('text_config') or config).get('ple_storage') or {}).get('manifest')
        m = json.loads((view / name).read_text()) if name else None
    except (OSError, ValueError, AttributeError, TypeError):
        return False
    if not isinstance(m, dict):
        return False
    if m.get('format') == FORMAT:
        return True
    if 'format' in m or any(k not in m for k in MLX_VLM_FIELDS):
        return False
    try:
        from mlx_vlm.models.qwen4_exp.ple_storage import QuantizedMMapNGramEmbedding
        version = QuantizedMMapNGramEmbedding.MANIFEST_VERSION
    except Exception:
        version = MLX_VLM_VERSION
    return m['version'] == version


def prepare(source, target, rows=False, cache_rows=0):
    """A view of the checkpoint at `source`, at `target` (empty or new, on the same volume): every
    file but the n-gram table's arrays, hard-linked or copied, and the table read from the SSD as
    its manifest says (rows=True: from one record a row, written beside it).  Returns the manifest."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'{target} is not empty')
    target.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != target.stat().st_dev:
        raise ValueError(f'{source} and {target} are on different volumes: no hard links')
    m = manifest(source, target)
    m['cache_rows'] = int(cache_rows)
    (target / 'ple-store.json').write_text(json.dumps(m, indent=1) + '\n')
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    index['weight_map'] = {k: f for k, f in index['weight_map'].items() if not SHARD.match(k)}
    files = sorted(set(index['weight_map'].values()))
    for name in files:
        os.link(source / name, target / name)
    index.setdefault('metadata', {})['external_ple_bytes'] = str(sum(
        int(np.prod(s[p]['shape'])) * DTYPES[s[p]['dtype']].itemsize for s in m['shards'] for p, _, _ in m['tensors']))
    (target / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    for f in source.iterdir():
        if f.is_file() and f.suffix != '.safetensors' and f.name not in ('config.json', 'model.safetensors.index.json'):
            shutil.copy2(f, target / f.name)
    config = json.loads((source / 'config.json').read_text())
    config.setdefault('text_config', {})['ple_storage'] = {'manifest': 'ple-store.json', 'cache_rows': int(cache_rows)}
    for key in ('quantization', 'quantization_config'):
        if isinstance(config.get(key), dict):
            config[key] = {k: v for k, v in config[key].items() if not (isinstance(v, dict) and m['table'] in k)}
    (target / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (target / 'EXTERNAL_PLE.json').write_text(json.dumps(
        {'format': 'quanteeze_external_ple_v1', 'source_model': str(source), 'external_ple_manifest': 'ple-store.json',
         'linked_weight_files': files}, indent=2) + '\n')
    if rows:
        m = write_rows(target)
    return m


# -- the reading side, on mlx-vlm's QuantizedMMapNGramEmbedding --------------------------------

def _init(self, path, cache_rows=None):
    """A table of this module's manifest: what mlx-vlm's class keeps, for its stats and for
    quanteeze.ngram_prefetch (_interleaved, _shards, _tensor_names), set from it."""
    path = Path(path)
    m = json.loads(path.read_text())
    self._generic = m
    self.quantization = m['quantization']
    self._tensors = [(p, d, int(c)) for p, d, c in m['tensors']]
    self._tensor_names = tuple(p for p, _, _ in self._tensors)
    self._tensor_layout = tuple((p, d) for p, d, _ in self._tensors)
    self._group_size = int(m['quantization']['group_size']) if m['quantization'] else 1
    self.row_width, self.row_count = int(m['row_width']), int(m['row_count'])
    source = (path.parent / m['source_root']).resolve()
    self._interleaved, self._shards = None, []
    if m['layout'] == 'interleaved':
        stride = int(m['row_stride'])
        data = path.parent / m['data_file']
        if data.stat().st_size != self.row_count * stride:
            raise ValueError(f'{data}: {data.stat().st_size} bytes, not {self.row_count} rows of {stride}')
        self._interleaved = _random(np.memmap(data, dtype=np.uint8, mode='r', shape=(self.row_count, stride)))
    else:
        files = {}
        for shard in m['shards']:
            start, count = int(shard['row_start']), int(shard['row_count'])
            arrays = {
                p: _within(files, source / shard[p]['file'], DTYPES[d], int(shard[p]['offset']),
                           tuple(shard[p]['shape']))
                for p, d, _ in self._tensors
            }
            self._shards.append((start, start + count, arrays))
    self.cache_rows = int(m.get('cache_rows', 0) if cache_rows is None else cache_rows)
    self._cache = OrderedDict()
    self._lookups = self._rows = self._hits = self._misses = self._bytes = 0
    self._elapsed = 0.0


def _within(files, file, dtype, offset, shape, advise=True):
    """Map `file` once per file via `quanteeze.mapped.view` and return a typed view at `offset`."""
    from . import mapped
    first = Path(file) not in files
    out = mapped.view(files, file, dtype, offset, shape)
    if first and advise:
        _random(files[Path(file)])
    return out


def _shared(self):
    """Collapse per-shard/tensor memmaps from mlx-vlm's loader into views of one mapping per file."""
    files = {}
    for k, (start, end, arrays) in enumerate(getattr(self, '_shards', None) or []):
        if all(isinstance(a, np.memmap) and a.filename for a in arrays.values()):
            self._shards[k] = (
                start,
                end,
                {
                    name: _within(files, Path(a.filename), a.dtype, int(a.offset), a.shape, advise=False)
                    for name, a in arrays.items()
                },
            )


def _random(table):
    """A memmapped table read row by row without the kernel's read-ahead (madvise MADV_RANDOM), off a Mac:
    a decode token's dozen rows of 1.4 KB each faulted in 128 KB around it on Linux (read_ahead_kb), some
    0.86 MB a token from a SATA SSD for Qwen3.8, its file cache crowded out by host-cached experts.
    The Mac's reads are left as they were measured."""
    if mx.metal.is_available():
        return table
    try:
        table._mmap.madvise(mmap.MADV_RANDOM)
    except (AttributeError, OSError, ValueError):
        pass
    return table


def _read_rows(self, row_ids):
    """Each distinct row of row_ids once: (one array a tensor, [rows, columns]), as mlx-vlm's."""
    row_ids = np.asarray(row_ids, dtype=np.int64)
    rows, missing = [None] * len(row_ids), []
    for i, r in enumerate(row_ids.tolist()):
        hit = self._cache.get(r)
        if hit is None:
            missing.append((i, r))
        else:
            self._hits += 1
            self._cache.move_to_end(r)
            rows[i] = hit
    ids = np.array([r for _, r in missing], np.int64)
    if ids.size and self._interleaved is not None:
        raw = np.array(self._interleaved[ids], copy=True)
        columns, at = [], 0
        for _, d, c in self._tensors:
            w = c * DTYPES[d].itemsize
            columns.append(raw[:, at:at + w].reshape(ids.size, -1).view(DTYPES[d]))
            at += w
        loaded = list(zip(*columns))
        self._bytes += raw.nbytes
    elif ids.size:
        loaded = [None] * ids.size
        for start, end, arrays in self._shards:
            sel = np.flatnonzero((ids >= start) & (ids < end))
            if sel.size:
                columns = [np.array(arrays[p][ids[sel] - start], copy=True) for p in self._tensor_names]
                self._bytes += sum(c.nbytes for c in columns)
                for k, row in zip(sel.tolist(), zip(*columns)):
                    loaded[k] = row
        if any(row is None for row in loaded):
            raise IndexError('n-gram row outside the table')
    else:
        loaded = []
    self._misses += ids.size
    for (i, r), row in zip(missing, loaded):
        rows[i] = row
        if self.cache_rows:
            self._cache[r] = row
            while len(self._cache) > self.cache_rows:
                self._cache.popitem(last=False)
    if not rows:
        return tuple(np.empty((0, c), DTYPES[d]) for _, d, c in self._tensors)
    return tuple(np.stack([row[k] for row in rows]) for k in range(len(self._tensors)))


def _array(values, dtype):
    """A numpy array of a safetensors dtype as MLX holds it (bf16 from its bits)."""
    return mx.array(values.view(np.uint16)).view(mx.bfloat16) if dtype == 'BF16' else mx.array(values)


def _lookup(self, row_ids):
    """The rows row_ids names, [..., row_width], as the model's own embedding gives them: dequantized
    in its scales' dtype, or as stored."""
    started = time.perf_counter()
    ids = np.asarray(row_ids).astype(np.int64, copy=False)
    if ids.size and (ids.min() < 0 or ids.max() >= self.row_count):
        raise IndexError('n-gram row outside the table')
    flat = ids.reshape(-1)
    if not flat.size:
        return mx.zeros((*ids.shape, self.row_width), dtype=mx.bfloat16)
    unique, inverse = np.unique(flat, return_inverse=True)
    parts = [_array(a, d) for a, (_, d, _) in zip(self._read_rows(unique), self._tensors)]
    self._hits += flat.size - unique.size
    q = self.quantization
    if q is None:
        values = parts[0]
    elif q['mode'] == 'affine':
        values = mx.dequantize(parts[0], parts[1], parts[2] if len(parts) > 2 else None,
                               group_size=q['group_size'], bits=q['bits'])
    else:
        values = mx.dequantize(parts[0], parts[1], group_size=q['group_size'], bits=q['bits'], mode=q['mode'])
    values = values[mx.array(inverse, dtype=mx.int64)]
    self._lookups += 1
    self._rows += ids.size
    self._elapsed += time.perf_counter() - started
    return values.reshape(*ids.shape, self.row_width)


def install(cls=None):
    """mlx-vlm's QuantizedMMapNGramEmbedding (or `cls`) takes this module's manifests too, its own as
    before.  Once a class."""
    if cls is None:
        from mlx_vlm.models.qwen4_exp.ple_storage import QuantizedMMapNGramEmbedding as cls
    if getattr(cls, '_quanteeze_ple', False):
        return cls
    own_init, own_read, own_call = cls.__init__, cls._read_rows, cls.__call__

    def __init__(self, manifest_path, *, cache_rows=None):
        if json.loads(Path(manifest_path).read_text()).get('format') == FORMAT:
            return _init(self, manifest_path, cache_rows)
        self._generic = None
        own_init(self, manifest_path, cache_rows=cache_rows)
        _shared(self)

    def read(self, row_ids):
        return (_read_rows if getattr(self, '_generic', None) else own_read)(self, row_ids)

    def call(self, row_ids):
        return (_lookup if getattr(self, '_generic', None) else own_call)(self, row_ids)

    cls.__init__, cls._read_rows, cls.__call__, cls.lookup = __init__, read, call, call
    cls._quanteeze_ple = True
    return cls
