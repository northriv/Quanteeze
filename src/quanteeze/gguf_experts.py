# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Read routed MoE experts directly from a pre-distributed GGUF (single file, split shards
`*-00001-of-0000N.gguf`, or directory) and serve them without converting to MLX containers.

Decodes each expert tensor directly in whatever standard ggml block format the GGUF file stores
(`IQ*`, `Q*_K`, `Q4_0`..`Q8_0`, `F16`, `BF16`, including `UD-*` builds), both loaded whole in memory
via `load_model()` and streamed on demand from SSD through `demand.Demand`.
"""
from concurrent.futures import ThreadPoolExecutor
import gc
import re
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import arch, demand, expert_qmv, iq_experts, q2_experts
from .q2_experts import NAMES, PROJECTIONS

PART_RE = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$')
# gguf_type -> fixed scalar byte size (8=string, 9=array handled explicitly)
_KV_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_ALIGN_FMTS = {4: '<I', 5: '<i', 10: '<Q', 11: '<q'}
_STACKED_RE = re.compile(r'^blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.weight$')
_PER_EXPERT_RE = re.compile(r'^blk\.(\d+)\.(ffn_(?:gate|up|down))\.(\d+)\.weight$')
_GGUF_TO_PROJ = {v: k for k, v in PROJECTIONS.items()}
_PER_EXPERT_TO_PROJ = {'ffn_gate': 'gate_proj', 'ffn_up': 'up_proj', 'ffn_down': 'down_proj'}


def parts(path):
    """Return the GGUF file list for `path` (a single `.gguf`, any split shard, or a directory)."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f'{path}: GGUF path does not exist')
    if p.is_dir():
        found = sorted(f for f in p.glob('*.gguf') if not f.name.startswith('.'))
        if not found:
            raise FileNotFoundError(f'{path}: no .gguf files found')
        return found
    m = PART_RE.match(p.name)
    if not m:
        return [p]
    shards = sorted(s for s in p.parent.glob(f'{m.group(1)}-*-of-{m.group(3)}.gguf') if PART_RE.match(s.name))
    return shards or [p]


class GGUFFile:
    """Tensor directory of one `.gguf` file: `name -> (dims, ggml_type, abs_offset)`."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, 'rb') as f:
            if f.read(4) != b'GGUF':
                raise ValueError(f'{self.path}: not a GGUF file')
            version = struct.unpack('<I', f.read(4))[0]
            if version < 2:
                raise ValueError(f'{self.path}: unsupported GGUF version {version} (need >= 2)')
            n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
            align = 32

            def read_str():
                length = struct.unpack('<Q', f.read(8))[0]
                return f.read(length).decode('utf-8', 'replace')

            def skip_val(vtype):
                if vtype == 8:
                    f.seek(struct.unpack('<Q', f.read(8))[0], 1)
                elif vtype == 9:
                    elem_type, count = struct.unpack('<IQ', f.read(12))
                    if elem_type in _KV_SIZES:
                        f.seek(count * _KV_SIZES[elem_type], 1)
                    else:
                        for _ in range(count):
                            skip_val(elem_type)
                elif vtype in _KV_SIZES:
                    f.seek(_KV_SIZES[vtype], 1)
                else:
                    raise ValueError(f'{self.path}: unknown GGUF value type {vtype}')

            for _ in range(n_kv):
                key = read_str()
                vtype = struct.unpack('<I', f.read(4))[0]
                if key == 'general.alignment' and vtype in _ALIGN_FMTS:
                    align = struct.unpack(_ALIGN_FMTS[vtype], f.read(_KV_SIZES[vtype]))[0]
                else:
                    skip_val(vtype)

            entries = []
            for _ in range(n_tensors):
                name = read_str()
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
                kind, offset = struct.unpack('<IQ', f.read(12))
                entries.append((name, dims, kind, offset))

            data_start = -(-f.tell() // align) * align
        self.tensors = {name: (dims, kind, data_start + offset) for name, dims, kind, offset in entries}


def mode_for(kind):
    """Mode tag for `iq_experts` ggml type `kind`, e.g. `'gguf:iq2_xxs'`."""
    if kind not in iq_experts.TYPES:
        raise ValueError(f'unsupported ggml expert type {kind}')
    return f'gguf:{iq_experts.TYPES[kind].lower()}'


def bits_for(kind):
    """Representative integer bit width for metadata reporting of ggml type `kind`."""
    pb = iq_experts.per_block(kind)
    return max(1, (8 * iq_experts.BLOCK[kind]) // pb)


def tensor_name(layer, projection):
    """Canonical GGUF stacked expert tensor name for `(layer, projection)`."""
    return f'blk.{layer}.{PROJECTIONS[projection]}.weight'


class ExpertGGUF:
    """Routed MoE experts read from GGUF file(s), implementing the `demand.Demand` checkpoint
    and `q2_experts.HostCopy` APIs.

    Only `weight` tensors exist (no `scales` or `biases`); raw blocks stay encoded in ggml layout
    (`uint8 [experts, out_dims, in_dims // per_block, block_bytes]`) and execute directly through
    `gguf_qmv` on Metal and CUDA or `iq_experts.decode` fallback.
    """

    def __init__(self, path):
        resolved = Path(path).expanduser().resolve()
        self.path = resolved
        self.files = parts(resolved)
        self.raw_tensors = {}
        self._by_key = {}                               # (layer, projection) -> entry dict

        per_expert = {}
        for f in self.files:
            gf = GGUFFile(f)
            for name, (dims, kind, offset) in gf.tensors.items():
                self.raw_tensors[name] = (f, dims, kind, offset)
                m = _STACKED_RE.match(name)
                if m and len(dims) == 3:
                    layer = int(m.group(1))
                    proj = _GGUF_TO_PROJ[m.group(2)]
                    cols, rows, n_exp = (int(d) for d in dims)
                    self._by_key[(layer, proj)] = {
                        'file': f, 'offset': offset, 'cols': cols, 'rows': rows,
                        'experts': n_exp, 'kind': kind, 'per_expert': None,
                    }
                    continue
                m2 = _PER_EXPERT_RE.match(name)
                if m2 and len(dims) == 2:
                    layer = int(m2.group(1))
                    proj = _PER_EXPERT_TO_PROJ[m2.group(2)]
                    exp_id = int(m2.group(3))
                    per_expert.setdefault((layer, proj), {})[exp_id] = (f, dims, kind, offset)

        for (layer, proj), experts_map in per_expert.items():
            if (layer, proj) in self._by_key:
                continue
            ids = sorted(experts_map)
            if ids != list(range(len(ids))):
                raise ValueError(f'{self.path}: layer {layer} {proj}: non-contiguous experts {ids[:4]}')
            metas = {(experts_map[e][1], experts_map[e][2]) for e in ids}
            if len(metas) != 1:
                raise ValueError(f'{self.path}: layer {layer} {proj}: mixed expert shapes/types')
            (cols, rows), kind = next(iter(metas))
            self._by_key[(layer, proj)] = {
                'file': None, 'offset': 0, 'cols': int(cols), 'rows': int(rows),
                'experts': len(ids), 'kind': kind,
                'per_expert': [(experts_map[e][0], experts_map[e][3]) for e in ids],
            }

        if not self._by_key:
            raise ValueError(f'{self.path}: no blk.<layer>.ffn_{{gate,up,down}}_exps.weight tensors found')

        self._pool, self._ahead = None, {}
        self.tensors = {}
        proj_order = {p: i for i, p in enumerate(PROJECTIONS)}
        for layer, proj in sorted(self._by_key, key=lambda lp: (lp[0], proj_order[lp[1]])):
            e = self._entry(layer, proj)
            kind = e['kind']
            pb, bb = iq_experts.per_block(kind), iq_experts.BLOCK[kind]
            shape = (e['experts'], e['rows'], e['cols'] // pb, bb)
            loc = e['offset'] if e['file'] is not None else e['per_expert']
            self.tensors[self.name(layer, proj, 'weight')] = (e['file'], 'U8', shape, loc)

    def _entry(self, layer, projection):
        key = (int(layer), projection)
        if key not in self._by_key:
            raise KeyError(f'{self.path}: missing expert tensor {tensor_name(layer, projection)}')
        entry = self._by_key[key]
        kind = entry['kind']
        if kind not in iq_experts.TYPES:
            raise ValueError(f'{tensor_name(layer, projection)}: unsupported ggml type {kind}')
        pb = iq_experts.per_block(kind)
        if entry['cols'] % pb:
            raise ValueError(f'{tensor_name(layer, projection)}: in_dims {entry["cols"]} not divisible by {pb}')
        return entry

    def name(self, layer, projection, part='weight'):
        return f'blk.{layer}.{PROJECTIONS[projection]}.{part}'

    def has(self, layer, projection, part):
        return part == 'weight' and (int(layer), projection) in self._by_key

    def mode(self, layer, projection):
        return mode_for(self._entry(layer, projection)['kind'])

    def quantization(self, layer, projection, input_dims=None):
        kind = self._entry(layer, projection)['kind']
        return iq_experts.per_block(kind), bits_for(kind)

    def nbytes(self, name):
        parts_name = name.split('.')
        if len(parts_name) == 4 and parts_name[0] == 'blk' and parts_name[3] == 'weight':
            proj = _GGUF_TO_PROJ.get(parts_name[2])
            if proj is not None and (int(parts_name[1]), proj) in self._by_key:
                e = self._entry(int(parts_name[1]), proj)
                kind = e['kind']
                return e['experts'] * e['rows'] * (e['cols'] // iq_experts.per_block(kind)) * iq_experts.BLOCK[kind]
        return 0

    def total_bytes(self, moe_indices):
        """Total bytes across `moe_indices` and `(gate_proj, up_proj, down_proj)` in this GGUF."""
        return sum(self.nbytes(self.name(l, p, 'weight')) for l in moe_indices for p in PROJECTIONS)

    def kinds(self, moe_indices):
        """Unique ggml type names used across `moe_indices` in order of appearance."""
        seen = []
        for l in moe_indices:
            for p in PROJECTIONS:
                tname = iq_experts.TYPES[self._entry(l, p)['kind']]
                if tname not in seen:
                    seen.append(tname)
        return seen

    def format_label(self, moe_indices):
        """Human-readable label listing the ggml expert format(s) stored in this GGUF."""
        seen = self.kinds(moe_indices)
        if len(seen) == 1:
            return f'gguf {seen[0]}'
        return f'gguf {"/".join(seen[:4])}{"..." if len(seen) > 4 else ""}'

    def validate(self, layers):
        """Verify that every MoE layer in `layers` matches expert count and dims in this GGUF."""
        moe = [(i, arch.switch_of(layer)) for i, layer in enumerate(layers) if arch.switch_of(layer) is not None]
        if not moe:
            raise ValueError('model has no MoE SwitchGLU layers')
        for layer_idx, sw in moe:
            for p in PROJECTIONS:
                e = self._entry(layer_idx, p)
                proj = getattr(sw, p)
                out_d, in_d = demand._proj_dims(proj)
                n_exp = proj.weight.shape[0]
                if (e['experts'], e['rows'], e['cols']) != (n_exp, out_d, in_d):
                    raise ValueError(
                        f'GGUF mismatch at layer {layer_idx} {p}: GGUF has {(e["experts"], e["rows"], e["cols"])}, '
                        f'model expects {(n_exp, out_d, in_d)}'
                    )

    @staticmethod
    def _key(name, rows):
        return name, None if rows is None else tuple(np.asarray(rows, np.int64).reshape(-1).tolist())

    def routed(self, layers=False):
        """Routed GGUF expert tensor names ordered layer by layer (`(layer, name)` with `layers`)."""
        proj_order = {p: i for i, p in enumerate(PROJECTIONS)}
        found = [
            (layer, self.name(layer, proj, 'weight'))
            for layer, proj in sorted(self._by_key, key=lambda lp: (lp[0], proj_order[lp[1]]))
        ]
        return found if layers else [name for _, name in found]

    def _from_file(self, name, rows=None, cache=False, out=None):
        file, _, shape, offset = self.tensors[name]
        if file is not None:
            return q2_experts.read_rows(file, offset, shape, np.uint8, rows, cache=cache, out=out)
        rows_arr = np.arange(shape[0], dtype=np.int64) if rows is None else np.asarray(rows, np.int64).reshape(-1)
        if out is None:
            out = np.empty((len(rows_arr),) + tuple(shape[1:]), np.uint8)
        for i, exp_id in enumerate(rows_arr):
            part_file, part_off = offset[int(exp_id)]
            q2_experts.read_rows(part_file, part_off, shape[1:], np.uint8, cache=cache, out=out[i])
        return out

    def _read(self, name, rows=None, cache=False, out=None):
        held = self.held()
        got = None if held is None else held.rows(name, rows, out)
        return got if got is not None else self._from_file(name, rows, cache=cache, out=out)

    def _read_numpy(self, layer, projection, rows, cache=False, out=None):
        self._entry(layer, projection)
        return self._read(self.name(layer, projection, 'weight'), rows, cache=cache, out=out)

    def buffers(self, layer, projection):
        e = self._entry(layer, projection)
        kind = e['kind']
        pb, bb = iq_experts.per_block(kind), iq_experts.BLOCK[kind]
        buf = mx.zeros((1, e['rows'], e['cols'] // pb, bb), mx.uint8, stream=mx.cpu)
        mx.eval(buf)
        return [buf, None, None]

    def experts(self, layer, projection, experts, cache=False):
        self._entry(layer, projection)
        t_name = self.name(layer, projection, 'weight')
        pending = self._ahead.pop(self._key(t_name, experts), None)
        raw = pending.result() if pending is not None else self._read(t_name, experts, cache=cache)
        return [mx.array(raw), None, None]

    def expert(self, layer, projection, expert_id, out, cache=True):
        buf = out[0]
        self._read_numpy(layer, projection, [int(expert_id)], cache=cache, out=np.asarray(buf))
        mx.eval(buf)
        return [buf, None, None]

    def ahead(self, items):
        """Start reading `[(tensor name, rows or None for all)]` on a background thread."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix='gguf-ahead')
        for name, rows in items:
            key = self._key(name, rows)
            if key not in self._ahead:
                self._ahead[key] = self._pool.submit(self._read, name, rows)

    def forget(self):
        """Let go of what was read ahead and not asked for."""
        for pending in self._ahead.values():
            pending.cancel()
        self._ahead.clear()

    def close(self):
        """Cancel pending read-ahead tasks and shut down the read-ahead worker pool."""
        self.forget()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def held(self):
        """The host's copy of this GGUF source's experts (`q2_experts.hold()`), or `None`."""
        return q2_experts._HELD.get(self.path)


def load_model(layers, source, on_layer=None):
    """Replace every MoE layer's `SwitchGLU` projections in `layers` with raw GGUF blocks loaded
    from `source` (`ExpertGGUF` or path) whole in memory and install `expert_qmv` dispatch."""
    src = source if isinstance(source, ExpertGGUF) else ExpertGGUF(source)
    src.validate(layers)
    moe = [(i, arch.switch_of(layer)) for i, layer in enumerate(layers) if arch.switch_of(layer) is not None]
    total = len(moe)
    for done, (layer_idx, sw) in enumerate(moe, 1):
        for p in PROJECTIONS:
            w = demand._to_device(src.experts(layer_idx, p, None, cache=False)[0])
            kind = src._entry(layer_idx, p)['kind']
            pb = iq_experts.per_block(kind)
            proj = demand.Projection(w, None, None, pb, bits_for(kind), mode=mode_for(kind))
            mx.eval(proj.parameters())
            setattr(sw, p, proj)
        expert_qmv.install_switch(sw)
        if done % 8 == 0:
            gc.collect()
            mx.clear_cache()
        if on_layer is not None:
            on_layer(done, total)
    gc.collect()
    mx.clear_cache()
    return src
