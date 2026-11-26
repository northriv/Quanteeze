# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Random-access reading of routed experts from an MLX safetensors checkpoint, past the OS file
cache (`F_NOCACHE` on macOS, `POSIX_FADV_DONTNEED` on Linux) or into a host-RAM copy
(`HostCopy`) on multi-memory systems.
"""
import fcntl
import json
import mmap
import os
import re
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import per_load

PROJECTIONS = {'gate_proj': 'ffn_gate_exps', 'up_proj': 'ffn_up_exps', 'down_proj': 'ffn_down_exps'}
NAMES = ('weight', 'scales', 'biases')
_SAFETENSORS = {'U8': np.uint8, 'U32': np.uint32, 'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32,
                'I8': np.uint8, 'F8_E8M0': np.uint8, 'F8_E4M3': np.uint8}
_PACKED = ('I8',)                     # DeepSeek's own FP4 experts: two values a byte, read as MLX's uint32 words


def read_rows(file, offset, shape, dtype, rows=None, cache=False, out=None):
    """Rows of a tensor stored at `offset` of `file` -- along its first axis, all of them by
    default -- read past the file cache (F_NOCACHE; on Linux, the pages read dropped after each
    run of rows with POSIX_FADV_DONTNEED).  A model's experts stream through once a pass when they
    are read layer by layer; kept as cached pages they would only push the rest of memory into
    swap.  Consecutive rows are read in one go.
    cache=True reads through the file cache; `out`, an array of the rows' shape and dtype, is read
    into (the memory of an MLX array, say)."""
    shape = tuple(shape)
    rows = np.arange(shape[0]) if rows is None else np.asarray(rows, np.int64).reshape(-1)
    size = int(np.prod(shape[1:], dtype=np.int64)) * np.dtype(dtype).itemsize
    if out is None:
        out = np.empty((len(rows),) + shape[1:], dtype)
    elif out.shape != (len(rows),) + shape[1:] or out.dtype != np.dtype(dtype) or not out.flags.c_contiguous:
        raise ValueError(f'{file}: rows {(len(rows),) + shape[1:]} {np.dtype(dtype)} into {out.shape} {out.dtype}')
    raw = memoryview(out.reshape(-1).view(np.uint8))
    fd = os.open(file, os.O_RDONLY)
    try:
        nocache = not cache and hasattr(fcntl, 'F_NOCACHE')         # macOS
        if nocache:
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        drop = not cache and not nocache and hasattr(os, 'posix_fadvise')   # Linux
        i = 0
        while i < len(rows):
            j = i + 1
            while j < len(rows) and rows[j] == rows[j - 1] + 1:
                j += 1
            start, length, done = offset + int(rows[i]) * size, (j - i) * size, 0
            while done < length:
                n = os.preadv(fd, [raw[i * size + done:i * size + min(length, done + (1 << 30))]], start + done)
                if n <= 0:
                    raise OSError(f'{file}: {length - done} bytes short at {start + done}')
                done += n
            if drop:                                       # whole pages: the kernel keeps any it was given in part
                first, end = start // mmap.PAGESIZE * mmap.PAGESIZE, -(-(start + length) // mmap.PAGESIZE) * mmap.PAGESIZE
                os.posix_fadvise(fd, first, end - first, os.POSIX_FADV_DONTNEED)
            i = j
    finally:
        os.close(fd)
    return out


def _as_mlx(array, dtype):
    """An array read as stored, as MLX takes it: bf16 bits as bf16, DeepSeek's FP4 bytes as words."""
    if dtype == 'BF16':
        return array.view(mx.bfloat16)
    return array.view(mx.uint32) if dtype in _PACKED else array


_MLX = {np.dtype(k): v for k, v in ((np.uint8, mx.uint8), (np.uint16, mx.uint16), (np.uint32, mx.uint32),
                                    (np.float16, mx.float16), (np.float32, mx.float32))}

EXPERT_RE = re.compile(r'^(.*\blayers)\.(\d+)\.(.+)\.(gate_proj|up_proj|down_proj)\.weight$')
# DeepSeek's own layout: one tensor an expert, w1 the gate, w3 the up and w2 the down projection
PER_EXPERT_RE = re.compile(r'^(.*\blayers)\.(\d+)\.(.+)\.experts\.(\d+)\.(w1|w2|w3|gate_proj|up_proj|down_proj)'
                           r'\.(weight|scales?|biases)$')
PER_EXPERT_PROJECTIONS = {'w1': 'gate_proj', 'w3': 'up_proj', 'w2': 'down_proj',                  # DeepSeek's own
                          'gate_proj': 'gate_proj', 'up_proj': 'up_proj', 'down_proj': 'down_proj'}  # GLM-5.3's MLX build


class Checkpoint:
    """An MLX checkpoint's safetensors, read in place: name -> (file, dtype, shape, offset).  Its
    routed experts are found by name: the weights over stacked experts (three dimensions) under
    '<...>layers.<n>.<path>.<projection>', whatever the path ('mlp.switch_mlp' in Qwen's, with
    'language_model.model.' before it in a VLM's).  In DeepSeek's own layout (TensorFold's
    bit-exact MXFP4 DeepSeek V4: 'layers.<n>.ffn.experts.<e>.w1.weight', one tensor an expert)
    each layer's experts read as if stacked, under '<...>layers.<n>.<path>.switch_mlp.<projection>':
    those names hold (None, dtype, shape, [(file, offset) an expert])."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.tensors = {}
        self._pool, self._ahead = None, {}
        for file in sorted(self.path.glob('*.safetensors')):
            with open(file, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(n))
            for name, entry in header.items():
                if name != '__metadata__':
                    self.tensors[name] = (file, entry['dtype'], tuple(entry['shape']), 8 + n + entry['data_offsets'][0])
        found = {}
        for name, (_, _, shape, _) in self.tensors.items():
            m = EXPERT_RE.match(name)
            if m and len(shape) == 3:
                found[(m.group(1), m.group(3))] = found.get((m.group(1), m.group(3)), 0) + 1
        self.template = None
        if found:
            head, path = max(found, key=found.get)
            self.template = head + '.{layer}.' + path + '.{projection}.{name}'
        else:
            self._stack_per_expert()

    def _stack_per_expert(self):
        """Name each layer's per-expert tensors as one stacked tensor (DeepSeek's own layout)."""
        groups = {}
        for name, (file, dtype, shape, offset) in self.tensors.items():
            m = PER_EXPERT_RE.match(name)
            if m and file is not None:
                part = 'scales' if m.group(6) == 'scale' else m.group(6)              # DeepSeek's own name
                key = (m.group(1), int(m.group(2)), m.group(3), PER_EXPERT_PROJECTIONS[m.group(5)], part)
                groups.setdefault(key, {})[int(m.group(4))] = (file, dtype, shape, offset)
        if not groups:
            return
        heads = {(head, path) for head, _, path, _, _ in groups}
        head, path = max(heads, key=lambda hp: sum(1 for k in groups if (k[0], k[2]) == hp))
        self.template = head + '.{layer}.' + path + '.switch_mlp.{projection}.{name}'
        for (h, layer, pth, projection, part), experts in groups.items():
            if (h, pth) != (head, path):
                continue
            ids = sorted(experts)
            if ids != list(range(len(ids))) or len({(experts[e][1], experts[e][2]) for e in ids}) != 1:
                raise ValueError(f'{self.path}: layer {layer} {projection}.{part}: experts {ids[:3]}... do not stack')
            dtype, shape = experts[0][1], experts[0][2]
            self.tensors[self.template.format(layer=layer, projection=projection, name=part)] = (
                None, dtype, (len(ids),) + tuple(shape), [(experts[e][0], experts[e][3]) for e in ids])

    def name(self, layer, projection, part='weight'):
        """The tensor name of one layer's routed experts' `part` ('weight', 'scales', 'biases')."""
        if self.template is None:
            raise ValueError(f'{self.path}: no routed experts found')
        return self.template.format(layer=layer, projection=projection, name=part)

    def has(self, layer, projection, part):
        return self.template is not None and self.name(layer, projection, part) in self.tensors

    def mode(self, layer, projection):
        """The quantization mode of one layer's expert projection: MLX's affine one when it has
        biases; None when unquantized (no scales); otherwise the microscaling format whose packing
        fits its arrays (MXFP4: 4 bits in groups of 32; NVFP4: 4 bits in groups of 16; MXFP8:
        8 bits in groups of 32)."""
        if not self.has(layer, projection, 'scales'):
            return None
        if self.has(layer, projection, 'biases'):
            return 'affine'
        words = self._words(self.name(layer, projection))                  # uint32 words a row
        groups = self.tensors[self.name(layer, projection, 'scales')][2][-1]
        for name, bits, group_size in (('mxfp4', 4, 32), ('nvfp4', 4, 16), ('mxfp8', 8, 32)):
            if words * 32 // bits == groups * group_size:
                return name
        raise ValueError(f'layer {layer} {projection}: no biases, and no known format packs '
                         f'{words} words a row with {groups} scales')

    def quantization(self, layer, projection, input_dims):
        """(group size, bits) of one layer's expert projection, from its arrays' shapes."""
        scales = self.tensors[self.name(layer, projection, 'scales')][2]
        return input_dims // scales[-1], self._words(self.name(layer, projection)) * 32 // input_dims

    def _words(self, name):
        """The uint32 words of a row of the tensor `name`, as MLX holds it."""
        _, dtype, shape, _ = self.tensors[name]
        return shape[-1] * np.dtype(_SAFETENSORS[dtype]).itemsize // 4

    @staticmethod
    def _key(name, rows):
        return name, None if rows is None else tuple(np.asarray(rows, np.int64).reshape(-1).tolist())

    def routed(self, layers=False):
        """The names of the routed experts' tensors, layer by layer ((layer, name) with `layers`)."""
        if self.template is None:
            return []
        pattern = re.compile(re.escape(self.template).replace(r'\{layer\}', r'(\d+)')
                             .replace(r'\{projection\}', '(?:gate_proj|up_proj|down_proj)')
                             .replace(r'\{name\}', '(?:' + '|'.join(NAMES) + ')') + '$')
        found = sorted((int(m.group(1)), name) for name in self.tensors if (m := pattern.match(name)))
        return found if layers else [name for _, name in found]

    def nbytes(self, name):
        _, dtype, shape, _ = self.tensors[name]
        return int(np.prod(shape, dtype=np.int64)) * np.dtype(_SAFETENSORS[dtype]).itemsize

    def _read(self, name, rows=None, cache=False, out=None):
        """A tensor's rows as numpy, into `out` when given: from the host's copy where it holds
        them (hold()), else from the file (_from_file)."""
        held = _HELD.get(self.path)
        got = None if held is None else held.rows(name, rows, out)
        return got if got is not None else self._from_file(name, rows, cache, out)

    def _from_file(self, name, rows=None, cache=False, out=None):
        """A tensor's rows read from its file (read_rows), into `out` when given; a stacked name of
        per-expert tensors, expert by expert."""
        file, dtype, shape, offset = self.tensors[name]
        if file is not None:
            return read_rows(file, offset, shape, _SAFETENSORS[dtype], rows, cache, out)
        rows = np.arange(shape[0]) if rows is None else np.asarray(rows, np.int64).reshape(-1)
        if out is None:
            out = np.empty((len(rows),) + tuple(shape[1:]), _SAFETENSORS[dtype])
        for i, e in enumerate(rows):
            part_file, part_offset = offset[int(e)]
            read_rows(part_file, part_offset, shape[1:], _SAFETENSORS[dtype], cache=cache, out=out[i])
        return out

    def ahead(self, items):
        """Start reading [(tensor name, rows or None for all)] on a thread of the reader's own, so
        they are there when asked for: the next layer's experts while this one runs."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix='read-ahead')
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
        """The host's copy of this checkpoint's experts (hold()), or None."""
        return _HELD.get(self.path)

    def _array(self, name, rows=None):
        dtype = self.tensors[name][1]
        pending = self._ahead.pop(self._key(name, rows), None)
        out = mx.array(pending.result() if pending is not None else self._read(name, rows))
        return _as_mlx(out, dtype)

    def arrays(self, names):
        """Whole tensors by name, read past the file cache (read_rows)."""
        return [self._array(name) for name in names]

    def experts(self, layer, projection, experts):
        """Some experts' quantized arrays of one layer and projection: [weight, scales, biases],
        biases None in a format without them (mxfp4), read past the file cache (read_rows)."""
        from . import parallel
        parallel.require(layer, 'expert read')
        names = [self.name(layer, projection, name) if self.has(layer, projection, name) else None for name in NAMES]
        return [None if name is None else self._array(name, experts) for name in names]

    def buffers(self, layer, projection):
        """Room for one expert's arrays of a layer's projection: [weight, scales, biases] as MLX
        arrays [1, ...] of their stored dtype (bf16 as uint16; None for a part it lacks), made on the
        CPU stream so that making them never waits for the GPU."""
        out = []
        for name in NAMES:
            key = self.name(layer, projection, name)
            if key not in self.tensors:
                out.append(None)
                continue
            dtype, shape = self.tensors[key][1:3]
            out.append(mx.zeros((1,) + tuple(shape[1:]), _MLX[np.dtype(_SAFETENSORS[dtype])], stream=mx.cpu))
        mx.eval([a for a in out if a is not None])
        return out

    def expert(self, layer, projection, expert, out, cache=True):
        """One expert's arrays of a layer's projection read into `out` (buffers())."""
        from . import parallel
        parallel.require(layer, 'expert read')
        keys = [None if array is None else self.name(layer, projection, name) for name, array in zip(NAMES, out)]
        got = []
        for key, array in zip(keys, out):
            if array is None:
                got.append(None)
                continue
            self._read(key, [expert], cache=cache, out=np.asarray(array))
            got.append(_as_mlx(array, self.tensors[key][1]))
        mx.eval([a for a in got if a is not None])
        return got

    def segments(self, layer, projection, expert, out):
        """Return `[(file, byte_offset, nbytes, ptr, u8_view, key)]` for direct C++ `pread` into `out`,
        or `None` if already ready in `HostCopy` or non-contiguous."""
        from . import parallel
        parallel.require(layer, 'expert read')
        held = self.held()
        segs = []
        for name, array in zip(NAMES, out):
            if array is None:
                continue
            key = self.name(layer, projection, name)
            if held is not None and held.holds_expert(layer, key, expert):
                return None
            entry = self.tensors.get(key)
            if entry is None:
                return None
            file, dtype, shape, offset = entry
            row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * np.dtype(_SAFETENSORS[dtype]).itemsize
            arr_np = np.asarray(array)
            if not arr_np.flags.c_contiguous or not arr_np.flags.writeable or arr_np.nbytes != row_bytes:
                return None
            if file is not None:
                target_file, byte_off = file, int(offset) + int(expert) * row_bytes
            else:
                target_file, byte_off = offset[int(expert)][0], int(offset[int(expert)][1])
            u8 = arr_np.view(np.uint8).reshape(-1)
            segs.append((str(target_file), byte_off, row_bytes, int(u8.ctypes.data), u8, key))
        return segs

    def made(self, layer, projection, out):
        """Wrap buffers `out` (populated by `segments()` read) as MLX arrays of each tensor's dtype."""
        return [
            None if array is None else _as_mlx(array, self.tensors[self.name(layer, projection, name)][1])
            for name, array in zip(NAMES, out)
        ]


_HELD = {}                       # a checkpoint's path -> its HostCopy (hold())


class HostCopy:
    """A checkpoint's routed experts held read-only in the host's memory.

    With `by_rows=False`, as many whole tensors as `budget` bytes hold, layer by layer from the first.
    With `by_rows=True`, an equal share of expert rows across every layer (`expert_in`, `slot_of`).
    """

    def __init__(self, checkpoint, budget, workers, layers=None, by_rows=False):
        self.path, self.arrays, self.pending, self.bytes = checkpoint.path, {}, {}, 0
        self.lock = threading.RLock()
        self.by_rows = bool(by_rows)
        self.layer_of, self.expert_in, self.slot_of, self.experts = {}, {}, {}, {}
        self._layer_names = {}
        self._reading_done = False
        routed = [(layer, n) for layer, n in checkpoint.routed(layers=True) if layers is None or layer in layers]
        names = [n for _, n in routed]
        self.total = sum(checkpoint.nbytes(name) for name in names)
        self._workers = max(1, workers)
        self._pool = ThreadPoolExecutor(self._workers, thread_name_prefix='host-copy')
        if not self.by_rows:
            for name in names:
                size = checkpoint.nbytes(name)
                if self.bytes + size > budget:
                    break
                _, dtype, shape, _ = checkpoint.tensors[name]
                self.arrays[name] = np.empty(shape, _SAFETENSORS[dtype])
                self.pending[name] = self._pool.submit(checkpoint._from_file, name, None, False, self.arrays[name])
                self.bytes += size
            self._reading_done = not self.pending
            return
        by_layer = {}
        for layer, name in routed:
            by_layer.setdefault(layer, []).append(name)
        per = {}
        count = {}
        for layer, layer_names in by_layer.items():
            count[layer] = checkpoint.tensors[layer_names[0]][2][0]
            per[layer] = sum(checkpoint.nbytes(n) for n in layer_names) / count[layer]
        chosen = {layer: [] for layer in by_layer}
        max_experts = max(count.values(), default=0)
        for e in range(max_experts):
            for layer in by_layer:
                if e < count[layer] and self.bytes + per[layer] <= budget:
                    chosen[layer].append(e)
                    self.bytes += per[layer]
        for layer, layer_names in by_layer.items():
            ids = np.sort(np.asarray(chosen[layer], np.int64))
            self.experts[layer] = count[layer]
            self.expert_in[layer] = ids.copy()
            self.slot_of[layer] = np.full(count[layer], -1, np.int64)
            self.slot_of[layer][ids] = np.arange(ids.size)
            if not ids.size:
                continue
            self._layer_names[layer] = tuple(layer_names)
            for name in layer_names:
                _, dtype, shape, _ = checkpoint.tensors[name]
                self.layer_of[name] = layer
                self.arrays[name] = np.empty((ids.size,) + tuple(shape[1:]), _SAFETENSORS[dtype])
                self.pending[name] = self._pool.submit(checkpoint._from_file, name, ids, False, self.arrays[name])
        self.bytes = int(sum(a.nbytes for a in self.arrays.values()))
        self._reading_done = not self.pending

    def _ready(self, name):
        pending = self.pending.get(name)
        return pending is not None and pending.done() and not pending.cancelled() and pending.exception() is None

    def _slots(self, name, rows):
        """Where a held tensor keeps `rows` (all for None): their slots, or None where one is not held."""
        held = self.arrays[name]
        layer = self.layer_of.get(name)
        if layer is None:
            if rows is None:
                return np.arange(held.shape[0])
            rows = np.asarray(rows, np.int64).reshape(-1)
            if rows.size and (rows.min() < 0 or rows.max() >= held.shape[0]):
                raise IndexError(f'{self.path}: {name}: rows {rows.min()}..{rows.max()} of {held.shape[0]}')
            return rows
        slot_of = self.slot_of[layer]
        rows = np.arange(slot_of.size) if rows is None else np.asarray(rows, np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= slot_of.size):
            raise IndexError(f'{self.path}: {name}: rows {rows.min()}..{rows.max()} of {slot_of.size}')
        slots = slot_of[rows]
        return None if (slots < 0).any() else slots

    def rows(self, name, rows=None, out=None):
        """A held tensor's `rows` (all of them for None), into `out` when given; None where it is
        not held (or one of the rows is not), or not read yet."""
        with self.lock:
            if not self._ready(name):
                return None
            slots = self._slots(name, rows)
            if slots is None:
                return None
            held = self.arrays[name]
            layer = self.layer_of.get(name)
            if rows is None and (layer is None or held.shape[0] == self.slot_of[layer].size):
                if out is None:
                    return held.copy()
                out[:held.shape[0]] = held
                return out
            return np.take(held, slots, axis=0, out=out, mode='clip')

    def has(self, layer, expert):
        with self.lock:
            slot_of = self.slot_of.get(layer)
            return bool(self.arrays) and slot_of is not None and slot_of[expert] >= 0

    def held_in(self, layer, experts):
        """Of `experts`, those this copy holds and has finished reading in `layer` (by row)."""
        experts = np.asarray(experts, np.int64).reshape(-1)
        with self.lock:
            slot_of = self.slot_of.get(layer)
            names = self._layer_names.get(layer, ())
            if slot_of is None or not names or not all(self._ready(n) for n in names):
                return experts[:0]
            return experts[slot_of[experts] >= 0]

    def holds_expert(self, layer, tensor_name, expert):
        """Whether `(layer, tensor_name, expert)` is already resident and ready in this `HostCopy`."""
        with self.lock:
            if self.by_rows:
                slot_of = self.slot_of.get(layer)
                names = self._layer_names.get(layer, ())
                exp_id = int(expert)
                return bool(
                    slot_of is not None
                    and names
                    and 0 <= exp_id < slot_of.size
                    and slot_of[exp_id] >= 0
                    and all(self._ready(n) for n in names)
                )
            return self._ready(tensor_name)

    def holds_layer(self, layer):
        """Whether this `by_rows` copy holds every expert of `layer` and has finished reading it."""
        with self.lock:
            slot_of = self.slot_of.get(layer)
            names = self._layer_names.get(layer, ())
            if slot_of is None or not names:
                return False
            return (
                self.arrays[names[0]].shape[0] == slot_of.size
                and all(self._ready(n) for n in names)
            )

    def copy_layer(self, layer, experts, targets):
        """Copy `experts` (all of `layer` when None) into `[(tensor_name, out_view), ...]` under one
        lock acquisition; returns True when every target was copied or False if any is missing."""
        if not targets:
            return True
        with self.lock:
            slot_of = self.slot_of.get(layer)
            if slot_of is None or not all(self._ready(name) for name, _ in targets):
                return False
            if experts is None:
                slots = slot_of
            else:
                exp_arr = np.asarray(experts, np.int64).reshape(-1)
                if exp_arr.size and (exp_arr.min() < 0 or exp_arr.max() >= slot_of.size):
                    raise IndexError(
                        f'{self.path}: layer {layer}: rows {exp_arr.min()}..{exp_arr.max()} of {slot_of.size}'
                    )
                slots = slot_of[exp_arr] if exp_arr.size else exp_arr
            if (slots < 0).any():
                return False
            m = int(slots.size)
            if m == 0:
                return True
            first_held = self.arrays[targets[0][0]]
            is_full = (
                m == first_held.shape[0]
                and (experts is None or np.array_equal(slots, np.arange(m, dtype=np.int64)))
            )
            work = [(self.arrays[name], out[:m]) for name, out in targets]

            def _copy_one(held, out_slice):
                if is_full:
                    out_slice[:] = held
                else:
                    np.take(held, slots, axis=0, out=out_slice, mode='clip')

            big = [item for item in work if item[1].nbytes >= (1 << 19)]
            if self._workers > 1 and len(big) > 1 and not self.reading():
                small = [item for item in work if item[1].nbytes < (1 << 19)]
                futs = [self._pool.submit(_copy_one, held, out_slice) for held, out_slice in big[:-1]]
                _copy_one(*big[-1])
                for held, out_slice in small:
                    _copy_one(held, out_slice)
                for fut in futs:
                    fut.result()
            else:
                for held, out_slice in work:
                    _copy_one(held, out_slice)
            return True

    def reading(self):
        """Whether any background read into `self.arrays` is still in flight."""
        with self.lock:
            if self._reading_done:
                return False
            busy = any(not p.done() for p in list(self.pending.values()))
            if not busy:
                self._reading_done = True
            return busy

    def read(self):
        """Bytes held and read so far."""
        with self.lock:
            return sum(
                self.arrays[n].nbytes
                for n, p in self.pending.items()
                if n in self.arrays and p.done() and not p.cancelled() and p.exception() is None
            )

    def wait(self):
        """Wait for every held tensor to be read; raises what a read raised."""
        for pending in list(self.pending.values()):
            if not pending.cancelled():
                pending.result()
        self._reading_done = True

    def close(self):
        with self.lock:
            self.arrays, self.pending = {}, {}
            self._reading_done = True
        self._pool.shutdown(wait=True, cancel_futures=True)


def _is_source(obj):
    return hasattr(obj, 'path') and hasattr(obj, 'tensors') and hasattr(obj, 'routed') and hasattr(obj, '_from_file')


def hold(path, budget, workers, layers=None, by_rows=False):
    """Hold `path`'s (or a checkpoint / GGUF expert source's) routed experts in host memory
    (`HostCopy`), as many as `budget` bytes take: one copy per source path."""
    ckpt = path if _is_source(path) else None
    resolved = ckpt.path if ckpt is not None else Path(path).expanduser().resolve()
    if resolved not in _HELD:
        if ckpt is None:
            if resolved.suffix.lower() == '.gguf' or (
                resolved.is_dir() and not any(resolved.glob('*.safetensors')) and any(resolved.glob('*.gguf'))
            ):
                from . import gguf_experts
                ckpt = gguf_experts.ExpertGGUF(resolved)
            else:
                ckpt = Checkpoint(resolved)
        _HELD[resolved] = HostCopy(ckpt, budget, workers, layers, by_rows)
    return _HELD[resolved]


def reading():
    """Whether any checkpoint's `HostCopy` is still reading into host memory."""
    return any(held.reading() for held in list(_HELD.values()))


@per_load.on_let_go
def let_go(path=None):
    """Let go of the host's copy of checkpoint `path`'s experts (all of them for None)."""
    target = path.path if _is_source(path) else path
    for key in list(_HELD) if target is None else [Path(target).expanduser().resolve()]:
        held = _HELD.pop(key, None)
        if held is not None:
            held.close()


def _mounts(mounts='/proc/mounts'):
    """Return `[(mount_point, fs_type)]` from Linux `/proc/mounts` or macOS `/sbin/mount`."""
    try:
        lines = Path(mounts).read_text().splitlines()
        return [(f[1].replace('\\040', ' '), f[2]) for f in (line.split() for line in lines) if len(f) >= 3]
    except OSError:
        pass
    if sys.platform != 'darwin' or mounts != '/proc/mounts':
        return []
    try:
        text = subprocess.run(['/sbin/mount'], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in text.splitlines():
        head, sep, tail = line.rpartition(' (')
        point = head.partition(' on ')[2]
        if sep and point:
            kind = tail.rstrip(')').split(',', 1)[0].strip()
            if kind:
                out.append((point, kind))
    return out


def file_system(path, mounts='/proc/mounts'):
    """The type of the file system `path` lies on (`mounts`; '' where it cannot be read)."""
    path, best = str(Path(path).expanduser().resolve()), ('', '')
    for point, kind in _mounts(mounts):
        if (path == point or path.startswith(point.rstrip('/') + '/')) and len(point) > len(best[0]):
            best = (point, kind)
    return best[1]


def in_memory(path, mounts='/proc/mounts'):
    """Whether `path` lies on a file system in memory (tmpfs: /dev/shm, say)."""
    return file_system(path, mounts) in ('tmpfs', 'ramfs')


NETWORK = (
    'lustre', 'nfs', 'nfs4', 'cifs', 'smb3', 'smbfs', 'afpfs', 'webdav',
    'beegfs', 'gpfs', 'ceph', 'fuse.sshfs', '9p',
)
LOCAL_READ_THREADS = 4
NETWORK_READ_THREADS = 16


def on_network(path, mounts='/proc/mounts'):
    """Whether `path` lies on a network's file system (Lustre, NFS, ...)."""
    return file_system(path, mounts) in NETWORK


def read_threads(path=None, mounts='/proc/mounts'):
    """Reader thread count from `QUANTEEZE_READ_THREADS` (`auto`: 16 on network FS, else 4)."""
    from . import variables
    raw = (variables.get('QUANTEEZE_READ_THREADS', variables.at_load()) or 'auto').strip().lower()
    if raw == 'auto':
        return NETWORK_READ_THREADS if (path and on_network(path, mounts)) else LOCAL_READ_THREADS
    return max(1, int(raw))
