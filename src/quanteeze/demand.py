# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Serving a model from the SSD when its routed experts do not fit in memory: each MoE layer holds
a bank of `slots` experts at uniform precision (either the checkpoint's own or uniform least-squares
`QUANTEEZE_BITS` via `requant.Flat`), reads missing experts on demand when routed to, and evicts
the least-used slots with exponential decay.
"""
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time
from types import SimpleNamespace
import weakref

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.linear import DECODE_BLOCK_SIZE

from . import arch, cuda_kernels, expert_qmv, prefill, q2_experts, requant
from .q2_experts import NAMES, PROJECTIONS

DECAY = 0.999
_LIVE = weakref.WeakSet()
_NEED_PREFILL = object()


class Projection(nn.Module):
    """One projection (`gate_proj`, `up_proj`, or `down_proj`) of an on-demand expert bank."""

    def __init__(self, weight, scales, biases, group_size, bits, mode='affine'):
        super().__init__()
        self.weight = weight
        if scales is not None:
            self.scales = scales
        if biases is not None:
            self.biases = biases
        self.group_size, self.bits, self.mode = group_size, bits, mode
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        if not requant.native(self.bits, self.mode):
            def unpacked():
                arrays = [self.weight, self.get('scales'), self.get('biases')]
                idx = indices
                if idx.size < self.weight.shape[0]:
                    flat = idx.flatten()
                    arrays = [a[flat] if a is not None else None for a in arrays]
                    idx = mx.arange(idx.size, dtype=idx.dtype).reshape(idx.shape)
                w = requant.dequantize(*arrays, self.group_size, self.bits, mode=self.mode).astype(x.dtype)
                return mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=idx,
                                    sorted_indices=sorted_indices and idx is indices)

            prefer_rows = indices.ndim == 1 and indices.size >= 16
            if isinstance(self.mode, str) and self.mode.startswith('gguf:'):
                from . import gguf_qmv
                if prefer_rows and gguf_qmv.takes_rows(self, x, indices):
                    return gguf_qmv.gather_rows(self, x, indices, unpacked)
                if gguf_qmv.takes(self, x, indices):
                    return gguf_qmv.gather(self, x, indices, unpacked)
                if gguf_qmv.takes_rows(self, x, indices):
                    return gguf_qmv.gather_rows(self, x, indices, unpacked)
            from . import kquant_qmv
            if self.mode in kquant_qmv.MODES:
                if prefer_rows and kquant_qmv.takes_rows(self, x, indices):
                    return kquant_qmv.gather_rows(self, x, indices, unpacked)
                if kquant_qmv.takes(self, x, indices):
                    return kquant_qmv.gather(self, x, indices, unpacked)
                if kquant_qmv.takes_rows(self, x, indices):
                    return kquant_qmv.gather_rows(self, x, indices, unpacked)
            return unpacked()
        return mx.gather_qmm(cuda_kernels.qmm_room(x), self.weight, self.scales, self.get('biases'),
                             rhs_indices=indices, transpose=True, group_size=self.group_size,
                             bits=self.bits, mode=self.mode, sorted_indices=sorted_indices)


_STAGE_RING = 8
_PREFILL_PAGE = 65536
_PREFILL_WORDS = _PREFILL_PAGE // 4


def _to_device(arr):
    """Ensure `arr` resides in pure GPU device memory (`device = 0` on CUDA) rather than
    `cudaMallocManaged` (`mx.array(numpy)`), avoiding UVM residency churn on resident banks."""
    if arr is None:
        return None
    if mx.metal.is_available() or mx.default_device() != mx.gpu:
        mx.eval(arr)
        return arr
    if arr.ndim == 0 or arr.shape[0] == 0:
        dst = mx.zeros(arr.shape, dtype=arr.dtype)
        mx.eval(dst)
        return dst
    if arr.dtype == mx.uint8 and (arr.size // arr.shape[0]) % 4 == 0:
        words = (arr.size // arr.shape[0]) // 4
        dst = mx.zeros((arr.shape[0], words), dtype=mx.uint32)
        mx.eval(dst)
        idx = mx.arange(arr.shape[0], dtype=mx.int32)
        dst[idx] = arr.reshape(arr.shape[0], -1).view(mx.uint32)
        out = dst.view(mx.uint8).reshape(arr.shape)
        del dst
        mx.eval(out)
        return out
    dst = mx.zeros(arr.shape, dtype=arr.dtype)
    mx.eval(dst)
    idx = mx.arange(arr.shape[0], dtype=mx.int32)
    dst[idx] = arr
    mx.eval(dst)
    return dst


WEIGHT_ROOM = 256


def _with_weight_room(arr):
    """Ensure `arr` resides on device with at least `WEIGHT_ROOM` bytes of trailing zero padding on
    non-Metal GPU backends so MLX 0.32.4's CUDA `qmm_sm80_kernel` never reads unmapped memory past
    the last expert row."""
    if arr is None:
        return None
    if mx.metal.is_available() or mx.default_device() != mx.gpu or arr.ndim == 0 or arr.size == 0:
        return _to_device(arr)
    if arr.dtype == mx.uint8 and arr.size % 4 == 0:
        words = arr.size // 4
        pad_words = -(-WEIGHT_ROOM // 4)
        base = mx.zeros((words + pad_words,), dtype=mx.uint32)
        mx.eval(base)
        idx = mx.arange(words, dtype=mx.int32)
        base[idx] = arr.reshape(-1).view(mx.uint32)
        out = base[:words].view(mx.uint8).reshape(arr.shape)
        del base
        mx.eval(out)
        return out
    pad = -(-WEIGHT_ROOM // arr.dtype.size)
    base = mx.zeros((arr.size + pad,), dtype=arr.dtype)
    mx.eval(base)
    idx = mx.arange(arr.size, dtype=mx.int32)
    base[idx] = arr.reshape(-1)
    out = base[:arr.size].reshape(arr.shape)
    del base
    mx.eval(out)
    return out


def _place_attr(proj, attr, idx, val):
    cur = getattr(proj, attr)
    sh = cur.shape
    on_cuda_w = attr == 'weight' and not mx.metal.is_available() and mx.default_device() == mx.gpu
    if cur.dtype == mx.uint8 and sh[0] > 0 and (cur.size // sh[0]) % 4 == 0:
        setattr(proj, attr, None)
        u32 = cur.reshape(sh[0], -1).view(mx.uint32)
        del cur
        u32[idx] = val.reshape(val.shape[0], -1).view(mx.uint32)
        out = u32.view(mx.uint8).reshape(sh)
        if on_cuda_w:
            out = _with_weight_room(out)
        setattr(proj, attr, out)
    else:
        cur[idx] = val
        if on_cuda_w:
            setattr(proj, attr, _with_weight_room(cur))


def _take_rows(arr, take_idx):
    if arr is None:
        return None
    sh = arr.shape
    if arr.dtype == mx.uint8 and sh[0] > 0 and (arr.size // sh[0]) % 4 == 0:
        taken = mx.take(arr.reshape(sh[0], -1).view(mx.uint32), take_idx, axis=0)
        return taken.view(mx.uint8).reshape((take_idx.size,) + sh[1:])
    return mx.take(arr, take_idx, axis=0)


class DemandSwitchGLU(nn.Module):
    """An MoE layer's `switch_mlp` holding `slots` experts in memory and reading missing ones on
    demand through `Demand`."""

    def __init__(self, layer, gate_proj, up_proj, down_proj, experts, num_experts, activation, demand):
        super().__init__()
        self.layer = layer
        self.gate_proj, self.up_proj, self.down_proj = gate_proj, up_proj, down_proj
        self.activation = activation
        self.weighs = expert_qmv.weighs(activation)
        self.demand = demand
        self.experts = np.array(experts, dtype=np.int64, copy=True)
        self.route = np.full(num_experts, -1, np.int64)
        self.route[self.experts] = np.arange(self.experts.size)
        self._identity = (
            demand is None
            and self.experts.size == num_experts
            and np.array_equal(self.experts, np.arange(num_experts, dtype=np.int64))
        )
        if self._identity:
            self._local = None
        elif demand is None:
            self._local = mx.array(np.maximum(self.route, 0).astype(np.uint32))
        else:
            self._local = _to_device(mx.array(np.maximum(self.route, 0).astype(np.uint32)))
        self.freeze()

    def _local_indices(self, indices):
        if self._identity:
            return indices if indices.dtype == mx.uint32 else indices.astype(mx.uint32)
        return self._local[indices]

    def place_many(self, slots, experts, arrays_by_proj):
        """Write `arrays_by_proj` (`{proj: [weight, scales, biases]}`) for `experts` into `slots`
        with one batched update per array so MLX donates each bank buffer in place."""
        slots_arr = np.asarray(slots, dtype=np.int64).reshape(-1)
        exps_arr = np.asarray(experts, dtype=np.int64).reshape(-1)
        if not slots_arr.size:
            return []
        if self.demand is not None:
            self.demand._before_switch_placed(self)
        old_exps = self.experts[slots_arr].copy()
        for slot, old in zip(slots_arr.tolist(), old_exps.tolist()):
            if old >= 0 and self.route[old] == slot:
                self.route[old] = -1
        self.experts[slots_arr] = exps_arr
        self.route[exps_arr] = slots_arr

        idx = mx.array(slots_arr.astype(np.int32))
        written = []
        for p in PROJECTIONS:
            proj = getattr(self, p)
            w, s, b = arrays_by_proj[p]
            _place_attr(proj, 'weight', idx, w)
            written.append(proj.weight)
            if s is not None and getattr(proj, 'scales', None) is not None:
                _place_attr(proj, 'scales', idx, s)
                written.append(proj.scales)
            if b is not None and getattr(proj, 'biases', None) is not None:
                _place_attr(proj, 'biases', idx, b)
                written.append(proj.biases)

        still_evicted = list(dict.fromkeys(
            old for old in old_exps.tolist() if old >= 0 and self.route[old] < 0
        ))
        upd_exps = np.array(still_evicted + exps_arr.tolist(), dtype=np.int32)
        upd_vals = np.array(([0] * len(still_evicted)) + slots_arr.tolist(), dtype=np.uint32)
        if upd_exps.size:
            self._local[mx.array(upd_exps)] = mx.array(upd_vals)
            written.append(self._local)
        if self.demand is not None:
            self.demand._on_switch_placed(self)
        return written

    def place(self, slot, expert, arrays):
        """Write `arrays` (`{proj: [weight, scales, biases]}`) for `expert` into `slot`."""
        return self.place_many([slot], [expert], arrays)

    def replace_all(self, experts, arrays_by_proj):
        """Replace the entire bank with `experts` and their stacked `[weight, scales, biases]`."""
        if self.demand is not None:
            self.demand._before_switch_placed(self)
        self.experts = np.array(experts, dtype=np.int64, copy=True)
        self.route.fill(-1)
        self.route[self.experts] = np.arange(self.experts.size)
        self._identity = False
        written = []
        for p in PROJECTIONS:
            proj = getattr(self, p)
            w_raw, s_raw, b_raw = arrays_by_proj[p]
            w, s, b = _with_weight_room(w_raw), _to_device(s_raw), _to_device(b_raw)
            proj.weight = w
            written.append(proj.weight)
            if s is not None:
                proj.scales = s
                written.append(proj.scales)
            if b is not None:
                proj.biases = b
                written.append(proj.biases)
        self._local = _to_device(mx.array(np.maximum(self.route, 0).astype(np.uint32)))
        written.append(self._local)
        if self.demand is not None:
            self.demand._on_switch_placed(self)
        return written

    def __call__(self, x, indices, scores=None):
        return self.demand.compute(self, x, indices, scores)


def _proj_dims(proj):
    """Return `(out_dims, in_dims)` for a routed-expert projection (unquantized, 3D quantized, or 4D ggml)."""
    if proj.weight.ndim == 4:
        return proj.weight.shape[1], proj.weight.shape[2] * int(getattr(proj, 'group_size', 256))
    bits = getattr(proj, 'bits', None)
    return proj.weight.shape[1], (proj.weight.shape[2] * 32 // bits if bits else proj.weight.shape[2])


def _proj_meta(w, s, bits_spec):
    from . import ggml_quants
    if w.ndim == 4:
        return ggml_quants.FORMAT_BITS[bits_spec], 256, bits_spec
    if bits_spec == 'mxfp4' and s.dtype == mx.uint8:
        in_dims = w.shape[2] * 8
        return 4, in_dims // s.shape[-1], 'mxfp4'
    eff_bits = (4 if bits_spec == 'mxfp4'
                else ggml_quants.FORMAT_BITS[bits_spec] if bits_spec in requant.GGML_MODES
                else int(bits_spec))
    in_dims = w.shape[2] * 32 // eff_bits
    return eff_bits, in_dims // s.shape[-1], 'affine'


class Demand:
    """On-demand expert streaming manager across all MoE layers of `model`."""

    def __init__(self, model, checkpoint, budget_bytes, workers=4, prompt_room=None):
        from . import cpu_experts, cuda_ple, fitting, native_reads
        self.checkpoint = checkpoint
        self.base = checkpoint.base if isinstance(checkpoint, requant.Flat) else checkpoint
        self.workers = max(1, int(workers))
        self.prompt_room = None if not prompt_room else float(prompt_room)
        self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix='demand-read')
        self._native = (
            native_reads.Readers(self.workers)
            if (hasattr(checkpoint, 'segments') and native_reads.enabled())
            else None
        )
        self.native_reads = 0
        self._stream_threads = set()
        self._stream_lock = threading.Lock()
        self._seq_lock = threading.Lock()
        self._prefill_lock = threading.RLock()
        self._stage_lock = threading.RLock()
        self._buf_lock = threading.Lock()
        self._pass_seq = 0
        self._stm = None
        self._stm_why = None
        self._free_least = cuda_ple.free_least()
        self._decode_steps = 0
        self._after_prefill = False
        self.trimmed = 0
        self._cpu_mode = cpu_experts.cpu_setting()
        self._cpu = None
        self._cpu_why = None
        self._cpu_load = cpu_experts.Load(cpu_experts.default_threads()).start() if self._cpu_mode == 'auto' else None
        self._cpu_status = []
        self._cpu_counts = []
        self._cpu_status_lock = threading.Lock()
        self._cpu_eval_tid = None
        self.cpu_served = 0
        self.cpu_late = 0
        self.cpu_wait_ms = 0.0

        layers = arch.decoder_layers(model)
        moe_indices = [i for i, l in enumerate(layers) if arch.switch_of(l) is not None]
        if not moe_indices:
            raise ValueError('model has no MoE SwitchGLU layers for on-demand serving')

        first_layer = moe_indices[0]
        first_switch = arch.switch_of(layers[first_layer])
        num_experts = first_switch.gate_proj.weight.shape[0]
        if isinstance(checkpoint, requant.Flat):
            bits_spec, gs = checkpoint.bits, checkpoint.group_size
            per_expert = 0.0
            for p in PROJECTIONS:
                out_d, in_d = _proj_dims(getattr(first_switch, p))
                eff_b, _, eff_mode = requant.effective_format(bits_spec, in_d, gs)
                per_expert += fitting.quantized_bytes(
                    out_d * in_d, eff_mode if eff_mode in requant.UNIFORM_MODES else eff_b
                )
        else:
            mode = self.base.mode(first_layer, 'gate_proj') or 'affine'
            _, in_dims = _proj_dims(first_switch.gate_proj)
            gs, bits = self.base.quantization(first_layer, 'gate_proj', in_dims)
            total_expert_bytes = (
                self.base.total_bytes(moe_indices)
                if hasattr(self.base, 'total_bytes')
                else sum(
                    self.base.nbytes(self.base.name(l, p, part))
                    for l in moe_indices for p in PROJECTIONS for part in NAMES if self.base.has(l, p, part)
                )
            )
            per_expert = total_expert_bytes // max(1, num_experts * len(moe_indices))

        self.per_expert = max(1, int(per_expert))
        per_layer_Slot = self.per_expert * len(moe_indices)
        self.slots = max(8, min(num_experts, int(budget_bytes // per_layer_Slot)))
        self.num_experts = num_experts
        self.moe_indices = moe_indices
        self.next_moe = {moe_indices[i]: moe_indices[i + 1] for i in range(len(moe_indices) - 1)}

        self.weight = {l: np.zeros(num_experts, np.float64) for l in moe_indices}
        self._advised_views = []
        self._stage_ring = [{} for _ in range(_STAGE_RING)]
        self._stage_inflight = [None] * _STAGE_RING
        self._stage_cursor = 0
        self.buffers = {}
        shared_bufs = {}
        can_share = (not isinstance(checkpoint, requant.Flat)
                     and hasattr(self.base, 'tensors') and hasattr(self.base, 'has') and hasattr(self.base, 'name'))
        for l in moe_indices:
            for p in PROJECTIONS:
                if can_share:
                    sig = (
                        p,
                        tuple(
                            (self.base.tensors[self.base.name(l, p, part)][1],
                             tuple(self.base.tensors[self.base.name(l, p, part)][2][1:]))
                            if self.base.has(l, p, part) else None
                            for part in NAMES
                        ),
                    )
                    if sig not in shared_bufs:
                        worker_bufs = [self.checkpoint.buffers(l, p) for _ in range(self.workers)]
                        for buf in worker_bufs:
                            for arr in (buf or ()):
                                if arr is not None:
                                    view = np.asarray(arr)
                                    if cuda_ple.advise_host_staging(view, enable=True):
                                        self._advised_views.append(view)
                        shared_bufs[sig] = worker_bufs
                    self.buffers[(l, p)] = shared_bufs[sig]
                else:
                    worker_bufs = [self.checkpoint.buffers(l, p) for _ in range(self.workers)]
                    for buf in worker_bufs:
                        for arr in (buf or ()):
                            if arr is not None:
                                view = np.asarray(arr)
                                if cuda_ple.advise_host_staging(view, enable=True):
                                    self._advised_views.append(view)
                    self.buffers[(l, p)] = worker_bufs
        self.switches = {}
        self.hits = 0
        self.misses = 0
        self.prefill_layers = 0
        self.read_s = 0.0
        self.prefill_read_s = 0.0
        self.host_served = 0
        self._prefill_stage = {}
        self._prefill_dev = {}
        self._prefill_active = None

        init_ids = np.arange(self.slots, dtype=np.int64)
        reader = requant.Reader(self.base.path, checkpoint.bits, gs) if isinstance(checkpoint, requant.Flat) else self.base
        for idx, l in enumerate(moe_indices):
            old_switch = arch.switch_of(layers[l])
            projs = {}
            for p in PROJECTIONS:
                w, s, b = reader.experts(l, p, init_ids)
                if isinstance(checkpoint, requant.Flat):
                    p_bits, p_gs, p_mode = _proj_meta(w, s, checkpoint.bits)
                else:
                    p_mode = self.base.mode(l, p) or 'affine'
                    _, in_d = _proj_dims(getattr(old_switch, p))
                    p_gs, p_bits = self.base.quantization(l, p, in_d)
                w, s, b = _with_weight_room(w), _to_device(s), _to_device(b)
                projs[p] = Projection(w, s, b, p_gs, p_bits, mode=p_mode)
                del w, s, b
                mx.eval(projs[p].parameters())
            d_switch = DemandSwitchGLU(
                l, projs['gate_proj'], projs['up_proj'], projs['down_proj'],
                init_ids, num_experts, old_switch.activation, self,
            )
            arch.set_switch(layers[l], d_switch)
            self.switches[l] = d_switch
            mx.clear_cache()

        self._layer_locks = {l: threading.RLock() for l in self.moe_indices}
        from . import stm_core
        self._stm = stm_core.create_driver(self)
        _LIVE.add(self)

    def _before_switch_placed(self, switch):
        if getattr(self, '_stm', None) is not None:
            self._stm.before_switch_placed(switch)

    def _on_switch_placed(self, switch):
        if getattr(self, '_stm', None) is not None:
            self._stm.on_switch_placed(switch)

    def _has_layer_inflight(self, layer):
        with self._stage_lock:
            return any(entry is not None and entry[0] == layer for entry in self._stage_inflight)

    def _finish_decode_step(self):
        with self._seq_lock:
            self._decode_steps += 1
            do_room = self._after_prefill or (self._decode_steps % 64 == 0)
            if do_room:
                self._after_prefill = False
        if do_room:
            self._room_back()

    def _mark_prefill(self):
        with self._seq_lock:
            self._after_prefill = True
            self._decode_steps = 0

    def thread_done(self):
        """Release any STM layer leases remaining from the calling thread's last pass."""
        if getattr(self, '_stm', None) is not None:
            self._stm.thread_done()

    def _room_back(self):
        if self._free_least is not None:
            from . import cuda_ple
            if cuda_ple.room_back(self._free_least):
                with self._seq_lock:
                    self.trimmed += 1

    def _host(self):
        """The host's copy of the experts by rows (q2_experts.HostCopy with `by_rows=True`), or None."""
        if isinstance(self.checkpoint, requant.Flat):
            return None
        held = self.base.held()
        return held if held is not None and held.by_rows else None

    def _cpu_allowed(self):
        """Worker threads currently permitted for CPU expert offload (`0` when disabled or yielded)."""
        if self._cpu_why is not None or self._cpu_mode == 'off':
            return 0
        if self._cpu_mode == 'auto':
            return int(self._cpu_load.allowed()) if self._cpu_load is not None else 0
        return self._cpu.pool.threads if self._cpu is not None else 1

    def _flush_cpu_counts(self):
        """Evaluate accumulated CUDA `CPU_WAIT` status telemetry into scalar counters."""
        with self._cpu_status_lock:
            statuses = self._cpu_status
            self._cpu_status = []
            counts = self._cpu_counts
            self._cpu_counts = []
        if statuses:
            counts.append(mx.sum(mx.stack(statuses)[:, 1::2], axis=0))
        if counts:
            totals = mx.sum(mx.stack(counts), axis=0)
            mx.eval(totals)
            late_add = int(totals[0].item())
            wait_ms_add = float(totals[1].item()) / 1e3
            with self._cpu_status_lock:
                self.cpu_late += late_add
                self.cpu_wait_ms += wait_ms_add
                if self.cpu_late > 0 and self._cpu_why is None:
                    self._cpu_why = 'CPU expert server timed out'

    def _step_cpu(self):
        """Fold the previous step's CPU statuses asynchronously and refresh allowed worker count."""
        with self._cpu_status_lock:
            self._cpu_eval_tid = threading.get_ident()
            statuses = self._cpu_status
            self._cpu_status = []
        if statuses:
            late = mx.sum(mx.stack(statuses)[:, 1::2], axis=0)
            mx.async_eval(late)
            with self._cpu_status_lock:
                self._cpu_counts.append(late)
                should_flush = len(self._cpu_counts) >= 16
            if should_flush:
                self._flush_cpu_counts()
        cpu = self._cpu
        if cpu is not None:
            allowed = self._cpu_allowed()
            cpu.active = allowed
            if cpu.server is not None:
                cpu.server.active(allowed)

    def _cpu_layer(self, layer, switch, hidden_dim, host):
        """Ensure the CPU worker pool/server are started and `layer` is registered, or return False."""
        from . import cpu_experts, cuda_kernels
        if self._cpu_why is not None or self._cpu_mode == 'off' or host is None or not self._has_tensor_table():
            return False
        if self._cpu is not None:
            if layer in self._cpu.layers:
                return True
            if layer in self._cpu.declined:
                return False

        act_name = type(switch.activation).__name__
        if 'SwiGLU' not in act_name:
            if self._cpu is not None:
                self._cpu.declined.add(layer)
            return False
        weighs = bool(getattr(switch, 'weighs', False))
        activation = 'deepseek_v41' if (weighs or act_name == 'DeepseekV41SwiGLU') else 'plain'
        limit = float(getattr(switch.activation, 'limit', 0.0) or 0.0)

        arrays, groups = [], {}
        layer_kind, layer_bits = None, None
        for p in PROJECTIONS:
            p_mode = self.base.mode(layer, p) or 'affine'
            if p_mode not in ('affine', 'mxfp4'):
                if self._cpu is not None:
                    self._cpu.declined.add(layer)
                return False
            out_d, in_d = _proj_dims(getattr(switch, p))
            if p_mode == 'affine':
                if not (self.base.has(layer, p, 'weight') and self.base.has(layer, p, 'scales')
                        and self.base.has(layer, p, 'biases')):
                    return False
                gs, bits = self.base.quantization(layer, p, in_d)
                w_st = self.base.tensors[self.base.name(layer, p, 'weight')][1]
                s_st = self.base.tensors[self.base.name(layer, p, 'scales')][1]
                b_st = self.base.tensors[self.base.name(layer, p, 'biases')][1]
                if (bits not in (2, 4, 8) or gs <= 0 or in_d % gs or in_d % (32 // bits)
                        or w_st != 'U32' or s_st != 'BF16' or b_st != 'BF16'):
                    return False
            else:
                if not (self.base.has(layer, p, 'weight') and self.base.has(layer, p, 'scales')) or self.base.has(
                    layer, p, 'biases'
                ):
                    return False
                gs, bits = self.base.quantization(layer, p, in_d)
                w_st = self.base.tensors[self.base.name(layer, p, 'weight')][1]
                s_st = self.base.tensors[self.base.name(layer, p, 'scales')][1]
                if bits != 4 or gs != 32 or in_d % 32 or w_st not in ('U32', 'I8', 'U8') or s_st not in ('U8', 'F8_E8M0'):
                    return False
            if p == 'gate_proj' and activation == 'deepseek_v41' and out_d % 32:
                return False
            if layer_kind is None:
                layer_kind, layer_bits = p_mode, bits
            elif (layer_kind, layer_bits) != (p_mode, bits):
                return False
            groups[p] = gs
            for part in NAMES:
                if not self.base.has(layer, p, part):
                    continue
                t_name = self.base.name(layer, p, part)
                arr = host.arrays.get(t_name)
                if arr is None or arr.shape[0] == 0:
                    return False
                arrays.append((p, part, arr))

        spec = cpu_experts.layer_spec(arrays, groups, layer_kind, layer_bits)
        try:
            if self._cpu is None:
                if cpu_experts.lib() is None or cpu_experts.isa() is None:
                    raise ValueError('CPU experts C++ library failed to build')
                if cpu_experts.openblas_one_thread():
                    print('quanteeze: numpy OpenBLAS held to 1 thread alongside CPU expert workers',
                          file=sys.stderr, flush=True)
                pool = cpu_experts.Pool()
                if self._cpu_load is not None:
                    self._cpu_load.threads = pool.threads
                rng = np.random.default_rng(0)
                n_held = int(next(iter(spec['arrays']['gate_proj'].values())).shape[0])
                for slot in range(cpu_experts.CHECKS):
                    xs = rng.normal(0, 1, hidden_dim).astype(np.float32)
                    if pool.expert(cpu_experts.layer_expert(spec, slot % n_held), xs) is None and cpu_experts.verified()[1]:
                        pool.close()
                        raise ValueError(cpu_experts.verified()[1])
                active_init = self._cpu_load.allowed() if self._cpu_load is not None else pool.threads
                on_cuda = cuda_kernels.available() and not mx.metal.is_available()
                try:
                    server = cpu_experts.Server(pool, self.moe_indices, hidden_dim) if on_cuda else None
                except Exception:
                    pool.close()
                    raise
                if server is not None:
                    server.active(active_init)
                self._cpu = SimpleNamespace(
                    pool=pool, server=server, layers=set(), specs={}, declined=set(),
                    active=active_init,
                )
                where = (f'server thread on CPUs {server.cpus}' if server is not None else 'direct pool')
                q8_info = ', q8' if (pool.q8 and pool.isa >= 1) else ''
                print(
                    f'quanteeze: decode misses in host memory computed on the CPU '
                    f'(QUANTEEZE_CPU_EXPERTS={self._cpu_mode}): {pool.threads} workers on CPUs {pool.cpus}, '
                    f'{where}, {cpu_experts.ISA_NAMES[pool.isa]}{q8_info}',
                    file=sys.stderr, flush=True,
                )
            if self._cpu.server is not None:
                self._cpu.server.add(layer, spec, limit, activation)
                self._cpu.server.start()
            self._cpu.specs[layer] = (spec, limit, activation)
            self._cpu.layers.add(layer)
            return True
        except (ValueError, OSError) as e:
            self._cpu_why = f'{type(e).__name__}: {e}'
            print(f'quanteeze: CPU expert offload off ({self._cpu_why})', file=sys.stderr, flush=True)
            return False

    def _native_submit_expert(self, layer, expert, worker_slot, cache=True):
        """Submit one expert's 3-projection read via `native_reads`, or return None."""
        from . import native_reads
        if self._native is None or not native_reads.enabled() or not hasattr(self.checkpoint, 'segments'):
            return None
        rooms = {}
        segs = []
        for p in PROJECTIONS:
            room = self.buffers[(layer, p)][worker_slot]
            got = self.checkpoint.segments(layer, p, int(expert), room)
            if not got:
                return None
            rooms[p] = room
            segs.extend(got)
        if len(segs) > native_reads.MAX_SEGMENTS:
            return None

        ckpt = self.checkpoint
        exp_id = int(expert)

        def finish():
            return exp_id, {p: ckpt.made(layer, p, rooms[p]) for p in PROJECTIONS}

        check = None
        if native_reads.checking():
            def check():
                for _file, _off, _length, _ptr, dst_u8, key in segs:
                    want = ckpt._from_file(key, [exp_id], cache=cache).view(np.uint8).reshape(-1)
                    if dst_u8.shape != want.shape or not np.array_equal(dst_u8, want):
                        native_reads.record_check(False, f'{key} expert {exp_id}')
                        return False
                native_reads.record_check(True)
                return True

        def fallback():
            self._native = None
            return self._read_one(layer, exp_id, worker_slot)

        return self._native.read(segs, finish=finish, cache=cache, priority=0, check=check, fallback=fallback)

    def _native_prefill_targets(self, remaining_targets, rows_arg):
        """Read unheld prefill targets directly via `native_reads`, or return False to fall back."""
        from . import native_reads, request_threads
        if self._native is None or not native_reads.enabled() or not hasattr(self.base, 'tensors'):
            return False
        prio = 1 if request_threads.others_serving() else 0
        segs = []
        for t_name, out_view in remaining_targets:
            entry = self.base.tensors.get(t_name)
            if entry is None:
                return False
            file, _st_dtype, shape, offset = entry
            rows_arr = np.arange(shape[0], dtype=np.int64) if rows_arg is None else np.asarray(rows_arg, dtype=np.int64).reshape(-1)
            if rows_arr.size == 0:
                continue
            if not out_view.flags.c_contiguous or not out_view.flags.writeable or out_view.shape[0] != rows_arr.size:
                return False
            u8_out = out_view.reshape(rows_arr.size, -1).view(np.uint8)
            row_bytes = int(u8_out.shape[1])
            if file is not None:
                i = 0
                while i < rows_arr.size:
                    j = i + 1
                    while j < rows_arr.size and rows_arr[j] == rows_arr[j - 1] + 1:
                        j += 1
                    segs.append((str(file), int(offset) + int(rows_arr[i]) * row_bytes, (j - i) * row_bytes,
                                 int(u8_out[i].ctypes.data), u8_out[i:j].reshape(-1), t_name, rows_arr[i:j]))
                    i = j
            else:
                for i, e in enumerate(rows_arr):
                    part_file, part_off = offset[int(e)]
                    segs.append((str(part_file), int(part_off), row_bytes,
                                 int(u8_out[i].ctypes.data), u8_out[i:i + 1].reshape(-1), t_name, rows_arr[i:i + 1]))
        if not segs:
            return True
        futs = []
        ok = False
        try:
            for idx in range(0, len(segs), native_reads.MAX_SEGMENTS):
                batch = segs[idx:idx + native_reads.MAX_SEGMENTS]
                check = None
                fallback = None
                if idx == 0 and native_reads.checking():
                    def check(b=batch):
                        for _f, _o, _l, _p, dst_u8, key, part_rows in b:
                            want = self.base._from_file(key, part_rows, cache=False).view(np.uint8).reshape(-1)
                            if dst_u8.shape != want.shape or not np.array_equal(dst_u8, want):
                                native_reads.record_check(False, f'{key} prefill')
                                return False
                        native_reads.record_check(True)
                        return True

                    def fallback(b=batch):
                        self._native = None
                        for _f, _o, _l, _p, dst_u8, key, part_rows in b:
                            dst_u8[:] = self.base._from_file(key, part_rows, cache=False).view(np.uint8).reshape(-1)
                fut = self._native.read(batch, cache=False, priority=prio, check=check, fallback=fallback)
                if fut is None:
                    return False
                futs.append(fut)
            for fut in futs:
                fut.result()
            with self._seq_lock:
                self.native_reads += len(futs)
            ok = True
            return True
        finally:
            if not ok:
                for fut in futs:
                    fut.close()

    def _load_disk_missing(self, layer, switch, disk_victims, disk_missing, in_pass=True):
        """Read `disk_missing` experts from disk via C++ STM lander, lock-free `pread`, or `_pool`."""
        from . import native_reads, parallel
        parallel.require(layer, 'demand load')
        stm = getattr(self, '_stm', None)
        if stm is not None and stm.can_serve_reads() and not native_reads.checking():
            self._drain_layer_inflight(layer)
            if stm.land_missing(layer, switch, disk_victims, disk_missing, in_pass=in_pass):
                return
        reacquire = False
        if stm is not None and stm.started:
            if in_pass:
                stm.layer_done(layer)
                reacquire = True
            stm.wait_layer_idle(layer)
        try:
            with self._buf_lock:
                for start in range(0, len(disk_victims), self.workers):
                    batch_slots = disk_victims[start:start + self.workers]
                    batch_exps = disk_missing[start:start + len(batch_slots)]
                    futures = []
                    native_flags = []
                    try:
                        for w_idx, e in enumerate(batch_exps):
                            fut = self._native_submit_expert(layer, int(e), w_idx, cache=True)
                            if fut is not None:
                                futures.append(fut)
                                native_flags.append(True)
                            else:
                                futures.append(self._pool.submit(self._read_one, layer, e, w_idx))
                                native_flags.append(False)
                        results = [fut.result() for fut in futures]
                    finally:
                        for flag, fut in zip(native_flags, futures):
                            if flag:
                                fut.close()
                    n_native = sum(1 for flag in native_flags if flag and native_reads._STATE['failed'] is None)
                    if n_native:
                        with self._seq_lock:
                            self.native_reads += n_native
                    if any(native_flags):
                        to_eval = [
                            a for flag, (_, d) in zip(native_flags, results) if flag
                            for p in PROJECTIONS for a in d[p] if a is not None
                        ]
                        if to_eval:
                            mx.eval(to_eval)
                    batch_exps_done = [int(r[0]) for r in results]
                    if len(results) == 1:
                        batch_arrays = results[0][1]
                    else:
                        batch_arrays = {
                            p: [
                                None if results[0][1][p][i] is None
                                else mx.concatenate([r[1][p][i] for r in results], axis=0)
                                for i in range(len(NAMES))
                            ]
                            for p in PROJECTIONS
                        }
                    batch_written = switch.place_many(batch_slots, batch_exps_done, batch_arrays)
                    if batch_written:
                        mx.eval(batch_written)
        finally:
            if reacquire:
                stm.begin_layer(layer)

    def _load_and_place(self, layer, switch, victims, in_host, disk_missing, *, in_pass=True):
        """Stage host-resident experts and/or load disk-missing experts into `switch` safely under STM."""
        from . import parallel
        parallel.require(layer, 'demand load')
        self._drain_layer_inflight(layer)
        stm = getattr(self, '_stm', None)
        written = []
        ring = None
        reacquire = False
        try:
            if in_host.size:
                ring, copied = self._stage_host_experts(layer, in_host)
                if stm is not None and stm.started:
                    if in_pass:
                        stm.layer_done(layer)
                        reacquire = True
                    stm.wait_layer_idle(layer)
                written = switch.place_many(victims[:in_host.size], in_host, copied)
            disk_victims = victims[in_host.size:]
            if disk_missing.size == 0:
                if written:
                    if in_pass:
                        with self._stage_lock:
                            self._stage_inflight[ring] = (layer, written)
                    else:
                        mx.eval(written)
                    del written
            else:
                if written:
                    mx.eval(written)
                    written.clear()
                self._load_disk_missing(layer, switch, disk_victims, disk_missing, in_pass=in_pass)
        finally:
            if reacquire and stm is not None:
                stm.begin_layer(layer)

    def _cpu_offload(self, layer, switch, x, indices, scores, ids, active_ids, present_mask, missing, host, in_host):
        """Compute `in_host` decode misses (`B=1..8`) on CPU in parallel with GPU-resident experts, or return None."""
        from . import cpu_experts
        if (x.ndim != 3 or x.shape[1] != 1 or not 1 <= x.shape[0] <= cpu_experts.SERVE_BMAX
                or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)):
            return None
        if self._cpu_allowed() <= 0 or not self._cpu_layer(layer, switch, int(x.shape[-1]), host):
            return None
        c = self._cpu
        b_rows, top_k = int(x.shape[0]), int(indices.shape[-1])
        in_host_set = set(int(e) for e in in_host.tolist())
        ids_flat = [int(e) for e in ids.tolist()]
        req_exps, req_xrows, inv_list = [], [], []
        if switch.weighs:
            for j, e in enumerate(ids_flat):
                b_idx = j // top_k
                if e in in_host_set and len(req_exps) < cpu_experts.SERVE_MMAX:
                    inv_list.append(len(req_exps))
                    req_exps.append(e)
                    req_xrows.append(b_idx)
                else:
                    inv_list.append(-1)
        else:
            pair_pos = {}
            for j, e in enumerate(ids_flat):
                b_idx = j // top_k
                key = (b_idx, e)
                if key in pair_pos:
                    inv_list.append(pair_pos[key])
                elif e in in_host_set and len(req_exps) < cpu_experts.SERVE_MMAX:
                    m_idx = len(req_exps)
                    pair_pos[key] = m_idx
                    inv_list.append(m_idx)
                    req_exps.append(e)
                    req_xrows.append(b_idx)
                else:
                    inv_list.append(-1)
        if not req_exps:
            return None

        unhandled_missing = {
            ids_flat[j] for j, iv in enumerate(inv_list) if iv < 0 and switch.route[ids_flat[j]] < 0
        }
        gpu_missing = np.array([int(e) for e in missing.tolist() if int(e) in unhandled_missing], dtype=np.int64)
        gpu_in_host = np.array([int(e) for e in in_host.tolist() if int(e) in unhandled_missing], dtype=np.int64)
        if gpu_missing.size:
            free_mask = np.ones(switch.experts.size, dtype=bool)
            if present_mask.any():
                free_mask[switch.route[active_ids[present_mask]]] = False
            cand = np.flatnonzero(free_mask)
            victims = cand[np.argsort(self.weight[layer][switch.experts[cand]], kind='stable')[:gpu_missing.size]]
            gpu_in_host_set = set(gpu_in_host.tolist())
            disk_missing = np.array([int(e) for e in gpu_missing.tolist() if int(e) not in gpu_in_host_set],
                                    dtype=np.int64)
            self._load_and_place(layer, switch, victims, gpu_in_host, disk_missing, in_pass=True)

        req_slots = host.slot_of[layer][np.asarray(req_exps, dtype=np.int64)].astype(np.int32)
        inv = np.asarray(inv_list, dtype=np.int32)
        weights = scores if switch.weighs else None
        has_gpu_slots = bool((inv < 0).any())

        if c.server is not None:
            rows_mx = mx.array(req_slots, dtype=mx.int32)
            inv_mx = mx.array(inv, dtype=mx.int32)
            token = cpu_experts.cpu_ask(c.server.addr[layer], x, rows_mx, inv_mx, weights)
            if has_gpu_slots:
                y_gpu = expert_qmv.decode(switch, x, switch._local_indices(indices), weights)
            else:
                y_gpu = mx.zeros((b_rows, 1, top_k, int(x.shape[-1])), dtype=x.dtype)
            out, status = cpu_experts.cpu_wait(c.server.addr[layer], token, y_gpu, inv_mx, most=int(req_slots.size))
            mx.async_eval(out, status)
            with self._cpu_status_lock:
                self._cpu_eval_tid = threading.get_ident()
                self._cpu_status.append(status)
        else:
            if has_gpu_slots:
                y_gpu = expert_qmv.decode(switch, x, switch._local_indices(indices), weights)
                mx.async_eval(y_gpu)
            xf = np.array(x.astype(mx.float32)).reshape(b_rows, -1)
            w_np = np.array(scores.astype(mx.float32)).reshape(-1) if switch.weighs else None
            spec, limit, activation = c.specs[layer]
            computed = []
            for m, slot in enumerate(req_slots.tolist()):
                w_val = 1.0
                if switch.weighs and w_np is not None:
                    for j, iv in enumerate(inv_list):
                        if iv == m:
                            w_val = float(w_np[j])
                            break
                res = c.pool.expert(cpu_experts.layer_expert(spec, int(slot)), xf[req_xrows[m]], limit, w_val, activation)
                if res is None:
                    err = cpu_experts.verified()[1]
                    if err:
                        with self._cpu_status_lock:
                            self._cpu_why = err
                    return None
                computed.append(res)
            stacked = np.stack(computed, axis=0)
            valid_inv = np.where(inv >= 0, inv, 0)
            cpu_y = mx.array(stacked[valid_inv]).astype(x.dtype).reshape(b_rows, 1, top_k, int(x.shape[-1]))
            out = mx.where(mx.array(inv >= 0).reshape(b_rows, 1, top_k, 1), cpu_y, y_gpu) if has_gpu_slots else cpu_y

        cpu_unique = len({int(e) for e in req_exps if int(e) not in unhandled_missing})
        with self._seq_lock:
            self.host_served += int(gpu_in_host.size + cpu_unique)
        with self._cpu_status_lock:
            self.cpu_served += int(len(req_exps))
        return out

    def _close_cpu(self):
        """Flush telemetry and stop the CPU worker pool, polling server, and load monitor."""
        with self._cpu_status_lock:
            eval_tid = self._cpu_eval_tid
        if eval_tid is not None and threading.get_ident() == eval_tid:
            self._flush_cpu_counts()
        else:
            with self._cpu_status_lock:
                self._cpu_status = []
                self._cpu_counts = []
        with self._cpu_status_lock:
            cpu = self._cpu
            self._cpu = None
        if cpu is not None:
            try:
                mx.synchronize()
            except Exception:
                pass
            if cpu.server is not None:
                cpu.server.close()
            if cpu.pool is not None:
                cpu.pool.close()
        if self._cpu_load is not None:
            self._cpu_load.close()
            self._cpu_load = None

    def _drain_layer_inflight(self, layer):
        """Ensure any async stage write targeting `layer` has completed and been cleared so bank
        buffers have refcount 1 for in-place donation."""
        with self._stage_lock:
            for k, entry in enumerate(self._stage_inflight):
                if entry is not None and entry[0] == layer:
                    mx.eval(entry[1])
                    self._stage_inflight[k] = None

    def _stage_host_experts(self, layer, in_host):
        """Copy `in_host` expert rows from `HostCopy` directly into reusable `cuMemAdvise`-hinted
        `stream=mx.cpu` staging buffers in the next ring slot."""
        from . import cuda_ple
        with self._stage_lock:
            ring = self._stage_cursor
            self._stage_cursor = (ring + 1) % _STAGE_RING
            if self._stage_inflight[ring] is not None:
                mx.eval(self._stage_inflight[ring][1])
                self._stage_inflight[ring] = None
            m = int(in_host.size)
            cap = max(8, 1 << max(0, m - 1).bit_length())
            copied = {}
            targets = []
            for p in PROJECTIONS:
                parts = []
                for part in NAMES:
                    if not self.base.has(layer, p, part):
                        parts.append(None)
                        continue
                    t_name = self.base.name(layer, p, part)
                    _, st_dtype, full_shape, _ = self.base.tensors[t_name]
                    tail = tuple(full_shape[1:])
                    np_dt = np.dtype(q2_experts._SAFETENSORS[st_dtype])
                    row_bytes = int(np.prod(tail, dtype=np.int64)) * np_dt.itemsize
                    used_bytes = m * row_bytes
                    need_bytes = cap * row_bytes
                    need_words = max(1, (need_bytes + 3) // 4)
                    slot_key = (p, part)
                    entry = self._stage_ring[ring].get(slot_key)
                    if entry is None or entry[0].shape[0] < need_words:
                        if entry is not None and entry[2]:
                            cuda_ple.advise_host_staging(entry[1], enable=False)
                        mlx_arr = mx.zeros((need_words,), mx.uint32, stream=mx.cpu)
                        mx.eval(mlx_arr)
                        np_view = np.asarray(mlx_arr).view(np.uint8)
                        adv = cuda_ple.advise_host_staging(np_view, enable=True)
                        entry = (mlx_arr, np_view, adv)
                        self._stage_ring[ring][slot_key] = entry
                    mlx_arr, np_u8, _ = entry
                    out_view = np_u8[:used_bytes].view(np_dt).reshape((m,) + tail)
                    targets.append((t_name, out_view))
                    if np_dt == np.uint32:
                        raw_part = mlx_arr[:used_bytes // 4].reshape((m,) + tail)
                    elif (used_bytes % 4) == 0:
                        raw_part = mlx_arr[:used_bytes // 4].view(q2_experts._MLX[np_dt]).reshape((m,) + tail)
                    else:
                        raw_part = mlx_arr.view(mx.uint8)[:used_bytes].view(q2_experts._MLX[np_dt]).reshape((m,) + tail)
                    parts.append(q2_experts._as_mlx(raw_part, st_dtype))
                copied[p] = parts
            host = self._host()
            if targets:
                if host is None or not host.copy_layer(layer, in_host, targets):
                    for t_name, out_view in targets:
                        self.base._read(t_name, in_host, cache=True, out=out_view)
            return ring, copied

    def _has_tensor_table(self):
        return (
            not isinstance(self.checkpoint, requant.Flat)
            and hasattr(self.base, 'tensors')
            and hasattr(self.base, '_read')
        )

    def _layer_expert_bytes(self, layer):
        if layer is None or not self._has_tensor_table():
            return self.per_expert
        total = 0
        for p in PROJECTIONS:
            for part in NAMES:
                if not self.base.has(layer, p, part):
                    continue
                t_name = self.base.name(layer, p, part)
                _, st_dtype, full_shape, _ = self.base.tensors[t_name]
                row_elems = int(np.prod(full_shape[1:], dtype=np.int64))
                total += row_elems * np.dtype(q2_experts._SAFETENSORS[st_dtype]).itemsize
        return max(1, total)

    def _prefill_cap(self, layer=None):
        per_exp = self._layer_expert_bytes(layer)
        prompt_cap = self.num_experts if not self.prompt_room else max(1, int(self.prompt_room // per_exp))
        if not mx.metal.is_available() and mx.default_device() == mx.gpu:
            from . import cuda_ple
            if not self._prefill_dev:
                mx.clear_cache()
            held_dev = sum(int(a.nbytes) for a in self._prefill_dev.values())
            reserve = max(int(self._free_least or 0), 256 << 20)

            def _usable_vram():
                room = cuda_ple.gpu_room()
                if room is None:
                    return None
                avail = int(room[0]) + int(room[1]) + held_dev
                return avail - reserve if avail > reserve else avail // 2

            usable = _usable_vram()
            if usable is not None:
                # Reclaim dead MLX cache tensors accumulated earlier in the layer (e.g. attention)
                # so transient pool occupancy cannot push uniform 4-bit out of in_layer_mode.
                if usable < self.num_experts * per_exp and mx.get_cache_memory() > 0:
                    mx.clear_cache()
                    usable = _usable_vram()
                    if usable is None:
                        return prompt_cap
                vram_cap = max(8, self.slots // 2, int(usable // per_exp))
                return min(prompt_cap, vram_cap)
        return prompt_cap

    def _load_prefill_projs(self, layer, switch, expert_ids, cap_rows):
        from . import cuda_ple
        exp_ids = np.asarray(expert_ids, np.int64).reshape(-1)
        m = int(exp_ids.size)
        cap_rows = max(m, int(cap_rows))
        on_cuda = not mx.metal.is_available() and mx.default_device() == mx.gpu
        t_read = time.perf_counter()
        if not self._has_tensor_table():
            reader = (
                requant.Reader(self.base.path, self.checkpoint.bits, self.checkpoint.group_size)
                if isinstance(self.checkpoint, requant.Flat) else self.base
            )
            projs = {}
            for p in PROJECTIONS:
                w, s, b = reader.experts(layer, p, exp_ids)
                if on_cuda:
                    w, s, b = _with_weight_room(w), _to_device(s), _to_device(b)
                ref = getattr(switch, p)
                projs[p] = Projection(w, s, b, ref.group_size, ref.bits, mode=ref.mode)
            mx.eval([
                a for pr in projs.values()
                for a in (pr.weight, pr.get('scales'), pr.get('biases')) if a is not None
            ])
            with self._seq_lock:
                self.prefill_read_s += time.perf_counter() - t_read
            return projs

        is_all = (m == self.num_experts and np.array_equal(exp_ids, np.arange(self.num_experts, dtype=np.int64)))
        rows_arg = None if is_all else exp_ids
        ahead_map = getattr(self.base, '_ahead', None)
        remaining_targets = []
        stage_meta = {}
        for p in PROJECTIONS:
            for part in NAMES:
                if not self.base.has(layer, p, part):
                    continue
                t_name = self.base.name(layer, p, part)
                _, st_dtype, full_shape, _ = self.base.tensors[t_name]
                tail = tuple(full_shape[1:])
                np_dt = np.dtype(q2_experts._SAFETENSORS[st_dtype])
                row_bytes = int(np.prod(tail, dtype=np.int64)) * np_dt.itemsize
                used_bytes = m * row_bytes
                need_bytes = cap_rows * row_bytes
                pages_used = max(1, (used_bytes + _PREFILL_PAGE - 1) // _PREFILL_PAGE)
                pages_cap = max(pages_used, (need_bytes + _PREFILL_PAGE - 1) // _PREFILL_PAGE)
                slot_key = (p, part)
                entry = self._prefill_stage.get(slot_key)
                if on_cuda:
                    if entry is None or entry[0] is None or entry[0].shape[0] < pages_cap:
                        if entry is not None and entry[2]:
                            cuda_ple.advise_host_staging(entry[1], enable=False)
                        stage_u32 = mx.zeros((pages_cap, _PREFILL_WORDS), mx.uint32, stream=mx.cpu)
                        mx.eval(stage_u32)
                        stage_np_u8 = np.asarray(stage_u32).view(np.uint8)
                        adv = cuda_ple.advise_host_staging(stage_np_u8, enable=True)
                        entry = (stage_u32, stage_np_u8, adv)
                        self._prefill_stage[slot_key] = entry
                    np_view = entry[1].reshape(-1)[:used_bytes].view(np_dt).reshape((m,) + tail)
                else:
                    if entry is None or entry[1].size < pages_cap * _PREFILL_PAGE:
                        stage_np_u8 = np.empty(pages_cap * _PREFILL_PAGE, np.uint8)
                        entry = (None, stage_np_u8, False)
                        self._prefill_stage[slot_key] = entry
                    np_view = entry[1].reshape(-1)[:used_bytes].view(np_dt).reshape((m,) + tail)
                stage_meta[slot_key] = (st_dtype, tail, np_dt, used_bytes, pages_used, pages_cap, entry, np_view)
                pending = None
                pending_all = None
                if ahead_map is not None:
                    pending = ahead_map.pop(self.base._key(t_name, rows_arg), None)
                    if pending is None and not is_all:
                        pending_all = ahead_map.pop(self.base._key(t_name, None), None)
                if pending is not None:
                    np_view[:] = pending.result()[:m]
                elif pending_all is not None:
                    np.take(pending_all.result(), exp_ids, axis=0, out=np_view, mode='clip')
                else:
                    remaining_targets.append((t_name, np_view))

        if remaining_targets:
            host = self._host()
            if host is None or not host.copy_layer(layer, rows_arg, remaining_targets):
                if not self._native_prefill_targets(remaining_targets, rows_arg):
                    for t_name, out_view in remaining_targets:
                        self.base._read(t_name, rows_arg, cache=False, out=out_view)

        if on_cuda and self._prefill_dev:
            needed_keys = set(stage_meta.keys())
            stale = [k for k in self._prefill_dev if k not in needed_keys]
            if stale:
                for k in stale:
                    del self._prefill_dev[k]
                mx.clear_cache()

        projs = {}
        for p in PROJECTIONS:
            parts = []
            for part in NAMES:
                slot_key = (p, part)
                meta = stage_meta.get(slot_key)
                if meta is None:
                    parts.append(None)
                    continue
                st_dtype, tail, np_dt, used_bytes, pages_used, pages_cap, entry, np_view = meta
                stage_u32 = entry[0]
                if on_cuda:
                    dev_u32 = self._prefill_dev.pop(slot_key, None)
                    if dev_u32 is None or dev_u32.shape[0] < pages_cap + 1:
                        if dev_u32 is not None:
                            dev_u32 = None
                            mx.clear_cache()
                        dev_u32 = mx.zeros((pages_cap + 1, _PREFILL_WORDS), mx.uint32)
                        mx.eval(dev_u32)
                    idx = mx.arange(pages_used, dtype=mx.int32)
                    dev_u32[idx] = stage_u32[:pages_used]
                    self._prefill_dev[slot_key] = dev_u32
                    if np_dt == np.uint32:
                        raw_dev = dev_u32.reshape(-1)[:used_bytes // 4].reshape((m,) + tail)
                    elif (used_bytes % 4) == 0:
                        raw_dev = dev_u32.reshape(-1)[:used_bytes // 4].view(q2_experts._MLX[np_dt]).reshape((m,) + tail)
                    else:
                        raw_dev = dev_u32.view(mx.uint8).reshape(-1)[:used_bytes].view(q2_experts._MLX[np_dt]).reshape((m,) + tail)
                    parts.append(q2_experts._as_mlx(raw_dev, st_dtype))
                else:
                    parts.append(q2_experts._as_mlx(mx.array(np_view), st_dtype))
            w, s, b = parts
            ref = getattr(switch, p)
            projs[p] = Projection(w, s, b, ref.group_size, ref.bits, mode=ref.mode)

        mx.eval([
            a for pr in projs.values()
            for a in (pr.weight, pr.get('scales'), pr.get('biases')) if a is not None
        ])
        with self._seq_lock:
            self.prefill_read_s += time.perf_counter() - t_read
        return projs

    def _update_bank_after_prefill(self, switch, seen_experts, source_ids, source_projs):
        """Promote highest-weight experts seen during prefill into `switch`'s resident bank once."""
        from . import native_reads
        layer = switch.layer
        candidates = np.union1d(switch.experts, seen_experts)
        top = np.sort(candidates[np.argsort(-self.weight[layer][candidates], kind='stable')[:self.slots]])
        new_exps = top[switch.route[top] < 0]
        if not new_exps.size:
            return
        self._drain_layer_inflight(layer)
        top_set = set(int(t) for t in top)
        gone = [s for s, e in enumerate(switch.experts) if int(e) not in top_set]
        stm = getattr(self, '_stm', None)
        last_map = {int(e): idx for idx, e in enumerate(source_ids)} if source_ids is not None else {}
        to_read = np.array([int(e) for e in new_exps if int(e) not in last_map], dtype=np.int64)
        if to_read.size == 0 and source_projs is not None:
            take_idx = mx.array([last_map[int(e)] for e in new_exps], dtype=mx.int32)
            stacked = {
                p: [
                    _take_rows(getattr(source_projs[p], part, None), take_idx)
                    for part in NAMES
                ]
                for p in PROJECTIONS
            }
            if stm is not None and stm.started:
                stm.wait_layer_idle(layer)
            written = switch.place_many(gone[:new_exps.size], new_exps, stacked)
            if written:
                mx.eval(written)
            del stacked, written
            return

        if (to_read.size == new_exps.size and stm is not None
                and stm.can_serve_reads() and not native_reads.checking()):
            if stm.land_missing(layer, switch, gone[:new_exps.size], new_exps, in_pass=False):
                return

        t_read = time.perf_counter()
        fetched = None
        if to_read.size and not isinstance(self.checkpoint, requant.Flat) and self._has_tensor_table():
            host = self._host()
            if host is not None:
                host_bufs = {}
                host_targets = []
                for p in PROJECTIONS:
                    for part in NAMES:
                        if not self.base.has(layer, p, part):
                            continue
                        t_name = self.base.name(layer, p, part)
                        _, st_dtype, full_shape, _ = self.base.tensors[t_name]
                        np_dt = np.dtype(q2_experts._SAFETENSORS[st_dtype])
                        buf = np.empty((int(to_read.size),) + tuple(full_shape[1:]), np_dt)
                        host_targets.append((t_name, buf))
                        host_bufs[(p, part)] = (buf, st_dtype)
                if host.copy_layer(layer, to_read, host_targets):
                    fetched = {
                        p: [
                            None if (p, part) not in host_bufs
                            else q2_experts._as_mlx(mx.array(host_bufs[(p, part)][0]), host_bufs[(p, part)][1])
                            for part in NAMES
                        ]
                        for p in PROJECTIONS
                    }
        if fetched is None:
            reader = (
                requant.Reader(self.base.path, self.checkpoint.bits, self.checkpoint.group_size)
                if isinstance(self.checkpoint, requant.Flat) else self.base
            )
            fetched = ({p: reader.experts(layer, p, to_read) for p in PROJECTIONS} if to_read.size else None)
        with self._seq_lock:
            self.prefill_read_s += time.perf_counter() - t_read
        fetch_map = {int(e): idx for idx, e in enumerate(to_read)}
        slot_arrays = {p: [[], [], []] for p in PROJECTIONS}
        for e in new_exps:
            e_int = int(e)
            if e_int in last_map:
                j = last_map[e_int]
                for p in PROJECTIONS:
                    lp = source_projs[p]
                    arrs = (
                        lp.weight[j:j + 1],
                        lp.scales[j:j + 1] if getattr(lp, 'scales', None) is not None else None,
                        lp.biases[j:j + 1] if getattr(lp, 'biases', None) is not None else None,
                    )
                    for idx_part, val in enumerate(arrs):
                        slot_arrays[p][idx_part].append(val)
            else:
                j = fetch_map[e_int]
                for p in PROJECTIONS:
                    for idx_part, a in enumerate(fetched[p]):
                        slot_arrays[p][idx_part].append(None if a is None else a[j:j + 1])
        stacked = {
            p: [
                None if slot_arrays[p][idx_part][0] is None
                else (slot_arrays[p][idx_part][0] if new_exps.size == 1
                      else mx.concatenate(slot_arrays[p][idx_part], axis=0))
                for idx_part in range(len(NAMES))
            ]
            for p in PROJECTIONS
        }
        if stm is not None and stm.started:
            stm.wait_layer_idle(layer)
        written = switch.place_many(gone[:new_exps.size], new_exps, stacked)
        if written:
            mx.eval(written)
        del stacked, written, fetched

    def _release_prefill_buffers(self):
        from . import cuda_ple
        self._prefill_active = None
        for _, np_view, adv in self._prefill_stage.values():
            if adv:
                cuda_ple.advise_host_staging(np_view, enable=False)
        self._prefill_stage.clear()
        self._prefill_dev.clear()
        self.base.forget()
        mx.clear_cache()
        self._room_back()

    def end_layer_prefill(self, switch, ok=True):
        """Finish a chunked layer prefill pass: update `switch`'s resident bank from the
        layer-wide expert weights and release layer/prompt temporary buffers."""
        with self._prefill_lock:
            with self._layer_locks[switch.layer]:
                active = self._prefill_active
                self._prefill_active = None
                if active is not None:
                    act_layer, projs, temp, seen_set = active
                    if ok and act_layer == switch.layer and seen_set:
                        seen_arr = np.array(sorted(seen_set), dtype=np.int64)
                        self._update_bank_after_prefill(switch, seen_arr, temp.experts, projs)
                    del active, projs, temp, seen_set
                if getattr(self, '_stm', None) is not None:
                    self._stm.layer_done(switch.layer)
                if switch.layer == self.moe_indices[-1]:
                    self._release_prefill_buffers()
                else:
                    mx.clear_cache()

    def _read_one(self, layer, expert, worker_slot):
        with self._stream_lock:
            self._stream_threads.add(threading.get_ident())
        out = {}
        for p in PROJECTIONS:
            buf = self.buffers[(layer, p)][worker_slot]
            out[p] = self.checkpoint.expert(layer, p, int(expert), buf, cache=True)
        return expert, out

    def compute(self, switch, x, indices, scores=None):
        is_decode = not (x.ndim == 3 and x.shape[1] > DECODE_BLOCK_SIZE and indices.size >= 64)
        if not is_decode or prefill.in_layer():
            with self._prefill_lock:
                with self._layer_locks[switch.layer]:
                    return self._compute_locked(switch, x, indices, scores, is_decode=is_decode)
        with self._layer_locks[switch.layer]:
            res = self._compute_locked(switch, x, indices, scores, is_decode=is_decode, allow_prefill=False)
        if isinstance(res, tuple) and len(res) == 2 and res[0] is _NEED_PREFILL:
            with self._prefill_lock:
                with self._layer_locks[switch.layer]:
                    return self._compute_locked(
                        switch, x, indices, scores, is_decode=is_decode, precomputed_needed=res[1],
                    )
        return res

    def _compute_locked(self, switch, x, indices, scores=None, *, is_decode=True, allow_prefill=True, precomputed_needed=None):
        from . import native_reads
        layer = switch.layer
        stm = getattr(self, '_stm', None)
        if precomputed_needed is None:
            if is_decode and stm is not None and (layer == self.moe_indices[0] or stm.pass_num == 0):
                with self._seq_lock:
                    self._pass_seq += 1
                    pseq = self._pass_seq
                stm.begin_pass(pseq, with_token=True)
            if is_decode and layer == self.moe_indices[0]:
                self._step_cpu()

            host = self._host()
            pre_stepped = False
            if (is_decode and not prefill.in_layer() and stm is not None
                    and stm.can_serve_reads() and not native_reads.checking()
                    and host is None and not self._has_layer_inflight(layer)):
                status, st_needed = stm.step_decode(
                    layer, switch, indices, scores, self.weight[layer], DECAY,
                )
                if status == 0:
                    out = expert_qmv.decode(switch, x, switch._local_indices(indices), scores if switch.weighs else None)
                    if layer == self.moe_indices[-1]:
                        self._finish_decode_step()
                        out = stm.after(out)
                        stm.end_pass()
                    else:
                        out = stm.after_layer(out)
                    return out
                needed = st_needed
                pre_stepped = True

            if not pre_stepped:
                if scores is None:
                    ids = np.array(indices).reshape(-1)
                    w_scores = np.ones_like(ids, dtype=np.float64)
                else:
                    scores_f32 = scores.astype(mx.float32)
                    mx.eval(indices, scores_f32)
                    ids = np.array(indices, copy=False).reshape(-1)
                    w_scores = np.array(scores_f32, copy=False).reshape(-1).astype(np.float64)

                self.weight[layer] *= DECAY
                np.add.at(self.weight[layer], ids, w_scores)

                if prefill.in_layer() and self._prefill_active is not None:
                    if self._prefill_active[0] == layer:
                        self._mark_prefill()
                        _, _, temp, seen_set = self._prefill_active
                        seen_set.update(int(e) for e in np.unique(ids))
                        y = expert_qmv.stock(temp, x, temp._local_indices(indices), scores if switch.weighs else None)
                        mx.eval(y)
                        return y
                    self._prefill_active = None

                needed = np.unique(ids)
        else:
            needed = precomputed_needed
        if not is_decode or needed.size > self.slots:
            if not allow_prefill:
                return (_NEED_PREFILL, needed)
            if not is_decode:
                self._mark_prefill()
            if needed.size > self.slots:
                cap = self._prefill_cap(layer)
                in_layer_mode = prefill.in_layer() and self.num_experts <= cap
                nxt = self.next_moe.get(layer)
                host = self._host()
                if (in_layer_mode and nxt is not None and self._has_tensor_table()
                        and not (host is not None and host.holds_layer(nxt))):
                    self.base.ahead([
                        (self.base.name(nxt, p, part), None)
                        for p in PROJECTIONS for part in NAMES if self.base.has(nxt, p, part)
                    ])
                if in_layer_mode:
                    all_ids = np.arange(self.num_experts, dtype=np.int64)
                    projs = self._load_prefill_projs(layer, switch, all_ids, self.num_experts)
                    temp = DemandSwitchGLU(
                        layer, projs['gate_proj'], projs['up_proj'], projs['down_proj'],
                        all_ids, self.num_experts, switch.activation, None,
                    )
                    self._prefill_active = (layer, projs, temp, set(int(e) for e in needed))
                    with self._seq_lock:
                        self.prefill_layers += 1
                    y = expert_qmv.stock(temp, x, temp._local_indices(indices), scores if switch.weighs else None)
                    mx.eval(y)
                    return y

                present = needed[switch.route[needed] >= 0]
                missing = needed[switch.route[needed] < 0]
                batches = [missing[k:k + cap] for k in range(0, missing.size, cap)]
                y = None
                last_batch, last_projs = None, None
                if present.size == 0 and len(batches) == 1:
                    batch = batches[0]
                    cap_rows = min(
                        cap,
                        min(self.num_experts, max(int(batch.size), ((int(batch.size) + 31) // 32) * 32)),
                    )
                    batch_projs = self._load_prefill_projs(layer, switch, batch, cap_rows)
                    temp = DemandSwitchGLU(
                        layer, batch_projs['gate_proj'], batch_projs['up_proj'], batch_projs['down_proj'],
                        batch, self.num_experts, switch.activation, None,
                    )
                    y = expert_qmv.stock(temp, x, temp._local_indices(indices), scores if switch.weighs else None)
                    mx.eval(y)
                    last_batch, last_projs = batch, batch_projs
                    del temp
                else:
                    indices_np = np.array(indices)
                    inside_p = None
                    if present.size > 0:
                        inside_p = switch.route[indices_np] >= 0
                        stand_in = mx.where(mx.array(inside_p), switch._local_indices(indices),
                                            mx.zeros_like(indices, dtype=mx.uint32))
                        y = expert_qmv.stock(switch, x, stand_in, scores if switch.weighs else None)
                        mx.eval(y)
                    for k, batch in enumerate(batches):
                        cap_rows = min(
                            cap,
                            min(self.num_experts, max(int(batch.size), ((int(batch.size) + 31) // 32) * 32)),
                        )
                        batch_projs = self._load_prefill_projs(layer, switch, batch, cap_rows)
                        temp = DemandSwitchGLU(
                            layer, batch_projs['gate_proj'], batch_projs['up_proj'], batch_projs['down_proj'],
                            batch, self.num_experts, switch.activation, None,
                        )
                        inside = (~inside_p) if (len(batches) == 1 and inside_p is not None) else np.isin(indices_np, batch)
                        stand_in = mx.where(mx.array(inside), temp._local_indices(indices),
                                            mx.zeros_like(indices, dtype=mx.uint32))
                        part = expert_qmv.stock(temp, x, stand_in, scores if switch.weighs else None)
                        y = part if y is None else mx.where(mx.array(inside)[..., None], part, y)
                        del part
                        mx.eval(y)
                        if k + 1 == len(batches):
                            last_batch, last_projs = batch, batch_projs
                        else:
                            del batch_projs
                        del temp
                with self._seq_lock:
                    self.prefill_layers += 1

                self._update_bank_after_prefill(switch, needed, last_batch, last_projs)
                del last_batch, last_projs
                if layer == self.moe_indices[-1] and not prefill.in_layer():
                    self._release_prefill_buffers()
                if is_decode and stm is not None:
                    if layer == self.moe_indices[-1]:
                        y = stm.after(y)
                        stm.end_pass()
                    else:
                        y = stm.after_layer(y)
                return y

        active_ids = needed
        if is_decode and stm is not None:
            stm.begin_layer(layer)

        present_mask = switch.route[active_ids] >= 0
        missing = active_ids[~present_mask]
        with self._seq_lock:
            self.hits += int(present_mask.sum())
            self.misses += int(missing.size)

        if missing.size:
            t0 = time.perf_counter()
            host = self._host()
            in_host = host.held_in(layer, missing) if host is not None else missing[:0]
            if is_decode and in_host.size:
                out = self._cpu_offload(
                    layer, switch, x, indices, scores, ids, active_ids, present_mask, missing, host, in_host
                )
                if out is not None:
                    with self._seq_lock:
                        self.read_s += time.perf_counter() - t0
                    if layer == self.moe_indices[-1]:
                        self._finish_decode_step()
                        if stm is not None:
                            out = stm.after(out)
                            stm.end_pass()
                    elif stm is not None:
                        out = stm.after_layer(out)
                    return out
                present_mask = switch.route[active_ids] >= 0
                missing = active_ids[~present_mask]
                in_host = host.held_in(layer, missing) if host is not None else missing[:0]
            if in_host.size:
                missing = np.concatenate([in_host, missing[~np.isin(missing, in_host)]])
            free_mask = np.ones(switch.experts.size, dtype=bool)
            if present_mask.any():
                free_mask[switch.route[active_ids[present_mask]]] = False
            cand = np.flatnonzero(free_mask)
            victims = cand[np.argsort(self.weight[layer][switch.experts[cand]], kind='stable')[:missing.size]]
            self._load_and_place(layer, switch, victims, in_host, missing[in_host.size:], in_pass=is_decode)
            if in_host.size:
                with self._seq_lock:
                    self.host_served += int(in_host.size)
            elapsed = time.perf_counter() - t0
            with self._seq_lock:
                if is_decode:
                    self.read_s += elapsed
                else:
                    self.prefill_read_s += elapsed

        local = switch._local_indices(indices)
        weights = scores if switch.weighs else None
        if is_decode:
            out = expert_qmv.decode(switch, x, local, weights)
            if layer == self.moe_indices[-1]:
                self._finish_decode_step()
                if stm is not None:
                    out = stm.after(out)
                    stm.end_pass()
            elif stm is not None:
                out = stm.after_layer(out)
            return out
        out = expert_qmv.stock(switch, x, local, weights)
        if layer == self.moe_indices[-1]:
            self._room_back()
        return out

    def summary(self, flush=None):
        with self._cpu_status_lock:
            eval_tid = self._cpu_eval_tid
        if flush is None:
            flush = (eval_tid is not None and threading.get_ident() == eval_tid)
        if flush:
            self._flush_cpu_counts()
        with self._cpu_status_lock:
            cpu = self._cpu
            cpu_served = self.cpu_served
            cpu_late = self.cpu_late
            cpu_wait_ms = round(self.cpu_wait_ms, 2)
            cpu_why = self._cpu_why
            server_counts = cpu.server.counts() if (cpu is not None and cpu.server is not None) else None
        with self._seq_lock:
            hits = self.hits
            misses = self.misses
            prefill_layers = self.prefill_layers
            prefill_read_s = self.prefill_read_s
            read_s = self.read_s
            host_served = self.host_served
            trimmed = self.trimmed
            native_reads = self.native_reads
        total = hits + misses
        out = {
            'slots': self.slots,
            'num_experts': self.num_experts,
            'hits': hits,
            'misses': misses,
            'hit_rate': round(hits / total, 4) if total else 1.0,
            'prefill_layers': prefill_layers,
            'prefill_read_s': round(prefill_read_s, 2),
            'read_s': round(read_s, 2),
            'host_served': host_served,
            'trimmed': trimmed,
        }
        if native_reads or self._native is not None:
            out['native_reads'] = native_reads
        if cpu is not None or cpu_served or cpu_why is not None:
            out['cpu_served'] = cpu_served
            out['cpu_late'] = cpu_late
            out['cpu_wait_ms'] = cpu_wait_ms
            if server_counts is not None:
                out['cpu_requests'] = server_counts['requests']
                out['cpu_experts'] = server_counts['experts']
                out['cpu_failed'] = server_counts['failed']
                out['cpu_ms'] = round(server_counts['compute_ns'] / 1e6, 2)
            if cpu_why is not None:
                out['cpu_off'] = cpu_why
        stm = getattr(self, '_stm', None)
        if stm is not None:
            out['stm'] = stm.summary()
        elif self._stm_why is not None:
            from . import stm_core
            m = stm_core.mode()
            out['stm'] = {'mode': m, 'asked': m, 'started': False, 'serving': False, 'why': self._stm_why}
        return out

    def close(self):
        from . import cuda_ple
        _LIVE.discard(self)
        self._close_cpu()
        stm = getattr(self, '_stm', None)
        if stm is not None:
            stm.close()
            self._stm = None
        self._prefill_active = None
        for _, np_view, adv in self._prefill_stage.values():
            if adv:
                cuda_ple.advise_host_staging(np_view, enable=False)
        self._prefill_stage.clear()
        self._prefill_dev.clear()
        for k, entry in enumerate(self._stage_inflight):
            if entry is not None:
                try:
                    mx.eval(entry[1])
                except Exception:
                    pass
                self._stage_inflight[k] = None
        for ring_map in self._stage_ring:
            for _, np_view, adv in ring_map.values():
                if adv:
                    cuda_ple.advise_host_staging(np_view, enable=False)
            ring_map.clear()
        for np_view in self._advised_views:
            cuda_ple.advise_host_staging(np_view, enable=False)
        self._advised_views.clear()
        if hasattr(self.base, 'close'):
            self.base.close()
        else:
            self.base.forget()
        clear = getattr(mx, 'clear_streams', None)
        with self._stream_lock:
            n = len(self._stream_threads)
            self._stream_threads.clear()
        if clear is not None and n > 0:
            barrier = threading.Barrier(n, timeout=1.0)

            def _release():
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    pass
                clear()

            futs = []
            for _ in range(n):
                try:
                    futs.append(self._pool.submit(_release))
                except RuntimeError:
                    break
            for fut in futs:
                try:
                    fut.result(timeout=2.0)
                except Exception:
                    pass
        self._pool.shutdown(wait=True, cancel_futures=True)
        if self._native is not None:
            self._native.close()
            self._native = None
