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

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.linear import DECODE_BLOCK_SIZE

from . import arch, expert_qmv, q2_experts, requant
from .q2_experts import NAMES, PROJECTIONS

DECAY = 0.999


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

            if isinstance(self.mode, str) and self.mode.startswith('gguf:'):
                from . import gguf_qmv
                if gguf_qmv.takes(self, x, indices):
                    return gguf_qmv.gather(self, x, indices, unpacked)
                if gguf_qmv.takes_rows(self, x, indices):
                    return gguf_qmv.gather_rows(self, x, indices, unpacked)
            from . import kquant_qmv
            if self.mode in kquant_qmv.MODES:
                if kquant_qmv.takes(self, x, indices):
                    return kquant_qmv.gather(self, x, indices, unpacked)
                if kquant_qmv.takes_rows(self, x, indices):
                    return kquant_qmv.gather_rows(self, x, indices, unpacked)
            return unpacked()
        return mx.gather_qmm(x, self.weight, self.scales, self.get('biases'), rhs_indices=indices, transpose=True,
                             group_size=self.group_size, bits=self.bits, mode=self.mode,
                             sorted_indices=sorted_indices)


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
        self._local = mx.array(np.maximum(self.route, 0).astype(np.uint32))
        self.freeze()

    def place(self, slot, expert, arrays):
        """Write `arrays` (`{proj: [weight, scales, biases]}`) for `expert` into `slot`."""
        old = int(self.experts[slot])
        if old >= 0 and self.route[old] == slot:
            self.route[old] = -1
        self.experts[slot] = expert
        self.route[expert] = slot
        idx = mx.array([slot], dtype=mx.int32)
        written = []
        for p in PROJECTIONS:
            proj = getattr(self, p)
            w, s, b = arrays[p]
            proj.weight[idx] = w
            written.append(proj.weight)
            if s is not None and getattr(proj, 'scales', None) is not None:
                proj.scales[idx] = s
                written.append(proj.scales)
            if b is not None and getattr(proj, 'biases', None) is not None:
                proj.biases[idx] = b
                written.append(proj.biases)
        self._local = mx.array(np.maximum(self.route, 0).astype(np.uint32))
        return written

    def replace_all(self, experts, arrays_by_proj):
        """Replace the entire bank with `experts` and their stacked `[weight, scales, biases]`."""
        self.experts = np.array(experts, dtype=np.int64, copy=True)
        self.route.fill(-1)
        self.route[self.experts] = np.arange(self.experts.size)
        written = []
        for p in PROJECTIONS:
            proj = getattr(self, p)
            w, s, b = arrays_by_proj[p]
            proj.weight = w
            written.append(proj.weight)
            if s is not None:
                proj.scales = s
                written.append(proj.scales)
            if b is not None:
                proj.biases = b
                written.append(proj.biases)
        self._local = mx.array(np.maximum(self.route, 0).astype(np.uint32))
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
    eff_bits = ggml_quants.FORMAT_BITS[bits_spec] if bits_spec in requant.GGML_MODES else int(bits_spec)
    in_dims = w.shape[2] * 32 // eff_bits
    return eff_bits, in_dims // s.shape[-1], 'affine'


class Demand:
    """On-demand expert streaming manager across all MoE layers of `model`."""

    def __init__(self, model, checkpoint, budget_bytes, workers=4, prompt_room=None):
        from . import cuda_ple, fitting
        self.checkpoint = checkpoint
        self.base = checkpoint.base if isinstance(checkpoint, requant.Flat) else checkpoint
        self.workers = max(1, int(workers))
        self.prompt_room = None if not prompt_room else float(prompt_room)
        self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix='demand-read')
        self._stream_threads = set()
        self._stream_lock = threading.Lock()
        self._free_least = cuda_ple.free_least()
        self._decode_steps = 0
        self._after_prefill = False
        self.trimmed = 0

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
                    out_d * in_d, eff_mode if eff_mode in fitting.GGML_MODES else eff_b
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
        self.buffers = {
            (l, p): [self.checkpoint.buffers(l, p)
                     for _ in range(self.workers)]
            for l in moe_indices for p in PROJECTIONS
        }
        self.switches = {}
        self.hits = 0
        self.misses = 0
        self.prefill_layers = 0
        self.read_s = 0.0
        self.host_served = 0

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
                projs[p] = Projection(w, s, b, p_gs, p_bits, mode=p_mode)
                mx.eval(projs[p].parameters())
            d_switch = DemandSwitchGLU(
                l, projs['gate_proj'], projs['up_proj'], projs['down_proj'],
                init_ids, num_experts, old_switch.activation, self,
            )
            arch.set_switch(layers[l], d_switch)
            self.switches[l] = d_switch
            mx.clear_cache()

    def _room_back(self):
        if self._free_least is not None:
            from . import cuda_ple
            if cuda_ple.room_back(self._free_least):
                self.trimmed += 1

    def _host(self):
        """The host's copy of the experts by rows (q2_experts.HostCopy with `by_rows=True`), or None."""
        if isinstance(self.checkpoint, requant.Flat):
            return None
        held = self.base.held()
        return held if held is not None and held.by_rows else None

    def _read_one(self, layer, expert, worker_slot):
        with self._stream_lock:
            self._stream_threads.add(threading.get_ident())
        out = {}
        for p in PROJECTIONS:
            buf = self.buffers[(layer, p)][worker_slot]
            out[p] = self.checkpoint.expert(layer, p, int(expert), buf, cache=True)
        return expert, out

    def compute(self, switch, x, indices, scores=None):
        layer = switch.layer
        is_decode = not (x.ndim == 3 and x.shape[1] > DECODE_BLOCK_SIZE and indices.size >= 64)
        ids = np.array(indices).reshape(-1)
        w_scores = (np.ones_like(ids, dtype=np.float64) if scores is None
                    else np.array(scores.astype(mx.float32)).reshape(-1).astype(np.float64))

        self.weight[layer] *= DECAY
        np.add.at(self.weight[layer], ids, w_scores)

        if not is_decode:
            self._after_prefill = True
            self._decode_steps = 0
            needed = np.unique(ids)
            missing = needed[switch.route[needed] < 0]
            if needed.size > self.slots:
                nxt = self.next_moe.get(layer)
                if nxt is not None and not isinstance(self.checkpoint, requant.Flat):
                    self.base.ahead([
                        (self.base.name(nxt, p, part), None)
                        for p in PROJECTIONS for part in NAMES if self.base.has(nxt, p, part)
                    ])
                reader = (requant.Reader(self.base.path, self.checkpoint.bits, self.checkpoint.group_size)
                          if isinstance(self.checkpoint, requant.Flat) else self.base)
                cap = needed.size if not self.prompt_room else max(1, int(self.prompt_room // self.per_expert))
                batches = [needed[k:k + cap] for k in range(0, needed.size, cap)]
                indices_np = np.array(indices) if len(batches) > 1 else None
                y = None
                last_batch, last_projs = None, None
                for k, batch in enumerate(batches):
                    t_read = time.perf_counter()
                    batch_projs = {}
                    for p in PROJECTIONS:
                        w, s, b = reader.experts(layer, p, batch)
                        ref = getattr(switch, p)
                        batch_projs[p] = Projection(w, s, b, ref.group_size, ref.bits, mode=ref.mode)
                    mx.eval([a for pr in batch_projs.values()
                             for a in (pr.weight, pr.get('scales'), pr.get('biases')) if a is not None])
                    self.read_s += time.perf_counter() - t_read
                    temp = DemandSwitchGLU(
                        layer, batch_projs['gate_proj'], batch_projs['up_proj'], batch_projs['down_proj'],
                        batch, self.num_experts, switch.activation, None,
                    )
                    if len(batches) == 1:
                        y = expert_qmv.stock(temp, x, temp._local[indices], scores if switch.weighs else None)
                    else:
                        inside = np.isin(indices_np, batch)
                        stand_in = mx.where(mx.array(inside), temp._local[indices],
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
                self.prefill_layers += 1

                candidates = np.union1d(switch.experts, needed)
                top = np.sort(candidates[np.argsort(-self.weight[layer][candidates], kind='stable')[:self.slots]])
                new_exps = top[switch.route[top] < 0]
                if new_exps.size:
                    top_set = set(int(t) for t in top)
                    gone = [s for s, e in enumerate(switch.experts) if int(e) not in top_set]
                    last_map = {int(e): idx for idx, e in enumerate(last_batch)} if last_batch is not None else {}
                    to_read = np.array([int(e) for e in new_exps if int(e) not in last_map], dtype=np.int64)
                    t_read = time.perf_counter()
                    fetched = ({p: reader.experts(layer, p, to_read) for p in PROJECTIONS}
                               if to_read.size else None)
                    self.read_s += time.perf_counter() - t_read
                    fetch_map = {int(e): idx for idx, e in enumerate(to_read)}
                    written = []
                    for slot, e in zip(gone, new_exps):
                        e_int = int(e)
                        if e_int in last_map:
                            j = last_map[e_int]
                            arrays = {
                                p: [
                                    last_projs[p].weight[j:j + 1],
                                    last_projs[p].scales[j:j + 1]
                                    if getattr(last_projs[p], 'scales', None) is not None else None,
                                    last_projs[p].biases[j:j + 1]
                                    if getattr(last_projs[p], 'biases', None) is not None else None,
                                ]
                                for p in PROJECTIONS
                            }
                        else:
                            j = fetch_map[e_int]
                            arrays = {
                                p: [None if a is None else a[j:j + 1] for a in fetched[p]]
                                for p in PROJECTIONS
                            }
                        written.extend(switch.place(slot, e_int, arrays))
                    if written:
                        mx.eval(written)
                del last_batch, last_projs
                mx.clear_cache()
                if layer == self.moe_indices[-1]:
                    self._room_back()
                return y

        active_ids = np.unique(ids)

        present_mask = switch.route[active_ids] >= 0
        self.hits += int(present_mask.sum())
        missing = active_ids[~present_mask]
        self.misses += int(missing.size)

        if missing.size:
            t0 = time.perf_counter()
            host = self._host()
            in_host = host.held_in(layer, missing) if host is not None else missing[:0]
            if in_host.size:
                missing = np.concatenate([in_host, missing[~np.isin(missing, in_host)]])
            copied = {p: self.base.experts(layer, p, in_host) for p in PROJECTIONS} if in_host.size else None
            in_use = set(int(e) for e in active_ids if switch.route[e] >= 0)
            candidates = [s for s, e in enumerate(switch.experts) if int(e) not in in_use]
            candidates.sort(key=lambda s: self.weight[layer][switch.experts[s]])
            victims = candidates[:missing.size]

            written = []
            for k in range(in_host.size):
                slot = victims[k]
                e = int(missing[k])
                arrays = {p: [None if a is None else a[k:k + 1] for a in copied[p]] for p in PROJECTIONS}
                written.extend(switch.place(slot, e, arrays))
            disk_victims = victims[in_host.size:]
            disk_missing = missing[in_host.size:]
            for start in range(0, len(disk_victims), self.workers):
                batch_slots = disk_victims[start:start + self.workers]
                batch_exps = disk_missing[start:start + len(batch_slots)]
                futures = [
                    self._pool.submit(self._read_one, layer, e, w_idx)
                    for w_idx, e in enumerate(batch_exps)
                ]
                for slot, fut in zip(batch_slots, futures):
                    e, arrays = fut.result()
                    written.extend(switch.place(slot, int(e), arrays))
                if written:
                    mx.eval(written)
                    written.clear()
            if written:
                mx.eval(written)
            if in_host.size:
                self.host_served += int(in_host.size)
            self.read_s += time.perf_counter() - t0

        local = switch._local[indices]
        weights = scores if switch.weighs else None
        if is_decode:
            out = expert_qmv.decode(switch, x, local, weights)
            if layer == self.moe_indices[-1]:
                self._decode_steps += 1
                if self._after_prefill or self._decode_steps % 64 == 0:
                    self._after_prefill = False
                    self._room_back()
            return out
        out = expert_qmv.stock(switch, x, local, weights)
        if layer == self.moe_indices[-1]:
            self._room_back()
        return out

    def summary(self):
        total = self.hits + self.misses
        return {
            'slots': self.slots,
            'num_experts': self.num_experts,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': round(self.hits / total, 4) if total else 1.0,
            'prefill_layers': self.prefill_layers,
            'read_s': round(self.read_s, 2),
            'host_served': self.host_served,
            'trimmed': self.trimmed,
        }

    def close(self):
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
