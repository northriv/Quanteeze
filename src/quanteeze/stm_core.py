# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""C++20 KAME STM coordination driver for QuantEEze's on-demand experts (QUANTEEZE_DEMAND_STM).

Binds the portable C++ STM core (vendor/stm_core/stm_core.cpp) into Demand:
- Maintains per-layer STM placement snapshots (route, absent mask, resident slot table).
- Tracks active GPU pass leases per layer (`begin_layer`, `pass_token`, `mx.depends`, `completed`)
  so asynchronous weight landings wait until no GPU command buffer reads that layer.
- Connects native pread readers (`NativeReaders.c_entrypoints()`), lock-free host pool memory
  (`allocator.host_alloc_addrs()`), and optional CUDA driver async HtoD copy handles.
- Supports `on` (full STM + lander ready), `shadow` (mirror placements & verify snapshots),
  and `off` (disabled).
"""
import ctypes
import os
import signal
import sys
import threading

import numpy as np

from . import native_ext, variables

_SAID = set()
TURN_PROMPT = 0
TURN_DECODE = 1
TURN_TIMEOUT_MS = 2000.0


def _warn_once(key, msg):
    if key not in _SAID:
        _SAID.add(key)
        print(f'quanteeze: {msg}', file=sys.stderr, flush=True)


def mode():
    """Return QUANTEEZE_DEMAND_STM normalized to 'auto', 'on', 'shadow', or 'off'."""
    raw = (variables.get('QUANTEEZE_DEMAND_STM') or 'auto').strip().lower()
    if raw in ('auto', 'on', 'shadow', 'off'):
        return raw
    return 'auto'


def enabled():
    """Whether QUANTEEZE_DEMAND_STM is active ('auto', 'on', or 'shadow')."""
    return mode() in ('auto', 'on', 'shadow')


def extension(quiet=False):
    """Build and load the `_stm_core` C++20 extension, or return None if unavailable."""
    return native_ext.extension('stm_core', with_mlx=True, with_stm=True, quiet=quiet)


def stop_if_write_failed(failed):
    """Abort serving gracefully if the CUDA lander left partial rows un-committed."""
    if failed <= 0:
        return
    why = f'STM lander CUDA copy failed ({failed}): stopping server'
    print(f'quanteeze: {why}', file=sys.stderr, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
    raise RuntimeError(f'quanteeze: {why}')


def attach_cuda(model):
    """Pass CUDA driver context & stream copy function pointers into `model`; returns True if attached."""
    import mlx.core as mx
    from . import cuda_kernels, cuda_ple
    if mx.metal.is_available() or not cuda_kernels.available():
        return False
    drv = cuda_ple._driver()
    if drv is None:
        return False
    cuda, context = drv
    addr = lambda f: ctypes.cast(f, ctypes.c_void_p).value  # noqa: E731
    model.attach_cuda(
        int(context.value),
        addr(cuda.cuCtxPushCurrent_v2),
        addr(cuda.cuCtxPopCurrent_v2),
        addr(cuda.cuStreamCreate),
        addr(cuda.cuMemcpyAsync),
        addr(cuda.cuStreamSynchronize),
        addr(cuda.cuStreamDestroy_v2),
    )
    return True


def attach_host(model, pinned=False):
    """Attach host staging allocator (kamepoolalloc if active, else aligned_alloc/free) + optional CUDA pin."""
    from . import allocator
    alloc_addr, free_addr = allocator.host_alloc_addrs()
    if not alloc_addr or not free_addr:
        libc = ctypes.CDLL(None)
        addr = lambda f: ctypes.cast(f, ctypes.c_void_p).value  # noqa: E731
        alloc_addr = addr(libc.aligned_alloc)
        free_addr = addr(libc.free)
    reg = unreg = dev = 0
    if pinned:
        from . import cuda_ple
        drv = cuda_ple._driver()
        if drv is not None:
            cuda, _ = drv
            addr = lambda f: ctypes.cast(f, ctypes.c_void_p).value  # noqa: E731
            reg = addr(cuda.cuMemHostRegister_v2)
            unreg = addr(cuda.cuMemHostUnregister)
            dev = addr(cuda.cuMemHostGetDevicePointer_v2)
    model.attach_host(int(alloc_addr), int(free_addr), int(reg), int(unreg), int(dev))
    return True


def attach_readers(model, readers):
    """Attach NativeReaders C function pointers to `model`; returns True if attached."""
    if readers is None:
        return False
    ep = readers.c_entrypoints()
    if ep is None:
        return False
    if isinstance(ep, dict):
        model.attach_readers(
            int(ep['handle']),
            int(ep['submit']),
            int(ep['status']),
            int(ep['release']),
            int(ep['times']),
            int(ep['backlog']),
            int(readers.workers),
        )
    else:
        handle, submit, status, release, times, backlog, workers = ep
        model.attach_readers(
            int(handle), int(submit), int(status), int(release),
            int(times), int(backlog), int(workers),
        )
    return True


def _ints(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def _bools(a):
    return np.ascontiguousarray(a, dtype=bool)


class _DriverPass(threading.local):
    """Per-thread pass lease state inside CoreDriver."""

    def __init__(self):
        self.token = None
        self._prev_out = None
        self._layer_out = None
        self._marked = {}
        self._version = {}
        self._gen = 0
        self.pass_num = 0


def _thread_own(name):
    return property(lambda self: getattr(self._tls, name), lambda self, v: setattr(self._tls, name, v))


class CoreDriver:
    """Portable STM driver bound to a `Demand` instance (on-demand MoE layers)."""

    TIMEOUT_MS = 5000.0
    STATS = (
        'stm_writes', 'stm_reads', 'stm_checks', 'stm_mismatched',
        'stm_landed', 'stm_not_fit', 'stm_failed', 'stm_write_failed',
    )

    def __init__(self, demand, ext, *, mode='on'):
        self.demand = demand
        self.ext = ext
        self.mode = mode
        self.index = {i: k for k, i in enumerate(sorted(demand.switches))}
        self.model = ext.Model(len(self.index))
        self.hints = self.model.hints
        self.started = False
        self.can_land = False
        self.why = None
        self._quant_synced = set()
        self._tls = _DriverPass()
        self._epoch = 0
        self._lock = threading.Lock()
        self.stats = {k: 0 for k in self.STATS}
        self._landings = (0, 0, 0)

        for i, k in self.index.items():
            s = demand.switches[i]
            ver = self.model.put_placement(
                k, _ints(s.route), _bools(s.route < 0), _ints(s.experts)
            )
            object.__setattr__(s, '_stm_read', ver)
            self.stats['stm_writes'] += 1

        if mode in ('auto', 'on'):
            self._start()

    def _register_layer_quants(self, layer):
        from .q2_experts import PROJECTIONS
        d = self.demand
        k = self.index[layer]
        s = d.switches[layer]
        for q, pname in enumerate(PROJECTIONS):
            proj = getattr(s, pname)
            w = proj.weight
            sc = getattr(proj, 'scales', None)
            bi = getattr(proj, 'biases', None)
            arrays = (w, sc, bi)
            indices = tuple(q * 3 + m if a is not None else -1 for m, a in enumerate(arrays))
            self.model.register_quant(
                k, q, w, sc, bi,
                int(getattr(proj, 'bits', 0) or 0),
                int(getattr(proj, 'group_size', 0) or 0),
                str(getattr(proj, 'mode', 'affine') or 'affine'),
                indices,
            )
        self._quant_synced.add(layer)

    def _ensure_quant(self, layer):
        if layer not in self._quant_synced and self.started:
            with self._lock:
                if layer not in self._quant_synced and self.started:
                    self._register_layer_quants(layer)

    def _start(self):
        import mlx.core as mx
        from . import cuda_ple, native_reads, requant
        from .q2_experts import NAMES, PROJECTIONS, _SAFETENSORS
        d = self.demand
        cuda = attach_cuda(self.model)
        attach_host(self.model, pinned=cuda)
        native = getattr(d, '_native', None)
        if native is not None:
            attach_readers(self.model, native)
        base = d.base
        file_fn = getattr(native, 'file_id', None) or getattr(native, '_file', None) if native is not None else None
        has_tensors = (
            native is not None
            and native_reads.enabled()
            and file_fn is not None
            and not isinstance(d.checkpoint, requant.Flat)
            and hasattr(base, 'tensors') and hasattr(base, 'has') and hasattr(base, 'name')
        )
        all_registered = bool(has_tensors)
        for i, k in self.index.items():
            self._register_layer_quants(i)
            if not has_tensors:
                continue
            s = d.switches[i]
            num_experts = int(s.route.size)
            for q, pname in enumerate(PROJECTIONS):
                proj = getattr(s, pname)
                arrays = (proj.weight, getattr(proj, 'scales', None), getattr(proj, 'biases', None))
                for m, (name, a) in enumerate(zip(NAMES, arrays)):
                    if a is None:
                        continue
                    if not base.has(i, pname, name):
                        all_registered = False
                        continue
                    tname = base.name(i, pname, name)
                    entry = base.tensors.get(tname)
                    if entry is None:
                        all_registered = False
                        continue
                    file_path, st_dtype, st_shape, offset = entry
                    row_bytes = int(a.nbytes // max(1, a.shape[0]))
                    item_np = _SAFETENSORS.get(st_dtype)
                    expected_bytes = (
                        int(np.prod(st_shape[1:], dtype=np.int64)) * np.dtype(item_np).itemsize
                        if item_np is not None else -1
                    )
                    if int(st_shape[0]) != num_experts or row_bytes <= 0 or expected_bytes != row_bytes:
                        all_registered = False
                        continue
                    if file_path is not None:
                        fid = file_fn(file_path)
                        if fid < 0:
                            all_registered = False
                            continue
                        files = np.full(num_experts, fid, dtype=np.int32)
                        offsets = np.int64(offset) + np.arange(num_experts, dtype=np.int64) * row_bytes
                    else:
                        fids = [file_fn(offset[e][0]) for e in range(num_experts)]
                        if any(fid < 0 for fid in fids):
                            all_registered = False
                            continue
                        files = np.array(fids, dtype=np.int32)
                        offsets = np.array([int(offset[e][1]) for e in range(num_experts)], dtype=np.int64)
                    self.model.register_tensor(k, q * 3 + m, files, offsets, row_bytes)
        if cuda:
            cuda_ple._while_alone(lambda: (mx.synchronize(), mx.clear_cache()))
        self.can_land = bool(has_tensors and all_registered)
        if self.can_land:
            self.why = None
        elif isinstance(d.checkpoint, requant.Flat):
            self.why = 'uniform re-quantization without disk cache (QUANTEEZE_QUANT_CACHE=off)'
        elif native is None or not native_reads.enabled():
            failed_detail = native_reads._STATE.get('failed')
            self.why = (
                f'native SSD reader verification failed ({failed_detail})'
                if failed_detail else 'native SSD readers disabled'
            )
        else:
            self.why = 'checkpoint byte segment table incomplete'
        self.model.start_lander()
        self.started = True

    def is_serving(self):
        """Return `(serving, why)` matching UI expectations (`True`/`False`/`None`)."""
        from . import native_reads
        if self.mode not in ('auto', 'on'):
            return None, None
        if not self.started:
            return False, self.why or 'STM driver not started'
        if self.demand._host() is not None:
            return False, 'host-RAM expert copy (HostCopy) active'
        if not self.can_land:
            return False, self.why or 'C++ landing disabled'
        if getattr(self.demand, '_native', None) is None or not native_reads.enabled():
            failed_detail = native_reads._STATE.get('failed')
            return False, (
                f'native SSD reader verification failed ({failed_detail})'
                if failed_detail else 'native SSD readers disabled'
            )
        return True, None

    def can_serve_reads(self):
        """True when direct C++ `land_now` and `demand_step` can handle SSD misses."""
        from . import native_reads
        return (
            self.mode in ('auto', 'on')
            and self.started
            and self.can_land
            and self.demand._host() is None
            and getattr(self.demand, '_native', None) is not None
            and native_reads.enabled()
        )

    def now_ns(self):
        """Return steady clock nanoseconds shared with native reader birth timestamps."""
        return int(self.ext.Model.now_ns())

    def turn_take(self, kind, birth=0, timeout_ms=TURN_TIMEOUT_MS):
        """Acquire GPU arbiter turn (`TURN_PROMPT` or `TURN_DECODE`), oldest birth first."""
        return int(self.model.turn_take(int(kind), int(birth), float(timeout_ms)))

    def turn_try(self, kind, birth=0):
        """Try to acquire GPU arbiter turn without waiting; returns True if acquired."""
        return bool(self.model.turn_try(int(kind), int(birth)))

    def turn_give(self):
        """Release the calling thread's GPU arbiter turn and wake waiters."""
        if self.model is not None:
            self.model.turn_give()

    def turn_counts(self):
        """Return turn arbitration counters as a dict."""
        s = self.model.turn_stats()
        return {
            'prompt_turns': int(s[0]),
            'decode_turns': int(s[1]),
            'prompt_turn_waits': int(s[2]),
            'decode_turn_waits': int(s[3]),
            'turn_wait_ms': round(s[4] / 1e6, 2),
            'turn_timeouts': int(s[5]),
        }

    def before_switch_placed(self, switch):
        """Drop C++ array references on `switch.layer` before Python mutates bank arrays."""
        self._quant_synced.discard(switch.layer)
        if self.started and self.model is not None and switch.layer in self.index:
            self.model.drop_quant(self.index[switch.layer])

    def on_switch_placed(self, switch):
        """Mirror a Python placement mutation into the layer's STM node."""
        if self.model is None or switch.layer not in self.index:
            return
        self._quant_synced.discard(switch.layer)
        k = self.index[switch.layer]
        ver = self.model.put_placement(
            k, _ints(switch.route), _bools(switch.route < 0), _ints(switch.experts)
        )
        object.__setattr__(switch, '_stm_read', ver)
        with self._lock:
            self.stats['stm_writes'] += 1

    def begin_pass(self, pass_num, *, with_token=True):
        """Start a pass on the calling thread: retire old tokens, check failures, allocate new token."""
        if self.model is None:
            return
        old_pnum = int(self.pass_num)
        prev_l = self._layer_out
        self._layer_out = None
        if prev_l is not None:
            try:
                import mlx.core as mx
                mx.eval(prev_l)
            except Exception:
                pass
        if self.started and old_pnum:
            self.model.release(old_pnum, -1)
        prev = self._prev_out
        if prev is not None:
            self._prev_out = None
            try:
                import mlx.core as mx
                mx.async_eval(prev)
            except Exception:
                pass
        self.pass_num = int(pass_num)
        self._gen += 1
        self._marked = {}
        self.token = None
        self.model.completed()
        failed = int(self.model.write_failed())
        if failed:
            with self._lock:
                self.stats['stm_write_failed'] = failed
            stop_if_write_failed(failed)
        self._sync_landing_counts()
        if self.started and with_token:
            self.token = self.model.pass_token(self.pass_num)

    def _sync_landing_counts(self):
        landed, not_fit, failed, _ = self.model.landings()
        with self._lock:
            seen = self._landings
            self.stats['stm_landed'] += landed - seen[0]
            self.stats['stm_not_fit'] += not_fit - seen[1]
            self.stats['stm_failed'] += failed - seen[2]
            self._landings = (landed, not_fit, failed)

    def begin_layer(self, layer):
        """Acquire a lease on `layer` for the calling thread's current pass and refresh placement if changed."""
        if not self.started or self.model is None or layer not in self.index:
            return
        k = self.index[layer]
        prev = self._layer_out
        if prev is not None:
            self._layer_out = None
            import mlx.core as mx
            mx.eval(prev)
        if self._marked.get(layer) != (self._epoch, self._gen):
            self.model.release(int(self.pass_num), k)
            ver = self.model.begin_layer(k, int(self.pass_num))
            self._version[layer] = ver
            self._marked[layer] = (self._epoch, self._gen)
            self.read(self.demand.switches[layer])

    def after_layer(self, y):
        """Remember the calling thread's lazy layer output so the next layer evaluates it before releasing leases."""
        self._layer_out = y
        return y

    def wait_layer_idle(self, layer, timeout_ms=None):
        """Wait until `layer` has no active pass leases or C++ writers before a Python switch mutation."""
        if not self.started or self.model is None or layer not in self.index:
            return True
        return bool(self.model.wait_idle(
            self.index[layer],
            float(self.TIMEOUT_MS if timeout_ms is None else timeout_ms),
        ))

    def read(self, switch):
        """Refresh `switch.route`, `switch.experts`, and `switch._local` if STM snapshot version advanced."""
        if self.model is None or switch.layer not in self.index:
            return False
        k = self.index[switch.layer]
        seen = self._version.pop(switch.layer, None)
        expected = int(self.hints[k]) if seen is None else int(seen)
        if switch.__dict__.get('_stm_read') == expected:
            return False
        got = self.model.placement(k)
        if got is None:
            return False
        version, route, _absent, experts = got
        import mlx.core as mx
        from .demand import _to_device
        route_arr = np.array(route, dtype=np.int64, copy=True)
        experts_arr = np.array(experts, dtype=np.int64, copy=True) if len(experts) else switch.experts
        switch.route = route_arr
        switch.experts = experts_arr
        if not getattr(switch, '_identity', False):
            switch._local = _to_device(mx.array(np.maximum(route_arr, 0).astype(np.uint32)))
        object.__setattr__(switch, '_stm_read', int(version))
        with self._lock:
            self.stats['stm_reads'] += 1
        return True

    def land_missing(self, layer, switch, slots, experts, *, in_pass=True, timeout_ms=None):
        """Read and copy `experts` into `slots` on `layer` in C++ with the GIL released."""
        if not self.can_serve_reads() or layer not in self.index or getattr(switch, '_identity', False):
            return False
        slots_vec = [int(s) for s in np.asarray(slots, dtype=np.int64).reshape(-1)]
        exp_vec = [int(e) for e in np.asarray(experts, dtype=np.int64).reshape(-1)]
        if not slots_vec:
            return True
        self._ensure_quant(layer)
        k = self.index[layer]
        pnum = int(self.pass_num) if in_pass else -1
        ok, new_ver, _read_ns, _copy_ns, landed_n = self.model.land_now(
            k, pnum, slots_vec, exp_vec,
            float(self.TIMEOUT_MS if timeout_ms is None else timeout_ms),
        )
        self._sync_landing_counts()
        if not ok:
            return False
        if in_pass:
            self._marked[layer] = (self._epoch, self._gen)
        self._version[layer] = int(new_ver)
        self.read(switch)
        lk = getattr(self.demand, '_seq_lock', None)
        if lk is not None:
            with lk:
                self.demand.native_reads += int(landed_n)
        else:
            self.demand.native_reads += int(landed_n)
        return True

    def step_decode(self, layer, switch, indices, scores, weight_arr, decay, timeout_ms=None):
        """Run decode routing, weight update, and missing-expert C++ landing without the GIL."""
        self._ensure_quant(layer)
        k = self.index[layer]
        prev_out = self._layer_out
        self._layer_out = None
        status, needed_vec, hits, misses, ver, elapsed_s = self.model.demand_step(
            k, int(self.pass_num), indices, scores, weight_arr, float(decay),
            float(self.TIMEOUT_MS if timeout_ms is None else timeout_ms),
            prev_out,
        )
        if status == 0:
            self._marked[layer] = (self._epoch, self._gen)
            self._version[layer] = int(ver)
            self.read(switch)
            lk = getattr(self.demand, '_seq_lock', None)
            if lk is not None:
                with lk:
                    self.demand.hits += int(hits)
                    self.demand.misses += int(misses)
                    if misses > 0:
                        self.demand.native_reads += int(misses)
                        self.demand.read_s += float(elapsed_s)
            else:
                self.demand.hits += int(hits)
                self.demand.misses += int(misses)
                if misses > 0:
                    self.demand.native_reads += int(misses)
                    self.demand.read_s += float(elapsed_s)
            self._sync_landing_counts()
            return 0, None
        if status == 1:
            self._marked.pop(layer, None)
            return 1, np.asarray(needed_vec, dtype=np.int64)
        self._marked[layer] = (self._epoch, self._gen)
        self._version[layer] = int(ver)
        self.read(switch)
        self._sync_landing_counts()
        return 2, np.asarray(needed_vec, dtype=np.int64)

    def after(self, y):
        """Attach the calling thread's pass token to output `y` so MLX retains it until GPU completion."""
        token = self.token
        if token is None:
            return y
        import mlx.core as mx
        out = mx.depends(y, token)
        out = out[0] if isinstance(out, (list, tuple)) else out
        self._prev_out = out
        return out

    def end_pass(self):
        """Drop the calling thread's pass token reference and verify snapshots when in shadow mode."""
        if self.model is None:
            return
        self._layer_out = None
        self.token = None
        if self.mode == 'shadow':
            self.check()

    def layer_done(self, layer):
        """Release the calling thread's lease on `layer`."""
        if self.model is None or layer not in self.index:
            return
        self._marked.pop(layer, None)
        self.model.release_layer(self.index[layer], int(self.pass_num))

    def evaluated(self, keep=None):
        """Release all layer holds held by the calling thread's current pass except `keep`."""
        if self.model is None or not self.started:
            return
        keep_k = -1 if keep is None else self.index.get(keep, -1)
        self.model.release(int(self.pass_num), keep_k)
        self._gen += 1
        if keep is not None and keep in self.index:
            self._marked[keep] = (self._epoch, self._gen)

    def thread_done(self):
        """Release any holds remaining from this thread's last pass after stream synchronization."""
        if self.model is None:
            return
        pnum = int(self.pass_num)
        self._prev_out = None
        self._layer_out = None
        self.token = None
        self._marked = {}
        self.model.release(pnum, -1)
        self.model.turn_give()
        self.model.completed()

    def check(self):
        """Verify STM placement snapshots against Python switches; returns list of mismatched layers."""
        if self.model is None:
            return []
        different = []
        for i, k in self.index.items():
            s = self.demand.switches[i]
            bits = self.model.check(k, _ints(s.route), _bools(s.route < 0), _ints(s.experts))
            if bits:
                names = [name for b, name in enumerate(self.ext.MISMATCH) if bits >> b & 1]
                different.append((i, names))
        with self._lock:
            self.stats['stm_checks'] += 1
            self.stats['stm_mismatched'] += len(different)
            mismatched = self.stats['stm_mismatched']
        if different and mismatched <= 5:
            print(f'quanteeze: STM shadow placement mismatch: {different[:4]}', file=sys.stderr, flush=True)
        return different

    def summary(self):
        """Return diagnostic summary dict for UI/monitoring without touching unsafe GPU state."""
        serving_state, why = self.is_serving()
        if self.model is None:
            return {'mode': self.mode, 'asked': self.mode, 'started': False, 'serving': serving_state, 'why': why}
        self._sync_landing_counts()
        with self._lock:
            st = dict(self.stats)
        out = {
            'mode': self.mode,
            'asked': self.mode,
            'started': self.started,
            'serving': serving_state,
            'why': why,
            'commits': int(self.model.total_commits()),
            'writes': st['stm_writes'],
            'reads': st['stm_reads'],
            'checks': st['stm_checks'],
            'mismatched': st['stm_mismatched'],
            'landed': st['stm_landed'],
            'not_fit': st['stm_not_fit'],
            'failed': st['stm_failed'],
            'write_failed': st['stm_write_failed'],
            'held_back': int(self.model.held_back()),
        }
        if self.started:
            tc = self.turn_counts()
            if tc['prompt_turns'] or tc['decode_turns']:
                out['turns'] = tc
        return out

    def room_put(self, nbytes):
        if self.model is not None:
            self.model.room_put(int(nbytes))

    def room_take(self, nbytes, overdraw=False, timeout_ms=0.0):
        if self.model is None:
            return True
        return bool(self.model.room_take(int(nbytes), bool(overdraw), float(timeout_ms)))

    def room_give(self, nbytes):
        if self.model is not None:
            self.model.room_give(int(nbytes))

    def room_free(self):
        return 0 if self.model is None else int(self.model.room_free())

    def close(self):
        """Shut down the C++ lander thread cleanly."""
        self._prev_out = None
        self._layer_out = None
        if self.model is not None:
            self.model.turn_give()
            self.model.stop_lander()
            self.started = False


for _name in ('token', '_prev_out', '_layer_out', '_marked', '_version', '_gen', 'pass_num'):
    setattr(CoreDriver, _name, _thread_own(_name))


class RoomDriver:
    """Standalone KAME STM work-room holder when concurrent requests run without an active `Demand` driver."""

    def __init__(self, ext):
        self.ext = ext
        self.model = ext.Model(0)
        self.started = True

    def room_put(self, nbytes):
        self.model.room_put(int(nbytes))

    def room_take(self, nbytes, overdraw=False, timeout_ms=0.0):
        return bool(self.model.room_take(int(nbytes), bool(overdraw), float(timeout_ms)))

    def room_give(self, nbytes):
        self.model.room_give(int(nbytes))

    def room_free(self):
        return int(self.model.room_free())


def room_driver():
    """Return a fresh standalone `RoomDriver`, or None if `_stm_core` is unavailable."""
    ext = extension(quiet=True)
    return None if ext is None else RoomDriver(ext)


def create_driver(demand):
    """Create a `CoreDriver` if QUANTEEZE_DEMAND_STM is active and `_stm_core` compiles; else None."""
    m = mode()
    if m == 'off':
        return None
    ext = extension(quiet=(m == 'auto'))
    if ext is None:
        demand._stm_why = '_stm_core extension did not build'
        if m != 'auto':
            _warn_once(('no_ext', m), f'QUANTEEZE_DEMAND_STM={m}, but _stm_core did not build: running without STM')
        return None
    return CoreDriver(demand, ext, mode=m)
