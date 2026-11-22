# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Routed MoE experts computed on the CPU from the host's copy (`q2_experts.HostCopy`).

Where a model's routed experts do not fit in GPU memory, those held in host RAM are normally copied
over PCIe on demand (`quanteeze.demand`). On multi-memory systems, reading and computing single
decode experts directly on the CPU from `HostCopy` arrays in parallel with GPU-resident experts
avoids PCIe weight transfers and avoids evicting hot GPU bank slots:

- C++ SIMD kernels (`vendor/cpu_experts/cpu_experts.cpp`: AVX-512, AVX2+FMA, or plain C++) read
  MLX `affine` (2, 4, 8 bits; bf16 scales and biases) and `mxfp4` (E2M1 codes, uint8 E8M0 scales)
  rows directly in host memory with the GIL released.
- A lock-free worker `Pool` (KAME `atomic_shared_ptr` + `XWaitCell`) splits projections in
  `CHUNK_ROWS`-row chunks, runs workers at `SCHED_IDLE` (else nice 19) pinned to physical cores
  (skipping `RESERVE_CORES` foreground cores), and steals stalled chunks after `STEAL_NS`.
- `Load` monitors `/proc/stat` and `/proc/pressure/cpu` to yield to desktop/other processes when
  `QUANTEEZE_CPU_EXPERTS=auto`.
- `Server` polls GPU-mapped pinned request/answer mailboxes (`quanteeze_cpu_ask` / `quanteeze_cpu_wait`)
  on CUDA so GPU-resident experts run concurrently while CPU workers compute host-resident misses.
"""
import ctypes
from functools import lru_cache
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import allocator, variables

CHECKS = 3
TOLERANCE = 2e-3
POLL = 0.25
CALM_S = 2.5
PRESSURE_AT = 5.0
RESERVE_CORES = 2
MAX_AUTO_THREADS = 24
BATCH_JOB_VARS = ('SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID', 'PJM_JOBID', 'SGE_TASK_ID')
PCIE_GBS_PER_LANE = {'2.5': 0.25, '5.0': 0.5, '8.0': 0.985, '16.0': 1.969, '32.0': 3.938, '64.0': 7.563}
PCIE_WIDE_GBS = 36.0
SPIN_US = 0
ISA_NAMES = ('plain', 'avx2', 'avx512')
ACTIVATIONS = ('plain', 'deepseek_v41')
PRIORITIES = ('idle', 'nice', 'normal', 'wake')
CPU_EXPERTS = ('off', 'on', 'auto')

SERVE_MMAX = 16
SERVE_BMAX = 8
SERVE_HEAD = 64
SERVE_SPIN_US = 100
QUIET_S = 1.0
CPU_TIMEOUT_NS = 2_000_000_000

VENDOR_DIR = Path(__file__).resolve().parent / 'vendor' / 'cpu_experts'
KAME_DIR = allocator.VENDOR_DIR
KAME_HEADERS = ('atomic.h', 'atomic_mfence.h', 'atomic_smart_ptr.h', 'xwaitcell.h')
FLAGS = ('-std=c++17', '-O2', '-fPIC', '-shared', '-pthread', '-fvisibility=hidden')

_STATE = {'checked': 0, 'failed': None}
_LIB = [None, False]
_LOCK = threading.Lock()
_SAID = []


def in_batch_job(environ=None):
    """Whether the current process runs inside an HPC batch scheduler allocation."""
    env = os.environ if environ is None else environ
    return any(bool(env.get(k)) for k in BATCH_JOB_VARS)


def default_priority(priority=None, environ=None):
    """Worker scheduling priority name (`'idle'`, `'nice'`, `'normal'`, or `'wake'`)."""
    raw = (priority or variables.get('QUANTEEZE_CPU_PRIORITY') or 'auto').strip().lower()
    if raw == 'auto':
        return 'normal' if in_batch_job(environ) else 'idle'
    if raw not in PRIORITIES:
        raise ValueError(f'QUANTEEZE_CPU_PRIORITY is auto or one of {PRIORITIES}, not {raw!r}')
    return raw


def parse_cpulist(text):
    """Parse a Linux sysfs cpulist string (e.g. `'0-31,64-95'`) into a set of CPU IDs."""
    out = set()
    for part in (text or '').strip().split(','):
        part = part.strip()
        if not part or part.startswith('-'):
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            if a.isdigit() and b.isdigit():
                out.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.add(int(part))
    return out


def pci_sysfs_dir(bdf=None, pci_root='/sys/bus/pci/devices'):
    """Return the sysfs PCI device directory for `bdf` (or CUDA device 0), or `None`."""
    if bdf is None:
        from . import cuda_ple
        bdf = cuda_ple.pci_bus_id()
    if not bdf:
        return None
    bdf = bdf.strip().lower()
    candidates = [bdf]
    if ':' in bdf:
        dom, rest = bdf.split(':', 1)
        if len(dom) > 4:
            try:
                candidates.append(f'{int(dom, 16):04x}:{rest}')
            except ValueError:
                pass
    root = Path(pci_root)
    for cand in candidates:
        dev = root / cand
        if dev.is_dir():
            return dev
    return None


def gpu_local_cpus(bdf=None, pci_root='/sys/bus/pci/devices', node_root='/sys/devices/system/node'):
    """CPUs on the same PCIe root / NUMA node as the GPU, or empty set if unavailable."""
    dev = pci_sysfs_dir(bdf, pci_root)
    if dev is None:
        return set()
    try:
        cpus = parse_cpulist((dev / 'local_cpulist').read_text())
        if cpus:
            return cpus
    except OSError:
        pass
    try:
        nid = int((dev / 'numa_node').read_text().strip())
        if nid >= 0:
            return parse_cpulist((Path(node_root) / f'node{nid}' / 'cpulist').read_text())
    except (OSError, ValueError):
        pass
    return set()


def pcie_bandwidth_gbs(bdf=None, pci_root='/sys/bus/pci/devices'):
    """Approximate unidirectional PCIe bandwidth (GB/s) for the GPU from Linux sysfs, or `None`."""
    dev = pci_sysfs_dir(bdf, pci_root)
    if dev is None:
        return None
    try:
        tokens = (dev / 'current_link_speed').read_text().split()
        width = int((dev / 'current_link_width').read_text().strip())
        if not tokens or width <= 0:
            return None
        speed_tok = tokens[0]
        if speed_tok not in PCIE_GBS_PER_LANE:
            speed_tok = f'{float(speed_tok):.1f}'
        per_lane = PCIE_GBS_PER_LANE.get(speed_tok)
        return None if per_lane is None else per_lane * width
    except (OSError, ValueError):
        return None


def auto_cpu_experts(bdf=None, pci_root='/sys/bus/pci/devices'):
    """Whether hardware heuristics recommend enabling CPU expert offload under `'auto'`."""
    import mlx.core as mx
    from . import cuda_kernels, cuda_ple
    if mx.metal.is_available() or not cuda_kernels.available() or cuda_ple.integrated():
        return False
    if (isa() or 0) < 1 or default_threads() < variables.integer('QUANTEEZE_CPU_MIN_THREADS'):
        return False
    bw = pcie_bandwidth_gbs(bdf=bdf, pci_root=pci_root)
    if bw is not None and bw > PCIE_WIDE_GBS and default_threads() < 12 and (isa() or 0) < 2:
        return False
    return True


def cpu_setting():
    """`QUANTEEZE_CPU_EXPERTS`: `'off'`, `'on'`, or `'auto'`."""
    explicit = variables.is_set('QUANTEEZE_CPU_EXPERTS')
    value = (variables.get('QUANTEEZE_CPU_EXPERTS') or 'auto').strip().lower()
    if value not in CPU_EXPERTS:
        raise ValueError(f'QUANTEEZE_CPU_EXPERTS is one of {CPU_EXPERTS}, not {value!r}')
    if value == 'auto' and not explicit:
        return 'auto' if auto_cpu_experts() else 'off'
    return value


def _cache_dir():
    preferred = Path.home() / '.quanteeze' / 'cpu_experts'
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / f'.probe-{os.getpid()}'
        probe.write_bytes(b'')
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / f'quanteeze-cpu-experts-{os.getuid()}'
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _compiler():
    cxx = allocator.find_compiler()
    if cxx is None:
        raise OSError('no C++17 compiler found (set CXX or install clang++ / g++)')
    version = subprocess.run([*cxx, '--version'], capture_output=True, text=True, timeout=30).stdout
    return cxx, version


def source_digest(compiler_version):
    """16-hex-char digest of `cpu_experts.cpp`, KAME headers, flags, target, and compiler."""
    h = hashlib.sha256()
    for part in [VENDOR_DIR / 'cpu_experts.cpp', *(KAME_DIR / name for name in KAME_HEADERS)]:
        h.update(part.name.encode('utf-8') + b'\0')
        h.update(part.read_bytes() + b'\0')
    h.update(' '.join((*FLAGS, sys.platform, platform.machine(), compiler_version)).encode('utf-8'))
    return h.hexdigest()[:16]


def ensure_library():
    """Compile and return the path to the native CPU-experts shared library."""
    source = VENDOR_DIR / 'cpu_experts.cpp'
    if not source.is_file():
        raise OSError(f'missing C++ source {source}')
    for header in KAME_HEADERS:
        if not (KAME_DIR / header).is_file():
            raise OSError(f'missing KAME header {KAME_DIR / header}')
    cxx, version = _compiler()
    root = _cache_dir()
    out = root / f'libquanteeze_cpu_experts-{source_digest(version)}{allocator.library_suffix()}'
    if out.is_file() and out.stat().st_size > 0:
        return out
    tmp = out.with_name(f'.{out.name}.tmp-{os.getpid()}-{threading.get_ident()}')
    cmd = [*cxx, *FLAGS, f'-I{KAME_DIR}', str(source), '-o', str(tmp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise OSError(f'cpu_experts.cpp did not build:\n{proc.stderr[-4000:]}')
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
        dsym = Path(f'{tmp}.dSYM')
        if dsym.is_dir():
            shutil.rmtree(dsym, ignore_errors=True)
    return out


def lib():
    """The compiled C++ library (`ctypes.CDLL`), or `None` when it does not build."""
    with _LOCK:
        if _LIB[1]:
            return _LIB[0]
        try:
            loaded = ctypes.CDLL(str(ensure_library()))
            f32 = ctypes.POINTER(ctypes.c_float)
            loaded.qz_isa.restype = ctypes.c_int
            loaded.qz_affine.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, f32, f32, f32, ctypes.c_int, ctypes.c_int,
                ctypes.c_int,
            ]
            loaded.qz_mxfp4.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, f32, f32, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
            ]
            loaded.qz_pool_new.restype = ctypes.c_void_p
            loaded.qz_pool_new.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ]
            loaded.qz_pool_free.argtypes = [ctypes.c_void_p]
            loaded.qz_pool_busy.restype = ctypes.c_int
            loaded.qz_pool_busy.argtypes = [ctypes.c_void_p]
            loaded.qz_pool_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            loaded.qz_pool_stall.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            loaded.qz_pool_q8.argtypes = [ctypes.c_void_p, ctypes.c_int]
            loaded.qz_pool_stall_before.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            loaded.qz_reference.argtypes = [ctypes.c_void_p, f32, f32]
            loaded.qz_expert_reference.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, f32, f32,
                ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
            ]
            loaded.qz_activate.argtypes = [
                f32, f32, f32, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int,
            ]
            loaded.qz_expert.restype = ctypes.c_int
            loaded.qz_expert.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, f32, f32, f32, ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int,
            ]
            loaded.qz_server_new.restype = ctypes.c_void_p
            loaded.qz_server_new.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
                ctypes.c_int,
            ]
            loaded.qz_server_layer.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int,
            ]
            for name in ('qz_server_start', 'qz_server_free'):
                getattr(loaded, name).argtypes = [ctypes.c_void_p]
            loaded.qz_server_busy.restype = ctypes.c_int
            loaded.qz_server_busy.argtypes = [ctypes.c_void_p]
            loaded.qz_server_active.argtypes = [ctypes.c_void_p, ctypes.c_int]
            loaded.qz_server_cpus.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            loaded.qz_server_counts.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            _LIB[0] = loaded
        except (OSError, subprocess.SubprocessError) as error:
            if not _SAID:
                _SAID.append(True)
                print(f'quanteeze: cpu_experts native library did not build ({error})', file=sys.stderr, flush=True)
            _LIB[0] = None
        _LIB[1] = True
        return _LIB[0]


def isa():
    """Supported SIMD level: 2 AVX-512, 1 AVX2+FMA, 0 plain C++, or `None` without the library."""
    got = lib()
    return None if got is None else int(got.qz_isa())


# ---------------------------------------------------------------- elementwise roundings & C++ reference


def to_bf16(a):
    """Round `float32` array to `bfloat16` (round-to-nearest-even) returned as `float32`."""
    b = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


def fp8_roundtrip(a):
    """Block-32 E4M3 / UE8M0 FP8 roundtrip matching `deepseek_v41.fakequant.fake_quant_fp8_ue8m0`."""
    v = np.ascontiguousarray(a, np.float32).reshape(-1, 32)
    amax = np.maximum(np.abs(v).max(axis=1, keepdims=True), np.float32(1e-4))
    bits = (amax * np.float32(1.0 / 448.0)).view(np.uint32)
    scale = (((bits >> 23) + ((bits & 0x7FFFFF) != 0)) << 23).astype(np.uint32).view(np.float32)
    q = np.clip(v / scale, -448.0, 448.0).astype(np.float32)
    e = (np.abs(q).view(np.uint32) >> 23).astype(np.int64) - 3
    step = (np.maximum(e, 118) << 23).astype(np.uint32).view(np.float32)
    return (np.rint(q / step) * step * scale).astype(np.float32).reshape(np.shape(a))


def activate(gate, up, limit=0.0, weight=1.0, activation='plain'):
    """SwiGLU activation (`silu(gate) * up * weight`, clipped at `limit` when positive)."""
    gate, up = np.asarray(gate, np.float32), np.asarray(up, np.float32)
    v41 = activation == 'deepseek_v41'
    if v41:
        gate, up = to_bf16(gate), to_bf16(up)
    if limit > 0:
        gate, up = np.minimum(gate, np.float32(limit)), np.clip(up, -np.float32(limit), np.float32(limit))
    with np.errstate(over='ignore'):
        h = (gate / (np.float32(1.0) + np.exp(-gate)) * up * np.float32(weight)).astype(np.float32)
    return fp8_roundtrip(to_bf16(h)) if v41 else h


def native_reference(expert, x, limit=0.0, weight=1.0, activation='plain', q8=False):
    """Single-threaded C++ `float64` reference (`qz_expert_reference`), avoiding BLAS worker threads."""
    x = np.ascontiguousarray(x, np.float32)
    gate, up, down = (_proj(expert[n]) for n in ('gate_proj', 'up_proj', 'down_proj'))
    out = np.empty(expert['down_proj']['rows'], np.float32)
    lib().qz_expert_reference(
        ctypes.byref(gate), ctypes.byref(up), ctypes.byref(down),
        _fptr(x), _fptr(out), ctypes.c_float(limit), ctypes.c_float(weight),
        ACTIVATIONS.index(activation), int(bool(q8)),
    )
    return out


def projection(weight, scales, biases=None, bits=4, group=64, kind='affine'):
    """Build a contiguous projection descriptor dict from host array slices."""
    w = np.ascontiguousarray(weight)
    if w.dtype != np.uint32:
        w = w.view(np.uint32)
    out = {
        'kind': kind,
        'weight': w,
        'scales': np.ascontiguousarray(scales, np.uint8 if kind == 'mxfp4' else np.uint16),
        'bits': int(bits),
        'group': 32 if kind == 'mxfp4' else int(group),
    }
    out['biases'] = None if biases is None else np.ascontiguousarray(biases, np.uint16)
    out['rows'] = out['weight'].shape[0]
    out['cols'] = out['weight'].shape[1] * 32 // (4 if kind == 'mxfp4' else int(bits))
    return out


# ---------------------------------------------------------------- C++ pool & affinity


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def project_rows(p, x, xsum, y, row0, row1, isa_level):
    """Compute rows `row0..row1` of `W x` into `y` in C++ with the GIL released."""
    if p['kind'] == 'mxfp4':
        done = lib().qz_mxfp4(_ptr(p['weight']), _ptr(p['scales']), p['cols'], _fptr(x), _fptr(y),
                              row0, row1, isa_level)
    else:
        done = lib().qz_affine(_ptr(p['weight']), _ptr(p['scales']), _ptr(p['biases']), p['cols'], p['bits'],
                               p['group'], _fptr(x), _fptr(xsum), _fptr(y), row0, row1, isa_level)
    if done != 0:
        raise ValueError(
            f'CPU experts cannot compute {p["kind"]} ({p["bits"]} bits, group {p["group"]}, cols {p["cols"]})'
        )


class Proj(ctypes.Structure):
    _fields_ = [
        ('kind', ctypes.c_int),
        ('w', ctypes.c_void_p),
        ('sc', ctypes.c_void_p),
        ('bi', ctypes.c_void_p),
        ('rows', ctypes.c_int),
        ('cols', ctypes.c_int),
        ('bits', ctypes.c_int),
        ('group', ctypes.c_int),
    ]


def _proj(p):
    return Proj(
        1 if p['kind'] == 'mxfp4' else 0,
        p['weight'].ctypes.data,
        p['scales'].ctypes.data,
        None if p['biases'] is None else p['biases'].ctypes.data,
        p['rows'],
        p['cols'],
        p['bits'],
        p['group'],
    )


def project(p, x, isa_level=None):
    """Single-thread C++ `W x` across all rows of projection `p`."""
    x = np.ascontiguousarray(x, np.float32)
    y = np.empty(p['rows'], np.float32)
    xsum = None if p['kind'] == 'mxfp4' else np.ascontiguousarray(x.reshape(-1, p['group']).sum(1, dtype=np.float32))
    project_rows(p, x, xsum, y, 0, p['rows'], isa() if isa_level is None else isa_level)
    return y


def topology(root='/sys/devices/system/cpu'):
    """Return `{cpu: (package, core)}` for CPUs allowed to this process (Linux sysfs, else `{}`)."""
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    out = {}
    for cpu in allowed:
        try:
            core = (Path(root) / f'cpu{cpu}' / 'topology' / 'core_id').read_text().strip()
            package = (Path(root) / f'cpu{cpu}' / 'topology' / 'physical_package_id').read_text().strip()
        except OSError:
            return {}
        out[cpu] = (package, core)
    return out


def cores(n, root='/sys/devices/system/cpu', siblings=True, local_cpus=None):
    """Select `n` CPUs (NUMA/socket-local last physical cores first, skipping `RESERVE_CORES`, then SMT siblings)."""
    top = topology(root)
    if not top:
        return []
    local_set = (
        gpu_local_cpus()
        if (local_cpus is None and Path(root) == Path('/sys/devices/system/cpu'))
        else (set(local_cpus) if local_cpus else set())
    )
    min_phys = min(len(set(top.values())), 4 + RESERVE_CORES)
    if local_set:
        local_top = {c: pc for c, pc in top.items() if c in local_set}
        if len(set(local_top.values())) >= min_phys:
            top = local_top
    if len({pkg for pkg, _ in top.values()}) > 1:
        by_pkg = {}
        for c, pc in top.items():
            by_pkg.setdefault(pc[0], set()).add(pc)
        best_pkg = max(by_pkg, key=lambda p: (len(by_pkg[p]), -int(p) if p.isdigit() else 0))
        if len(by_pkg[best_pkg]) >= min(MAX_AUTO_THREADS + RESERVE_CORES, min_phys):
            top = {c: pc for c, pc in top.items() if pc[0] == best_pkg}
    by_core = {}
    for cpu, core in sorted(top.items()):
        by_core.setdefault(core, []).append(cpu)
    physical = sorted(by_core.values())
    usable = (physical[RESERVE_CORES:] or physical)[::-1]
    depth = max(map(len, usable), default=0) if siblings else 1
    out = [cpus[k] for k in range(depth) for cpus in usable if k < len(cpus)]
    return out[:max(1, n)]


def default_threads():
    """Configured or auto-detected worker thread count for `Pool` / `Load`."""
    given = (variables.get('QUANTEEZE_CPU_THREADS') or 'auto').strip().lower()
    if given != 'auto':
        return max(1, int(given))
    allowed_n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 2)
    auto = len(cores(MAX_AUTO_THREADS, siblings=False)) or max(1, allowed_n - RESERVE_CORES)
    return max(1, min(MAX_AUTO_THREADS, int(auto)))


class Pool:
    """Lock-free C++ thread pool executing single-expert MLPs directly from `HostCopy` memory."""

    def __init__(self, threads=None, isa_level=None, allowed=None, priority=None, spin_us=None, cpus=None, q8=None):
        self.threads = default_threads() if threads is None else max(1, int(threads))
        prio_name = default_priority(priority)
        self.priority = PRIORITIES.index(prio_name)
        self.isa = isa() if isa_level is None else isa_level
        self.allowed = allowed or (lambda: self.threads)
        self.spin_us = SPIN_US if spin_us is None else int(spin_us)
        self.cpus = cores(self.threads) if cpus is None else list(cpus)
        held = np.array(self.cpus or [-1], np.int32)
        self._handle = (
            lib().qz_pool_new(self.threads, self.priority, self.spin_us, held.ctypes.data, len(self.cpus))
            if lib() is not None else None
        )
        self.q8 = (not variables.off('QUANTEEZE_CPU_Q8')) if q8 is None else bool(q8)
        if self._handle is not None:
            lib().qz_pool_q8(ctypes.c_void_p(self._handle), int(self.q8))

    def expert(self, expert, x, limit=0.0, weight=1.0, activation='plain'):
        """Compute one expert's output `[D]` float32 for `x [D]`, returning `None` if disabled or invalid."""
        workers = min(self.threads, int(self.allowed()))
        if workers <= 0 or self._handle is None or self.isa is None or _STATE['failed']:
            return None
        x = np.ascontiguousarray(x, np.float32)
        projections = [_proj(expert[n]) for n in ('gate_proj', 'up_proj', 'down_proj')]

        def run(style):
            out = np.empty(expert['down_proj']['rows'], np.float32)
            hidden = np.empty(expert['gate_proj']['rows'], np.float32)
            done = lib().qz_expert(
                ctypes.c_void_p(self._handle), workers,
                *(ctypes.byref(p) for p in projections),
                _fptr(x), _fptr(out), _fptr(hidden),
                ctypes.c_float(limit), ctypes.c_float(weight),
                ACTIVATIONS.index(style), self.isa,
            )
            return done, out

        done, out = run('plain' if _STATE['checked'] < CHECKS else activation)
        if done in (-2, -3):
            return None
        if done != 0:
            _STATE['failed'] = 'unsupported expert projection shape for CPU offload'
            return None
        if _STATE['checked'] < CHECKS:
            want = native_reference(expert, x, limit, weight, q8=self.q8 and self.isa >= 1)
            scale = max(float(np.abs(want).max()), 1e-6)
            err = float(np.abs(out - want).max()) / scale
            _STATE['checked'] += 1
            if not err <= TOLERANCE:
                _STATE['failed'] = f'{ISA_NAMES[self.isa]}: {err:.2e} max relative error vs reference'
                return None
            if activation != 'plain':
                done, out = run(activation)
                if done != 0:
                    return None
        return out

    STATS = (
        'rung', 'woke', 'wake_ns', 'wake_ns_most', 'by_workers', 'by_caller',
        'stolen', 'workers_ns', 'caller_ns', 'caller_cpus', 'worker_cpus', 'moved',
    )

    def stats(self):
        """Return pool execution and work-stealing telemetry counters since creation."""
        if self._handle is None:
            return None
        out = np.zeros(len(self.STATS), np.int64)
        lib().qz_pool_stats(ctypes.c_void_p(self._handle), out.ctypes.data)
        got = dict(zip(self.STATS, (int(v) for v in out)))
        for k in ('caller_cpus', 'worker_cpus'):
            mask = got[k] & ((1 << 64) - 1)
            got[k] = {cpu for cpu in range(64) if (mask >> cpu) & 1}
        return got

    def busy(self):
        """Return number of workers currently inside a chunk."""
        return 0 if self._handle is None else int(lib().qz_pool_busy(ctypes.c_void_p(self._handle)))

    def close(self):
        if self._handle is not None:
            lib().qz_pool_free(ctypes.c_void_p(self._handle))
            self._handle = None


def verified():
    """Return `(checked_count, failure_reason_or_None)` for CPU expert sanity verification."""
    return _STATE['checked'], _STATE['failed']


# ---------------------------------------------------------------- HostCopy spec & mailbox Server


def free_cpus(taken):
    """CPUs allowed to this process whose physical cores do not overlap with `taken`."""
    top = topology()
    if not top:
        return []
    busy = {top[c] for c in taken if c in top}
    out = [c for c, core in sorted(top.items()) if core not in busy]
    return out or sorted(top)


def openblas_one_thread():
    """Restrict loaded OpenBLAS libraries to 1 thread so NumPy operations never starve workers."""
    names = (
        'scipy_openblas_set_num_threads64_',
        'scipy_openblas_set_num_threads',
        'openblas_set_num_threads64_',
        'openblas_set_num_threads',
    )
    try:
        maps = open('/proc/self/maps').read().split('\n')
    except OSError:
        return False
    libs = sorted({line.split()[-1] for line in maps if 'openblas' in line.lower() and line.split()[-1].startswith('/')})
    for path in libs:
        try:
            handle = ctypes.CDLL(path)
        except OSError:
            continue
        for name in names:
            fn = getattr(handle, name, None)
            if fn is not None:
                fn.argtypes = [ctypes.c_int]
                fn(1)
                return True
    return False


def layer_expert(spec, slot):
    """Zero-copy projection view dict for `slot` in a `layer_spec()` dictionary."""
    out = {}
    for p, parts in spec['arrays'].items():
        a = {name: parts[name][slot] for name in parts}
        w = a['weight']
        if w.dtype != np.uint32:
            w = w.view(np.uint32)
        out[p] = projection(
            w,
            a['scales'].view(np.uint8 if spec['kind'] == 'mxfp4' else np.uint16),
            None if a.get('biases') is None else a['biases'].view(np.uint16),
            spec['bits'],
            spec['groups'][p],
            kind=spec['kind'],
        )
    return out


def layer_spec(arrays, group_sizes, mode, bits):
    """Build a layer descriptor for `Server.add` from `(proj, name, host_array)` triples."""
    parts = {}
    for p, name, a in arrays:
        parts.setdefault(p, {})[name] = a
    return {
        'arrays': parts,
        'groups': dict(group_sizes),
        'kind': 'mxfp4' if mode == 'mxfp4' else 'affine',
        'bits': 4 if mode == 'mxfp4' else int(bits),
    }


class Server:
    """Zero-GIL C++ mailbox server answering GPU `CPU_ASK` / `CPU_WAIT` expert requests."""

    native = True
    COUNTS = ('requests', 'experts', 'failed', 'compute_ns')

    def __init__(self, pool, layers, hidden_dim, spin_us=SERVE_SPIN_US, cpus=None, pinned=True):
        self.pool, self.D = pool, int(hidden_dim)
        n = (max(layers) + 1) if layers else 1
        self.box_words, self.ans_words = SERVE_HEAD + SERVE_BMAX * self.D, SERVE_HEAD + SERVE_MMAX * self.D
        if pinned:
            from . import cuda_ple
            self.box = cuda_ple.pinned(
                (n, self.box_words), np.int32, "CPU experts' requests", 'CPU offload disabled'
            )
            self.ans = cuda_ple.pinned(
                (n, self.ans_words), np.int32, "CPU experts' answers", 'CPU offload disabled'
            )
            b, a = cuda_ple._device(self.box), cuda_ple._device(self.ans)
            if b is None or a is None:
                raise ValueError('no mapped pinned memory available for CPU expert mailbox')
            self.box[:], self.ans[:] = 0, 0
            import mlx.core as mx
            self.addr = {
                i: mx.array(np.array([b + i * self.box_words * 4, a + i * self.ans_words * 4], np.uint64))
                for i in layers
            }
            mx.eval(list(self.addr.values()))
        else:
            self.box = np.zeros((n, self.box_words), np.int32)
            self.ans = np.zeros((n, self.ans_words), np.int32)
            self.addr = {}
        self.specs = {}
        self._keep = []
        self.spin_us = int(spin_us)
        self._handle = None
        self._started = False
        self.cpus = free_cpus(pool.cpus) if cpus is None else list(cpus)
        self._make()

    def _make(self):
        if lib() is None or self.pool._handle is None:
            raise ValueError('cpu_experts native library unavailable')
        self._handle = lib().qz_server_new(
            ctypes.c_void_p(self.pool._handle),
            self.pool.isa,
            self.D,
            self.box.shape[0],
            self.box.ctypes.data,
            self.box_words,
            self.ans.ctypes.data,
            self.ans_words,
            self.spin_us,
        )
        held = np.array(self.cpus or [0], np.int32)
        lib().qz_server_cpus(ctypes.c_void_p(self._handle), held.ctypes.data, len(self.cpus))

    def add(self, i, spec, limit=0.0, activation='plain'):
        """Register layer `i`'s `HostCopy` slot pointers with the server."""
        self.specs[i] = (spec, float(limit), activation)
        expert = layer_expert(spec, 0)
        like = (Proj * 3)(*[_proj(expert[n]) for n in ('gate_proj', 'up_proj', 'down_proj')])
        bases = (ctypes.c_void_p * 9)()
        rowb = np.zeros(9, np.int64)
        for q, p in enumerate(('gate_proj', 'up_proj', 'down_proj')):
            for k, name in enumerate(('weight', 'scales', 'biases')):
                a = spec['arrays'][p].get(name)
                if a is not None:
                    bases[q * 3 + k] = a.ctypes.data
                    rowb[q * 3 + k] = a.strides[0]
        self._keep += [like, bases, rowb]
        lib().qz_server_layer(
            ctypes.c_void_p(self._handle),
            int(i),
            like,
            bases,
            rowb.ctypes.data,
            ctypes.c_float(limit),
            ACTIVATIONS.index(activation),
            int(expert['gate_proj']['rows']),
        )

    def start(self):
        if not self._started:
            lib().qz_server_start(ctypes.c_void_p(self._handle))
            self._started = True
        return self

    def active(self, n):
        lib().qz_server_active(ctypes.c_void_p(self._handle), int(n))

    def busy(self):
        return int(lib().qz_server_busy(ctypes.c_void_p(self._handle))) if self._handle else 0

    def quiet(self, timeout=QUIET_S):
        """Wait up to `timeout` seconds until no CPU expert request or worker chunk is in flight."""
        end = time.monotonic() + timeout
        while self.busy():
            if time.monotonic() > end:
                return False
            time.sleep(0.0001)
        return True

    def counts(self):
        out = np.zeros(len(self.COUNTS), np.int64)
        if self._handle:
            lib().qz_server_counts(ctypes.c_void_p(self._handle), out.ctypes.data)
        return dict(zip(self.COUNTS, (int(v) for v in out)))

    def close(self):
        if self._handle:
            lib().qz_server_free(ctypes.c_void_p(self._handle))
            self._handle = None


# ---------------------------------------------------------------- background CPU load monitor


def _allowed_cpus():
    if not hasattr(os, 'sched_getaffinity'):
        return None
    try:
        return set(os.sched_getaffinity(0))
    except OSError:
        return None


class Load:
    """Monitors other processes' CPU usage and PSI pressure and bounds active CPU offload workers."""

    def __init__(self, threads, yield_at=None, cores=None, clock=time.monotonic, read=None, least=None, cpus=None):
        self.threads = threads
        self.yield_at = float(variables.number('QUANTEEZE_CPU_YIELD_AT') if yield_at is None else yield_at)
        self.least = max(1, int(variables.integer('QUANTEEZE_CPU_MIN_THREADS') if least is None else least))
        aff = set(cpus) if cpus is not None else _allowed_cpus()
        self.cores = cores or (len(aff) if aff else None) or os.cpu_count() or 1
        self.clock = clock
        self.read = read or (lambda: _read_proc(cpus=aff))
        self.others, self.pressure = 0.0, 0.0
        self.workers = 0
        self._calm_since = None
        self._last = None
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        """Sample host CPU usage and return the worker count permitted for the next step."""
        total, own, pressure = self.read()
        now = self.clock()
        if self._last is not None and total is not None:
            t0, own0 = self._last[1], self._last[2]
            busy = max(total[0] - t0[0], 0)
            span = max(total[1] - t0[1], 1)
            mine = max(own - own0, 0)
            self.others = max(0.0, 100.0 * (busy - mine) / span)
        self.pressure = pressure if pressure is not None else 0.0
        self._last = (now, total, own) if total is not None else self._last
        others_cores = self.others / 100.0 * self.cores
        ceiling = max(0, min(self.threads, int(self.cores - RESERVE_CORES - np.ceil(others_cores))))
        if self.others >= self.yield_at or self.pressure >= PRESSURE_AT:
            self.workers, self._calm_since = 0, None
        elif self.others < self.yield_at / 2 and self.pressure < PRESSURE_AT / 5:
            if self._calm_since is None:
                self._calm_since = now
            elif now - self._calm_since >= CALM_S and self.workers < ceiling:
                self.workers = self.least if self.workers == 0 else self.workers + 1
                self._calm_since = now
        self.workers = min(self.workers, ceiling)
        if self.workers < self.least:
            self.workers = 0
        return self.workers

    def allowed(self):
        return self.workers

    def start(self):
        def run():
            try:
                self.sample()
            except Exception:
                self.workers = 0
            while not self._stop.wait(POLL):
                try:
                    self.sample()
                except Exception:
                    self.workers = 0

        self._thread = threading.Thread(target=run, name='cpu-expert-load', daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)


def _read_proc(stat='/proc/stat', own='/proc/self/stat', psi='/proc/pressure/cpu', cpus=None):
    """Read `(total_jiffies_busy_all, own_jiffies, psi_some_avg10)` from Linux `/proc`."""
    total, mine, some = None, 0, None
    n_all = 0
    try:
        agg_total = None
        sel_busy, sel_span, n_sel = 0, 0, 0
        with open(stat) as f:
            for line in f:
                if not line.startswith('cpu'):
                    break
                fields = line.split()
                if not fields:
                    continue
                tag = fields[0]
                vals = [int(v) for v in fields[1:11]]
                if len(vals) < 8:
                    continue
                busy = sum(vals[:3]) + sum(vals[5:8])
                span = sum(vals[:8])
                if tag == 'cpu':
                    agg_total = (busy, span)
                elif tag[3:].isdigit():
                    n_all += 1
                    if cpus and int(tag[3:]) in cpus:
                        sel_busy += busy
                        sel_span += span
                        n_sel += 1
        total = (sel_busy, sel_span) if n_sel > 0 else agg_total
    except (OSError, ValueError):
        pass
    try:
        with open(own) as f:
            line = f.read()
        fields = line[line.rfind(')') + 2:].split()
        mine = int(fields[11]) + int(fields[12])
    except (OSError, ValueError, IndexError):
        pass
    if not (cpus and n_all > len(cpus) and psi == '/proc/pressure/cpu'):
        try:
            with open(psi) as f:
                for line in f:
                    if line.startswith('some '):
                        for item in line.split():
                            if item.startswith('avg10='):
                                some = float(item.split('=', 1)[1])
        except (OSError, ValueError):
            pass
    return total, mine, some


# ---------------------------------------------------------------- CUDA mailbox handshake


CPU_ASK = r"""
    volatile int* box = reinterpret_cast<volatile int*>(addr[0]);
    volatile float* boxf = reinterpret_cast<volatile float*>(addr[0]);
    __shared__ int stamp, any;
    if (threadIdx.x == 0) {
      stamp = box[0] + 1;
      any = 0;
      for (int m = 0; m < M; ++m) any |= rows[m] >= 0;
    }
    __syncthreads();
    if (!any) {
      if (threadIdx.x == 0) { token[0] = 0; token[1] = 0; token[2] = 0; token[3] = 0; }
      return;
    }
    for (int d = threadIdx.x; d < B * D; d += blockDim.x) boxf[HEAD + d] = static_cast<float>(x[d]);
    if (threadIdx.x == 0) {
      box[1] = M;
      box[2 + 3 * MMAX] = B;
      const int k_per = B > 1 ? (K / B) : K;
      for (int m = 0; m < M; ++m) {
        box[2 + m] = static_cast<int>(rows[m]);
        float w = 1.0f;
        int xr = 0;
        for (int j = 0; j < K; ++j) {
          if (static_cast<int>(inv[j]) == m) {
            if (WEIGHS) w = static_cast<float>(weights[j]);
            xr = k_per > 0 ? (j / k_per) : 0;
            break;
          }
        }
        boxf[2 + MMAX + m] = w;
        box[2 + 2 * MMAX + m] = xr;
      }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      box[0] = stamp;
      __threadfence_system();
      token[0] = stamp;
      token[1] = 0;
      token[2] = 0;
      token[3] = 0;
    }
"""

CPU_WAIT = r"""
    volatile int* ans = reinterpret_cast<volatile int*>(addr[1]);
    volatile float* ansf = reinterpret_cast<volatile float*>(addr[1]);
    __shared__ int got;
    if (threadIdx.x == 0) {
      const int stamp = token[0];
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      got = 0;
      t1 = t0;
      while (stamp != 0) {
        if (ans[0] == stamp) { got = 1; break; }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > static_cast<unsigned long long>(TIMEOUT_NS)) break;
        __nanosleep(2000);
      }
      __threadfence_system();
      status[0] = got;
      status[1] = stamp != 0 && !got;
      status[2] = stamp == 0;
      status[3] = static_cast<int>((t1 - t0) / 1000);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < K * D; i += blockDim.x) {
      const int j = i / D, d = i - j * D;
      const int m = static_cast<int>(inv[j]);
      if (got && m >= 0 && m < M && ans[2 + m] != 0) out[i] = static_cast<T>(ansf[HEAD + m * D + d]);
      else out[i] = y[i];
    }
"""


@lru_cache(maxsize=None)
def _cpu_ask_kernel():
    from . import cuda_kernels
    return cuda_kernels.kernel('quanteeze_cpu_ask', ['rows', 'inv', 'weights', 'x', 'addr'], ['token'], CPU_ASK)


@lru_cache(maxsize=None)
def _cpu_wait_kernel():
    from . import cuda_kernels
    return cuda_kernels.kernel('quanteeze_cpu_wait', ['token', 'y', 'inv', 'addr'], ['out', 'status'], CPU_WAIT)


def cpu_ask(addr, x, rows, inv, weights=None):
    """Post a layer's CPU expert request (`rows [M]`, `inv [K]`, `x`) to pinned mailbox `addr`."""
    import mlx.core as mx
    k, d, most = int(inv.size), int(x.shape[-1]), int(rows.size)
    b_rows = max(1, int(x.size) // d)
    if most > SERVE_MMAX:
        raise ValueError(f'cpu_ask(): {most} rows asked, {SERVE_MMAX} at most')
    if b_rows > SERVE_BMAX:
        raise ValueError(f'cpu_ask(): {b_rows} batch rows asked, {SERVE_BMAX} at most')
    w = weights.reshape(-1).astype(mx.float32) if weights is not None else mx.ones((k,), mx.float32)
    return _cpu_ask_kernel()(
        inputs=[rows.astype(mx.int32).reshape(-1), inv.astype(mx.int32).reshape(-1), w, x.reshape(-1), addr],
        template=[
            ('B', b_rows), ('M', most), ('K', k), ('D', d), ('HEAD', SERVE_HEAD),
            ('MMAX', SERVE_MMAX), ('WEIGHS', int(weights is not None)),
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(4,)],
        output_dtypes=[mx.int32],
    )[0]


def cpu_wait(addr, token, y, inv, most=None):
    """Wait for the CPU's answer to `token` and merge CPU-computed slots into `y`."""
    import mlx.core as mx
    k, d = int(inv.size), int(y.shape[-1])
    if most is None:
        most = k
    out, status = _cpu_wait_kernel()(
        inputs=[token, y.reshape(-1), inv.astype(mx.int32).reshape(-1), addr],
        template=[
            ('T', y.dtype), ('M', int(most)), ('K', k), ('D', d),
            ('HEAD', SERVE_HEAD), ('TIMEOUT_NS', CPU_TIMEOUT_NS),
        ],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(int(y.size),), (4,)],
        output_dtypes=[y.dtype, mx.int32],
    )
    return out.reshape(y.shape), status
