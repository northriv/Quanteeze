# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Host memory pool allocator integration (`kamepoolalloc`, `rss-zero-loss` branch).

Bundles and preloads `kamepoolalloc` (https://github.com/northriv/kamepoolalloc — dual-licensed
Apache-2.0 OR GPL-2.0-or-later) via `DYLD_INSERT_LIBRARIES` on macOS or `LD_PRELOAD` on Linux
when `QUANTEEZE_ALLOCATOR` is `auto` (default) or `on`.

Why this helps MLX / MoE serving:
- Lock-free per-thread small-slot freelists (`16 B .. 32 KiB`) with lazy 1-page watermark pre-fill
  eliminate host `malloc`/`free` contention across Python, `libc++`, tokenizer, and prefetch threads
  without faulting untouched chunk pages into RSS.
- Two-level warm recycle caches (`32 KiB .. 32 MiB`) keep temporary safetensor slice buffers read
  during on-demand SSD expert/vision/PLE streaming mapped and warm across requests while returning
  idle memory promptly via `trim()` (`kame_pool_trim`) on model switch or footprint pressure.
"""
import ctypes
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from . import variables

REPO_URL = 'https://github.com/northriv/kamepoolalloc'
BRANCH = 'rss-zero-loss'
VENDOR_DIR = Path(__file__).resolve().parent / 'vendor' / 'kamepoolalloc'
SOURCES = (
    'allocator.cpp',
    'allocator.h',
    'allocator_prv.h',
    'kame_pool.h',
    'atomic.h',
    'atomic_mfence.h',
    'atomic_smart_ptr.h',
)
PRELOAD_SENTINEL = '_KPA_PRELOAD_TRIED'
_API_CACHE = {'pid': None, 'fns': None}


def supported(plat=None):
    """Whether drop-in dynamic-linker preload is supported on `plat` (macOS and Linux)."""
    plat = sys.platform if plat is None else plat
    return plat == 'darwin' or plat.startswith('linux')


def preload_var(plat=None):
    """Dynamic linker preload environment variable name for `plat`, or `None`."""
    plat = sys.platform if plat is None else plat
    if plat == 'darwin':
        return 'DYLD_INSERT_LIBRARIES'
    if plat.startswith('linux'):
        return 'LD_PRELOAD'
    return None


def library_suffix(plat=None):
    """Shared library filename suffix for `plat`."""
    plat = sys.platform if plat is None else plat
    return '.dylib' if plat == 'darwin' else '.so'


def _api():
    """Bound `ctypes` symbols from the current process if `kamepoolalloc` is preloaded, else `None`."""
    pid = os.getpid()
    if _API_CACHE['pid'] == pid:
        return _API_CACHE['fns']
    try:
        proc = ctypes.CDLL(None)
        reserved = proc.kame_pool_reserved_bytes
        usable = proc.kame_pool_malloc_usable_size
        cache_cap = proc.kame_pool_get_large_cache_cap
        trim_fn = proc.kame_pool_trim
    except (AttributeError, OSError):
        _API_CACHE.update(pid=pid, fns=None)
        return None
    reserved.argtypes = []
    reserved.restype = ctypes.c_size_t
    usable.argtypes = [ctypes.c_void_p]
    usable.restype = ctypes.c_size_t
    cache_cap.argtypes = []
    cache_cap.restype = ctypes.c_size_t
    trim_fn.argtypes = []
    trim_fn.restype = None
    fns = {
        'reserved_bytes': reserved,
        'usable_size': usable,
        'large_cache_cap': cache_cap,
        'trim': trim_fn,
    }
    _API_CACHE.update(pid=pid, fns=fns)
    return fns


def active():
    """Whether `kamepoolalloc` is loaded and active in this process."""
    return _api() is not None


def reserved_bytes():
    """Total region bytes currently mapped by `kamepoolalloc` in this process (`0` if inactive)."""
    fns = _api()
    return int(fns['reserved_bytes']()) if fns is not None else 0


def usable_size(ptr):
    """Pool-owned allocation usable byte size for `ptr` (`0` for foreign pointers or when inactive)."""
    fns = _api()
    return int(fns['usable_size'](ctypes.c_void_p(ptr))) if fns is not None and ptr else 0


def large_cache_cap():
    """Target warm large-recycle cache cap in bytes (`0` if inactive)."""
    fns = _api()
    return int(fns['large_cache_cap']()) if fns is not None else 0


def trim():
    """Release unowned per-thread freelists and L1/L2 recycle caches back to the OS."""
    fns = _api()
    if fns is None:
        return False
    fns['trim']()
    return True


def state():
    """Allocator status dictionary surfaced in `home.live()`."""
    if not active():
        return {'active': False, 'name': 'system'}
    return {
        'active': True,
        'name': 'kamepoolalloc',
        'branch': BRANCH,
        'reserved_mb': round(reserved_bytes() / 1e6, 1),
        'warm_cache_cap_mb': round(large_cache_cap() / 1e6, 1),
    }


def source_digest():
    """12-hex-char SHA-256 digest of the bundled `kamepoolalloc` C++ sources and target ABI."""
    h = hashlib.sha256()
    h.update(f'{sys.platform}:{platform.machine()}\0'.encode('utf-8'))
    for name in SOURCES:
        h.update(name.encode('utf-8') + b'\0')
        h.update((VENDOR_DIR / name).read_bytes() + b'\0')
    return h.hexdigest()[:12]


def default_cache_dir():
    """Writable directory holding the compiled `libkamepoolalloc` shared library."""
    preferred = Path.home() / '.quanteeze' / 'kamepoolalloc'
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / f'.probe-{os.getpid()}'
        probe.write_bytes(b'')
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / f'quanteeze-kamepoolalloc-{os.getuid()}'
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def find_compiler():
    """Return argv prefix for an available C++17 compiler (`CXX`, `clang++`, `c++`, or `g++`), or `None`."""
    cxx_env = (os.environ.get('CXX') or '').strip()
    if cxx_env:
        parts = shlex.split(cxx_env)
        if parts and shutil.which(parts[0]):
            return parts
    for candidate in ('clang++', 'c++', 'g++'):
        found = shutil.which(candidate)
        if found:
            return [found]
    return None


def ensure_library(cache_dir=None, notify=None):
    """Return path to compiled `libkamepoolalloc`, building it atomically on first run."""
    if not supported():
        raise RuntimeError(f'unsupported platform {sys.platform!r}')
    missing = [name for name in SOURCES if not (VENDOR_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f'missing bundled kamepoolalloc sources: {", ".join(missing)}')
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'libkamepoolalloc-{source_digest()}{library_suffix()}'
    if target.is_file() and target.stat().st_size > 0:
        return target

    cxx = find_compiler()
    if cxx is None:
        raise RuntimeError('no C++17 compiler found (set CXX or install clang++ / g++)')

    if notify is not None:
        notify(f'building bundled kamepoolalloc ({BRANCH}) with {Path(cxx[0]).name} -> {target}')

    tmp = target.with_name(f'.{target.name}.tmp-{os.getpid()}-{threading.get_ident()}')
    cmd = [
        *cxx,
        '-std=c++17',
        '-O3',
        '-shared',
        '-fPIC',
        '-DKAMEPOOLALLOC_DYLIB',
        '-fvisibility-inlines-hidden',
        '-fno-semantic-interposition',
        '-pthread',
        f'-I{VENDOR_DIR}',
        str(VENDOR_DIR / 'allocator.cpp'),
        '-o',
        str(tmp),
    ]
    if sys.platform.startswith('linux'):
        cmd.append('-ldl')
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or '').strip().splitlines()
            tail = detail[-1] if detail else f'exit {proc.returncode}'
            raise RuntimeError(f'compiler failed ({tail})')
        os.replace(tmp, target)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('compiler timed out building kamepoolalloc') from exc
    finally:
        tmp.unlink(missing_ok=True)
        dsym = Path(f'{tmp}.dSYM')
        if dsym.is_dir():
            shutil.rmtree(dsym, ignore_errors=True)
    return target


def preload_env(lib_path, env=None):
    """Return a copy of `env` with `lib_path` prepended to `DYLD_INSERT_LIBRARIES` or `LD_PRELOAD`."""
    var = preload_var()
    if var is None:
        raise RuntimeError(f'unsupported platform {sys.platform!r}')
    out = dict(os.environ if env is None else env)
    lib_str = str(lib_path)
    existing = out.get(var, '')
    parts = [p for p in existing.split(':') if p]
    if lib_str not in parts:
        out[var] = f'{lib_str}:{existing}' if existing else lib_str
    return out


def _configured_mode():
    """Effective `QUANTEEZE_ALLOCATOR` mode (`auto`, `on`, or `off`), checking saved settings before `control.start()`."""
    raw = variables.value('QUANTEEZE_ALLOCATOR')
    if not raw:
        try:
            from . import control
            raw = control.saved().get('QUANTEEZE_ALLOCATOR')
        except Exception:
            raw = None
    return (raw or variables.SETTINGS['QUANTEEZE_ALLOCATOR'].default or 'auto').strip().lower()


def _status_line():
    cap_gib = large_cache_cap() / (1 << 30)
    return f'host allocator: kamepoolalloc ({BRANCH}, warm recycle cap {cap_gib:g} GiB; {REPO_URL})'


def ensure_active(argv=None):
    """Ensure `kamepoolalloc` is preloaded according to `QUANTEEZE_ALLOCATOR`, re-execing once if needed.

    Returns a status line to print on stderr (or `None` when explicitly `off` and not preloaded).
    """
    mode = _configured_mode()
    if mode not in ('auto', 'on', 'off'):
        raise SystemExit(f'quanteeze: QUANTEEZE_ALLOCATOR={mode}: expected auto, on, or off')
    if mode == 'off':
        return _status_line() if active() else None
    if active():
        return _status_line()
    if not supported():
        if mode == 'on':
            raise SystemExit(
                f'quanteeze: QUANTEEZE_ALLOCATOR=on is only supported on macOS and Linux (got {sys.platform})'
            )
        return None
    if os.environ.get(PRELOAD_SENTINEL) == '1':
        if mode == 'on':
            raise SystemExit(
                f'quanteeze: QUANTEEZE_ALLOCATOR=on: dynamic linker ignored {preload_var()} '
                '(check macOS SIP / restricted Python binary)'
            )
        return f'host allocator: system malloc (dynamic linker ignored {preload_var()})'
    try:
        lib_path = ensure_library(
            notify=lambda msg: print(f'quanteeze: {msg}', file=sys.stderr, flush=True),
        )
    except RuntimeError as err:
        if mode == 'on':
            raise SystemExit(f'quanteeze: QUANTEEZE_ALLOCATOR=on: {err}') from None
        return f'host allocator: system malloc ({err})'

    env = preload_env(lib_path, os.environ)
    env[PRELOAD_SENTINEL] = '1'
    args = sys.argv[1:] if argv is None else list(argv)
    os.execve(
        sys.executable,
        [sys.executable, '-m', 'quanteeze.serve', *args],
        env,
    )
