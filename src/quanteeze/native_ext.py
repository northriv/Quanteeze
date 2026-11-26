# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Build and load compiled C++ extensions for QuantEEze (QUANTEEZE_NATIVE).

Compiles small C++20 nanobind modules on demand against Python, nanobind, MLX, and the bundled
kamepoolalloc / kamestm STM sources, caching the resulting shared library by compiler, ABI, and
source fingerprint under ~/.cache/quanteeze/native (or QUANTEEZE_CACHE_DIR).
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time

from . import variables

_EXTENSIONS = {}
_FAILED = set()
_LOCK = threading.Lock()
_SAID = set()

STM_SOURCES = ('threadlocal.cpp', 'xthread.cpp', 'xtime.cpp')

CACHE_KEEP_DAYS = 30
CACHE_MOST_BYTES = 1 << 30
_HEX16 = re.compile(r'^[0-9a-f]{16}$')


def _warn_once(key, msg):
    if key not in _SAID:
        _SAID.add(key)
        print(f'quanteeze: {msg}', file=sys.stderr, flush=True)


def mode():
    """Return QUANTEEZE_NATIVE as snapshotted at most recent load, else current env (`'auto'` default)."""
    return (variables.get('QUANTEEZE_NATIVE', variables.at_load()) or 'auto').strip().lower()


def available():
    """Whether C++ extension builds are enabled by QUANTEEZE_NATIVE."""
    return not variables.off('QUANTEEZE_NATIVE', variables.at_load())


def allowed(verified=True):
    """Whether a native component should be used (`'on'`, or `'auto'` when `verified`)."""
    m = mode()
    return m == 'on' or (m == 'auto' and bool(verified))


def _cache_root():
    root = os.environ.get('QUANTEEZE_CACHE_DIR')
    base = Path(root).expanduser() if root else (Path.home() / '.cache' / 'quanteeze')
    return base / 'native'


def _used(*paths):
    """Mark cached build artifacts as recently used so `_trim` retains them; return the first path."""
    for p in paths:
        try:
            os.utime(p)
        except OSError:
            pass
    return paths[0]


def _held(path):
    """Return total bytes stored under `path`."""
    if path.is_dir():
        return sum(_held(p) for p in path.iterdir())
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _trim(cache, now=None):
    """Prune cached builds unused for `CACHE_KEEP_DAYS`, always retaining the newest tag per module."""
    if not cache.is_dir():
        return
    old = (time.time() if now is None else now) - CACHE_KEEP_DAYS * 86400
    for stem_dir in cache.iterdir():
        if not stem_dir.is_dir():
            if stem_dir.name.startswith('tmp') and _mtime(stem_dir) < old:
                stem_dir.unlink(missing_ok=True)
            continue
        tags = []
        for entry in stem_dir.iterdir():
            if entry.is_dir() and _HEX16.match(entry.name):
                mtime = max(
                    [
                        _mtime(entry),
                        *(
                            _mtime(f)
                            for f in entry.iterdir()
                            if f.name == '.ok' or not (f.name.startswith('.') or f.name.startswith('tmp'))
                        ),
                    ],
                    default=0.0,
                )
                for leftover in list(entry.iterdir()):
                    if leftover.is_file() and leftover.name != '.ok' and (
                        leftover.name.startswith('.') or leftover.name.startswith('tmp')
                    ) and _mtime(leftover) < old:
                        leftover.unlink(missing_ok=True)
                tags.append((mtime, entry))
            elif entry.is_file() and (
                entry.name.startswith('.') or entry.name.startswith('tmp')
            ) and _mtime(entry) < old:
                entry.unlink(missing_ok=True)
        tags.sort(key=lambda pair: pair[0], reverse=True)
        for mtime, stale in tags[1:]:
            if mtime < old:
                shutil.rmtree(stale, ignore_errors=True)


def _room(cache):
    """Prune old builds first and raise `OSError` if the cache still exceeds `CACHE_MOST_BYTES`."""
    _trim(cache)
    held = _held(cache) if cache.is_dir() else 0
    if held >= CACHE_MOST_BYTES:
        raise OSError(
            f'the native builds hold {held / 1e9:.2f} GB in {cache}, {CACHE_MOST_BYTES / 1e9:.2f} GB at most: '
            f'nothing more is built there (old ones are pruned after {CACHE_KEEP_DAYS} days unused; '
            'delete some to build now)'
        )


def _cxx():
    env_cxx = os.environ.get('CXX')
    if env_cxx:
        found = shutil.which(env_cxx) or (env_cxx if Path(env_cxx).is_file() else None)
        if found:
            return found
    for name in ('c++', 'clang++', 'g++'):
        found = shutil.which(name)
        if found:
            return found
    return None


def _toolchain(with_mlx=True):
    """Locate C++ compiler, Python headers, nanobind, bundled KAME headers, and MLX."""
    cxx = _cxx()
    if cxx is None:
        return None, 'no C++20 compiler found (tried $CXX, c++, clang++, g++)'
    try:
        import nanobind
    except ImportError:
        return None, 'nanobind package is not installed'
    py_inc = sysconfig.get_path('include')
    if not py_inc or not (Path(py_inc) / 'Python.h').is_file():
        return None, f'Python.h not found under {py_inc!r}'
    nb_inc = Path(nanobind.include_dir())
    nb_src = Path(nanobind.source_dir())
    robin_inc = nb_inc.parent / 'ext' / 'robin_map' / 'include'
    if not (nb_src / 'nb_combined.cpp').is_file():
        return None, f'nanobind source not found under {nb_src}'
    kame_dir = Path(__file__).resolve().parent / 'vendor' / 'kamepoolalloc'
    mlx_inc = mlx_lib = None
    if with_mlx:
        try:
            import mlx.core as mx
        except ImportError:
            return None, 'mlx package is not available'
        mlx_root = Path(mx.__file__).resolve().parent
        mlx_inc = mlx_root / 'include'
        mlx_lib = mlx_root / 'lib'
        lib_name = 'libmlx.dylib' if sys.platform == 'darwin' else 'libmlx.so'
        if not (mlx_inc / 'mlx' / 'mlx.h').is_file() or not (mlx_lib / lib_name).is_file():
            return None, f'MLX headers or {lib_name} not found under {mlx_root}'
    return {
        'cxx': cxx,
        'py_inc': Path(py_inc),
        'nb_inc': nb_inc,
        'nb_src': nb_src,
        'robin_inc': robin_inc,
        'kame_dir': kame_dir,
        'mlx_inc': mlx_inc,
        'mlx_lib': mlx_lib,
    }, None


def _fingerprint(name, sources, watch_dirs, tc, with_mlx, with_stm, extra_files=()):
    h = hashlib.sha256()
    h.update(f'{name}|{sys.version_info[:3]}|{sys.platform}|{tc["cxx"]}|{with_mlx}|{with_stm}\n'.encode())
    h.update(f'{tc["nb_inc"]}|{tc["mlx_lib"]}\n'.encode())
    try:
        out = subprocess.check_output([tc['cxx'], '--version'], stderr=subprocess.STDOUT, timeout=5)
        h.update(out[:256])
    except Exception:
        pass
    seen = set()
    files = [Path(__file__), *sources, *extra_files]
    for d in watch_dirs:
        if d is not None and Path(d).is_dir():
            for p in sorted(Path(d).iterdir()):
                if p.is_file() and p.suffix in ('.h', '.hpp', '.cpp', '.cc'):
                    files.append(p)
    for p in files:
        p = Path(p).resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        h.update(f'=={p.name}==\n'.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _probe(mod_name, so_path):
    """Load the freshly compiled extension in a child Python process before dlopening it here."""
    code = (
        'import importlib.util, sys; '
        'spec = importlib.util.spec_from_file_location(sys.argv[1], sys.argv[2]); '
        'mod = importlib.util.module_from_spec(spec); '
        'spec.loader.exec_module(mod)'
    )
    res = subprocess.run([sys.executable, '-c', code, mod_name, str(so_path)],
                         capture_output=True, text=True, timeout=30)
    if res.returncode != 0:
        msg = (res.stderr or res.stdout or '').strip()
        raise RuntimeError(f'probe failed ({res.returncode}): {msg}')


def _load(mod_name, so_path):
    spec = importlib.util.spec_from_file_location(mod_name, str(so_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'could not create module spec for {so_path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_extension(name, *, with_mlx=True, with_stm=False, quiet=False, force=False):
    """Compile (if needed) and load vendor/<name>/<name>.cpp as extension _<name>. Returns None if unavailable."""
    src_dir = Path(__file__).resolve().parent / 'vendor' / name
    cpp = src_dir / f'{name}.cpp'
    if not cpp.is_file():
        if not quiet:
            _warn_once(('missing', name), f'extension source not found: {cpp}')
        return None
    tc, why = _toolchain(with_mlx=with_mlx)
    if tc is None:
        if not quiet:
            _warn_once(('tc', name), f'cannot build {name}: {why}')
        return None

    mod_name = f'_{name}'
    suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
    sources = [cpp, tc['nb_src'] / 'nb_combined.cpp']
    watch_dirs = [src_dir]
    kame = tc['kame_dir']
    extra_files = ()
    if with_stm:
        for s in STM_SOURCES:
            sources.append(kame / s)
        watch_dirs.append(kame)
    else:
        extra_files = tuple(sorted(kame.glob('*.h')) + sorted(kame.glob('*.hpp')))

    tag = _fingerprint(name, sources, watch_dirs, tc, with_mlx, with_stm, extra_files=extra_files)
    out_dir = _cache_root() / mod_name / tag
    so_path = out_dir / f'{mod_name}{suffix}'
    ok_path = out_dir / '.ok'

    if so_path.is_file() and ok_path.is_file():
        _used(so_path, out_dir)
        try:
            return _load(mod_name, so_path)
        except Exception as err:
            if not quiet:
                _warn_once(('reload', mod_name), f'rebuilding {mod_name} after load failure: {err}')
            try:
                ok_path.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        _room(_cache_root())
    except OSError as err:
        if not quiet:
            _warn_once(('room', name), f'cannot build {name}: {err}')
        if force or mode() == 'on':
            raise
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    includes = [
        f'-I{tc["py_inc"]}',
        f'-I{tc["nb_inc"]}',
        f'-I{tc["nb_src"]}',
        f'-I{tc["robin_inc"]}',
        f'-I{src_dir}',
        f'-I{tc["kame_dir"]}',
    ]
    if with_mlx:
        includes.append(f'-I{tc["mlx_inc"]}')

    cmd = [
        tc['cxx'],
        '-std=c++20',
        '-O3',
        '-fPIC',
        '-fvisibility=hidden',
        '-pthread',
        *includes,
        *(str(s) for s in sources),
    ]
    cmd.append('-DNB_DOMAIN=mlx' if with_mlx else '-DNB_DOMAIN=quanteeze')
    if with_stm:
        cmd.append('-Wno-deprecated-declarations')
    if sys.platform == 'darwin':
        cmd += ['-shared', '-undefined', 'dynamic_lookup']
        if with_mlx:
            cmd += [f'-Wl,-rpath,{tc["mlx_lib"]}', str(tc['mlx_lib'] / 'libmlx.dylib')]
    else:
        cmd += ['-shared', '-ldl']
        if with_mlx:
            cmd += [f'-Wl,-rpath,{tc["mlx_lib"]}', str(tc['mlx_lib'] / 'libmlx.so')]

    fd, tmp_name = tempfile.mkstemp(prefix=f'.{mod_name}-', suffix=suffix, dir=str(out_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        cmd += ['-o', str(tmp_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode != 0:
            if not quiet:
                detail = (res.stderr or res.stdout or '').strip()
                lines = detail.splitlines()
                errs = [ln for ln in lines if 'error:' in ln or 'fatal error:' in ln]
                shown = '\n'.join((errs[:12] + ['---'] + lines[-12:]) if errs else lines[-20:])
                _warn_once(('compile', name), f'compiling {name}.cpp failed ({res.returncode}):\n{shown}')
            return None
        _probe(mod_name, tmp_path)
        os.replace(tmp_path, so_path)
        ok_path.write_text('ok\n')
        _used(so_path, out_dir)
        return _load(mod_name, so_path)
    except Exception as err:
        if not quiet:
            _warn_once(('build', name), f'building {name} failed: {err}')
        return None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def extension(name='stm_core', *, with_mlx=True, with_stm=True, force=False, quiet=False):
    """Return the cached compiled C++ extension _<name>, building it on first use; None if disabled/failed."""
    m = mode()
    if not force and not available():
        return None
    key = (name, bool(with_mlx), bool(with_stm))
    fail_key = ('ext', m, key)
    with _LOCK:
        cached = _EXTENSIONS.get(key)
        if cached is not None:
            return cached
        if not force and fail_key in _FAILED:
            return None
        mod = build_extension(name, with_mlx=with_mlx, with_stm=with_stm, quiet=quiet, force=force)
        if mod is not None:
            _EXTENSIONS[key] = mod
        elif not force:
            _FAILED.add(fail_key)
        return mod


def _library_fingerprint(name, sources, watch_dirs, cxx):
    h = hashlib.sha256()
    h.update(f'lib|{name}|{sys.platform}|{cxx}\n'.encode())
    try:
        out = subprocess.check_output([cxx, '--version'], stderr=subprocess.STDOUT, timeout=5)
        h.update(out[:256])
    except Exception:
        pass
    seen = set()
    files = [Path(__file__), *sources]
    for d in watch_dirs:
        if d is not None and Path(d).is_dir():
            for p in sorted(Path(d).iterdir()):
                if p.is_file() and p.suffix in ('.h', '.hpp', '.cpp', '.cc'):
                    files.append(p)
    for p in files:
        p = Path(p).resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        h.update(f'=={p.name}==\n'.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def library_path(name='nvrtc_pch'):
    """Return the expected cache path for pure-C++ shared library `lib<name>`, or None if unavailable."""
    cxx = _cxx()
    if cxx is None:
        return None
    src_dir = Path(__file__).resolve().parent / 'vendor' / name
    cpp = src_dir / f'{name}.cpp'
    if not cpp.is_file():
        return None
    kame_dir = Path(__file__).resolve().parent / 'vendor' / 'kamepoolalloc'
    tag = _library_fingerprint(name, [cpp], [src_dir, kame_dir], cxx)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    return _cache_root() / f'lib{name}' / tag / f'lib{name}{suffix}'


def build_library(name='nvrtc_pch', *, quiet=False, force=False):
    """Compile (without importing MLX or nanobind) pure-C++ shared library `vendor/<name>/<name>.cpp`."""
    src_dir = Path(__file__).resolve().parent / 'vendor' / name
    cpp = src_dir / f'{name}.cpp'
    if not cpp.is_file():
        if not quiet:
            _warn_once(('missing', name), f'library source not found: {cpp}')
        return None
    cxx = _cxx()
    if cxx is None:
        if not quiet:
            _warn_once(('tc', name), f'cannot build {name}: no C++20 compiler found')
        return None
    kame_dir = Path(__file__).resolve().parent / 'vendor' / 'kamepoolalloc'
    tag = _library_fingerprint(name, [cpp], [src_dir, kame_dir], cxx)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    out_dir = _cache_root() / f'lib{name}' / tag
    so_path = out_dir / f'lib{name}{suffix}'
    ok_path = out_dir / '.ok'
    if so_path.is_file() and ok_path.is_file():
        _used(so_path, out_dir)
        return so_path
    try:
        _room(_cache_root())
    except OSError as err:
        if not quiet:
            _warn_once(('room', name), f'cannot build {name}: {err}')
        if force or mode() == 'on':
            raise
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [cxx, '-std=c++20', '-O3', '-fPIC', '-fvisibility=hidden', '-pthread', f'-I{kame_dir}', str(cpp)]
    cmd += ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-ldl']
    fd, tmp_name = tempfile.mkstemp(prefix=f'.lib{name}-', suffix=suffix, dir=str(out_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        res = subprocess.run([*cmd, '-o', str(tmp_path)], capture_output=True, text=True, timeout=180)
        if res.returncode != 0:
            if not quiet:
                detail = (res.stderr or res.stdout or '').strip()
                lines = detail.splitlines()
                errs = [ln for ln in lines if 'error:' in ln or 'fatal error:' in ln]
                shown = '\n'.join((errs[:12] + ['---'] + lines[-12:]) if errs else lines[-20:])
                _warn_once(('compile', name), f'compiling {name}.cpp failed ({res.returncode}):\n{shown}')
            return None
        os.replace(tmp_path, so_path)
        ok_path.write_text('ok\n')
        _used(so_path, out_dir)
        return so_path
    except Exception as err:
        if not quiet:
            _warn_once(('build', name), f'building {name} failed: {err}')
        return None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def library(name='nvrtc_pch', *, force=False, quiet=False):
    """Return path to cached pure-C++ shared library `lib<name>`, building on demand without importing MLX."""
    m = mode()
    if not force and not available():
        return None
    key = ('lib', name)
    fail_key = ('lib', m, name)
    with _LOCK:
        cached = _EXTENSIONS.get(key)
        if cached is not None and Path(cached).is_file():
            _used(Path(cached), Path(cached).parent)
            return cached
        if not force and fail_key in _FAILED:
            return None
        built = build_library(name, quiet=quiet, force=force)
        if built is not None:
            _EXTENSIONS[key] = built
        elif not force:
            _FAILED.add(fail_key)
        return built
