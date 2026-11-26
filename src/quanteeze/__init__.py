# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Quanteeze: uniform least-squares MoE quantization & high-speed MLX inference server."""
import importlib
import importlib.abc
import importlib.util
import os
import sys

# A Metal command buffer holds this many operations and MB before MLX commits it (where the environment
# says nothing).  MLX reads them as it is imported, so here, before mlx.core is imported.
MLX_BUFFERS = {'MLX_MAX_OPS_PER_BUFFER': '300', 'MLX_MAX_MB_PER_BUFFER': '4000'}
if sys.platform == 'darwin':
    for _name, _value in MLX_BUFFERS.items():
        os.environ.setdefault(_name, _value)


def _gpu_model(pattern='/proc/driver/nvidia/gpus/*/information'):
    """The first NVIDIA GPU's model as the driver names it ('NVIDIA_A100-SXM4-40GB'), or None."""
    import glob
    for info in sorted(glob.glob(pattern)):
        with open(info) as f:
            for line in f:
                if line.startswith('Model:'):
                    return '_'.join(line.split(':', 1)[1].split())
    return None


if sys.platform != 'darwin' and _gpu_model():
    os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER', '200')
    os.environ.setdefault('MLX_MAX_MB_PER_BUFFER', '100000')
    os.environ.setdefault('MLX_CUDA_GRAPH_CACHE_SIZE', '100')
    os.environ.setdefault('CUDA_CACHE_MAXSIZE', str(4 << 30))
    try:
        from importlib.metadata import version as _version
        os.environ.setdefault('MLX_PTX_CACHE_DIR', os.path.join(
            os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
            'mlx', _version('mlx'), _gpu_model(), 'ptx',
        ))
    except Exception:
        pass


NVRTC_PCH = [None]              # vendor/nvrtc_pch/nvrtc_pch.cpp CDLL handle when loaded, else None
NVRTC_PCH_DIR = [None]          # per-process directory where NVRTC writes PCH files, or None
_PCH_PREFIX = 'quanteeze-nvrtc-pch-'
_PCH_ALIVE = '.alive'           # flock'd while this process lives
_PCH_HELD = []                  # open fd(s) holding the flock


def _pch_dir(lib):
    """Create a per-process PCH directory under $TMPDIR (or /tmp) and bind it via `qz_nvrtc_pch_dir`.

    Sweeps stale same-host, same-uid PCH directories whose `.alive` flock can be acquired (i.e. dead
    processes), holds an exclusive flock on `.alive` while alive, and removes the directory via a
    PID-guarded `atexit` callback.
    """
    import atexit
    import fcntl
    import shutil
    import socket
    import tempfile
    import time

    root = tempfile.gettempdir()
    prefix = f'{_PCH_PREFIX}{socket.gethostname().split(".")[0]}-'
    uid = os.getuid()
    for name in os.listdir(root):
        if not name.startswith(prefix):
            continue
        old = os.path.join(root, name)
        try:
            if os.stat(old).st_uid != uid:
                continue
            fd = os.open(os.path.join(old, _PCH_ALIVE), os.O_RDONLY)
        except FileNotFoundError:
            try:
                if time.time() - os.stat(old).st_mtime > 600:
                    shutil.rmtree(old, ignore_errors=True)
            except OSError:
                pass
            continue
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            continue
        else:
            shutil.rmtree(old, ignore_errors=True)
        finally:
            os.close(fd)

    path = tempfile.mkdtemp(prefix=f'{prefix}{os.getpid()}-', dir=root)
    fd = None
    try:
        fd = os.open(os.path.join(path, _PCH_ALIVE), os.O_WRONLY | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not lib.qz_nvrtc_pch_dir(path.encode()):
            raise OSError('nvrtc_pch library rejected PCH directory')
        _PCH_HELD.append(fd)
    except BaseException:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        shutil.rmtree(path, ignore_errors=True)
        raise
    pid = os.getpid()
    atexit.register(lambda: os.getpid() == pid and shutil.rmtree(path, ignore_errors=True))
    return path


def _nvrtc_pch():
    """Load libnvrtc_pch with RTLD_GLOBAL before MLX imports so custom kernels add --pch."""
    if 'mlx.core' in sys.modules:
        return
    from . import native_ext, variables
    if variables.off('QUANTEEZE_NVRTC_PCH') or not native_ext.available():
        return
    try:
        import ctypes
        lib_path = native_ext.library('nvrtc_pch', quiet=True)
        if lib_path is None:
            return
        lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        NVRTC_PCH[0] = lib
        NVRTC_PCH_DIR[0] = _pch_dir(lib)
    except Exception as e:                                                # noqa: BLE001 -- NVRTC as MLX calls it
        print(f'quanteeze: NVRTC precompiled headers (QUANTEEZE_NVRTC_PCH) not used: {type(e).__name__}: {e}',
              file=sys.stderr, flush=True)


def nvrtc_pch_counts():
    """Return `(compiles, pch_added, pch_made)` recorded by `nvrtc_pch`, or None when not loaded."""
    if NVRTC_PCH[0] is None:
        return None
    import ctypes
    out = (ctypes.c_uint64 * 3)()
    NVRTC_PCH[0].qz_nvrtc_pch_counts(out)
    return int(out[0]), int(out[1]), int(out[2])


if sys.platform != 'darwin' and _gpu_model():
    _nvrtc_pch()

# mlx-vlm's modules quanteeze patches as they are imported, before any model is built or loaded.
PATCHES = {
    'mlx_vlm.utils': lambda m: importlib.import_module('quanteeze.checkpoints').install(m),
    'mlx_vlm.models.qwen4_exp.ple_storage':
        lambda m: importlib.import_module('quanteeze.ple_store').install(m.QuantizedMMapNGramEmbedding),
}


class _Patcher(importlib.abc.MetaPathFinder):
    """Runs PATCHES[name] on a module of PATCHES once it is executed (`quanteeze.checkpoints`: an
    MTP head under another name; `quanteeze.ple_store`: an n-gram table in any layout)."""

    def __init__(self):
        self.busy = set()

    def find_spec(self, fullname, path, target=None):
        if fullname not in PATCHES or fullname in self.busy:
            return None
        self.busy.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self.busy.discard(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return spec
        exec_module = spec.loader.exec_module

        def run(module):
            exec_module(module)
            PATCHES[fullname](module)

        spec.loader.exec_module = run
        return spec


for _name, _patch in PATCHES.items():
    if _name in sys.modules:
        _patch(sys.modules[_name])
if not any(isinstance(f, _Patcher) for f in sys.meta_path):
    sys.meta_path.insert(0, _Patcher())
