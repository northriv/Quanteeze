# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Zero-GIL, lock-free C++ `pread` reader pool for `Demand` on-demand SSD streaming."""
import ctypes
import errno
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from . import allocator, variables

CHECKS = 3
MAX_SEGMENTS = 12
STATES = ('queued', 'taken', 'done', 'failed', 'cancelled')
_DONE, _FAILED, _CANCELLED = 2, 3, 4

VENDOR_DIR = Path(__file__).resolve().parent / 'vendor' / 'native_reads'
KAME_DIR = allocator.VENDOR_DIR
KAME_HEADERS = ('atomic.h', 'atomic_mfence.h', 'atomic_queue.h', 'atomic_smart_ptr.h', 'xwaitcell.h')
FLAGS = ('-std=c++17', '-O2', '-fPIC', '-shared', '-pthread', '-fvisibility=hidden')

_STATE = {'checked': 0, 'failed': None}
_LIBS = [None, False]  # [(py_lib, c_lib), attempted]
_LOCK = threading.Lock()
_SAID = []


def _cache_dir():
    preferred = Path.home() / '.quanteeze' / 'native_reads'
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / f'.probe-{os.getpid()}'
        probe.write_bytes(b'')
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / f'quanteeze-native-reads-{os.getuid()}'
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _compiler():
    cxx = allocator.find_compiler()
    if cxx is None:
        raise OSError('no C++17 compiler found (set CXX or install clang++ / g++)')
    version = subprocess.run([*cxx, '--version'], capture_output=True, text=True, timeout=30).stdout
    return cxx, version


def source_digest(compiler_version):
    """16-hex-char digest of `readers.cpp`, KAME headers, flags, target, and compiler."""
    h = hashlib.sha256()
    for part in [VENDOR_DIR / 'readers.cpp', *(KAME_DIR / name for name in KAME_HEADERS)]:
        h.update(part.name.encode('utf-8') + b'\0')
        h.update(part.read_bytes() + b'\0')
    h.update(' '.join((*FLAGS, sys.platform, platform.machine(), compiler_version)).encode('utf-8'))
    return h.hexdigest()[:16]


def ensure_library():
    """Compile and return the path to the native readers shared library."""
    source = VENDOR_DIR / 'readers.cpp'
    if not source.is_file():
        raise OSError(f'missing C++ source {source}')
    for header in KAME_HEADERS:
        if not (KAME_DIR / header).is_file():
            raise OSError(f'missing KAME header {KAME_DIR / header}')
    cxx, version = _compiler()
    root = _cache_dir()
    out = root / f'libquanteeze_readers-{source_digest(version)}{allocator.library_suffix()}'
    if out.is_file() and out.stat().st_size > 0:
        return out
    tmp = out.with_name(f'.{out.name}.tmp-{os.getpid()}-{threading.get_ident()}')
    cmd = [*cxx, *FLAGS, f'-I{KAME_DIR}', str(source), '-o', str(tmp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise OSError(f'readers.cpp did not build:\n{proc.stderr[-4000:]}')
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
        dsym = Path(f'{tmp}.dSYM')
        if dsym.is_dir():
            shutil.rmtree(dsym, ignore_errors=True)
    return out


def libs():
    """Return `(py_lib, c_lib)` (`PyDLL` for fast lock-free calls, `CDLL` for waits), or `None`."""
    with _LOCK:
        if _LIBS[1]:
            return _LIBS[0]
        try:
            path = str(ensure_library())
            py_lib = ctypes.PyDLL(path)
            c_lib = ctypes.CDLL(path)

            c_lib.qz_rd_new.argtypes = [ctypes.c_int]
            c_lib.qz_rd_new.restype = ctypes.c_void_p
            c_lib.qz_rd_free.argtypes = [ctypes.c_void_p]
            c_lib.qz_rd_free.restype = None
            c_lib.qz_rd_wait.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
            ]
            c_lib.qz_rd_wait.restype = ctypes.c_int32

            py_lib.qz_rd_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            py_lib.qz_rd_file.restype = ctypes.c_int32
            py_lib.qz_rd_submit.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            py_lib.qz_rd_submit.restype = ctypes.c_int64
            py_lib.qz_rd_status.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            py_lib.qz_rd_status.restype = ctypes.c_int32
            py_lib.qz_rd_cancel.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            py_lib.qz_rd_cancel.restype = ctypes.c_int32
            py_lib.qz_rd_release.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            py_lib.qz_rd_release.restype = None
            py_lib.qz_rd_counts.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
            py_lib.qz_rd_counts.restype = None
            py_lib.qz_rd_times.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
            py_lib.qz_rd_times.restype = None
            py_lib.qz_rd_backlog.argtypes = [ctypes.c_void_p]
            py_lib.qz_rd_backlog.restype = ctypes.c_int64

            c_lib.qz_rd_submit.argtypes = py_lib.qz_rd_submit.argtypes
            c_lib.qz_rd_submit.restype = ctypes.c_int64
            c_lib.qz_rd_status.argtypes = py_lib.qz_rd_status.argtypes
            c_lib.qz_rd_status.restype = ctypes.c_int32
            c_lib.qz_rd_release.argtypes = py_lib.qz_rd_release.argtypes
            c_lib.qz_rd_release.restype = None
            c_lib.qz_rd_times.argtypes = py_lib.qz_rd_times.argtypes
            c_lib.qz_rd_times.restype = None
            c_lib.qz_rd_backlog.argtypes = py_lib.qz_rd_backlog.argtypes
            c_lib.qz_rd_backlog.restype = ctypes.c_int64

            _LIBS[0] = (py_lib, c_lib)
        except OSError as exc:
            if not _SAID:
                _SAID.append(True)
                print(f'quanteeze: native SSD readers unavailable ({exc}); using Python thread pool',
                      file=sys.stderr, flush=True)
            _LIBS[0] = None
        _LIBS[1] = True
        return _LIBS[0]


def setting():
    """Return validated `QUANTEEZE_DEMAND_READ_NATIVE` (`'auto'`, `'on'`, or `'off'`)."""
    raw = (variables.get('QUANTEEZE_DEMAND_READ_NATIVE') or 'auto').strip().lower()
    if raw not in ('auto', 'on', 'off'):
        raise ValueError(f'QUANTEEZE_DEMAND_READ_NATIVE: auto, on, or off, not {raw}')
    return raw


def enabled():
    """Whether `Demand` should use the zero-GIL C++ `pread` reader pool."""
    if setting() == 'off' or _STATE['failed'] is not None:
        return False
    return libs() is not None


def checking():
    """Whether native read verification against `_from_file` is still active."""
    return _STATE['checked'] < CHECKS and _STATE['failed'] is None


def record_check(ok, detail=''):
    """Update bit-verification check counters; disables native reads on mismatch."""
    if ok:
        _STATE['checked'] += 1
        return True
    _STATE['failed'] = detail or 'byte mismatch'
    print(f'quanteeze: native SSD reader verification failed ({_STATE["failed"]}); falling back to Python readers',
          file=sys.stderr, flush=True)
    return False


class NativeReadFuture:
    """Future-compatible handle for a queued native C++ `pread` request."""

    def __init__(self, readers, tok, finish=None, check=None, fallback=None, keepalive=None):
        self._readers = readers
        self._handle = readers._raw_handle
        self._tok = int(tok)
        self._finish = finish
        self._check = check
        self._fallback = fallback
        self._keepalive = keepalive
        self._mu = threading.Lock()
        self._settle_mu = threading.Lock()
        self._waiters = 0
        self._want_release = False
        self._released = False
        self._cancelled = False
        self._has_result = False
        self._result = None
        self._exc = None

    def _try_release_locked(self, clear_keepalive=True):
        if self._want_release and not self._released and self._waiters == 0:
            self._released = True
            handle = self._handle
            self._handle = None
            if clear_keepalive:
                self._keepalive = None
            return handle
        return None

    def _release(self, clear_keepalive=True):
        with self._mu:
            self._want_release = True
            handle = self._try_release_locked(clear_keepalive=clear_keepalive)
        if handle is not None:
            self._readers._py.qz_rd_release(handle, self._tok)

    def cancelled(self):
        return self._cancelled

    def done(self):
        with self._mu:
            if self._has_result or self._exc is not None or self._cancelled or self._released or self._want_release:
                return True
            handle = self._handle
            if handle is None:
                return True
            self._waiters += 1
        try:
            return int(self._readers._py.qz_rd_status(handle, self._tok)) >= _DONE
        finally:
            with self._mu:
                self._waiters -= 1
                rel = self._try_release_locked()
            if rel is not None:
                self._readers._py.qz_rd_release(rel, self._tok)

    def cancel(self):
        with self._mu:
            if self._has_result or self._exc is not None or self._cancelled or self._released or self._want_release:
                return False
            handle = self._handle
            if handle is None:
                self._cancelled = True
                self._want_release = True
                self._keepalive = None
                return True
            self._waiters += 1
        did = False
        try:
            did = bool(self._readers._py.qz_rd_cancel(handle, self._tok))
        finally:
            with self._mu:
                self._waiters -= 1
                if did:
                    self._cancelled = True
                    self._want_release = True
                rel = self._try_release_locked()
        if rel is not None:
            self._readers._py.qz_rd_release(rel, self._tok)
        return did

    def result(self, timeout=None):
        with self._mu:
            if self._has_result:
                return self._result
            if self._exc is not None:
                raise self._exc
            if self._cancelled:
                raise RuntimeError('read cancelled')
            handle = self._handle
            if handle is None:
                raise RuntimeError('native reader pool closed')
            self._waiters += 1
        timeout_us = -1 if timeout is None else max(0, int(float(timeout) * 1e6))
        err = ctypes.c_int32(0)
        fidx = ctypes.c_int32(-1)
        st = -1
        rel = None
        try:
            st = int(self._readers._c.qz_rd_wait(handle, self._tok, timeout_us, ctypes.byref(err), ctypes.byref(fidx)))
        finally:
            with self._mu:
                self._waiters -= 1
                if st >= 0:
                    self._want_release = True
                rel = self._try_release_locked(clear_keepalive=False)
        if rel is not None:
            self._readers._py.qz_rd_release(rel, self._tok)
        if st < 0:
            raise TimeoutError('timed out waiting for native SSD read')
        try:
            with self._settle_mu:
                if self._has_result:
                    return self._result
                if self._exc is not None:
                    raise self._exc
                if self._cancelled:
                    raise RuntimeError('read cancelled')
                if st == _CANCELLED:
                    self._cancelled = True
                    self._exc = RuntimeError('read cancelled')
                    raise self._exc
                if st == _FAILED:
                    code = int(err.value)
                    path = self._readers.path_of(int(fidx.value))
                    if code < 0:
                        self._exc = OSError(errno.EIO, 'short read past end of file', path)
                    else:
                        self._exc = OSError(code or errno.EIO, os.strerror(code) if code > 0 else 'read failed', path)
                    raise self._exc
                if self._check is not None:
                    if not self._check():
                        if self._fallback is not None:
                            self._result = self._fallback()
                            self._has_result = True
                            return self._result
                        self._exc = RuntimeError('native SSD read failed verification')
                        raise self._exc
                self._result = self._finish() if self._finish is not None else None
                self._has_result = True
                return self._result
        finally:
            with self._mu:
                if self._released:
                    self._keepalive = None

    def close(self):
        readers = getattr(self, '_readers', None)
        mu = getattr(self, '_mu', None)
        if readers is None or mu is None:
            return
        with mu:
            if self._released or self._want_release:
                return
            handle = self._handle
            if handle is None:
                self._want_release = True
                self._keepalive = None
                return
            self._waiters += 1
        did_cancel = False
        try:
            did_cancel = bool(readers._py.qz_rd_cancel(handle, self._tok))
            if not did_cancel:
                err = ctypes.c_int32(0)
                fidx = ctypes.c_int32(-1)
                readers._c.qz_rd_wait(handle, self._tok, -1, ctypes.byref(err), ctypes.byref(fidx))
        finally:
            with mu:
                self._waiters -= 1
                if did_cancel and not (self._has_result or self._exc is not None):
                    self._cancelled = True
                self._want_release = True
                rel = self._try_release_locked()
        if rel is not None:
            readers._py.qz_rd_release(rel, self._tok)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Readers:
    """Lock-free C++ `pread` pool holding open cached and uncached FDs per checkpoint file."""

    COUNTS = ('submitted', 'completed', 'cancelled', 'failed', 'bytes_read', 'refused')

    def __init__(self, workers):
        pair = libs()
        if pair is None:
            raise RuntimeError('native SSD reader library is unavailable')
        self._py, self._c = pair
        self.workers = max(0, int(workers))
        raw = self._c.qz_rd_new(self.workers)
        if not raw:
            raise RuntimeError('qz_rd_new returned NULL')
        self._raw_handle = ctypes.c_void_p(raw)
        self._lock = threading.Lock()
        self._closed = False
        self._last_counts = {k: 0 for k in self.COUNTS}
        self._file_ids = {}
        self._paths = []

    def _file_id_locked(self, path):
        key = str(Path(path).expanduser().resolve())
        fid = self._file_ids.get(key)
        if fid is not None:
            return fid
        if self._closed:
            raise RuntimeError('native reader pool closed')
        fid = int(self._py.qz_rd_file(self._raw_handle, key.encode('utf-8')))
        if fid < 0:
            raise RuntimeError(f'too many open files in native reader pool ({len(self._paths)})')
        self._file_ids[key] = fid
        self._paths.append(key)
        return fid

    def file_id(self, path):
        with self._lock:
            return self._file_id_locked(path)

    def path_of(self, fid):
        with self._lock:
            if 0 <= fid < len(self._paths):
                return self._paths[fid]
            return None

    def read(self, segments, finish=None, cache=True, priority=0, check=None, fallback=None):
        """Submit up to `MAX_SEGMENTS` `(path, offset, length, dst_ptr)` reads as one ticket.

        Returns `NativeReadFuture`, or `None` when all `SLAB * SLABS` tickets are currently in use.
        """
        nseg = len(segments)
        if nseg <= 0 or nseg > MAX_SEGMENTS:
            raise ValueError(f'native read expects 1..{MAX_SEGMENTS} segments, got {nseg}')
        f_arr = (ctypes.c_int32 * nseg)()
        o_arr = (ctypes.c_int64 * nseg)()
        l_arr = (ctypes.c_int64 * nseg)()
        d_arr = (ctypes.c_uint64 * nseg)()
        with self._lock:
            if self._closed:
                return None
            for idx, seg in enumerate(segments):
                path, offset, length, dst = seg[0], int(seg[1]), int(seg[2]), int(seg[3])
                f_arr[idx] = self._file_id_locked(path)
                o_arr[idx] = offset
                l_arr[idx] = length
                d_arr[idx] = dst
            tok = int(self._py.qz_rd_submit(
                self._raw_handle,
                int(priority),
                1 if cache else 0,
                nseg,
                f_arr,
                o_arr,
                l_arr,
                d_arr,
            ))
        if tok == -1:
            return None
        if tok < 0:
            raise ValueError(f'invalid native read segments (code {tok})')
        return NativeReadFuture(self, tok, finish=finish, check=check, fallback=fallback, keepalive=tuple(segments))

    def _counts_locked(self):
        if self._closed:
            return dict(self._last_counts)
        arr = (ctypes.c_int64 * len(self.COUNTS))()
        self._py.qz_rd_counts(self._raw_handle, arr)
        return dict(zip(self.COUNTS, (int(arr[i]) for i in range(len(self.COUNTS)))))

    def counts(self):
        with self._lock:
            return self._counts_locked()

    def c_entrypoints(self):
        """Return C-callable `(handle, submit, status, release, times, backlog, workers)` addresses."""
        if self._closed:
            return None
        addr = lambda name: int(ctypes.cast(getattr(self._c, name), ctypes.c_void_p).value)  # noqa: E731
        return (
            int(self._raw_handle.value),
            addr('qz_rd_submit'),
            addr('qz_rd_status'),
            addr('qz_rd_release'),
            addr('qz_rd_times'),
            addr('qz_rd_backlog'),
            self.workers,
        )

    def close(self):
        lock = getattr(self, '_lock', None)
        if lock is None:
            return
        raw_handle = None
        with lock:
            if not self._closed:
                self._last_counts = self._counts_locked()
                self._closed = True
                raw_handle = self._raw_handle
        if raw_handle is not None:
            self._c.qz_rd_free(raw_handle)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
