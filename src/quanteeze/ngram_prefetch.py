# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Bring the n-gram table's pages in from parallel threads before a lookup gathers its rows.

Flash-Next's n-gram (PLE) table, about 32 GB, stays on the SSD.  mlx-vlm's
QuantizedMMapNGramEmbedding gathers each lookup's rows from np.memmap arrays,
and every page not in the page cache is a page fault that waits on the SSD
alone.  A prefill chunk asks for rows scattered over the whole table, so a
cold prefill read one page at a time: about 300 MB/s, from an SSD that
serves random 16 KB reads at 2.1 GB/s when 16 are in flight.

DeepSeek V4.1's Engram tables (mlx-vlm's OffloadedEngramEmbedding, 203 GB in DeepSeek's own
checkpoint) are memmapped and gathered the same way: a 5,000-token prompt's two Engram layers
gathered some 120k rows, in 32 s of a 151 s prefill (on demand).

use() reads ahead for mlx-vlm's QuantizedMMapNGramEmbedding and OffloadedEngramEmbedding, once
however often it is called (the server and every script that loads Flash-Next call it; previously
each imported the class and installed it itself); install(embedding_cls) wraps a
class's _read_rows.  Before the gather, a pool of
threads reads the first and last element of every row the lookup misses,
each thread a slice of them, so the gather that follows finds their pages in
the page cache.  numpy's fancy indexing releases the GIL, so the threads'
page faults wait on the SSD together: 32k random pages took 0.23 s from 16
threads against 2.8 s from one, and 0.7 ms when they were cached.  The rows
come back as before.  A lookup of fewer than MIN_ROWS rows (a decode step)
goes a row a job: a decode step's 24 cold rows took 0.74 ms from 16 threads
against 2.4 ms in one, out of a step of some 25 ms.  Both of mlx-vlm's
layouts are covered: one interleaved row
store, or each tensor's range in the checkpoint's shards.
QUANTEEZE_PLE_THREADS sets the pool's size; 0 turns it off.

hold(model, budget, workers) is for a file system no read-ahead makes fast: on Lustre a read waits
about 0.2 s whatever it reads, and a lookup's rows lie anywhere in the table, so each new token's
are new reads (Flash-Next split over 4 A100s, a decode step 16 s, a 42-token cold
prefill 168 s on the rank holding the table; 0.12 s and 11 s with the checkpoint in /dev/shm).  It
copies the tables into the host's memory, read once, in parts of PART_BYTES by `workers` threads
past the file cache, and each memmap is swapped for its copy as the copy is read, so lookups go to
the file until then and to memory after.  The read-ahead leaves the copies alone.  On CUDA a table
the GPU can read (quanteeze.cuda_ple.reads) is copied into pinned memory it maps, where a decode
step's lookup then reads its rows.
"""
import fcntl
import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

from . import cuda_ple, variables

_USED = {}
_HELD = []
STATS = {'again': 0}
THREADS = 16
MIN_ROWS = 64
PART_ROWS = 512          # rows a thread takes at least
PART_BYTES = 64 << 20    # a read of the host's copy at most
ADVISE = [None]          # QUANTEEZE_NGRAM_ADVISE


def _touch(job):
    """Read each row's first and last element, so every page the rows lie on."""
    array, rows = job
    return int(array[rows, 0].sum()) + int(array[rows, -1].sum())


def _advising():
    if ADVISE[0] is None:
        ADVISE[0] = hasattr(mmap, 'MADV_WILLNEED') and not variables.off('QUANTEEZE_NGRAM_ADVISE')
    return ADVISE[0]


def _madvise(mm, start, length):
    mm.madvise(mmap.MADV_WILLNEED, start, length)


def _base(mm):
    return np.frombuffer(mm, np.uint8, count=1).ctypes.data


def _advise(located):
    """Advise the OS kernel via `madvise(MADV_WILLNEED)` for each row's page span on the calling thread."""
    plan = []
    for array, rows in located:
        mm = getattr(array, '_mmap', None)
        if mm is None or getattr(mm, 'closed', False) or array.ndim < 1 or not array.flags.c_contiguous:
            return False
        stride = array.strides[0]
        first = array.ctypes.data - _base(mm)
        if first < 0 or first + array.nbytes > len(mm):
            return False
        plan.append((mm, first, stride, rows))
    page = mmap.PAGESIZE
    for mm, first, stride, rows in plan:
        for r in np.asarray(rows, np.int64).reshape(-1).tolist():
            at = first + r * stride
            start = at - at % page
            _madvise(mm, start, at + stride - start)
    return True


class Prefetcher:
    """Brings the pages of memmapped rows into the page cache from a pool of threads, or via
    `madvise(MADV_WILLNEED)` for small lookups when `advise` is enabled."""

    def __init__(self, threads=THREADS, min_rows=MIN_ROWS, advise=None):
        self.threads = threads
        self.min_rows = min_rows
        self.advise = advise
        self._pool = ThreadPoolExecutor(threads, thread_name_prefix='ple-prefetch') if threads else None

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def _advised(self, located):
        if self._pool is None or not (_advising() if self.advise is None else self.advise):
            return False
        count = sum(rows.size for _, rows in located)
        return 0 < count < self.min_rows and _advise(located)

    def fetch(self, located):
        """Touch the rows in located, [(memmap, row indices)]; returns how many, or 0 if left alone."""
        if self._advised(located):
            return sum(rows.size for _, rows in located)
        jobs = self._jobs(located)
        if not jobs:
            return 0
        for _ in self._pool.map(_touch, jobs):
            pass
        return sum(rows.size for _, rows in located)

    def ahead(self, located):
        """Asynchronous read-ahead for `located` without blocking the caller."""
        if self._advised(located):
            return
        for job in self._jobs(located) or ():
            self._pool.submit(_touch, job)

    def _jobs(self, located):
        count = sum(rows.size for _, rows in located)
        if self._pool is None or not count:
            return None
        if count < self.min_rows:                   # a decode step: its few page faults overlap
            return [(array, rows[i:i + 1]) for array, rows in located for i in range(rows.size)]
        return [(array, part) for array, rows in located
                for part in np.array_split(rows, max(1, min(self.threads, rows.size // PART_ROWS)))]


_RANDOM = set()
_RANDOM_HELD = []


def read_randomly(array):
    """Disable OS sequential read-ahead (`MADV_RANDOM`) on non-Mac hosts for large random-access mmap tables."""
    mm = getattr(array, '_mmap', None)
    if mm is None or id(mm) in _RANDOM or _on_mac():
        return
    try:
        _advise_random(mm)
    except (AttributeError, OSError, ValueError):
        return
    _RANDOM.add(id(mm))
    _RANDOM_HELD.append(mm)


def _advise_random(mm):
    mm.madvise(mmap.MADV_RANDOM)


def _on_mac():
    import mlx.core as mx
    return mx.metal.is_available()


def located(embedding, row_ids):
    """The memmaps holding rows row_ids of a QuantizedMMapNGramEmbedding, or of an
    OffloadedEngramEmbedding (each of its arrays, _arrays), with their indices."""
    if getattr(embedding, '_arrays', None) is not None and not hasattr(embedding, '_interleaved'):
        out = [(array, row_ids) for array in embedding._arrays.values()]
    elif embedding._interleaved is not None:
        out = [(embedding._interleaved, row_ids)]
    else:
        out = []
        for start, end, arrays in embedding._shards:
            rows = row_ids[(row_ids >= start) & (row_ids < end)] - start
            if rows.size:
                out.extend((arrays[name], rows) for name in embedding._tensor_names)
    out = [(array, rows) for array, rows in out if isinstance(array, np.memmap)]     # not the host's copies
    for array, _ in out:
        read_randomly(array)
    return out


def _memmaps(embedding):
    """(where, key, memmap) of each memmap an embedding reads its rows from, where[key] being it:
    the arrays of an OffloadedEngramEmbedding, or of a QuantizedMMapNGramEmbedding's shards, or its
    interleaved store (the object's own attribute)."""
    if getattr(embedding, '_arrays', None) is not None and not hasattr(embedding, '_interleaved'):
        found = [(embedding._arrays, k, a) for k, a in embedding._arrays.items()]
    elif embedding._interleaved is not None:
        found = [(vars(embedding), '_interleaved', embedding._interleaved)]
    else:
        found = [(arrays, k, a) for _, _, arrays in embedding._shards for k, a in arrays.items()]
    return [(where, k, a) for where, k, a in found if isinstance(a, np.memmap) and a.size]


def tables(model):
    """The n-gram tables `model` holds, each once: mlx-vlm's QuantizedMMapNGramEmbedding (an
    attribute of a module) and OffloadedEngramEmbedding (a module)."""
    from mlx_vlm.models.deepseek_v41.engram import OffloadedEngramEmbedding
    from mlx_vlm.models.qwen4_exp.ple_storage import QuantizedMMapNGramEmbedding
    found = {}
    for module in model.modules():
        for value in (module, *vars(module).values()):
            if isinstance(value, (QuantizedMMapNGramEmbedding, OffloadedEngramEmbedding)):
                found[id(value)] = value
    return list(found.values())


def _read(file, offset, out):
    """`out`'s bytes from `file` at `offset`, past the file cache (F_NOCACHE on macOS; on Linux the
    pages dropped after, as q2_experts.read_rows does): the copy is what stays."""
    raw = memoryview(out.reshape(-1).view(np.uint8))
    fd = os.open(file, os.O_RDONLY)
    try:
        if hasattr(fcntl, 'F_NOCACHE'):
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        done = 0
        while done < len(raw):
            n = os.preadv(fd, [raw[done:done + (1 << 30)]], offset + done)
            if n <= 0:
                raise OSError(f'{file}: {len(raw) - done} bytes short at {offset + done}')
            done += n
        if not hasattr(fcntl, 'F_NOCACHE') and hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(fd, offset, len(raw), os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


class HostCopy:
    """The n-gram tables of `embeddings` in the host's memory, as many whole arrays as `budget`
    bytes hold, in order, but those whose file `skip(file)` says to leave (one in memory already).
    Each array is read in parts of PART_BYTES by `workers` threads, and swapped in for its memmap
    once read."""

    def __init__(self, embeddings, budget, workers, skip=lambda file: False):
        self.bytes = self.total = self._read = 0
        self.error = None
        self._entries = []
        self._pool = ThreadPoolExecutor(max(1, workers), thread_name_prefix='ngram-copy')
        for embedding in embeddings:
            empty = cuda_ple.empty if cuda_ple.reads(embedding) else np.empty     # pinned: the GPU reads it there
            for where, key, array in _memmaps(embedding):
                self.total += array.nbytes
                if self.bytes + array.nbytes > budget or skip(array.filename):
                    continue
                copy = empty(array.shape, array.dtype)
                row = array.nbytes // array.shape[0]
                step = max(1, PART_BYTES // row)
                parts = [self._pool.submit(_read, array.filename, array.offset + a * row, copy[a:a + step])
                         for a in range(0, array.shape[0], step)]
                self._entries.append((where, key, copy, parts))
                self.bytes += array.nbytes
        self._swapper = threading.Thread(target=self._swap, name='ngram-copy-swap', daemon=True)
        self._swapper.start()

    def _swap(self):
        for where, key, copy, parts in self._entries:
            wait(parts)
            if any(p.cancelled() for p in parts):
                return
            failed = next((p.exception() for p in parts if p.exception() is not None), None)
            if failed is not None:                       # this one read from the file, as before
                self.error = self.error or failed
                continue
            where[key] = copy
            self._read += copy.nbytes

    def unread(self):
        """Bytes to hold not read yet (the host's memory they will take)."""
        return self.bytes - self._read if self._swapper.is_alive() else 0

    def wait(self):
        """Wait for every array to be read and swapped in; returns the bytes held, raises the first
        read that failed."""
        self._swapper.join()
        if self.error is not None:
            raise self.error
        return self._read

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._swapper.join()


def hold(model, budget, workers, skip=lambda file: False):
    """Hold `model`'s n-gram tables in the host's memory (HostCopy), as many as `budget` bytes
    take, the copy before let go; returns the copy, or None where the model has no table."""
    let_go()
    found = tables(model)
    if not found:
        return None
    _HELD.append(HostCopy(found, budget, workers, skip))
    return _HELD[0]


def unread():
    """Bytes the host's copy of the tables will yet take."""
    return sum(held.unread() for held in _HELD)


def let_go():
    """Stop reading the host's copy of the tables; what was read stays with the model's tables."""
    while _HELD:
        _HELD.pop().close()


def close_all():
    """Stop active `HostCopy` readers and `Prefetcher` worker pools."""
    let_go()
    seen = set()
    for p in list(_USED.values()):
        if id(p) not in seen:
            seen.add(id(p))
            p.close()
    _USED.clear()


def _kept(rows):
    for a in (rows.values() if isinstance(rows, dict) else rows if isinstance(rows, (tuple, list)) else (rows,)):
        if isinstance(a, np.ndarray):
            a.flags.writeable = False
    return dict(rows) if isinstance(rows, dict) else rows


def install(embedding_cls, prefetcher=None):
    """Bring in the pages every lookup of `embedding_cls` will touch; returns the Prefetcher."""
    if prefetcher is None:
        prefetcher = Prefetcher(variables.integer('QUANTEEZE_PLE_THREADS'))
    read_rows = embedding_cls._read_rows

    def _read_rows(self, row_ids):
        small = isinstance(row_ids, np.ndarray) and row_ids.size < prefetcher.min_rows
        key = (row_ids.dtype.str, row_ids.shape, row_ids.tobytes()) if small else None
        last = self.__dict__.get('_quanteeze_last')
        if key is not None and last is not None and last[0] == key:
            STATS['again'] += 1
            return dict(last[1]) if isinstance(last[1], dict) else last[1]
        ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
        if getattr(self, 'cache_rows', False) and self._cache:
            ids = np.array([i for i in ids.tolist() if i not in self._cache], dtype=np.int64)
        prefetcher.fetch(located(self, ids))
        rows = read_rows(self, row_ids)
        if key is not None:
            object.__setattr__(self, '_quanteeze_last', (key, _kept(rows)))
        return rows

    embedding_cls._read_rows = _read_rows
    return prefetcher


def ahead(embedding, row_ids):
    """Start bringing in the pages of `embedding`'s `row_ids` asynchronously without blocking."""
    prefetcher = next(iter(_USED.values()), None)
    if prefetcher is not None and (getattr(embedding, '_arrays', None) is not None or hasattr(embedding, '_interleaved')):
        prefetcher.ahead(located(embedding, np.asarray(row_ids, dtype=np.int64).reshape(-1)))


def use():
    """The read-ahead on mlx-vlm's QuantizedMMapNGramEmbedding (Flash-Next's n-gram table) and
    OffloadedEngramEmbedding (DeepSeek V4.1's Engram tables), installed the first time, one pool
    for both; returns its Prefetcher."""
    from mlx_vlm.models.deepseek_v41.engram import OffloadedEngramEmbedding
    from mlx_vlm.models.qwen4_exp.ple_storage import QuantizedMMapNGramEmbedding
    prefetcher = next(iter(_USED.values()), None)
    for cls in (QuantizedMMapNGramEmbedding, OffloadedEngramEmbedding):
        if cls not in _USED:
            prefetcher = _USED[cls] = install(cls, prefetcher)
    return prefetcher
