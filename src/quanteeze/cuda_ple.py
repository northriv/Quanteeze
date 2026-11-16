# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A decode step's n-gram (PLE) rows read by the GPU from the host's copy of the table, so that the host
never waits for the token there (QUANTEEZE_CUDA_PLE).

Here the host's copy of a table (quanteeze.ngram_prefetch.HostCopy) is allocated pinned and mapped
into the GPU's address space (empty(): cuMemHostAlloc, the driver's, since MLX links CUDA's runtime
statically), and once every array of the table is there a lookup is one kernel, its row ids left on
the GPU: a thread a word of a row finds the row's shard, reads the word, its group's scale and bias
over PCIe and dequantizes its eight values as MLX's affine_dequantize does (scale * value + bias in
bfloat16), so the rows are mx.dequantize's, bit for bit.
"""
import atexit
import ctypes
import sys
from collections import namedtuple
from functools import lru_cache

import mlx.core as mx
import numpy as np

from . import cuda_kernels

CHECKS = 3
MAX_ROWS = 256
ON = [True]                     # QUANTEEZE_CUDA_PLE
HOST = [True]                   # QUANTEEZE_HOST_NGRAM
TOKENS = [None]
PENDING = [None]
STATS = dict(ngram=0, served=0, checked=0, mismatched=0, pinned_bytes=0)
HOST_STATS = dict(served=0, checked=0, mismatched=0, context_read=0, verified=0, wrong=0)
CHOICE = {}                     # (shards, row width, group size, rows): whether it matched the host's
CHECKED = {}
SAID = []
PIN_SAID = []                   # what pinned() could not pin, said once each
_DRIVER = []                    # [(libcuda, the device's primary context)] or [None]
_INTEGRATED = []                # [bool] for CU_DEVICE_ATTRIBUTE_INTEGRATED on device 0
_PCI_BUS_ID = []                # [str | None] lowercase PCI BDF of CUDA device 0
_EXITING = []

_CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
PORTABLE_DEVICEMAP = 0x01 | 0x02    # CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP
Table = namedtuple('Table', 'starts pointers shards group_size strides')

HEADER = r"""
// a word of row `id` (its 8 values from the k-th) as MLX's affine_dequantize gives them: the row's shard
// found among the S whose first rows `starts` holds, the word, its group's scale and bias read where
// `pointers` says (each shard's weight, scales and biases; host memory the GPU maps)
template <typename T, int S, int W, int GS, int WSTRIDE, int SSTRIDE, int BSTRIDE>
__device__ __forceinline__ void qz_ple_word(T* out, int64_t id, int k, const int64_t* starts,
                                            const uint64_t* pointers) {
  int lo = 0;
  int hi = S;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (starts[mid] <= id) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const int64_t local = id - starts[lo];
  const uint8_t* wrow = reinterpret_cast<const uint8_t*>(pointers[3 * lo]) + local * WSTRIDE;
  const uint8_t* srow = reinterpret_cast<const uint8_t*>(pointers[3 * lo + 1]) + local * SSTRIDE;
  const uint8_t* brow = reinterpret_cast<const uint8_t*>(pointers[3 * lo + 2]) + local * BSTRIDE;
  const uint32_t w = reinterpret_cast<const uint32_t*>(wrow)[k];
  const int g = (8 * k) / GS;
  const T scale = reinterpret_cast<const T*>(srow)[g];
  const T bias = reinterpret_cast<const T*>(brow)[g];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint8_t d = (w >> (4 * i)) & 0x0f;
    out[i] = scale * static_cast<T>(d) + bias;
  }
}
"""

ROW = 'qz_ple_word<T, S, W, GS, WSTRIDE, SSTRIDE, BSTRIDE>'

SOURCE = r"""
    constexpr int WORDS = W / 8;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N * WORDS) {
      return;
    }
    const int r = t / WORDS;
    const int k = t - r * WORDS;
    ROW(values + int64_t(r) * W + 8 * k, ids[r], k, starts, pointers);
""".replace('ROW', ROW)

NGRAM_SOURCE = r"""
    constexpr int L = CTX + 1;
    constexpr int WORDS = W / 8;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= CTX * HPN * WORDS) {
      return;
    }
    int64_t h[L];
#pragma unroll
    for (int q = 0; q < CTX; ++q) {
      h[q] = context[q];
    }
    h[CTX] = static_cast<int64_t>(token[0]);
    if (t == 0) {
#pragma unroll
      for (int q = 0; q < CTX; ++q) {
        window[q] = h[q + 1];
      }
    }
    int64_t previous = -1;
#pragma unroll
    for (int q = 0; q < L - 1; ++q) {
      if (h[q] == int64_t(EOS)) {
        previous = q;
      }
    }
    const int64_t in_segment = int64_t(L - 1) - (previous + 1);
    int64_t shifted[L];
#pragma unroll
    for (int shift = 0; shift < L; ++shift) {
      const int64_t source = int64_t(L - 1 - shift);
      const bool valid = in_segment >= shift && source >= 0;
      shifted[shift] = shift == 0 || valid ? h[source > 0 ? source : 0] : int64_t(EOS);
    }
    const int r = t / WORDS;
    const int k = t - r * WORDS;
    const int ngram = 2 + r / HPN;
    int64_t mixed = Multiply{}(shifted[0], multipliers[0]);
    for (int q = 1; q < ngram; ++q) {
      mixed = BitwiseXor{}(mixed, Multiply{}(shifted[q], multipliers[q]));
    }
    const int64_t id = Add{}(Remainder{}(mixed, sizes[r]), offsets[r]);
    ROW(values + int64_t(r) * W + 8 * k, id, k, starts, pointers);
""".replace('ROW', ROW)


@lru_cache(maxsize=None)
def _kernel():
    return cuda_kernels.kernel('quanteeze_ple_rows', ['ids', 'starts', 'pointers'], ['values'], SOURCE, HEADER)


@lru_cache(maxsize=None)
def _ngram_kernel():
    return cuda_kernels.kernel('quanteeze_ple_ngram', ['token', 'context', 'multipliers', 'sizes', 'offsets', 'starts',
                                                       'pointers'], ['values', 'window'], NGRAM_SOURCE, HEADER)


def _driver():
    """(libcuda, the primary context of device 0, the one MLX's runtime uses), or None where there is none."""
    if not _DRIVER:
        found = None
        try:
            cuda = ctypes.CDLL('libcuda.so.1')
            device, context = ctypes.c_int(), ctypes.c_void_p()
            if not (cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(device), 0)
                    or cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device)):
                cuda.cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
                cuda.cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p,
                                                              ctypes.c_uint]
                cuda.cuMemFreeHost.argtypes = [ctypes.c_void_p]
                cuda.cuCtxPushCurrent_v2.argtypes = [ctypes.c_void_p]
                cuda.cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
                if hasattr(cuda, 'cuMemAdvise'):
                    cuda.cuMemAdvise.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
                if hasattr(cuda, 'cuDeviceGetAttribute'):
                    cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
                if hasattr(cuda, 'cuDeviceGetPCIBusId'):
                    cuda.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
                found = (cuda, context)
        except (OSError, AttributeError):
            pass
        _DRIVER.append(found)
    return _DRIVER[0]


def gpu_room():
    """`(free_bytes, unused_pool_bytes)` on CUDA device 0 via `libcuda`, or `None` when unavailable."""
    found = _driver()
    if found is None:
        return None
    cuda, context = found
    free, total, device, pool = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_int(), ctypes.c_void_p()
    reserved, used = ctypes.c_uint64(), ctypes.c_uint64()
    cuda.cuCtxPushCurrent_v2(context)
    try:
        if (
            cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
            or cuda.cuDeviceGet(ctypes.byref(device), 0)
            or cuda.cuDeviceGetDefaultMemPool(ctypes.byref(pool), device)
            or cuda.cuMemPoolGetAttribute(pool, 5, ctypes.byref(reserved))
            or cuda.cuMemPoolGetAttribute(pool, 7, ctypes.byref(used))
        ):
            return None
    except AttributeError:
        return None
    finally:
        cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
    return int(free.value), max(0, int(reserved.value) - int(used.value))


def integrated():
    """Whether CUDA device 0 is an integrated GPU sharing physical memory with the host."""
    if _INTEGRATED:
        return _INTEGRATED[0]
    if mx.metal.is_available() or not cuda_kernels.available():
        _INTEGRATED.append(False)
        return False
    found = _driver()
    if found is None:
        _INTEGRATED.append(False)
        return False
    cuda, _ = found
    result = False
    try:
        device, val = ctypes.c_int(), ctypes.c_int()
        if (
            hasattr(cuda, 'cuDeviceGetAttribute')
            and not cuda.cuDeviceGet(ctypes.byref(device), 0)
            and not cuda.cuDeviceGetAttribute(ctypes.byref(val), _CU_DEVICE_ATTRIBUTE_INTEGRATED, device.value)
        ):
            result = bool(val.value)
    except (AttributeError, OSError):
        result = False
    _INTEGRATED.append(result)
    return result


def pci_bus_id():
    """Lowercase PCI bus-device-function string for CUDA device 0, or `None` when unavailable."""
    if _PCI_BUS_ID:
        return _PCI_BUS_ID[0]
    if mx.metal.is_available() or not cuda_kernels.available():
        _PCI_BUS_ID.append(None)
        return None
    found = _driver()
    if found is None:
        _PCI_BUS_ID.append(None)
        return None
    cuda, _ = found
    result = None
    try:
        device = ctypes.c_int()
        buf = ctypes.create_string_buffer(32)
        if (
            hasattr(cuda, 'cuDeviceGetPCIBusId')
            and not cuda.cuDeviceGet(ctypes.byref(device), 0)
            and not cuda.cuDeviceGetPCIBusId(buf, 32, device.value)
        ):
            text = buf.value.decode('ascii', errors='ignore').strip('\x00 \t\r\n').lower()
            result = text or None
    except (AttributeError, OSError):
        result = None
    _PCI_BUS_ID.append(result)
    return result


ROOM_SLACK = 1e9


def free_least():
    """`QUANTEEZE_GPU_FREE_GB` in bytes on CUDA GPUs, or `None` when disabled or not on CUDA."""
    from . import variables
    if variables.off('QUANTEEZE_GPU_FREE_GB') or mx.metal.is_available() or not cuda_kernels.available():
        return None
    return variables.number('QUANTEEZE_GPU_FREE_GB') * 1e9


def room_back(least):
    """Trim MLX and the CUDA driver memory pool (`mx.clear_cache()`) when free VRAM drops below
    `least` bytes while unused pool memory is at least `ROOM_SLACK`."""
    room = gpu_room()
    if room is None:
        return None
    if room[0] >= least or room[1] < ROOM_SLACK:
        return False
    try:
        mx.synchronize()
    except RuntimeError:
        pass
    mx.clear_cache()
    return True


_ADVISE_SET_PREFERRED_LOCATION = 3
_ADVISE_UNSET_PREFERRED_LOCATION = 4
_ADVISE_SET_ACCESSED_BY = 5
_ADVISE_UNSET_ACCESSED_BY = 6
_CU_DEVICE_CPU = -1


def advise_host_staging(np_view, enable=True):
    """Advise CUDA Unified Memory (`cuMemAdvise`) to keep a reusable `stream=mx.cpu` staging buffer
    in host RAM (`PreferredLocation = CPU`) with direct GPU PTEs (`AccessedBy = device 0`) so CPU
    writes and GPU `Scatter` reads do not ping-pong page faults across PCIe."""
    if mx.metal.is_available() or np_view is None or int(np_view.nbytes) < 4096:
        return False
    found = _driver() if cuda_kernels.available() else None
    if found is None:
        return False
    cuda, context = found
    if not hasattr(cuda, 'cuMemAdvise'):
        return False
    ptr = ctypes.c_uint64(int(np_view.ctypes.data))
    size = ctypes.c_size_t(int(np_view.nbytes))
    device = ctypes.c_int()
    cuda.cuCtxPushCurrent_v2(context)
    try:
        if cuda.cuDeviceGet(ctypes.byref(device), 0):
            return False
        if enable:
            r1 = cuda.cuMemAdvise(ptr, size, _ADVISE_SET_PREFERRED_LOCATION, _CU_DEVICE_CPU)
            r2 = cuda.cuMemAdvise(ptr, size, _ADVISE_SET_ACCESSED_BY, device.value)
            return not (r1 or r2)
        cuda.cuMemAdvise(ptr, size, _ADVISE_UNSET_ACCESSED_BY, device.value)
        cuda.cuMemAdvise(ptr, size, _ADVISE_UNSET_PREFERRED_LOCATION, _CU_DEVICE_CPU)
        return True
    except (AttributeError, OSError):
        return False
    finally:
        cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))


class _Pinned:
    """A cuMemHostAlloc allocation as numpy sees an array (its base), freed with the last array on it."""

    def __init__(self, shape, dtype, host, device):
        self.host, self.device = host, device
        self.__array_interface__ = dict(shape=tuple(shape), typestr=dtype.str, data=(host, False), version=3)

    def __del__(self):
        if _EXITING:
            return
        cuda, context = _DRIVER[0]
        cuda.cuCtxPushCurrent_v2(context)
        cuda.cuCtxSynchronize()
        cuda.cuMemFreeHost(self.host)
        cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
        STATS['pinned_bytes'] -= int(np.prod(self.__array_interface__['shape'])) * np.dtype(
            self.__array_interface__['typestr']).itemsize


atexit.register(lambda: _EXITING.append(True))


def empty(shape, dtype):
    """np.empty(shape, dtype) in the host's memory pinned and mapped for the GPU, which reads it where
    it is; np.empty's own where that cannot be (no CUDA, QUANTEEZE_CUDA_PLE off, the driver refused)."""
    return pinned(shape, dtype, "the n-gram table's copy", 'its lookups on the host') if ON[0] else np.empty(shape, dtype)


def pinned(shape, dtype, what='a copy', otherwise='read as before'):
    """np.empty(shape, dtype) pinned and mapped for the GPU (cuMemHostAlloc), whatever ON says;
    np.empty's own where it cannot be (no CUDA, the driver refused: said once, as `what` and
    `otherwise`).  Used for quanteeze.cuda_ple's n-gram tables."""
    dtype = np.dtype(dtype)
    size = int(np.prod(shape)) * dtype.itemsize
    driver = _driver() if size and cuda_kernels.available() else None
    if driver is None:
        return np.empty(shape, dtype)
    cuda, context = driver
    host, device = ctypes.c_void_p(), ctypes.c_uint64()
    cuda.cuCtxPushCurrent_v2(context)
    try:
        failed = cuda.cuMemHostAlloc(ctypes.byref(host), size, PORTABLE_DEVICEMAP)
        if not failed:
            failed = cuda.cuMemHostGetDevicePointer_v2(ctypes.byref(device), host, 0)
            if failed:
                cuda.cuMemFreeHost(host)
    finally:
        cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
    if failed:
        if what not in PIN_SAID:
            PIN_SAID.append(what)
            print(f'quanteeze: {what} not pinned for the GPU (cuMemHostAlloc: error {failed}); {otherwise}',
                  file=sys.stderr, flush=True)
        return np.empty(shape, dtype)
    STATS['pinned_bytes'] += size
    return np.asarray(_Pinned(shape, dtype, host.value, device.value))


def is_pinned(array):
    """Whether `array` lies in pinned memory created by `empty()` or `pinned()`."""
    return isinstance(array, np.ndarray) and _device(array) is not None


def _device(array):
    """Where the GPU reads `array`, a whole array empty() gave or a view of it whose rows are
    contiguous; None for any other."""
    base = array
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, _Pinned) or not array.flags.c_contiguous:
        return None
    return base.device + (array.__array_interface__['data'][0] - base.host)


def reads(embedding):
    """Whether a lookup of this table could be read by the GPU once held pinned: 4 bits, affine, in
    groups of a multiple of 8 values, bfloat16 scales and biases."""
    q = getattr(embedding, 'quantization', None)
    if not isinstance(q, dict) or not hasattr(embedding, '_interleaved'):
        return False
    group, width = int(q.get('group_size') or 0), int(getattr(embedding, 'row_width', 0) or 0)
    return (q.get('bits') == 4 and q.get('mode', 'affine') == 'affine' and group > 0 and group % 8 == 0
            and width > 0 and width % group == 0
            and tuple(getattr(embedding, '_tensor_layout', ())) == (('weight', 'U32'), ('scales', 'BF16'),
                                                                     ('biases', 'BF16')))


def _table(embedding):
    """The table's shards as the kernel reads them (their first rows, each tensor's pointer, the bytes a
    row of each), once every array of it is held pinned; None until then."""
    table = embedding.__dict__.get('_quanteeze_pinned')
    if table is not None or not reads(embedding):
        return table
    width, group = embedding.row_width, int(embedding.quantization['group_size'])
    if embedding._interleaved is not None:
        rows = embedding._interleaved
        base = _device(rows)
        if base is None:
            return None
        sizes = (width // 2, width // group * 2)
        starts, pointers = [0, rows.shape[0]], [base, base + sizes[0], base + sizes[0] + sizes[1]]
        strides = (rows.strides[0],) * 3
    else:
        starts, pointers, strides = [], [], None
        for start, end, arrays in embedding._shards:
            tensors = [arrays[name] for name in ('weight', 'scales', 'biases')]
            where = [_device(a) for a in tensors]
            row = tuple(a.strides[0] for a in tensors)
            if None in where or (strides is not None and row != strides):
                return None
            starts.append(start)
            pointers.extend(where)
            strides = row
        starts.append(embedding._shards[-1][1])
    if (any(p % 4 for p in pointers[::3]) or any(p % 2 for p in pointers) or strides[0] % 4
            or strides[1] % 2 or strides[2] % 2):
        return None
    table = Table(mx.array(starts, dtype=mx.int64), mx.array(pointers, dtype=mx.uint64), len(starts) - 1, group,
                  strides)
    embedding._quanteeze_pinned = table
    return table


def _template(embedding, table):
    return [('T', mx.bfloat16), ('S', table.shards), ('W', embedding.row_width), ('GS', table.group_size),
            ('WSTRIDE', table.strides[0]), ('SSTRIDE', table.strides[1]), ('BSTRIDE', table.strides[2])]


def _grid(threads, block=64):
    return dict(grid=(-(-threads // block) * block, 1, 1), threadgroup=(block, 1, 1))


def rows(embedding, table, row_ids):
    """The rows row_ids names, [..., row_width] in bfloat16, read by the GPU from the pinned table."""
    n, width = row_ids.size, embedding.row_width
    return _kernel()(inputs=[row_ids.astype(mx.int64).reshape(-1), table.starts, table.pointers],
                     template=_template(embedding, table) + [('N', n)], **_grid(n * width // 8),
                     output_shapes=[(*row_ids.shape, width)], output_dtypes=[mx.bfloat16])[0]


def ngram(module, table, token, context):
    """(the token's n-gram rows [1, 1, heads, row_width], its context's window slid [1, CTX]) as
    Qwen4ExpNGramEmbedding computes them for one token after `context`, in one kernel."""
    embedding, ctx = module.ngram_embedding, module.context_len
    heads = ctx * module.heads_per_ngram
    return _ngram_kernel()(
        inputs=[token, context, module.layer_multipliers, module.ngram_heads_vocab_sizes, module.ngram_heads_offsets,
                table.starts, table.pointers],
        template=_template(embedding, table) + [('CTX', ctx), ('HPN', module.heads_per_ngram),
                                                ('EOS', int(module.eos_token_id))],
        **_grid(heads * embedding.row_width // 8),
        output_shapes=[(1, 1, heads, embedding.row_width), (1, ctx)], output_dtypes=[mx.bfloat16, mx.int64])


def lookup(embedding, row_ids, own):
    """What `own` (the host's lookup) gives for row_ids, read by the GPU where it can (CUDA, the whole
    table pinned, at most MAX_ROWS rows, this shape matched the host's); None where it cannot."""
    if not ON[0] or not isinstance(row_ids, mx.array) or not 0 < row_ids.size <= MAX_ROWS:
        return None
    if not cuda_kernels.available():
        return None
    table = _table(embedding)
    if table is None:
        return None
    key = (table.shards, embedding.row_width, table.group_size, row_ids.size)
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS['served'] += 1
            return rows(embedding, table, row_ids)
        return None
    want = own(embedding, row_ids)
    try:
        got = rows(embedding, table, row_ids)
        same = got.shape == want.shape and mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)).item()
    except Exception as error:
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: the n-gram rows\' kernel unavailable ({type(error).__name__}: {error})',
                  file=sys.stderr, flush=True)
        same = False
    STATS['checked'] += 1
    STATS['mismatched'] += not same
    CHOICE[key], CHECKED[key] = bool(same), (seen + 1 if same else CHECKS)
    return want


def _one_token(module, input_ids, cache):
    context = None if cache is None else cache[3]
    eos = getattr(module, 'eos_token_id', None)
    return (
        isinstance(input_ids, mx.array) and input_ids.shape == (1, 1) and isinstance(context, mx.array)
        and context.dtype == mx.int64 and context.shape == (1, module.context_len)
        and 1 <= module.context_len <= 3 and module.ngram_size == module.context_len + 1
        and module.ngram_heads == module.context_len * module.heads_per_ngram
        and not getattr(cache, 'history_capacity', 0) and isinstance(eos, int) and 0 <= eos < 2 ** 31
    )


def embeds(module, input_ids, cache):
    """Whether embed() computes this step as `module` would: `_one_token()` with a whole pinned table."""
    return _one_token(module, input_ids, cache) and _table(module.ngram_embedding) is not None


def _constants(module):
    got = module.__dict__.get('_quanteeze_host_ngram')
    if got is None:
        got = tuple(
            np.array(a, dtype=np.int64)
            for a in (module.layer_multipliers, module.ngram_heads_vocab_sizes, module.ngram_heads_offsets)
        )
        module.__dict__['_quanteeze_host_ngram'] = got
    return got


def host_ids(module, context, token):
    """Compute 1-token n-gram row ids `[1, 1, heads]` on the host matching `Qwen4ExpNGramEmbedding`."""
    multipliers, sizes, offsets = _constants(module)
    ctx, eos = module.context_len, int(module.eos_token_id)
    h = np.array(list(context) + [int(token)], dtype=np.int64)
    last = len(h) - 1
    previous = max([q for q in range(last) if h[q] == eos], default=-1)
    in_segment = last - (previous + 1)
    shifted = np.array(
        [h[max(last - shift, 0)] if shift == 0 or (in_segment >= shift and last - shift >= 0) else eos
         for shift in range(last + 1)],
        dtype=np.int64,
    )
    products = shifted * multipliers[:last + 1]
    heads = np.arange(ctx * module.heads_per_ngram)
    ngram_sizes = 2 + heads // module.heads_per_ngram
    mixed = np.zeros(heads.size, dtype=np.int64)
    for i, n in enumerate(ngram_sizes.tolist()):
        m = products[0]
        for q in range(1, n):
            m = np.bitwise_xor(m, products[q])
        mixed[i] = m
    return (np.remainder(mixed, sizes) + offsets).reshape(1, 1, -1)


def _context(cache):
    window, held = cache.__dict__.get('_quanteeze_window', (None, None)), cache[3]
    if window[0] is held:
        return window[1]
    HOST_STATS['context_read'] += 1
    return np.array(held, dtype=np.int64).reshape(-1)


def _verify():
    pending, PENDING[0] = PENDING[0], None
    if pending is None:
        return
    ids, token = pending
    HOST_STATS['verified'] += 1
    actual = int(np.array(ids).reshape(-1)[0])
    if actual != token:
        HOST_STATS['wrong'] += 1
        HOST[0] = False
        print(
            f'quanteeze: n-gram ids built from token {token}, not input {actual}; switching to GPU ids',
            file=sys.stderr,
            flush=True,
        )


def host_embed(module, input_ids, cache, own):
    """Compute 1-token PLE n-gram rows using host-computed `host_ids` from `TOKENS` without GPU sync."""
    tokens, TOKENS[0] = TOKENS[0], None
    _verify()
    if (
        not HOST[0] or tokens is None or not isinstance(tokens, mx.array) or tokens.shape != (1,)
        or not _one_token(module, input_ids, cache)
    ):
        return None
    token = int(np.array(tokens)[0])
    context = _context(cache)
    ids = host_ids(module, context, token)
    window = np.concatenate([context, [token]])[-module.context_len:].astype(np.int64)
    key = ('host', module.context_len, module.heads_per_ngram)
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if not choice:
            return None
        HOST_STATS['served'] += 1
        values = module.ngram_embedding(ids)
        held = mx.array(window[None])
        cache[3] = held
        cache.__dict__['_quanteeze_window'] = (held, window)
        PENDING[0] = (input_ids, token)
        return values.reshape(*values.shape[:-2], -1)
    want = own(module, input_ids, cache)
    try:
        got = module.ngram_embedding(ids)
        got = got.reshape(*got.shape[:-2], -1)
        same = (
            int(np.array(input_ids).reshape(-1)[0]) == token and got.shape == want.shape
            and mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)).item()
            and np.array_equal(np.array(cache[3]).reshape(-1), window)
        )
    except Exception as error:
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: host n-gram ids unavailable ({type(error).__name__}: {error})',
                  file=sys.stderr, flush=True)
        same = False
    HOST_STATS['checked'] += 1
    HOST_STATS['mismatched'] += not same
    CHOICE[key], CHECKED[key] = bool(same), (seen + 1 if same else CHECKS)
    return want


def track_tokens(cls=None):
    """Hook `GenerationBatch._step` to expose already-evaluated `_next_tokens` in `TOKENS`."""
    if cls is None:
        from mlx_vlm.generate.ar import GenerationBatch as cls
    if cls.__dict__.get('_quanteeze_tokens', False):
        return cls
    own = cls._step

    def _step(self, *args, **kwargs):
        TOKENS[0] = getattr(self, '_next_tokens', None)
        try:
            return own(self, *args, **kwargs)
        finally:
            TOKENS[0] = None

    cls._step = _step
    cls._quanteeze_tokens = True
    return cls


def embed(module, input_ids, cache, own):
    """What `own` (Qwen4ExpNGramEmbedding.__call__) gives for a decode step's token, its n-gram ids and
    rows in one kernel (ngram()) where this shape matched `own`; None where embeds() says no."""
    if not ON[0] or not cuda_kernels.available() or not embeds(module, input_ids, cache):
        return None
    table = module.ngram_embedding._quanteeze_pinned
    key = ('ngram', table.shards, module.ngram_embedding.row_width, table.group_size, module.context_len,
           module.heads_per_ngram, str(input_ids.dtype))
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if not choice:
            return None
        STATS['ngram'] += 1
        values, window = ngram(module, table, input_ids, cache[3])
        cache[3] = window
        return values.reshape(1, 1, -1)
    context = cache[3]
    want = own(module, input_ids, cache)
    try:
        values, window = ngram(module, table, input_ids, context)
        same = (values.reshape(want.shape).shape == want.shape
                and mx.array_equal(values.reshape(want.shape).view(mx.uint16), want.view(mx.uint16)).item()
                and mx.array_equal(window, cache[3]).item())
    except Exception as error:
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: the n-gram kernel unavailable ({type(error).__name__}: {error})', file=sys.stderr,
                  flush=True)
        same = False
    STATS['checked'] += 1
    STATS['mismatched'] += not same
    CHOICE[key], CHECKED[key] = bool(same), (seen + 1 if same else CHECKS)
    return want


def install(cls=None, module_cls=None):
    """mlx-vlm's QuantizedMMapNGramEmbedding (or `cls`) looks up through lookup() where it can, through
    its own __call__ (quanteeze.ple_store's for its manifests) elsewhere; and its Qwen4ExpNGramEmbedding
    (or `module_cls`) embeds a decode step's token through embed() where it can.  Once a class."""
    if cls is None:
        from mlx_vlm.models.qwen4_exp.ple_storage import QuantizedMMapNGramEmbedding as cls
    if module_cls is None:
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpNGramEmbedding as module_cls
    if not cls.__dict__.get('_quanteeze_cuda_ple', False):
        own = cls.__call__

        def call(self, row_ids):
            got = lookup(self, row_ids, own)
            return own(self, row_ids) if got is None else got

        cls.__call__ = cls.lookup = call
        cls._quanteeze_cuda_ple = True
    if not module_cls.__dict__.get('_quanteeze_cuda_ple', False):
        own_embed = module_cls.__call__

        def embed_call(self, input_ids, cache):
            got = embed(self, input_ids, cache, own_embed)
            if got is None:
                got = host_embed(self, input_ids, cache, own_embed)
            return own_embed(self, input_ids, cache) if got is None else got

        module_cls.__call__ = embed_call
        module_cls._quanteeze_cuda_ple = True
    return cls


def summary():
    return dict(STATS, host=dict(HOST_STATS), shapes={str(k): v for k, v in CHOICE.items()})
