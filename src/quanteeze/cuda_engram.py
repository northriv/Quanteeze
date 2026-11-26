# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""DeepSeek V4.1's Engram rows and token embedding read by the GPU from the host's copy of their tables, so that the
host never waits for the token there (QUANTEEZE_CUDA_PLE, as Flash-Next's n-gram rows: quanteeze.cuda_ple).

A V4.1 decode step looks rows up on the host twice before its first layer is built: the token's embedding
(quanteeze.arch.deepseek_v41.SSDEmbedding: np.asarray(ids)) and the Engram's (mlx-vlm's
OffloadedEngramEmbedding.__call__: np.asarray(indices), in layers 1 and 14).  Each waits for the token, so the
host built a step only once the GPU had finished the one before, though mlx-vlm's loop builds a step while the
GPU runs the last.

Here the host's copy of a table (quanteeze.ngram_prefetch.HostCopy, which holds V4.1's Engram and token
embedding tables when budget allows) is allocated pinned and mapped into the GPU's address space (cuda_ple.empty),
and once every array of a table is there a lookup is one kernel, its row ids left on the GPU: a thread a word of
a row reads it over PCIe.  The Engram's rows are mxfp8 (e4m3 values, an e8m0 scale a group of 32) and come back as
mx.dequantize gives them on CUDA (fp_dequantize: bfloat16(scale * float(e4m3)), the e4m3 value by its bits
as quanteeze.cuda_qmv's qmv_fp8 makes it); the embedding's are bfloat16 as they are.  Lookups of up to MAX_ROWS
rows (a decode step's 24 a layer; a prefill's stay on the host, where the wait costs a chunk nothing).  The first
CHECKS lookups of a shape are compared with the host's, and it serves the shape only if they are the same.
"""
import sys
from functools import lru_cache

import mlx.core as mx
import numpy as np

from . import cuda_kernels, cuda_ple, kernel_checks, per_thread

CHECKS = 3
MAX_ROWS = 256
STATS = dict(engram=0, embedding=0, checked=0, mismatched=0)
CHOICE, CHECKED = kernel_checks.tables('cuda_engram')  # (kind, row width): whether it matched the host's
SAID = []
MXFP8 = dict(group_size=32, bits=8, mode='mxfp8')

# a word of an Engram row: its 4 e4m3 values, its group's e8m0 scale, as MLX's fp_dequantize computes them
ENGRAM_SOURCE = r"""
  int64_t n = 1;
  for (int d = 0; d < ids_ndim; ++d) n *= ids_shape[d];
  constexpr int WORDS = W / 4;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= n * WORDS) return;
  const int64_t r = t / WORDS;
  const int k = static_cast<int>(t - r * WORDS);
  const int64_t id = ids[r];
  const uint32_t w = reinterpret_cast<const uint32_t*>(pointers[0] + static_cast<uint64_t>(id) * W)[k];
  const uint32_t sb = reinterpret_cast<const uint8_t*>(pointers[1] + static_cast<uint64_t>(id) * (W / 32))[(4 * k) / 32];
  const float scale = sb == 0 ? __uint_as_float(0x00400000u) : sb == 255 ? __uint_as_float(0x7fc00000u)
                                                              : __uint_as_float(sb << 23);
  uint16_t* o = reinterpret_cast<uint16_t*>(out) + r * W + 4 * k;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t b = (w >> (8 * i)) & 0xffu;
    const float v = (b & 0x7fu) == 0x7fu ? __uint_as_float(0x7fc00000u)
                                         : __uint_as_float(((b & 0x80u) << 24) | ((b & 0x7fu) << 20)) * 0x1p120f;
    o[i] = __bfloat16_as_ushort(__float2bfloat16_rn(v * scale));
  }
"""

# 16 bytes of a bfloat16 row
ROWS_SOURCE = r"""
  int64_t n = 1;
  for (int d = 0; d < ids_ndim; ++d) n *= ids_shape[d];
  constexpr int PARTS = W / 8;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= n * PARTS) return;
  const int64_t r = t / PARTS;
  const int k = static_cast<int>(t - r * PARTS);
  const int64_t id = ids[r];
  reinterpret_cast<uint4*>(out)[r * PARTS + k] =
      reinterpret_cast<const uint4*>(pointers[0] + static_cast<uint64_t>(id) * (W * 2))[k];
"""


@lru_cache(maxsize=None)
def _engram_kernel():
    return cuda_kernels.kernel('quanteeze_engram_rows', ['ids', 'pointers'], ['out'], ENGRAM_SOURCE)


@lru_cache(maxsize=None)
def _rows_kernel():
    return cuda_kernels.kernel('quanteeze_bf16_rows', ['ids', 'pointers'], ['out'], ROWS_SOURCE)


def _grid(threads, block=128):
    return dict(grid=(-(-threads // block) * block, 1, 1), threadgroup=(block, 1, 1))


def _kind(embedding):
    """'engram' for an OffloadedEngramEmbedding in mxfp8 (weight uint32, scales uint8, no biases), 'embedding'
    for an SSDEmbedding (bfloat16 rows), whose widths the kernels take; None for any other."""
    arrays = getattr(embedding, '_arrays', None)
    if isinstance(arrays, dict) and not hasattr(embedding, '_interleaved'):
        if getattr(embedding, '_quantization', None) != MXFP8 or set(arrays) != {'weight', 'scales'}:
            return None
        w, s = arrays['weight'], arrays['scales']
        ok = (w.dtype == np.uint32 and s.dtype == np.uint8 and w.ndim == s.ndim == 2 and w.shape[0] == s.shape[0]
              and w.shape[1] * 4 == s.shape[1] * 32)
        return 'engram' if ok else None
    rows = vars(embedding).get('_rows') if hasattr(embedding, '__dict__') else None
    if isinstance(rows, np.ndarray) and rows.dtype == np.uint16 and rows.ndim == 2 and rows.shape[1] % 8 == 0:
        return 'embedding'
    return None


def reads(embedding):
    """Whether a lookup of this table could be read by the GPU once held pinned (quanteeze.ngram_prefetch)."""
    return _kind(embedding) is not None


def _table(embedding):
    """(kind, the row's values, the arrays' device pointers) once every array of the table is held pinned; None
    until then.  Kept on it: what is held is not swapped again."""
    table = vars(embedding).get('_quanteeze_pinned') if hasattr(embedding, '__dict__') else None
    if table is not None:
        return table
    kind = _kind(embedding)
    if kind == 'engram':
        arrays = [embedding._arrays['weight'], embedding._arrays['scales']]
        width = arrays[0].shape[1] * 4
    elif kind == 'embedding':
        arrays = [embedding._rows]
        width = arrays[0].shape[1]
    else:
        return None
    where = [cuda_ple._device(a) for a in arrays]
    if None in where or where[0] % 16:
        return None
    table = (kind, width, mx.array(where, dtype=mx.uint64))
    object.__setattr__(embedding, '_quanteeze_pinned', table)
    return table


def rows(table, ids):
    """The rows `ids` names, [*ids.shape, width] in bfloat16, read by the GPU from the pinned table."""
    kind, width, pointers = table
    threads = ids.size * (width // 4 if kind == 'engram' else width // 8)
    kernel = _engram_kernel() if kind == 'engram' else _rows_kernel()
    return kernel(inputs=[ids.astype(mx.int64), pointers], template=[('W', width)], **_grid(threads),
                  output_shapes=[(*ids.shape, width)], output_dtypes=[mx.bfloat16])[0]


def lookup(embedding, ids, own):
    """What `own` (the host's lookup) gives for `ids`, read by the GPU where it can (CUDA, the whole table
    pinned, at most MAX_ROWS rows, this shape matched the host's); None where it cannot."""
    if not cuda_ple.ON[0] or not isinstance(ids, mx.array) or not 0 < ids.size <= MAX_ROWS:
        return None
    if not cuda_kernels.available():
        return None
    table = _table(embedding)
    if table is None:
        return None
    key = (table[0], table[1])
    choice, seen = kernel_checks.read(CHOICE, CHECKED, key)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS[table[0]] += 1
            return rows(table, ids)
        return None
    want = own(embedding, ids)
    try:
        got = rows(table, ids)
        same = kernel_checks.same_bits(got, want)
    except Exception as error:
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: the Engram\'s and embedding\'s row kernels unavailable ({type(error).__name__}: '
                  f'{error})', file=sys.stderr, flush=True)
        same = False
    STATS['checked'] += 1
    STATS['mismatched'] += not same
    per_thread.checked(CHOICE, CHECKED, key, bool(same), CHECKS)
    return want


def _wrap(cls):
    if cls.__dict__.get('_quanteeze_cuda_engram', False):
        return
    own = cls.__call__

    def call(self, ids):
        got = lookup(self, ids, own)
        return own(self, ids) if got is None else got

    cls.__call__ = call
    cls._quanteeze_cuda_engram = True


def install(*classes):
    """mlx-vlm's OffloadedEngramEmbedding and quanteeze's SSDEmbedding (or `classes`) look up through lookup()
    where they can, through their own __call__ elsewhere.  Once a class."""
    if not classes:
        from mlx_vlm.models.deepseek_v41.engram import OffloadedEngramEmbedding

        from .arch.deepseek_v41 import SSDEmbedding
        classes = (OffloadedEngramEmbedding, SSDEmbedding)
    for cls in classes:
        _wrap(cls)
    return classes


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
