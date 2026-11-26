# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A greedy decode token's head on CUDA: lm_head's rows as MLX's qmv computes them, and their argmax,
in two kernels, bit for bit.

quantized_argmax() is the method for a step head() takes (one token, no mask, an affine 4-bit
bfloat16 head with biases that cuda_qmv takes); the first CHECKS times a shape is met its token and
its rows are compared with the original's, and it serves the shape only if they are the same.
QUANTEEZE_CUDA_HEAD=off keeps mlx-vlm's.
"""
import sys
from functools import lru_cache

import mlx.core as mx

from . import cuda_kernels, cuda_qmv, kernel_checks, per_thread

CHECKS = 3
ON = [True]                     # QUANTEEZE_CUDA_HEAD
BITS = (4, 8)                   # bits head() accepts (qz_qmv4_row, qz_qmv8_row)
STATS = dict(served=0, checked=0, mismatched=0)
CHOICE, CHECKED = kernel_checks.tables('cuda_head')  # (K, N, group size[, 8 bits]): whether it matched the original
SAID = []

HEADER = cuda_qmv.HEADER + r"""
// a row's bfloat16 value v as MLX's ArgMax orders it (a NaN above all, 0 and -0 alike), above the
// complement of its index (the smaller of equals the larger key)
__device__ __forceinline__ unsigned long long qz_argmax_key(__nv_bfloat16 v, unsigned index) {
  const unsigned b = *reinterpret_cast<const unsigned short*>(&v);
  unsigned order;
  if ((b & 0x7FFFu) > 0x7F80u) {
    order = 0xFFFFu;
  } else if ((b & 0x7FFFu) == 0u) {
    order = 0x8000u;
  } else {
    order = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
  }
  return (static_cast<unsigned long long>(order) << 32) | (0xFFFFFFFFu - index);
}
"""

# A warp a row, 8 a block (MLX's qmv's launch); a block's best key out.
ROWS_SOURCE = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int wid = warp.meta_group_rank();
    const int row = ROW0 + blockIdx.x * 8 + wid;
    __shared__ unsigned long long keys[8];
    unsigned long long key = 0ull;
    if (row < ROW0 + N_OUT) {
      float sum;
      if constexpr (BITS == 8) {
        sum = qz_qmv8_row<K_IN, GS, true>(x, w, scales, biases, row);
      } else {
        sum = qz_qmv4_row<K_IN, GS>(x, w, scales, biases, row);
      }
      key = qz_argmax_key(__float2bfloat16_rn(sum), static_cast<unsigned>(row));
    }
    if (warp.thread_rank() == 0) {
      keys[wid] = key;
    }
    block.sync();
    if (block.thread_rank() == 0) {
      unsigned long long best = keys[0];
      for (int i = 1; i < 8; ++i) {
        best = keys[i] > best ? keys[i] : best;
      }
      best_keys[blockIdx.x] = best;
    }
"""

# One block of 1024: the largest of the blocks' keys, its index out (or the key, /*KEY*/).
BEST_SOURCE = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    constexpr int PER = (BLOCKS + 1023) / 1024;
    unsigned long long best = 0ull;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int i = static_cast<int>(block.thread_rank()) + j * 1024;
      if (i < BLOCKS) {
        const unsigned long long k = keys[i];
        best = k > best ? k : best;
      }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      const unsigned long long other = warp.shfl_down(best, offset);
      best = other > best ? other : best;
    }
    __shared__ unsigned long long warps[32];
    if (warp.thread_rank() == 0) {
      warps[warp.meta_group_rank()] = best;
    }
    block.sync();
    if (block.thread_rank() == 0) {
      for (int i = 1; i < static_cast<int>(warp.meta_group_size()); ++i) {
        best = warps[i] > best ? warps[i] : best;
      }
      /*OUT*/
    }
"""
_TOKEN = 'token[0] = 0xFFFFFFFFu - static_cast<uint32_t>(best & 0xFFFFFFFFull);'
_KEY = 'best_key[0] = best;'


@lru_cache(maxsize=None)
def _rows_kernel():
    return cuda_kernels.kernel('quanteeze_head_rows', ['x', 'w', 'scales', 'biases'], ['best_keys'], ROWS_SOURCE,
                               HEADER)


@lru_cache(maxsize=None)
def _best_kernel(key=False):
    if key:
        return cuda_kernels.kernel('quanteeze_head_best_key', ['keys'], ['best_key'],
                                   BEST_SOURCE.replace('/*OUT*/', _KEY))
    return cuda_kernels.kernel('quanteeze_head_best', ['keys'], ['token'], BEST_SOURCE.replace('/*OUT*/', _TOKEN))


def _best(keys, key=False):
    return _best_kernel(key)(inputs=[keys], template=[('BLOCKS', keys.size)], grid=(1024, 1, 1),
                             threadgroup=(1024, 1, 1), output_shapes=[(1,) if key else (1, 1)],
                             output_dtypes=[mx.uint64 if key else mx.uint32])[0]


def share(n):
    """All rows of a head of n."""
    return 0, n


def head(linear, x):
    """mx.argmax of linear(x) [1, 1] (uint32) for one token x [1, 1, K], the rows never written."""
    n, k = linear.weight.shape[0], x.shape[-1]
    first, count = share(n)
    blocks = max(1, -(-count // 8))
    keys = _rows_kernel()(inputs=[mx.contiguous(x.reshape(-1)), linear.weight, linear.scales, linear.biases],
                          template=[('GS', int(linear.group_size)), ('ROW0', first), ('N_OUT', count),
                                    ('K_IN', k), ('BITS', int(linear.bits))],
                          grid=(32 * blocks, 8, 1), threadgroup=(32, 8, 1), output_shapes=[(blocks,)],
                          output_dtypes=[mx.uint64])[0]
    return _best(keys)


def takes(linear, x, token_mask=None):
    """Whether head() computes this step's token as the original does: on CUDA, one token of bfloat16
    and no mask, an affine 4- or 8-bit head with biases (no bias added) that cuda_qmv's rows take."""
    return (token_mask is None and cuda_kernels.available() and isinstance(x, mx.array) and x.ndim == 3
            and x.shape[:2] == (1, 1) and type(linear).__name__ == 'QuantizedLinear' and 'bias' not in linear
            and isinstance(linear.get('scales'), mx.array) and getattr(linear, 'bits', None) in BITS
            and cuda_qmv.one_row(x, linear.weight, linear.scales, linear.get('biases'), True, linear.group_size,
                                 linear.bits, getattr(linear, 'mode', 'affine')))


def argmax(linear, x, token_mask, own):
    """own() (the original quantized_argmax) through head() where this shape matched it."""
    if not ON[0] or not takes(linear, x, token_mask):
        return own()
    bits = int(linear.bits)
    key = (x.shape[-1], linear.weight.shape[0], int(linear.group_size)) + ((bits,) if bits != 4 else ())
    choice, seen = kernel_checks.read(CHOICE, CHECKED, key)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS['served'] += 1
            return head(linear, x)
        return own()
    want = own()
    try:
        from mlx_vlm.models.quantized_verifier import singleton_quantized_linear
        got = head(linear, x)
        rows = cuda_qmv.qmv(x.reshape(1, -1), linear.weight, linear.scales, linear.biases, linear.group_size, bits)
        theirs = singleton_quantized_linear(linear, x)
        same = (want is not None and kernel_checks.same_bits(got, want)
                and kernel_checks.same_bits(rows.reshape(theirs.shape), theirs))
    except Exception as error:
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: the head\'s kernels unavailable ({type(error).__name__}: {error})', file=sys.stderr,
                  flush=True)
        same = False
    STATS['checked'] += 1
    STATS['mismatched'] += not same
    per_thread.checked(CHOICE, CHECKED, key, bool(same), CHECKS)
    return want


def install(cls=None):
    """mlx-vlm's Qwen3_5BatchInvariantForward (or `cls`, Qwen4-Exp's forward inherits it) takes a greedy
    decode's token through argmax() where it can.  Once a class."""
    if cls is None:
        from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward as cls
    if not cls.__dict__.get('_quanteeze_cuda_head', False):
        own = cls.quantized_argmax

        def quantized_argmax(self, linear, x, token_mask=None):
            return argmax(linear, x, token_mask, lambda: own(self, linear, x, token_mask=token_mask))

        cls.quantized_argmax = quantized_argmax
        cls._quanteeze_cuda_head = True
    return cls


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
