# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A decode token's routing in one CUDA kernel: the router's softmax, its top-k and their weights
renormalized, bit for bit what MLX's CUDA ops compute (quanteeze.arch.qwen4_exp's
_router_and_shared: softmax(precise=True), argpartition, take_along_axis, scores / their sum).

None is taken on trust: the first CHECKS times a shape is met (a block's first step, which runs
outside any compiled graph) the kernel's output is compared bit for bit with MLX's ops, and it
serves the shape only if they are the same.  QUANTEEZE_CUDA_ROUTER=off keeps MLX's ops.
"""
import sys
from functools import lru_cache

import mlx.core as mx

from . import cuda_kernels

CHECKS = 3
ON = [True]                     # QUANTEEZE_CUDA_ROUTER
STATS = dict(served=0, checked=0, mismatched=0)
CHOICE = {}                     # (experts, k, dtype): whether it matched MLX's ops
CHECKED = {}
SAID = []

HEADER = r"""
"""

SOURCE = r"""
    const int E = static_cast<int>(experts[0]);     // a row's experts: read, not a template's (one build a k)
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = warp.thread_rank();
    const int w = warp.meta_group_rank();
    const int nw = warp.meta_group_size();
    const T* in = logits + row * E;
    cg::greater<float> max_op;
    cg::plus<float> plus_op;
    // MLX's softmax kernel, a row of E with BLOCK threads, eight values each
    float vals[8];
    for (int i = 0; i < 8; ++i) {
      int c = t * 8 + i;
      vals[i] = c < E ? static_cast<float>(in[c]) : -cuda::std::numeric_limits<float>::infinity();
    }
    float maxval = cuda::std::numeric_limits<float>::lowest();
    float normalizer = 0.0f;
    float prevmax = maxval;
    for (int i = 0; i < 8; ++i) {
      maxval = max_op(maxval, vals[i]);
    }
    normalizer = normalizer * __expf(prevmax - maxval);
    for (int i = 0; i < 8; ++i) {
      normalizer = normalizer + __expf(vals[i] - maxval);
    }
    prevmax = maxval;
    maxval = cg::reduce(warp, maxval, max_op);
    normalizer = normalizer * __expf(prevmax - maxval);
    normalizer = cg::reduce(warp, normalizer, plus_op);
    __shared__ float local_max[32];
    __shared__ float local_normalizer[32];
    prevmax = maxval;
    if (lane == 0) {
      local_max[w] = maxval;
    }
    block.sync();
    maxval = lane < nw ? local_max[lane] : -cuda::std::numeric_limits<float>::infinity();
    maxval = cg::reduce(warp, maxval, max_op);
    normalizer = normalizer * __expf(prevmax - maxval);
    if (lane == 0) {
      local_normalizer[w] = normalizer;
    }
    block.sync();
    normalizer = lane < nw ? local_normalizer[lane] : 0.0f;
    normalizer = cg::reduce(warp, normalizer, plus_op);
    normalizer = 1 / normalizer;
    // each gate rounded, keyed (its bits, index + 1): a softmax gate is >= 0, whose bits order as its
    // values do, and the index breaks ties as the stable sort does; 0 marks none or taken
    unsigned int key[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int c = t * 8 + i;
      float v = c < E ? static_cast<float>(in[c]) : 0.0f;
      T gate = static_cast<T>(__expf(v - maxval) * normalizer);
      key[i] = c < E ? (static_cast<unsigned int>(__bfloat16_as_ushort(gate)) << 16) | static_cast<unsigned int>(c + 1)
                     : 0u;
    }
    // the K greatest, one a round: each thread's best left, the warp's by redux, the warps' through
    // shared memory, a buffer a round
    __shared__ unsigned int best[2][32];
    unsigned int picked[K];
#pragma unroll
    for (int r = 0; r < K; ++r) {
      unsigned int b = 0u;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        b = key[i] > b ? key[i] : b;
      }
      b = __reduce_max_sync(0xffffffffu, b);
      if (nw > 1) {
        if (lane == 0) {
          best[r & 1][w] = b;
        }
        block.sync();
        b = best[r & 1][0];
        for (int j = 1; j < nw; ++j) {
          b = best[r & 1][j] > b ? best[r & 1][j] : b;
        }
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (key[i] == b) {
          key[i] = 0u;
        }
      }
      picked[r] = b;
    }
    float pv[K];
    int pi[K];
#pragma unroll
    for (int r = 0; r < K; ++r) {
      pv[r] = static_cast<float>(__ushort_as_bfloat16(static_cast<unsigned short>(picked[r] >> 16)));
      pi[r] = static_cast<int>(picked[r] & 0xffffu) - 1;
    }
    // ascending, as argpartition's last K; their sum as row_reduce_simple adds K <= 16 bfloat16
    // (the first eight in order from 0, the rest, then the two added), each divided by it
    if (t == 0) {
      T first = static_cast<T>(0.0f);
      T second = static_cast<T>(0.0f);
#pragma unroll
      for (int j = 0; j < K; ++j) {
        T sj = static_cast<T>(pv[K - 1 - j]);
        if (j < 8) {
          first = Add{}(first, sj);
        } else {
          second = Add{}(second, sj);
        }
      }
      T total = K > 8 ? Add{}(first, second) : first;
#pragma unroll
      for (int j = 0; j < K; ++j) {
        indices[row * K + j] = static_cast<uint32_t>(pi[K - 1 - j]);
        scores[row * K + j] = Divide{}(static_cast<T>(pv[K - 1 - j]), total);
      }
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return cuda_kernels.kernel('quanteeze_route', ['logits', 'experts'], ['indices', 'scores'], SOURCE, HEADER)


_COUNTS = {}                    # experts: their count as an int32 array [1], made once (route())


def _count(e):
    n = _COUNTS.get(e)
    if n is None:
        n = _COUNTS[e] = mx.array([e], dtype=mx.int32)
        mx.eval(n)
    return n


def _threads(experts):
    """MLX's softmax block for a row of `experts`: ceil(experts / 8) threads, a warp's power of two."""
    need, block = -(-experts // 8), 32
    while block < need and block < 1024:
        block *= 2
    return block


def takes(logits, k):
    """Whether route() computes these rows' routing as MLX's ops do: bfloat16 logits of at most
    8192 experts (an index in 16 bits), top 16 at most, 1 to 64 rows."""
    if not cuda_kernels.available() or logits.dtype != mx.bfloat16 or logits.ndim < 1:
        return False
    e = logits.shape[-1]
    rows = logits.size // e if e > 0 else 0
    return e > 0 and logits.size == rows * e and 1 <= rows <= 64 and 0 < k <= min(16, e) and e <= 8192


def route(logits, k):
    """(indices, scores) of router logits rows: argpartition's top k of their precise softmax
    and those gates renormalized, in one kernel."""
    e = logits.shape[-1]
    rows = max(1, logits.size // e)
    block = _threads(e)
    shape = (*logits.shape[:-1], k)
    return tuple(_kernel()(inputs=[mx.contiguous(logits), _count(e)], template=[('T', logits.dtype), ('K', int(k))],
                           grid=(block * rows, 1, 1), threadgroup=(block, 1, 1), output_shapes=[shape, shape],
                           output_dtypes=[mx.uint32, logits.dtype]))


def original(logits, k):
    """MLX's ops, as _router_and_shared runs them."""
    gates = mx.softmax(logits, axis=-1, precise=True)
    indices = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
    scores = mx.take_along_axis(gates, indices, axis=-1)
    return indices, scores / scores.sum(axis=-1, keepdims=True)


def routed(logits, k):
    """(indices, scores) through route() where this shape matched MLX's ops, else theirs."""
    if not ON[0] or not takes(logits, k):
        return original(logits, k)
    e = logits.shape[-1]
    rows = max(1, logits.size // e)
    key = (e, int(k), str(logits.dtype)) if rows == 1 else (rows, e, int(k), str(logits.dtype))
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS['served'] += 1
            return route(logits, k)
        return original(logits, k)
    want = original(logits, k)
    try:
        got = route(logits, k)
        same = all(mx.array_equal(a, b).item() for a, b in zip(got, want))
    except Exception as error:
        if 'transformations' in str(error):                 # traced (mx.compile): nothing to compare yet
            return want
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: the routing kernel unavailable ({type(error).__name__}: {error})', file=sys.stderr,
                  flush=True)
        same = False
    STATS['checked'] += 1
    STATS['mismatched'] += not same
    CHOICE[key], CHECKED[key] = bool(same), (seen + 1 if same else CHECKS)
    return want


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
