# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Metal decode and speculative-verify routing and MoE weighted summation (`quanteeze.metal_router`).

Replaces MLX's multi-op routing chain (Softmax, ArgPartition, Slice, StopGradient, GatherAxis, Sum,
Divide) with one Metal kernel (`quanteeze_metal_route` for 1-token decode; `quanteeze_metal_route_rows`
for 2..`ROWS_MOST` verify tokens), and replaces the compiled output combination graph
`(output * scores[..., None]).sum(axis=-2) + shared` with a second kernel (`quanteeze_metal_weighted` /
`quanteeze_metal_weighted_rows`), reducing Python graph-building overhead and small-kernel GPU launch gaps.

Both paths match MLX's per-row arithmetic order bit-for-bit:
- Routing kernels: MLX's `softmax_single_row` (`precise=True`), stable ascending `(gate, index)`
  top-k selection matching `argpartition`, and sequential `T` reduction + division for weight
  renormalization (one threadgroup per token row).
- Weighted-sum kernels: column-wise float32 product rounded to `T`, pairwise row `j` + `j + 8`
  column reduction in `T` matching MLX's `sum(axis=-2)`, and residual `shared` addition in `T`
  (one grid row of threads per token row).

Every shape is verified bit-for-bit against the MLX reference for the first `CHECKS` invocations
(`kernel_checks`) before serving that shape, with verify-pass shapes keyed separately by row count.
1-token decode keeps its dedicated single-row kernels. `QUANTEEZE_METAL_ROUTER=off` disables all of them.
"""
import sys
from functools import lru_cache

import mlx.core as mx

from . import kernel_checks, per_thread

CHECKS = 3
ON = [True]                     # QUANTEEZE_METAL_ROUTER
STATS = dict(served=0, checked=0, mismatched=0)
CHOICE, CHECKED = kernel_checks.tables('metal_router')  # (experts, k, dtype) or (rows, experts, k, dtype)
SAID = []
MOST = 4096                     # maximum expert count: MLX's softmax_single_row limit (SOFTMAX_LOOPED_LIMIT)
ROWS_MOST = 8                   # maximum verify token rows per call (`arch.qwen4_exp.VERIFY_ROWS_MOST`)

_SOURCE = r"""
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    const int E = int(logits_shape[0]);
    const int lid = int(thread_position_in_threadgroup.x);
    const int lane = int(thread_index_in_simdgroup);
    const int sg = int(simdgroup_index_in_threadgroup);
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];
    threadgroup uint local_key[SIMD_SIZE];
    threadgroup uint picked[16];
    float ld[N_READS];
    for (int i = 0; i < N_READS; i++) {
      const int c = lid * N_READS + i;
      ld[i] = c < E ? float(logits[c]) : -INFINITY;
    }
    if (sg == 0) {
      local_max[lane] = -INFINITY;
      local_normalizer[lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxval = -FLT_MAX;
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < ld[i]) ? ld[i] : maxval;
    }
    maxval = simd_max(maxval);
    if (lane == 0) {
      local_max[sg] = maxval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      maxval = simd_max(local_max[lane]);
      if (lane == 0) {
        local_max[0] = maxval;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];
    float normalizer = 0;
    for (int i = 0; i < N_READS; i++) {
      float e = fast::exp(ld[i] - maxval);
      ld[i] = e;
      normalizer += e;
    }
    normalizer = simd_sum(normalizer);
    if (lane == 0) {
      local_normalizer[sg] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      normalizer = simd_sum(local_normalizer[lane]);
      if (lane == 0) {
        local_normalizer[0] = normalizer;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1 / local_normalizer[0];
    uint key[N_READS];
    for (int i = 0; i < N_READS; i++) {
      const int c = lid * N_READS + i;
      T gate = T(ld[i] * normalizer);
      key[i] = c < E ? ((uint(as_type<ushort>(gate)) + 1u) << 16) | uint(c) : 0u;
    }
    for (int j = 0; j < K; j++) {
      uint best = 0;
      for (int i = 0; i < N_READS; i++) {
        best = key[i] > best ? key[i] : best;
      }
      best = simd_max(best);
      if (lane == 0) {
        local_key[sg] = best;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) {
        uint all = simd_max(lane < int((threads_per_threadgroup.x + SIMD_SIZE - 1) / SIMD_SIZE) ? local_key[lane] : 0u);
        if (lane == 0) {
          picked[j] = all;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint got = picked[j];
      for (int i = 0; i < N_READS; i++) {
        key[i] = key[i] == got ? 0u : key[i];
      }
    }
    if (lid == 0) {
      T total = T(0);
      for (int j = K - 1; j >= 0; j--) {
        total = T(float(total) + float(as_type<T>(ushort((picked[j] >> 16) - 1u))));
      }
      for (int j = 0; j < K; j++) {
        const uint got = picked[j];
        ids[K - 1 - j] = got & 0xFFFFu;
        gates[K - 1 - j] = T(float(as_type<T>(ushort((got >> 16) - 1u))) / float(total));
      }
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name='quanteeze_metal_route',
        input_names=['logits'],
        output_names=['ids', 'gates'],
        source=_SOURCE,
    )


def _rows_source(source, replaced, head):
    """Derive a multi-row Metal kernel source from a single-row source."""
    for old, new in [*replaced, (head[0], head[0] + head[1].lstrip('\n'))]:
        if source.count(old) < 1 or (old != head[0] and source.count(old) != 1):
            raise ValueError(f'metal_router: {old!r} expected once in single-row source')
        source = source.replace(old, new, 1)
    return source


_ROWS_SOURCE = _rows_source(
    _SOURCE,
    [
        ('const int E = int(logits_shape[0]);', 'const int E = int(logits_shape[1]);'),
        ('float(logits[c])', 'float(row_logits[c])'),
        ('ids[K - 1 - j] =', 'row_ids[K - 1 - j] ='),
        ('gates[K - 1 - j] =', 'row_gates[K - 1 - j] ='),
    ],
    (
        'const int E = int(logits_shape[1]);\n',
        r"""
    const int row = int(threadgroup_position_in_grid.y);
    auto row_logits = logits + size_t(row) * size_t(E);
    auto row_ids = ids + size_t(row) * K;
    auto row_gates = gates + size_t(row) * K;
""",
    ),
)


@lru_cache(maxsize=None)
def _rows_kernel():
    return mx.fast.metal_kernel(
        name='quanteeze_metal_route_rows',
        input_names=['logits'],
        output_names=['ids', 'gates'],
        source=_ROWS_SOURCE,
    )


def _rows(logits, k):
    """Return token row count (`1..ROWS_MOST`) supported by `route()`, or `0` when unsupported."""
    if not ON[0] or not mx.metal.is_available() or logits.ndim < 1 or logits.dtype not in (mx.bfloat16, mx.float16):
        return 0
    e = logits.shape[-1]
    if not (0 < k <= 16 and k <= e <= MOST):
        return 0
    if logits.size == e:
        return 1
    if logits.ndim == 3 and logits.shape[0] == 1 and 1 < logits.shape[1] <= ROWS_MOST:
        return int(logits.shape[1])
    return 0


def takes(logits, k):
    """Whether `route()` supports `logits`: Metal bf16/f16 1-row or single-batch 2..`ROWS_MOST` verify rows."""
    return _rows(logits, k) > 0


def route(logits, k):
    """Return `(ids uint32 [..., k], scores [..., k])`: top-k softmax gates and renormalized weights."""
    e = logits.shape[-1]
    rows = logits.size // e
    threads = 32 * ((-(-e // 4) + 31) // 32)
    shape = (*logits.shape[:-1], k)
    if rows == 1:
        return tuple(
            _kernel()(
                inputs=[mx.contiguous(logits).reshape(-1)],
                template=[('T', logits.dtype), ('K', int(k))],
                grid=(threads, 1, 1),
                threadgroup=(threads, 1, 1),
                output_shapes=[shape, shape],
                output_dtypes=[mx.uint32, logits.dtype],
            )
        )
    return tuple(
        _rows_kernel()(
            inputs=[mx.contiguous(logits).reshape(rows, e)],
            template=[('T', logits.dtype), ('K', int(k))],
            grid=(threads, rows, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[shape, shape],
            output_dtypes=[mx.uint32, logits.dtype],
        )
    )


def pending(experts, k, dtype, x):
    """Return True while decode or verify shape (`1..ROWS_MOST` rows) still needs eager bit-for-bit verification."""
    if not (
        ON[0]
        and mx.metal.is_available()
        and x.ndim >= 2
        and x.shape[0] == 1
        and 0 < x.shape[1] <= ROWS_MOST
        and dtype in (mx.bfloat16, mx.float16)
        and 0 < k <= 16
        and k <= experts <= MOST
    ):
        return False
    rows = int(x.shape[1])
    key = (int(experts), int(k), str(dtype))
    choice, seen = kernel_checks.read(CHOICE, CHECKED, key if rows == 1 else (rows,) + key)
    return choice is None or seen < CHECKS


def routed(logits, k, theirs):
    """Compute `(ids, scores)` with `route()` once verified bit-for-bit `CHECKS` times against `theirs()`."""
    rows = _rows(logits, k)
    if not rows:
        return theirs()
    key = (int(logits.shape[-1]), int(k), str(logits.dtype))
    key = key if rows == 1 else (rows,) + key
    choice, seen = kernel_checks.read(CHOICE, CHECKED, key)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS['served'] += 1
            return route(logits, k)
        return theirs()
    want = theirs()
    try:
        got = route(logits, k)
        same = kernel_checks.same_bits(got, want)
    except Exception as error:
        if 'transformations' in str(error):
            return want
        if not SAID:
            SAID.append(True)
            print(
                f'quanteeze: the Metal routing kernel unavailable ({type(error).__name__}: {error})',
                file=sys.stderr,
                flush=True,
            )
        same = False
    STATS['checked'] += 1
    STATS['mismatched'] += not same
    per_thread.checked(CHOICE, CHECKED, key, bool(same), CHECKS)
    return want


_WEIGHTED = r"""
    const uint d = thread_position_in_grid.x;
    const uint D = uint(shared_shape[shared_ndim - 1]);
    if (d >= D) {
      return;
    }
    T part[8];
    for (int j = 0; j < 8 && j < K; j++) {
      part[j] = T(float(output[j * D + d]) * float(scores[j]));
      if (j + 8 < K) {
        part[j] = T(float(part[j]) + float(T(float(output[(j + 8) * D + d]) * float(scores[j + 8]))));
      }
    }
    T total = T(0);
    for (int j = 0; j < 8 && j < K; j++) {
      total = T(float(total) + float(part[j]));
    }
    out[d] = T(float(total) + float(shared[d]));
"""


@lru_cache(maxsize=None)
def _weighted_kernel():
    return mx.fast.metal_kernel(
        name='quanteeze_metal_weighted',
        input_names=['output', 'scores', 'shared'],
        output_names=['out'],
        source=_WEIGHTED,
    )


_WEIGHTED_ROWS = _rows_source(
    _WEIGHTED,
    [
        ('float(output[j * D + d]) * float(scores[j])', 'float(row_output[j * D + d]) * float(row_scores[j])'),
        (
            'float(output[(j + 8) * D + d]) * float(scores[j + 8])',
            'float(row_output[(j + 8) * D + d]) * float(row_scores[j + 8])',
        ),
        ('out[d] = T(float(total) + float(shared[d]));', 'out[row * D + d] = T(float(total) + float(shared[row * D + d]));'),
    ],
    (
        '    }\n',
        r"""
    const size_t row = size_t(thread_position_in_grid.y);
    auto row_output = output + row * K * D;
    auto row_scores = scores + row * K;
""",
    ),
)


@lru_cache(maxsize=None)
def _weighted_rows_kernel():
    return mx.fast.metal_kernel(
        name='quanteeze_metal_weighted_rows',
        input_names=['output', 'scores', 'shared'],
        output_names=['out'],
        source=_WEIGHTED_ROWS,
    )


def _weighs(output, scores, shared):
    """Return token row count (`1..ROWS_MOST`) supported by `weigh()`, or `0` when unsupported."""
    if not ON[0] or not mx.metal.is_available() or output.ndim != 4:
        return 0
    one, t, k, d = output.shape
    if (
        one != 1
        or not 0 < t <= ROWS_MOST
        or output.dtype not in (mx.bfloat16, mx.float16)
        or not scores.dtype == output.dtype == shared.dtype
        or not 0 < k <= 16
    ):
        return 0
    return t if tuple(scores.shape) == (1, t, k) and tuple(shared.shape) == (1, t, d) else 0


def weighs(output, scores, shared):
    """Whether `weigh()` supports `[1, T, k, D]` output, `[1, T, k]` scores, and `[1, T, D]` shared tensors."""
    return _weighs(output, scores, shared) > 0


def weigh(output, scores, shared):
    """Return `(output * scores[..., None]).sum(axis=-2) + shared` (`[1, T, D]`) in one Metal kernel."""
    t, k, d = output.shape[1], output.shape[2], output.shape[3]
    threads = min(256, 32 * -(-d // 32))
    kernel = _weighted_kernel() if t == 1 else _weighted_rows_kernel()
    return kernel(
        inputs=[mx.contiguous(output), mx.contiguous(scores), mx.contiguous(shared)],
        template=[('T', output.dtype), ('K', int(k))],
        grid=(32 * -(-d // 32), t, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[shared.shape],
        output_dtypes=[output.dtype],
    )[0]


WEIGHED_CHOICE, WEIGHED_CHECKED = kernel_checks.tables('metal_router.weighted')


class _Weighed:
    """Verification state for `weighted()` matching `kernel_checks.served()` module protocol."""

    CHECKS = CHECKS
    CHOICE, CHECKED = WEIGHED_CHOICE, WEIGHED_CHECKED
    STATS = dict(served=0, checked=0, mismatched=0)
    SAID = []


def weighted(output, scores, shared, theirs):
    """Compute weighted expert sum + `shared` via `weigh()` once verified bit-for-bit against `theirs()`."""
    t = _weighs(output, scores, shared)
    if not t:
        return theirs()
    key = (int(output.shape[2]), int(output.shape[3]), str(output.dtype))
    return kernel_checks.served(
        _Weighed,
        key if t == 1 else (t,) + key,
        lambda: weigh(output, scores, shared),
        theirs,
        'the Metal weighted-sum kernel',
    )


def summary():
    return dict(
        STATS,
        weighted=dict(_Weighed.STATS),
        shapes={str(k): v for k, v in CHOICE.items()},
        weighted_shapes={str(k): v for k, v in WEIGHED_CHOICE.items()},
    )
