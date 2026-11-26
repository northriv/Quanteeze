# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""GLM-5.3's sparse-attention indexer at a decode token with its choice of positions in one kernel,
bit for bit what mlx-vlm computes (GLM-5.3's decode token, after its
routing, linear attention and hyper-connections, its sparse attention).

The indexer (Glm5NextIndexer) picks the positions an attention layer's query reads: pools of 4 keys,
every pool while there are at most 512 (a context of 2048 tokens), else the 512 its scores rank
highest; the chosen pools' positions in order, padded to 2048 with -1, then the positions after the
last whole pool (the tail).  Its scores are float32 work (a projection, a matmul, an argpartition);
the rest is integer bookkeeping, and at one token mlx-vlm runs it as some forty kernels: validity
masks, first-key argmax, pool indices, the candidates, an argsort to put the choice in order, gathers,
pad, cumsum, the tail, concatenate.  Eleven layers at a context of some 200 tokens (one layer's real
weights, a decode step each): 3.2 ms of which the choice is 1.1 or more.

select() is one kernel, a threadgroup of 1024: the first valid position and how many are valid
(argmax and cumsum of the validity), then the positions: with every pool, first_key + i for the first
4 * pools and -1 after; with the scored 512, their bits set in a bitmap, counted and scanned so that
each pool writes its 4 positions at its rank in ascending order (what the argsort and gathers give);
then the tail, as many as visible % 4 from first_key + visible - that.  The scores are mlx-vlm's ops
as it runs them, and the caches (validity, pool windows, pooled keys) are updated as it updates them.
On one layer's real weights, 61 decode steps from contexts of 1 to 4110 tokens, across the switch to
scores at 2052: the same positions, every step.  Eleven layers: 3.2 to 1.5 ms (some 200 tokens), 3.65
to 2.3 (some 4100).

Only a decode step of one request (one token, no padding mask, no speculation), every position valid
or not, pools up to 32 * WORDS.  None is taken on trust: the first CHECKS times each path (every pool,
scored) is met, mlx-vlm's indexer runs and select() is compared with it bit for bit, and the path is
served only if they were the same.  CHECKS is 4, so that the tail is checked at each of its four
lengths.  QUANTEEZE_INDEXER_KERNEL=off keeps mlx-vlm's.
"""
import sys
from functools import lru_cache

import mlx.core as mx
from mlx_vlm.models.glm5_next.language import _score_index_keys

from . import kernel_checks, per_thread

CHECKS = 4
ON = [True]                     # QUANTEEZE_INDEXER_KERNEL
WORDS = 4096                    # the bitmap's words: pools up to 131072, a context of 524288 tokens
STATS = dict(served=0, checked=0, mismatched=0)
CHOICE, CHECKED = kernel_checks.tables('glm_indexer')  # (path, top k, pool): whether it matched mlx-vlm's
SAID = []

_SELECT = r"""
    uint t = thread_position_in_threadgroup.x;
    const int kv = int(sizes[0]);
    const int pools = int(sizes[1]);
    const int chosen = int(sizes[2]);
    const bool scored = sizes[3] != 0;
    threadgroup int counts[32];
    threadgroup int firsts[32];
    threadgroup int total[2];
    threadgroup atomic_uint bits[WORDS];
    // the first valid position and how many are valid (argmax of the validity, the cumsum's last)
    int count = 0;
    int first = kv;
    for (int i = int(t); i < kv; i += 1024) {
      if (valid[i]) {
        count += 1;
        first = metal::min(first, i);
      }
    }
    count = simd_sum(count);
    first = simd_min(first);
    if (thread_index_in_simdgroup == 0) {
      counts[simdgroup_index_in_threadgroup] = count;
      firsts[simdgroup_index_in_threadgroup] = first;
    }
    if (scored) {                     // the words the pools take, only where they are scored
      for (int w = int(t); w < (pools + 31) / 32; w += 1024) {
        atomic_store_explicit(&bits[w], 0u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      int c = 0;
      int f = kv;
      for (int s = 0; s < 32; ++s) {
        c += counts[s];
        f = metal::min(f, firsts[s]);
      }
      total[0] = c;
      total[1] = c > 0 ? f : kv;
    }
    if (scored) {
      for (int i = int(t); i < chosen; i += 1024) {
        int p = selected[i];
        atomic_fetch_or_explicit(&bits[p / 32], 1u << (p % 32), memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int visible = total[0];
    const int first_key = total[1];
    // the tail: the positions after the last whole pool, as many as visible % KPOOL
    if (t == 0) {
      int tail_count = visible % KPOOL;
      int tail_start = first_key + visible - tail_count;
      for (int j = 0; j < KPOOL - 1; ++j) {
        int p = tail_start + j;
        int safe = metal::clamp(p, 0, metal::max(kv - 1, 0));
        bool ok = j < tail_count && p < kv && valid[safe];
        topk[TOPK + j] = ok ? p : -1;
      }
    }
    if (!scored) {
      // every pool, in order: position first_key + i for the first KPOOL * pools, then -1
      for (int i = int(t); i < TOPK; i += 1024) {
        topk[i] = i < KPOOL * pools ? first_key + i : -1;
      }
      return;
    }
    // the chosen pools in ascending order: each thread a run of words, their set bits counted and
    // scanned, then each pool's KPOOL positions written at its rank
    const int words = (pools + 31) / 32;
    const int per = (words + 1023) / 1024;
    int begin = int(t) * per;
    int end = metal::min(begin + per, words);
    int mine = 0;
    for (int w = begin; w < end; ++w) {
      mine += popcount(atomic_load_explicit(&bits[w], memory_order_relaxed));
    }
    // the threads' exclusive prefix sum: within each simdgroup, then over the 32 simdgroups' sums
    // (counts, free since total was taken)
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    int before = simd_prefix_exclusive_sum(mine);
    if (lane == 31) {
      counts[sg] = before + mine;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      counts[lane] = simd_prefix_exclusive_sum(counts[lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int rank = counts[sg] + before;
    for (int w = begin; w < end; ++w) {
      uint b = atomic_load_explicit(&bits[w], memory_order_relaxed);
      while (b != 0u) {
        int bit = ctz(b);
        b &= b - 1u;
        int p = w * 32 + bit;
        for (int j = 0; j < KPOOL; ++j) {
          if (rank * KPOOL + j < TOPK) {
            topk[rank * KPOOL + j] = first_key + p * KPOOL + j;
          }
        }
        rank += 1;
      }
    }
    int filled = metal::min(chosen, TOPK / KPOOL) * KPOOL;
    for (int i = filled + int(t); i < TOPK; i += 1024) {
      topk[i] = -1;
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(name='quanteeze_glm_select', input_names=['valid', 'selected', 'sizes'],
                                output_names=['topk'], source=_SELECT)


@lru_cache(maxsize=None)
def _metal():
    return mx.metal.is_available()


@lru_cache(maxsize=None)
def _none():
    return mx.zeros((1,), mx.int32)


def _speculating(c):
    return getattr(c, '_speculation', None) is not None or getattr(c, 'is_speculating', False)


def applies(indexer, x, padding_mask, cache, pool_cache):
    """Whether decode() takes this step: one token of one request, no padding mask, caches that are
    not speculating, the tail always chosen, pools the bitmap holds."""
    return (ON[0] and _metal() and padding_mask is None and cache is not None and pool_cache is not None
            and x.shape[:2] == (1, 1) and indexer.always_select_tail and indexer.index_kpool > 1
            and isinstance(pool_cache.offset, int) and pool_cache.offset < 32 * WORDS
            and not _speculating(cache) and not _speculating(pool_cache))


def _path(indexer, pools):
    chosen = min(indexer.index_topk // indexer.index_kpool, pools)
    return chosen, 0 < chosen < pools


def select(indexer, x, q_resid, valid_state, pool_keys):
    """The positions [1, 1, top k + pool - 1] this token reads: mlx-vlm's scores where there are more
    pools than it takes, the rest select's kernel."""
    pools, kv = pool_keys.shape[1], valid_state.shape[2]
    chosen, scored = _path(indexer, pools)
    if scored:
        q, weights = indexer._project_queries(x, q_resid)
        scores = _score_index_keys(q, pool_keys, weights, indexer.softmax_scale)
        selected = mx.argpartition(-scores, kth=chosen - 1, axis=-1)[..., :chosen].astype(mx.int32).reshape(-1)
    else:
        selected = _none()
    kp = indexer.index_kpool
    return _kernel()(inputs=[valid_state.reshape(-1), selected, mx.array([kv, pools, chosen, int(scored)], mx.int32)],
                     template=[('TOPK', indexer.index_topk), ('KPOOL', kp), ('WORDS', WORDS)],
                     grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
                     output_shapes=[(1, 1, indexer.index_topk + kp - 1)], output_dtypes=[mx.int32])[0]


def decode(indexer, x, q_resid, cache, pool_cache, offset):
    """The indexer's step: its caches updated as mlx-vlm updates them, then select()."""
    k, gate = indexer._project_keys(x)
    query_valid = mx.ones((1, 1), dtype=mx.bool_)
    valid_state, _ = cache.update_and_fetch(query_valid[:, None, :, None], mx.zeros((1, 1, 1, 0), dtype=mx.bool_))
    ready_k, ready_gate, _ = pool_cache.accumulate_windows(k, gate.astype(k.dtype), offset)
    pool_keys = pool_cache.update_and_fetch(indexer._compress_pools(ready_k, ready_gate))
    return select(indexer, x, q_resid, valid_state, pool_keys)


def step(indexer, x, q_resid, cache, pool_cache, offset, stock):
    """The indexer's step: decode() where this path matched stock() (mlx-vlm's __call__) bit for bit
    the first CHECKS times it was met, else stock()."""
    pools = pool_cache.offset + (1 if pool_cache.remainder == pool_cache.ratio - 1 else 0)   # after this token
    key = ('scored' if _path(indexer, pools)[1] else 'every', indexer.index_topk, indexer.index_kpool)
    choice, seen = kernel_checks.read(CHOICE, CHECKED, key)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS['served'] += 1
            return decode(indexer, x, q_resid, cache, pool_cache, offset)
        return stock()
    want = stock()
    STATS['checked'] += 1
    try:
        pooled = pool_cache.pooled
        pool_keys = pooled if pooled is not None else mx.zeros((1, 0, indexer.head_dim), x.dtype)
        got = select(indexer, x, q_resid, cache.keys[..., :cache.offset, :], pool_keys)
        same = kernel_checks.same_bits(got, want)
    except Exception as error:                              # a kernel that does not build: mlx-vlm's
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: GLM-5.3\'s indexer kernel unavailable ({type(error).__name__}: {error})',
                  file=sys.stderr, flush=True)
        same = False
    STATS['mismatched'] += not same
    per_thread.checked(CHOICE, CHECKED, key, bool(same), CHECKS)
    return want


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
