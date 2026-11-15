# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Flash-Next's full attention in a decode token in fewer GPU kernels, each bit for bit what it replaces.

A full attention layer (Qwen4ExpAttention) attends densely while its prefix fits the QSA indexer's
budget (2048 tokens: mlx-vlm's _qsa_attention then runs the batch-invariant forward's _attention), and
on MLX's CUDA backend that is some forty kernels a layer.  Here:

- qkv(): the three projections' rows in one grid, each row MLX's qmv as quanteeze.cuda_qmv computes it;
- prepare(): a warp a head of q and of k: the RMSNorm as quanteeze.hyper's norm repeats it, the rotation
  of the first 64 values on bfloat16 with MLX's Multiply, Add and Negative, and q times the scale;
- the cache's update, mlx-vlm's;
- scores(): a block a query head: its scores as gemv_batched computes them, kept in shared memory, then
  MLX's softmax over them as it runs;
- the probabilities times v: MLX's matmul (cuBLAS), the same call on the same arrays;
- out(): sigmoid(gate) times the attention's output into shared memory, then o_proj's rows.

None is taken on trust: the first CHECKS times a softmax's block size is met, the new key and value and the
layer's output are compared bit for bit with mlx-vlm's (whose key and value then go into the cache), and
the kernels serve that size only if all three are the same.  QUANTEEZE_CUDA_ATTN=off: mlx-vlm's.
"""
import sys
from functools import lru_cache

import mlx.core as mx

from . import cuda_kernels, cuda_qmv

CHECKS = 3
ON = [True]                     # QUANTEEZE_CUDA_ATTN
STATS = dict(served=0, checked=0, mismatched=0)
DECLINED = {}                   # why a step was left to mlx-vlm: its count
CHOICE = {}                     # (shape, softmax block): whether it matched mlx-vlm's
CHECKED = {}
SAID = []
MAX_KEYS = 2056                 # a dense prefix's keys at most (the budget's 2048 and a block's tail), shared memory

# A warp a row, 8 rows a block: q's rows, then k's, then v's.
_QKV = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int r = blockIdx.x * 8 + warp.meta_group_rank();
    if (r >= NQ + NK + NV) {
      return;
    }
    const bool is_q = r < NQ;
    const bool is_k = !is_q && r < NQ + NK;
    const int row = is_q ? r : (is_k ? r - NQ : r - NQ - NK);
    const uint32_t* w = is_q ? wq : (is_k ? wk : wv);
    const __nv_bfloat16* sc = is_q ? sq : (is_k ? sk : sv);
    const __nv_bfloat16* bs = is_q ? bq : (is_k ? bk : bv);
    const float sum = qz_qmv4_row<K_IN, GS>(x, w, sc, bs, row);
    if (warp.thread_rank() == 0) {
      (is_q ? q : (is_k ? k : v))[row] = __float2bfloat16_rn(sum);
    }
"""

_PREPARE = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const bool is_q = head < NH;
    const int h = is_q ? head : head - NH;
    const __nv_bfloat16* xr = is_q ? q + h * 2 * HD : k + h * HD;
    const __nv_bfloat16* wr = is_q ? qw : kw;
    float acc = 0.0f;
#pragma unroll
    for (int r = 0; r < HD / 128; ++r) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float v = static_cast<float>(xr[r * 128 + lane * 4 + j]);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
      }
    }
    acc = cg::reduce(warp, acc, QzSum{});
    const float rs = rsqrtf(__fmaf_rn(acc, INV, EPS));
    __shared__ __nv_bfloat16 y[HD];
    for (int d = lane; d < HD; d += 32) {
      float a = __fmul_rn(static_cast<float>(xr[d]), rs);
      float b = __fadd_rn(1.0f, static_cast<float>(wr[d]));
      y[d] = static_cast<__nv_bfloat16>(__fmul_rn(a, b));
    }
    __syncwarp();
    __nv_bfloat16* out = is_q ? qs + h * HD : kr + h * HD;
    for (int d = lane; d < HD; d += 32) {
      __nv_bfloat16 val = y[d];
      if (d < ROT) {
        const int j = d % (ROT / 2);
        float p;
        if constexpr (PMODE == 0) {
          const float p0 = static_cast<float>(pos[0]);
          p = p0 > 0.0f ? p0 : 0.0f;
        } else if constexpr (PMODE == 2) {
          p = static_cast<float>(pos_shape[0]);
        } else if constexpr (PMODE == 1) {
          p = static_cast<float>(pos[0]);
        } else {
          p = static_cast<float>(pos[static_cast<int64_t>(sel[j]) * pos_strides[0]]);
        }
        const float f = p * inv_freq[j];
        const __nv_bfloat16 c = static_cast<__nv_bfloat16>(Cos{}(f));
        const __nv_bfloat16 s = static_cast<__nv_bfloat16>(Sin{}(f));
        const __nv_bfloat16 rot = d < ROT / 2 ? Negative{}(y[d + ROT / 2]) : y[d - ROT / 2];
        val = Add{}(Multiply{}(val, c), Multiply{}(rot, s));
      }
      if (is_q) {
        val = Multiply{}(static_cast<__nv_bfloat16>(SCALE), val);
      }
      out[d] = val;
    }
"""

_SCORES = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int h = blockIdx.x;
    const int lane = warp.thread_rank();
    const int wid = warp.meta_group_rank();
    const int L = keys_shape[2];
    int BLOCK = 32;
    while (BLOCK < (L + 7) / 8 && BLOCK < 1024) {
      BLOCK *= 2;
    }
    const __nv_bfloat16* kb = keys + (h / REP) * keys_strides[1];
    const int64_t ks = keys_strides[2];
    const __nv_bfloat16* qh = q + h * HD;
    float qv[HD / 32];
#pragma unroll
    for (int c = 0; c < HD / 128; ++c) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        qv[c * 4 + j] = static_cast<float>(qh[c * 128 + lane * 4 + j]);
      }
    }
    __shared__ __nv_bfloat16 sc[MAXL];
    constexpr int NW = NT / 32;
    for (int row0 = wid; row0 < L; row0 += NW * ROWS) {
      __nv_bfloat16 kv[ROWS][HD / 32];
#pragma unroll
      for (int u = 0; u < ROWS; ++u) {
        const int row = row0 + u * NW;
        if (row < L) {
          const __nv_bfloat16* kr = kb + row * ks;
#pragma unroll
          for (int c = 0; c < HD / 128; ++c) {
            uint2 raw = *reinterpret_cast<const uint2*>(kr + c * 128 + lane * 4);
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              kv[u][c * 4 + j] = p[j];
            }
          }
        }
      }
#pragma unroll
      for (int u = 0; u < ROWS; ++u) {
        const int row = row0 + u * NW;
        if (row < L) {
          float sum = 0.0f;
#pragma unroll
          for (int c = 0; c < HD / 128; ++c) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              sum += static_cast<float>(kv[u][c * 4 + j]) * qv[c * 4 + j];
            }
          }
          sum = cg::reduce(warp, sum, cg::plus<float>{});
          if (lane == 0) {
            sc[row] = static_cast<__nv_bfloat16>(sum);
          }
        }
      }
    }
    block.sync();
    cg::greater<float> max_op;
    cg::plus<float> plus_op;
    const float lowest = __int_as_float(0xff7fffff);
    const float neg_inf = __int_as_float(0xff800000);
    const __nv_bfloat16 neg_inf_t = static_cast<__nv_bfloat16>(neg_inf);
    const bool active = threadIdx.x < BLOCK;
    float prevmax;
    float maxval = lowest;
    float normalizer = 0.0f;
    if (active) {
      for (int r = 0; r < (L + BLOCK * 8 - 1) / (BLOCK * 8); r++) {
        const int index = r * BLOCK + threadIdx.x;
        __nv_bfloat16 vals[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          vals[i] = (8 * index + i) < L ? sc[8 * index + i] : neg_inf_t;
        }
        prevmax = maxval;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          maxval = max_op(maxval, static_cast<float>(vals[i]));
        }
        normalizer = normalizer * __expf(prevmax - maxval);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          normalizer = normalizer + __expf(static_cast<float>(vals[i]) - maxval);
        }
      }
      prevmax = maxval;
      maxval = cg::reduce(warp, maxval, max_op);
      normalizer = normalizer * __expf(prevmax - maxval);
      normalizer = cg::reduce(warp, normalizer, plus_op);
    }
    __shared__ float local_max[32];
    __shared__ float local_normalizer[32];
    prevmax = maxval;
    if (active && lane == 0) {
      local_max[wid] = maxval;
    }
    block.sync();
    if (active) {
      maxval = lane < BLOCK / 32 ? local_max[lane] : neg_inf;
      maxval = cg::reduce(warp, maxval, max_op);
      normalizer = normalizer * __expf(prevmax - maxval);
    }
    if (active && lane == 0) {
      local_normalizer[wid] = normalizer;
    }
    block.sync();
    if (active) {
      normalizer = lane < BLOCK / 32 ? local_normalizer[lane] : 0.0f;
      normalizer = cg::reduce(warp, normalizer, plus_op);
      normalizer = 1 / normalizer;
      __nv_bfloat16* out = probs + static_cast<int64_t>(h) * L;
      for (int r = 0; r < (L + BLOCK * 8 - 1) / (BLOCK * 8); r++) {
        const int index = r * BLOCK + threadIdx.x;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int e = 8 * index + i;
          if (e < L) {
            out[e] = static_cast<__nv_bfloat16>(__expf(static_cast<float>(sc[e]) - maxval) * normalizer);
          }
        }
      }
    }
"""

_OUT = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    __shared__ __align__(16) __nv_bfloat16 xs[DIN];
    for (int i = block.thread_rank(); i < DIN; i += block.size()) {
      const int h = i / HD;
      const int d = i - h * HD;
      xs[i] = Multiply{}(attn[i], Sigmoid{}(q[h * 2 * HD + HD + d]));
    }
    block.sync();
    const int row = blockIdx.x * 8 + warp.meta_group_rank();
    if (row >= N_OUT) {
      return;
    }
    const float sum = qz_qmv4_row<DIN, GS>(xs, w, scales, biases, row);
    if (warp.thread_rank() == 0) {
      y[row] = __float2bfloat16_rn(sum);
    }
"""


@lru_cache(maxsize=None)
def _kernel(part):
    if part == 'qkv':
        return cuda_kernels.kernel('quanteeze_attn_qkv', ['x', 'wq', 'sq', 'bq', 'wk', 'sk', 'bk', 'wv', 'sv', 'bv'],
                                   ['q', 'k', 'v'], _QKV, cuda_qmv.HEADER)
    if part == 'scores':
        return mx.fast.cuda_kernel(name='quanteeze_attn_scores', input_names=['q', 'keys'], output_names=['probs'],
                                   source=_SCORES, header=cuda_kernels.HEADER, ensure_row_contiguous=False)
    return cuda_kernels.kernel('quanteeze_attn_out', ['attn', 'q', 'w', 'scales', 'biases'], ['y'], _OUT,
                               cuda_qmv.HEADER)


@lru_cache(maxsize=None)
def _prepare_kernel(eps, scale):
    inv = float(mx.array(1.0 / 256, dtype=mx.float32).item())        # head_dim's, as mean's 1/n
    header = (cuda_kernels.HEADER + f'constexpr float EPS = {float(eps)!r}f;\nconstexpr float INV = {inv!r}f;\n'
              f'constexpr float SCALE = {float(scale)!r}f;\n')
    name = 'quanteeze_attn_prepare_' + ''.join(c if c.isalnum() else '_' for c in f'{eps}_{scale}')
    return mx.fast.cuda_kernel(name=name, input_names=['q', 'k', 'qw', 'kw', 'inv_freq', 'pos', 'sel'],
                               output_names=['qs', 'kr'], source=_PREPARE, header=header, ensure_row_contiguous=False)


def softmax_block(keys):
    """The block MLX's softmax runs a row of `keys` scores with (dispatch_block_dim of its 8-value reads)."""
    threads = -(-keys // 8)
    block = 32
    while block < threads and block < 1024:
        block *= 2
    return block


def _quantized(linear, rows, cols):
    return (getattr(linear, 'bits', None) == 4 and getattr(linear, 'mode', 'affine') == 'affine'
            and getattr(linear, 'biases', None) is not None and linear.group_size in (32, 64, 128)
            and linear.weight.dtype == mx.uint32 and tuple(linear.weight.shape) == (rows, cols // 8)
            and linear.scales.dtype == mx.bfloat16 and linear.biases.dtype == mx.bfloat16
            and 'bias' not in linear)


def applies(attention, x, cache, position_ids, mask):
    """Whether attend() computes this step: one token of one request, bfloat16, no mask, a QSAKVCache (a
    request served alone) or a BatchQSAKVCache without left padding, of bfloat16, the prefix still dense after
    this token, Flash-Next's shapes (why not, counted in DECLINED)."""
    if not ON[0]:
        return False
    reason = _declined(attention, x, cache, position_ids, mask)
    if reason is not None:
        DECLINED[reason] = DECLINED.get(reason, 0) + 1
    return reason is None


def _declined(attention, x, cache, position_ids, mask):
    if mask is not None:
        return f'mask {type(mask).__name__}'
    if x.dtype != mx.bfloat16 or x.ndim != 3 or x.shape[:2] != (1, 1):
        return f'input {x.shape} {x.dtype}'
    if not cuda_kernels.available():
        return 'no CUDA'
    from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, QSAKVCache, Qwen4ExpRMSNorm
    if type(cache) not in (QSAKVCache, BatchQSAKVCache):
        return f'cache {type(cache).__name__}'
    batched = type(cache) is BatchQSAKVCache
    kv = _kv(cache)
    if kv.keys is None or kv.keys.dtype != mx.bfloat16 or kv.values.dtype != mx.bfloat16 or kv.keys.shape[0] != 1:
        return 'cache state'
    if batched and (getattr(kv, '_right_padding', None) is not None or not isinstance(kv.offset, mx.array)
                    or kv.offset.shape != (1,) or kv.offset.dtype != mx.int32):
        return 'cache state'
    if not batched and not isinstance(cache.offset, int):
        return 'cache state'
    from mlx_vlm.models.qwen3_5.language import _qwen3_5_left_padding_info
    padding = _qwen3_5_left_padding_info(cache)
    if (padding is not None or batched) and (padding is None or padding[0] != (0,)):
        return 'left padding'
    indexer = attention.indexer
    index_len = (0 if cache.index_keys is None else cache.index_offset if batched
                 else cache.index_keys.shape[1])
    if (index_len + 1) // indexer.compress_ratio > indexer.block_topk or _length(cache) + 1 > MAX_KEYS:
        return 'sparse'
    heads, kv_heads, hd = attention.num_attention_heads, attention.num_key_value_heads, attention.head_dim
    rope = attention.rotary_emb
    d = x.shape[-1]
    if (hd != 256 or heads % kv_heads or kv.keys.shape[1] != kv_heads or kv.keys.shape[-1] != hd
            or getattr(attention, 'scale', None) is None):
        return 'heads'
    if (getattr(rope, 'style', None) != 'interleaved' or getattr(rope, 'fused_apply', True)
            or getattr(rope, 'attention_scaling', None) != 1.0 or not getattr(rope, 'cast_output', False)
            or rope.dim % 2 or rope.dim > hd or rope.inv_freq.dtype != mx.float32 or rope.inv_freq.size != rope.dim // 2):
        return 'rope'
    norms = (attention.q_norm, attention.k_norm)
    if (not all(type(n) is Qwen4ExpRMSNorm and n.group_size is None and n.weight.shape == (hd,)
                and n.weight.dtype == mx.bfloat16 for n in norms) or attention.q_norm.eps != attention.k_norm.eps):
        return 'norms'
    if not (_quantized(attention.q_proj, 2 * heads * hd, d) and _quantized(attention.k_proj, kv_heads * hd, d)
            and _quantized(attention.v_proj, kv_heads * hd, d) and _quantized(attention.o_proj, d, heads * hd)
            and len({attention.q_proj.group_size, attention.k_proj.group_size, attention.v_proj.group_size}) == 1):
        return 'projections'
    if position_ids is not None:
        if position_ids.dtype not in (mx.int32, mx.int64) or not (
                position_ids.shape == (1, 1) or position_ids.shape == (3, 1, 1) and rope.position_selector is not None):
            return f'positions {position_ids.shape} {position_ids.dtype}'
    return None


def _kv(cache):
    """The cache's keys, values and offset: a BatchQSAKVCache's BatchKVCache, or the QSAKVCache itself."""
    return getattr(cache, 'kv_cache', cache)


def _length(cache):
    """The keys the cache holds."""
    return cache.kv_cache._idx if hasattr(cache, 'kv_cache') else cache.offset


def _qkv(attention, x):
    """The q, k and v projections of x in one kernel: (q's 2 HD a head, k, v as (1, kv heads, 1, HD))."""
    q_, k_, v_ = attention.q_proj, attention.k_proj, attention.v_proj
    nq, nk, nv = q_.weight.shape[0], k_.weight.shape[0], v_.weight.shape[0]
    q, k, v = _kernel('qkv')(
        inputs=[x, q_.weight, q_.scales, q_.biases, k_.weight, k_.scales, k_.biases, v_.weight, v_.scales, v_.biases],
        template=[('GS', int(q_.group_size)), ('K_IN', x.shape[-1]), ('NQ', nq), ('NK', nk), ('NV', nv)],
        grid=(32 * -(-(nq + nk + nv) // 8), 8, 1), threadgroup=(32, 8, 1),
        output_shapes=[(1, 1, nq), (1, 1, nk), (1, attention.num_key_value_heads, 1, attention.head_dim)],
        output_dtypes=[x.dtype] * 3)
    return q, k, v


def _prepare(attention, q, k, cache, position_ids):
    """q normed, rotated and scaled (1, kv heads, heads a kv head, 1, HD); k normed and rotated (1, kv heads, 1, HD)."""
    heads, kv_heads, hd = attention.num_attention_heads, attention.num_key_value_heads, attention.head_dim
    rope = attention.rotary_emb
    if position_ids is None:
        pos, mode = mx.broadcast_to(attention.q_norm.weight[:1], (_length(cache),)), 2
    else:
        pos, mode = position_ids, (3 if position_ids.ndim == 3 else 1)
    sel = rope.position_selector if mode == 3 else rope.inv_freq
    qs, kr = _prepare_kernel(attention.q_norm.eps, attention.scale)(
        inputs=[q, k, attention.q_norm.weight, attention.k_norm.weight, rope.inv_freq, pos, sel],
        template=[('NH', heads), ('HD', hd), ('ROT', rope.dim), ('PMODE', mode),
                  ('PBITS', 64 if pos.dtype == mx.int64 else 32 if mode != 2 else 16)],
        grid=(32 * (heads + kv_heads), 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, kv_heads, heads // kv_heads, 1, hd), (1, kv_heads, 1, hd)],
        output_dtypes=[q.dtype, k.dtype])
    return qs, kr


def _finish(attention, qs, keys, values, q):
    """The layer's output from the scaled queries, the cache's keys and values, and q's projection (its gates)."""
    heads, kv_heads, hd = attention.num_attention_heads, attention.num_key_value_heads, attention.head_dim
    length = keys.shape[2]
    threads = 512
    probs = _kernel('scores')(
        inputs=[qs, keys],
        template=[('HD', hd), ('REP', heads // kv_heads), ('NT', threads), ('ROWS', 4), ('MAXL', MAX_KEYS)],
        grid=(threads * heads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(1, kv_heads, heads // kv_heads, 1, length)], output_dtypes=[qs.dtype])[0]
    output = mx.matmul(probs, mx.expand_dims(values, 2))
    o = attention.o_proj
    n = o.weight.shape[0]
    return _kernel('out')(
        inputs=[output, q, o.weight, o.scales, o.biases],
        template=[('GS', int(o.group_size)), ('DIN', heads * hd), ('HD', hd), ('N_OUT', n)],
        grid=(32 * -(-n // 8), 8, 1), threadgroup=(32, 8, 1),
        output_shapes=[(1, 1, n)], output_dtypes=[qs.dtype])[0]


class _Recorded:
    """The cache as _prepare_projected_qkv and the SDPA see it, the key and value it is given recorded."""

    def __init__(self, cache):
        self.cache = cache
        self.offset = cache.offset
        self.fetched = None

    def update_and_fetch(self, keys, values):
        self.given = (keys, values)
        self.fetched = self.cache.update_and_fetch(keys, values)
        return self.fetched


def attend(forward, attention, x, cache, position_ids):
    """mlx-vlm's _qsa_attention of a step applies() takes (dense): its indexer's ops, then the kernels or,
    for a softmax block not yet matched, mlx-vlm's _attention."""
    indexer = attention.indexer
    projected = forward._linear(indexer.index_qk_proj, x)
    if indexer.select_from_projected(projected, cache, position_ids) is not None:
        raise AssertionError('a QSA step applies() took as dense selected blocks')
    key = (tuple(x.shape), attention.num_attention_heads, attention.num_key_value_heads, attention.head_dim,
           softmax_block(_length(cache) + 1))
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if not choice:
            return forward._attention(attention, x, None, cache, position_ids, None)
        STATS['served'] += 1
        q, k, v = _qkv(attention, x)
        qs, kr = _prepare(attention, q, k, cache, position_ids)
        keys, values = cache.update_and_fetch(kr, v)
        return _finish(attention, qs, keys, values, q)
    STATS['checked'] += 1
    try:
        q, k, v = _qkv(attention, x)
        qs, kr = _prepare(attention, q, k, cache, position_ids)
    except Exception as error:
        return _refuse(key, error, lambda: forward._attention(attention, x, None, cache, position_ids, None))
    recorded = _Recorded(cache)
    want = forward._attention(attention, x, None, recorded, position_ids, None)
    try:
        got = _finish(attention, qs, *recorded.fetched, q)
        parts = dict(key=(kr, recorded.given[0]), value=(v, recorded.given[1]), output=(got, want))
        differ = [name for name, (a, b) in parts.items() if not mx.array_equal(a, b).item()]
    except Exception as error:
        return _refuse(key, error, lambda: want)
    if differ and not SAID:
        SAID.append(True)
        print(f'quanteeze: the full attention\'s kernels differ from mlx-vlm\'s in {", ".join(differ)} '
              f'({key}); mlx-vlm\'s serves it', file=sys.stderr, flush=True)
    STATS['mismatched'] += bool(differ)
    CHOICE[key], CHECKED[key] = not differ, (seen + 1 if not differ else CHECKS)
    return want


def _refuse(key, error, theirs):
    if not SAID:
        SAID.append(True)
        print(f'quanteeze: the full attention\'s kernels unavailable ({type(error).__name__}: {error})',
              file=sys.stderr, flush=True)
    STATS['mismatched'] += 1
    CHOICE[key], CHECKED[key] = False, CHECKS
    return theirs()


def summary():
    return dict(STATS, declined=dict(DECLINED), shapes={str(k): v for k, v in CHOICE.items()})
