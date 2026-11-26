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

from . import cuda_kernels, cuda_qmv, kernel_checks, per_load, per_thread

CHECKS = 3
ON = [True]                     # QUANTEEZE_CUDA_ATTN
ROWS_ON = [True]                # QUANTEEZE_CUDA_ATTN_ROWS: a verify pass's tokens too
ROWS_MOST = 7                   # max verify pass tokens handled cleanly before MLX switches to qmm
STATS = dict(served=0, checked=0, mismatched=0, rows=0)
DECLINED = {}                   # why a step was left to mlx-vlm: its count
CHOICE, CHECKED = kernel_checks.tables('cuda_attn')  # (shape, softmax block): whether it matched mlx-vlm's
SAID = []
MAX_KEYS = 2056                 # a dense prefix's keys at most (the budget's 2048 and a block's tail), shared memory


@per_load.on_let_go
def _let_go():
    """Clear per-thread attention provenance cache on model unload while keeping kernel checks."""
    _MADE.clear()
    _PROVEN[0] = (None, False)

# A warp a row, 8 rows a block: q's rows, then k's, then v's; each of T tokens in turn.
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
    for (int t = 0; t < T; ++t) {
      const float sum = qz_qmv4_row<K_IN, GS>(x + static_cast<int64_t>(t) * K_IN, w, sc, bs, row);
      if (warp.thread_rank() == 0) {
        const int64_t at = is_q ? static_cast<int64_t>(t) * NQ + row : is_k ? static_cast<int64_t>(t) * NK + row
                                                                             : ((row / HD) * T + t) * HD + row % HD;
        (is_q ? q : (is_k ? k : v))[at] = __float2bfloat16_rn(sum);
      }
    }
"""

_PREPARE = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int head = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x;
    const bool is_q = head < NH;
    const int h = is_q ? head : head - NH;
    const __nv_bfloat16* xr = is_q ? q + static_cast<int64_t>(t) * NH * 2 * HD + h * 2 * HD
                                   : k + static_cast<int64_t>(t) * NKV * HD + h * HD;
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
    __nv_bfloat16* out = (is_q ? qs : kr) + (static_cast<int64_t>(h) * T + t) * HD;
    for (int d = lane; d < HD; d += 32) {
      __nv_bfloat16 val = y[d];
      if (d < ROT) {
        const int j = d % (ROT / 2);
        float p;
        if constexpr (PMODE == 0) {
          const float p0 = static_cast<float>(pos[0]);
          p = (p0 > 0.0f ? p0 : 0.0f) + static_cast<float>(t);
        } else if constexpr (PMODE == 2) {
          p = static_cast<float>(pos_shape[0] + t);
        } else if constexpr (PMODE == 1) {
          p = static_cast<float>(pos[t * pos_strides[1]]);
        } else {
          p = static_cast<float>(pos[static_cast<int64_t>(sel[j]) * pos_strides[0] + t * pos_strides[2]]);
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

_SCORES_HEAD = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int h = blockIdx.x;
    const int lane = warp.thread_rank();
    const int wid = warp.meta_group_rank();
    const int L = keys_shape[2];
    const __nv_bfloat16* qh = q + h * HD;
"""

_ROWS_HEAD = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = warp.thread_rank();
    const int wid = warp.meta_group_rank();
    const int L = keys_shape[2] - T + t + 1;
    const __nv_bfloat16* qh = q + (static_cast<int64_t>(h) * T + t) * HD;
"""

_DOTS = r"""
    int BLOCK = 32;
    while (BLOCK < (L + 7) / 8 && BLOCK < 1024) {
      BLOCK *= 2;
    }
    const __nv_bfloat16* kb = keys + (h / REP) * keys_strides[1];
    const int64_t ks = keys_strides[2];
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
"""

_SOFTMAX = r"""
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

_SCORES = _SCORES_HEAD + _DOTS + r"""
    __nv_bfloat16* const out = probs + static_cast<int64_t>(h) * L;
""" + _SOFTMAX

_PAIR = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = warp.thread_rank();
    const int wid = warp.meta_group_rank();
    const int L = scores_shape[4];
    const int64_t at = (static_cast<int64_t>(h) * T + t) * L;
    const int visible = L - T + t + 1;
    const __nv_bfloat16 masked = __ushort_as_bfloat16(0xff7f);
    __shared__ __nv_bfloat16 sc[MAXL];
    for (int e = threadIdx.x; e < L; e += blockDim.x) {
      sc[e] = e < visible ? scores[at + e] : masked;
    }
    block.sync();
    int BLOCK = 32;
    while (BLOCK < (L + 7) / 8 && BLOCK < 1024) {
      BLOCK *= 2;
    }
    __nv_bfloat16* const out = probs + at;
""" + _SOFTMAX

_OUT = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int row = blockIdx.x * 8 + warp.meta_group_rank();
    __shared__ __align__(16) __nv_bfloat16 xs[DIN];
    for (int t = 0; t < T; ++t) {
      const __nv_bfloat16* qt = q + static_cast<int64_t>(t) * 2 * DIN;
      if (t > 0) {
        block.sync();
      }
      for (int i = block.thread_rank(); i < DIN; i += block.size()) {
        const int h = i / HD;
        const int d = i - h * HD;
        xs[i] = Multiply{}(ATTN, Sigmoid{}(qt[h * 2 * HD + HD + d]));
      }
      block.sync();
      if (row < N_OUT) {
        const float sum = qz_qmv4_row<DIN, GS>(xs, w, scales, biases, row);
        if (warp.thread_rank() == 0) {
          y[static_cast<int64_t>(t) * N_OUT + row] = __float2bfloat16_rn(sum);
        }
      }
    }
"""
_ATTN_ONE = 'attn[(static_cast<int64_t>(h) * T + t) * HD + d]'


def _pick(name, rows):
    """Select token t's buffer among `rows` kernel outputs named name0, name1, ..."""
    pick = f'{name}{rows - 1}'
    for i in range(rows - 2, -1, -1):
        pick = f'(t == {i} ? {name}{i} : {pick})'
    return pick


@lru_cache(maxsize=None)
def _kernel(part, bits=4, rows=0):
    suffix, row = ('' if bits == 4 else f'_b{bits}'), f'qz_qmv{bits}_row<'
    if part == 'qkv':
        return cuda_kernels.kernel(
            'quanteeze_attn_qkv' + suffix,
            ['x', 'wq', 'sq', 'bq', 'wk', 'sk', 'bk', 'wv', 'sv', 'bv'],
            ['q', 'k', 'v'],
            _QKV.replace('qz_qmv4_row<', row),
            cuda_qmv.HEADER,
        )
    if part == 'scores':
        return mx.fast.cuda_kernel(name='quanteeze_attn_scores', input_names=['q', 'keys'], output_names=['probs'],
                                   source=_SCORES, header=cuda_kernels.HEADER, ensure_row_contiguous=False)
    if part == 'rows':
        source = _ROWS_HEAD + _DOTS + f"""
    __nv_bfloat16* const out = {_pick('probs', rows)} + static_cast<int64_t>(h) * L;
""" + _SOFTMAX
        return mx.fast.cuda_kernel(
            name=f'quanteeze_attn_rows{rows}',
            input_names=['q', 'keys'],
            output_names=[f'probs{i}' for i in range(rows)],
            source=source,
            header=cuda_kernels.HEADER,
            ensure_row_contiguous=False,
        )
    if part == 'pair':
        return cuda_kernels.kernel('quanteeze_attn_pair', ['scores'], ['probs'], _PAIR)
    names = [f'attn{i}' for i in range(rows)] if rows else ['attn']
    attn = f'{_pick("attn", rows)}[static_cast<int64_t>(h) * HD + d]' if rows else _ATTN_ONE
    return cuda_kernels.kernel(
        f'quanteeze_attn_out{suffix}' + (f'_rows{rows}' if rows else ''),
        names + ['q', 'w', 'scales', 'biases'],
        ['y'],
        _OUT.replace('qz_qmv4_row<', row).replace('ATTN', attn),
        cuda_qmv.HEADER,
    )


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


BITS = (4, 8)


def _quantized(linear, rows, cols):
    bits = getattr(linear, 'bits', None)
    return (bits in BITS and getattr(linear, 'mode', 'affine') == 'affine'
            and getattr(linear, 'biases', None) is not None and linear.group_size in (32, 64, 128)
            and linear.weight.dtype == mx.uint32 and tuple(linear.weight.shape) == (rows, cols * bits // 32)
            and linear.scales.dtype == mx.bfloat16 and linear.biases.dtype == mx.bfloat16
            and 'bias' not in linear)


def applies(attention, x, cache, position_ids, mask):
    """Whether attend() computes this step: 1 token (no mask) or 2..ROWS_MOST tokens (causal mask)."""
    if not ON[0]:
        return False
    reason = _declined(attention, x, cache, position_ids, mask)
    if reason is not None:
        DECLINED[reason] = DECLINED.get(reason, 0) + 1
    return reason is None


def _declined(attention, x, cache, position_ids, mask):
    rows = x.shape[1] if x.ndim == 3 else 0
    if rows == 1 and mask is not None:
        return f'mask {type(mask).__name__}'
    if x.dtype != mx.bfloat16 or x.ndim != 3 or x.shape[0] != 1 or not 1 <= rows <= ROWS_MOST:
        return f'input {x.shape} {x.dtype}'
    if rows > 1 and not ROWS_ON[0]:
        return 'rows off'
    if rows > 1 and not (isinstance(mask, str) and mask == 'causal' or isinstance(mask, mx.array)):
        return f'mask {type(mask).__name__} of {rows} tokens'
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
    if (index_len + rows) // indexer.compress_ratio > indexer.block_topk or _length(cache) + rows > MAX_KEYS:
        return 'sparse'
    if rows > 1 and index_len != _length(cache):
        return 'indexer keys'
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
            and len({attention.q_proj.group_size, attention.k_proj.group_size, attention.v_proj.group_size}) == 1
            and len({attention.q_proj.bits, attention.k_proj.bits, attention.v_proj.bits}) == 1):
        return 'projections'
    if position_ids is not None:
        if position_ids.dtype not in (mx.int32, mx.int64) or not (
                position_ids.shape == (1, rows)
                or position_ids.shape == (3, 1, rows) and rope.position_selector is not None):
            return f'positions {position_ids.shape} {position_ids.dtype}'
    if rows > 1 and not _causal(mask, rows, _length(cache), _kv(cache)):
        return 'mask not causal'
    return None


_PROVEN = [(None, False)]
_MADE = {}
STATS_MASKS = dict(made=0, read=0)


def install_masks():
    """Record causal masks produced by `BatchKVCache.make_mask` without reading back to the host."""
    import threading
    import weakref

    from mlx_vlm.models import cache as caches
    cls = caches.BatchKVCache
    if getattr(cls.make_mask, '_quanteeze_made', False):
        return
    made = cls.make_mask

    def make_mask(self, N, return_array=False, **kwargs):
        mask = made(self, N, return_array=return_array, **kwargs)
        if (not any(v is not None for v in kwargs.values()) and getattr(self, '_right_padding', None) is None
                and isinstance(mask, mx.array)):
            lp = getattr(self, 'left_padding', None)
            unpadded = True if lp is None else not any(int(p) for p in lp.tolist())
            _MADE[threading.get_ident()] = (mask, int(N), int(self._idx), weakref.ref(self), lp, unpadded)
        return mask

    make_mask._quanteeze_made = True
    cls.make_mask = make_mask


def _causal(mask, rows, length, kv=None):
    """Return whether `mask` is causal over `rows` tokens after `length` cached tokens."""
    import threading
    if isinstance(mask, str):
        return mask == 'causal'
    if (mask.dtype != mx.bool_ or not 2 <= mask.ndim <= 4 or any(n != 1 for n in mask.shape[:-2])
            or tuple(mask.shape[-2:]) != (rows, length + rows)):
        return False
    made = _MADE.get(threading.get_ident())
    maker = made[3]() if made is not None else None
    if (made is not None and made[0] is mask and made[1] == rows and made[2] == length and made[5]
            and maker is not None and getattr(maker, 'left_padding', None) is made[4]):
        STATS_MASKS['made'] += 1
        _PROVEN[0] = (mask, True)
        return True
    seen, verdict = _PROVEN[0]
    if seen is mask:
        return verdict
    STATS_MASKS['read'] += 1
    want = mx.arange(length + rows)[None] <= (length + mx.arange(rows))[:, None]
    verdict = bool(mx.array_equal(mask.reshape(rows, length + rows), want).item())
    _PROVEN[0] = (mask, verdict)
    return verdict


def _kv(cache):
    """The cache's keys, values and offset: a BatchQSAKVCache's BatchKVCache, or the QSAKVCache itself."""
    return getattr(cache, 'kv_cache', cache)


def _length(cache):
    """The keys the cache holds."""
    return cache.kv_cache._idx if hasattr(cache, 'kv_cache') else cache.offset


def _qkv(attention, x):
    """The q, k and v projections of x's T tokens in one kernel."""
    q_, k_, v_ = attention.q_proj, attention.k_proj, attention.v_proj
    nq, nk, nv = q_.weight.shape[0], k_.weight.shape[0], v_.weight.shape[0]
    t = x.shape[1]
    q, k, v = _kernel('qkv', int(q_.bits))(
        inputs=[x, q_.weight, q_.scales, q_.biases, k_.weight, k_.scales, k_.biases, v_.weight, v_.scales, v_.biases],
        template=[('GS', int(q_.group_size)), ('K_IN', x.shape[-1]), ('NQ', nq), ('NK', nk), ('NV', nv),
                  ('HD', attention.head_dim), ('T', t)],
        grid=(32 * -(-(nq + nk + nv) // 8), 8, 1), threadgroup=(32, 8, 1),
        output_shapes=[(1, t, nq), (1, t, nk), (1, attention.num_key_value_heads, t, attention.head_dim)],
        output_dtypes=[x.dtype] * 3)
    return q, k, v


def _prepare(attention, q, k, cache, position_ids):
    """q normed, rotated and scaled (1, kv heads, heads a kv head, T, HD); k normed and rotated."""
    heads, kv_heads, hd = attention.num_attention_heads, attention.num_key_value_heads, attention.head_dim
    rope = attention.rotary_emb
    t = q.shape[1]
    if position_ids is None:
        pos, mode = mx.broadcast_to(attention.q_norm.weight[:1], (_length(cache),)), 2
    else:
        pos, mode = position_ids, (3 if position_ids.ndim == 3 else 1)
    sel = rope.position_selector if mode == 3 else rope.inv_freq
    qs, kr = _prepare_kernel(attention.q_norm.eps, attention.scale)(
        inputs=[q, k, attention.q_norm.weight, attention.k_norm.weight, rope.inv_freq, pos, sel],
        template=[('NH', heads), ('NKV', kv_heads), ('HD', hd), ('ROT', rope.dim), ('PMODE', mode), ('T', t),
                  ('PBITS', 64 if pos.dtype == mx.int64 else 32 if mode != 2 else 16)],
        grid=(32 * (heads + kv_heads), t, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, kv_heads, heads // kv_heads, t, hd), (1, kv_heads, t, hd)],
        output_dtypes=[q.dtype, k.dtype])
    return qs, kr


def _finish(attention, qs, keys, values, q):
    """Layer output from scaled queries of T tokens, cached keys/values, and q projection (with gates)."""
    heads, kv_heads, hd = attention.num_attention_heads, attention.num_key_value_heads, attention.head_dim
    rep, length, t = heads // kv_heads, keys.shape[2], qs.shape[3]
    threads = 512
    if t == 1:
        probs = _kernel('scores')(
            inputs=[qs, keys],
            template=[('HD', hd), ('REP', rep), ('NT', threads), ('ROWS', 4), ('MAXL', MAX_KEYS)],
            grid=(threads * heads, 1, 1), threadgroup=(threads, 1, 1),
            output_shapes=[(1, kv_heads, rep, 1, length)], output_dtypes=[qs.dtype])[0]
        attn = [mx.matmul(probs, mx.expand_dims(values, 2))]
    elif t == 2:
        scores = mx.matmul(qs, mx.swapaxes(mx.expand_dims(keys, 2), -1, -2))
        probs = _kernel('pair')(
            inputs=[scores], template=[('T', t), ('MAXL', MAX_KEYS)],
            grid=(threads * heads, t, 1), threadgroup=(threads, 1, 1),
            output_shapes=[scores.shape], output_dtypes=[qs.dtype])[0]
        attn = [mx.matmul(probs, mx.expand_dims(values, 2))]
    else:
        probs = _kernel('rows', rows=t)(
            inputs=[qs, keys],
            template=[('HD', hd), ('REP', rep), ('NT', threads), ('ROWS', 4), ('MAXL', MAX_KEYS), ('T', t)],
            grid=(threads * heads, t, 1), threadgroup=(threads, 1, 1),
            output_shapes=[(1, kv_heads, rep, 1, length - t + i + 1) for i in range(t)],
            output_dtypes=[qs.dtype] * t)
        attn = [mx.matmul(p, mx.expand_dims(values[:, :, :p.shape[-1], :], 2)) for p in probs]
    o = attention.o_proj
    n = o.weight.shape[0]
    return _kernel('out', int(o.bits), rows=t if t > 2 else 0)(
        inputs=attn + [q, o.weight, o.scales, o.biases],
        template=[('GS', int(o.group_size)), ('DIN', heads * hd), ('HD', hd), ('N_OUT', n), ('T', t)],
        grid=(32 * -(-n // 8), 8, 1), threadgroup=(32, 8, 1),
        output_shapes=[(1, t, n)], output_dtypes=[qs.dtype])[0]


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
    """Execute dense QSA attention for 1 decode token or 2..ROWS_MOST verify tokens using CUDA kernels."""
    indexer = attention.indexer
    rows = x.shape[1]
    projected = forward._linear(indexer.index_qk_proj, x)
    selection = indexer.select_from_projected(projected, cache, position_ids)
    if (selection is not None if rows == 1 else
            selection is None or selection.key_len // indexer.compress_ratio > indexer.block_topk):
        raise AssertionError('a QSA step applies() took as dense selected blocks')

    def theirs(c):
        mask = None if rows == 1 else 'causal' if rows == 2 else indexer.build_mask(selection)
        return forward._attention(attention, x, mask, c, position_ids, None)

    length = _length(cache)
    blocks = (softmax_block(length + rows) if rows <= 2
              else tuple(softmax_block(length + i + 1) for i in range(rows)))
    key = (tuple(x.shape), attention.num_attention_heads, attention.num_key_value_heads, attention.head_dim,
           blocks, int(attention.q_proj.bits), int(attention.o_proj.bits))
    choice, seen = kernel_checks.read(CHOICE, CHECKED, key)
    if choice is not None and seen >= CHECKS:
        if not choice:
            return theirs(cache)
        STATS['served'] += 1
        if rows > 1:
            STATS['rows'] += 1
        q, k, v = _qkv(attention, x)
        qs, kr = _prepare(attention, q, k, cache, position_ids)
        keys, values = cache.update_and_fetch(kr, v)
        return _finish(attention, qs, keys, values, q)
    STATS['checked'] += 1
    try:
        q, k, v = _qkv(attention, x)
        qs, kr = _prepare(attention, q, k, cache, position_ids)
    except Exception as error:
        return _refuse(key, error, lambda: theirs(cache))
    recorded = _Recorded(cache)
    want = theirs(recorded)
    try:
        got = _finish(attention, qs, *recorded.fetched, q)
        parts = dict(key=(kr, recorded.given[0]), value=(v, recorded.given[1]), output=(got, want))
        differ = [name for name, (a, b) in parts.items() if not kernel_checks.same_bits(a, b)]
    except Exception as error:
        return _refuse(key, error, lambda: want)
    if differ and not SAID:
        SAID.append(True)
        print(f'quanteeze: the full attention\'s kernels differ from mlx-vlm\'s in {", ".join(differ)} '
              f'({key}); mlx-vlm\'s serves it', file=sys.stderr, flush=True)
    STATS['mismatched'] += bool(differ)
    per_thread.checked(CHOICE, CHECKED, key, not differ, CHECKS)
    return want


def _refuse(key, error, theirs):
    if not SAID:
        SAID.append(True)
        print(f'quanteeze: the full attention\'s kernels unavailable ({type(error).__name__}: {error})',
              file=sys.stderr, flush=True)
    STATS['mismatched'] += 1
    per_thread.checked(CHOICE, CHECKED, key, False, CHECKS)
    return theirs()


def summary():
    return dict(STATS, masks=dict(STATS_MASKS), declined=dict(DECLINED), shapes={str(k): v for k, v in CHOICE.items()})
