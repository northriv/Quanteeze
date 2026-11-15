# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""GLM-5.3's routing of a decode token (mlx-vlm's MoEGate: the gate's logits, sigmoid scores, the top
k by score plus the correction bias, their scores renormalized and scaled; one group) in two Metal
kernels where mlx-vlm runs ten, each bit for bit what it replaces
(GLM-5.3's decode token is host bound, its routing first).

GLM-5.3 whole in memory at low bits (short context): 15.7 ms a token, the host
building and encoding its kernels 13.3 of it, the routing of its 42 MoE layers some 2.  A layer's
gate in mlx-vlm:

- the logits, x in float32 times the weight transposed.  The weight is bfloat16 in the checkpoint,
  so MLX casts it (288 x 4096) to float32 each call before its gemv: three kernels, the two casts
  and the product.  logits() is one, reading x and the weight as bfloat16, in MLX's gemv's order
  for these shapes: a simdgroup four rows, lane l the columns 4l..4l+3 of each 128, the products
  added in order, the lanes' sums by simd_shuffle_down 16, 8, 4, 2, 1.  A product of two bfloat16
  is exact in float32, so whether it is fused with its add does not matter.
- _expert_select, compiled: seven kernels (Sigmoid; the bias added and negated; ArgPartition;
  AsType; GatherAxis; Sum; the add, divide and multiply).  route() is one, a simdgroup a row:
  - each score as MLX's Sigmoid writes it, plus the bias;
  - the top k as argpartition gave them on every row tried: the k greatest, descending, ties to
    the lower index.  They are picked k times, by simd_max of the scores' bits made to order as
    their values (0 and -0 alike, NaN below every number), then simd_min of the indices at it;
  - their scores summed in order from 0, as MLX's small row reduction adds them, each divided by
    the sum plus 1e-20 and times the scaling factor.

Four of GLM-5.3's gates on random hidden states (ties forced): 1800 rows, the same bits.  42 gates
of its shape a step on the host: 2.31 to 0.99 ms (route() alone 1.41).

None is taken on trust: the first CHECKS times a shape is met, each kernel's output is compared bit
for bit with mlx-vlm's, and it serves the shape only if they are the same.  logits() takes one row
only (a decode of one request: more rows are MLX's matmul, another order), and a weight of fewer
than 16 columns a row (MLX's gemv splits a longer row among eight simdgroups: 64 x 1024 differed);
route() up to ROWS rows.
QUANTEEZE_SIGMOID_ROUTER_KERNELS=off keeps mlx-vlm's.
"""
import sys
from functools import lru_cache

import mlx.core as mx
from mlx_vlm.models.glm5_next.language import _expert_select
from mlx_vlm.models.linear import tiled_linear

CHECKS = 3
ROWS = 32                       # route()'s rows at most: a decode step of a batch, or a draft's
ON = [True]                     # QUANTEEZE_SIGMOID_ROUTER_KERNELS
STATS = dict(logits=0, route=0, checked=0, mismatched=0)
CHOICE = {}                     # (part, shape): whether it matched mlx-vlm's
CHECKED = {}
SAID = []

_LOGITS = r"""
    uint lane = thread_index_in_simdgroup;
    int row = (int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup)) * 4;
    if (row >= N) {
      return;
    }
    const device bfloat16_t* m = w + size_t(row) * K;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float inter[4];
    float v[4];
    int bn = int(lane) * 4;
    for (int i = 0; i < K / 128; ++i) {
      for (int tn = 0; tn < 4; ++tn) {
        v[tn] = static_cast<float>(x[bn + tn]);
      }
      int offset = 0;
      for (int tm = 0; tm < 4; ++tm) {
        for (int tn = 0; tn < 4; ++tn) {
          inter[tn] = static_cast<float>(m[offset + bn + tn]);
        }
        for (int tn = 0; tn < 4; ++tn) {
          result[tm] += inter[tn] * v[tn];
        }
        offset += K;
      }
      bn += 128;
    }
    for (int tm = 0; tm < 4; ++tm) {
      for (ushort sn = 16; sn >= 1; sn >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], sn);
      }
    }
    if (lane == 0) {
      for (int tm = 0; tm < 4; ++tm) {
        logits[row + tm] = result[tm];
      }
    }
"""

_ROUTE = r"""
    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    const device float* in = logits + row * E;
    constexpr int PER = (E + 31) / 32;
    // each choice (the score as MLX's Sigmoid writes it, plus the bias) keyed so that its bits
    // order as its value: 0 and -0 alike, NaN below every number
    uint key[PER];
    bool left[PER];
    float sig[PER];                   // each score, kept for the K picked
    for (int i = 0; i < PER; ++i) {
      int c = int(lane) + 32 * i;
      left[i] = c < E;
      key[i] = 0u;
      sig[i] = 0.0f;
      if (c < E) {
        float x = in[c];
        float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
        sig[i] = (x < 0) ? y : 1 - y;
        float v = sig[i] + bias[c];
        uint b = as_type<uint>(v == 0.0f ? 0.0f : v);
        key[i] = metal::isnan(v) ? 0u : ((b & 0x80000000u) ? ~b : (b | 0x80000000u));
      }
    }
    // the K greatest, one a round, ties to the lower index
    uint picked[K];
    float score[K];
    for (int r = 0; r < K; ++r) {
      uint best = 0u;
      uint at = 0xffffffffu;
      bool has = false;
      for (int i = 0; i < PER; ++i) {
        if (left[i] && (!has || key[i] > best)) {
          best = key[i];
          at = lane + 32 * i;
          has = true;
        }
      }
      uint top = simd_max(has ? best : 0u);
      uint idx = simd_min((has && best == top) ? at : 0xffffffffu);
      if (lane == idx % 32) {
        left[idx / 32] = false;
      }
      picked[r] = idx;
      score[r] = simd_shuffle(sig[idx / 32], ushort(idx % 32));
    }
    // their scores summed in order from 0, each divided by the sum plus 1e-20 and scaled
    if (lane == 0) {
      float s[K];
      float total = 0.0f;
      for (int r = 0; r < K; ++r) {
        s[r] = score[r];
        total = s[r] + total;
      }
      float d = total + 1e-20f;
      for (int r = 0; r < K; ++r) {
        indices[row * K + r] = int(picked[r]);
        weights[row * K + r] = NORM ? (s[r] / d) * scale[0] : s[r] * scale[0];
      }
    }
"""


@lru_cache(maxsize=None)
def _logits_kernel():
    return mx.fast.metal_kernel(name='quanteeze_gate_logits', input_names=['x', 'w'], output_names=['logits'],
                                source=_LOGITS)


@lru_cache(maxsize=None)
def _route_kernel():
    return mx.fast.metal_kernel(name='quanteeze_sigmoid_route', input_names=['logits', 'bias', 'scale'],
                                output_names=['indices', 'weights'], source=_ROUTE)


@lru_cache(maxsize=None)
def _metal():
    return mx.metal.is_available()


@lru_cache(maxsize=None)
def _scale(scale):
    return mx.array([scale], mx.float32)


def logits(x, w):
    """x [..., D] (one row, bfloat16) times w [E, D] (bfloat16) transposed, in float32: [..., E]."""
    e, d = w.shape
    return _logits_kernel()(inputs=[x, w], template=[('N', e), ('K', d)], grid=(128 * -(-e // 16), 1, 1),
                            threadgroup=(128, 1, 1), output_shapes=[(*x.shape[:-1], e)],
                            output_dtypes=[mx.float32])[0]


def route(logits, bias, k, scale, norm):
    """(indices int32, weights float32) [..., k] of float32 logits [..., E]: sigmoid scores, the top k
    by score plus bias, their scores divided by their sum (norm) and times scale."""
    e = logits.shape[-1]
    shape = (*logits.shape[:-1], k)
    return tuple(_route_kernel()(inputs=[logits, bias, _scale(float(scale))],
                                 template=[('E', e), ('K', int(k)), ('NORM', bool(norm))],
                                 grid=(32, logits.size // e, 1), threadgroup=(32, 1, 1),
                                 output_shapes=[shape, shape], output_dtypes=[mx.int32, mx.float32]))


def original_logits(gate, x):
    """mlx-vlm's MoEGate's logits."""
    return tiled_linear(lambda v: v @ gate.weight.T, x.astype(mx.float32))


def original_route(gate, logits):
    """mlx-vlm's MoEGate's route of its logits."""
    return _expert_select(logits, gate.e_score_correction_bias, gate.top_k, gate.n_group, gate.topk_group,
                          gate.routed_scaling_factor, gate.norm_topk_prob)


def gate_logits(gate, x):
    """The gate's logits of x: logits() where this shape matched mlx-vlm's, else theirs."""
    w = gate.weight
    if not (ON[0] and _metal() and x.size == x.shape[-1] and x.dtype == mx.bfloat16 and w.dtype == mx.bfloat16
            and w.ndim == 2 and w.shape[1] == x.shape[-1] and w.shape[1] % 128 == 0 and w.shape[0] % 4 == 0
            and w.shape[1] < 16 * w.shape[0]):
        return original_logits(gate, x)
    return _served('logits', tuple(w.shape), lambda: logits(x, w), lambda: original_logits(gate, x))


def gate_route(gate, logits):
    """The gate's (indices, weights) of its logits: route() where this shape matched mlx-vlm's, else
    theirs."""
    e, k, bias = logits.shape[-1], gate.top_k, gate.e_score_correction_bias
    rows = logits.size // e
    if not (ON[0] and _metal() and gate.n_group == 1 and logits.dtype == mx.float32 and bias.dtype == mx.float32
            and 0 < k <= min(32, e) and e <= 2048 and rows <= ROWS):
        return original_route(gate, logits)
    norm = k > 1 and bool(gate.norm_topk_prob)
    scale = float(gate.routed_scaling_factor)
    return _served('route', (e, k, norm, scale, rows), lambda: route(logits, bias, k, scale, norm),
                   lambda: original_route(gate, logits))


def _served(part, key, mine, theirs):
    """mine() where this part's shape matched theirs() bit for bit the first CHECKS times it was
    met, else theirs()."""
    key = (part,) + key
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS[part] += 1
            return mine()
        return theirs()
    want = theirs()
    STATS['checked'] += 1
    try:
        got = mine()
        pairs = zip(got, want) if isinstance(want, tuple) else [(got, want)]
        same = all(a.dtype == b.dtype and mx.array_equal(a, b).item() for a, b in pairs)
    except Exception as error:                              # a kernel that does not build: the original
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: the routing kernels unavailable ({type(error).__name__}: {error})',
                  file=sys.stderr, flush=True)
        same = False
    STATS['mismatched'] += not same
    CHOICE[key], CHECKED[key] = bool(same), (seen + 1 if same else CHECKS)
    return want


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
