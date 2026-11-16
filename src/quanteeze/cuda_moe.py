# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A decode token's MoE block on CUDA in four kernels, bit for bit what MLX's ops compute, and the
block's output injected into the streams in the last of them.

feed_forward() is Qwen3_5BatchInvariantForward._feed_forward and _inject for a step this takes;
the first CHECKS times a shape is met the streams it gives are compared bit for bit with the
original's, and it serves the shape only if they are the same.  QUANTEEZE_CUDA_FUSE=off keeps the
block's kernels as they were.
"""
import sys
from functools import lru_cache

import mlx.core as mx

from . import cuda_experts, cuda_kernels, cuda_qmv, cuda_router

CHECKS = 3
STATS = dict(served=0, checked=0, mismatched=0)
CHOICE = {}                     # (slots, experts, hidden, inner, shared inner): whether it matched
CHECKED = {}
SAID = []

# Blocks [0, RB) the router's rows, a warp each; [RB, RB + SB) the shared expert's, 4 gate rows in
# warps 0..3 and the same 4 up rows in warps 4..7, then their swiglu; the last the gate's one row.
IN_SOURCE = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int wid = warp.meta_group_rank();
    const int lane = warp.thread_rank();
    constexpr int RB = (E + 7) / 8;
    constexpr int SB = (IS + 3) / 4;
    const int b = blockIdx.x;
    __shared__ T gu[2][4];
    if (b < RB) {
      const int row = b * 8 + wid;
      if (row < E) {
        const float s = qz_qmv8_row<K, RGS>(x, rw, rsc, rbi, row);
        if (lane == 0) {
          logits[row] = static_cast<T>(s);
        }
      }
    } else if (b < RB + SB) {
      const int q = wid & 3;
      const bool up = wid >= 4;
      const int row = (b - RB) * 4 + q;
      if (row < IS) {
        const float s = qz_qmv4_row<K, SGS>(x, up ? uw : gw, up ? usc : gsc, up ? ubi : gbi, row);
        if (lane == 0) {
          gu[up][q] = static_cast<T>(s);
        }
      }
      block.sync();
      const int t = threadIdx.x;
      if (t < 4 && (b - RB) * 4 + t < IS) {
        const T g = gu[0][t];
        sh[(b - RB) * 4 + t] = Multiply{}(Multiply{}(g, Sigmoid{}(g)), gu[1][t]);
      }
    } else if (wid == 0) {
      const float s = qz_qmv8_row<K, GGS>(x, ew, esc, ebi, 0);
      if (lane == 0) {
        sgate[0] = Sigmoid{}(static_cast<T>(s));
      }
    }
"""

# A block 8 columns d0..d0+7: warps 0..SLOTS-1 a slot's expert's down rows there, the next 8 the
# shared expert's; then a thread a column and a stream.
OUT_SOURCE = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int wid = warp.meta_group_rank();
    const int lane = warp.thread_rank();
    const int d0 = blockIdx.x * 8;
    __shared__ T outs[SLOTS][8];
    __shared__ T shd[8];
    if (wid < SLOTS) {
      float c[1][4];
      qz_expert_rows<D, IR, GS, false>(c, h + static_cast<size_t>(wid) * IR, w, scales, biases, w, scales, biases,
                                       static_cast<int>(idx[wid]), d0);
      if (lane < 4) {
        outs[wid][2 * lane] = __float2bfloat16_rn(c[0][0]);
        outs[wid][2 * lane + 1] = __float2bfloat16_rn(c[0][1]);
      }
    } else if (wid < SLOTS + 8) {
      const int c = wid - SLOTS;
      const float s = qz_qmv4_row<IS, SGS>(sh, sw, ssc, sbi, d0 + c);
      if (lane == 0) {
        shd[c] = static_cast<T>(s);
      }
    }
    block.sync();
    const int t = threadIdx.x;
    if (t < 8 * HC) {
      const int c = t % 8;
      const int i = t / 8;
      const int d = d0 + c;
      T total = static_cast<T>(0.0f);
      for (int j = 0; j < SLOTS; ++j) {
        total = Add{}(total, QzMul{}(outs[j][c], scores[j]));
      }
      const T branch = Add{}(total, QzMul{}(sgate[0], shd[c]));
      hidden[i * D + d] = Add{}(hyper[i * D + d], QzMul{}(branch, iw[i]));
    }
"""


OUT16_SOURCE = r"""
    constexpr int ROWW = IR / 8 + 4;                // an expert row's words in shared memory (banks apart)
    constexpr int SBR = IR / GS + 4;                // its scales in shared memory, in bfloat16 (8 bytes more)
    constexpr int SBP = (IR / GS) / 4;              // ... in 8-byte pieces
    constexpr int CPS = 8;                          // chunks a stage: 64 bytes a row
    constexpr int NS = IR / 16 / CPS;
    constexpr int WARP_SMEM = 16 * ROWW * 4 + 2 * 16 * SBR * 2 + IR * 2;
    extern __shared__ __align__(16) unsigned char smem[];
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int wid = warp.meta_group_rank();
    const int lane = warp.thread_rank();
    const int d0 = blockIdx.x * 16;
    __shared__ T outs[SLOTS][16];
    __shared__ T shd[16];
    if (wid < SLOTS) {
      const int g = lane >> 2;
      const int t = lane & 3;
      unsigned char* mine = smem + wid * WARP_SMEM;
      uint32_t* ws = reinterpret_cast<uint32_t*>(mine);
      __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(mine + 16 * ROWW * 4);
      const size_t grow = static_cast<size_t>(idx[wid]) * D + d0;
      for (int p = lane; p < 2 * 16 * SBP; p += 32) {
        const int a = p / (16 * SBP);
        const int q = p % (16 * SBP);
        qz_cp8(sb + (a * 16 + q / SBP) * SBR + (q % SBP) * 4, (a ? biases : scales) + grow * (IR / GS) + q * 4);
      }
      __nv_bfloat16* xr = reinterpret_cast<__nv_bfloat16*>(mine + 16 * ROWW * 4 + 2 * 16 * SBR * 2);
      for (int p = lane; p < IR / 8; p += 32) {
        qz_cp16(xr + p * 8, h + static_cast<size_t>(wid) * IR + p * 8);
      }
#pragma unroll
      for (int s = 0; s < NS; ++s) {
        for (int p = lane; p < 64; p += 32) {
          qz_cp16(ws + (p / 4) * ROWW + s * 16 + (p % 4) * 4, w + (grow + p / 4) * (IR / 8) + s * 16 + (p % 4) * 4);
        }
        qz_cp_commit();
      }
      const uint32_t* wrow = ws + g * ROWW;
      const __nv_bfloat16* srow = sb + g * SBR;
      const __nv_bfloat16* brow = sb + (16 + g) * SBR;
      float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int s = 0; s < NS; ++s) {
        qz_cp_wait_dyn(NS - 1 - s);
        __syncwarp();
#pragma unroll
        for (int cc = 0; cc < CPS; ++cc) {
          const int kc = s * CPS + cc;
          unsigned x0 = 0u, x1 = 0u;
          if (g == 0) {
            x0 = *reinterpret_cast<const unsigned*>(xr + kc * 16 + 2 * t);
            x1 = *reinterpret_cast<const unsigned*>(xr + kc * 16 + 2 * t + 8);
          }
          qz_rows16_chunk<ROWW, SBR, GS>(c, wrow, 2 * kc, srow, brow, kc, x0, x1, t);
        }
      }
      if (t == 0) {
        outs[wid][g] = __float2bfloat16_rn(c[0]);
        outs[wid][g + 8] = __float2bfloat16_rn(c[2]);
      }
    } else if (wid < SLOTS + 8) {
#pragma unroll 1
      for (int q = 0; q < 2; ++q) {
        const int c = (wid - SLOTS) * 2 + q;
        const float s = qz_qmv4_row<IS, SGS>(sh, sw, ssc, sbi, d0 + c);
        if (lane == 0) {
          shd[c] = static_cast<T>(s);
        }
      }
    }
    block.sync();
    const int t = threadIdx.x;
    if (t < 16 * HC) {
      const int c = t % 16;
      const int i = t / 16;
      const int d = d0 + c;
      T total = static_cast<T>(0.0f);
      for (int j = 0; j < SLOTS; ++j) {
        total = Add{}(total, QzMul{}(outs[j][c], scores[j]));
      }
      const T branch = Add{}(total, QzMul{}(sgate[0], shd[c]));
      hidden[i * D + d] = Add{}(hyper[i * D + d], QzMul{}(branch, iw[i]));
    }
"""


def _out16_smem(slots, inner, group_size):
    return slots * (16 * (inner // 8 + 4) * 4 + 2 * 16 * (inner // group_size + 4) * 2 + inner * 2)


def _out16(slots, d, inner, group_size):
    """Whether moe_out() takes OUT16_SOURCE: 16 columns a block, the inner rows in stages of 128 (at most
    8), their scales in 8-byte pieces, up to 16 slots, the shared memory within an A100's."""
    return (d % 16 == 0 and inner % 128 == 0 and inner // 128 <= 8 and (inner // group_size) % 4 == 0
            and slots <= 16 and _out16_smem(slots, inner, group_size) <= 99 * 1024)


@lru_cache(maxsize=None)
def _out16_kernel(slots, inner, group_size):
    return cuda_kernels.kernel(f'quanteeze_moe_out16_{slots}_{inner}_{group_size}',
                               ['h', 'idx', 'scores', 'w', 'scales', 'biases', 'sh', 'sw', 'ssc', 'sbi', 'sgate',
                                'hyper', 'iw'],
                               ['hidden'], OUT16_SOURCE, cuda_qmv.HEADER + cuda_experts.HEADER,
                               shared_memory=_out16_smem(slots, inner, group_size))


@lru_cache(maxsize=None)
def _in_kernel():
    return cuda_kernels.kernel('quanteeze_moe_in', ['x', 'rw', 'rsc', 'rbi', 'gw', 'gsc', 'gbi', 'uw', 'usc', 'ubi',
                                                    'ew', 'esc', 'ebi'],
                               ['logits', 'sh', 'sgate'], IN_SOURCE, cuda_qmv.HEADER)


@lru_cache(maxsize=None)
def _out_kernel():
    return cuda_kernels.kernel('quanteeze_moe_out', ['h', 'idx', 'scores', 'w', 'scales', 'biases', 'sh', 'sw', 'ssc',
                                                     'sbi', 'sgate', 'hyper', 'iw'],
                               ['hidden'], OUT_SOURCE, cuda_qmv.HEADER + cuda_experts.HEADER)


def moe_in(block, x):
    """(the router's logits [1, 1, E], the shared expert's swiglu(gate(x), up(x)) [1, 1, inner],
    sigmoid(shared_expert_gate(x)) [1, 1, 1]) for one token x [1, 1, K], in one kernel."""
    router, gate = block.gate, block.shared_expert_gate
    shared = block.shared_expert
    e, inner, k = router.weight.shape[0], shared.gate_proj.weight.shape[0], x.shape[-1]
    blocks = -(-e // 8) + -(-inner // 4) + 1
    g, u = shared.gate_proj, shared.up_proj
    logits, sh, sgate = _in_kernel()(
        inputs=[mx.contiguous(x), router.weight, router.scales, router.biases, g.weight, g.scales, g.biases,
                u.weight, u.scales, u.biases, gate.weight, gate.scales, gate.biases],
        template=[('T', x.dtype), ('K', k), ('E', e), ('RGS', int(router.group_size)), ('IS', inner),
                  ('SGS', int(g.group_size)), ('GGS', int(gate.group_size))],
        grid=(256 * blocks, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, e), (1, 1, inner), (1, 1, cuda_kernels.roomy(1, x.dtype))], output_dtypes=[x.dtype] * 3)
    return logits, sh, sgate[..., :1]


def moe_out(block, h, idx, scores, sh, sgate, hyper, weights):
    """The streams hyper [1, 1, hc * D] with the block's output injected (weights [1, 1, hc])."""
    dn, sd = block.switch_mlp.down_proj, block.shared_expert.down_proj
    slots, d, inner, gs = idx.size, dn.weight.shape[1], int(h.shape[-1]), int(dn.group_size)
    inputs = [h, idx, scores, dn.weight, dn.scales, dn.biases, mx.contiguous(sh), sd.weight, sd.scales, sd.biases,
              sgate, mx.contiguous(hyper), mx.contiguous(weights)]
    template = [('T', h.dtype), ('D', d), ('IR', inner), ('GS', gs), ('SLOTS', slots), ('IS', int(sh.shape[-1])),
                ('SGS', int(sd.group_size)), ('HC', int(weights.size))]
    if _out16(slots, d, inner, gs):
        return _out16_kernel(slots, inner, gs)(
            inputs=inputs, template=template, grid=(32 * (slots + 8) * (d // 16), 1, 1),
            threadgroup=(32 * (slots + 8), 1, 1), output_shapes=[hyper.shape], output_dtypes=[hyper.dtype])[0]
    return _out_kernel()(
        inputs=inputs, template=template, grid=(32 * (slots + 8) * (d // 8), 1, 1),
        threadgroup=(32 * (slots + 8), 1, 1), output_shapes=[hyper.shape], output_dtypes=[hyper.dtype])[0]


def _eight_bit(p, k, dtype):
    return (type(p).__name__ == 'QuantizedLinear' and getattr(p, 'mode', None) == 'affine' and p.bits == 8
            and p.group_size in (32, 64, 128) and p.get('biases') is not None and 'bias' not in p
            and p.weight.dtype == mx.uint32 and p.scales.dtype == dtype and p.biases.dtype == dtype
            and p.weight.shape[-1] * 4 == k == p.scales.shape[-1] * p.group_size)


def _four_bit(p, x):
    return (type(p).__name__ == 'QuantizedLinear' and isinstance(p.get('scales'), mx.array) and 'bias' not in p
            and cuda_qmv.takes(x, p.weight, p.scales, p.get('biases'), True, p.group_size, p.bits, p.mode))


def takes(block, x, hyper, weights):
    """Whether feed_forward() computes this step as the original does: on CUDA, one token of bfloat16;
    an 8-bit router and shared expert gate, a 4-bit shared SwiGLU expert, routing and routed experts
    their kernels take; at most 24 slots."""
    if not cuda_kernels.available() or x.dtype != mx.bfloat16 or x.ndim != 3 or x.shape[:2] != (1, 1):
        return False
    shared, switch = block.get('shared_expert'), block.get('switch_mlp')
    k, slots = x.shape[-1], int(block.top_k)
    if shared is None or switch is None or not all(hasattr(shared, n) for n in ('gate_proj', 'up_proj', 'down_proj')):
        return False
    gate = block.get('shared_expert_gate')
    if gate is None or not _eight_bit(block.gate, k, x.dtype) or not _eight_bit(gate, k, x.dtype):
        return False
    if gate.weight.shape[0] != 1 or slots + 8 > 32:
        return False
    if not (_four_bit(shared.gate_proj, x) and _four_bit(shared.up_proj, x)
            and shared.gate_proj.weight.shape == shared.up_proj.weight.shape):
        return False
    inner = shared.gate_proj.weight.shape[0]
    if not _four_bit(shared.down_proj, mx.zeros((1, 1, inner), x.dtype)) or shared.down_proj.weight.shape[0] != k:
        return False
    logits = mx.zeros((1, 1, block.gate.weight.shape[0]), x.dtype)
    indices = mx.zeros((1, 1, slots), mx.uint32)
    return (cuda_router.takes(logits, slots) and cuda_experts.takes(switch, x, indices)
            and getattr(switch.down_proj, 'mode', None) == 'affine'
            and switch.down_proj.weight.shape[1] == k and k % 8 == 0
            and hyper.dtype == x.dtype and weights.dtype == x.dtype and hyper.size == weights.size * k)


def feed_forward(forward, block, x, hyper, weights):
    """forward._inject(forward._feed_forward(block, x), hyper, weights) for a step takes() takes,
    through moe_in(), the routing, the experts' gate and up, and moe_out() where its shape matched."""
    def theirs():
        return forward._inject(forward._feed_forward(block, x), hyper, weights)
    if not cuda_kernels.FUSE[0] or not takes(block, x, hyper, weights):
        return theirs()

    def mine():
        logits, sh, sgate = moe_in(block, x)
        indices, scores = cuda_router.route(logits, block.top_k)
        idx = indices.reshape(-1)
        h = cuda_experts.gate_up(block.switch_mlp, x.reshape(1, -1), idx)
        return moe_out(block, h, idx, scores.reshape(-1), sh, sgate, hyper, weights)
    key = (int(block.top_k), block.gate.weight.shape[0], x.shape[-1], block.switch_mlp.down_proj.weight.shape[-1] * 8,
           block.shared_expert.gate_proj.weight.shape[0])
    choice, seen = CHOICE.get(key), CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if choice:
            STATS['served'] += 1
            return mine()
        return theirs()
    want = theirs()
    try:
        got = mine()
        same = mx.array_equal(got, want).item()
    except Exception as error:
        if not SAID:
            SAID.append(True)
            print(f'quanteeze: the MoE block\'s kernels unavailable ({type(error).__name__}: {error})',
                  file=sys.stderr, flush=True)
        same = False
    STATS['checked'] += 1
    STATS['mismatched'] += not same
    CHOICE[key], CHECKED[key] = bool(same), (seen + 1 if same else CHECKS)
    return want


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
