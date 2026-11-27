# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Flash-Next's hyper-connections in fewer GPU kernels, each bit for bit what it replaces.

With quanteeze.cuda_kernels.FUSE (QUANTEEZE_CUDA_FUSE) the norm and the two projections are one
kernel on CUDA (normed_project()): every block of the projections' grid reads the 4 streams into
shared memory and sums their squares itself, as norm() does, so that the projections read their
input there and their weights all ahead; and the step of a layer (HyperForward._layer) injects the
attention's output into the streams in the MoE block's hyper-connection's kernel (_qwen4_inject's
product and sum, on bfloat16) and the MoE block's output in the block's last kernel
(quanteeze.cuda_moe), where the original injects each with two kernels of its own.

None is taken on trust: the first CHECKS times a shape is met, the kernel's output is compared
bit for bit with the original's, and it serves the shape only if they are the same.  Only a step of
one token (a decode); QUANTEEZE_HC_KERNELS=off keeps the forward's own path.
"""
# Portions of the Metal kernel ports below are adapted from Apple MLX
# (Copyright © 2023-2026 Apple Inc., MIT License — see LICENSE).
import sys
import weakref
from functools import lru_cache
from types import SimpleNamespace

import mlx.core as mx

from . import cuda_kernels, cuda_ple_layer, kernel_checks
from .arch import qwen4_exp
from .arch.qwen4_exp import _BatchInvariantForward

CHECKS = 3
ON = [True]                     # QUANTEEZE_HC_KERNELS
STATS = dict(norm=0, mix=0, project=0, fused=0, rows=0, batch=0, checked=0, mismatched=0)
ROWS_MOST = 8                   # max tokens/requests in verify-rows or batch kernels (2 to 8)
ROWS_COMPILE_AFTER = 2          # verify passes served outside a graph before compiling on Metal
CHOICE, CHECKED = kernel_checks.tables('hyper')  # (part, shape, dtype): whether it matched the original
SAID = []

_NORM = r"""
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint nsg = simdgroups_per_threadgroup;
    const device T* xr = x + row * GROUP;
    const device T* wr = w + (row % (DIM / GROUP)) * GROUP;
    device T* yr = y + row * GROUP;
    float total = 0;
    const int blocks = GROUP / (TG * 4);
    const int extra = GROUP - blocks * TG * 4;
    const device T* in = xr + lid * 4;
    for (int b = 0; b < blocks; ++b) {
      for (int i = 0; i < 4; ++i) {
        float v = static_cast<float>(in[i]);
        float sq = v * v;
        total = sq + total;
      }
      in += TG * 4;
    }
    int index = lid * 4;
    if (index + 4 <= extra) {
      for (int i = 0; i < 4; ++i) {
        float v = static_cast<float>(in[i]);
        float sq = v * v;
        total = sq + total;
      }
    } else {
      for (int i = 0; index + i < extra; ++i) {
        float v = static_cast<float>(in[i]);
        float sq = v * v;
        total = sq + total;
      }
    }
    threadgroup float shared[32];
    total = simd_sum(total);
    if (nsg > 1) {
      if (lane == 0) {
        shared[sg] = total;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float v = lane < nsg ? shared[lane] : 0.0f;
      total = simd_sum(v);
    }
    float mean = total * (1.0f / float(GROUP));
    float rs = metal::precise::rsqrt(mean + EPS);
    for (int k = lid; k < GROUP; k += TG) {
      float v = static_cast<float>(xr[k]);
      float a = v * rs;
      float b = 1.0f + static_cast<float>(wr[k]);
      yr[k] = static_cast<T>(a * b);
    }
"""

_MIX = r"""
    uint d = thread_position_in_grid.x;
    if (d >= D) {
      return;
    }
    uint row = thread_position_in_grid.y;
    const device T* p = projected + row * HC * D;
    const device T* s = normed + row * HC * D;
    T total = T(0);
    for (int i = 0; i < HC; ++i) {
      T x = p[i * D + d];
      auto e = 1 / (1 + metal::precise::exp(metal::abs(x)));
      T mix = (x < 0) ? e : 1 - e;
      T prod = mix * s[i * D + d];
      total = total + prod;
    }
    y[row * D + d] = total * T(1.0f / float(HC));
"""

_NORM_CUDA = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const T* xr = x + row * GROUP;
    const T* wr = w + (row % (DIM / GROUP)) * GROUP;
    T* yr = y + row * GROUP;
    float acc = 0.0f;
    constexpr int full = GROUP / (TG * 4);
    constexpr int fin = full * TG * 4;
    const T* in = xr + tid * 4;
    for (int r = 0; r < full; ++r) {
      for (int j = 0; j < 4; ++j) {
        float v = static_cast<float>(in[j]);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
      }
      in += TG * 4;
    }
    if (fin < GROUP) {
      for (int j = 0; j < 4; ++j) {
        float v = (fin + tid * 4 + j < GROUP) ? static_cast<float>(in[j]) : 0.0f;
        acc = __fadd_rn(acc, __fmul_rn(v, v));
      }
    }
    __shared__ float smem[32];
    acc = cg::reduce(warp, acc, QzSum{});
    if (warp.meta_group_size() > 1) {
      if (warp.thread_rank() == 0) {
        smem[warp.meta_group_rank()] = acc;
      }
      block.sync();
      acc = warp.thread_rank() < warp.meta_group_size() ? smem[warp.thread_rank()] : 0.0f;
      acc = cg::reduce(warp, acc, QzSum{});
    }
    float rs = rsqrtf(__fmaf_rn(acc, INV, EPS));
    for (int k = tid; k < GROUP; k += TG) {
      float a = __fmul_rn(static_cast<float>(xr[k]), rs);
      float b = __fadd_rn(1.0f, static_cast<float>(wr[k]));
      yr[k] = static_cast<T>(__fmul_rn(a, b));
    }
"""

_MIX_CUDA = r"""
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) {
      return;
    }
    unsigned int row = blockIdx.y;
    const T* p = projected + row * HC * D;
    const T* s = normed + row * HC * D;
    T total = static_cast<T>(0.0f);
    for (int i = 0; i < HC; ++i) {
      T sg = Sigmoid{}(p[i * D + d]);
      T prod = QzMul{}(sg, s[i * D + d]);
      total = Add{}(total, prod);
    }
    y[row * D + d] = Multiply{}(total, static_cast<T>(1.0f / float(HC)));
"""


_PROJECT = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const bool inject = n_tile >= DOWN_TILES;
    const device uint32_t* w = inject ? w2 : w1;
    const device T* sc = inject ? scales2 : scales1;
    const device T* bs = inject ? biases2 : biases1;
    device T* yb = inject ? y2 : y1;
    const int out_vec_size = inject ? N2 : N1;
    const bool partial = inject;
    const uint tile = inject ? n_tile - DOWN_TILES : n_tile;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int packs_per_thread = BITS <= 2 ? 1 : 2;
    constexpr int pack_factor = get_pack_factor<BITS, 32>();
    constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * SIMD_SIZE;
    constexpr int scale_step_per_thread = GS / values_per_thread;
    const int in_vec_size = K_SIZE;
    typedef float U;
    thread U x_thread[values_per_thread];
    thread U result[results_per_simdgroup] = {0};
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / GS;
    const int out_row = tile * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
    int last_row = results_per_simdgroup - 1;
    if (partial) {
      if (out_row >= out_vec_size) {
        return;
      }
      last_row = min(last_row, out_vec_size - 1 - out_row);
    }
    const device uint8_t* ws = (const device uint8_t*)w + out_row * in_vec_size_w
        + simd_lid * packs_per_thread * bytes_per_pack;
    sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    bs += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* xs = x + simd_lid * values_per_thread;
    for (int k = 0; k < in_vec_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, BITS>(xs, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        int src = partial ? min(row, last_row) : row;
        auto wl = (const device uint8_t*)(ws + src * in_vec_size_w);
        const device T* sl = sc + src * in_vec_size_g;
        const device T* bl = bs + src * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        result[row] += qdot<U, values_per_thread, BITS>(wl, x_thread, s, b, sum);
      }
      ws += block_size * bytes_per_pack / pack_factor;
      sc += block_size / GS;
      bs += block_size / GS;
      xs += block_size;
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      U r = simd_sum(result[row]);
      if (simd_lid == 0 && row <= last_row) {
        T a = static_cast<T>(r) / T(HC);
        auto e = 1 / (1 + metal::precise::exp(metal::abs(a)));
        T sg = (a < 0) ? e : 1 - e;
        yb[out_row + row] = inject ? T(2) * sg : a * sg;
      }
    }
"""


_PROJECT_CUDA = r"""
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int r = blockIdx.x * 8 + warp.meta_group_rank();
    if (r >= N1 + N2) {
      return;
    }
    const bool injecting = r >= N1;
    const int row = injecting ? r - N1 : r;
    const float sum = injecting ? qz_qmv4_row<K_IN, GS>(x, w2, scales2, biases2, row)
                                : qz_qmv4_row<K_IN, GS>(x, w1, scales1, biases1, row);
    if (warp.thread_rank() == 0) {
      T a = Divide{}(static_cast<T>(sum), static_cast<T>(HC));
      T sg = Sigmoid{}(a);
      if (injecting) {
        y2[row] = Multiply{}(static_cast<T>(2.0f), sg);
      } else {
        y1[row] = Multiply{}(a, sg);
      }
    }
"""


@lru_cache(maxsize=None)
def _project_kernel(bits, group_size):
    if cuda_kernels.available():
        from . import cuda_qmv
        source = _rowed(_PROJECT_CUDA, [('qz_qmv4_row<K_IN, GS>', 'qz_qmv8_row<K_IN, GS, true>')] if bits == 8 else [])
        return cuda_kernels.kernel(f'quanteeze_hc_project{"_b8" if bits == 8 else ""}_gs{group_size}',
                                   ['x', 'w1', 'scales1', 'biases1', 'w2', 'scales2', 'biases2'],
                                   ['y1', 'y2'], source, cuda_qmv.HEADER)
    from . import verify_qmv
    return mx.fast.metal_kernel(name=f'quanteeze_hc_project_b{bits}_gs{group_size}',
                                input_names=['x', 'w1', 'scales1', 'biases1', 'w2', 'scales2', 'biases2'],
                                output_names=['y1', 'y2'], header=verify_qmv._header(bits, group_size),
                                source=_PROJECT)


ONE_BITS = (4, 8)


def _attr(obj, key):
    return obj.get(key) if hasattr(obj, 'get') else getattr(obj, key, None)


def _row_projection(p, x):
    """Whether a one-token CUDA kernel computes projection `p` of a row `x` as MLX's qmv does (4 or 8 bits)."""
    from . import cuda_qmv
    return (isinstance(_attr(p, 'scales'), mx.array) and _attr(p, 'bias') is None and p.bits in ONE_BITS
            and cuda_qmv.one_row(x, p.weight, p.scales, _attr(p, 'biases'), True, p.group_size, p.bits, p.mode))


def projectable(down, inject, x):
    """Whether project() computes these two projections of x [1, 1, K] as MLX's qmv_fast / qmv does."""
    from . import verify_qmv
    k = x.shape[-1]
    if cuda_kernels.available():
        return (all(_row_projection(p, x) for p in (down, inject)) and down.group_size == inject.group_size
                and down.bits == inject.bits)
    return (all(getattr(p, 'mode', None) == 'affine' and _attr(p, 'biases') is not None and _attr(p, 'bias') is None
                    and p.scales.dtype == x.dtype for p in (down, inject))
            and down.bits == inject.bits and down.group_size == inject.group_size and down.bits in (2, 4, 8)
            and k % 512 == 0 and down.weight.shape[0] % 8 == 0 and inject.weight.shape[0] < 8
            and k == down.scales.shape[-1] * down.group_size == inject.scales.shape[-1] * inject.group_size
            and verify_qmv.helpers() is not None)


def project(down, inject, x, hc):
    """(silu(down(x) / hc), 2 sigmoid(inject(x) / hc)) for x [1, 1, K], in one kernel."""
    n1, n2 = down.weight.shape[0], inject.weight.shape[0]
    if cuda_kernels.available():
        y1, y2 = _project_kernel(int(down.bits), int(down.group_size))(
            inputs=[mx.contiguous(x), down.weight, down.scales, down.biases, inject.weight, inject.scales,
                    inject.biases],
            template=[('T', x.dtype), ('K_IN', int(x.shape[-1])), ('GS', int(down.group_size)), ('N1', n1),
                      ('N2', n2), ('HC', int(hc))],
            grid=(32 * -(-(n1 + n2) // 8), 8, 1), threadgroup=(32, 8, 1),
            output_shapes=[(1, 1, n1), (1, 1, cuda_kernels.roomy(n2, x.dtype))], output_dtypes=[x.dtype, x.dtype])
        return y1, y2[..., :n2]
    tiles = n1 // 8
    return tuple(_project_kernel(int(down.bits), int(down.group_size))(
        inputs=[mx.contiguous(x), down.weight, down.scales, down.biases, inject.weight, inject.scales, inject.biases],
        template=[('T', x.dtype), ('K_SIZE', int(x.shape[-1])), ('N1', n1), ('N2', n2), ('DOWN_TILES', tiles),
                  ('HC', int(hc))],
        grid=(32, 2 * (tiles + 1), 1), threadgroup=(32, 2, 1), output_shapes=[(1, 1, n1), (1, 1, n2)],
        output_dtypes=[x.dtype, x.dtype]))


_PROJECT_ROWS = _PROJECT.replace(
    '    uint n_tile = threadgroup_position_in_grid.y;\n',
    '    uint n_tile = threadgroup_position_in_grid.y;\n    uint tok = threadgroup_position_in_grid.z;\n').replace(
    '    device T* yb = inject ? y2 : y1;\n', '    device T* yb = inject ? y2 + tok * N2 : y1 + tok * N1;\n').replace(
    '    const device T* xs = x + simd_lid * values_per_thread;\n',
    '    const device T* xs = x + tok * K_SIZE + simd_lid * values_per_thread;\n')
assert _PROJECT_ROWS.count('tok') == 4


@lru_cache(maxsize=None)
def _project_rows_kernel(bits, group_size):
    from . import verify_qmv
    return mx.fast.metal_kernel(name=f'quanteeze_hc_project_rows_b{bits}_gs{group_size}',
                                input_names=['x', 'w1', 'scales1', 'biases1', 'w2', 'scales2', 'biases2'],
                                output_names=['y1', 'y2'], header=verify_qmv._header(bits, group_size),
                                source=_PROJECT_ROWS)


def project_rows(down, inject, x, hc):
    """project() of each token of a pass x [1, T, K] on Metal, in one kernel: (low [1, T, n1], injection [1, T, n2])."""
    n1, n2 = down.weight.shape[0], inject.weight.shape[0]
    t, tiles = x.shape[1], n1 // 8
    return tuple(_project_rows_kernel(int(down.bits), int(down.group_size))(
        inputs=[mx.contiguous(x), down.weight, down.scales, down.biases, inject.weight, inject.scales, inject.biases],
        template=[('T', x.dtype), ('K_SIZE', int(x.shape[-1])), ('N1', n1), ('N2', n2), ('DOWN_TILES', tiles),
                  ('HC', int(hc))],
        grid=(32, 2 * (tiles + 1), t), threadgroup=(32, 2, 1), output_shapes=[(1, t, n1), (1, t, n2)],
        output_dtypes=[x.dtype, x.dtype]))


_IN_CUDA = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    constexpr int KI = HC * D;
    constexpr int NT = 256;
    constexpr int WPR = TG / 32;
    constexpr int VECS = KI / 8;
    __shared__ __align__(16) T xs[KI];
    __shared__ float part[HC][WPR];
    __shared__ float rs[HC];
    const int tid = threadIdx.x;
    const int wid = warp.meta_group_rank();
    const int lane = warp.thread_rank();
    const int share = (VECS + gridDim.x - 1) / gridDim.x;
    const int first = blockIdx.x * share;
    for (int v = tid; v < VECS; v += NT) {
      uint4 hv = reinterpret_cast<const uint4*>(h)[v];
/*INJECT*/
      reinterpret_cast<uint4*>(xs)[v] = hv;
    }
    block.sync();
    for (int vw = wid; vw < HC * WPR; vw += NT / 32) {
      const int row = vw / WPR;
      const int t = (vw % WPR) * 32 + lane;
      const T* in = xs + row * D + t * 4;
      float acc = 0.0f;
      constexpr int full = D / (TG * 4);
      constexpr int fin = full * TG * 4;
      for (int r = 0; r < full; ++r) {
        for (int j = 0; j < 4; ++j) {
          float v = static_cast<float>(in[j]);
          acc = __fadd_rn(acc, __fmul_rn(v, v));
        }
        in += TG * 4;
      }
      if (fin < D) {
        for (int j = 0; j < 4; ++j) {
          float v = (fin + t * 4 + j < D) ? static_cast<float>(in[j]) : 0.0f;
          acc = __fadd_rn(acc, __fmul_rn(v, v));
        }
      }
      acc = cg::reduce(warp, acc, QzSum{});
      if (lane == 0) {
        part[row][vw % WPR] = acc;
      }
    }
    block.sync();
    if (wid < HC) {
      float acc = part[wid][0];
      if (WPR > 1) {
        acc = lane < WPR ? part[wid][lane] : 0.0f;
        acc = cg::reduce(warp, acc, QzSum{});
      }
      if (lane == 0) {
        rs[wid] = rsqrtf(__fmaf_rn(acc, INV, EPS));
      }
    }
    block.sync();
    for (int k = tid; k < KI; k += NT) {
      const int row = k / D;
      float a = __fmul_rn(static_cast<float>(xs[k]), rs[row]);
      float b = __fadd_rn(1.0f, static_cast<float>(nw[(row % (DIM / D)) * D + k % D]));
      xs[k] = static_cast<T>(__fmul_rn(a, b));
    }
    block.sync();
    for (int v = first + tid; v < first + share && v < VECS; v += NT) {
      reinterpret_cast<uint4*>(normed)[v] = reinterpret_cast<const uint4*>(xs)[v];
    }
    const int r = blockIdx.x * 8 + wid;
    if (r < N1 + N2) {
      const bool injecting = r >= N1;
      const int row = injecting ? r - N1 : r;
      const float sum = qz_qmv4_row_ahead<KI, GS>(xs, injecting ? w2 : w1, injecting ? scales2 : scales1,
                                                  injecting ? biases2 : biases1, row);
      if (lane == 0) {
        T a = Divide{}(static_cast<T>(sum), static_cast<T>(HC));
        T sg = Sigmoid{}(a);
        if (injecting) {
          y2[row] = Multiply{}(static_cast<T>(2.0f), sg);
        } else {
          y1[row] = Multiply{}(a, sg);
        }
      }
    }
"""

_INJECT_CUDA = r"""
      {
        T* hp = reinterpret_cast<T*>(&hv);
        const uint4 bv = *reinterpret_cast<const uint4*>(br + (v * 8) % D);
        const T* bp = reinterpret_cast<const T*>(&bv);
        const T wi = iw[(v * 8) / D];
        for (int e = 0; e < 8; ++e) {
          hp[e] = Add{}(hp[e], QzMul{}(bp[e], wi));
        }
        if (v >= first && v < first + share) {
          reinterpret_cast<uint4*>(hidden)[v] = hv;
        }
      }
"""


@lru_cache(maxsize=None)
def _in_kernel(eps, group, injecting, bits=4):
    from . import cuda_qmv
    inverse = float(mx.array(1.0 / group, dtype=mx.float32).item())
    weights = ['nw', 'w1', 'scales1', 'biases1', 'w2', 'scales2', 'biases2']
    name = (f'quanteeze_hc_in{"_injected" if injecting else ""}{"_b8" if bits == 8 else ""}_{eps!r}'
            .replace('.', '_').replace('-', 'm'))
    source = _IN_CUDA.replace('/*INJECT*/', _INJECT_CUDA if injecting else '')
    if bits == 8:
        source = _rowed(source, [('qz_qmv4_row_ahead<KI, GS>', 'qz_qmv8_row<KI, GS, true>')])
    return cuda_kernels.kernel(name, ['h'] + (['br', 'iw'] if injecting else []) + weights,
                               ['y1', 'y2', 'normed'] + (['hidden'] if injecting else []), source,
                               cuda_qmv.HEADER + f'constexpr float EPS = {float(eps)!r}f;\n'
                               f'constexpr float INV = {inverse!r}f;\n')


def in_fusable(module, x):
    """Whether normed_project() computes this hyper-connection's norm and projections of the streams
    x [1, 1, hc * D]: on CUDA, a norm over each stream, the projections project() takes, the streams
    fitting a block's shared memory."""
    norm_, down, inject = module.hc_norm, module.input_mix_weight_down, _attr(module, 'block_inject_weight')
    d, size = module.hidden_size, x.shape[-1]
    return (cuda_kernels.available() and inject is not None and x.dtype == mx.bfloat16 and x.size == size
            and size == module.hc_count * d and (norm_.group_size or norm_.weight.size) == d and d % 8 == 0
            and norm_.weight.size % d == 0 and norm_.weight.dtype == x.dtype and size * 2 <= 32768
            and projectable(down, inject, x))


def normed_project(module, x, branch=None, weights=None):
    """(silu(down(normed) / hc), 2 sigmoid(inject(normed) / hc), normed) for the streams x [1, 1, hc * D],
    normed = norm(x), in one kernel."""
    norm_, down, inject = module.hc_norm, module.input_mix_weight_down, _attr(module, 'block_inject_weight')
    hc, d = int(module.hc_count), int(module.hidden_size)
    n1, n2 = down.weight.shape[0], inject.weight.shape[0]
    injecting = branch is not None
    blocks = -(-(n1 + n2) // 8)
    out = _in_kernel(float(norm_.eps), d, injecting, int(down.bits))(
        inputs=[mx.contiguous(x)] + ([mx.contiguous(branch), mx.contiguous(weights)] if injecting else [])
        + [norm_.weight, down.weight, down.scales, down.biases, inject.weight, inject.scales, inject.biases],
        template=[('T', x.dtype), ('HC', hc), ('D', d), ('TG', _threads(d)), ('DIM', int(norm_.weight.size)),
                  ('GS', int(down.group_size)), ('N1', n1), ('N2', n2)],
        grid=(256 * blocks, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, n1), (1, 1, cuda_kernels.roomy(n2, x.dtype)), x.shape] + ([x.shape] if injecting else []),
        output_dtypes=[x.dtype] * (4 if injecting else 3))
    return (out[0], out[1][..., :n2], *out[2:])


_UP_MIX_CUDA = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    constexpr int PER = 8 / HC;
    __shared__ T ups[PER][HC];
    const int wid = warp.meta_group_rank();
    const int dl = wid / HC;
    const int i = wid % HC;
    const int d = blockIdx.x * PER + dl;
    if (d < D) {
      const float sum = qz_qmv4_row<K_IN, GS>(low, w, scales, biases, i * D + d);
      if (warp.thread_rank() == 0) {
        ups[dl][i] = static_cast<T>(sum);
      }
    }
    block.sync();
    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    const int dc = blockIdx.x * PER + t;
    if (t < PER && dc < D) {
      T total = static_cast<T>(0.0f);
      for (int j = 0; j < HC; ++j) {
        T prod = QzMul{}(Sigmoid{}(ups[t][j]), normed[j * D + dc]);
        total = Add{}(total, prod);
      }
      y[dc] = Multiply{}(total, static_cast<T>(1.0f / float(HC)));
    }
"""


@lru_cache(maxsize=None)
def _up_mix_kernel(group_size, bits=4):
    from . import cuda_qmv
    eight = [('qz_qmv4_row<K_IN, GS>', 'qz_qmv8_row<K_IN, GS, true>')]
    source = _rowed(_UP_MIX_CUDA, eight if bits == 8 else [])
    return cuda_kernels.kernel(f'quanteeze_hc_up_mix{"_b8" if bits == 8 else ""}_gs{group_size}',
                               ['low', 'w', 'scales', 'biases', 'normed'], ['y'], source, cuda_qmv.HEADER)


def up_mixable(up, low, hc):
    """Whether up_mix() computes mix_streams(up(low), normed) as MLX does: 4 or 8 bits on CUDA."""
    return cuda_kernels.available() and not 8 % hc and _row_projection(up, low)


def up_mix(up, low, normed, hc, d):
    """_qwen4_hyper_pointwise(hc, d)'s mix_streams(up(low), normed) for one token, in one kernel."""
    per = 8 // hc
    return _up_mix_kernel(int(up.group_size), int(up.bits))(
        inputs=[mx.contiguous(low), up.weight, up.scales, up.biases, mx.contiguous(normed)],
        template=[('T', low.dtype), ('K_IN', int(low.shape[-1])), ('GS', int(up.group_size)), ('HC', int(hc)),
                  ('D', int(d))],
        grid=(32 * -(-d // per), 8, 1), threadgroup=(32, 8, 1), output_shapes=[(*low.shape[:-1], d)],
        output_dtypes=[low.dtype])[0]


_IN_ROWS_PROLOGUE = r"""
    const size_t tok = blockIdx.y;
    const T* hr = h + tok * (HC * D);
    T* normedr = normed + tok * (HC * D);
    T* y1r = y1 + tok * N1;
    T* y2r = y2 + tok * N2R;
"""
_IN_ROWS_INJECTED = r"""
    const T* brr = br + tok * D;
    const T* iwr = iw + tok * HC;
    T* hiddenr = hidden + tok * (HC * D);
"""
_UP_MIX_ROWS_CUDA = r"""
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    constexpr int PER = 8 / HC;
    __shared__ T ups[R][PER][HC];
    const int wid = warp.meta_group_rank();
    const int dl = wid / HC;
    const int i = wid % HC;
    const int d = blockIdx.x * PER + dl;
    if (d < D) {
      float sums[R];
      qz_qmv4_rows<K_IN, GS, R>(low, w, scales, biases, i * D + d, sums);
      if (warp.thread_rank() == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
          ups[r][dl][i] = static_cast<T>(sums[r]);
        }
      }
    }
    block.sync();
    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    const int r = t / PER;
    const int c = t % PER;
    const int dc = blockIdx.x * PER + c;
    if (r < R && dc < D) {
      const T* nr = normed + static_cast<int64_t>(r) * (HC * D);
      T total = static_cast<T>(0.0f);
      for (int j = 0; j < HC; ++j) {
        T prod = QzMul{}(Sigmoid{}(ups[r][c][j]), nr[j * D + dc]);
        total = Add{}(total, prod);
      }
      y[static_cast<int64_t>(r) * D + dc] = Multiply{}(total, static_cast<T>(1.0f / float(HC)));
    }
"""


def _rowed(source, pairs):
    for a, b in pairs:
        if source.count(a) < 1:
            raise ValueError(f'hyper: {a!r} not in source')
        source = source.replace(a, b)
    return source


@lru_cache(maxsize=None)
def _in_rows_kernel(eps, group, injecting, bits=4):
    from . import cuda_qmv
    inverse = float(mx.array(1.0 / group, dtype=mx.float32).item())
    weights = ['nw', 'w1', 'scales1', 'biases1', 'w2', 'scales2', 'biases2']
    name = (f'quanteeze_hc_in_rows{"_injected" if injecting else ""}{"_b8" if bits == 8 else ""}_{eps!r}'
            .replace('.', '_').replace('-', 'm'))
    body = _IN_CUDA.replace('/*INJECT*/', _INJECT_CUDA if injecting else '')
    pairs = [('reinterpret_cast<const uint4*>(h)[v]', 'reinterpret_cast<const uint4*>(hr)[v]'),
             ('reinterpret_cast<uint4*>(normed)[v]', 'reinterpret_cast<uint4*>(normedr)[v]'),
             ('y2[row]', 'y2r[row]'), ('y1[row]', 'y1r[row]')]
    if injecting:
        pairs += [('br + (v * 8) % D', 'brr + (v * 8) % D'), ('iw[(v * 8) / D]', 'iwr[(v * 8) / D]'),
                  ('reinterpret_cast<uint4*>(hidden)[v]', 'reinterpret_cast<uint4*>(hiddenr)[v]')]
    if bits == 8:
        pairs.append(('qz_qmv4_row_ahead<KI, GS>', 'qz_qmv8_row<KI, GS>'))
    source = _IN_ROWS_PROLOGUE + (_IN_ROWS_INJECTED if injecting else '') + _rowed(body, pairs)
    return cuda_kernels.kernel(name, ['h'] + (['br', 'iw'] if injecting else []) + weights,
                               ['y1', 'y2', 'normed'] + (['hidden'] if injecting else []), source,
                               cuda_qmv.HEADER + f'constexpr float EPS = {float(eps)!r}f;\n'
                               f'constexpr float INV = {inverse!r}f;\n')


@lru_cache(maxsize=None)
def _up_mix_rows_kernel(group_size, bits=4):
    from . import cuda_qmv
    source = _rowed(_UP_MIX_ROWS_CUDA, [('qz_qmv4_rows<', 'qz_qmv8_rows<')] if bits == 8 else [])
    return cuda_kernels.kernel(f'quanteeze_hc_up_mix_rows{"_b8" if bits == 8 else ""}_gs{group_size}',
                               ['low', 'w', 'scales', 'biases', 'normed'], ['y'], source, cuda_qmv.HEADER)


def _one_row(n, dtype):
    return SimpleNamespace(dtype=dtype, ndim=3, size=n, shape=(1, 1, n))


ROWS_BITS = (4, 8)


def rows_fusable(module, x):
    """Whether _rows() computes this hyper-connection for a pass of 2 to ROWS_MOST tokens x [1, T, hc * D]."""
    if not (cuda_kernels.available() and cuda_kernels.FUSE[0] and x.ndim == 3 and x.shape[0] == 1
            and 1 < x.shape[1] <= ROWS_MOST and x.dtype == mx.bfloat16):
        return False
    return _rows_bits(module, x.shape[-1], x.dtype) is not None


def _rows_bits(module, width, dtype):
    known = getattr(module, '_quanteeze_rows_bits', None)
    if known is None:
        known = {}
        object.__setattr__(module, '_quanteeze_rows_bits', known)
    if (width, dtype) not in known:
        known[(width, dtype)] = _rows_bits_of(module, width, dtype)
    return known[(width, dtype)]


def _rows_bits_of(module, width, dtype):
    from . import cuda_qmv
    norm_, down, inject, up = (module.hc_norm, module.input_mix_weight_down, _attr(module, 'block_inject_weight'),
                               module.input_mix_weight_up)
    d, hc = int(module.hidden_size), int(module.hc_count)
    if not (inject is not None and dtype == mx.bfloat16 and width == hc * d
            and (norm_.group_size or norm_.weight.size) == d and d % 8 == 0 and norm_.weight.size % d == 0
            and norm_.weight.dtype == dtype and width * 2 <= 32768 and 8 % hc == 0):
        return None
    row, low = _one_row(width, dtype), _one_row(down.weight.shape[0], dtype)
    if not all(isinstance(_attr(p, 'scales'), mx.array) and _attr(p, 'bias') is None and p.bits in ROWS_BITS
               and cuda_qmv.computes(r, p.weight, p.scales, _attr(p, 'biases'), True, p.group_size, p.bits, p.mode)
               for r, p in ((row, down), (row, inject), (low, up))):
        return None
    if down.bits != inject.bits or down.group_size != inject.group_size:
        return None
    return int(down.bits), int(up.bits)


def rows_metal(module, x):
    """Whether HyperForward._rows_metal() computes this hyper-connection for 2..ROWS_MOST tokens on Metal."""
    if not (mx.metal.is_available() and not cuda_kernels.available() and x.ndim == 3 and x.shape[0] == 1
            and 1 < x.shape[1] <= ROWS_MOST and x.dtype in (mx.bfloat16, mx.float16)):
        return False
    known = getattr(module, '_quanteeze_rows_metal', None)
    width = (x.shape[-1], x.dtype)
    if known is not None and width in known:
        return known[width]
    norm_, down, inject = module.hc_norm, module.input_mix_weight_down, _attr(module, 'block_inject_weight')
    ok = (inject is not None and x.shape[-1] == module.hc_count * module.hidden_size
          and (norm_.group_size or norm_.weight.size) == module.hidden_size
          and norm_.weight.size % module.hidden_size == 0
          and projectable(down, inject, _one_row(x.shape[-1], x.dtype)))
    if known is None:
        known = {}
        object.__setattr__(module, '_quanteeze_rows_metal', known)
    known[width] = ok
    return ok


def normed_project_rows(module, x, branch=None, weights=None):
    """normed_project() of each token of a pass x [1, T, hc * D], in one CUDA kernel."""
    norm_, down, inject = module.hc_norm, module.input_mix_weight_down, _attr(module, 'block_inject_weight')
    hc, d = int(module.hc_count), int(module.hidden_size)
    n1, n2 = down.weight.shape[0], inject.weight.shape[0]
    t = x.shape[1]
    n2r = cuda_kernels.roomy(n2, x.dtype)
    injecting = branch is not None
    blocks = -(-(n1 + n2) // 8)
    out = _in_rows_kernel(float(norm_.eps), d, injecting, int(down.bits))(
        inputs=[mx.contiguous(x)] + ([mx.contiguous(branch), mx.contiguous(weights)] if injecting else [])
        + [norm_.weight, down.weight, down.scales, down.biases, inject.weight, inject.scales, inject.biases],
        template=[('T', x.dtype), ('HC', hc), ('D', d), ('TG', _threads(d)), ('DIM', int(norm_.weight.size)),
                  ('GS', int(down.group_size)), ('N1', n1), ('N2', n2), ('N2R', n2r)],
        grid=(256 * blocks, t, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1, t, n1), (1, t, n2r), x.shape] + ([x.shape] if injecting else []),
        output_dtypes=[x.dtype] * (4 if injecting else 3))
    return (out[0], out[1][..., :n2], *out[2:])


def up_mix_rows(up, low, normed, hc, d):
    """up_mix() of each token of a pass (low [1, T, K], normed [1, T, hc * D]) in one CUDA kernel: [1, T, D]."""
    per = 8 // hc
    t = low.shape[1]
    return _up_mix_rows_kernel(int(up.group_size), int(up.bits))(
        inputs=[mx.contiguous(low), up.weight, up.scales, up.biases, mx.contiguous(normed)],
        template=[('T', low.dtype), ('K_IN', int(low.shape[-1])), ('GS', int(up.group_size)), ('HC', int(hc)),
                  ('D', int(d)), ('R', int(t))],
        grid=(32 * -(-d // per), 8, 1), threadgroup=(32, 8, 1), output_shapes=[(1, t, d)],
        output_dtypes=[low.dtype])[0]


@lru_cache(maxsize=None)
def _norm_kernel(eps, group):
    name = f'quanteeze_hc_norm_{eps!r}'.replace('.', '_').replace('-', 'm')
    if cuda_kernels.available():
        inverse = float(mx.array(1.0 / group, dtype=mx.float32).item())
        return cuda_kernels.kernel(name, ['x', 'w'], ['y'], _NORM_CUDA,
                                   f'constexpr float EPS = {float(eps)!r}f;\nconstexpr float INV = {inverse!r}f;\n')
    return mx.fast.metal_kernel(name=name, input_names=['x', 'w'], output_names=['y'], source=_NORM,
                                header=f'constant constexpr float EPS = {float(eps)!r}f;\n')


@lru_cache(maxsize=None)
def _mix_kernel():
    if cuda_kernels.available():
        return cuda_kernels.kernel('quanteeze_hc_mix', ['projected', 'normed'], ['y'], _MIX_CUDA)
    return mx.fast.metal_kernel(name='quanteeze_hc_mix', input_names=['projected', 'normed'],
                                output_names=['y'], source=_MIX)


def _threads(group):
    """A row's threads: Metal's, four values a thread, as MLX's row reduction; CUDA's, as
    row_reduce_simple launches a float32 row (a quarter of its reads' warps, 1 to 32)."""
    reads = -(-group // 4)
    if cuda_kernels.available():
        return max(min(-(-reads // 32) // 4, 32), 1) * 32
    return min(1024, -(-reads // 32) * 32)


def norm(x, weight, group, eps):
    """_qwen4_rms_norm(group, eps)(x, weight) in one kernel."""
    tg = _threads(group)
    rows = x.size // group
    return _norm_kernel(float(eps), group)(
        inputs=[mx.contiguous(x), weight], template=[('T', x.dtype), ('GROUP', group), ('TG', tg),
                                                     ('DIM', weight.size)],
        grid=(rows * tg, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def mix(projected, normed, hc, d):
    """_qwen4_hyper_pointwise(hc, d)'s mix_streams(projected, normed) in one kernel."""
    rows = projected.size // (hc * d)
    return _mix_kernel()(
        inputs=[mx.contiguous(projected), mx.contiguous(normed)], template=[('T', projected.dtype), ('HC', hc),
                                                                            ('D', d)],
        grid=(d, rows, 1), threadgroup=(min(256, d), 1, 1), output_shapes=[(*projected.shape[:-1], d)],
        output_dtypes=[projected.dtype])[0]


def _served(part, key, mine, theirs):
    """mine() where this part's shape matched theirs() bit for bit the first CHECKS times it was
    met, else theirs()."""
    return kernel_checks.served(sys.modules[__name__], (part,) + key, mine, theirs,
                                'hyper-connection kernels', stat=part)


class HyperForward(_BatchInvariantForward):
    """The batch-invariant forward with decode and verify-pass hyper-connections in fewer kernels."""

    def _layer(self, layer, hidden, input_ids, mask, cache, position_ids):
        if (not ON[0] or not cuda_kernels.FUSE[0] or not cuda_kernels.available() or hidden.shape[0] != 1
                or not 1 <= hidden.shape[1] <= ROWS_MOST
                or not all(n in layer for n in ('attn_hyper_connection', 'mlp_hyper_connection', 'mlp'))):
            return super()._layer(layer, hidden, input_ids, mask, cache, position_ids)
        return self._fused_layer(layer, hidden, input_ids, mask, cache, position_ids)

    def _fused_layer(self, layer, hidden, input_ids, mask, cache, position_ids):
        if 'ple' in layer:
            hidden = cuda_ple_layer.added(self, layer.ple, hidden, input_ids, cache, mask)
        mixed, hyper_input, injection = self._hyper_connection(layer.attn_hyper_connection, hidden)
        if layer.is_linear:
            branch = self._gated_delta(layer.linear_attn, mixed, mask, cache)
        else:
            branch = self._qsa_attention(layer.self_attn, mixed, cache, position_ids, mask)
        mixed, hyper_input, injection = self._hyper_connection(layer.mlp_hyper_connection, hyper_input, branch,
                                                               injection)
        return self._feed_forward_into(layer.mlp, mixed, hyper_input, injection)

    def _hyper_connection(self, module, hidden_states, branch=None, weights=None):
        """(mixed, the streams, injection) of the streams hidden_states, or of _inject(branch,
        hidden_states, weights) given a branch."""
        if ON[0] and hidden_states.shape[1] != 1 and rows_fusable(module, hidden_states):
            return self._rows(module, hidden_states, branch, weights)
        if ON[0] and hidden_states.shape[1] != 1 and rows_metal(module, hidden_states):
            return self._rows_metal(module, hidden_states, branch, weights)
        n = hidden_states.shape[0]
        if (ON[0] and hidden_states.ndim == 3 and hidden_states.shape[1] == 1 and 1 < n <= ROWS_MOST
                and rows_fusable(module, rows := hidden_states.reshape(1, n, -1))):
            return self._batch(module, hidden_states, rows, branch, weights)
        if not ON[0] or hidden_states.shape[1] != 1 or hidden_states.dtype not in (mx.bfloat16, mx.float16):
            if branch is not None:
                hidden_states = self._inject(branch, hidden_states, weights)
            return super()._hyper_connection(module, hidden_states)
        key = (tuple(hidden_states.shape), str(hidden_states.dtype), _bits(module))
        if (qwen4_exp.COMPILE[0] and hidden_states.shape[0] == 1
                and _settled(key, _parts(module, hidden_states, branch))):
            return self._compiled_hyper(module, hidden_states, key, branch, weights)
        return self._kernels(module, hidden_states, key, branch, weights)

    def _rows(self, module, hidden_states, branch=None, weights=None):
        """_hyper_connection for 2..ROWS_MOST tokens on CUDA in two decode-arithmetic kernels."""
        hc, d = int(module.hc_count), int(module.hidden_size)
        t = hidden_states.shape[1]
        key = (tuple(hidden_states.shape), str(hidden_states.dtype), branch is not None,
               _rows_bits(module, hidden_states.shape[-1], hidden_states.dtype))

        def kernels(m, h, b=None, w=None):
            out = normed_project_rows(m, h, b, w)
            low, injection, normed = out[:3]
            mixed = up_mix_rows(m.input_mix_weight_up, low, normed, hc, d)
            return mixed, (out[3] if b is not None else h), injection

        def mine():
            return kernels(module, hidden_states, branch, weights)

        if qwen4_exp.COMPILE[0] and kernel_checks.settled(CHOICE, CHECKED, ('rows',) + key, CHECKS):
            out = self._compiled_rows(module, key, kernels, hidden_states, branch, weights, after=0)
            if out is not None:
                STATS['rows'] += 1
                return out

        def theirs():
            each = [self._hyper_connection(module, hidden_states[:, i:i + 1],
                                           None if branch is None else branch[:, i:i + 1],
                                           None if weights is None else weights[:, i:i + 1]) for i in range(t)]
            return tuple(mx.concatenate([e[j] for e in each], axis=1) for j in range(3))

        return _served('rows', key, mine, theirs)

    def _rows_metal(self, module, hidden_states, branch=None, weights=None):
        """_hyper_connection for 2..ROWS_MOST tokens on Metal."""
        hc, d = int(module.hc_count), int(module.hidden_size)
        t = hidden_states.shape[1]
        key = (tuple(hidden_states.shape), str(hidden_states.dtype), branch is not None)

        def kernels(m, h, b=None, w=None):
            h = h if b is None else self._inject(b, h, w)
            normed = norm(h, m.hc_norm.weight, m.hc_norm.group_size or m.hc_norm.weight.size, m.hc_norm.eps)
            low, injection = project_rows(m.input_mix_weight_down, m.block_inject_weight, normed, hc)
            return mix(self._linear(m.input_mix_weight_up, low), normed, hc, d), h, injection

        def mine():
            return kernels(module, hidden_states, branch, weights)

        if qwen4_exp.COMPILE[0] and kernel_checks.settled(CHOICE, CHECKED, ('rows',) + key, CHECKS):
            out = self._compiled_rows(module, key, kernels, hidden_states, branch, weights)
            if out is not None:
                STATS['rows'] += 1
                return out

        def theirs():
            each = [self._hyper_connection(module, hidden_states[:, i:i + 1],
                                           None if branch is None else branch[:, i:i + 1],
                                           None if weights is None else weights[:, i:i + 1]) for i in range(t)]
            return tuple(mx.concatenate([e[j] for e in each], axis=1) for j in range(3))

        return _served('rows', key, mine, theirs)

    def _compiled_rows(self, module, key, kernels, hidden_states, branch=None, weights=None, after=None):
        """Compiled verify-rows graph of `module` after `after` ungraphed passes."""
        name = f'_hyper_rows_graph{key[0][1]}' + ('_injected' if branch is not None else '')
        graph = getattr(module, name, None)
        if graph is False:
            return None
        if graph is None or isinstance(graph, int):
            seen = graph or 0
            if seen < (ROWS_COMPILE_AFTER if after is None else after):
                object.__setattr__(module, name, seen + 1)
                return None
            ref = weakref.ref(module)
            graph = (mx.compile(lambda h: kernels(ref(), h)) if branch is None
                     else mx.compile(lambda h, b, w: kernels(ref(), h, b, w)))
            object.__setattr__(module, name, graph)
        try:
            out = graph(hidden_states) if branch is None else graph(hidden_states, branch, weights)
            qwen4_exp.COMPILE_STATS['hyper'] += 1
            return out
        except Exception as error:
            object.__setattr__(module, name, False)
            qwen4_exp.refused('a verify pass\'s hyper-connection', error)
            return None

    def _batch(self, module, hidden_states, rows, branch=None, weights=None):
        """_hyper_connection of several requests decoding at once [B, 1, hc * D] in CUDA rows kernels."""
        n = hidden_states.shape[0]
        key = (tuple(hidden_states.shape), str(hidden_states.dtype), branch is not None,
               _rows_bits(module, hidden_states.shape[-1], hidden_states.dtype))

        def mine():
            out = normed_project_rows(module, rows, None if branch is None else branch.reshape(1, n, -1),
                                      None if weights is None else weights.reshape(1, n, -1))
            low, injection, normed = out[:3]
            mixed = up_mix_rows(module.input_mix_weight_up, low, normed, int(module.hc_count),
                                int(module.hidden_size))
            streams = out[3] if branch is not None else rows
            return tuple(o.reshape(n, 1, *o.shape[2:]) for o in (mixed, streams, injection))

        def theirs():
            streams = hidden_states if branch is None else self._inject(branch, hidden_states, weights)
            return super(HyperForward, self)._hyper_connection(module, streams)

        return _served('batch', key, mine, theirs)

    def _compiled_hyper(self, module, hidden_states, key, branch=None, weights=None):
        name = '_hyper_graph' if branch is None else '_hyper_graph_injected'
        graph = getattr(module, name, None)
        if graph is None:
            ref = weakref.ref(module)
            graph = (mx.compile(lambda h: self._kernels(ref(), h, key)) if branch is None
                     else mx.compile(lambda h, b, w: self._kernels(ref(), h, key, b, w)))
            object.__setattr__(module, name, graph)
        if graph:
            try:
                out = graph(hidden_states) if branch is None else graph(hidden_states, branch, weights)
                qwen4_exp.COMPILE_STATS['hyper'] += 1
                return out
            except Exception as error:
                object.__setattr__(module, name, False)
                qwen4_exp.refused('a hyper-connection', error)
        return self._kernels(module, hidden_states, key, branch, weights)

    def _kernels(self, module, hidden_states, key, branch=None, weights=None):
        from mlx_vlm.models.qwen4_exp.language import _qwen4_hyper_pointwise
        hc, d = module.hc_count, module.hidden_size
        activate, mix_streams, injection_gate = _qwen4_hyper_pointwise(hc, d)
        norm_ = module.hc_norm
        down = module.input_mix_weight_down
        inject = _attr(module, 'block_inject_weight')
        if _parts(module, hidden_states, branch)[0][0] == 'fused':
            def theirs():
                h = hidden_states if branch is None else self._inject(branch, hidden_states, weights)
                normed = norm_(h)
                return ((activate(self._linear(down, normed)), injection_gate(self._linear(inject, normed)), normed)
                        + (() if branch is None else (h,)))
            out = _served('fused', key + (branch is not None,),
                          lambda: normed_project(module, hidden_states, branch, weights), theirs)
            low, injection, normed = out[:3]
            hidden_states = out[3] if branch is not None else hidden_states
        else:
            if branch is not None:
                hidden_states = self._inject(branch, hidden_states, weights)
            normed = _served('norm', key, lambda: norm(hidden_states, norm_.weight,
                                                       norm_.group_size or norm_.weight.size, norm_.eps),
                             lambda: norm_(hidden_states))
            if inject is not None and projectable(down, inject, normed):
                low, injection = _served(
                    'project', key, lambda: project(down, inject, normed, hc),
                    lambda: (activate(self._linear(down, normed)), injection_gate(self._linear(inject, normed))))
            else:
                low = activate(self._linear(down, normed))
                injection = None if inject is None else injection_gate(self._linear(inject, normed))
        up_linear = module.input_mix_weight_up
        if up_mixable(up_linear, low, hc):
            mixed = _served('mix', key, lambda: up_mix(up_linear, low, normed, hc, d),
                            lambda: mix_streams(self._linear(up_linear, low), normed))
        else:
            up = self._linear(up_linear, low)
            mixed = _served('mix', key, lambda: mix(up, normed, hc, d), lambda: mix_streams(up, normed))
        if inject is None:
            return mixed
        return mixed, hidden_states, injection


def install():
    """The language model's batch-invariant forward (one object for every Flash-Next) given the
    kernels; whether it was."""
    from mlx_vlm.models.qwen4_exp import language
    forward = language._QWEN4_BATCH_INVARIANT_FORWARD
    if type(forward) is _BatchInvariantForward:
        forward.__class__ = HyperForward
    return type(forward) is HyperForward


def _bits(module):
    return getattr(module.input_mix_weight_down, 'bits', None), getattr(module.input_mix_weight_up, 'bits', None)


def _fusable(module, hidden_states):
    known = getattr(module, '_quanteeze_fusable', None)
    if known is None:
        known = {}
        object.__setattr__(module, '_quanteeze_fusable', known)
    width = (tuple(hidden_states.shape), hidden_states.dtype)
    if width not in known:
        known[width] = in_fusable(module, hidden_states)
    return known[width]


def _parts(module, hidden_states, branch):
    """The parts _kernels() computes this module's step with: the norm and projections in one kernel
    (fused, its key telling whether a branch is injected) or apart."""
    if cuda_kernels.FUSE[0] and _fusable(module, hidden_states):
        return ('fused', branch is not None), ('mix',)
    return ('norm',), ('mix',), ('project',)


def _settled(key, parts):
    """Whether every part's checks for this shape are made: a compiled graph may then serve it."""
    for part in parts:
        seen = kernel_checks.read(CHOICE, CHECKED, (part[0],) + key + part[1:])[1]
        if seen < CHECKS and not (part[0] == 'project' and seen == 0):
            return False
    return True


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
