# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A verify pass's quantized projections with the weights read once, each token's arithmetic the
one-token decode's.

A speculative verify pass computes 2-8 tokens at once, and its tokens must come out as decoding
them one at a time would, bit for bit.  mlx-vlm keeps that by reading every quantized weight once
a token: its exact multi-token kernel (quantized_verifier.optimized_affine_linear) takes only
projections whose input is a multiple of 512 and output a multiple of 8 -- the preconditions of
MLX's qmv_fast, whose arithmetic it copies -- and every other one goes a token a row through
gather_qmm (singleton_quantized_linear) or a call a token.  Flash-Next (hidden 1280,
hyper-connections of 320 and 4) has hardly a projection that passes; GLM-5.3 never tries it, its
models.linear.native_batch_linear going straight to the per-token paths.

Here the tokens of the pass are computed up to PER (4) to a threadgroup, the weights read once
for each threadgroup's tokens, each token's reduction exactly MLX's for one vector: four ports, of qmv_quad (an
input of 64 or 128), of qmv_fast, of qmv_fast with partial rows (an output not a multiple of 8)
and of the generic qmv (any other input; a tail read safely), their load_vector,
load_vector_safe, qdot and qdot_safe taken word for word from the installed MLX's own
quantized.h (helpers()), so they are its code, not a copy of it.

Which kernel MLX dispatches a one-token decode to is decided in its C++, which the wheel does not
carry.  _applies() names the one port whose original MLX's rules choose for a shape (bits, group
size, input, output), and that is not taken on trust either: the first CHECKS times a shape is
met, the port is run and compared bit for bit with MLX's own outputs for the pass's tokens one
at a time, on the pass's own input; if it matches it serves the shape from then on, and if not
the shape keeps the path it had.  (One port a shape: a port that is not MLX's can match by
chance -- the generic qmv did on one input of four at an input of 128.)  install() hooks GLM's
native_batch_linear, the Qwen verifiers' fallback after mlx-vlm's own kernel declines
(speculative.ops.linear), and Flash-Next's singleton projections (qwen4_exp.language).
Each projection of a checkpoint is checked and its paths are timed.

The hyper-connections' expansion (GLM-5.3, DeepSeek V4: mlx-vlm's deepseek_v4.hyper_connection
.hc_expand, after each attention and each MoE) is checked the same way (verify_hc_expand): for a
few tokens mlx-vlm expands with a kernel of its own (exact_hc_expand), meant to keep the
decode's arithmetic, and on GLM-5.3 it did not (it
found the first to part, 1.2e-4 in layer 0's attention, and greedy decoding with MTP parted
from it some 80 tokens in).  The first CHECKS times a shape is met, each way is compared bit for
bit with the decode's own expansion, a token at a time: the block through the decode's own
operation at once ('block'), then mlx-vlm's kernel ('kernel'); the first that matches serves the
shape, and with none it goes a token at a time ('tokens', exact by construction).

GLM-5.3's sparse attention (glm5_next Glm5NextAttention._attend) goes a token at a time in a
verify pass -- the token's top-k latents gathered, embed_q, the attention, unembed_out, an
mx.async_eval -- 11 layers times T loops, which made most of a pass's step to T tokens (the Mac:
23.8 ms of CPU encoding and waiting at T = 2 against 4.4 at
T = 1, the graph alone 6.6).  verify_attend computes the T tokens at once with the tokens as
the batch: each is still one query on its own keys, the decode's kernels; checked the same way,
the loop kept where it does not match.

DeepSeek V4.1's projections are MXFP8, a microscaling format (fp_quantized.h): FP_PORTS port MLX's
fp_qmv_fast_impl (with and without partial rows) and fp_qmv_impl the same way, their types and
helpers from the installed fp4.h, fp8.h and fp_quantized.h (fp_helpers()); for 2 to 8 rows MLX
computes such a projection with fp_qmv_wide, whose sums go another way.  verify_multi takes
mlx-vlm's QuantizedMultiLinear (V4.1's wo_a, each output group its own weights) a group at a
time.  quanteeze.arch.deepseek_v41 routes its projections here in its verify passes alone.
"""
# Portions of the Metal kernel ports below are adapted from Apple MLX
# (Copyright © 2023-2026 Apple Inc., MIT License — see LICENSE).
import re
import sys
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

CHECKS = 2                         # passes a shape is checked against MLX's one-token decode
MAX_TOKENS = 8                     # a verify pass's tokens at most (models.linear.DECODE_BLOCK_SIZE)
PER = 4                            # tokens a threadgroup at most, the pass's split evenly: on the Mac
                                   # 8 at once ran 4-7x slower than 4 at large outputs (the
                                   # per-token arrays out of registers), and 2 by 2 a pass of 3 computed
                                   # a fourth; QUANTEEZE_VERIFY_QMV sets it
CHOICE = {}                        # shape: the port serving it, or False for none
CHECKED = {}                       # shape: the passes it was checked on
STATS = dict(served=0, declined=0, checked=0, mismatched=0)
HC_CHOICE = {}                     # (tokens, hc, width, dtype): 'block', 'kernel' or 'tokens'
HC_CHECKED = {}
HC_WAYS = ('block', 'kernel')
ATTEND_CHOICE = {}                 # (tokens, heads, width, dtype): which STAGES at once, or False (the loop); not by
                                   # the top-k, which grows a token a step while the context is short
ATTEND_CHECKED = {}
_SAID = set()


def _header_path():
    import mlx
    return Path(next(iter(mlx.__path__))) / 'include' / 'mlx' / 'backend' / 'metal' / 'kernels' / 'quantized.h'


@lru_cache(maxsize=None)
def helpers():
    """MLX's get_pack_factor, get_bytes_per_pack, load_vector, load_vector_safe, qdot and qdot_safe,
    as the installed quantized.h has them, or None where they cannot be found."""
    try:
        text = _header_path().read_text()
    except OSError:
        return None
    start = text.find('template <int bits, int wsize = 8>\ninline constexpr short get_pack_factor')
    end = text.find('qouter(')
    if start < 0 or end < 0:
        return None
    end = text.rfind('template <', start, end)
    body = text[start:end]
    if not all(re.search(rf'\b{name}\(', body) for name in
               ('get_pack_factor', 'get_bytes_per_pack', 'load_vector', 'load_vector_safe', 'qdot', 'qdot_safe')):
        return None
    return body


def _header(bits, group_size):
    return ('using namespace metal;\n#define MLX_MTL_CONST static constant constexpr const\n'
            'MLX_MTL_CONST int SIMD_SIZE = 32;\nMLX_MTL_CONST int QUAD_SIZE = 4;\n' + helpers()
            + f'\nconstant constexpr int BITS = {int(bits)};\nconstant constexpr int GS = {int(group_size)};\n')


_COMMON = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint t_block = threadgroup_position_in_grid.z;            // VERIFY_T tokens a threadgroup
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int packs_per_thread = __PACKS__;
    constexpr int pack_factor = get_pack_factor<BITS, 32>();
    constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * SIMD_SIZE;
    constexpr int scale_step_per_thread = GS / values_per_thread;
    const int in_vec_size = K_SIZE;
    const int out_vec_size = N_SIZE;
    typedef float U;
    thread U x_thread[VERIFY_T][values_per_thread];
    thread U result[VERIFY_T][results_per_simdgroup];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < results_per_simdgroup; ++row) {
        result[t][row] = 0;
      }
    }
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / GS;
    const int out_row = n_tile * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
    const device uint8_t* ws = (const device uint8_t*)w;
    const device T* sc = scales;
    const device T* bs = biases;
    const device T* xs = x + t_block * VERIFY_T * in_vec_size + simd_lid * values_per_thread;
    device T* yb = y + t_block * VERIFY_T * out_vec_size;
"""

# MLX's qmv_fast_impl (partial_rows: __PARTIAL__), for VERIFY_T vectors at once.
_FAST = _COMMON + r"""
    int last_row = results_per_simdgroup - 1;
    if (__PARTIAL__) {
      if (out_row >= out_vec_size) {
        return;
      }
      last_row = min(last_row, out_vec_size - 1 - out_row);
    }
    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    bs += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    device T* ys = yb + out_row;
    const int aligned_end = (in_vec_size / block_size) * block_size;
    U sum[VERIFY_T];
    for (int k = 0; k < aligned_end; k += block_size) {
      for (int t = 0; t < VERIFY_T; ++t) {
        sum[t] = load_vector<T, U, values_per_thread, BITS>(xs + t * in_vec_size, x_thread[t]);
      }
      for (int row = 0; row < results_per_simdgroup; row++) {
        int src = __PARTIAL__ ? min(row, last_row) : row;
        auto wl = (const device uint8_t*)(ws + src * in_vec_size_w);
        const device T* sl = sc + src * in_vec_size_g;
        const device T* bl = bs + src * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot<U, values_per_thread, BITS>(wl, x_thread[t], s, b, sum[t]);
        }
      }
      ws += block_size * bytes_per_pack / pack_factor;
      sc += block_size / GS;
      bs += block_size / GS;
      xs += block_size;
    }
    if (aligned_end < in_vec_size) {
      bool in_bounds = (aligned_end + static_cast<int>(simd_lid) * values_per_thread) < in_vec_size;
      if (in_bounds) {
        for (int t = 0; t < VERIFY_T; ++t) {
          sum[t] = load_vector<T, U, values_per_thread, BITS>(xs + t * in_vec_size, x_thread[t]);
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
          int src = __PARTIAL__ ? min(row, last_row) : row;
          auto wl = (const device uint8_t*)(ws + src * in_vec_size_w);
          const device T* sl = sc + src * in_vec_size_g;
          const device T* bl = bs + src * in_vec_size_g;
          U s = sl[0];
          U b = bl[0];
          for (int t = 0; t < VERIFY_T; ++t) {
            result[t][row] += qdot<U, values_per_thread, BITS>(wl, x_thread[t], s, b, sum[t]);
          }
        }
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      for (int t = 0; t < VERIFY_T; ++t) {
        U r = simd_sum(result[t][row]);
        if (simd_lid == 0 && row <= last_row) {
          ys[t * out_vec_size + row] = static_cast<T>(r);
        }
      }
    }
"""

# MLX's qmv_impl, for VERIFY_T vectors at once: one pack a thread, the input's tail read safely;
# fewer than 8 outputs guarded row by row, else the last tile moved back to the end.
_GENERIC = _COMMON + r"""
    if (out_row >= out_vec_size) {
      return;
    }
    const bool small = out_vec_size < (num_simdgroups * results_per_simdgroup);
    const int base_row = small ? out_row : min(out_vec_size - results_per_simdgroup, out_row);
    const int rows = small ? min(results_per_simdgroup, out_vec_size - out_row) : results_per_simdgroup;
    ws += base_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    sc += base_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    bs += base_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    device T* ys = yb + base_row;
    U sum[VERIFY_T];
    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      for (int t = 0; t < VERIFY_T; ++t) {
        sum[t] = load_vector<T, U, values_per_thread, BITS>(xs + t * in_vec_size, x_thread[t]);
      }
      for (int row = 0; row < rows; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot<U, values_per_thread, BITS>(wl, x_thread[t], s, b, sum[t]);
        }
      }
      ws += block_size * bytes_per_pack / pack_factor;
      sc += block_size / GS;
      bs += block_size / GS;
      xs += block_size;
    }
    const int remaining = clamp(static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0,
                                values_per_thread);
    if (remaining > 0) {
      for (int t = 0; t < VERIFY_T; ++t) {
        sum[t] = load_vector_safe<T, U, values_per_thread, BITS>(xs + t * in_vec_size, x_thread[t], remaining);
      }
      for (int row = 0; row < rows; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot_safe<U, values_per_thread, BITS>(wl, x_thread[t], s, b, sum[t], remaining);
        }
      }
    }
    for (int row = 0; row < rows; row++) {
      for (int t = 0; t < VERIFY_T; ++t) {
        U r = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          ys[t * out_vec_size + row] = static_cast<T>(r);
        }
      }
    }
"""

# MLX's qmv_quad_impl (an input of 64 or 128, bits a power of two), for VERIFY_T vectors at once:
# a quad of threads a row, eight rows a quad, 64 rows a threadgroup of one simdgroup.
_QUAD = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint t_block = threadgroup_position_in_grid.z;
    uint quad_gid = thread_index_in_simdgroup / QUAD_SIZE;
    uint quad_lid = thread_index_in_simdgroup % QUAD_SIZE;
    constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;
    constexpr int pack_factor = 32 / BITS;
    constexpr int values_per_thread = K_SIZE / QUAD_SIZE;
    constexpr int packs_per_thread = values_per_thread / pack_factor;
    constexpr int scale_step_per_thread = GS / values_per_thread;
    constexpr int results_per_quadgroup = 8;
    const int in_vec_size = K_SIZE;
    const int out_vec_size = N_SIZE;
    typedef float U;
    thread U x_thread[VERIFY_T][values_per_thread];
    thread U result[VERIFY_T][results_per_quadgroup];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < results_per_quadgroup; ++row) {
        result[t][row] = 0;
      }
    }
    const int in_vec_size_w = in_vec_size / pack_factor;
    const int in_vec_size_g = in_vec_size / GS;
    const int out_row = n_tile * quads_per_simd * results_per_quadgroup + quad_gid;
    const device uint32_t* wq = w + out_row * in_vec_size_w + quad_lid * packs_per_thread;
    const device T* sc = scales + out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
    const device T* bs = biases + out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
    const device T* xs = x + t_block * VERIFY_T * in_vec_size + quad_lid * values_per_thread;
    device T* ys = y + t_block * VERIFY_T * out_vec_size + out_row;
    U sum[VERIFY_T];
    for (int t = 0; t < VERIFY_T; ++t) {
      sum[t] = load_vector<T, U, values_per_thread, BITS>(xs + t * in_vec_size, x_thread[t]);
    }
    for (int row = 0; row < results_per_quadgroup; row++) {
      if (row * quads_per_simd + out_row < out_vec_size) {
        auto wl = (const device uint8_t*)(wq + row * in_vec_size_w * quads_per_simd);
        U s = sc[row * in_vec_size_g * quads_per_simd];
        U b = bs[row * in_vec_size_g * quads_per_simd];
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot<U, values_per_thread, BITS>(wl, x_thread[t], s, b, sum[t]);
        }
      }
    }
    for (int row = 0; row < results_per_quadgroup; row++) {
      for (int t = 0; t < VERIFY_T; ++t) {
        U r = quad_sum(result[t][row]);
        if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {
          ys[t * out_vec_size + row * quads_per_simd] = static_cast<T>(r);
        }
      }
    }
"""

PORTS = ('quad', 'fast', 'fast_partial', 'generic')


# The microscaling formats' (mxfp8, mxfp4, nvfp4: DeepSeek V4.1's attention, shared experts, Engram
# and compressors' projections) qmv_fast, from the installed fp_quantized.h (fp_helpers()): MLX
# computes a short pass's tokens through another kernel (fp_qmv_wide, for 2 to vector_limit rows),
# whose sums go another way -- V4.1's verify pass parted from one-token decodes at layer 0's wq_b.
# Here each token's arithmetic is
# fp_qmv_fast_impl's for one vector, the weights read once for VERIFY_T tokens.
@lru_cache(maxsize=None)
def fp_helpers():
    """fp4.h and fp8.h (their types), and fp_quantized.h's get_pack_factor through qdot_safe, as the
    installed MLX has them, or None where they cannot be found."""
    kernels = _header_path().parent
    try:
        text = (kernels / 'fp_quantized.h').read_text()
        types = ''.join((kernels / name).read_text() for name in ('fp4.h', 'fp8.h'))
    except OSError:
        return None
    start = text.find('template <int wsize = 8, int bits = 4>\ninline constexpr short get_pack_factor')
    end = text.find('qouter(')
    if start < 0 or end < 0:
        return None
    end = text.rfind('template <', start, end)
    body = text[start:end]
    if not all(re.search(rf'\b{name}\(', body) for name in
               ('get_pack_factor', 'get_bytes_per_pack', 'dequantize_scale', 'load_vector', 'qdot')):
        return None
    return types.replace('#pragma once', '') + body


def _fp_header(bits, group_size):
    return ('using namespace metal;\n#define MLX_MTL_CONST static constant constexpr const\n'
            'MLX_MTL_CONST int SIMD_SIZE = 32;\nMLX_MTL_CONST int QUAD_SIZE = 4;\n' + fp_helpers()
            + f'\nconstant constexpr int BITS = {int(bits)};\nconstant constexpr int GS = {int(group_size)};\n')


_FP_FAST = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint t_block = threadgroup_position_in_grid.z;            // VERIFY_T tokens a threadgroup
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    constexpr int packs_per_thread = 2;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int pack_factor = get_pack_factor<32, BITS>();
    constexpr int bytes_per_pack = get_bytes_per_pack<32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * SIMD_SIZE;
    constexpr int scale_step_per_thread = GS / values_per_thread;
    const int in_vec_size = K_SIZE;
    const int out_vec_size = N_SIZE;
    typedef float U;
    thread U x_thread[VERIFY_T][values_per_thread];
    thread U result[VERIFY_T][results_per_simdgroup];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < results_per_simdgroup; ++row) {
        result[t][row] = 0;
      }
    }
    const device uint8_t* ws = (const device uint8_t*)w;
    const device uint8_t* sc = scales;
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / GS;
    const int out_row = n_tile * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
    int last_row = results_per_simdgroup - 1;
    if (__PARTIAL__) {
      if (out_row >= out_vec_size) {
        return;
      }
      last_row = min(last_row, out_vec_size - 1 - out_row);
    }
    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* xs = x + t_block * VERIFY_T * in_vec_size + simd_lid * values_per_thread;
    device T* ys = y + t_block * VERIFY_T * out_vec_size + out_row;
    for (int k = 0; k < in_vec_size; k += block_size) {
      for (int t = 0; t < VERIFY_T; ++t) {
        load_vector<T, U, values_per_thread>(xs + t * in_vec_size, x_thread[t]);
      }
      for (int row = 0; row < results_per_simdgroup; row++) {
        int src = __PARTIAL__ ? min(row, last_row) : row;
        auto wl = (const device uint8_t*)(ws + src * in_vec_size_w);
        const device uint8_t* sl = sc + src * in_vec_size_g;
        U s = dequantize_scale<U, GS>(sl[0]);
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot<U, values_per_thread, BITS>(wl, x_thread[t], s);
        }
      }
      ws += block_size * bytes_per_pack / pack_factor;
      sc += block_size / GS;
      xs += block_size;
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      for (int t = 0; t < VERIFY_T; ++t) {
        U r = simd_sum(result[t][row]);
        if (simd_lid == 0 && row <= last_row) {
          ys[t * out_vec_size + row] = static_cast<T>(r);
        }
      }
    }
"""
# fp_qmv_impl, for VERIFY_T vectors at once: one pack a thread, the input's tail read safely; fewer
# than 8 outputs guarded row by row (its tail by qdot), else the last tile moved back (by qdot_safe).
_FP_GENERIC = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint t_block = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int packs_per_thread = 1;
    constexpr int pack_factor = get_pack_factor<32, BITS>();
    constexpr int bytes_per_pack = get_bytes_per_pack<32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * SIMD_SIZE;
    constexpr int scale_step_per_thread = GS / values_per_thread;
    const int in_vec_size = K_SIZE;
    const int out_vec_size = N_SIZE;
    typedef float U;
    thread U x_thread[VERIFY_T][values_per_thread];
    thread U result[VERIFY_T][results_per_simdgroup];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < results_per_simdgroup; ++row) {
        result[t][row] = 0;
      }
    }
    const device uint8_t* ws = (const device uint8_t*)w;
    const device uint8_t* sc = scales;
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / GS;
    const int out_row = n_tile * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
    if (out_row >= out_vec_size) {
      return;
    }
    const bool small = out_vec_size < (num_simdgroups * results_per_simdgroup);
    const int base_row = small ? out_row : min(out_vec_size - results_per_simdgroup, out_row);
    const int rows = small ? min(results_per_simdgroup, out_vec_size - out_row) : results_per_simdgroup;
    ws += base_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    sc += base_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* xs = x + t_block * VERIFY_T * in_vec_size + simd_lid * values_per_thread;
    device T* ys = y + t_block * VERIFY_T * out_vec_size + base_row;
    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      for (int t = 0; t < VERIFY_T; ++t) {
        load_vector<T, U, values_per_thread>(xs + t * in_vec_size, x_thread[t]);
      }
      for (int row = 0; row < rows; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device uint8_t* sl = sc + row * in_vec_size_g;
        U s = dequantize_scale<U, GS>(sl[0]);
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot<U, values_per_thread, BITS>(wl, x_thread[t], s);
        }
      }
      ws += block_size * bytes_per_pack / pack_factor;
      sc += block_size / GS;
      xs += block_size;
    }
    const int remaining = clamp(static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0,
                                values_per_thread);
    if (remaining > 0) {
      for (int t = 0; t < VERIFY_T; ++t) {
        load_vector_safe<T, U, values_per_thread>(xs + t * in_vec_size, x_thread[t], remaining);
      }
      for (int row = 0; row < rows; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device uint8_t* sl = sc + row * in_vec_size_g;
        U s = dequantize_scale<U, GS>(sl[0]);
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += small ? qdot<U, values_per_thread, BITS>(wl, x_thread[t], s)
                                  : qdot_safe<U, values_per_thread, BITS>(wl, x_thread[t], s, remaining);
        }
      }
    }
    for (int row = 0; row < rows; row++) {
      for (int t = 0; t < VERIFY_T; ++t) {
        U r = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          ys[t * out_vec_size + row] = static_cast<T>(r);
        }
      }
    }
"""
FP_PORTS = ('fp_fast', 'fp_fast_partial', 'fp_generic')
FP_MODES = {'mxfp8': (32, 8), 'mxfp4': (32, 4), 'nvfp4': (16, 4)}         # mode: (group size, bits)


def source(port, bits, multi=False):
    """A port's kernel source, for T_TOTAL tokens (_ragged)."""
    if port == 'fp_generic':
        text = _FP_GENERIC
    elif port.startswith('fp_'):
        text = _FP_FAST.replace('__PARTIAL__', 'true' if port == 'fp_fast_partial' else 'false')
    elif port == 'quad':
        text = _QUAD
    elif port == 'generic':
        text = _GENERIC.replace('__PACKS__', '1')
    else:
        packs = '1' if bits <= 2 else '2'
        text = _FAST.replace('__PACKS__', packs).replace('__PARTIAL__', 'true' if port == 'fast_partial' else 'false')
    return _ragged(text, multi)


def _ragged(text, multi):
    """A port for T_TOTAL tokens however they fall into blocks of VERIFY_T: a threadgroup's tokens
    from ROW0, LIVE of them real; one past them reads the last real one's input and writes what it
    gives, the same bits to the same place (the tokens were padded with zeros: a
    concatenation and a slice a projection a pass where they did not fall evenly).  multi: the
    rows of a group of a multi-head linear (_multi_source), T_TOTAL each."""
    start = 'uint t_block = threadgroup_position_in_grid.z;'
    assert text.count(start) == 1
    rows = ('const int group_ = t_block / T_BLOCKS;\n'
            '    const int block_ = int(t_block) - group_ * T_BLOCKS;\n'
            '    const int ROW0 = group_ * T_TOTAL + block_ * VERIFY_T;\n'
            '    const int LIVE = metal::min(VERIFY_T, T_TOTAL - block_ * VERIFY_T);') if multi else (
            'const int ROW0 = int(t_block) * VERIFY_T;\n'
            '    const int LIVE = metal::min(VERIFY_T, T_TOTAL - ROW0);')
    text = text.replace(start, start + '\n    ' + rows)
    for old, new in (('t_block * VERIFY_T * in_vec_size', 'ROW0 * in_vec_size'),
                     ('t_block * VERIFY_T * out_vec_size', 'ROW0 * out_vec_size'),
                     ('xs + t * in_vec_size', 'xs + metal::min(t, LIVE - 1) * in_vec_size'),
                     ('ys[t * out_vec_size', 'ys[metal::min(t, LIVE - 1) * out_vec_size')):
        assert old in text, old
        text = text.replace(old, new)
    return text


def _multi_source(port, bits):
    """A microscaling port's source for every group of a multi-head linear in one kernel: the
    threadgroups' z runs over the groups' token blocks (T_BLOCKS a group), and a group's weights and
    scales are its own; each row's arithmetic is the port's."""
    text = source(port, bits, multi=True)
    for row in ('out_row', 'base_row'):
        text = text.replace(f'ws += {row} * in_vec_size_w', f'ws += (group_ * out_vec_size + {row}) * in_vec_size_w')
        text = text.replace(f'sc += {row} * in_vec_size_g', f'sc += (group_ * out_vec_size + {row}) * in_vec_size_g')
    return text


@lru_cache(maxsize=None)
def _kernel(port, bits, group_size, multi=False):
    fp = port.startswith('fp_')
    return mx.fast.metal_kernel(name=f'quanteeze_verify_{port}_b{bits}_gs{group_size}' + ('_multi' if multi else ''),
                                input_names=['x', 'w', 'scales'] + ([] if fp else ['biases']), output_names=['y'],
                                header=(_fp_header if fp else _header)(bits, group_size),
                                source=_multi_source(port, bits) if multi else source(port, bits))


def _applies(port, k, n, bits=4, group_size=64):
    """Whether MLX dispatches a one-token decode of this shape to the port's original, as its
    quantized matmul chooses (one port a shape): qmv_quad for an input of 64 or 128 at bits a
    power of two; else qmv_fast for an input that is a multiple of 512 (with partial rows, an
    output that is not one of 8); else the generic qmv.  Previously the generic port was
    tried wherever another applied, and on the Mac it matched GLM-5.3's f_b_proj (an input of
    128: qmv_quad's) on one input of four -- by chance, so a shape could have been served by a
    port that does not compute what the decode does.  A microscaling format's, _fp_applies."""
    if port.startswith('fp_'):
        return _fp_applies(port, k, n, bits)
    quad = k in (64, 128) and bits in (2, 4, 8) and group_size >= k // 4
    if port == 'quad':
        return quad
    if port == 'fast':
        return not quad and k % 512 == 0 and n % 8 == 0
    if port == 'fast_partial':
        return not quad and k % 512 == 0 and n % 8 != 0
    return not quad and k % 512 != 0


def _fp_applies(port, k, n, bits):
    """_applies for a microscaling format: fp_qmv_fast for an input of whole blocks (a thread's
    two packs, 32 threads: 256 inputs at 8 bits, 512 at 4), with partial rows for an output not
    one of 8; else the generic fp_qmv (an input of 64 or 128 goes to fp_qmv_quad, not ported)."""
    block = 2 * (32 // bits) * 32
    if k in (64, 128):
        return False
    if port == 'fp_fast':
        return k % block == 0 and n % 8 == 0
    if port == 'fp_fast_partial':
        return k % block == 0 and n % 8 != 0
    return k % block != 0


def run(port, linear, x, per=None):
    """The port's outputs for x [1, T, K]: [1, T, N], at most `per` tokens a threadgroup (PER), as
    evenly as they go (8 by 4, 5 by 3): the weights are read once for each threadgroup's tokens;
    the last threadgroup's may be fewer (_ragged)."""
    k, t, n = x.shape[-1], x.shape[1], linear.weight.shape[0]
    blocks = -(-t // max(1, min(per or PER, t)))
    per = -(-t // blocks)
    x = mx.contiguous(x)
    if port == 'quad':
        grid, group = (32, -(-n // 64), blocks), (32, 1, 1)
    else:
        grid, group = (32, 2 * ((n + 7) // 8), blocks), (32, 2, 1)
    inputs = [x, linear.weight, linear.scales] + ([] if port.startswith('fp_') else [linear.biases])
    out = _kernel(port, int(linear.bits), int(linear.group_size))(
        inputs=inputs,
        template=[('T', x.dtype), ('VERIFY_T', int(per)), ('K_SIZE', int(k)), ('N_SIZE', int(n)),
                  ('T_TOTAL', int(t))],
        grid=grid, threadgroup=group, output_shapes=[(1, t, n)], output_dtypes=[x.dtype])[0]
    if 'bias' in linear:
        out = out + linear['bias']
    return out


def run_multi(port, multi, x, per=None):
    """A microscaling port's outputs for x [1, G, T, K] through a multi-head linear's G groups at once
    (_multi_source): [1, G, T, N], each group's as run() gives it alone."""
    groups, t, k = x.shape[1], x.shape[2], x.shape[3]
    n = multi['weight'].shape[1]
    blocks = -(-t // max(1, min(per or PER, t)))
    per = -(-t // blocks)
    x = mx.contiguous(x)
    return _kernel(port, int(multi.bits), int(multi.group_size), True)(
        inputs=[x, mx.contiguous(multi['weight']), mx.contiguous(multi['scales'])],
        template=[('T', x.dtype), ('VERIFY_T', int(per)), ('K_SIZE', int(k)), ('N_SIZE', int(n)),
                  ('T_BLOCKS', int(blocks)), ('T_TOTAL', int(t))],
        grid=(32, 2 * ((n + 7) // 8), groups * blocks), threadgroup=(32, 2, 1),
        output_shapes=[(1, groups, t, n)], output_dtypes=[x.dtype])[0]


def eligible(linear, x):
    if not (isinstance(linear, nn.QuantizedLinear) and x.ndim == 3 and x.shape[0] == 1
            and 1 < x.shape[1] <= MAX_TOKENS and x.dtype in (mx.bfloat16, mx.float16)
            and x.shape[-1] == linear.scales.shape[-1] * linear.group_size
            and mx.metal.is_available() and mx.default_device() == mx.gpu):
        return False
    mode = getattr(linear, 'mode', 'affine')
    if mode in FP_MODES:                              # a microscaling format: uint8 scales, no biases
        return ((int(linear.group_size), int(linear.bits)) == FP_MODES[mode] and linear.scales.dtype == mx.uint8
                and fp_helpers() is not None)
    return (mode == 'affine' and linear.get('biases') is not None and linear.bits in (2, 3, 4, 5, 6, 8)
            and linear.scales.dtype == x.dtype and linear.biases.dtype == x.dtype and helpers() is not None)


def _shape(linear, x):
    return (int(linear.bits), int(linear.group_size), int(x.shape[-1]), int(linear.weight.shape[0]), str(x.dtype),
            getattr(linear, 'mode', 'affine'))


def _decode(linear, x):
    """MLX's own outputs for x's tokens one at a time, as a decode computes them."""
    return mx.concatenate([linear(mx.contiguous(x[:, t:t + 1])) for t in range(x.shape[1])], axis=1)


class _Group:
    """One group of a quantized multi-head linear (mlx-vlm's QuantizedMultiLinear: DeepSeek V4.1's
    wo_a) as run() takes a projection."""

    def __init__(self, multi, g):
        self.weight, self.scales = multi['weight'][g], multi['scales'][g]
        self.biases = None if multi.get('biases') is None else multi['biases'][g]
        self.bits, self.group_size, self.mode = multi.bits, multi.group_size, multi.mode

    def __contains__(self, name):
        return False


def verify_multi(multi, x):
    """x [1, G, T, K] through a quantized multi-head linear (G groups, each its own weights) as T
    one-token calls would compute it, a group's weights read once, the groups in one kernel
    (run_multi: a kernel a group took wo_a from 8.7 to 16.9 ms a pass of 2, V4.1 on the Mac);
    None where no port matched
    (the caller keeps its path).  The port each group's shape takes is checked as verify_linear
    checks a projection's, against the multi-head call's own one-token outputs."""
    if not (x.ndim == 4 and x.shape[0] == 1 and 1 < x.shape[2] <= MAX_TOKENS and multi.mode in FP_MODES
            and fp_helpers() is not None and mx.metal.is_available() and mx.default_device() == mx.gpu):
        return None
    groups, t, k = x.shape[1], x.shape[2], x.shape[3]
    one = _Group(multi, 0)
    key = (int(one.bits), int(one.group_size), int(k), int(one.weight.shape[0]), str(x.dtype), multi.mode, 'multi')
    choice = CHOICE.get(key)
    if choice is False:
        STATS['declined'] += 1
        return None

    def ours(port):
        return run_multi(port, multi, x)
    seen = CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        STATS['served'] += 1
        return ours(choice)
    from mlx_vlm.models.mla import QuantizedMultiLinear
    want = mx.concatenate([QuantizedMultiLinear.__call__(multi, mx.contiguous(x[:, :, i:i + 1])) for i in range(t)], axis=2)
    STATS['checked'] += 1
    for port in [choice] if choice is not None else [p for p in FP_PORTS if _applies(p, k, key[3], key[0], key[1])]:
        try:
            same = mx.array_equal(ours(port), want).item()
        except Exception:                                 # noqa: BLE001 -- a kernel that does not build
            same = False
        if same:
            CHOICE[key], CHECKED[key] = port, seen + 1
            return want
    STATS['mismatched'] += 1
    CHOICE[key] = False
    return want


def verify_linear(linear, x):
    """x [1, T, K] through a quantized projection as T one-token decodes would compute it, the weights
    read once; None where no port is known to match (the caller keeps its path)."""
    if not eligible(linear, x):
        return None
    key = _shape(linear, x)
    choice = CHOICE.get(key)
    if choice is False:
        STATS['declined'] += 1
        return None
    seen = CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        STATS['served'] += 1
        return run(choice, linear, x)
    want = _decode(linear, x)
    STATS['checked'] += 1
    k, n = key[2], key[3]
    ports = FP_PORTS if key[5] in FP_MODES else PORTS
    order = [choice] if choice is not None else [p for p in ports if _applies(p, k, n, key[0], key[1])]
    for port in order:
        try:
            got = run(port, linear, x)
            same = mx.array_equal(got, want).item()
        except Exception as error:                          # a kernel that does not build: none for it
            if port not in _SAID:
                _SAID.add(port)
                print(f'quanteeze: verify kernel {port} unavailable ({type(error).__name__}: {error})',
                      file=sys.stderr, flush=True)
            same = False
        if same:
            CHOICE[key], CHECKED[key] = port, seen + 1
            return want
    STATS['mismatched'] += 1
    CHOICE[key] = False
    return want


def _hc_ways():
    from mlx_vlm.models.deepseek_v4 import hyper_connection as hc
    return hc


def _hc_tokens(op, x, residual, post, comb):
    """The decode's expansion, a token at a time."""
    return mx.concatenate([op(x[:, t:t + 1], residual[:, t:t + 1], post[:, t:t + 1], comb[:, t:t + 1])
                           for t in range(x.shape[1])], axis=1)


def verify_hc_expand(x, residual, post, comb, original):
    """A hyper-connection's expansion of x [B, T, D] into residual [B, T, hc, D] as T one-token
    decodes compute it; `original` is mlx-vlm's hc_expand, for what this does not take."""
    hc = _hc_ways()
    t = x.shape[1]
    if not 1 < t <= MAX_TOKENS:
        return original(x, residual, post, comb)
    key = (int(t), int(residual.shape[2]), int(x.shape[-1]), str(x.dtype))
    op = hc._hc_expand_op
    ways = {'block': lambda: op(x, residual, post, comb),
            'kernel': lambda: hc.exact_hc_expand(x, residual, post, comb),
            'tokens': lambda: _hc_tokens(op, x, residual, post, comb)}
    choice = HC_CHOICE.get(key)
    seen = HC_CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        STATS['hc_' + choice] = STATS.get('hc_' + choice, 0) + 1
        return ways[choice]()
    want = ways['tokens']()
    for way in ([choice] if choice is not None and choice != 'tokens' else HC_WAYS):
        try:
            got = ways[way]()
            same = got is not None and mx.array_equal(got, want).item()
        except Exception:
            same = False
        if same:
            HC_CHOICE[key], HC_CHECKED[key] = way, seen + 1
            return want
    HC_CHOICE[key], HC_CHECKED[key] = 'tokens', CHECKS
    return want


STAGES = ('embed_q', 'attention', 'unembed_out')


def _attend_at_once(attention, q, latent, topk, kv_cache, at_once=(True, True, True)):
    """GLM-5.3's _attend loop for q [1, H, T, D] with the T tokens as the batch: [1, H, T, Dv].
    at_once: for each of STAGES, whether it goes for the T tokens in one call, else a call a
    token as the loop makes it (the same calls: what the loop computes), with no async_eval
    between them."""
    from mlx_vlm.models.glm5_next import language as glm
    t = q.shape[2]
    rows = topk[0][:, None]                                          # [T, 1, K]
    valid = (rows >= 0) & (rows < latent.shape[2])
    safe = mx.clip(rows, 0, max(latent.shape[2] - 1, 0))
    latents = mx.broadcast_to(latent, (t,) + tuple(latent.shape[1:]))
    selected = mx.take_along_axis(latents, safe[:, None, 0, :, None], axis=2)     # [T, 1, K, C]
    embed, attend, unembed = at_once
    if embed:
        query = attention.embed_q(q[0].transpose(1, 0, 2)[:, :, None])             # [T, H, 1, C]
    else:
        query = mx.concatenate([attention.embed_q(q[:, :, i:i + 1]) for i in range(t)], axis=0)
    if attend:
        out = glm.scaled_dot_product_attention(query, selected, selected, cache=kv_cache, scale=attention.scale,
                                               mask=valid[:, None])
    else:
        out = mx.concatenate([glm.scaled_dot_product_attention(query[i:i + 1], selected[i:i + 1],
                                                               selected[i:i + 1], cache=kv_cache,
                                                               scale=attention.scale, mask=valid[i:i + 1, None])
                              for i in range(t)], axis=0)
    if unembed:
        out = attention.unembed_out(out)                                           # [T, H, 1, Dv]
    else:
        out = mx.concatenate([attention.unembed_out(out[i:i + 1]) for i in range(t)], axis=0)
    return out[:, :, 0].transpose(1, 0, 2)[None]                                   # [1, H, T, Dv]


def _stages(attention, q, latent, topk, kv_cache):
    """Which of STAGES give the loop's outputs for the T tokens at once, each given the loop's own
    inputs: (embed_q, attention, unembed_out), True where the same bit for bit."""
    from mlx_vlm.models.glm5_next import language as glm
    t = q.shape[2]
    rows = topk[0][:, None]
    valid = (rows >= 0) & (rows < latent.shape[2])
    safe = mx.clip(rows, 0, max(latent.shape[2] - 1, 0))
    selected = mx.take_along_axis(mx.broadcast_to(latent, (t,) + tuple(latent.shape[1:])),
                                  safe[:, None, 0, :, None], axis=2)
    queries = mx.concatenate([attention.embed_q(q[:, :, i:i + 1]) for i in range(t)], axis=0)
    outs = mx.concatenate([glm.scaled_dot_product_attention(queries[i:i + 1], selected[i:i + 1], selected[i:i + 1],
                                                            cache=kv_cache, scale=attention.scale,
                                                            mask=valid[i:i + 1, None]) for i in range(t)], axis=0)
    unembedded = mx.concatenate([attention.unembed_out(outs[i:i + 1]) for i in range(t)], axis=0)
    at_once = (attention.embed_q(q[0].transpose(1, 0, 2)[:, :, None]),
               glm.scaled_dot_product_attention(queries, selected, selected, cache=kv_cache, scale=attention.scale,
                                                mask=valid[:, None]),
               attention.unembed_out(outs))
    return tuple(bool(mx.array_equal(a, b).item()) for a, b in zip(at_once, (queries, outs, unembedded)))


def verify_attend(attention, original, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
    """Glm5NextAttention._attend for a verify pass: the stages that matched the loop for the tokens
    at once, the others a call a token, and one async_eval a layer rather than one a token (held
    back altogether, the Mac's pass of 2 tokens took 53.6 ms against 46.4: the GPU waited for the
    whole graph); anything else as it was.  On the Mac,
    the whole at once did not match the loop, so a mismatch is now told stage by
    stage (said once a shape) and the stages that match go at once."""
    t = q.shape[2]
    if (last_only or attention.training or q.shape[0] != 1 or not 1 < t <= MAX_TOKENS
            or hasattr(kv_cache, 'bits') or 'TurboQuant' in type(kv_cache).__name__):
        return original(q, latent, new_latent, topk, kv_cache, projected_cache, last_only)
    key = (int(t), int(q.shape[1]), int(q.shape[-1]), str(q.dtype))
    choice, seen = ATTEND_CHOICE.get(key), ATTEND_CHECKED.get(key, 0)
    if choice is not None and seen >= CHECKS:
        if not choice:
            return original(q, latent, new_latent, topk, kv_cache, projected_cache, last_only)
        STATS['attend_at_once'] = STATS.get('attend_at_once', 0) + 1
        out = _attend_at_once(attention, q, latent, topk, kv_cache, choice)
        mx.async_eval(out)                  # once a layer, as the loop's last: the GPU starts on what is built
        return out.transpose(0, 2, 1, 3).reshape(1, t, -1), topk
    want = original(q, latent, new_latent, topk, kv_cache, projected_cache, last_only)

    def matches(at_once):
        got = _attend_at_once(attention, q, latent, topk, kv_cache, at_once).transpose(0, 2, 1, 3).reshape(1, t, -1)
        return mx.array_equal(got, want[0]).item()
    try:
        tried = choice or (True, True, True)
        same = matches(tried)
        if not same and choice is None:
            stages = _stages(attention, q, latent, topk, kv_cache)
            print(f'quanteeze: GLM attention for {t} tokens at once: '
                  + ', '.join(f'{n} {"=" if ok else "x"}' for n, ok in zip(STAGES, stages))
                  + '; the others a token at a time', file=sys.stderr, flush=True)
            tried = stages
            same = matches(tried)
    except Exception as error:
        if 'attend' not in _SAID:
            _SAID.add('attend')
            print(f'quanteeze: verify attention at once unavailable ({type(error).__name__}: {error})',
                  file=sys.stderr, flush=True)
        same = False
    ATTEND_CHOICE[key] = tuple(tried) if same else False
    ATTEND_CHECKED[key] = seen + 1 if same else CHECKS
    return want


def install():
    """Route the verify passes' quantized projections through verify_linear where it matches: GLM's
    native_batch_linear, the Qwen verifiers' fallback after mlx-vlm's own kernel, and Flash-Next's
    singleton projections; the hyper-connections' expansion through verify_hc_expand; and GLM-5.3's
    sparse attention through verify_attend.  A decode step of several requests at once ([B, 1, K])
    goes through verify_linear too, else a row at a time: so each request decodes as it would
    alone, whatever it is batched with (Flash-Next's greedy text was the same alone, twice and
    beside a 6,941-token prompt for 150 tokens; mlx-vlm's batched kernel had parted it at token
    25)."""
    from mlx_vlm.models import linear as projections
    from mlx_vlm.speculative.ops import linear as ops

    batch = projections.native_batch_linear

    def native_batch_linear(module, x):
        out = verify_linear(module, x) if isinstance(module, nn.QuantizedLinear) else None
        return batch(module, x) if out is None else out
    projections.native_batch_linear = native_batch_linear

    optimized = ops._target_verify_optimized_affine_linear

    def target_verify_optimized_affine_linear(linear, x):
        if x.ndim == 3 and x.shape[0] > 1 and x.shape[1] == 1:
            # rows of a batch, a token each (a decode step of several requests): as each row's own
            # decode computes it, not by mlx-vlm's kernel, which rounded Flash-Next's 10240x4
            # hyper-connection projection otherwise
            out = verify_linear(linear, x.reshape(1, x.shape[0], x.shape[2]))
            return out.reshape(x.shape[0], 1, -1) if out is not None else ops._target_verify_singletons(linear, x)
        out = optimized(linear, x)
        return verify_linear(linear, x) if out is None else out
    ops._target_verify_optimized_affine_linear = target_verify_optimized_affine_linear

    try:
        hc = _hc_ways()
    except Exception:
        hc = None
    if hc is not None:
        expand = hc.hc_expand

        def hc_expand(x, residual, post, comb, *, use_kernel=True):
            if not use_kernel:
                return expand(x, residual, post, comb, use_kernel=False)
            return verify_hc_expand(x, residual, post, comb, expand)
        hc.hc_expand = hc_expand

    try:
        from mlx_vlm.models.glm5_next import language as glm
    except Exception:
        glm = None
    if glm is not None and not getattr(glm.Glm5NextAttention, '_quanteeze_attend', False):
        attend = glm.Glm5NextAttention._attend

        def _attend(self, *args, **kwargs):
            return verify_attend(self, lambda *a, **k: attend(self, *a, **k), *args, **kwargs)
        glm.Glm5NextAttention._attend = _attend
        glm.Glm5NextAttention._quanteeze_attend = True

    try:
        from mlx_vlm.models.qwen4_exp import language as qwen4
    except Exception:
        return
    singleton = qwen4.singleton_quantized_linear

    def singleton_quantized_linear(linear, x):
        out = verify_linear(linear, x)
        return singleton(linear, x) if out is None else out
    qwen4.singleton_quantized_linear = singleton_quantized_linear


def summary():
    return dict(STATS, shapes=len(CHOICE), matched=sum(1 for v in CHOICE.values() if v),
                ports={p: sum(1 for v in CHOICE.values() if v == p) for p in PORTS},
                hc_expand={str(k): v for k, v in HC_CHOICE.items()},
                attend_at_once={str(k): v if not v else '/'.join(n for n, ok in zip(STAGES, v) if ok) or 'none'
                                for k, v in ATTEND_CHOICE.items()})
