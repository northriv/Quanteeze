# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""What quanteeze's GPU kernels share on MLX's CUDA backend (mx.fast.cuda_kernel, an A100's): whether
it is the backend in use, and a header giving a kernel MLX's own element-wise functors (unary_ops.cuh,
binary_ops.cuh: Sigmoid, Rsqrt, Multiply, Add as MLX's kernels and compiled graphs apply them) and a
sum for cooperative groups' reductions as MLX's reduce ops add, so that a kernel can repeat MLX's CUDA
arithmetic op for op and match it bit for bit.

A kernel's source is CUDA C++ inside the signature MLX generates (each input a const pointer, each
output a pointer, template constants as constexpr), in namespace mlx::core::cu with cg for
cooperative groups; its grid is counted in threads, as Metal's.  What a kernel computes and how it
was matched against MLX (the order of each sum, where nvcc fuses a multiply and an add) is told where
the kernel is (quanteeze.hyper, quanteeze.linear_attn).
"""
import mlx.core as mx

FUSE = [True]                   # QUANTEEZE_CUDA_FUSE: a layer's parts fused across one another

HEADER = r"""
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include <cooperative_groups/reduce.h>
// MLX's Multiply rounded to its type before anything adds to it.  On sm_90 and later cuda_bf16's
// operator* is a bare mul.bf16 (mul.f16 for half), which ptxas may fuse with an add that follows
// into one fma; on sm_80 it is fma.rn with -0, which it never fuses, and mul.rn is that same rounding.
// For a product whose sum MLX adds in another kernel.
struct QzMul {
  template <typename U>
  __device__ __forceinline__ U operator()(U a, U b) const {
    if constexpr (cuda::std::is_same_v<U, __nv_bfloat16> || cuda::std::is_same_v<U, __half>) {
      return __hmul_rn(a, b);
    } else if constexpr (cuda::std::is_same_v<U, float>) {
      return __fmul_rn(a, b);
    } else {
      return mlx::core::cu::Multiply{}(a, b);
    }
  }
};
struct QzSum {
  template <typename U>
  __device__ __forceinline__ U operator()(U a, U b) const { return a + b; }
};
"""


def available():
    """Whether MLX runs on its CUDA backend here and can build a CUDA kernel."""
    return bool(getattr(getattr(mx, 'cuda', None), 'is_available', lambda: False)()) and hasattr(mx.fast, 'cuda_kernel')


def roomy(n, dtype):
    """How many elements a kernel's output of n `dtype` values is given: 16 bytes at least.  MLX's CUDA
    allocator keeps arrays of 8 bytes or less in a pool of managed memory, which a kernel reading them
    took 8.5 us longer to run, written just before by another kernel or not; a view of the first n of a
    larger array reads as fast as any."""
    return max(int(n), -(-16 // dtype.size))


def kernel(name, input_names, output_names, source, header='', shared_memory=0):
    """mx.fast.cuda_kernel with HEADER before `header` (and `shared_memory` bytes of dynamic shared memory,
    extern __shared__ in the source)."""
    return mx.fast.cuda_kernel(name=name, input_names=input_names, output_names=output_names, source=source,
                               header=HEADER + header, shared_memory=shared_memory)
