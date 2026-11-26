# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""GLM-5.3's hyper-connections (DeepSeek-V4's mHC: four streams collapsed into a branch's input, its
output expanded back) at a decode token in two kernels where mlx-vlm runs nine, each bit for bit what
it replaces (GLM-5.3's decode token is host bound; after its routing
and linear attention, its hyper-connections, 90 a token).

mlx-vlm's HyperConnection.apply_branch at one token:

- before the branch, six kernels: the streams cast to float32, their RMSNorm, the mixing weight (fn,
  24 x 16384, bfloat16 in the checkpoint) cast to float32 and its gemv, the sinkhorn-and-collapse
  kernel, and the branch's RMSNorm.  For blocks of 2 to 8 tokens mlx-vlm has a kernel that does all
  of it as the one-token path does (fast_ops.exact_hc_normalized_norm); pre() runs it on one token
  too.  On 48 inputs of GLM-5.3's size, the normalized input, post and comb were the same bits.
- after the branch, three: the residual cast to float32, comb transposed times it (MLX's matmul),
  and post times the output added, cast back (compiled).  expand() is one kernel, a thread a column.
  MLX's matmul of these shapes, here, is the neural accelerators' float32: comb's values cut to 10
  mantissa bits and the four products (then exact) added in pairs, (p0 + p1) + (p2 + p3): 81920 of
  81920 values matched that, no other order or rounding did.  post times the output is rounded
  before the add, as the compiled kernel computes it.  mlx-vlm's own short-block expand
  (fast_ops.exact_hc_expand, an 8 x 8 simdgroup MMA) differed from the one-token path everywhere.
  On a GPU whose matmul is another arithmetic, the checks find expand() different and keep mlx-vlm's.

90 hyper-connections a step on the host (a stand-in branch): 3.62 to 2.58 ms.

None is taken on trust: the first CHECKS times a shape is met, each part's output is compared bit for
bit with mlx-vlm's, and it serves the shape only if they are the same.  Only a step of one token
(more are mlx-vlm's fused path); QUANTEEZE_HC_KERNELS=off keeps mlx-vlm's (as it does Flash-Next's,
quanteeze.hyper).
"""
import sys
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection, hc_expand
from mlx_vlm.models.fast_ops import exact_hc_normalized_norm

from . import kernel_checks

CHECKS = 3
ON = [True]                     # QUANTEEZE_HC_KERNELS
STATS = dict(pre=0, expand=0, checked=0, mismatched=0)
CHOICE, CHECKED = kernel_checks.tables('glm_hyper')  # (part, shape): whether it matched mlx-vlm's
SAID = []

_EXPAND = r"""
    uint d = thread_position_in_grid.x;
    if (d >= D) {
      return;
    }
    float r[HC];
    for (int j = 0; j < HC; ++j) {
      r[j] = static_cast<float>(residual[j * D + d]);
    }
    float xf = static_cast<float>(x[d]);
    for (int h = 0; h < HC; ++h) {
      // MLX's matmul of comb transposed and the residual: comb cut to 10 mantissa bits, the products
      // (exact) added in pairs
      float p[HC];
      for (int j = 0; j < HC; ++j) {
        float c = as_type<float>(as_type<uint>(comb[j * HC + h]) & 0xffffe000u);
        p[j] = c * r[j];
      }
      float mm = (p[0] + p[1]) + (p[2] + p[3]);
      volatile float product = post[h] * xf;
      out[h * D + d] = static_cast<T>(product + mm);
    }
"""


@lru_cache(maxsize=None)
def _expand_kernel():
    return mx.fast.metal_kernel(name='quanteeze_glm_hc_expand', input_names=['x', 'residual', 'post', 'comb'],
                                output_names=['out'], source=_EXPAND)


@lru_cache(maxsize=None)
def _metal():
    return mx.metal.is_available()


def applies(connection, x):
    """Whether this step's hyper-connection is pre() and expand()'s: one token of four streams."""
    return (ON[0] and _metal() and not connection.training and x.ndim == 4 and x.shape[:2] == (1, 1)
            and x.shape[2] == 4 == connection.hc_mult and x.dtype in (mx.bfloat16, mx.float16))


def expand(x, residual, post, comb):
    """The branch's output x [1, 1, D] expanded into the streams: post times x plus comb transposed
    times the residual [1, 1, 4, D], in its dtype."""
    d = x.shape[-1]
    return _expand_kernel()(inputs=[x, residual, post, comb], template=[('T', x.dtype), ('HC', 4), ('D', d)],
                            grid=(d, 1, 1), threadgroup=(256, 1, 1), output_shapes=[residual.shape],
                            output_dtypes=[x.dtype])[0]


def collapsed(connection, norm, x):
    """(the branch's input, post, comb): mlx-vlm's exact kernel where it takes the shape and it
    matched the one-token path, else that path."""
    def theirs():
        c, post, comb = HyperConnection.__call__(connection, x)
        return norm(c), post, comb
    w = connection.fn
    if not (isinstance(norm, nn.RMSNorm) and norm.weight.dtype == x.dtype and x.shape[2:] == (4, 4096)
            and w.shape == (24, 4 * 4096) and w.dtype in (mx.bfloat16, mx.float16)):      # what the kernel takes
        return theirs()
    key = (tuple(x.shape), str(x.dtype), tuple(connection.fn.shape), str(connection.fn.dtype),
           connection.sinkhorn_iters, connection.hc_eps, norm.eps)
    return _served('pre', key, lambda: exact_hc_normalized_norm(connection, norm, x), theirs)


def expanded(connection, output, x, post, comb):
    """The branch's output expanded: expand() where it matched mlx-vlm's, else mlx-vlm's."""
    def theirs():
        return hc_expand(output, x, post, comb, use_kernel=not connection.training)
    if output.shape != (1, 1, x.shape[-1]) or output.dtype != x.dtype or post.dtype != mx.float32:
        return theirs()
    return _served('expand', (tuple(x.shape), str(x.dtype)), lambda: expand(output, x, post, comb), theirs)


def apply_branch(connection, x, norm, branch, *args, **kwargs):
    """HyperConnection.apply_branch at one token through collapsed() and expanded()."""
    normed, post, comb = collapsed(connection, norm, x)
    output = branch(normed, *args, **kwargs)
    if isinstance(output, tuple):
        return (expanded(connection, output[0], x, post, comb), *output[1:])
    return expanded(connection, output, x, post, comb)


def _served(part, key, mine, theirs):
    """mine() where this part's shape matched theirs() bit for bit the first CHECKS times it was
    met, else theirs()."""
    return kernel_checks.served(sys.modules[__name__], (part,) + key, mine, theirs,
                                "GLM-5.3's hyper-connection kernels", stat=part)


def summary():
    return dict(STATS, shapes={str(k): v for k, v in CHOICE.items()})
