# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Shared Metal kernel device helpers matching MLX's own unary rounding and Edge semantics."""

# MLX's sigmoid unary op: 1 / (1 + exp(|x|)), and 1 - y for x >= 0.
SIGMOID = """
template <typename X>
inline X mlx_sigmoid(X x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""
