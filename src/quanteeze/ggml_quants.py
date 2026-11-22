# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Standard ggml `Q2_K`, `Q3_K`, and `IQ2_XS` quantizers and dequantizers in MLX array ops.

- `Q2_K` (2.625 bpw, 84 bytes per 256): 16-value sub-blocks `w ~ d * sc * q - dmin * m`, `q` in `0..3`,
  4-bit `sc` and `m` against block float16 `d` and `dmin`.
- `Q3_K` (3.4375 bpw, 110 bytes per 256): 16-value sub-blocks `w ~ d * sc * q`, `q` in `-4..3`,
  signed 6-bit `sc` against block float16 `d`.
- `IQ2_XS` (2.3125 bpw, 74 bytes per 256): 8-value groups on ggml's 512-point grid (magnitudes 8, 25, 43)
  with 7 explicit signs + 8th by even parity, 4-bit scale per 16 values `d * (0.5 + l) * 0.25`.

All packed outputs match `quanteeze.iq_experts.decode` bit-for-bit in float32."""
import mlx.core as mx
import numpy as np

from . import cuda_kernels
from .iq_experts import BLOCK, IQ2_XS, Q2_K, Q3_K, QK, compact_grids, tables

BITS = {
    'q2_k': BLOCK[Q2_K] * 8 / QK,
    'q3_k': BLOCK[Q3_K] * 8 / QK,
    'iq2_xs': BLOCK[IQ2_XS] * 8 / QK,
}                                                   # 2.625, 3.4375, 2.3125
ITERATIONS = 8
CANDIDATES = np.linspace(0.6, 1.4, 9)
ROWS = 64
GPU_PACK = [True]                                   # fused Metal/CUDA packing when on GPU (tests toggle False)
_KERNELS = {}


def _gpu_backend():
    if not GPU_PACK[0] or mx.default_device() != mx.gpu:
        return None
    if mx.metal.is_available():
        return 'metal'
    if cuda_kernels.available():
        return 'cuda'
    return None


def _f16(a):
    return a.astype(mx.float16).astype(mx.float32)


def _weights(w, plain):
    rows, cols = w.shape
    b = w.reshape(rows, cols // QK, QK)
    if plain is not None:
        return plain(b)
    sigma2 = 2 * (b * b).mean(-1, keepdims=True)
    return mx.sqrt(sigma2 + b * b)


def _packed(fields, shifts, axis):
    return (fields.astype(mx.uint32) << shifts).sum(axis).astype(mx.uint8)


def _rows(w):
    w = mx.array(w) if not isinstance(w, mx.array) else w
    if w.ndim < 2 or w.shape[-1] % QK:
        raise ValueError(f"ggml blocks need rows of a multiple of {QK} values, not {w.shape}")
    lead = tuple(w.shape[:-1])
    return w.astype(mx.float32).reshape(-1, w.shape[-1]), lead, w.shape[-1] // QK


def _scale_min(w, top, iterations):
    rows, cols = w.shape
    n = rows * cols // QK
    c = _weights(w, mx.abs).reshape(n, 16, 16)
    g = w.reshape(n, 16, 16)
    base = mx.minimum(g.min(-1, keepdims=True), 0.0)
    scale = mx.maximum((g.max(-1, keepdims=True) - base) / top, 1e-30)
    for _ in range(iterations):
        q = mx.clip(mx.round((g - base) / scale), 0, top)
        s0, s1, s2 = c.sum(-1, keepdims=True), (c * q).sum(-1, keepdims=True), (c * q * q).sum(-1, keepdims=True)
        t0, t1 = (c * g).sum(-1, keepdims=True), (c * q * g).sum(-1, keepdims=True)
        det = s0 * s2 - s1 * s1
        ok = det > 1e-30 * mx.maximum(s0 * s2, 1e-30)
        s = mx.where(ok, (s0 * t1 - s1 * t0) / mx.where(ok, det, 1.0), scale)
        b = mx.where(ok, (t0 - s * s1) / mx.maximum(s0, 1e-30), base)
        pinned = b > 0
        s = mx.where(pinned, t1 / mx.maximum(s2, 1e-30), s)
        b = mx.where(pinned, 0.0, b)
        scale, base = mx.where(s > 0, s, scale), b
    d = _f16(scale.max(1, keepdims=True) / 15)
    dmin = _f16((-base).max(1, keepdims=True) / 15)
    sc = mx.where(d > 0, mx.clip(mx.round(scale / mx.where(d > 0, d, 1.0)), 0, 15), 0.0)
    m = mx.where(dmin > 0, mx.clip(mx.round(-base / mx.where(dmin > 0, dmin, 1.0)), 0, 15), 0.0)
    step, low = d * sc, dmin * m
    q = mx.where(step > 0, mx.clip(mx.round((g + low) / mx.where(step > 0, step, 1.0)), 0, top), 0.0)
    return sc, m, d, dmin, q, step * q - low


def _q2_k_pack_ops(w, iterations=ITERATIONS):
    """Reference pure-MLX array-op implementation of `Q2_K` packing."""
    w, lead, nb = _rows(w)
    n = w.shape[0] * nb
    sc, m, d, dmin, q, values = _scale_min(w, 3, iterations)
    scales = _packed(mx.stack([sc.reshape(n, 16), m.reshape(n, 16)], axis=-1), mx.array([0, 4], mx.uint32), -1)
    codes = _packed(q.reshape(n, 2, 4, 32), (2 * mx.arange(4, dtype=mx.uint32))[:, None], 2).reshape(n, 64)
    body = mx.concatenate([scales, codes], axis=-1).reshape(*lead, nb, 80)
    return (
        body,
        d.reshape(*lead, nb).astype(mx.float16),
        dmin.reshape(*lead, nb).astype(mx.float16),
        values.reshape(*lead, w.shape[1]),
    )


def _symmetric(g, c, scale, iterations):
    for _ in range(iterations):
        q = mx.clip(mx.round(g / scale), -4, 3)
        s2 = (c * q * q).sum(-1, keepdims=True)
        scale = mx.where(s2 > 0, (c * q * g).sum(-1, keepdims=True) / mx.where(s2 > 0, s2, 1.0), scale)
    q = mx.clip(mx.round(g / scale), -4, 3)
    return scale, (c * (g - scale * q) ** 2).sum(-1, keepdims=True)


def _q3_k_pack_ops(w, iterations=ITERATIONS):
    """Reference pure-MLX array-op implementation of `Q3_K` packing."""
    w, lead, nb = _rows(w)
    rows, cols = w.shape
    n = rows * nb
    c = _weights(w, lambda b: b * b).reshape(n, 16, 16)
    g = w.reshape(n, 16, 16)
    top = mx.take_along_axis(g, mx.argmax(mx.abs(g), axis=-1, keepdims=True), axis=-1)
    top = mx.where(top == 0, 1e-30, top)
    at_low, low_err = _symmetric(g, c, top / -4, iterations)
    at_high, high_err = _symmetric(g, c, top / 3, iterations)
    scale = mx.where(high_err < low_err, at_high, at_low)
    biggest = mx.take_along_axis(scale, mx.argmax(mx.abs(scale), axis=1, keepdims=True), axis=1)
    d = _f16(-biggest / 32)
    sc = mx.where(d != 0, mx.clip(mx.round(scale / mx.where(d != 0, d, 1.0)), -32, 31), 0.0)
    step = d * sc
    q = mx.where(step != 0, mx.clip(mx.round(g / mx.where(step != 0, step, 1.0)), -4, 3), 0.0)
    values = (step * q).reshape(*lead, cols)
    q = q.reshape(n, QK)
    u = (sc.reshape(n, 16) + 32).astype(mx.uint32)
    high = _packed((q >= 0).reshape(n, 8, 32), mx.arange(8, dtype=mx.uint32)[:, None], 1)
    codes = _packed(mx.where(q >= 0, q, q + 4).reshape(n, 2, 4, 32), (2 * mx.arange(4, dtype=mx.uint32))[:, None], 2)
    low4 = _packed(mx.stack([u[:, 0:8] & 15, u[:, 8:16] & 15], axis=-1), mx.array([0, 4], mx.uint32), -1)
    top2 = _packed((u.reshape(n, 4, 4) >> 4) & 3, (2 * mx.arange(4, dtype=mx.uint32))[:, None], 1)
    body = mx.concatenate([high, codes.reshape(n, 64), low4, top2], axis=-1).reshape(*lead, nb, 108)
    return body, d.reshape(*lead, nb).astype(mx.float16), None, values


def _nearest(t, c, sigma, grid, grid2):
    a = (c * t) @ grid.T
    b = c @ grid2.T
    k = mx.argmin(sigma * sigma * b - 2 * sigma * a, axis=-1)
    pick = lambda m: mx.take_along_axis(m, k[..., None], axis=-1)[..., 0]  # noqa: E731
    return k, pick(a), pick(b)


def _iq2_xs_pack_ops(w):
    """Reference pure-MLX array-op implementation of `IQ2_XS` packing."""
    w, lead, nb = _rows(w)
    rows, cols = w.shape
    grid = tables()[IQ2_XS]
    grid2 = grid * grid
    weights = _weights(w, None)
    bodies, ds, values = [], [], []
    shifts7 = mx.arange(7, dtype=mx.uint32)
    for r in range(0, rows, ROWS):
        x = w[r:r + ROWS]
        n = x.shape[0] * nb
        t = mx.abs(x).reshape(n, 16, 2, 8)
        c = weights[r:r + ROWS].reshape(n, 16, 2, 8)
        neg = x.reshape(n, 16, 2, 8) < 0
        odd = (neg.astype(mx.int32).sum(-1, keepdims=True) % 2) == 1
        cheapest = mx.argmin(c * t * t, axis=-1, keepdims=True) == mx.arange(8)
        flip = odd & cheapest
        t = mx.where(flip, -t, t)
        neg = mx.logical_xor(neg, flip)
        top = mx.maximum(mx.abs(t).max(axis=(-1, -2), keepdims=True), 1e-30) / 43
        best_err, best_sigma = None, None
        for f in CANDIDATES:
            _, a, b = _nearest(t, c, top * f, grid, grid2)
            sigma = (a.sum(-1, keepdims=True) / mx.maximum(b.sum(-1, keepdims=True), 1e-30))[..., None]
            _, a, b = _nearest(t, c, sigma, grid, grid2)
            err = sigma[..., 0, 0] ** 2 * b.sum(-1) - 2 * sigma[..., 0, 0] * a.sum(-1)
            if best_err is None:
                best_err, best_sigma = err, sigma
            else:
                better = err < best_err
                best_err = mx.where(better, err, best_err)
                best_sigma = mx.where(better[..., None, None], sigma, best_sigma)
        sigma = mx.maximum(best_sigma, 0.0)
        d = _f16(sigma.max(axis=1, keepdims=True) * 4 / 15.5)
        l = mx.where(d > 0, mx.clip(mx.round(4 * sigma / mx.where(d > 0, d, 1.0) - 0.5), 0, 15), 0.0)
        quantized = d * (0.5 + l) * 0.25
        k, _, _ = _nearest(t, c, quantized, grid, grid2)
        sign = mx.where(neg, -1.0, 1.0)
        values.append((sign * quantized * mx.take(grid, k, axis=0)).reshape(x.shape[0], cols))
        sign7 = (neg.reshape(n, 32, 8)[..., :7].astype(mx.uint32) << shifts7).sum(-1)
        u = k.reshape(n, 32).astype(mx.uint32) | (sign7 << 9)
        qs = mx.stack([u & 255, u >> 8], axis=-1).reshape(n, 64).astype(mx.uint8)
        lu = l.reshape(n, 16).astype(mx.uint32)
        sc = (lu[:, 0::2] | (lu[:, 1::2] << 4)).astype(mx.uint8)
        body_chunk = mx.concatenate([qs, sc], axis=-1)
        d_chunk = d.reshape(n).astype(mx.float16)
        mx.eval(body_chunk, d_chunk, values[-1])
        bodies.append(body_chunk)
        ds.append(d_chunk)
    body = (bodies[0] if len(bodies) == 1 else mx.concatenate(bodies, axis=0)).reshape(*lead, nb, 72)
    d_all = (ds[0] if len(ds) == 1 else mx.concatenate(ds, axis=0)).reshape(*lead, nb)
    val_all = (values[0] if len(values) == 1 else mx.concatenate(values, axis=0)).reshape(*lead, cols)
    return body, d_all, None, val_all


_METAL_HEADER = r"""
static constant float IQ2_MAG[4] = {8.0f, 25.0f, 43.0f, 0.0f};
"""

_Q2_K_METAL = r"""
  const size_t tid = size_t(thread_position_in_grid.x);
  const size_t blk = tid >> 5;
  const uint lane = uint(tid) & 31u;
  if (blk >= meta[0]) return;
  const uint sub = lane >> 1;

  const device T* wr = w + blk * 256u + size_t(lane) * 8u;
  float g[8], c[8];
  float gmin_loc = INFINITY, gmax_loc = -INFINITY;
  for (int i = 0; i < 8; i++) {
    const float gv = float(wr[i]);
    g[i] = gv;
    c[i] = metal::fabs(gv);
    gmin_loc = metal::fmin(gmin_loc, gv);
    gmax_loc = metal::fmax(gmax_loc, gv);
  }
  const float gmin = metal::fmin(gmin_loc, simd_shuffle_xor(gmin_loc, 1u));
  const float gmax = metal::fmax(gmax_loc, simd_shuffle_xor(gmax_loc, 1u));
  float base = metal::fmin(gmin, 0.0f);
  float scale = metal::fmax((gmax - base) / 3.0f, 1e-30f);

  for (uint it = 0u; it < meta[1]; it++) {
    float ls0 = 0.0f, ls1 = 0.0f, ls2 = 0.0f, lt0 = 0.0f, lt1 = 0.0f;
    for (int i = 0; i < 8; i++) {
      const float qv = metal::clamp(metal::rint((g[i] - base) / scale), 0.0f, 3.0f);
      const float cq = c[i] * qv;
      ls0 += c[i];
      ls1 += cq;
      ls2 += cq * qv;
      lt0 += c[i] * g[i];
      lt1 += cq * g[i];
    }
    const float s0 = ls0 + simd_shuffle_xor(ls0, 1u);
    const float s1 = ls1 + simd_shuffle_xor(ls1, 1u);
    const float s2 = ls2 + simd_shuffle_xor(ls2, 1u);
    const float t0 = lt0 + simd_shuffle_xor(lt0, 1u);
    const float t1 = lt1 + simd_shuffle_xor(lt1, 1u);
    const float det = s0 * s2 - s1 * s1;
    const bool ok = det > 1e-30f * metal::fmax(s0 * s2, 1e-30f);
    float s = ok ? (s0 * t1 - s1 * t0) / det : scale;
    float b = ok ? (t0 - s * s1) / metal::fmax(s0, 1e-30f) : base;
    if (b > 0.0f) {
      s = t1 / metal::fmax(s2, 1e-30f);
      b = 0.0f;
    }
    if (s > 0.0f) scale = s;
    base = b;
  }

  const float max_scale = simd_max(scale);
  const float max_neg_base = metal::fmax(simd_max(-base), 0.0f);
  const half dh = half(max_scale / 15.0f);
  const half dminh = half(max_neg_base / 15.0f);
  const float d = float(dh);
  const float dmin = float(dminh);
  if (lane == 0u) {
    d_out[blk] = dh;
    dmin_out[blk] = dminh;
  }

  const uint sc_u = (d > 0.0f) ? uint(metal::clamp(metal::rint(scale / d), 0.0f, 15.0f)) : 0u;
  const uint m_u = (dmin > 0.0f) ? uint(metal::clamp(metal::rint(-base / dmin), 0.0f, 15.0f)) : 0u;
  const float step = d * float(sc_u);
  const float low = dmin * float(m_u);

  device uint8_t* out_b = body + blk * 80u;
  if ((lane & 1u) == 0u) out_b[sub] = uint8_t(sc_u | (m_u << 4));

  const uint h = lane >> 4;
  const uint sh = 2u * ((lane >> 2) & 3u);
  const uint lbase = (lane & 3u) * 8u;
  uint w0 = 0u, w1 = 0u;
  for (int i = 0; i < 8; i++) {
    const uint qv = (step > 0.0f) ? uint(metal::clamp(metal::rint((g[i] + low) / step), 0.0f, 3.0f)) : 0u;
    const uint p = qv << (sh + 8u * (uint(i) & 3u));
    if (i < 4) w0 |= p;
    else w1 |= p;
  }
  w0 |= simd_shuffle_xor(w0, 4u);
  w0 |= simd_shuffle_xor(w0, 8u);
  w1 |= simd_shuffle_xor(w1, 4u);
  w1 |= simd_shuffle_xor(w1, 8u);
  if ((lane & 12u) == 0u) {
    device uint8_t* dst = out_b + 16u + 32u * h + lbase;
    for (uint k = 0u; k < 4u; k++) {
      dst[k] = uint8_t((w0 >> (8u * k)) & 0xffu);
      dst[4u + k] = uint8_t((w1 >> (8u * k)) & 0xffu);
    }
  }
"""

_Q3_K_METAL = r"""
  const size_t tid = size_t(thread_position_in_grid.x);
  const size_t blk = tid >> 5;
  const uint lane = uint(tid) & 31u;
  if (blk >= meta[0]) return;

  const device T* wr = w + blk * 256u + size_t(lane) * 8u;
  float g[8], c[8];
  float best_abs = -1.0f, best_g = 0.0f;
  uint best_idx = lane * 8u;
  for (int i = 0; i < 8; i++) {
    const float gv = float(wr[i]);
    g[i] = gv;
    c[i] = gv * gv;
    const float av = metal::fabs(gv);
    if (av > best_abs) {
      best_abs = av;
      best_g = gv;
      best_idx = lane * 8u + uint(i);
    }
  }
  const float other_abs = simd_shuffle_xor(best_abs, 1u);
  const float other_g = simd_shuffle_xor(best_g, 1u);
  const uint other_idx = simd_shuffle_xor(best_idx, 1u);
  if (other_abs > best_abs || (other_abs == best_abs && other_idx < best_idx)) best_g = other_g;
  const float top = (best_g == 0.0f) ? 1e-30f : best_g;

  float at_lo = top / -4.0f;
  for (uint it = 0u; it < meta[1]; it++) {
    float l2 = 0.0f, lt = 0.0f;
    for (int i = 0; i < 8; i++) {
      const float qv = metal::clamp(metal::rint(g[i] / at_lo), -4.0f, 3.0f);
      const float cq = c[i] * qv;
      l2 += cq * qv;
      lt += cq * g[i];
    }
    const float s2 = l2 + simd_shuffle_xor(l2, 1u);
    const float t1 = lt + simd_shuffle_xor(lt, 1u);
    if (s2 > 0.0f) at_lo = t1 / s2;
  }
  float le_lo = 0.0f;
  for (int i = 0; i < 8; i++) {
    const float qv = metal::clamp(metal::rint(g[i] / at_lo), -4.0f, 3.0f);
    const float err = g[i] - at_lo * qv;
    le_lo += c[i] * err * err;
  }
  const float low_err = le_lo + simd_shuffle_xor(le_lo, 1u);

  float at_hi = top / 3.0f;
  for (uint it = 0u; it < meta[1]; it++) {
    float l2 = 0.0f, lt = 0.0f;
    for (int i = 0; i < 8; i++) {
      const float qv = metal::clamp(metal::rint(g[i] / at_hi), -4.0f, 3.0f);
      const float cq = c[i] * qv;
      l2 += cq * qv;
      lt += cq * g[i];
    }
    const float s2 = l2 + simd_shuffle_xor(l2, 1u);
    const float t1 = lt + simd_shuffle_xor(lt, 1u);
    if (s2 > 0.0f) at_hi = t1 / s2;
  }
  float le_hi = 0.0f;
  for (int i = 0; i < 8; i++) {
    const float qv = metal::clamp(metal::rint(g[i] / at_hi), -4.0f, 3.0f);
    const float err = g[i] - at_hi * qv;
    le_hi += c[i] * err * err;
  }
  const float high_err = le_hi + simd_shuffle_xor(le_hi, 1u);

  const float scale = (high_err < low_err) ? at_hi : at_lo;

  uint bsub = lane >> 1;
  float babs = metal::fabs(scale);
  float bval = scale;
  for (uint off = 16u; off >= 2u; off >>= 1) {
    const float oa = simd_shuffle_xor(babs, off);
    const float ov = simd_shuffle_xor(bval, off);
    const uint os = simd_shuffle_xor(bsub, off);
    if (oa > babs || (oa == babs && os < bsub)) {
      babs = oa;
      bval = ov;
      bsub = os;
    }
  }

  const half dh = half(-bval / 32.0f);
  const float d = float(dh);
  if (lane == 0u) d_out[blk] = dh;

  const int sc_i = (d != 0.0f) ? int(metal::clamp(metal::rint(scale / d), -32.0f, 31.0f)) : 0;
  const uint u = uint(sc_i + 32);
  const float step = d * float(sc_i);
  int qi[8];
  for (int i = 0; i < 8; i++) {
    qi[i] = (step != 0.0f) ? int(metal::clamp(metal::rint(g[i] / step), -4.0f, 3.0f)) : 0;
  }

  device uint8_t* out_b = body + blk * 108u;

  const uint hb = lane >> 2;
  uint h0 = 0u, h1 = 0u;
  for (int i = 0; i < 8; i++) {
    if (qi[i] >= 0) {
      const uint bit = 1u << (hb + 8u * (uint(i) & 3u));
      if (i < 4) h0 |= bit;
      else h1 |= bit;
    }
  }
  h0 |= simd_shuffle_xor(h0, 4u);
  h0 |= simd_shuffle_xor(h0, 8u);
  h0 |= simd_shuffle_xor(h0, 16u);
  h1 |= simd_shuffle_xor(h1, 4u);
  h1 |= simd_shuffle_xor(h1, 8u);
  h1 |= simd_shuffle_xor(h1, 16u);
  if (lane < 4u) {
    device uint8_t* hd = out_b + lane * 8u;
    for (uint k = 0u; k < 4u; k++) {
      hd[k] = uint8_t((h0 >> (8u * k)) & 0xffu);
      hd[4u + k] = uint8_t((h1 >> (8u * k)) & 0xffu);
    }
  }

  const uint h = lane >> 4;
  const uint sh = 2u * ((lane >> 2) & 3u);
  const uint lbase = (lane & 3u) * 8u;
  uint c0 = 0u, c1 = 0u;
  for (int i = 0; i < 8; i++) {
    const uint qlow = uint(qi[i] >= 0 ? qi[i] : qi[i] + 4);
    const uint p = qlow << (sh + 8u * (uint(i) & 3u));
    if (i < 4) c0 |= p;
    else c1 |= p;
  }
  c0 |= simd_shuffle_xor(c0, 4u);
  c0 |= simd_shuffle_xor(c0, 8u);
  c1 |= simd_shuffle_xor(c1, 4u);
  c1 |= simd_shuffle_xor(c1, 8u);
  if ((lane & 12u) == 0u) {
    device uint8_t* cd = out_b + 32u + 32u * h + lbase;
    for (uint k = 0u; k < 4u; k++) {
      cd[k] = uint8_t((c0 >> (8u * k)) & 0xffu);
      cd[4u + k] = uint8_t((c1 >> (8u * k)) & 0xffu);
    }
  }

  const uint u_hi = simd_shuffle_xor(u, 16u);
  if ((lane & 1u) == 0u && lane < 16u) out_b[96u + (lane >> 1)] = uint8_t((u & 15u) | ((u_hi & 15u) << 4));

  uint t2 = ((u >> 4) & 3u) << (2u * (lane >> 3));
  t2 |= simd_shuffle_xor(t2, 8u);
  t2 |= simd_shuffle_xor(t2, 16u);
  if ((lane & 1u) == 0u && lane < 8u) out_b[104u + (lane >> 1)] = uint8_t(t2);
"""

_IQ2_XS_METAL = r"""
  threadgroup uint16_t sh_grid[512];
  const uint ltid = thread_position_in_threadgroup.x;
  sh_grid[ltid] = grid_u16[ltid];
  sh_grid[ltid + 256u] = grid_u16[ltid + 256u];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const size_t tid = size_t(thread_position_in_grid.x);
  const size_t blk = tid >> 5;
  const uint lane = uint(tid) & 31u;
  if (blk >= meta[0]) return;

  const device T* wr = w + blk * 256u + size_t(lane) * 8u;
  float x[8];
  float lsq = 0.0f;
  for (int i = 0; i < 8; i++) {
    const float xi = float(wr[i]);
    x[i] = xi;
    lsq += xi * xi;
  }
  const float sigma2 = simd_sum(lsq) * (1.0f / 128.0f);

  float c[8], t[8];
  uint neg_mask = 0u;
  float max_abs = 0.0f;
  for (int i = 0; i < 8; i++) {
    const float xi = x[i];
    const float ai = metal::fabs(xi);
    c[i] = metal::sqrt(sigma2 + xi * xi);
    t[i] = ai;
    if (xi < 0.0f) neg_mask |= (1u << uint(i));
    max_abs = metal::fmax(max_abs, ai);
  }
  if ((metal::popcount(neg_mask) & 1u) != 0u) {
    int best_i = 0;
    float best_loss = c[0] * t[0] * t[0];
    for (int i = 1; i < 8; i++) {
      const float loss = c[i] * t[i] * t[i];
      if (loss < best_loss) {
        best_loss = loss;
        best_i = i;
      }
    }
    t[best_i] = -t[best_i];
    neg_mask ^= (1u << uint(best_i));
  }
  float ct[8];
  for (int i = 0; i < 8; i++) ct[i] = c[i] * t[i];

  const float sub_max = metal::fmax(max_abs, simd_shuffle_xor(max_abs, 1u));
  const float top = metal::fmax(sub_max, 1e-30f) / 43.0f;

  constexpr float CANDS[9] = {0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f};
  float s0[9], bsc0[9], ba0[9], bb0[9];
  for (int ci = 0; ci < 9; ci++) s0[ci] = top * CANDS[ci];
  for (int k = 0; k < 512; k++) {
    const uint16_t gw = sh_grid[k];
    float ak = 0.0f, bk = 0.0f;
    for (int i = 0; i < 8; i++) {
      const float gv = IQ2_MAG[(gw >> (2u * uint(i))) & 3u];
      ak += ct[i] * gv;
      bk += c[i] * (gv * gv);
    }
    for (int ci = 0; ci < 9; ci++) {
      const float s = s0[ci];
      const float sc = (s * s) * bk - (2.0f * s) * ak;
      if (k == 0 || sc < bsc0[ci]) {
        bsc0[ci] = sc;
        ba0[ci] = ak;
        bb0[ci] = bk;
      }
    }
  }

  float s1[9], bsc1[9], ba1[9], bb1[9];
  for (int ci = 0; ci < 9; ci++) {
    const float asum = ba0[ci] + simd_shuffle_xor(ba0[ci], 1u);
    const float bsum = bb0[ci] + simd_shuffle_xor(bb0[ci], 1u);
    s1[ci] = asum / metal::fmax(bsum, 1e-30f);
  }
  for (int k = 0; k < 512; k++) {
    const uint16_t gw = sh_grid[k];
    float ak = 0.0f, bk = 0.0f;
    for (int i = 0; i < 8; i++) {
      const float gv = IQ2_MAG[(gw >> (2u * uint(i))) & 3u];
      ak += ct[i] * gv;
      bk += c[i] * (gv * gv);
    }
    for (int ci = 0; ci < 9; ci++) {
      const float s = s1[ci];
      const float sc = (s * s) * bk - (2.0f * s) * ak;
      if (k == 0 || sc < bsc1[ci]) {
        bsc1[ci] = sc;
        ba1[ci] = ak;
        bb1[ci] = bk;
      }
    }
  }

  float best_err = 0.0f, best_sigma = 0.0f;
  for (int ci = 0; ci < 9; ci++) {
    const float asum = ba1[ci] + simd_shuffle_xor(ba1[ci], 1u);
    const float bsum = bb1[ci] + simd_shuffle_xor(bb1[ci], 1u);
    const float s = s1[ci];
    const float err = (s * s) * bsum - (2.0f * s) * asum;
    if (ci == 0 || err < best_err) {
      best_err = err;
      best_sigma = s;
    }
  }

  const float sigma = metal::fmax(best_sigma, 0.0f);
  const float max_sigma = simd_max(sigma);
  const half dh = half((max_sigma * 4.0f) / 15.5f);
  const float d = float(dh);
  if (lane == 0u) d_out[blk] = dh;
  const uint lu = (d > 0.0f) ? uint(metal::clamp(metal::rint((4.0f * sigma) / d - 0.5f), 0.0f, 15.0f)) : 0u;
  const float quantized = d * (0.5f + float(lu)) * 0.25f;

  uint best_k = 0u;
  float best_qsc = 0.0f;
  const float q2 = quantized * quantized;
  const float q_two = 2.0f * quantized;
  for (int k = 0; k < 512; k++) {
    const uint16_t gw = sh_grid[k];
    float ak = 0.0f, bk = 0.0f;
    for (int i = 0; i < 8; i++) {
      const float gv = IQ2_MAG[(gw >> (2u * uint(i))) & 3u];
      ak += ct[i] * gv;
      bk += c[i] * (gv * gv);
    }
    const float sc = q2 * bk - q_two * ak;
    if (k == 0 || sc < best_qsc) {
      best_qsc = sc;
      best_k = uint(k);
    }
  }

  device uint8_t* out_b = body + blk * 72u;
  const uint u16 = best_k | ((neg_mask & 0x7fu) << 9);
  out_b[2u * lane] = uint8_t(u16 & 0xffu);
  out_b[2u * lane + 1u] = uint8_t(u16 >> 8);
  const uint lu_hi = simd_shuffle_xor(lu, 2u);
  if ((lane & 3u) == 0u) out_b[64u + (lane >> 2)] = uint8_t(lu | (lu_hi << 4));
"""

_CUDA_HEADER = r"""
#include <cstdint>
__device__ __forceinline__ float qz_ggml_wsum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float qz_ggml_wmax(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float qz_clamp(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}
static constexpr float QZ_IQ2_MAG[4] = {8.0f, 25.0f, 43.0f, 0.0f};
"""

_Q2_K_CUDA = r"""
  const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t blk = tid >> 5;
  const unsigned lane = unsigned(tid) & 31u;
  if (blk >= meta[0]) return;
  const unsigned sub = lane >> 1;

  const T* wr = w + blk * 256u + size_t(lane) * 8u;
  float g[8], c[8];
  float gmin_loc = INFINITY, gmax_loc = -INFINITY;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const float gv = static_cast<float>(wr[i]);
    g[i] = gv;
    c[i] = fabsf(gv);
    gmin_loc = fminf(gmin_loc, gv);
    gmax_loc = fmaxf(gmax_loc, gv);
  }
  const float gmin = fminf(gmin_loc, __shfl_xor_sync(0xffffffffu, gmin_loc, 1));
  const float gmax = fmaxf(gmax_loc, __shfl_xor_sync(0xffffffffu, gmax_loc, 1));
  float base = fminf(gmin, 0.0f);
  float scale = fmaxf((gmax - base) / 3.0f, 1e-30f);

  for (unsigned it = 0u; it < meta[1]; it++) {
    float ls0 = 0.0f, ls1 = 0.0f, ls2 = 0.0f, lt0 = 0.0f, lt1 = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const float qv = qz_clamp(rintf((g[i] - base) / scale), 0.0f, 3.0f);
      const float cq = c[i] * qv;
      ls0 += c[i];
      ls1 += cq;
      ls2 += cq * qv;
      lt0 += c[i] * g[i];
      lt1 += cq * g[i];
    }
    const float s0 = ls0 + __shfl_xor_sync(0xffffffffu, ls0, 1);
    const float s1 = ls1 + __shfl_xor_sync(0xffffffffu, ls1, 1);
    const float s2 = ls2 + __shfl_xor_sync(0xffffffffu, ls2, 1);
    const float t0 = lt0 + __shfl_xor_sync(0xffffffffu, lt0, 1);
    const float t1 = lt1 + __shfl_xor_sync(0xffffffffu, lt1, 1);
    const float det = s0 * s2 - s1 * s1;
    const bool ok = det > 1e-30f * fmaxf(s0 * s2, 1e-30f);
    float s = ok ? (s0 * t1 - s1 * t0) / det : scale;
    float b = ok ? (t0 - s * s1) / fmaxf(s0, 1e-30f) : base;
    if (b > 0.0f) {
      s = t1 / fmaxf(s2, 1e-30f);
      b = 0.0f;
    }
    if (s > 0.0f) scale = s;
    base = b;
  }

  const float max_scale = qz_ggml_wmax(scale);
  const float max_neg_base = fmaxf(qz_ggml_wmax(-base), 0.0f);
  const __half dh = __float2half(max_scale / 15.0f);
  const __half dminh = __float2half(max_neg_base / 15.0f);
  const float d = __half2float(dh);
  const float dmin = __half2float(dminh);
  if (lane == 0u) {
    d_out[blk] = dh;
    dmin_out[blk] = dminh;
  }

  const unsigned sc_u = (d > 0.0f) ? static_cast<unsigned>(qz_clamp(rintf(scale / d), 0.0f, 15.0f)) : 0u;
  const unsigned m_u = (dmin > 0.0f) ? static_cast<unsigned>(qz_clamp(rintf(-base / dmin), 0.0f, 15.0f)) : 0u;
  const float step = d * float(sc_u);
  const float low = dmin * float(m_u);

  uint8_t* out_b = body + blk * 80u;
  if ((lane & 1u) == 0u) out_b[sub] = uint8_t(sc_u | (m_u << 4));

  const unsigned h = lane >> 4;
  const unsigned sh = 2u * ((lane >> 2) & 3u);
  const unsigned lbase = (lane & 3u) * 8u;
  unsigned w0 = 0u, w1 = 0u;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const unsigned qv = (step > 0.0f) ? static_cast<unsigned>(qz_clamp(rintf((g[i] + low) / step), 0.0f, 3.0f)) : 0u;
    const unsigned p = qv << (sh + 8u * (unsigned(i) & 3u));
    if (i < 4) w0 |= p;
    else w1 |= p;
  }
  w0 |= __shfl_xor_sync(0xffffffffu, w0, 4);
  w0 |= __shfl_xor_sync(0xffffffffu, w0, 8);
  w1 |= __shfl_xor_sync(0xffffffffu, w1, 4);
  w1 |= __shfl_xor_sync(0xffffffffu, w1, 8);
  if ((lane & 12u) == 0u) {
    uint8_t* dst = out_b + 16u + 32u * h + lbase;
    #pragma unroll
    for (unsigned k = 0u; k < 4u; k++) {
      dst[k] = uint8_t((w0 >> (8u * k)) & 0xffu);
      dst[4u + k] = uint8_t((w1 >> (8u * k)) & 0xffu);
    }
  }
"""

_Q3_K_CUDA = r"""
  const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t blk = tid >> 5;
  const unsigned lane = unsigned(tid) & 31u;
  if (blk >= meta[0]) return;

  const T* wr = w + blk * 256u + size_t(lane) * 8u;
  float g[8], c[8];
  float best_abs = -1.0f, best_g = 0.0f;
  unsigned best_idx = lane * 8u;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const float gv = static_cast<float>(wr[i]);
    g[i] = gv;
    c[i] = gv * gv;
    const float av = fabsf(gv);
    if (av > best_abs) {
      best_abs = av;
      best_g = gv;
      best_idx = lane * 8u + unsigned(i);
    }
  }
  const float other_abs = __shfl_xor_sync(0xffffffffu, best_abs, 1);
  const float other_g = __shfl_xor_sync(0xffffffffu, best_g, 1);
  const unsigned other_idx = __shfl_xor_sync(0xffffffffu, best_idx, 1);
  if (other_abs > best_abs || (other_abs == best_abs && other_idx < best_idx)) best_g = other_g;
  const float top = (best_g == 0.0f) ? 1e-30f : best_g;

  float at_lo = top / -4.0f;
  for (unsigned it = 0u; it < meta[1]; it++) {
    float l2 = 0.0f, lt = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const float qv = qz_clamp(rintf(g[i] / at_lo), -4.0f, 3.0f);
      const float cq = c[i] * qv;
      l2 += cq * qv;
      lt += cq * g[i];
    }
    const float s2 = l2 + __shfl_xor_sync(0xffffffffu, l2, 1);
    const float t1 = lt + __shfl_xor_sync(0xffffffffu, lt, 1);
    if (s2 > 0.0f) at_lo = t1 / s2;
  }
  float le_lo = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const float qv = qz_clamp(rintf(g[i] / at_lo), -4.0f, 3.0f);
    const float err = g[i] - at_lo * qv;
    le_lo += c[i] * err * err;
  }
  const float low_err = le_lo + __shfl_xor_sync(0xffffffffu, le_lo, 1);

  float at_hi = top / 3.0f;
  for (unsigned it = 0u; it < meta[1]; it++) {
    float l2 = 0.0f, lt = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const float qv = qz_clamp(rintf(g[i] / at_hi), -4.0f, 3.0f);
      const float cq = c[i] * qv;
      l2 += cq * qv;
      lt += cq * g[i];
    }
    const float s2 = l2 + __shfl_xor_sync(0xffffffffu, l2, 1);
    const float t1 = lt + __shfl_xor_sync(0xffffffffu, lt, 1);
    if (s2 > 0.0f) at_hi = t1 / s2;
  }
  float le_hi = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const float qv = qz_clamp(rintf(g[i] / at_hi), -4.0f, 3.0f);
    const float err = g[i] - at_hi * qv;
    le_hi += c[i] * err * err;
  }
  const float high_err = le_hi + __shfl_xor_sync(0xffffffffu, le_hi, 1);

  const float scale = (high_err < low_err) ? at_hi : at_lo;

  unsigned bsub = lane >> 1;
  float babs = fabsf(scale);
  float bval = scale;
  for (int off = 16; off >= 2; off >>= 1) {
    const float oa = __shfl_xor_sync(0xffffffffu, babs, off);
    const float ov = __shfl_xor_sync(0xffffffffu, bval, off);
    const unsigned os = __shfl_xor_sync(0xffffffffu, bsub, off);
    if (oa > babs || (oa == babs && os < bsub)) {
      babs = oa;
      bval = ov;
      bsub = os;
    }
  }

  const __half dh = __float2half(-bval / 32.0f);
  const float d = __half2float(dh);
  if (lane == 0u) d_out[blk] = dh;

  const int sc_i = (d != 0.0f) ? int(qz_clamp(rintf(scale / d), -32.0f, 31.0f)) : 0;
  const unsigned u = unsigned(sc_i + 32);
  const float step = d * float(sc_i);
  int qi[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    qi[i] = (step != 0.0f) ? int(qz_clamp(rintf(g[i] / step), -4.0f, 3.0f)) : 0;
  }

  uint8_t* out_b = body + blk * 108u;

  const unsigned hb = lane >> 2;
  unsigned h0 = 0u, h1 = 0u;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    if (qi[i] >= 0) {
      const unsigned bit = 1u << (hb + 8u * (unsigned(i) & 3u));
      if (i < 4) h0 |= bit;
      else h1 |= bit;
    }
  }
  h0 |= __shfl_xor_sync(0xffffffffu, h0, 4);
  h0 |= __shfl_xor_sync(0xffffffffu, h0, 8);
  h0 |= __shfl_xor_sync(0xffffffffu, h0, 16);
  h1 |= __shfl_xor_sync(0xffffffffu, h1, 4);
  h1 |= __shfl_xor_sync(0xffffffffu, h1, 8);
  h1 |= __shfl_xor_sync(0xffffffffu, h1, 16);
  if (lane < 4u) {
    uint8_t* hd = out_b + lane * 8u;
    #pragma unroll
    for (unsigned k = 0u; k < 4u; k++) {
      hd[k] = uint8_t((h0 >> (8u * k)) & 0xffu);
      hd[4u + k] = uint8_t((h1 >> (8u * k)) & 0xffu);
    }
  }

  const unsigned h = lane >> 4;
  const unsigned sh = 2u * ((lane >> 2) & 3u);
  const unsigned lbase = (lane & 3u) * 8u;
  unsigned c0 = 0u, c1 = 0u;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const unsigned qlow = unsigned(qi[i] >= 0 ? qi[i] : qi[i] + 4);
    const unsigned p = qlow << (sh + 8u * (unsigned(i) & 3u));
    if (i < 4) c0 |= p;
    else c1 |= p;
  }
  c0 |= __shfl_xor_sync(0xffffffffu, c0, 4);
  c0 |= __shfl_xor_sync(0xffffffffu, c0, 8);
  c1 |= __shfl_xor_sync(0xffffffffu, c1, 4);
  c1 |= __shfl_xor_sync(0xffffffffu, c1, 8);
  if ((lane & 12u) == 0u) {
    uint8_t* cd = out_b + 32u + 32u * h + lbase;
    #pragma unroll
    for (unsigned k = 0u; k < 4u; k++) {
      cd[k] = uint8_t((c0 >> (8u * k)) & 0xffu);
      cd[4u + k] = uint8_t((c1 >> (8u * k)) & 0xffu);
    }
  }

  const unsigned u_hi = __shfl_xor_sync(0xffffffffu, u, 16);
  if ((lane & 1u) == 0u && lane < 16u) out_b[96u + (lane >> 1)] = uint8_t((u & 15u) | ((u_hi & 15u) << 4));

  unsigned t2 = ((u >> 4) & 3u) << (2u * (lane >> 3));
  t2 |= __shfl_xor_sync(0xffffffffu, t2, 8);
  t2 |= __shfl_xor_sync(0xffffffffu, t2, 16);
  if ((lane & 1u) == 0u && lane < 8u) out_b[104u + (lane >> 1)] = uint8_t(t2);
"""

_IQ2_XS_CUDA = r"""
  extern __shared__ uint16_t sh_grid[];
  const unsigned ltid = threadIdx.x;
  sh_grid[ltid] = grid_u16[ltid];
  sh_grid[ltid + 256u] = grid_u16[ltid + 256u];
  __syncthreads();

  const size_t tid = size_t(blockIdx.x) * blockDim.x + ltid;
  const size_t blk = tid >> 5;
  const unsigned lane = unsigned(tid) & 31u;
  if (blk >= meta[0]) return;

  const T* wr = w + blk * 256u + size_t(lane) * 8u;
  float x[8];
  float lsq = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const float xi = static_cast<float>(wr[i]);
    x[i] = xi;
    lsq += xi * xi;
  }
  const float sigma2 = qz_ggml_wsum(lsq) * (1.0f / 128.0f);

  float c[8], t[8];
  unsigned neg_mask = 0u;
  float max_abs = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const float xi = x[i];
    const float ai = fabsf(xi);
    c[i] = sqrtf(sigma2 + xi * xi);
    t[i] = ai;
    if (xi < 0.0f) neg_mask |= (1u << unsigned(i));
    max_abs = fmaxf(max_abs, ai);
  }
  if ((__popc(neg_mask) & 1) != 0) {
    int best_i = 0;
    float best_loss = c[0] * t[0] * t[0];
    #pragma unroll
    for (int i = 1; i < 8; i++) {
      const float loss = c[i] * t[i] * t[i];
      if (loss < best_loss) {
        best_loss = loss;
        best_i = i;
      }
    }
    t[best_i] = -t[best_i];
    neg_mask ^= (1u << unsigned(best_i));
  }
  float ct[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) ct[i] = c[i] * t[i];

  const float sub_max = fmaxf(max_abs, __shfl_xor_sync(0xffffffffu, max_abs, 1));
  const float top = fmaxf(sub_max, 1e-30f) / 43.0f;

  static constexpr float CANDS[9] = {0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f};
  float s0[9], bsc0[9], ba0[9], bb0[9];
  #pragma unroll
  for (int ci = 0; ci < 9; ci++) s0[ci] = top * CANDS[ci];
  for (int k = 0; k < 512; k++) {
    const uint16_t gw = sh_grid[k];
    float ak = 0.0f, bk = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const float gv = QZ_IQ2_MAG[(gw >> (2u * unsigned(i))) & 3u];
      ak += ct[i] * gv;
      bk += c[i] * (gv * gv);
    }
    #pragma unroll
    for (int ci = 0; ci < 9; ci++) {
      const float s = s0[ci];
      const float sc = (s * s) * bk - (2.0f * s) * ak;
      if (k == 0 || sc < bsc0[ci]) {
        bsc0[ci] = sc;
        ba0[ci] = ak;
        bb0[ci] = bk;
      }
    }
  }

  float s1[9], bsc1[9], ba1[9], bb1[9];
  #pragma unroll
  for (int ci = 0; ci < 9; ci++) {
    const float asum = ba0[ci] + __shfl_xor_sync(0xffffffffu, ba0[ci], 1);
    const float bsum = bb0[ci] + __shfl_xor_sync(0xffffffffu, bb0[ci], 1);
    s1[ci] = asum / fmaxf(bsum, 1e-30f);
  }
  for (int k = 0; k < 512; k++) {
    const uint16_t gw = sh_grid[k];
    float ak = 0.0f, bk = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const float gv = QZ_IQ2_MAG[(gw >> (2u * unsigned(i))) & 3u];
      ak += ct[i] * gv;
      bk += c[i] * (gv * gv);
    }
    #pragma unroll
    for (int ci = 0; ci < 9; ci++) {
      const float s = s1[ci];
      const float sc = (s * s) * bk - (2.0f * s) * ak;
      if (k == 0 || sc < bsc1[ci]) {
        bsc1[ci] = sc;
        ba1[ci] = ak;
        bb1[ci] = bk;
      }
    }
  }

  float best_err = 0.0f, best_sigma = 0.0f;
  #pragma unroll
  for (int ci = 0; ci < 9; ci++) {
    const float asum = ba1[ci] + __shfl_xor_sync(0xffffffffu, ba1[ci], 1);
    const float bsum = bb1[ci] + __shfl_xor_sync(0xffffffffu, bb1[ci], 1);
    const float s = s1[ci];
    const float err = (s * s) * bsum - (2.0f * s) * asum;
    if (ci == 0 || err < best_err) {
      best_err = err;
      best_sigma = s;
    }
  }

  const float sigma = fmaxf(best_sigma, 0.0f);
  const float max_sigma = qz_ggml_wmax(sigma);
  const __half dh = __float2half((max_sigma * 4.0f) / 15.5f);
  const float d = __half2float(dh);
  if (lane == 0u) d_out[blk] = dh;
  const unsigned lu = (d > 0.0f) ? static_cast<unsigned>(qz_clamp(rintf((4.0f * sigma) / d - 0.5f), 0.0f, 15.0f)) : 0u;
  const float quantized = d * (0.5f + float(lu)) * 0.25f;

  unsigned best_k = 0u;
  float best_qsc = 0.0f;
  const float q2 = quantized * quantized;
  const float q_two = 2.0f * quantized;
  for (int k = 0; k < 512; k++) {
    const uint16_t gw = sh_grid[k];
    float ak = 0.0f, bk = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const float gv = QZ_IQ2_MAG[(gw >> (2u * unsigned(i))) & 3u];
      ak += ct[i] * gv;
      bk += c[i] * (gv * gv);
    }
    const float sc = q2 * bk - q_two * ak;
    if (k == 0 || sc < best_qsc) {
      best_qsc = sc;
      best_k = unsigned(k);
    }
  }

  uint8_t* out_b = body + blk * 72u;
  const unsigned u16 = best_k | ((neg_mask & 0x7fu) << 9);
  out_b[2u * lane] = uint8_t(u16 & 0xffu);
  out_b[2u * lane + 1u] = uint8_t(u16 >> 8);
  const unsigned lu_hi = __shfl_xor_sync(0xffffffffu, lu, 2);
  if ((lane & 3u) == 0u) out_b[64u + (lane >> 2)] = uint8_t(lu | (lu_hi << 4));
"""


def _prep_gpu_rows(w):
    w = mx.array(w) if not isinstance(w, mx.array) else w
    if w.ndim < 2 or w.shape[-1] % QK:
        raise ValueError(f"ggml blocks need rows of a multiple of {QK} values, not {w.shape}")
    if w.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        w = w.astype(mx.float32)
    lead = tuple(w.shape[:-1])
    nb = w.shape[-1] // QK
    return mx.contiguous(w.reshape(-1, QK)), lead, nb


def _kernel(backend, mode):
    key = (backend, mode)
    k = _KERNELS.get(key)
    if k is not None:
        return k
    if backend == 'metal':
        if mode == 'q2_k':
            k = mx.fast.metal_kernel('ggml_q2_k_pack', ['w', 'meta'], ['body', 'd_out', 'dmin_out'], _Q2_K_METAL)
        elif mode == 'q3_k':
            k = mx.fast.metal_kernel('ggml_q3_k_pack', ['w', 'meta'], ['body', 'd_out'], _Q3_K_METAL)
        else:
            k = mx.fast.metal_kernel('ggml_iq2_xs_pack', ['w', 'grid_u16', 'meta'], ['body', 'd_out'],
                                     _IQ2_XS_METAL, header=_METAL_HEADER)
    else:
        if mode == 'q2_k':
            k = cuda_kernels.kernel('ggml_q2_k_pack_cuda', ['w', 'meta'], ['body', 'd_out', 'dmin_out'],
                                    _Q2_K_CUDA, header=_CUDA_HEADER)
        elif mode == 'q3_k':
            k = cuda_kernels.kernel('ggml_q3_k_pack_cuda', ['w', 'meta'], ['body', 'd_out'],
                                    _Q3_K_CUDA, header=_CUDA_HEADER)
        else:
            k = cuda_kernels.kernel('ggml_iq2_xs_pack_cuda', ['w', 'grid_u16', 'meta'], ['body', 'd_out'],
                                    _IQ2_XS_CUDA, header=_CUDA_HEADER, shared_memory=1024)
    _KERNELS[key] = k
    return k


def pack_arrays(w, mode, iterations=ITERATIONS):
    """Pack `w [..., cols]` into `(body, d, dmin)` without computing reconstructed `values`."""
    if mode not in BODY:
        raise ValueError(f'unknown ggml format {mode!r}')
    backend = _gpu_backend()
    if backend is None:
        body, d, dmin, _ = _OPS_FORMATS[mode](w, iterations) if mode in ('q2_k', 'q3_k') else _OPS_FORMATS[mode](w)
        return body, d, dmin
    w_flat, lead, nb = _prep_gpu_rows(w)
    n = w_flat.shape[0]
    body_bytes = BODY[mode]
    if n == 0:
        body = mx.zeros((*lead, nb, body_bytes), dtype=mx.uint8)
        d = mx.zeros((*lead, nb), dtype=mx.float16)
        dmin = mx.zeros((*lead, nb), dtype=mx.float16) if mode == 'q2_k' else None
        return body, d, dmin
    grid = (((n + 7) // 8) * 256, 1, 1)
    tg = (256, 1, 1)
    kern = _kernel(backend, mode)
    if mode == 'q2_k':
        meta = mx.array([n, int(iterations)], dtype=mx.uint32)
        body, d, dmin = kern(
            inputs=[w_flat, meta],
            template=[('T', w_flat.dtype)],
            grid=grid, threadgroup=tg,
            output_shapes=[(n, body_bytes), (n,), (n,)],
            output_dtypes=[mx.uint8, mx.float16, mx.float16],
        )
        return body.reshape(*lead, nb, body_bytes), d.reshape(*lead, nb), dmin.reshape(*lead, nb)
    if mode == 'q3_k':
        meta = mx.array([n, int(iterations)], dtype=mx.uint32)
        body, d = kern(
            inputs=[w_flat, meta],
            template=[('T', w_flat.dtype)],
            grid=grid, threadgroup=tg,
            output_shapes=[(n, body_bytes), (n,)],
            output_dtypes=[mx.uint8, mx.float16],
        )
        return body.reshape(*lead, nb, body_bytes), d.reshape(*lead, nb), None
    grid_u16 = compact_grids()[IQ2_XS]
    meta = mx.array([n], dtype=mx.uint32)
    body, d = kern(
        inputs=[w_flat, grid_u16, meta],
        template=[('T', w_flat.dtype)],
        grid=grid, threadgroup=tg,
        output_shapes=[(n, body_bytes), (n,)],
        output_dtypes=[mx.uint8, mx.float16],
    )
    return body.reshape(*lead, nb, body_bytes), d.reshape(*lead, nb), None


def q2_k_pack(w, iterations=ITERATIONS):
    """`w [..., cols]` packed as `Q2_K`: `(body uint8 [..., blocks, 80], d float16 [..., blocks],
    dmin float16 [..., blocks], values float32 [..., cols])`."""
    if _gpu_backend() is None:
        return _q2_k_pack_ops(w, iterations)
    body, d, dmin = pack_arrays(w, 'q2_k', iterations)
    return body, d, dmin, dequantize(body, d, dmin, 'q2_k')


def q3_k_pack(w, iterations=ITERATIONS):
    """`w [..., cols]` packed as `Q3_K`: `(body uint8 [..., blocks, 108], d float16 [..., blocks],
    None, values float32 [..., cols])`."""
    if _gpu_backend() is None:
        return _q3_k_pack_ops(w, iterations)
    body, d, _ = pack_arrays(w, 'q3_k', iterations)
    return body, d, None, dequantize(body, d, None, 'q3_k')


def iq2_xs_pack(w):
    """`w [..., cols]` packed as `IQ2_XS`: `(body uint8 [..., blocks, 72], d float16 [..., blocks],
    None, values float32 [..., cols])`, where `body` holds the 64 bytes of `(grid_idx | sign7 << 9)`
    little-endian uint16 words followed by the 8 bytes of 4-bit sub-block scales."""
    if _gpu_backend() is None:
        return _iq2_xs_pack_ops(w)
    body, d, _ = pack_arrays(w, 'iq2_xs')
    return body, d, None, dequantize(body, d, None, 'iq2_xs')


_OPS_FORMATS = {'q2_k': _q2_k_pack_ops, 'q3_k': _q3_k_pack_ops, 'iq2_xs': _iq2_xs_pack_ops}
FORMATS = {'q2_k': q2_k_pack, 'q3_k': q3_k_pack, 'iq2_xs': iq2_xs_pack}
FORMAT_BITS = {'q2_k': 2, 'q3_k': 3, 'iq2_xs': 2}
BODY = {'q2_k': 80, 'q3_k': 108, 'iq2_xs': 72}


def blocks(body, d, dmin=None, mode=None):
    """Assemble packed `(body, d, dmin)` into raw ggml block bytes `uint8 [n, BLOCK_BYTES]`."""
    body_np = np.array(body)
    d_bytes = np.array(d.astype(mx.float16) if isinstance(d, mx.array) else d).reshape(-1, 1).astype(np.float16).view(np.uint8)
    body_flat = body_np.reshape(-1, body_np.shape[-1])
    if mode == 'iq2_xs' or (mode is None and body_flat.shape[-1] == 72 and dmin is None):
        return np.concatenate([d_bytes, body_flat], axis=-1)
    parts = [body_flat, d_bytes]
    if dmin is not None:
        parts.append(np.array(dmin).reshape(-1, 1).astype(np.float16).view(np.uint8))
    return np.concatenate(parts, axis=-1)


def q2_k(w, iterations=ITERATIONS):
    """`w [rows, cols]` as raw ggml `Q2_K` blocks: `(blocks uint8 [rows * cols / 256, 84], values float32 [rows, cols])`."""
    body, d, dmin, values = q2_k_pack(w, iterations)
    return blocks(body, d, dmin, mode='q2_k'), values


def q3_k(w, iterations=ITERATIONS):
    """`w [rows, cols]` as raw ggml `Q3_K` blocks: `(blocks uint8 [rows * cols / 256, 110], values float32 [rows, cols])`."""
    body, d, _, values = q3_k_pack(w, iterations)
    return blocks(body, d, mode='q3_k'), values


def iq2_xs(w):
    """`w [rows, cols]` as raw ggml `IQ2_XS` blocks: `(blocks uint8 [rows * cols / 256, 74], values float32 [rows, cols])`."""
    body, d, _, values = iq2_xs_pack(w)
    return blocks(body, d, mode='iq2_xs'), values


QUANTIZERS = {'q2_k': q2_k, 'q3_k': q3_k, 'iq2_xs': iq2_xs}


def _shifted(q, shifts, mask):
    return (q[..., None, :] >> mx.array(shifts, mx.uint8)[:, None]) & mask


def dequantize(body, d, dmin, mode):
    """Unpack `(body [..., blocks, BODY[mode]], d [..., blocks], dmin)` to `float32 [..., blocks * 256]`."""
    lead = tuple(body.shape[:-2])
    nb = body.shape[-2]
    if mode == 'q2_k':
        s = body[..., 0:16]
        q = _shifted(body[..., 16:80].reshape(*lead, nb, 2, 32), (0, 2, 4, 6), 3).reshape(*lead, nb, 16, 16)
        scale = d.astype(mx.float32)[..., None] * (s & 15).astype(mx.float32)
        low = dmin.astype(mx.float32)[..., None] * (s >> 4).astype(mx.float32)
        out = scale[..., None] * q.astype(mx.float32) - low[..., None]
    elif mode == 'q3_k':
        s = body[..., 96:108]
        u = mx.concatenate([s[..., 0:8] & 15, s[..., 0:8] >> 4], axis=-1).astype(mx.int32) | (
            (_shifted(s[..., 8:12], (0, 2, 4, 6), 3).reshape(*lead, nb, 16).astype(mx.int32)) << 4
        )
        low = _shifted(body[..., 32:96].reshape(*lead, nb, 2, 32), (0, 2, 4, 6), 3).reshape(*lead, nb, 16, 16)
        high = _shifted(body[..., 0:32], tuple(range(8)), 1).reshape(*lead, nb, 16, 16)
        q = low.astype(mx.int32) - ((1 - high.astype(mx.int32)) << 2)
        scale = d.astype(mx.float32)[..., None] * (u - 32).astype(mx.float32)
        out = scale[..., None] * q.astype(mx.float32)
    elif mode == 'iq2_xs':
        t = tables()
        q = (
            body[..., 0:64:2].astype(mx.uint32) | (body[..., 1:64:2].astype(mx.uint32) << 8)
        ).reshape(*lead, nb, 8, 4)
        s = body[..., 64:72].astype(mx.float32)
        low, high = mx.remainder(s, 16), mx.floor(s / 16)
        df = d.astype(mx.float32)[..., None]
        db = mx.stack([df * (0.5 + low) * 0.25, df * (0.5 + high) * 0.25], axis=-1)
        halves = mx.take(db, mx.array([0, 0, 1, 1]), axis=-1)
        out = halves[..., None] * mx.take(t[IQ2_XS], q & 511, axis=0) * mx.take(t['ksigns'], q >> 9, axis=0)
    else:
        raise ValueError(f'unknown ggml format {mode!r}')
    return out.reshape(*lead, nb * QK)
