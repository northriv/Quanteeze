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

from .iq_experts import BLOCK, IQ2_XS, Q2_K, Q3_K, QK, tables

BITS = {
    'q2_k': BLOCK[Q2_K] * 8 / QK,
    'q3_k': BLOCK[Q3_K] * 8 / QK,
    'iq2_xs': BLOCK[IQ2_XS] * 8 / QK,
}                                                   # 2.625, 3.4375, 2.3125
ITERATIONS = 8
CANDIDATES = np.linspace(0.6, 1.4, 9)
ROWS = 64


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


def q2_k_pack(w, iterations=ITERATIONS):
    """`w [..., cols]` packed as `Q2_K`: `(body uint8 [..., blocks, 80], d float16 [..., blocks],
    dmin float16 [..., blocks], values float32 [..., cols])`."""
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


def q3_k_pack(w, iterations=ITERATIONS):
    """`w [..., cols]` packed as `Q3_K`: `(body uint8 [..., blocks, 108], d float16 [..., blocks],
    None, values float32 [..., cols])`."""
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


def iq2_xs_pack(w):
    """`w [..., cols]` packed as `IQ2_XS`: `(body uint8 [..., blocks, 72], d float16 [..., blocks],
    None, values float32 [..., cols])`, where `body` holds the 64 bytes of `(grid_idx | sign7 << 9)`
    little-endian uint16 words followed by the 8 bytes of 4-bit sub-block scales."""
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
