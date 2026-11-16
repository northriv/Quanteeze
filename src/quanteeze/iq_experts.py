# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Standard ggml K-quant and IQ codebook tables and block decoders (`dequantize_row_*` in MLX array
ops, exact in float32).  Codebooks come from `quanteeze.iq_grids`."""
import mlx.core as mx
import numpy as np

from . import iq_grids

QK = 256
F16, Q4_0, Q4_1, IQ1_M, BF16, MXFP4, Q2_0 = 1, 2, 3, 29, 30, 39, 42
IQ2_XXS, IQ2_XS, IQ1_S, IQ2_S = 16, 17, 19, 22
Q5_0, IQ3_XXS, IQ4_NL, IQ4_XS = 6, 18, 20, 23
Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 7, 8, 10, 11, 12, 13, 14
TYPES = {
    F16: 'F16', Q4_0: 'Q4_0', Q4_1: 'Q4_1',
    IQ2_XXS: 'IQ2_XXS', IQ2_XS: 'IQ2_XS', IQ1_S: 'IQ1_S', IQ1_M: 'IQ1_M', IQ2_S: 'IQ2_S',
    Q5_0: 'Q5_0', IQ3_XXS: 'IQ3_XXS', IQ4_NL: 'IQ4_NL', IQ4_XS: 'IQ4_XS',
    Q5_1: 'Q5_1', Q8_0: 'Q8_0', BF16: 'BF16', MXFP4: 'MXFP4', Q2_0: 'Q2_0',
    Q2_K: 'Q2_K', Q3_K: 'Q3_K', Q4_K: 'Q4_K', Q5_K: 'Q5_K', Q6_K: 'Q6_K',
}
BY_NAME = {name: kind for kind, name in TYPES.items()}
BLOCK = {
    F16: 64, Q4_0: 18, Q4_1: 20,
    IQ2_XXS: 66, IQ2_XS: 74, IQ1_S: 50, IQ1_M: 56, IQ2_S: 82,
    Q5_0: 22, IQ3_XXS: 98, IQ4_NL: 18, IQ4_XS: 136,
    Q5_1: 24, Q8_0: 34, BF16: 64, MXFP4: 17, Q2_0: 18,
    Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210,
}
VALUES = {F16: 32, Q4_0: 32, Q4_1: 32, Q5_0: 32, Q5_1: 32, IQ4_NL: 32, Q8_0: 32, BF16: 32, MXFP4: 32, Q2_0: 64}
IQ1S_DELTA = 0.125
MXFP4_MAG = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0)


def _grid(hexes, values):
    words = np.frombuffer(bytes.fromhex(''.join(hexes)), '>u2')
    return np.asarray(values, np.float32)[(words[:, None] >> (2 * np.arange(8))) & 3]


def _signs(n):
    """[n, 8]: -1 where bit j of the byte is set, else 1."""
    return np.where((np.arange(n)[:, None] >> np.arange(8)) & 1, -1.0, 1.0).astype(np.float32)


_KSIGNS = np.array([i | (bin(i).count('1') & 1) << 7 for i in range(128)])
_TABLES = {}
_COMPACT_GRIDS = {}


def tables():
    """The codebooks and sign tables as MLX arrays (made once)."""
    if not _TABLES:
        _TABLES.update({
            IQ2_XXS: mx.array(_grid(iq_grids.IQ2XXS, (8, 25, 43))),
            IQ2_XS: mx.array(_grid(iq_grids.IQ2XS, (8, 25, 43))),
            IQ2_S: mx.array(_grid(iq_grids.IQ2S, (8, 25, 43))),
            IQ1_S: mx.array(_grid(iq_grids.IQ1S, (-1, 0, 1))),
            'ksigns': mx.array(_signs(256)[_KSIGNS]),
            'signs': mx.array(_signs(256)),
            IQ3_XXS: mx.array(_grid3(iq_grids.IQ3XXS, iq_grids.IQ3XXS_MAP)),
            IQ4_NL: mx.array(np.asarray(iq_grids.IQ4NL, np.float32)),
            MXFP4: mx.array(np.asarray(MXFP4_MAG, np.float32)),
        })
    return _TABLES


def compact_grids():
    """Compact uint16 IQ codebook tables for Metal and CUDA kernels (made once)."""
    if not _COMPACT_GRIDS:
        on_cuda = not mx.metal.is_available() and mx.default_device() == mx.gpu

        def dev(arr):
            if on_cuda:
                arr = arr + mx.zeros_like(arr)
            mx.eval(arr)
            return arr

        def u16_be(hexes):
            return dev(mx.array(np.frombuffer(bytes.fromhex(''.join(hexes)), '>u2').astype(np.uint16)))

        def u16_le(hexes):
            return dev(mx.array(np.frombuffer(bytes.fromhex(''.join(hexes)), '<u2').astype(np.uint16)))

        iq1_grid = u16_be(iq_grids.IQ1S)
        _COMPACT_GRIDS.update({
            IQ2_XXS: u16_be(iq_grids.IQ2XXS),
            IQ2_XS: u16_be(iq_grids.IQ2XS),
            IQ2_S: u16_be(iq_grids.IQ2S),
            IQ1_S: iq1_grid,
            IQ1_M: iq1_grid,
            IQ3_XXS: u16_le(iq_grids.IQ3XXS),
        })
    return _COMPACT_GRIDS


def _grid3(hexes, values):
    raw = np.frombuffer(bytes.fromhex(''.join(hexes)), np.uint8)
    digits = np.stack([raw & 15, raw >> 4], axis=-1).reshape(256, 4)
    return np.asarray(values, np.float32)[digits]


def per_block(kind):
    """Values per block of ggml type `kind`."""
    return VALUES.get(kind, QK)


def _shifted(q, shifts, mask):
    return (q[..., :, None, :] >> mx.array(shifts, dtype=mx.uint8)[:, None]) & mask


def _u16(b, at, n):
    return b[:, at:at + 2 * n:2].astype(mx.uint32) | (b[:, at + 1:at + 2 * n:2].astype(mx.uint32) << 8)


def _f16(b, at):
    return _u16(b, at, 1).astype(mx.uint16).view(mx.float16).astype(mx.float32)


def _d(b):
    return _u16(b, 0, 1).astype(mx.uint16).view(mx.float16).astype(mx.float32)


def _scale_min(s):
    d, m, md = s[:, 0:4], s[:, 4:8], s[:, 8:12]
    sc = mx.concatenate([d & 63, (md & 15) | ((d >> 2) & 48)], axis=-1)
    mn = mx.concatenate([m & 63, (md >> 4) | ((m >> 2) & 48)], axis=-1)
    return sc.astype(mx.float32), mn.astype(mx.float32)


def _k_quant(b, kind, n):
    if kind == Q2_K:
        s = b[:, 0:16]
        dl = (_f16(b, 80) * (s & 15).astype(mx.float32))[..., None]
        ml = (_f16(b, 82) * (s >> 4).astype(mx.float32))[..., None]
        q = _shifted(b[:, 16:80].reshape(n, 2, 32), (0, 2, 4, 6), 3).reshape(n, 16, 16).astype(mx.float32)
        return (dl * q - ml).reshape(n, QK)
    if kind == Q3_K:
        s = b[:, 96:108]
        low = _shifted(s[:, None, 0:8], (0, 4), 255).reshape(n, 16)
        high = _shifted(s[:, None, 8:12], (0, 2, 4, 6), 255).reshape(n, 16)
        sc = ((low & 15) | ((high & 3) << 4)).astype(mx.int32) - 32
        dl = (_f16(b, 108) * sc.astype(mx.float32))[..., None]
        ql = _shifted(b[:, 32:96].reshape(n, 2, 32), (0, 2, 4, 6), 3).reshape(n, 16, 16)
        qh = _shifted(b[:, None, 0:32], tuple(range(8)), 1).reshape(n, 16, 16) ^ 1
        q = (ql.astype(mx.int32) - (qh.astype(mx.int32) << 2)).astype(mx.float32)
        return (dl * q).reshape(n, QK)
    if kind in (Q4_K, Q5_K):
        sc, mn = _scale_min(b[:, 4:16])
        dd, dm = (_f16(b, 0) * sc)[..., None], (_f16(b, 2) * mn)[..., None]
        codes = b[:, 16:144] if kind == Q4_K else b[:, 48:176]
        q = _shifted(codes.reshape(n, 4, 32), (0, 4), 15).reshape(n, 8, 32)
        if kind == Q5_K:
            q = q | (_shifted(b[:, None, 16:48], tuple(range(8)), 1).reshape(n, 8, 32) << 4)
        return (dd * q.astype(mx.float32) - dm).reshape(n, QK)
    dsc = (_f16(b, 208) * b[:, 192:208].view(mx.int8).astype(mx.float32))[..., None]
    ql = _shifted(b[:, 0:128].reshape(n, 2, 64), (0, 4), 15).reshape(n, 8, 32)
    qh = _shifted(b[:, 128:192].reshape(n, 2, 32), (0, 2, 4, 6), 3).reshape(n, 8, 32)
    q = ((ql | (qh << 4)).astype(mx.int32) - 32).reshape(n, 16, 16).astype(mx.float32)
    return (dsc * q).reshape(n, QK)


def _nibbles(q):
    return mx.concatenate([q & 15, q >> 4], axis=-1)


def _halves(b, at, d):
    s = b[:, at:at + 8].astype(mx.float32)
    low, high = mx.remainder(s, 16), mx.floor(s / 16)
    db = mx.stack([d * (0.5 + low) * 0.25, d * (0.5 + high) * 0.25], axis=-1)
    return mx.take(db, mx.array([0, 0, 1, 1]), axis=2)


DECODE_CHUNK = 1 << 18


def _decode_chunk(b, kind):
    t = tables()
    n = b.shape[0]
    d = _d(b)
    l4 = mx.arange(4, dtype=mx.uint32)
    if kind == IQ2_XXS:
        q = b[:, 2:66].reshape(n, 8, 8)
        aux = q[:, :, 4:].astype(mx.uint32)
        aux = aux[..., 0] | (aux[..., 1] << 8) | (aux[..., 2] << 16) | (aux[..., 3] << 24)
        db = d * (0.5 + (aux >> 28).astype(mx.float32)) * 0.25
        signs = mx.take(t['ksigns'], (aux[:, :, None] >> (7 * l4)) & 127, axis=0)
        y = db[:, :, None, None] * mx.take(t[IQ2_XXS], q[:, :, :4].astype(mx.uint32), axis=0) * signs
    elif kind == IQ2_XS:
        q = _u16(b, 2, 32).reshape(n, 8, 4)
        y = _halves(b, 66, d)[..., None] * mx.take(t[IQ2_XS], q & 511, axis=0) * mx.take(t['ksigns'], q >> 9, axis=0)
    elif kind == IQ2_S:
        low = b[:, 2:34].reshape(n, 8, 4).astype(mx.uint32)
        high = b[:, 66:74].astype(mx.uint32)[:, :, None]
        index = low | ((high << (8 - 2 * l4)) & 0x300)
        signs = mx.take(t['signs'], b[:, 34:66].reshape(n, 8, 4).astype(mx.uint32), axis=0)
        y = _halves(b, 74, d)[..., None] * mx.take(t[IQ2_S], index, axis=0) * signs
    elif kind == IQ1_S:
        qh = _u16(b, 34, 8)
        dl = d * (2 * ((qh >> 12) & 7) + 1).astype(mx.float32)
        delta = mx.where((qh & 0x8000) > 0, -IQ1S_DELTA, IQ1S_DELTA)
        index = b[:, 2:34].reshape(n, 8, 4).astype(mx.uint32) | (((qh[:, :, None] >> (3 * l4)) & 7) << 8)
        y = dl[:, :, None, None] * (mx.take(t[IQ1_S], index, axis=0) + delta[:, :, None, None])
    elif kind == IQ1_M:
        qs = b[:, 0:32].reshape(n, 8, 4).astype(mx.uint32)
        qh_raw = b[:, 32:48].astype(mx.uint32)
        qh_nib = mx.stack([qh_raw & 15, qh_raw >> 4], axis=-1).reshape(n, 8, 4)
        index = qs | ((qh_nib & 7) << 8)
        delta = mx.where((qh_nib & 8) > 0, -IQ1S_DELTA, IQ1S_DELTA)
        sc = _u16(b, 48, 4)
        d_bits = (
            ((sc[:, 0:1] >> 12) & 0x000f)
            | ((sc[:, 1:2] >> 8) & 0x00f0)
            | ((sc[:, 2:3] >> 4) & 0x0f00)
            | (sc[:, 3:4] & 0xf000)
        )
        d_m = d_bits.astype(mx.uint16).view(mx.float16).astype(mx.float32)
        sub_sc = ((sc[:, :, None] >> (3 * l4)) & 7).reshape(n, 8, 2)
        dl = d_m[:, :, None] * (2 * mx.take(sub_sc, mx.array([0, 0, 1, 1]), axis=2).astype(mx.float32) + 1)
        y = dl[..., None] * (mx.take(t[IQ1_S], index, axis=0) + delta[..., None])
    elif kind == IQ3_XXS:
        q = b[:, 2:66].reshape(n, 8, 4, 2).astype(mx.uint32)
        aux = b[:, 66:98].reshape(n, 8, 4).astype(mx.uint32)
        aux = aux[..., 0] | (aux[..., 1] << 8) | (aux[..., 2] << 16) | (aux[..., 3] << 24)
        db = d * (0.5 + (aux >> 28).astype(mx.float32)) * 0.5
        signs = mx.take(t['ksigns'], (aux[:, :, None] >> (7 * l4)) & 127, axis=0)
        grid = mx.take(t[IQ3_XXS], q, axis=0).reshape(n, 8, 4, 8)
        y = db[:, :, None, None] * grid * signs
    elif kind == IQ4_XS:
        high = _u16(b, 2, 1)
        low = b[:, 4:8].astype(mx.uint32)
        low = mx.stack([low & 15, low >> 4], axis=-1).reshape(n, 8)
        ib = mx.arange(8, dtype=mx.uint32)
        ls = low | (((high >> (2 * ib)) & 3) << 4)
        dl = d * (ls.astype(mx.float32) - 32)
        codes = _nibbles(b[:, 8:136].reshape(n, 8, 16))
        y = dl[:, :, None] * mx.take(t[IQ4_NL], codes.astype(mx.uint32), axis=0)
    elif kind == IQ4_NL:
        y = d * mx.take(t[IQ4_NL], _nibbles(b[:, 2:18]).astype(mx.uint32), axis=0)
    elif kind == F16:
        y = _u16(b, 0, 32).astype(mx.uint16).view(mx.float16).astype(mx.float32)
    elif kind == BF16:
        y = _u16(b, 0, 32).astype(mx.uint16).view(mx.bfloat16).astype(mx.float32)
    elif kind == Q4_0:
        y = d * (_nibbles(b[:, 2:18]).astype(mx.int32) - 8).astype(mx.float32)
    elif kind == Q4_1:
        y = d * _nibbles(b[:, 4:20]).astype(mx.float32) + _f16(b, 2)
    elif kind == Q2_0:
        q = ((b[:, 2:18, None] >> mx.array([0, 2, 4, 6], dtype=mx.uint8)) & 3).reshape(n, 64)
        y = d * (q.astype(mx.int32) - 1).astype(mx.float32)
    elif kind == Q5_0:
        qh = b[:, 2:6].astype(mx.uint32)
        qh = qh[:, 0] | (qh[:, 1] << 8) | (qh[:, 2] << 16) | (qh[:, 3] << 24)
        five = (qh[:, None] >> mx.arange(32, dtype=mx.uint32)) & 1
        q = _nibbles(b[:, 6:22]).astype(mx.int32) | (five.astype(mx.int32) << 4)
        y = d * (q - 16).astype(mx.float32)
    elif kind == Q5_1:
        qh = b[:, 4:8].astype(mx.uint32)
        qh = qh[:, 0] | (qh[:, 1] << 8) | (qh[:, 2] << 16) | (qh[:, 3] << 24)
        five = (qh[:, None] >> mx.arange(32, dtype=mx.uint32)) & 1
        q = (_nibbles(b[:, 8:24]).astype(mx.uint32) | (five << 4)).astype(mx.float32)
        y = d * q + _f16(b, 2)
    elif kind == Q8_0:
        y = d * b[:, 2:34].view(mx.int8).astype(mx.float32)
    elif kind == MXFP4:
        e = b[:, 0:1].astype(mx.uint32)
        e_bits = mx.where(e < 2, mx.array(0x00200000, dtype=mx.uint32) << e, (e - 1) << 23)
        y = e_bits.view(mx.float32) * mx.take(t[MXFP4], _nibbles(b[:, 1:17]).astype(mx.uint32), axis=0)
    elif kind in (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K):
        y = _k_quant(b, kind, n)
    else:
        raise ValueError(f'ggml type {kind}: not one of {sorted(TYPES.values())}')
    return y.reshape(n, per_block(kind))


def decode(blocks, kind, dtype=mx.float32):
    """ggml's values of `blocks` (`uint8 [n, BLOCK[kind]]`, MLX or numpy) of type `kind`:
    `dtype [n, per_block(kind)]` (default `float32`), matching ggml's `dequantize_row_<type>`."""
    b = mx.array(blocks) if not isinstance(blocks, mx.array) else blocks
    n = b.shape[0]
    if n <= DECODE_CHUNK:
        out = _decode_chunk(b, kind)
        return out if dtype == mx.float32 else out.astype(dtype)
    parts = []
    for start in range(0, n, DECODE_CHUNK):
        part = _decode_chunk(b[start:start + DECODE_CHUNK], kind)
        if dtype != mx.float32:
            part = part.astype(dtype)
        mx.eval(part)
        parts.append(part)
    return mx.concatenate(parts, axis=0)
