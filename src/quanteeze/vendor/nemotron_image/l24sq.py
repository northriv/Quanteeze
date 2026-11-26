# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this file is governed by the NVIDIA License in LICENSE beside it: non-commercially only (for research or
# evaluation purposes).  Derived from Nemotron-Labs-Image-Tokenizer/modeling_l24sq_vit.py of
# nvidia/Nemotron-Labs-Image-30B-A3B (commit d7c2ba6a59a384080f0bdd5b1e846da03804cdd1); modified: its encoder,
# quantizer and decoder written again in MLX in place of PyTorch (Quanteeze, 2026-11-26).  See NOTICE.md.
"""Nemotron-Labs-Image-Tokenizer (l24sq_vit) in MLX: the encoder, the Leech-lattice quantizer and the
decoder, written from the repository's modeling_l24sq_vit.py (no torch, no remote code).

    python -m quanteeze.vendor.nemotron_image.l24sq recon <tokenizer dir> <image> <out.png>      # encode, decode, PSNR
    python -m quanteeze.vendor.nemotron_image.l24sq decode <tokenizer dir> <codes.npz> <out.png> # codes, grid (gh, gw)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

import mlx.core as mx

EPS = 1e-6          # RMSNorm(dim) in the reference: eps default 1e-6


def rope(gh, gw, head_dim, theta):
    """cos, sin [gh*gw, head_dim]: the first half rotates by the row, the second by the column."""
    q = head_dim // 4
    inv = 1.0 / (theta ** (np.arange(q, dtype=np.float32) * 2 / (head_dim // 2)))
    h = np.repeat(np.arange(gh, dtype=np.float32), gw)
    w = np.tile(np.arange(gw, dtype=np.float32), gh)
    ah, aw = np.outer(h, inv), np.outer(w, inv)
    eh, ew = np.concatenate([ah, ah], -1), np.concatenate([aw, aw], -1)
    cos = np.concatenate([np.cos(eh), np.cos(ew)], -1)
    sin = np.concatenate([np.sin(eh), np.sin(ew)], -1)
    return mx.array(cos), mx.array(sin)


def rotate_half(x):
    a, b = mx.split(x, 2, axis=-1)
    return mx.concatenate([-b, a], axis=-1)


def rotate_half_2d(x):
    a, b = mx.split(x, 2, axis=-1)
    return mx.concatenate([rotate_half(a), rotate_half(b)], axis=-1)


class ViT:
    def __init__(self, w, prefix, cfg, dtype):
        self.w = {k[len(prefix):]: v.astype(dtype) for k, v in w.items() if k.startswith(prefix)}
        self.layers, self.heads, self.width = cfg['layers'], cfg['heads'], cfg['width']
        self.head_dim = self.width // self.heads
        self.theta = cfg['vit_rope_theta']
        self.dtype = dtype

    def lin(self, x, name, bias=False):
        y = x @ self.w[name + '.weight'].T
        return y + self.w[name + '.bias'] if bias else y

    def blocks(self, x, gh, gw):
        cos, sin = rope(gh, gw, self.head_dim, self.theta)
        cos, sin = cos[:, None, :].astype(self.dtype), sin[:, None, :].astype(self.dtype)
        n = x.shape[0]
        for i in range(self.layers):
            p = f'blocks.{i}.'
            h = mx.fast.rms_norm(x, self.w[p + 'norm1.weight'], EPS)
            q = self.lin(h, p + 'attn.q_proj').reshape(n, self.heads, self.head_dim)
            k = self.lin(h, p + 'attn.k_proj').reshape(n, self.heads, self.head_dim)
            v = self.lin(h, p + 'attn.v_proj').reshape(n, self.heads, self.head_dim)
            q = q * cos + rotate_half_2d(q) * sin
            k = k * cos + rotate_half_2d(k) * sin
            o = mx.fast.scaled_dot_product_attention(
                q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None],
                scale=1.0 / math.sqrt(self.head_dim))
            o = o[0].transpose(1, 0, 2).reshape(n, self.width)
            x = x + self.lin(o, p + 'attn.out_proj')
            h = mx.fast.rms_norm(x, self.w[p + 'norm2.weight'], EPS)
            m = mx.sigmoid(g := self.lin(h, p + 'mlp.gate_proj')) * g * self.lin(h, p + 'mlp.up_proj')
            x = x + self.lin(m, p + 'mlp.down_proj')
        return mx.fast.rms_norm(x, self.w['norm.weight'], EPS)


def normalize(x):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class Tokenizer:
    def __init__(self, folder, dtype=mx.float32):
        folder = Path(folder)
        self.cfg = json.loads((folder / 'config.json').read_text())
        w = mx.load(str(folder / 'model.safetensors'))
        self.p = self.cfg['patch_size']
        self.enc = ViT(w, 'encoder.', self.cfg, dtype)
        self.dec = ViT(w, 'decoder.', self.cfg, dtype)
        self.top = {k: v.astype(dtype) for k, v in w.items()
                    if k.split('.')[0] in ('quant_embed', 'post_quant_embed', 'quantize')}
        self.dtype = dtype

    def encode(self, image):
        """image: float [H, W, 3] in [0, 1], H and W multiples of the patch -> (codes [gh*gw], (gh, gw))."""
        p = self.p
        x = mx.array(image, dtype=self.dtype) * 2 - 1                     # zero_mean
        gh, gw = x.shape[0] // p, x.shape[1] // p
        patches = x.reshape(gh, p, gw, p, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, 3 * p * p)
        wpe = self.enc.w['patch_embed.weight'].reshape(self.cfg['width'], 3 * p * p)
        h = patches @ wpe.T + self.enc.w['patch_embed.bias']
        h = self.enc.blocks(h, gh, gw)
        z = normalize(h @ self.top['quant_embed.weight'].T + self.top['quant_embed.bias'])
        cb = normalize(self.top['quantize.embedding.weight'])
        codes = mx.concatenate([mx.argmax(z[i:i + 1024] @ cb.T, axis=1) for i in range(0, z.shape[0], 1024)])
        return codes, (gh, gw)

    def decode(self, codes, grid):
        """codes [gh*gw] -> uint8 [H, W, 3]."""
        gh, gw = grid
        p = self.p
        z = normalize(self.top['quantize.embedding.weight'][mx.array(codes)])
        h = z @ self.top['post_quant_embed.weight'].T + self.top['post_quant_embed.bias']
        h = self.dec.blocks(h, gh, gw)
        h = mx.tanh(self.dec.lin(h, 'ffn.0', bias=True))
        h = self.dec.lin(h, 'conv_out', bias=True)                         # [N, (c sh sw)]
        img = h.reshape(gh, gw, 3, p, p).transpose(0, 3, 1, 4, 2).reshape(gh * p, gw * p, 3)
        img = mx.clip(img.astype(mx.float32), -1, 1)
        return np.array(mx.round((img + 1) / 2 * 255)).astype(np.uint8)


def load_image(path, multiple):
    from PIL import Image
    im = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
    h, w = min(im.shape[0], 1024) // multiple * multiple, min(im.shape[1], 1024) // multiple * multiple
    return im[:h, :w]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * math.log10(255 ** 2 / mse)


def main():
    from PIL import Image
    what, folder = sys.argv[1], sys.argv[2]
    tok = Tokenizer(folder)
    if what == 'recon':
        im = load_image(sys.argv[3], tok.p)
        codes, grid = tok.encode(im)
        out = tok.decode(codes, grid)
        ref = np.round(im * 255).astype(np.uint8)
        again, _ = tok.encode(out.astype(np.float32) / 255)
        print(f'grid {grid}; PSNR {psnr(ref, out):.2f} dB; codes kept on re-encoding '
              f'{float(mx.mean(again == codes)):.3f}; distinct codes {len(set(np.array(codes).tolist()))}')
        Image.fromarray(np.concatenate([ref, out], axis=1)).save(sys.argv[4])
    elif what == 'decode':
        d = np.load(sys.argv[3])
        Image.fromarray(tok.decode(d['codes'], tuple(int(g) for g in d['grid']))).save(sys.argv[4])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
