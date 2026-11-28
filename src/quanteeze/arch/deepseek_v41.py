# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Inference hooks for mlx-vlm's DeepSeek-V4.1 MoE block (`model_type: deepseek_v41`).

`prepare(model, path)` keeps FP8 MLA/shared-expert/indexer/Engram projections in MXFP8, keeps the
vocabulary head in BF16 (`BF16Head`), streams the input token embedding from SSD (`SSDEmbedding`),
drops the unused vision tower (`no_vision`), packs FP4 routed experts into MXFP4, and installs the
DSpark speculative taps (`speculative`).

`install(model)` gives every MoE block `RowGate`, hooks `expert_qmv` on its `SwitchGLU`, and makes
short verify passes bit-for-bit exact (`exact_projections`).
"""
import hashlib
import threading
import weakref
from collections import OrderedDict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.deepseek_v41.language import (Compressor, DeepseekV41Block, DeepseekV41Cache, DeepseekV41MoE,
                                                   DeepseekV41MoEGate, LanguageModel, ParallelHead,
                                                   make_identity_pre_mix)
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0

from .. import kernel_checks, per_load, per_thread



def images_mode(path=None, model=None):
    """`QUANTEEZE_V41_READS_IMAGES`: `'on'`, `'ssd'`, or `'off'`.  When `'on'`, follows
    `QUANTEEZE_VISION_SSD` (`auto` -> `'ssd'` when `path`/`model` is tight on local storage)."""
    from .. import variables, vision_ssd
    raw = (variables.get('QUANTEEZE_V41_READS_IMAGES') or 'on').strip().lower()
    if raw in ('0', 'false', 'no', 'off'):
        return 'off'
    if raw == 'ssd':
        return 'ssd'
    if raw in ('1', 'true', 'yes', 'on'):
        return 'ssd' if vision_ssd.enabled(path, model=model) else 'on'
    raise ValueError(f'QUANTEEZE_V41_READS_IMAGES: on, ssd, or off, not {raw}')


def prepare(model, path):
    """The routed experts and the projections as the checkpoint at `path` stores them, which
    mlx-vlm loads as bf16: DeepSeek's own V4.1 keeps its experts in FP4 (two values a byte, an E8M0
    scale a group of 32) -- four times the bytes in bf16; quantized to MXFP4 they are those bytes
    again -- only where the switches are not quantized and
    the checkpoint's experts are FP4 (q2_experts.Checkpoint.mode); and its attention's, shared
    experts', indexer's and Engram's projections in FP8 (fp8_projections()).  The MTP head is left
    as it is (quanteeze.dspark reads it), the language model given the taps its drafter reads
    (speculative()).  The token embedding is read from the checkpoint a row at a time
    (ssd_embedding()), and the vision tower let go (no_vision()) where QUANTEEZE_V41_READS_IMAGES
    is off or streamed block by block from SSD where `images_mode(path, model=model) == 'ssd'`.
    Returns whether it changed anything."""
    from mlx_vlm.models.switch_layers import SwitchLinear
    from .. import q2_experts
    speculative(model)                                  # the DSpark drafter's taps, inert without it
    done = bf16_head(model)                             # first: its float32 (2.6 GB) freed before fp8_projections
    done = fp8_projections(model, path) > 0 or done
    done = ssd_embedding(model, path) or done
    blocks = [(i, layer.ffn) for i, layer in enumerate(getattr(model, 'language_model', model).layers)
              if isinstance(getattr(layer, 'ffn', None), DeepseekV41MoE)]
    if blocks and isinstance(blocks[0][1].switch_mlp.gate_proj, SwitchLinear):
        try:
            mode = q2_experts.Checkpoint(path).mode(blocks[0][0], 'gate_proj')
        except Exception:
            mode = None
        if mode == 'mxfp4':
            nn.quantize(model, group_size=32, bits=4, mode='mxfp4',
                        class_predicate=lambda p, m: hasattr(m, 'to_quantized') and 'switch_mlp' in p
                        and 'mtp' not in p.split('.'))
            done = True
    mode_img = images_mode(path, model=model)
    if mode_img == 'off':
        done = no_vision(model) or done
    else:
        images_one_at_a_time(model)
        fused_vision()
        vision_block_by_block()
        if mode_img == 'ssd':
            done = (vision_from_ssd(model, path) > 0) or done
    return done


HEAD_ROWS = 8                   # rows of logits a call reads the head once for, in BF16Head's kernel:
                                # a decode's, a verify pass's (6), a DSpark block's (5)
HEAD_CHUNK = 16384              # rows of the head made float32 at a time, for more rows than that
_HEAD = r"""
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row0 = (threadgroup_position_in_grid.x * SGS + sg) * ROWS;
    float acc[M_ROWS][ROWS];
    for (int m = 0; m < M_ROWS; m++) for (int r = 0; r < ROWS; r++) acc[m][r] = 0.0f;
    for (uint k = lane * 8; k < K; k += 256) {
        float xs[M_ROWS][8];
        for (int m = 0; m < M_ROWS; m++) for (int j = 0; j < 8; j++) xs[m][j] = x[m * K + k + j];
        for (int r = 0; r < ROWS; r++) {
            uint row = row0 + r;
            if (row >= N) break;
            const device bfloat* wr = w + (ulong)row * K + k;
            float ws[8];
            for (int j = 0; j < 8; j++) ws[j] = static_cast<float>(wr[j]);
            for (int m = 0; m < M_ROWS; m++) {
                float s = 0.0f;
                for (int j = 0; j < 8; j++) s += xs[m][j] * ws[j];
                acc[m][r] += s;
            }
        }
    }
    for (int r = 0; r < ROWS; r++) {
        uint row = row0 + r;
        for (int m = 0; m < M_ROWS; m++) {
            float t = simd_sum(acc[m][r]);
            if (lane == 0 && row < N) out[m * N + row] = t;
        }
    }
"""
_CUDA_HEAD = r"""
  const uint lane = threadIdx.x & 31u;
  const uint sg = threadIdx.x >> 5;
  const uint row0 = (blockIdx.x * WARPS + sg) * ROWS;
  float acc[M_ROWS][ROWS];
  for (int m = 0; m < M_ROWS; m++) for (int r = 0; r < ROWS; r++) acc[m][r] = 0.0f;
  for (uint k = lane * 8u; k < K; k += 256u) {
    float xs[M_ROWS][8];
    for (int m = 0; m < M_ROWS; m++) for (int j = 0; j < 8; j++) xs[m][j] = x[m * K + k + j];
    for (int r = 0; r < ROWS; r++) {
      const uint row = row0 + r;
      if (row >= N) break;
      const auto* wr = w + static_cast<unsigned long long>(row) * K + k;
      float ws[8];
      for (int j = 0; j < 8; j++) ws[j] = static_cast<float>(wr[j]);
      for (int m = 0; m < M_ROWS; m++) {
        float s = 0.0f;
        for (int j = 0; j < 8; j++) s += xs[m][j] * ws[j];
        acc[m][r] += s;
      }
    }
  }
  for (int r = 0; r < ROWS; r++) {
    const uint row = row0 + r;
    for (int m = 0; m < M_ROWS; m++) {
      float t = acc[m][r];
      for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
      if (lane == 0u && row < N) out[m * N + row] = t;
    }
  }
"""
_KERNELS = {}
HEAD_KERNEL = [None]
HEAD_KERNEL_CHECKS = 2
HEAD_KERNEL_TOL = 1e-4
HEAD_KERNEL_STATE = {}


def _on_cuda():
    from .. import cuda_v41
    return cuda_v41._gpu()


def _head_kernel_on():
    if HEAD_KERNEL[0] is None:
        from .. import variables
        HEAD_KERNEL[0] = not variables.off('QUANTEEZE_V41_HEAD_KERNEL', variables.at_load())
    return HEAD_KERNEL[0]


def _cuda_head(rows, weight):
    """Direct BF16-to-FP32 vocabulary head kernel on CUDA for `m <= HEAD_ROWS` without FP32 weight copies."""
    m, k = rows.shape
    n = weight.shape[0]
    if k % 256 != 0 or not _head_kernel_on():
        return None
    key = (m, k, n)
    state = HEAD_KERNEL_STATE.get(key)
    if state is not None and not state[0]:
        return None
    from .. import cuda_kernels
    if 'cuda_head' not in _KERNELS:
        _KERNELS['cuda_head'] = cuda_kernels.kernel('quanteeze_cuda_bf16_head', ['x', 'w'], ['out'], _CUDA_HEAD)
    per = 4 * 8
    try:
        got = _KERNELS['cuda_head'](
            inputs=[rows, weight],
            template=[('M_ROWS', m), ('ROWS', 4), ('WARPS', 8), ('K', k), ('N', n)],
            grid=(-(-n // per) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(m, n)],
            output_dtypes=[mx.float32],
        )[0]
    except Exception:
        HEAD_KERNEL_STATE[key] = (False, HEAD_KERNEL_CHECKS)
        return None
    if state is not None and state[0] and state[1] >= HEAD_KERNEL_CHECKS:
        return got
    parts = []
    for s in range(0, n, HEAD_CHUNK):
        parts.append(rows @ weight[s:s + HEAD_CHUNK].astype(mx.float32).T)
        mx.eval(parts[-1])
    want = mx.concatenate(parts, axis=-1)
    mx.eval(got, want)
    scale = max(1.0, float(mx.max(mx.abs(want)).item()))
    ok = bool(mx.all(mx.isfinite(got)).item()) and float(mx.max(mx.abs(got - want)).item()) <= HEAD_KERNEL_TOL * scale
    seen = (state[1] if state is not None else 0) + 1
    HEAD_KERNEL_STATE[key] = (ok, seen if ok else HEAD_KERNEL_CHECKS)
    return want


class BF16Head(ParallelHead):
    """The vocabulary projection with its weight in bf16, as the checkpoint stores it, its logits
    float32 as ParallelHead's: mlx-vlm keeps the weight float32 so as not to cast it each
    token, which reads twice its bytes (129,280 x 5,120: 5.11 ms a decode).
    A few rows (HEAD_ROWS: a decode's, a verify pass's) go through a kernel that reads the bf16 weight once and
    sums in float32 (2.45 ms; within 1.6e-6 of the float32 product, its order of sums aside);
    more (a prompt's) through the float32 product, HEAD_CHUNK rows of the weight made float32 at a
    time, or wherever there is no Metal.

    `keep` (`mlx-vlm`'s `logits_to_keep`): on a pass of more than `HEAD_ROWS` tokens (such as a
    prompt's final step), each `HEAD_CHUNK` product is evaluated and sliced to the last `keep`
    positions before the next chunk's float32 copy is made, avoiding 2.65 GB of live float32 weight
    copies and full-prompt logits on both Metal and CUDA."""

    def __call__(self, x, keep=None):
        k, n = x.shape[-1], self.weight.shape[0]
        if keep is not None and (x.ndim < 3 or keep >= x.shape[-2]):
            keep = None
        if keep is not None and x.reshape(-1, k).shape[0] > HEAD_ROWS:
            rows = x.reshape(-1, k).astype(mx.float32)
            lead = tuple(x.shape[:-1])
            parts = []
            for s in range(0, n, HEAD_CHUNK):
                part = (rows @ self.weight[s:s + HEAD_CHUNK].astype(mx.float32).T).reshape(*lead, -1)[..., -keep:, :]
                mx.eval(part)
                parts.append(part)
            return mx.concatenate(parts, axis=-1)
        rows = x.reshape(-1, k).astype(mx.float32)
        m = rows.shape[0]
        if m <= HEAD_ROWS and k % 256 == 0 and mx.metal.is_available() and mx.default_device() == mx.gpu:
            if 'head' not in _KERNELS:
                _KERNELS['head'] = mx.fast.metal_kernel(name='quanteeze_bf16_head', input_names=['x', 'w'],
                                                        output_names=['out'], source=_HEAD)
            per = 4 * 8
            out = _KERNELS['head'](inputs=[rows, self.weight], template=[('M_ROWS', m), ('ROWS', 4), ('SGS', 8),
                                                                          ('K', k), ('N', n)],
                                   grid=(-(-n // per) * 256, 1, 1), threadgroup=(256, 1, 1),
                                   output_shapes=[(m, n)], output_dtypes=[mx.float32])[0]
        elif m <= HEAD_ROWS and _on_cuda():
            out = _cuda_head(rows, self.weight)
            if out is None:
                parts = []
                for s in range(0, n, HEAD_CHUNK):
                    parts.append(rows @ self.weight[s:s + HEAD_CHUNK].astype(mx.float32).T)
                    mx.eval(parts[-1])
                out = mx.concatenate(parts, axis=-1)
        else:
            out = mx.concatenate([rows @ self.weight[s:s + HEAD_CHUNK].astype(mx.float32).T
                                  for s in range(0, n, HEAD_CHUNK)], axis=-1)
        out = out.reshape(*x.shape[:-1], n)
        return out if keep is None else out[..., -keep:, :]


def bf16_head(model):
    """The head as BF16Head where its float32 weight holds bf16's values (V4.1's: BF16 in the
    checkpoint, made float32 by mlx-vlm's sanitize), 1.3 GB less held and read each token.
    Whether it did."""
    lm = getattr(model, 'language_model', model)
    head = getattr(lm, 'head', None)
    if type(head) is not ParallelHead or head.weight.dtype != mx.float32:
        return False
    half = head.weight.astype(mx.bfloat16)
    if not mx.array_equal(half.astype(mx.float32), head.weight).item():
        return False
    head.weight = half
    head.__class__ = BF16Head
    return True


class SSDEmbedding(nn.Module):
    """The token embedding read from the checkpoint a row at a time (numpy's memory map of its
    BF16 table), not held: a decode reads one row of its 129,280 (10 KB), and the pages the OS
    keeps.  Looking rows up waits for the token ids, as the Engram's tables read from the SSD do
    in the same pass (OffloadedEngramEmbedding), so no step waits where it did not."""

    def __init__(self, path, key='embed.weight'):
        super().__init__()
        import json
        import struct
        from pathlib import Path
        import numpy as np
        from .. import vision_ssd
        wmap = vision_ssd.weight_map(path)
        if not wmap or key not in wmap:
            raise KeyError(key)
        shard = Path(path) / wmap[key]
        with open(shard, 'rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            tensor = json.loads(f.read(length))[key]
        if tensor['dtype'] != 'BF16':
            raise ValueError(f'{key}: {tensor["dtype"]}, not BF16')
        start, _ = tensor['data_offsets']
        object.__setattr__(self, '_rows', np.memmap(shard, mode='r', dtype=np.uint16, offset=8 + length + start,
                                                    shape=tuple(tensor['shape'])))

    @property
    def shape(self):
        return self._rows.shape

    def __call__(self, ids):
        import numpy as np
        return mx.array(self._rows[np.asarray(ids)]).view(mx.bfloat16)


def ssd_embedding(model, path):
    """The language model's token embedding as SSDEmbedding where the checkpoint's BF16 table
    holds the same values (checked on its first, last and some random rows), 1.32 GB less held.
    Whether it did."""
    import numpy as np
    lm = getattr(model, 'language_model', model)
    embed = getattr(lm, 'embed_tokens', None)
    if type(embed) is not nn.Embedding or embed.weight.dtype != mx.bfloat16:
        return False
    try:
        ssd = SSDEmbedding(path)
    except (OSError, ValueError, KeyError):
        return False
    n = embed.weight.shape[0]
    rows = mx.array(np.unique(np.r_[0, n - 1, np.random.default_rng(0).integers(0, n, 64)]))
    if ssd.shape != tuple(embed.weight.shape) or not mx.array_equal(ssd(rows), embed(rows)).item():
        return False
    lm.embed_tokens = ssd
    return True


def _fused_attention(self, x, cos, sin):
    """mlx-vlm's VisionAttention as MLX's fused attention computes it: no [heads, n, n] scores."""
    from mlx_vlm.models.deepseek_v4.vision import apply_rotary
    n = x.shape[0]
    q, k, v = (t.reshape(n, self.n_heads, self.head_dim) for t in mx.split(self.wqkv(x), 3, axis=-1))
    q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    o = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None],
                                             v.transpose(1, 0, 2)[None], scale=self.scale)[0]
    return self.wo(o.transpose(1, 0, 2).reshape(n, -1))


def fused_vision():
    """V4.1's vision tower attending through MLX's fused attention."""
    from mlx_vlm.models.deepseek_v41 import vision
    if vision.VisionAttention.__call__ is _fused_attention:
        return False
    vision.VisionAttention.__call__ = _fused_attention
    return True


def _block_evaluated(block):
    def __call__(self, x, cos, sin):
        ssd = self.__dict__.get('_quanteeze_from_ssd')
        if ssd is not None:
            ssd.load()
        try:
            y = block(self, x, cos, sin)
            mx.eval(y)
        finally:
            if ssd is not None:
                ssd.release()
        return y
    __call__._quanteeze_evaluated = block
    return __call__


def vision_block_by_block():
    """Evaluate each of V4.1's 32 vision blocks before building the next so MLX holds one block's
    intermediates (~0.34 GB) rather than all 32 blocks at once (7-12.3 GB on high-resolution images),
    with identical arithmetic bit for bit."""
    from mlx_vlm.models.deepseek_v41 import vision
    if getattr(vision.VisionBlock.__call__, '_quanteeze_evaluated', False):
        return False
    vision.VisionBlock.__call__ = _block_evaluated(vision.VisionBlock.__call__)
    return True


def vision_from_ssd(model, path):
    """Stream V4.1's `vision` tower and `aligner` BF16 weights from `path` on demand one block at a time,
    releasing matching weights from memory until an image is encoded."""
    import sys
    from mlx.utils import tree_flatten
    from .. import vision_ssd
    vision_mod = model.get('vision') if isinstance(model, dict) else getattr(model, 'vision', None)
    aligner_mod = model.get('aligner') if isinstance(model, dict) else getattr(model, 'aligner', None)
    params = []
    if isinstance(vision_mod, nn.Module):
        params.extend((['vision.' + k, 'model.vision.' + k], vision_mod, k, a)
                      for k, a in tree_flatten(vision_mod.parameters()))
    if isinstance(aligner_mod, nn.Module):
        params.extend((['aligner.' + k, 'model.aligner.' + k], aligner_mod, k, a)
                      for k, a in tree_flatten(aligner_mod.parameters()))
    if not params:
        return 0
    taken, maps, freed = vision_ssd.matching(path, params)
    if not taken:
        return 0
    rest = dict(taken)
    blocks = getattr(vision_mod, 'blocks', None) or []
    for b, blk in enumerate(blocks):
        mine = {name: v for name, v in taken.items() if v[0] is vision_mod and v[1].startswith(f'blocks.{b}.')}
        for name in mine:
            del rest[name]
        if mine:
            object.__setattr__(blk, '_quanteeze_from_ssd', vision_ssd.VisionWeights(maps, mine))
    object.__setattr__(model, '_quanteeze_vision_rest', vision_ssd.VisionWeights(maps, rest) if rest else None)
    vision_ssd.VisionWeights(maps, taken).release()
    print(f'quanteeze: DeepSeek V4.1 vision/aligner streamed from checkpoint on demand: '
          f'{freed / 1e9:.2f} GB released ({len(taken)} weights)', file=sys.stderr, flush=True)
    return freed


def no_vision(model):
    """The vision tower and its aligner let go (0.97 GB), where QUANTEEZE_V41_READS_IMAGES is off:
    images are then not served to V4.1 (quanteeze.dsml leaves them out of its prompts and marks
    where each was).  Whether it did."""
    let_go = False
    for name in ('vision', 'aligner'):
        module = model.get(name) if isinstance(model, dict) else None
        if isinstance(module, nn.Module) and module.parameters():
            model[name] = nn.Module()
            let_go = True
    lm = getattr(model, 'language_model', None)
    if let_go and lm is not None:
        object.__setattr__(lm, '_quanteeze_text', True)        # SpecLanguageModel makes no image mask
    return let_go


IMAGES_KEPT_GB = 1.0            # images' embeddings kept for the requests after (images_one_at_a_time): a
IMAGES = OrderedDict()          # 1920x1080 image's some 24 MB; (patches' digest, grid) -> embeddings
IMAGE_STATS = dict(encoded=0, reused=0)


def images_one_at_a_time(model):
    """Each image a prompt carries through the vision tower and aligner, evaluated before the
    next, and cached across requests (IMAGES_KEPT_GB)."""
    cls = type(model)
    base = getattr(cls, 'encode_image', None)
    if base is None or getattr(base, '_quanteeze_one_at_a_time', False):
        return False

    def encode_image(self, patches, n_vit_h, n_vit_w):
        key = (hashlib.blake2b(np.asarray(patches).tobytes(), digest_size=16).digest(), int(n_vit_h), int(n_vit_w))
        out = IMAGES.get(key)
        if out is not None:
            IMAGES.move_to_end(key)
            IMAGE_STATS['reused'] += 1
            return out
        rest = self.__dict__.get('_quanteeze_vision_rest')
        if rest is not None:
            rest.load()
        try:
            out = base(self, patches, n_vit_h, n_vit_w)
            mx.eval(out)
        finally:
            if rest is not None:
                rest.release()
        IMAGE_STATS['encoded'] += 1
        IMAGES[key] = out
        while IMAGES and sum(a.nbytes for a in IMAGES.values()) > IMAGES_KEPT_GB * 1e9:
            IMAGES.popitem(last=False)
        return out
    encode_image._quanteeze_one_at_a_time = True
    cls.encode_image = encode_image
    return True


def fp8_projections(model, path):
    """The linear layers the checkpoint at `path` stores in FP8 (an E8M0 '.scale' beside the
    weight, a block of 32 x 32; the routed experts, the Engram's embedding and the MTP head
    aside), which mlx-vlm dequantizes to bf16, as MXFP8 again: the same values (a group of 32 of a
    row takes its block's scale, or a smaller one: each requantized value checked, a layer whose
    are not the same left in bf16), half the bytes, and a decode's one-token products at
    32768 x 1280 and 5120 x 8192 2.4 times as fast (0.134 to 0.055 ms, 0.126 to 0.052), wo_a's
    eight 1024 x 4096 (a MultiLinear, its weight a head's matrix each: QuantizedMultiLinear) 2.3
    (0.096 to 0.041).  On DeepSeek V4.1 Flash that is 6.4 of its 19.1 GB beside the experts, and
    the shared experts' 1.4 more (named w1, w2, w3 in the checkpoint, gate_proj, down_proj and
    up_proj in mlx-vlm: they were left in bf16, 2.83 GB).  Returns how many it
    quantized."""
    from .. import vision_ssd
    names = vision_ssd.weight_map(path)
    if not names:
        return 0
    shared = {f'.shared_experts.{w}': f'.shared_experts.{p}' for w, p in (('w1', 'gate_proj'), ('w2', 'down_proj'),
                                                                          ('w3', 'up_proj'))}

    def named(k):                                       # the checkpoint's name as mlx-vlm's module
        for w, p in shared.items():
            k = k.replace(w, p)
        return 'language_model.' + k
    fp8 = {named(k[:-len('.scale')]) for k in names
           if k.startswith('layers.') and k.endswith('.scale') and '.experts.' not in k and not k.endswith('.embed.scale')}
    count = [0]

    from mlx.utils import tree_flatten
    from mlx_vlm.models.mla import MultiLinear

    def same(p, m):
        if (p not in fp8 or not isinstance(m, (nn.Linear, MultiLinear)) or m.weight.ndim not in (2, 3)
                or m.weight.shape[-1] % 32):
            return False
        q = mx.quantize(m.weight, group_size=32, bits=8, mode='mxfp8')
        ok = mx.array_equal(mx.dequantize(*q, group_size=32, bits=8, mode='mxfp8', dtype=m.weight.dtype), m.weight).item()
        count[0] += ok
        return ok

    for prefix in sorted({'.'.join(n.split('.')[:3]) + '.' for n in fp8}):
        done = []

        def one(p, m, prefix=prefix):
            if not p.startswith(prefix) or not same(p, m):
                return False
            done.append(p)
            return True
        nn.quantize(model, group_size=32, bits=8, mode='mxfp8', class_predicate=one)
        if done:
            modules = dict(model.named_modules())
            mx.eval([v for p in done for _, v in tree_flatten(modules[p].parameters())])
    return count[0]



ROWS = 8                        # a pass of at most this many tokens multiplies its float32 rows one at a time
PLAIN_EVAL_DEFAULT = 1          # QUANTEEZE_EVAL_LAYERS auto for V4.1 on Metal: hand each layer on as it is built
PLAIN_EVAL = [0]                # quanteeze.serve: a pass of up to ROWS tokens (a decode, an MTP verify pass) handed to
                                # the GPU this many layers at a time as it is built, on Metal (0: the whole pass at its
                                # end, mlx-vlm's).  Its token's SSD embedding and Engram rows wait for the last token,
                                # so the GPU otherwise stands idle while the host builds the whole pass


def by_rows(x, w):
    """x @ w.T, a pass of a few tokens (2 to ROWS) a row at a time, as a decode multiplies its one:
    MLX's matrix product of a few rows is not its vector product's bits (the router's, 5120 x 384,
    and the hyper-connections' mix, 20480 x 24, both float32), and on the Mac it took 2.5 and 7 ms
    more a 3-token pass over the 40 layers than the rows one by one."""
    from .. import prefill
    n = x.shape[-2] if x.ndim >= 2 else 1
    if 1 < n <= ROWS:
        return mx.concatenate([x[..., i:i + 1, :] @ w.T for i in range(n)], axis=-2)
    parts = prefill.window_parts(n)
    if parts is not None:
        at, outs = 0, []
        for m in parts:
            outs.append(by_rows(x[..., at:at + m, :], w))
            at += m
        return mx.concatenate(outs, axis=-2)
    return x @ w.T


@mx.compile
def hc_mix_rows(x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_eps):
    """mlx-vlm's hc_mix_coeffs, the mix a row at a time for a few tokens (by_rows)."""
    from mlx_vlm.models.deepseek_v41.language import hc_split_sinkhorn
    mixes = by_rows(mx.fast.rms_norm(x.flatten(-2).astype(mx.float32), None, norm_eps), hc_fn)
    return hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)


def hc_mix_kernel(x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_eps):
    """`hc_mix_coeffs` using `cuda_v41.split` to run the Sinkhorn rounds in one CUDA kernel."""
    from .. import cuda_v41
    mixes = by_rows(mx.fast.rms_norm(x.flatten(-2).astype(mx.float32), None, norm_eps), hc_fn)
    return cuda_v41.split(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)


@mx.compile
def _gate_scores(logits, bias):
    """The router's scores and what it picks by (the scores less the correction bias), fused."""
    scores = mx.sqrt(nn.softplus(logits))
    return scores, -(scores + bias)


@mx.compile
def _gate_weights(scores, inds, scale):
    """The picked experts' scores, renormalized and scaled, fused."""
    weights = mx.take_along_axis(scores, inds, axis=-1)
    return weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * scale


class RowGate(DeepseekV41MoEGate):
    """The stock router, its scores by_rows, its elementwise steps compiled (_gate_scores,
    _gate_weights: 24 graph nodes a layer were 9, the same bits), or fused into `cuda_v41.route` on CUDA."""

    def logits(self, x):
        return by_rows(x.astype(mx.float32), self.weight)

    def by_ops(self, logits, bias=None):
        if bias is None:
            bias = self.bias
        scores, key = _gate_scores(logits, bias)
        inds = mx.stop_gradient(mx.argpartition(key, kth=self.top_k - 1, axis=-1)[..., :self.top_k].astype(mx.int32))
        if self.norm_topk_prob and self.top_k > 1:
            weights = _gate_weights(scores, inds, mx.array(self.routed_scaling_factor, dtype=mx.float32))
        else:
            weights = mx.take_along_axis(scores, inds, axis=-1) * self.routed_scaling_factor
        return inds, weights, scores

    def from_logits(self, logits, bias=None):
        from .. import cuda_v41
        if bias is None:
            bias = self.bias
        if (self.norm_topk_prob and self.top_k > 1 and cuda_v41.ROUTE[0] and cuda_v41.takes_bias(logits, bias)
                and cuda_v41.takes_route(logits, self.top_k)):
            inds, weights, _ = cuda_v41.route(logits, bias, self.routed_scaling_factor, self.top_k)
            return inds, weights
        inds, weights, _ = self.by_ops(logits, bias)
        return inds, weights

    def __call__(self, x, image_mask=None):
        bias = mx.where(image_mask[..., None], self.bias_vl, self.bias) if image_mask is not None else self.bias
        return self.from_logits(self.logits(x), bias)


_BOUND = threading.local()       # the positions a compression source's call rotates stay below this


def _window_indices(start_pos, seqlen, win):
    """For a pass of `seqlen` tokens at `start_pos` on a ring of `win`: the ring's slots holding the
    window before it (None at the start), the runs of slots it writes ((slot a, slot b, from, to)
    of its last min(seqlen, win) tokens), and the mask of the positions each query sees -- the
    values mlx-vlm's _window_part makes with arange in each of the 40 layers, made here once a
    pass and shared by them (35 graph nodes a layer were 4 on a decode)."""
    windows = per_thread.PASS.memo.setdefault('v41_windows', {})
    key = (start_pos, seqlen, win)
    got = windows.get(key)
    if got is None:
        if len(windows) > 16:
            windows.clear()
        import numpy as np
        base, end = max(0, start_pos - win + 1), start_pos + seqlen
        history = mx.array(np.arange(base, start_pos, dtype=np.int32) % win) if start_pos else None
        retained = min(seqlen, win)
        writes, at = [], 0
        while at < retained:
            slot = (end - retained + at) % win
            n = min(retained - at, win - slot)
            writes.append((slot, slot + n, at, at + n))
            at += n
        positions, queries = np.arange(base, end), np.arange(start_pos, end)
        valid = (positions[None, :] <= queries[:, None]) & (positions[None, :] > queries[:, None] - win)
        got = windows[key] = (history, writes, mx.array(valid[None, None]))
    return got


def _window_part(self, x, start_pos, cache):
    """mlx-vlm's _window_part, its indices and mask _window_indices's, the ring written by slices."""
    batch, seqlen = x.shape[0], x.shape[1]
    win = self.window_size
    kv = self.kv_norm(self.wkv(fake_quant_fp8_ue8m0(x)))
    kv = self.rope(kv[:, None], start_pos).reshape(batch, seqlen, self.head_dim)
    kv = fake_quant_fp8_ue8m0(kv)
    buffer = cache.window[self.layer_idx]
    if buffer is None:
        buffer = mx.zeros((batch, win, self.head_dim), dtype=kv.dtype)
    elif buffer.shape[0] < batch:
        buffer = mx.pad(buffer, [(0, batch - buffer.shape[0]), (0, 0), (0, 0)])
    history, writes, valid = _window_indices(start_pos, seqlen, win)
    part = kv if history is None else mx.concatenate([mx.take(buffer[:batch], history, axis=1), kv], axis=1)
    tail = kv[:, -min(seqlen, win):]
    for a, b, c, d in writes:
        buffer[:batch, a:b] = tail[:, c:d]
    cache.window[self.layer_idx] = buffer
    return part, valid


def _compress_part(self, x, qr, start_pos, cache):
    """mlx-vlm's _compress_part, the positions its latents can take bounded first (_BOUND)."""
    _BOUND.top = start_pos + x.shape[1] + self.compress_ratio + 1
    try:
        return _COMPRESS_PART[0](self, x, qr, start_pos, cache)
    finally:
        _BOUND.top = None


def _rope_at_positions(x, positions, rope_dim, theta, yarn):
    """mlx-vlm's _apply_rope_at_positions without its host sync: the table's length from the
    bound _compress_part set (a row of the table is the same whatever its length: numpy's, by
    position), not from the positions' maximum (.item(), four a pass: the GPU waited for each,
    and the next token's graph could not be built while it ran)."""
    from mlx_vlm.models.deepseek_v41.language import _apply_index_rotary, _index_cos_sin
    top = getattr(_BOUND, 'top', None)
    if top is None:
        return _ROPE_AT_POSITIONS[0](x, positions, rope_dim, theta, yarn)
    cos, sin = _index_cos_sin(top, rope_dim, theta, yarn)
    shape = (1,) * (x.ndim - positions.ndim - 1) + positions.shape + (cos.shape[-1],)
    return _apply_index_rotary(x, cos[positions].reshape(shape), sin[positions].reshape(shape), rope_dim)


EXPERT_ROWS = 1                 # a verify pass's slots naming an expert, a threadgroup each (expert_qmv.ROWS): its
                                # tokens' rows together (VERIFY_T of them, read once) took a pass of 6 from 114 to
                                # 142 ms, of 4 from 75 to 97, of 2 about the same;
                                # the tokens share an expert seldom (80% of a pass of 2's slots distinct, 52% of 6's)
_COMPRESS_PART, _ROPE_AT_POSITIONS, _WINDOW_PART = [None], [None], [None]
_ATTENTION, _INDEXER = [None], [None]


def _verifying(x):
    """Whether a call is a verify pass's of 2 to ROWS tokens (SpecLanguageModel: VERIFYING)."""
    return VERIFYING[0] and x.ndim >= 2 and 1 < x.shape[1] <= ROWS


GROUPED = [True]                # the indexer's queries of one shape together in a verify pass (_grouped)
EXACT = [True]                  # a verify pass as one-token decodes, bit for bit (QUANTEEZE_MTP_EXACT=bits, the default;
                                # serve._drafter); distribution drafted degenerate text served (not for use);
                                # off (distribution): VERIFYING never set, the pass as mlx-vlm computes its tokens
                                # together but for the rows and experts (by_rows, expert_qmv: as fast).  A pass of
                                # 2 tokens 58 -> 53 ms, 4 78 -> 69, 6 108 -> 91, its build 13 -> 7 at 6; with the
                                # attention alone together 57, 72 and 99.
GROUP_CHOICE, GROUP_CHECKED = kernel_checks.tables('arch.deepseek_v41.group')
GROUP_STATS = dict(served=0, checked=0, mismatched=0, declined=0, alone=0)


@per_load.on_let_go
def let_go():
    """Clear per-model cached image embeddings and reloadable head kernel preference on unload."""
    IMAGES.clear()
    HEAD_KERNEL[0] = None


def _shapes(keys):
    """The queries 0.. grouped by their key (one shape each): [(key, [queries])], first seen first."""
    groups = {}
    for s, key in enumerate(keys):
        groups.setdefault(key, []).append(s)
    return list(groups.items())


def group_key(g, keys, topk, source, given, q_shape):
    """Key `_grouped` by `keys.bit_length()` size class instead of exact position so cache stays bounded."""
    return 'index', g, keys.bit_length(), keys > topk, source, given, tuple(q_shape[2:])


def _grouped(key, together, apart):
    """A group of a verify pass's queries of one shape, its outputs (arrays, each [G, ...]):
    `together` computes them with the queries in the batch, as many one-query decodes at once;
    `apart` a query at a time, as the decodes do.  A key is served together once its first
    verify_qmv.CHECKS groups gave apart's bits; a key that did not, apart from then on."""
    from .. import verify_qmv
    choice, seen = kernel_checks.read(GROUP_CHOICE, GROUP_CHECKED, key)
    if not GROUPED[0] or choice is False:
        GROUP_STATS['declined'] += 1
        return apart()
    if choice and seen >= verify_qmv.CHECKS:
        GROUP_STATS['served'] += 1
        return together()
    want, got = apart(), together()
    GROUP_STATS['checked'] += 1
    same = kernel_checks.same_bits(tuple(want), tuple(got))
    per_thread.checked(GROUP_CHOICE, GROUP_CHECKED, key, same, verify_qmv.CHECKS)
    if not same:
        GROUP_STATS['mismatched'] += 1
    return want


def _in_order(groups, outs):
    """The groups' outputs ([G, ...] each, in the groups' order) as one [T, ...] in the queries' order."""
    order = [s for _, queries in groups for s in queries]
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
    if order == sorted(order):
        return out
    place = [0] * len(order)
    for i, s in enumerate(order):
        place[s] = i
    return mx.take(out, mx.array(place), axis=0)


def _indexer(self, x, qr, latent, start_pos, offset, cache):
    """mlx-vlm's Indexer.__call__; in a verify pass each query's scores, candidate blocks and top-k as
    a one-token decode at its position computes them: the index keys it had ((position + 1) //
    ratio), its own compress length, its own candidates, and min(index_topk, those keys) picked --
    MLX's product of the pass's queries at once is not one query's bits (up to 4e-3 in a score,
    enough to pick another position).  The queries that had as many keys go
    together, in the batch (_grouped: each a one-query decode's shape).  The rows are padded with -1
    to the longest; each query's count is left on the cache (_quanteeze_picked) for the attention."""
    if not _verifying(x):
        return _INDEXER[0](self, x, qr, latent, start_pos, offset, cache)
    from mlx_vlm.models.deepseek_v41.language import (_apply_index_rotary, _index_cos_sin, _index_scores,
                                                      fake_quant_fp4_ue8m0, select_candidate_blocks)
    batch, seqlen = x.shape[0], x.shape[1]
    ratio, end_pos = self.compress_ratio, start_pos + seqlen
    if self.owns_k:
        if latent is not None:
            self._publish_keys(latent, start_pos, cache)
        cache.index_k = cache.keys[self.layer_idx]
    cos, sin = _index_cos_sin(end_pos, self.rope_head_dim, self.rope_theta, self.yarn)
    positions = mx.arange(start_pos, end_pos)
    q = self.wq_b(fake_quant_fp8_ue8m0(qr)).reshape(batch, seqlen, self.n_heads, self.index_head_dim)
    q = _apply_index_rotary(q, cos[positions][None, :, None, :], sin[positions][None, :, None, :], self.rope_head_dim)
    q = fake_quant_fp4_ue8m0(q)
    weights = self.weights_proj(x) * (self.scale * self.n_heads ** -0.5)
    given = cache._quanteeze_candidates if self.uses_candidates else None

    def candidates_of(queries):
        """The candidate-source layer's blocks for these queries, [G, 1, keys]."""
        rows = [given[s] for s in queries]
        if all(r[0] is rows[0][0] for r in rows) and [r[1] for r in rows] == list(range(rows[0][0].shape[0])):
            return rows[0][0]
        return mx.concatenate([r[0][r[1]:r[1] + 1] for r in rows], axis=0)

    def one(s):
        keys = (start_pos + s + 1) // ratio
        lens = (mx.arange(start_pos + s + 1, start_pos + s + 2) // ratio)[:, None]
        scores = _index_scores(mx.contiguous(q[:, s:s + 1]), cache.index_k[:batch, :keys], weights[:, s:s + 1], lens,
                               candidates_of([s]) if given is not None else None)
        top = min(self.index_topk, keys)
        idxs = mx.sort(mx.argsort(-scores, axis=-1)[..., :top], axis=-1).astype(mx.int32)
        row = mx.where(idxs < lens, idxs + offset, -1).astype(mx.int32)
        if self.is_candidate_source:
            return row, select_candidate_blocks(scores, lens, self.candidate_topk_blocks, self.candidate_block_size)
        return (row,)

    def together(queries):
        g, keys = len(queries), (start_pos + queries[0] + 1) // ratio
        at = mx.array(queries)
        lens = mx.full((g, 1, 1), keys, dtype=mx.int32)
        q_g = mx.contiguous(mx.take(q, at, axis=1).transpose(1, 0, 2, 3))
        w_g = mx.take(weights, at, axis=1).transpose(1, 0, 2)
        scores = _index_scores(q_g, mx.broadcast_to(cache.index_k[:batch, :keys], (g, keys, q.shape[-1])), w_g, lens,
                               candidates_of(queries) if given is not None else None)
        top = min(self.index_topk, keys)
        idxs = mx.sort(mx.argsort(-scores, axis=-1)[..., :top], axis=-1).astype(mx.int32)
        row = mx.where(idxs < lens, idxs + offset, -1).astype(mx.int32)
        if self.is_candidate_source:
            return row, select_candidate_blocks(scores, lens, self.candidate_topk_blocks, self.candidate_block_size)
        return (row,)

    def apart(queries):
        outs = [one(s) for s in queries]
        return tuple(mx.concatenate([o[i] for o in outs], axis=0) for i in range(len(outs[0])))

    groups = _shapes([(start_pos + s + 1) // ratio for s in range(seqlen)])
    outs, picked, candidates = [], [0] * seqlen, {}
    for keys, queries in groups:
        if batch != 1 or len(queries) == 1:
            GROUP_STATS['alone'] += 1
            out = apart(queries)
        else:
            out = _grouped(group_key(len(queries), keys, self.index_topk, self.is_candidate_source,
                                     given is not None, q.shape),
                           lambda queries=queries: together(queries), lambda queries=queries: apart(queries))
        outs.append(out[0])
        for i, s in enumerate(queries):
            picked[s] = min(self.index_topk, keys)
            if self.is_candidate_source:
                candidates[s] = (out[1], i)
    if self.is_candidate_source:
        cache._quanteeze_candidates = candidates
    width = max(picked)
    cache._quanteeze_picked = picked
    outs = [r if r.shape[-1] == width else
            mx.concatenate([r, mx.full(r.shape[:-1] + (width - r.shape[-1],), -1, mx.int32)], axis=-1) for r in outs]
    return _in_order(groups, outs).transpose(1, 0, 2)


ATTENTION_PIECE = 128
ATTENTION_ALIGN = 128
_SPARSE_ATTENTION = [None]


def _pieces(n, piece=ATTENTION_PIECE):
    return [(a, min(a + piece, n)) for a in range(0, n, piece)]


def _sparse_attention(self, q, window, pool, indices, window_mask):
    """`DeepseekV41Attention._sparse_attention` evaluated in `ATTENTION_PIECE`-query slices on long
    prefill passes so `_gather_pooled` holds at most `ATTENTION_PIECE` queries of gathered keys at once."""
    seqlen = q.shape[2]
    if VERIFYING[0] or seqlen <= ATTENTION_PIECE or _SPARSE_ATTENTION[0] is None:
        return _SPARSE_ATTENTION[0](self, q, window, pool, indices, window_mask)
    outs = []
    for a, b in _pieces(seqlen):
        wmask = window_mask if window_mask is None or window_mask.shape[-2] != seqlen else window_mask[..., a:b, :]
        part = _SPARSE_ATTENTION[0](self, q[:, :, a:b], window, pool, indices[:, a:b], wmask)
        mx.eval(part)
        outs.append(part)
    return mx.concatenate(outs, axis=2)


def _window_attention(self, cache, x):
    """Non-compressed `DeepseekV41Attention._attention` evaluated in `ATTENTION_PIECE`-query slices on long prefill passes."""
    start_pos = cache.offset
    batch, seqlen = x.shape[0], x.shape[1]
    qr = self.q_norm(self.wq_a(fake_quant_fp8_ue8m0(x)))
    q = self.wq_b(fake_quant_fp8_ue8m0(qr)).reshape(batch, seqlen, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
    q = self.rope(q, start_pos)
    window_kv, window_mask = self._window_part(x, start_pos, cache)
    kv = window_kv[:, None].astype(q.dtype)
    sink = self.attn_sink.astype(q.dtype)
    outs = []
    for a, b in _pieces(seqlen):
        wmask = window_mask if window_mask is None or window_mask.shape[-2] != seqlen else window_mask[..., a:b, :]
        part = mx.fast.scaled_dot_product_attention(q[:, :, a:b], kv, kv, scale=self.scale, mask=wmask, sinks=sink)
        mx.eval(part)
        outs.append(part)
    out = self.rope(mx.concatenate(outs, axis=2), start_pos, inverse=True)
    out = out.reshape(batch, self.o_groups, -1, seqlen, self.head_dim)
    out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
    out = self.wo_a(out)
    out = out.transpose(0, 2, 1, 3).flatten(-2)
    return self.wo_b(fake_quant_fp8_ue8m0(out))


def _attention(self, cache, x):
    """mlx-vlm's DeepseekV41Attention._attention; in a verify pass each query attends as a one-token
    decode at its position does: its own window of the pass's keys (the decode's, in its order),
    the compressed positions its indexer picked for it, so many and no -1 among them, and the
    window alone where it had none or the layer compresses nothing -- MLX's products of the pass's
    queries at once are not one query's bits (1.3e-5), nor its fused attention for a few queries
    at every width (a tiny model's heads of 32).  Nor are they in the batch (unlike the indexer's,
    _grouped): a query's window product, [64, 512] by [512, 128], is not one query's in a batch of
    2 to 6 at V4.1's shapes, so the queries go one at a time.  The ring, the
    pool and the hand-offs are written as the pass writes them."""
    if not _verifying(x):
        if not self.compress_ratio and x.shape[1] > ATTENTION_PIECE:
            return _window_attention(self, cache, x)
        if x.shape[1] <= ATTENTION_PIECE and type(self)._window_part is _window_part:
            from .. import v41_attention
            y = v41_attention.attend(self, cache, x, _window_indices)
            if y is not None:
                return y
        return _ATTENTION[0](self, cache, x)
    start_pos = cache.offset
    batch, seqlen = x.shape[0], x.shape[1]
    qr = self.q_norm(self.wq_a(fake_quant_fp8_ue8m0(x)))
    q = self.wq_b(fake_quant_fp8_ue8m0(qr)).reshape(batch, seqlen, self.n_heads, self.head_dim)
    q = self.rope(q.transpose(0, 2, 1, 3), start_pos)
    window_kv, _ = self._window_part(x, start_pos, cache)
    pool, idxs = self._compress_part(x, qr, start_pos, cache) if self.compress_ratio else (None, None)
    picked = getattr(cache, '_quanteeze_picked', None)
    base, win, outs = max(0, start_pos - self.window_size + 1), self.window_size, []
    for s in range(seqlen):
        at = start_pos + s
        a, b = max(0, at - win + 1) - base, at - base + 1
        local, valid = window_kv[:, a:b], mx.ones((1, 1, 1, b - a), dtype=mx.bool_)
        q_s = mx.contiguous(q[:, :, s:s + 1])
        if self.compress_ratio and (at + 1) // self.compress_ratio and idxs.shape[-1] and picked is not None:
            outs.append(self._sparse_attention(q_s, local, pool, idxs[:, s:s + 1, :picked[s]], valid))
        else:
            kv = local[:, None].astype(q.dtype)
            outs.append(mx.fast.scaled_dot_product_attention(q_s, kv, kv, scale=self.scale, mask=valid,
                                                             sinks=self.attn_sink.astype(q.dtype)))
    out = self.rope(mx.concatenate(outs, axis=2), start_pos, inverse=True)
    out = out.reshape(batch, self.o_groups, -1, seqlen, self.head_dim)
    out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
    out = self.wo_a(out)
    out = out.transpose(0, 2, 1, 3).flatten(-2)
    return self.wo_b(fake_quant_fp8_ue8m0(out))


VERIFYING = per_thread.Field('exact')  # within a verify pass (SpecLanguageModel, speculative_verify=True)


class ExactLinear(nn.QuantizedLinear):
    """A projection of the target's in a microscaling format (MXFP8: the attention's, the shared
    experts', the Engram's and the compressors' and indexers') whose tokens in a verify pass of 2 to
    MAX_TOKENS come out as one-token decodes give them (verify_qmv.verify_linear: MLX's own
    one-vector kernel, the weights read once); any other pass -- a prompt's, whose chunks then
    compute as they always did -- or a shape no port matches, as it was.  MLX's kernel for a few
    rows sums otherwise: V4.1's verify pass parted from one-token decodes at layer 0's wq_b, its
    logits up to 2.8 apart and a greedy token in four another."""

    def __call__(self, x):
        from .. import verify_qmv
        out = (verify_qmv.verify_linear(self, x) if VERIFYING[0] and x.ndim == 3
               and 1 < x.shape[1] <= verify_qmv.MAX_TOKENS else None)
        return super().__call__(x) if out is None else out


def _exact_multi():
    """The class wo_a (mlx-vlm's QuantizedMultiLinear: each output group its own weights) takes:
    a verify pass's tokens through verify_qmv.verify_multi, each group as one-token calls compute
    it -- with the attention's core exact, wo_a was the next to part (layer 7, 2e-3; the Mac)
    -- else as it was."""
    from mlx_vlm.models.mla import QuantizedMultiLinear

    class ExactMultiLinear(QuantizedMultiLinear):
        def __call__(self, x, transpose=True):
            from .. import verify_qmv
            out = verify_qmv.verify_multi(self, x) if VERIFYING[0] and transpose else None
            return super().__call__(x, transpose) if out is None else out
    return ExactMultiLinear


class RowLinear(nn.Linear):
    """A plain projection of the target's (the compressors' wkv and wgate, the indexers' wk and
    weights_proj: no scales in the checkpoint) whose verify pass of 2 to ROWS tokens multiplies a
    row at a time, as a decode does (by_rows): with the projections in MXFP8 exact, the
    compressors' were the first to part (layer 2's wkv, 2.8e-7).  A prompt's
    chunks as they always did: the Engram, called on the whole prompt where a prefill goes a layer
    at a time (quanteeze.prefill), would otherwise compute its chunks otherwise."""

    def __call__(self, x):
        n = x.shape[-2] if x.ndim >= 2 else 1
        if not (VERIFYING[0] and 1 < n <= ROWS):
            return super().__call__(x)
        return mx.concatenate([super(RowLinear, self).__call__(x[..., i:i + 1, :]) for i in range(n)], axis=-2)


_EXACT_MULTI = [None]


def exact_projections(model):
    """The target's own projections (not the MTP head's, nor the drafter's, which is another model)
    a short pass exact: those in a microscaling format ExactLinear, the plain ones RowLinear;
    returns how many."""
    from ..verify_qmv import FP_MODES
    count = 0
    for name, module in model.named_modules():
        parts = name.split('.')
        if 'layers' not in parts or 'mtp' in parts:
            continue
        if type(module) is nn.QuantizedLinear and getattr(module, 'mode', 'affine') in FP_MODES:
            module.__class__ = ExactLinear
        elif type(module).__name__ == 'QuantizedMultiLinear' and getattr(module, 'mode', 'affine') in FP_MODES:
            if _EXACT_MULTI[0] is None:
                _EXACT_MULTI[0] = _exact_multi()
            module.__class__ = _EXACT_MULTI[0]
        elif type(module) is nn.Linear:
            module.__class__ = RowLinear
        else:
            continue
        count += 1
    return count


def _before_moe(block, h, pre_mix, image_mask, cache):
    from mlx_vlm.models.deepseek_v41 import language
    residual = h
    attn_pre, attn_post, attn_comb = block.hc_mixes(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    x = block.hc_pre(h, pre_mix)
    x = block.attn_norm(x)
    x = block.attn(x, cache)
    x = language.hc_expand(x, residual, attn_post, attn_comb)
    residual = x
    ffn_pre, ffn_post, ffn_comb = block.hc_mixes(x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
    x = block.hc_pre(x, attn_pre)
    x = block.ffn_norm(x)
    return x, residual, ffn_pre, ffn_post, ffn_comb, image_mask


def _from_moe(block, x, residual, ffn_pre, ffn_post, ffn_comb, image_mask):
    from mlx_vlm.models.deepseek_v41 import language
    x = block.ffn(x, image_mask)
    x = language.hc_expand(x, residual, ffn_post, ffn_comb)
    return x, ffn_pre


def install(model):
    """Give each MoE block of `model` its router RowGate, the blocks'
    hyper-connections hc_mix_rows (or `hc_mix_kernel` on CUDA), the compression sources' rotation no host sync
    (_rope_at_positions), the window its indices once a pass (_window_part), and the projections
    in a microscaling format a short pass exact (exact_projections), a verify pass's experts a slot
    a threadgroup (EXPERT_ROWS); returns {layer index: block}.
    The MTP head's blocks are left alone."""
    from mlx_vlm.models.deepseek_v41 import language

    from .. import cuda_engram, cuda_kernels, cuda_v41, expert_qmv, prefill
    prefill.split(DeepseekV41Block, _before_moe, _from_moe)
    v41_blocks = [m for n, m in model.named_modules() if type(m) is DeepseekV41Block and 'mtp' not in n.split('.')]
    lm = getattr(model, 'language_model', model)
    split_in_one = cuda_v41.choose(v41_blocks, lm.config) if hasattr(lm, 'config') else False
    cuda_v41.install_fp8()
    cuda_v41.install_prepost()
    cuda_v41.install_attention()
    if cuda_v41._gpu():
        cuda_engram.install()
    language.hc_mix_coeffs = hc_mix_kernel if split_in_one else hc_mix_rows   # DeepseekV41Block.hc_mixes reads it by name
    expert_qmv.ROWS[0] = EXPERT_ROWS
    if _COMPRESS_PART[0] is None:
        _COMPRESS_PART[0] = language.DeepseekV41Attention._compress_part
        _ROPE_AT_POSITIONS[0] = language._apply_rope_at_positions
        _WINDOW_PART[0] = language.DeepseekV41Attention._window_part
        language.DeepseekV41Attention._compress_part = _compress_part
        language._apply_rope_at_positions = _rope_at_positions
        language.DeepseekV41Attention._window_part = _window_part
    if _SPARSE_ATTENTION[0] is None:
        _SPARSE_ATTENTION[0] = language.DeepseekV41Attention._sparse_attention
        language.DeepseekV41Attention._sparse_attention = _sparse_attention
    if _ATTENTION[0] is None:                             # a verify pass's queries each as a decode's
        _ATTENTION[0], _INDEXER[0] = language.DeepseekV41Attention._attention, language.Indexer.__call__
        language.DeepseekV41Attention._attention = _attention
        language.Indexer.__call__ = _indexer
    blocks = {}
    for name, module in model.named_modules():
        parts = name.split('.')
        if type(module) is DeepseekV41MoE and 'layers' in parts and 'mtp' not in parts:
            layer = int(parts[parts.index('layers') + 1])
            if type(module.gate) is DeepseekV41MoEGate:
                module.gate.__class__ = RowGate
            object.__setattr__(module, '_layer_idx', layer)
            expert_qmv.install_switch(module.switch_mlp)
            blocks[layer] = module
    cuda_v41.install_route([b.gate for b in blocks.values() if type(b.gate) is RowGate])
    exact_projections(model)
    return blocks


class SpecCompressor(Compressor):
    """A pooling compressor (ratio 2) that keeps what it was given during a verify pass, for
    SpecLanguageModel.rollback_speculative_cache to put its partial group back as the accepted
    tokens left it."""

    def __call__(self, x, start_pos, cache):
        kept = getattr(cache, '_quanteeze_spec', None)
        if kept is not None and self.compress_ratio > 1:
            kept[self.layer_idx] = x
        return super().__call__(x, start_pos, cache)


def _host_ids(engram):
    """Whether `engram`'s offloaded embedding accepts host `np.ndarray` row IDs (`QUANTEEZE_ENGRAM_IDS_ONCE`)."""
    from .. import cuda_engram, cuda_ple, cuda_v41, variables
    embed = getattr(engram, 'embed', None)
    if getattr(embed, '_arrays', None) is None or variables.off('QUANTEEZE_ENGRAM_IDS_ONCE'):
        return False
    if cuda_ple.ON[0] and cuda_v41._gpu() and cuda_engram._table(embed) is not None:
        return False
    return True


def _one_row(entry, seqlen):
    """Single unpadded cache row backing a pass of `seqlen` tokens, or None if batched/padded."""
    rows = getattr(entry, 'rows', None)
    if rows is None:
        return entry
    if len(rows) != 1:
        return None
    bounds = getattr(entry, '_bounds', None)
    return rows[0] if bounds is not None and tuple(bounds(0, seqlen)) == (0, seqlen) else None


def _engram_ahead(lm, hashes, now, ids=None):
    """Prefetch the next Engram layer's pages (`QUANTEEZE_ENGRAM_READ_AHEAD`) while intermediate layers execute."""
    from .. import ngram_prefetch, variables
    if variables.off('QUANTEEZE_ENGRAM_READ_AHEAD'):
        return
    for j in range(now + 1, len(lm.layers)):
        nxt = getattr(lm.layers[j], 'engram', None)
        arrays = getattr(getattr(nxt, 'embed', None), '_arrays', None)
        if isinstance(arrays, dict):
            if any(isinstance(a, np.memmap) for a in arrays.values()):
                row_ids = ids[:, :, nxt.layer_hash_index, :] if ids is not None else np.asarray(hashes[:, :, nxt.layer_hash_index, :])
                ngram_prefetch.ahead(nxt.embed, row_ids)
            return


class SpecLanguageModel(LanguageModel):
    """V4.1's language model as mlx-vlm's DFlash round loop drives a target (quanteeze.dspark):
    capture_layer_ids gives what enters those layers, averaged over the hc copies, as DeepSeek's
    DSpark reads them (inference/model.py: h.mean(dim=2) before layers 37-39); a prompt's, only
    its last CONTEXT positions (the drafter keeps no more: its window), so a long prompt holds
    no 30 KB a token; speculative_verify=True keeps what a rejected draft overwrites, and
    rollback_speculative_cache() puts it back: each layer's window slots, its pooling group's
    partial state (replayed from the accepted tokens), and its compressed KV and index keys cut
    to the accepted length.  The Engram's token history is read only below the offset, and the
    hand-offs are published afresh each pass, so neither is put back.  Without either argument
    it is LanguageModel's call; with images not served (no_vision()) it makes no image mask: the
    routers' text bias and every token in the Engram's n-grams, as an all-False mask gives, bit
    for bit, without its comparison and selections in each layer."""

    def __call__(self, input_ids=None, inputs_embeds=None, cache=None, image_mask=None, engram_hashes=None,
                 inputs=None, n_to_process=None, capture_layer_ids=None, speculative_verify=False, **kwargs):
        per_thread.PASS.entries[id(self)] = weakref.ref(cache[0]) if cache else None
        entry = cache[0] if cache else DeepseekV41Cache(len(self.layers))
        if input_ids is None:
            input_ids = inputs
        image_from_ids = image_mask is None and input_ids is not None and not getattr(self, '_quanteeze_text', False)
        if image_from_ids:
            image_mask = input_ids == self.config.image_token_id
        engram_mask = None if image_mask is None else ~image_mask
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, seqlen = h.shape[0], h.shape[1]
        state = self._snapshot(entry, seqlen) if speculative_verify else None
        h = mx.broadcast_to(h[..., None, :], (batch, seqlen, self.config.hc_mult, self.config.hidden_size))
        host_hashes = None
        if engram_hashes is None and input_ids is not None:
            self._ensure_engram_hash()
            if self.engram_hash is not None:
                own = _one_row(entry, seqlen)
                before = getattr(own, 'engram', None) if own is not None else None
                engram_hashes = entry.map(lambda row, ids, mask: self.engram_hash(ids, row.offset, row, token_mask=mask),
                                          input_ids, engram_mask)
                if own is not None and (image_mask is None or image_from_ids):
                    from .. import engram_host
                    host_hashes = engram_host.pass_hashes(
                        self.engram_hash, own, input_ids, before, engram_hashes,
                        self.config.image_token_id if image_from_ids else None,
                    )
        capture = set(capture_layer_ids or ())
        captured = []
        pre_mix = make_identity_pre_mix(batch, seqlen, self.config.hc_mult)
        per_thread.PASS.verify = bool(speculative_verify)
        VERIFYING[0] = bool(speculative_verify) and EXACT[0]    # its projections as decodes (ExactLinear)
        one_decode = seqlen == 1 and not speculative_verify
        host_engram_ids = host_hashes
        if one_decode and host_engram_ids is not None:
            _engram_ahead(self, engram_hashes, -1, host_engram_ids)
        plain = PLAIN_EVAL[0] if seqlen <= ROWS and mx.metal.is_available() else 0
        last = len(self.layers) - 1
        try:
            for i, layer in enumerate(self.layers):
                if layer.engram is not None and (engram_hashes is not None or host_engram_ids is not None):
                    idx = layer.engram.layer_hash_index
                    if (host_engram_ids is not None or one_decode) and _host_ids(layer.engram):
                        if host_engram_ids is None:
                            host_engram_ids = np.asarray(engram_hashes)
                        h = layer.engram(h, host_engram_ids[:, :, idx, :], engram_mask)
                    else:
                        h = layer.engram(h, engram_hashes[:, :, idx, :], engram_mask)
                    if one_decode:
                        _engram_ahead(self, engram_hashes, i, host_engram_ids)
                if i in capture:
                    tail = h if speculative_verify else h[:, -CONTEXT:]
                    captured.append(tail.astype(mx.float32).mean(axis=2).astype(h.dtype))
                h, pre_mix = layer(h, pre_mix, image_mask, entry)
                if plain and (i + 1) % plain == 0 and i < last:
                    mx.async_eval(h, pre_mix)                 # the GPU on what is built (PLAIN_EVAL)
        finally:
            VERIFYING[0] = per_thread.PASS.verify = False
            for row, _ in state or ():
                row._quanteeze_spec = None
        h = DeepseekV41Block.hc_pre(h, pre_mix)
        keep = kwargs.get('logits_to_keep')
        if isinstance(self.head, BF16Head) and isinstance(keep, int) and keep > 0 and not speculative_verify:
            logits = self.head(self.norm(h), keep=keep)
        else:
            logits = self.head(self.norm(h))
        entry.advance(seqlen)
        return LanguageModelOutput(logits=logits, hidden_states=captured or None, gdn_states=state)

    def chunked_prefill_policy(self, *, draft_kind=None, **kwargs):
        """mlx-vlm prefills a prompt in chunks (the prefill step) only where the model says it may
        with a drafter: none said so for V4.1, so with its DSpark head drafting (MTP Auto), since
        a prompt went through in one pass, and one of 26,378 tokens took MLX from 85.7 to
        173 GB and ran out of memory (8,000 tokens: 110, the Mac).  DSpark's rounds (draft kind
        'dflash') take the context the chunks captured, joined (mlx-vlm's SpeculativePrefill): each
        chunk's last CONTEXT positions, just before the next chunk's, so the last CONTEXT of them --
        all the drafter reads -- are the prompt's last, as a pass of the whole would capture them."""
        return draft_kind in (None, 'dflash')

    def entry(self):
        """The cache entry of the last pass, while it lives (a drafter reads the offset there)."""
        ref = per_thread.PASS.entries.get(id(self))
        return None if ref is None else ref()

    @staticmethod
    def _snapshot(entry, width):
        """(row, what a verify pass of `width` tokens overwrites) for each row of the cache."""
        rows = entry.rows if hasattr(entry, 'rows') else [entry]
        state = []
        for row in rows:
            start = row.offset
            slots = None
            window = []
            for w in row.window:
                if w is not None and slots is None:
                    slots = (start + mx.arange(width)) % w.shape[1]
                window.append(None if w is None else mx.take(w, slots, axis=1))
            row._quanteeze_spec = {}
            state.append((row, {'start': start, 'width': width, 'slots': slots, 'window': window,
                                'kv_state': list(row.kv_state), 'score_state': list(row.score_state),
                                'x': row._quanteeze_spec}))
        return state

    def rollback_speculative_cache(self, caches, state, accepted, block_size):
        """Each row back to its accepted tokens and the bonus: accepted + 1 of the pass."""
        if isinstance(accepted, mx.array):
            accepted = accepted.reshape(-1).tolist()
        elif isinstance(accepted, int):
            accepted = [accepted]
        accepted = list(accepted)
        if len(accepted) == 1:
            accepted = accepted * len(state)
        for (row, kept), a in zip(state, accepted):
            self._rewind(row, kept, int(a) + 1)

    def _rewind(self, row, kept, keep):
        start, width = kept['start'], kept['width']
        if keep >= width:
            return
        row.offset = start + keep
        if kept['slots'] is not None:
            slots = kept['slots'][keep:]
            for w, saved in zip(row.window, kept['window']):
                if saved is not None:
                    w[:, slots] = saved[:, keep:]
        ratios = self.config.compress_ratios
        for i, ratio in enumerate(ratios):
            if not ratio:
                continue
            n = (start + keep) // ratio
            for arrays in (row.compress, row.keys):
                if arrays[i] is not None and arrays[i].shape[1] > n:
                    arrays[i] = arrays[i][:, :n]
        for i, x in kept['x'].items():
            compressor = self.layers[i].attn.compressor
            ratio, batch = compressor.compress_ratio, x.shape[0]
            row.kv_state[i], row.score_state[i] = kept['kv_state'][i], kept['score_state'][i]
            compressor._grow_state(batch, row)
            xf = x[:, :keep].astype(mx.float32)
            kv, score = compressor.wkv(xf), compressor.wgate(xf)
            for n in range(keep):
                slot = (start + n) % ratio
                row.kv_state[i] = compressor._write_slot(row.kv_state[i], batch, slot, kv[:, n:n + 1])
                row.score_state[i] = compressor._write_slot(row.score_state[i], batch, slot, score[:, n:n + 1])


CONTEXT = 128                   # a prompt's positions SpecLanguageModel captures, at its end: the drafter's window


def speculative(model):
    """Give V4.1's language model and its pooling compressors the classes above (once)."""
    lm = getattr(model, 'language_model', model)
    if type(lm) is LanguageModel:
        lm.__class__ = SpecLanguageModel
    for layer in lm.layers:
        compressor = getattr(getattr(layer, 'attn', None), 'compressor', None)
        if type(compressor) is Compressor:
            compressor.__class__ = SpecCompressor
    return lm
