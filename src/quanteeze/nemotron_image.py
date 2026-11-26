# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Nemotron-Labs-Image-30B-A3B drawn with MLX: mlx-lm's generation of the converted model (e.g., 4 bits), the
model card's prompt, template and sampling, then the pixel decoder written in MLX
(quanteeze.vendor.nemotron_image.l24sq).  Not served by Quanteeze (its experts are two projections, not
SwiGLU's three: arch.is_experts finds none); used by the MCP tool (quanteeze.image_mcp).

The model, its decoder (src/quanteeze/vendor/nemotron_image/, derived from NVIDIA's code) and generated outputs
are under the NVIDIA License: for research or evaluation only, not commercially (the vendor folder's LICENSE and
NOTICE.md).  No model weights are distributed in this repository.

The model writes <|img_start|>, a header ("height=H width=W"), then H/16 rows of W/16 image tokens each after a
<|vis_row|>, and <|img_end|>; an image token is 131075 + its code in the decoder's codebook.  A short or missing row
is filled with code 0 (how it broke shows in the picture), and told.

draw() loads the model and the decoder for the one image and lets them go after (room() checks for their size and
ROOM_SPARE_GB free first).
"""
import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SYSTEM = ("Generate an image matching the requested resolution by detailing the color, quantity, text, shape, "
          "size, texture, spatial relationships of the objects and background.")
PATCH = 16
TEMPERATURE = 0.9
SIZES = range(512, 1537, 64)            # sides offered (1024 x 1024 verified)
HEADER_TOKENS = 192                     # what comes before the grid and after it, at most, besides its rows
ROOM_SPARE_GB = 6.0
MANIFEST = 'unitron_extension_manifest.json'
DECODER = 'Nemotron-Labs-Image-Tokenizer'


def max_tokens(width, height):
    """Tokens enough for an image: each row's tokens and its <|vis_row|>, and HEADER_TOKENS (4,352 at 1024 x 1024
    with the card's own 4,352)."""
    return (height // PATCH) * (width // PATCH + 1) + HEADER_TOKENS


def check_size(width, height):
    for side in (width, height):
        if int(side) not in SIZES:
            raise ValueError(f'a side is one of {SIZES.start}..{SIZES.stop - 1} in steps of {SIZES.step}, not {side}')


def find(roots):
    """(the converted model, the decoder's folder) among the model folders `roots` (<root>/<org>/<name>): an MLX
    checkpoint with the manifest and a quantization, and the original's Nemotron-Labs-Image-Tokenizer -- beside the
    converted model where it was copied there, else in a checkpoint of the original weights."""
    models, decoders = [], []
    for root in roots:
        for d in sorted(Path(root).glob('*/Nemotron-Labs-Image*')):
            try:
                config = json.loads((d / 'config.json').read_text())
            except (OSError, ValueError):
                continue
            if (d / MANIFEST).is_file() and 'quantization' in config:
                models.append(d)
            if (d / DECODER / 'model.safetensors').is_file():
                decoders.append(d / DECODER)
    if not models:
        raise ValueError('no converted Nemotron-Labs-Image checkpoint (MLX, with its manifest) in '
                         + ', '.join(map(str, roots)))
    if not decoders:
        raise ValueError(f'no {DECODER} folder (the original repository\'s) in ' + ', '.join(map(str, roots)))
    model = models[0]
    near = [d for d in decoders if d.parent == model]
    return model, (near or decoders)[0]


def _gb(folder):
    return sum(p.stat().st_size for p in Path(folder).glob('*.safetensors')) / 1e9


def available_gb():
    """GB the machine may give a new process now: free, inactive, speculative and purgeable pages (a Mac), or
    MemAvailable (Linux); None where it cannot be told."""
    try:
        if sys.platform == 'darwin':
            out = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r'page size of (\d+) bytes', out).group(1))
            pages = 0
            for name in ('free', 'inactive', 'speculative', 'purgeable'):
                m = re.search(rf'Pages {name}:\s+(\d+)\.', out)
                pages += int(m.group(1)) if m else 0
            return pages * page / 1e9
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1e6
    except Exception:                                   # noqa: BLE001 -- not checked then
        return None
    return None


def room(model, decoder):
    """None where the machine has the room for the model and the decoder (and ROOM_SPARE_GB), else why not."""
    need = _gb(model) + _gb(decoder) + ROOM_SPARE_GB
    have = available_gb()
    if have is not None and have < need:
        return (f'{have:.0f} GB of memory free, {need:.0f} GB wanted (the model {_gb(model):.0f}, the decoder '
                f'{_gb(decoder):.1f}, {ROOM_SPARE_GB:g} spare): unload what holds it, or draw later')
    return None


def prompt_ids(tokenizer, prompt, width, height):
    """The card's prompt: its system text, "Generate a WxH image." and the user's words, thinking off, no BOS."""
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': f'Generate a {width}x{height} image.\n\n{prompt}'}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def grid(tokens, ranges, width, height):
    """(codes [gh, gw], what it says): the image tokens between <|img_start|> and <|img_end|>, row by row, a short or
    missing row filled with code 0; its header's size where it gives one.  `ranges`: the manifest's
    unitron_token_ranges."""
    start, end, row = (ranges['framing'][k] for k in ('<|img_start|>', '<|img_end|>', '<|vis_row|>'))
    v0, v1 = ranges['vision']['start_id'], ranges['vision']['end_id']
    if start not in tokens:
        raise ValueError('the model began no image (no <|img_start|>)')
    i0 = tokens.index(start)
    i1 = tokens.index(end) if end in tokens else len(tokens)
    head, rows, cur, stray = [], [], None, 0
    for x in tokens[i0 + 1:i1]:
        if x == row:
            if cur is not None:
                rows.append(cur)
            cur = []
        elif v0 <= x < v1:
            if cur is None:
                cur = []
            cur.append(x - v0)
        elif cur is None:
            head.append(x)
        else:
            stray += 1
    if cur is not None:
        rows.append(cur)
    gh, gw = height // PATCH, width // PATCH
    codes = np.zeros((gh, gw), dtype=np.int64)
    for r, got in enumerate(rows[:gh]):
        codes[r, :min(gw, len(got))] = got[:gw]
    lengths = sorted({len(r) for r in rows})
    whole = len(rows) == gh and lengths == [gw] and end in tokens and not stray
    said = {'rows': len(rows), 'lengths': lengths, 'closed': end in tokens, 'stray': stray, 'whole': whole,
            'head': head}
    return codes, said


def draw(model_dir, decoder_dir, prompt, width=1024, height=1024, seed=None, out=None, keep=('png',)):
    """One image: {'png' (its path where kept), 'seed', 'tokens', 'seconds', 'decode_tok_s', 'rows', 'lengths',
    'closed', 'whole', 'header'}.  out: the stem written to (<out>.png, and .ids.npy and .codes.npz where `keep`
    names them).  The model and decoder are let go after."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load_model
    from PIL import Image
    from transformers import PreTrainedTokenizerFast

    try:
        from .vendor.nemotron_image.l24sq import Tokenizer      # NVIDIA License: non-commercial (its NOTICE.md)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            'quanteeze.vendor.nemotron_image.l24sq is not installed '
            '(optional third-party decoder under the NVIDIA License: research or evaluation only)'
        ) from e

    check_size(width, height)
    folder = Path(model_dir)
    ranges = json.loads((folder / MANIFEST).read_text())['unitron_token_ranges']
    end, v1 = ranges['framing']['<|img_end|>'], ranges['vision']['end_id']
    tok = PreTrainedTokenizerFast(tokenizer_file=str(folder / 'tokenizer.json'))
    tok.chat_template = (folder / 'chat_template.jinja').read_text()
    ids = prompt_ids(tok, prompt, width, height)
    seed = int(seed) if seed is not None else int(time.time() * 1000) % (2 ** 31)
    t = time.perf_counter()
    model, _ = load_model(folder)
    loaded = time.perf_counter() - t
    try:
        pad = mx.arange(model.args.vocab_size) >= v1              # rows past the vision range: padding

        def no_padding(tokens, logits):
            return mx.where(pad, -mx.inf, logits)

        mx.random.seed(seed)
        tokens, t0, first = [], time.perf_counter(), None
        for token, _ in generate_step(mx.array(ids), model, max_tokens=max_tokens(width, height),
                                      sampler=make_sampler(temp=TEMPERATURE, top_p=1.0),
                                      logits_processors=[no_padding]):
            token = int(token)
            first = first if first is not None else time.perf_counter() - t0
            tokens.append(token)
            if token == end:
                break
        seconds = time.perf_counter() - t0
    finally:
        del model
        gc.collect()
        mx.clear_cache()
    codes, said = grid(tokens, ranges, width, height)
    image = Tokenizer(str(decoder_dir)).decode(codes.reshape(-1), codes.shape)
    gc.collect()
    mx.clear_cache()
    result = {'seed': seed, 'tokens': len(tokens), 'seconds': round(seconds, 1), 'load_seconds': round(loaded, 1),
              'decode_tok_s': round((len(tokens) - 1) / max(seconds - (first or 0), 1e-9), 1),
              'header': tok.decode(said.pop('head')), **said, 'png': None}
    if out is not None:
        out = str(out)
        if 'ids' in keep:
            np.save(out + '.ids.npy', np.array(tokens, dtype=np.int64))
        if 'codes' in keep:
            np.savez(out + '.codes.npz', codes=codes.reshape(-1), grid=np.array(codes.shape))
        if 'png' in keep:
            Image.fromarray(image).save(out + '.png')
            result['png'] = out + '.png'
    return result
