# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Image models mlx-vlm draws with its own pipelines (mlx_vlm.generate.image): Qwen-Image-2.1, Z-Image-Turbo and
FLUX.2-klein-4B, each at its own defaults (steps, guidance).  Not served by Quanteeze; the MCP tool
(quanteeze.image_mcp) offers them beside Nemotron-Labs-Image.

A checkpoint is a diffusers tree (model_index.json, transformer/, text_encoder/, vae/) in a model folder
(<root>/<org>/<name>), found by its name (MODELS[...].names).  Their licenses differ (MODELS[...].license):
Z-Image-Turbo and FLUX.2-klein-4B are Apache-2.0, commercial use allowed; Qwen-Image-2.1 is Qwen's Research
License, research or evaluation only.  (mlx-community's FLUX.2 4- to 8-bit copies are mflux's single-file layout,
which mlx-vlm does not load; FLUX.2-klein-9B and FLUX-3 klein are not under Apache-2.0, and are not offered.)

draw() loads the pipeline for the one image and lets it go after, where the memory has room (room()).
"""
import gc
import time
from dataclasses import dataclass
from pathlib import Path

from . import nemotron_image

PARTS = ('transformer', 'text_encoder', 'vae')
SIZES = range(256, 2049, 16)            # sides offered, multiples of 16 (mlx-vlm's FLUX.2 and Z-Image ask so)


@dataclass(frozen=True)
class Model:
    name: str                           # the tool's name of it
    names: tuple                        # folder names it is found by (fnmatch, case as on the disk)
    about: str                          # what the tool tells the agent
    license: str
    commercial: bool


MODELS = {m.name: m for m in (
    Model('z-image-turbo', ('Z-Image-Turbo*',),
          'Z-Image-Turbo (Tongyi-MAI, 6B): photographs and illustrations, English and Chinese text, 9 steps, about 12 s',
          'Apache-2.0', True),
    Model('flux2-klein', ('FLUX.2-klein-4B', 'FLUX.2-klein-4B-bf16'),
          'FLUX.2-klein-4B (Black Forest Labs): photographs and illustrations, 4 steps, fast',
          'Apache-2.0', True),
    Model('qwen-image', ('Qwen-Image-2.1*',),
          'Qwen-Image-2.1: good at text in the picture and at layouts, 30 steps',
          'Qwen Research License: research or evaluation only', False),
)}


def check_size(width, height):
    for side in (width, height):
        if int(side) not in SIZES:
            raise ValueError(f'a side is one of {SIZES.start}..{SIZES.stop - 1} in steps of {SIZES.step}, not {side}')


def find(name, roots):
    """The first checkpoint of MODELS[name] in the model folders `roots`: a diffusers tree mlx-vlm loads."""
    model = MODELS[name]
    for root in roots:
        for pattern in model.names:
            for d in sorted(Path(root).glob(f'*/{pattern}')):
                if (d / 'model_index.json').is_file() and all((d / p).is_dir() for p in PARTS):
                    return d
    raise ValueError(f'no {name} checkpoint ({" or ".join(model.names)}: model_index.json, transformer/, '
                     f'text_encoder/, vae/) in ' + ', '.join(map(str, roots)))


def gb(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*.safetensors')) / 1e9


def room(path):
    """None where the machine has the room for the pipeline (and nemotron_image.ROOM_SPARE_GB), else why not."""
    need = gb(path) + nemotron_image.ROOM_SPARE_GB
    have = nemotron_image.available_gb()
    if have is not None and have < need:
        return (f'{have:.0f} GB of memory free, {need:.0f} GB wanted (the pipeline {gb(path):.0f}, '
                f'{nemotron_image.ROOM_SPARE_GB:g} spare): unload what holds it, or draw later')
    return None


def draw(path, prompt, width=1024, height=1024, seed=None, out=None, steps=None, negative_prompt=None):
    """One image: {'png' (its path where `out`, a stem, is given), 'seed', 'steps', 'seconds', 'load_seconds'}."""
    import mlx.core as mx
    from mlx_vlm.generate.image import ImageGenerationRequest, generate_image, load_image_generation_model
    check_size(width, height)
    seed = int(seed) if seed is not None else int(time.time() * 1000) % (2 ** 31)
    t = time.perf_counter()
    pipeline = load_image_generation_model(str(path))
    loaded = time.perf_counter() - t
    try:
        t0 = time.perf_counter()
        extra = {'negative_prompt': negative_prompt} if negative_prompt else {}
        result = generate_image(pipeline, ImageGenerationRequest(prompt=prompt, seed=seed, steps=steps, width=width,
                                                                 height=height, extra=extra))
        image = result.to_pil()
        done = {'seed': int(result.seed), 'steps': int(result.steps), 'seconds': round(time.perf_counter() - t0, 1),
                'load_seconds': round(loaded, 1), 'png': None}
    finally:
        del pipeline
        gc.collect()
        mx.clear_cache()
    if out is not None:
        image.save(str(out) + '.png')
        done['png'] = str(out) + '.png'
    return done
