# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""The home page, /quanteeze/ (beside the detailed settings page, /quanteeze/settings): pick a
model, choose its uniform expert precision (original, or 8, 4, 3, 2, or 1 bit by least squares) and
how it is held in memory or streamed on demand from SSD, configure memory/PLE/MTP/heat presets,
monitor live server status, and download models from Hugging Face once the user agrees to their size.

- The catalog (`catalog()`): the checkpoints on this machine (LM Studio's models, Hugging Face's
  cache, beside the model served), each with its size, routed expert count, original expert GB,
  uniform least-squares GB at 1-8 bits (`uniform_gb()`), PLE / Engram SSD status, and how many GB
  `fitting.room` leaves for the experts here (`room_gb()`).
- A start (`start()`): sets `QUANTEEZE_BITS` (`''` for checkpoint precision, or `1`, `2`, `3`, `4`,
  `8` for uniform least-squares quantization) and `QUANTEEZE_FROM_SSD_GB`, prepares or selects the
  PLE SSD view as `QUANTEEZE_NGRAM_SSD` asks, and switches the served model through
  `quanteeze.control.switch_model`.
- The presets (`PRESETS`): memory (`QUANTEEZE_MEMORY_RESERVE`), n-gram PLE table
  (`QUANTEEZE_NGRAM_SSD`), MTP speculative decoding (`QUANTEEZE_MTP`, `QUANTEEZE_MTP_EXACT`), and
  thermal throttling (`QUANTEEZE_THROTTLE_AT`, `QUANTEEZE_THROTTLE_DUTY`).
- A download (`download()`): fetches a Hugging Face repository into the local cache on a background
  thread after `info()` reports its file count and size.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path

import numpy as np

from . import arch, control, fitting, guards, mtp, q2_experts, variables

PAGE_PATH, API, ROUTES = control.HOME_PATH, control.HOME_API, control.HOME_ROUTES
REPO = re.compile(r'^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$')
EXPERTS = ('num_experts', 'n_routed_experts', 'num_local_experts')
UNIFORM_BITS = (1, 2, 3, 4, 8)
UNIFORM_GGML = ('q3_k', 'q2_k', 'iq2_xs')
UNIFORM_MODES = ('mxfp4', *UNIFORM_GGML)
UNIFORM_CHOICES = (*UNIFORM_BITS, *UNIFORM_MODES)
ITEMSIZE = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'U16': 2, 'I16': 2,
    'F16': 2, 'BF16': 2, 'U32': 4, 'I32': 4, 'F32': 4,
    'U64': 8, 'I64': 8, 'F64': 8, 'F8_E4M3': 1, 'F8_E5M2': 1,
}
PRESETS = {
    'memory': {
        'roomy': {
            'label': 'Roomy',
            'about': 'more memory kept for long conversations and other apps (reserve 0.45)',
            'values': {'QUANTEEZE_MEMORY_RESERVE': '0.45'},
        },
        'standard': {
            'label': 'Standard',
            'about': 'the default: three tenths of the working set kept free (reserve 0.30)',
            'values': {'QUANTEEZE_MEMORY_RESERVE': ''},
        },
        'most': {
            'label': 'Most for experts',
            'about': 'more memory for the experts; long prompts and other apps have less room (reserve 0.15)',
            'values': {'QUANTEEZE_MEMORY_RESERVE': '0.15'},
        },
    },
    'wired': {
        'off': {
            'label': 'Off',
            'about': "the default: MLX's memory wired only while it generates, as mlx-vlm wires it",
            'values': {'QUANTEEZE_WIRED_GB': ''},
        },
        'auto': {
            'label': 'Auto',
            'about': 'wired from the load on, up to the working set Metal recommends (what mlx-vlm wires while it '
                     'generates): less of the model compressed before the first request. macOS still lets it go '
                     'within 30 s of the GPU falling idle, so the idle model is not kept from the compressor; while '
                     'it loads, other apps have less room',
            'values': {'QUANTEEZE_WIRED_GB': 'auto'},
        },
    },
    'ngram': {
        'auto': {
            'label': 'Auto',
            'about': 'the n-gram table read from the SSD where, held in memory, it would leave the experts less room '
                     'than they take; held in memory where they fit whole beside it',
            'values': {'QUANTEEZE_NGRAM_SSD': ''},
        },
        'ssd': {
            'label': 'SSD',
            'about': 'always read from the SSD row by row: its memory left to the experts',
            'values': {'QUANTEEZE_NGRAM_SSD': 'on'},
        },
        'memory': {
            'label': 'Memory',
            'about': 'always held in memory beside the experts: no reads from the SSD',
            'values': {'QUANTEEZE_NGRAM_SSD': 'off'},
        },
    },
    'mtp': {
        'auto': {
            'label': 'Auto',
            'about': 'drafts tokens ahead with the model\'s own MTP / DSpark head where that pays here, verified '
                     'bit for bit against single-token decode',
            'values': {'QUANTEEZE_MTP': '', 'QUANTEEZE_MTP_EXACT': ''},
        },
        'dflash2': {
            'label': 'DFlash2',
            'about': 'block-diffusion drafter (DFlash2) predicting a token block in parallel and tracing the '
                     'coherent path via CandidateSelector',
            'values': {'QUANTEEZE_MTP': 'dflash2', 'QUANTEEZE_MTP_EXACT': 'bits'},
        },
        'exact': {
            'label': 'On (exact)',
            'about': 'drafts with the model\'s own MTP head wherever it has one, verified bit for bit',
            'values': {'QUANTEEZE_MTP': 'on', 'QUANTEEZE_MTP_EXACT': 'bits'},
        },
        'fast': {
            'label': 'On (fast)',
            'about': 'drafts with the model\'s own MTP head, verifying tokens together in one multi-token pass',
            'values': {'QUANTEEZE_MTP': 'on', 'QUANTEEZE_MTP_EXACT': 'distribution'},
        },
        'off': {
            'label': 'Off',
            'about': 'one token a pass; the drafter head\'s 1-2 GB stay free',
            'values': {'QUANTEEZE_MTP': 'off', 'QUANTEEZE_MTP_EXACT': ''},
        },
    },
    'heat': {
        'quiet': {
            'label': 'Quiet',
            'about': 'slows down from the first warmth, so the fans stay low',
            'values': {'QUANTEEZE_THROTTLE_AT': '1', 'QUANTEEZE_THROTTLE_DUTY': '1:0.7,2:0.5,3:0.3,4:0.2'},
        },
        'standard': {
            'label': 'Standard',
            'about': 'the default: slows down once the machine reports heavy heat',
            'values': {'QUANTEEZE_THROTTLE_AT': '', 'QUANTEEZE_THROTTLE_DUTY': ''},
        },
        'full': {
            'label': 'Full speed',
            'about': 'never slows down for heat (the machine still protects itself)',
            'values': {'QUANTEEZE_THROTTLE_AT': 'off', 'QUANTEEZE_THROTTLE_DUTY': ''},
        },
    },
}

_MODELS = {}
_DOWNLOAD = {'repo': None, 'state': 'idle', 'error': None, 'bytes': 0, 'total': None, 'path': None}


def _decoder_layers(path):
    """The decoder layers' count in a checkpoint's config (its text config's), or None."""
    config = json.loads((Path(path) / 'config.json').read_text())
    return config.get('text_config', config).get('num_hidden_layers') or config.get('num_hidden_layers')


# ---------------------------------------------------------------- where things are

def hub():
    """Hugging Face's cache of repositories."""
    return Path(
        os.environ.get('HF_HUB_CACHE')
        or Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'hub'
    )


VOLUMES = Path('/Volumes')


def model_roots():
    """Directories holding checkpoints as <org>/<name>: LM Studio's, external volumes under
    `/Volumes/*/lmstudio`, and the two above the model served (or the one the server started with)."""
    roots = [Path.home() / '.lmstudio' / 'models']
    try:
        roots += sorted(VOLUMES.glob('*/lmstudio'))
    except OSError:
        pass
    served = control.chosen() or control.current() or control.argument(sys.argv[1:], '--model')
    if served and Path(served).expanduser().is_dir():
        roots.append(Path(served).expanduser().resolve().parent.parent)
    out = []
    for r in roots:
        try:
            ok = r.is_dir() and os.access(r, os.R_OK | os.X_OK)
        except OSError:
            ok = False
        if ok and r not in out:
            out.append(r)
    return out


def repo_of(path):
    """The Hugging Face repository a checkpoint's directory came from, as far as its path says:
    the cache's models--<org>--<name>, or LM Studio's <org>/<name>; else None."""
    parts = Path(path).expanduser().parts
    for p in parts:
        if p.startswith('models--'):
            return '/'.join(p[len('models--'):].split('--', 1))
    if len(parts) >= 3 and parts[-3] == 'models':
        return f'{parts[-2]}/{parts[-1]}'
    return None


def checkpoints():
    """The checkpoint directories found (a config.json and safetensors)."""
    found = set()
    for root in model_roots():
        found |= {p.parent for p in root.glob('*/*/config.json')} | {p.parent for p in root.glob('*/config.json')}
    if hub().is_dir():
        for repo in hub().glob('models--*'):
            snaps = sorted(repo.glob('snapshots/*/config.json'), key=lambda p: p.stat().st_mtime)
            if snaps:
                found.add(snaps[-1].parent)
    models = []
    for d in sorted(found):
        if (not any(d.glob('*.safetensors')) or _drafter(d) or d.name.endswith('.ple-view')
                or '.ple-view.part' in d.name):
            continue
        models.append(d)
    return models


def _drafter(path):
    """Whether a checkpoint is a speculative drafter (`..._mtp`, `dflash`, `dflash2`, `dspark`, `eagle3`) rather
    than a standalone model."""
    try:
        cfg = json.loads((Path(path) / 'config.json').read_text())
    except (OSError, ValueError):
        return False
    mtype = str(cfg.get('model_type', '')).lower()
    archs = [str(a) for a in (cfg.get('architectures') or ())]
    return (
        mtype.endswith('_mtp')
        or mtype in ('dflash', 'dflash2', 'dspark', 'eagle3')
        or 'dflash_config' in cfg
        or any('DraftModel' in a or 'DSparkModel' in a for a in archs)
    )


# ---------------------------------------------------------------- the catalog

def _config(path):
    config = json.loads((Path(path) / 'config.json').read_text())
    return config.get('text_config', config), config


def _gb(path):
    """GB of a checkpoint's safetensors files; of a view of one (quanteeze.ple_store), of the tensors
    its index names alone."""
    path = Path(path)
    index = path / 'model.safetensors.index.json'
    if (path / 'EXTERNAL_PLE.json').is_file() and index.is_file():
        from . import ple_store
        weights = json.loads(index.read_text()).get('weight_map') or {}
        total = 0
        for _, header in ple_store._headers(path, sorted(set(weights.values()))).values():
            total += sum(t['data_offsets'][1] - t['data_offsets'][0] for name, t in header.items()
                         if name in weights and isinstance(t, dict) and 'data_offsets' in t)
        return total / 1e9
    return sum(os.path.getsize(p) for p in path.glob('*.safetensors') if p.exists()) / 1e9


def model_info(path):
    """What the catalog says of a checkpoint: its name, repository, model type, GB, whether it has
    routed experts, its original expert GB and uniform least-squares GB at 1-8 bits, and the GB of
    it the server does not hold (`unheld()`)."""
    path = Path(path)
    key = (str(path), (path / 'config.json').stat().st_mtime)
    if key in _MODELS:
        return dict(_MODELS[key])
    text, whole = _config(path)
    n_experts = next((int(text.get(k) or whole.get(k)) for k in EXPERTS if text.get(k) or whole.get(k)), 0)
    moe = n_experts > 0
    info = {
        'path': str(path),
        'name': repo_of(path) or path.name,
        'repo': repo_of(path),
        'model_type': text.get('model_type') or whole.get('model_type'),
        'gb': round(_gb(path), 1),
        'moe': bool(moe),
        'layers': _decoder_layers(path),
        'experts': n_experts if moe else None,
        'experts_per_tok': int(text.get('num_experts_per_tok') or 0) or None,
    }
    drafts, why = mtp.auto(path, whole.get('model_type'))
    info['mtp'] = {'head': mtp.head(path) > 0, 'auto': drafts, 'why': why}
    try:
        info['ple'] = ple_info(path)
    except (ValueError, OSError, KeyError) as e:
        info['ple'] = {'gb': None, 'view': None, 'is_view': False, 'error': f'{type(e).__name__}: {e}'}
    try:
        info['unheld_gb'], info['engram'] = unheld(path, whole.get('model_type'))
    except (ValueError, OSError, KeyError) as e:
        info['unheld_gb'], info['engram'] = 0.0, None
        info['unheld_error'] = f'{type(e).__name__}: {e}'
    if moe:
        try:
            info['experts_gb'] = round(experts_bytes(path) / 1e9, 1)
        except (ValueError, OSError, KeyError) as e:
            info['experts_gb'] = None
            info['experts_error'] = f'{type(e).__name__}: {e}'
        try:
            info['uniform_gb'], info['stock_bits'] = uniform_gb(path)
        except (ValueError, OSError, KeyError):
            info['uniform_gb'], info['stock_bits'] = None, None
    _MODELS[key] = info
    return dict(info)


def _sizes(path, named):
    """{name: bytes} of the tensors of the checkpoint at `path` (its index's) whose names `named`
    takes, from their safetensors headers alone; {} where it has no index."""
    from . import ple_store
    path = Path(path)
    index = path / 'model.safetensors.index.json'
    if not index.is_file():
        return {}
    weights = json.loads(index.read_text()).get('weight_map') or {}
    out = {}
    for _, header in ple_store._headers(path, sorted({f for k, f in weights.items() if named(k)})).values():
        for name, t in header.items():
            if name in weights and named(name) and isinstance(t, dict) and 'data_offsets' in t:
                out[name] = t['data_offsets'][1] - t['data_offsets'][0]
    return out


def unheld(path, model_type=None):
    """(GB of the checkpoint at `path` the server does not hold beside the routed experts, and
    {'gb'} of its Engram tables or None)."""
    known = arch.get(model_type)
    other = re.compile(known.not_held).search if known is not None and known.not_held else lambda k: False
    sizes = _sizes(path, lambda k: arch.ROW_BY_ROW in k or other(k))
    engram = sum(n for k, n in sizes.items() if arch.ROW_BY_ROW in k)
    return round(sum(sizes.values()) / 1e9, 1), {'gb': round(engram / 1e9, 1)} if engram else None


# ---------------------------------------------------------------- the n-gram (PLE) table on the SSD

PLE_VIEWS = Path.home() / '.quanteeze' / 'ple-views'
_PLE = {'state': 'idle', 'path': None, 'view': None, 'error': None}


def _ple_targets(path):
    """Where a checkpoint's view (`quanteeze.ple_store.prepare`) goes: under PLE_VIEWS, else beside it
    (`<name>.ple-view`, or `<name>.quanteeze.ple-view` where another tool's view has that name)."""
    path = Path(path).resolve()
    name = f'{path.name}-{hashlib.sha256(str(path).encode()).hexdigest()[:8]}'
    return [PLE_VIEWS / name, path.with_name(f'{path.name}.ple-view'),
            path.with_name(f'{path.name}.quanteeze.ple-view')]


def _is_view(target):
    """Whether `target` is a view made whole whose manifest this server reads (ple_store.readable)."""
    from . import ple_store
    return ((target / 'EXTERNAL_PLE.json').is_file() and (target / 'config.json').is_file()
            and ple_store.readable(target))


def ple_view(path):
    """The checkpoint's view whose table is read from the SSD, when one was made whole; else None.
    A directory in its place that is not such a view (another tool's, its manifest in a format of
    its own) is passed over."""
    for target in _ple_targets(path):
        if _is_view(target):
            return target
    return None


def _make_view(path, prepare=None):
    """The checkpoint's view made (`quanteeze.ple_store.prepare`: hard links, a manifest and its
    config), into a directory of its own first, renamed when whole."""
    from . import ple_store
    prepare = prepare or ple_store.prepare
    errors = []
    for target in _ple_targets(path):
        if target.exists() and not _is_view(target):
            errors.append(f'{target}: there already, not a view this server reads')
            continue
        part = target.with_name(f'{target.name}.part-{os.getpid()}')
        try:
            shutil.rmtree(part, ignore_errors=True)
            part.parent.mkdir(parents=True, exist_ok=True)
            prepare(path, part)
            try:
                os.replace(part, target)
            except OSError:
                if ple_view(path) is None:
                    raise
                shutil.rmtree(part, ignore_errors=True)
            _MODELS.clear()
            return ple_view(path) or target
        except Exception as e:
            shutil.rmtree(part, ignore_errors=True)
            errors.append(f'{target.parent}: {type(e).__name__}: {e}')
    raise ValueError('; '.join(errors))


def ngram_source(path, prepare=None):
    """What mlx-vlm is given to load for the checkpoint at `path` (or its view), as
    QUANTEEZE_NGRAM_SSD says, and why (None where it has no n-gram table)."""
    path = Path(path).expanduser()
    source = path
    named = (path / 'EXTERNAL_PLE.json').is_file()
    if named:
        source = Path(json.loads((path / 'EXTERNAL_PLE.json').read_text()).get('source_model') or path)
        if not _is_view(path) and source != path and (source / 'config.json').is_file():
            # another tool's view, its manifest in a format of its own: its checkpoint served instead
            path, named = source, False
    try:
        table = ple_info(source)
    except (ValueError, OSError, KeyError):
        table = None
    if not table or table.get('is_view') or table.get('gb') is None:
        return path, None
    gb = f'its n-gram table ({table["gb"]:g} GB)'
    mode = (variables.get('QUANTEEZE_NGRAM_SSD') or 'auto').lower()
    if mode == 'off':
        return source, f'{gb} held in memory (QUANTEEZE_NGRAM_SSD=off)'
    if mode == 'auto' and ngram_whole(model_info(source)):
        return source, f'{gb} held in memory: the experts fit whole beside it'
    view = (path if named else None) or ple_view(source) or _make_view(source, prepare)
    return view, f'{gb} read from the SSD' + (' (QUANTEEZE_NGRAM_SSD=on)' if mode == 'on' else
                                              ': held, it would leave the experts less room than they take')


def ngram_read(info):
    """Whether a load of this checkpoint or view reads its n-gram table from the SSD, as ngram_source
    chooses; None where it has no table."""
    p = info.get('ple')
    if not p or not p.get('gb'):
        return None
    mode = (variables.get('QUANTEEZE_NGRAM_SSD') or 'auto').lower()
    if mode in ('on', 'off'):
        return mode == 'on'
    return not ngram_whole(model_info(p['source']) if p.get('is_view') else info)


def load_room_gb(info):
    """room_gb as its load would leave the routed experts: beside the rest without the n-gram table
    where a checkpoint's is read from the SSD, with it where a view's checkpoint is loaded instead."""
    room, ssd, p = room_gb(info), ngram_read(info), info.get('ple')
    if room is None or ssd is None or ssd == bool(p.get('is_view')):
        return room
    return room_gb(info, p['gb'] if p.get('is_view') else -p['gb'])


def ngram_whole(info):
    """Whether a checkpoint's routed experts fit whole here beside its n-gram table held in memory."""
    room = room_gb(info)
    return room is not None and info.get('experts_gb') is not None and info['experts_gb'] <= room


def ple_info(path):
    """None for a checkpoint with no n-gram (PLE) table held in memory; else {'gb', 'view', 'is_view'}."""
    from . import ple_store
    path = Path(path)
    if (path / 'EXTERNAL_PLE.json').is_file():
        source = json.loads((path / 'EXTERNAL_PLE.json').read_text()).get('source_model')
        here = source and Path(source).is_dir() and not (Path(source) / 'EXTERNAL_PLE.json').is_file()
        table = ple_info(Path(source)) if here else None
        return {'gb': table and table.get('gb'), 'view': str(path), 'is_view': True, 'source': source if here else None}
    table = _sizes(path, ple_store.SHARD.match)
    if not table:
        return None
    view = ple_view(path)
    return {'gb': round(sum(table.values()) / 1e9, 1), 'view': str(view) if view else None, 'is_view': False}


def prepare_ple(path, prepare=None):
    """Make the checkpoint's view on a thread of its own; `ple_state()` tells how it goes."""
    from . import ple_store
    path = Path(path).expanduser().resolve()
    if not ple_info(path):
        raise ValueError(f'{path}: no n-gram table to read from the SSD')
    if ple_view(path):
        raise ValueError(f'{path}: its view is made already ({ple_view(path)})')
    if _PLE['state'] == 'preparing':
        raise ValueError(f'making the view of {_PLE["path"]} already')
    prepare = prepare or ple_store.prepare
    _PLE.update(state='preparing', path=str(path), view=None, error=None)

    def run():
        try:
            _PLE.update(state='done', view=str(_make_view(path, prepare)))
        except ValueError as e:
            _PLE.update(state='failed', error=str(e))

    threading.Thread(target=run, name='quanteeze-ple', daemon=True).start()
    return ple_state()


def ple_state():
    return dict(_PLE)


# ---------------------------------------------------------------- catalog & uniform bit-widths

def catalog():
    """{'models', 'served', 'bits', 'from_ssd_gb', 'working_set_gb', 'machine'}: each local
    checkpoint with its original and uniform least-squares expert sizes and available room."""
    models = checkpoints()
    infos = [model_info(m) for m in models]
    for m in infos:
        m['ngram_ssd'] = ngram_read(m)
        m['room_gb'] = load_room_gb(m)
        m['whole'] = None if m['room_gb'] is None or m.get('experts_gb') is None else m['experts_gb'] <= m['room_gb']
    served = control.chosen() or control.current()
    if served and (Path(served) / 'EXTERNAL_PLE.json').is_file():
        served = json.loads((Path(served) / 'EXTERNAL_PLE.json').read_text()).get('source_model') or served
    return {
        'models': infos,
        'served': served,
        'bits': control._uniform_bits(),
        'from_ssd_gb': variables.get('QUANTEEZE_FROM_SSD_GB'),
        'working_set_gb': _working_set_gb(),
        'machine': machine(),
    }


def _working_set_gb():
    w = guards.recommended_budget()
    return round(w / 1e9, 1) if w else None


def experts_bytes(path, layers=None):
    """Bytes a checkpoint's routed experts take at the original precision (decoder layers only; of `layers`
    alone when given, e.g., a rank's run over several GPUs)."""
    by = layer_experts_bytes(path)
    return sum(b for layer, b in by.items() if layers is None or layer in layers)


def layer_experts_bytes(path):
    """`{decoder layer: bytes its routed experts take at the original precision}` of a checkpoint."""
    reader = q2_experts.Checkpoint(path)
    out = {}
    for layer in range(_decoder_layers(reader.path) or 0):
        for p in ('gate_proj', 'up_proj', 'down_proj'):
            for part in ('weight', 'scales', 'biases'):
                if reader.has(layer, p, part):
                    _, dtype, shape, _ = reader.tensors[reader.name(layer, p, part)]
                    out[layer] = out.get(layer, 0) + int(np.prod(shape)) * ITEMSIZE.get(dtype, 1)
    return out


def uniform_bytes(path):
    """`({bits: bytes}, stock_bits)` the checkpoint's routed experts take uniformly at each of
    `UNIFORM_CHOICES` (least squares at 1-8 bits, `'mxfp4'`, plus standard ggml `q3_k`, `q2_k`, `iq2_xs`)."""
    from . import requant
    reader = requant.Flat(path, 2)
    weights = 0
    stock_bits = None
    for layer in range(_decoder_layers(reader.path) or 0):
        for p in ('gate_proj', 'up_proj', 'down_proj'):
            if reader.base.has(layer, p, 'weight'):
                _, dtype, shape, _ = reader.base.tensors[reader.base.name(layer, p, 'weight')]
                b = reader._quantization(layer, p)[2] or (ITEMSIZE.get(dtype, 2) * 8)
                if stock_bits is None:
                    stock_bits = b
                weights += int(np.prod(shape)) * ITEMSIZE.get(dtype, 1) * 8 // b
    return {b: fitting.quantized_bytes(weights, b) for b in UNIFORM_CHOICES}, stock_bits


def uniform_gb(path):
    """`(dict of {bits: gb}, stock_bits)` as shown in the UI."""
    by_bits, stock_bits = uniform_bytes(path)
    return {b: round(n / 1e9, 1) for b, n in by_bits.items()}, stock_bits


def room_gb(info, table=0.0):
    """GB the server would leave a model's routed experts here: `fitting.room` beside the rest
    (the checkpoint less its experts and what the server does not hold of it, `unheld_gb`), `table`
    GB more of it, the reserve as set now; None where unknown."""
    if not info.get('experts_gb'):
        return None
    rest = info['gb'] - info['experts_gb'] - (info.get('unheld_gb') or 0) + table
    space = fitting.room(max(0.0, rest * 1e9))
    return None if space is None else round(space / 1e9, 1)


def start(path, bits=None, from_ssd_gb=None):
    """Load the checkpoint at `path` with uniform expert quantization `bits` (`None`/`''`/`'off'`
    for checkpoint precision, `1`/`2`/`3`/`4`/`8` for uniform least-squares quantization, `'mxfp4'`
    for uniform MXFP4, or `'q3_k'`/`'q2_k'`/`'iq2_xs'` for uniform ggml quantization) and optional
    `from_ssd_gb` (`None`/`''` for automatic in-memory when fitting else on-demand SSD).  Returns
    `control.model_state()`."""
    if not (Path(path).expanduser() / 'config.json').is_file():
        raise ValueError(f'{path}: no checkpoint there')
    bits_norm = str(bits).strip().lower() if bits not in (None, '', 0) else ''
    valid = {*map(str, UNIFORM_BITS), *UNIFORM_MODES}
    if bits_norm not in ('', 'off') and bits_norm not in valid:
        choices = [*map(str, UNIFORM_BITS), *UNIFORM_MODES]
        raise ValueError(f'bits: {", ".join(choices)}, or off, not {bits!r}')
    previous = {name: variables.get(name) for name in ('QUANTEEZE_BITS', 'QUANTEEZE_FROM_SSD_GB')}
    changes = {'QUANTEEZE_BITS': bits_norm if bits_norm not in ('', 'off') else ''}
    if from_ssd_gb is not None:
        changes['QUANTEEZE_FROM_SSD_GB'] = str(from_ssd_gb).strip()
    out = control.change(changes)
    if out['rejected']:
        raise ValueError('; '.join(f'{n}: {why}' for n, why in out['rejected'].items()))
    resolved = ngram_source(path)[0]
    try:
        return control.switch_model(resolved, undo=previous)
    except ValueError:
        control.change(previous)
        raise


def machine():
    """What the page calls the machine it serves on: 'this Mac' on macOS, else 'this machine'."""
    return 'this Mac' if sys.platform == 'darwin' else 'this machine'


# ---------------------------------------------------------------- memory, PLE, MTP, and heat presets

def presets():
    """Each preset group's choices and the one the settings now match ('custom' where none)."""
    out = {}
    for group, choices in PRESETS.items():
        now = None
        for key, p in choices.items():
            if all(variables.get(n) == (v or variables.get(n, {})) for n, v in p['values'].items()):
                now = key
        out[group] = {
            'choices': [{'key': k, 'label': p['label'], 'about': p['about']} for k, p in choices.items()],
            'now': now or 'custom',
        }
    if sys.platform != 'darwin':
        out['heat']['note'] = 'Only macOS tells quanteeze how hot it runs: here none of these slows it down.'
    import mlx.core as mx
    if not mx.metal.is_available():
        out['wired']['note'] = (
            "Only Metal wires memory (the Mac's memory, shared with the GPU): a GPU's own "
            "memory is not compressed or paged by the system, and here neither choice does anything."
        )
    out['loaded'] = control._LOADED[0] is not None
    out['sampling'] = {key: os.environ.get(name) or None for name, key in control.SAMPLING.items()}
    return out


def choose_preset(group, key):
    """Set a preset's values (`quanteeze.control.change`: saved, read at the next load)."""
    try:
        values = PRESETS[group][key]['values']
    except KeyError:
        raise ValueError(f'no preset {group}/{key}') from None
    out = control.change(dict(values))
    if out['rejected']:
        raise ValueError('; '.join(f'{n}: {why}' for n, why in out['rejected'].items()))
    return presets()


# ---------------------------------------------------------------- downloads

def _repo(repo):
    if not isinstance(repo, str) or not REPO.match(repo):
        raise ValueError(f'{repo!r}: a Hugging Face repository is <org>/<name>')
    return repo


def info(repo):
    """A repository's files and GB, as Hugging Face tells them, and where they would go."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    repo = _repo(repo)
    try:
        m = HfApi().model_info(repo, files_metadata=True)
    except GatedRepoError:
        raise ValueError(
            f'{repo}: gated on Hugging Face; accept its terms there and log in (hf auth login) first'
        ) from None
    except RepositoryNotFoundError:
        raise ValueError(f'{repo}: Hugging Face has no such repository, or it is private') from None
    sizes = [s.size or 0 for s in m.siblings or []]
    return {
        'repo': repo,
        'files': len(sizes),
        'gb': round(sum(sizes) / 1e9, 2),
        'gated': bool(getattr(m, 'gated', False)),
        'where': str(hub() / f'models--{repo.replace("/", "--")}'),
        'local': local(repo),
    }


def local(repo):
    """The checkpoint of `repo` here (its catalog path), or None."""
    for d in checkpoints():
        if repo_of(d) == repo:
            return str(d)
    return None


def _landed(repo):
    blobs = hub() / f'models--{repo.replace("/", "--")}' / 'blobs'
    return sum(p.stat().st_size for p in blobs.glob('*') if p.is_file()) if blobs.is_dir() else 0


def download_state():
    out = dict(_DOWNLOAD)
    if out['state'] == 'downloading' and out['repo']:
        out['bytes'] = _landed(out['repo'])
    return out


def download(repo, total=None, fetch=None):
    """Fetch `repo` into Hugging Face's cache on a thread of its own; one at a time."""
    repo = _repo(repo)
    if _DOWNLOAD['state'] == 'downloading':
        raise ValueError(f'downloading {_DOWNLOAD["repo"]} already')
    if fetch is None:
        from huggingface_hub import snapshot_download as fetch
    _DOWNLOAD.update(repo=repo, state='downloading', error=None, bytes=0, total=total, path=None)

    def run():
        try:
            _DOWNLOAD['path'] = str(fetch(repo_id=repo))
            _DOWNLOAD.update(state='done', bytes=_landed(repo))
            _MODELS.clear()
        except Exception as e:
            _DOWNLOAD.update(state='failed', error=f'{type(e).__name__}: {e}')

    threading.Thread(target=run, name='quanteeze-download', daemon=True).start()
    return download_state()


# ---------------------------------------------------------------- what is served now

_READINGS = guards.Readings()


def _serve():
    """The server's module as it runs."""
    import importlib
    main = sys.modules.get('__main__')
    return main if isinstance(getattr(main, '_SERVING', None), dict) else importlib.import_module('quanteeze.serve')


def live():
    """What the server is doing: MLX's memory, thermal pressure and throttle, uniform quantization /
    on-demand SSD stats, MTP speculative decoding stats, and each request's phase."""
    import mlx.core as mx
    from . import allocator, phases
    serve = _serve()
    keeper = getattr(serve, '_KEEPER', [None])[0]
    out = {
        'memory': {
            'active_gb': round(mx.get_active_memory() / 1e9, 1),
            'peak_gb': round(mx.get_peak_memory() / 1e9, 1),
            'cache_gb': round(mx.get_cache_memory() / 1e9, 1),
            'working_set_gb': _working_set_gb(),
            'keeper': None if keeper is None else keeper.state(),
            'allocator': allocator.state(),
        }
    }
    try:
        out['heat'] = {'level': _READINGS.thermal()}
    except Exception:
        out['heat'] = {'level': None}
    throttle = serve._SERVING.get('throttle')
    if throttle is not None:
        out['heat'].update(throttle_at=throttle.at, rested_s=round(getattr(throttle, 'rested', 0.0) or 0.0, 1))
    demand_inst = serve._SERVING.get('demand')
    if demand_inst is not None:
        out['demand'] = demand_inst.summary()
    out['serving'] = dict(serve._SERVING.get('info') or {})
    if mtp.ACTIVE[0] is not None:
        out['mtp'] = mtp.summary(mtp.ACTIVE[0][0])
    out['model'] = control.chosen() or control.current()
    out['requests'] = phases.state(control._SWITCH['inflight'])
    return out


# ---------------------------------------------------------------- connecting an app

def _lan_address():
    """This machine's address on its network (the one a route out would leave from), or None."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('192.0.2.1', 9))
            return s.getsockname()[0]
    except OSError:
        return None


def connect():
    """How an app reaches this server: the OpenAI-compatible base URL, the Anthropic one (Claude
    Code), the model it serves, and whether other machines can reach it."""
    argv = sys.argv[1:]
    host, port = control.argument(argv, '--host', '127.0.0.1'), control.argument(argv, '--port', control.DEFAULT_PORT)
    every = host in ('0.0.0.0', '::', '')
    shown = (_lan_address() or '127.0.0.1') if every else host
    shown = f'[{shown}]' if ':' in shown else shown
    base = f'http://{shown}:{port}'
    return {
        'openai': f'{base}/v1',
        'anthropic': base,
        'model': control.serving(),
        'local_only': host in ('127.0.0.1', 'localhost', '::1'),
        'claude_code': f'ANTHROPIC_BASE_URL={base} ANTHROPIC_AUTH_TOKEN=local claude',
        'codex': (
            'codex -c model_provider="quanteeze" -c model="local" '
            f'-c \'model_providers.quanteeze={{name="Quanteeze",base_url="{base}/v1",'
            'wire_api="responses",requires_openai_auth=false,stream_idle_timeout_ms=7200000}\''
        ),
        'curl': f'curl {base}/v1/chat/completions -H "Content-Type: application/json" '
                '-d \'{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}\'',
    }


# ---------------------------------------------------------------- the routes

def handle(method, route, body):
    """A route of API (`route` the part after it) with its JSON body: its reply's JSON."""
    if (method, route) == ('GET', 'catalog'):
        return catalog()
    if (method, route) == ('POST', 'start'):
        return start(body.get('path') or '', body.get('bits'), body.get('from_ssd_gb'))
    if (method, route) == ('GET', 'presets'):
        return presets()
    if (method, route) == ('POST', 'presets'):
        return choose_preset(body.get('group'), body.get('key'))
    if (method, route) == ('POST', 'download/info'):
        return info(body.get('repo'))
    if (method, route) == ('GET', 'connect'):
        return connect()
    if (method, route) == ('GET', 'live'):
        return live()
    if (method, route) == ('POST', 'ple'):
        return prepare_ple(body.get('path') or '')
    if (method, route) == ('GET', 'ple'):
        return ple_state()
    if (method, route) == ('POST', 'download'):
        return download(
            body.get('repo'),
            float(body['gb']) * 1e9 if isinstance(body.get('gb'), (int, float)) else None,
        )
    return download_state()
