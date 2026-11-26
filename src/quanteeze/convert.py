# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A checkpoint of float weights converted to MLX from the home page: mlx-lm's or mlx-vlm's convert, at 4 or 8 bits.

The page lists every folder with a config.json and safetensors (quanteeze.home.checkpoints), a repository's own
float weights too.  Such a checkpoint -- no quantization in its config, most tensors float -- is marked
(unquantized()), and the page offers to convert it (from the original weights, not from GGUF, which would quantize
values quantized already).

plan() tells what a conversion would do before it starts, for the page to show and the user to agree to: the tool
(mlx-lm for a text model, mlx-vlm for one that reads images or that mlx-lm does not know), where it writes (beside
the checkpoint as <name>-MLX-<bits>bit in its model folder; a Hugging Face snapshot's into LM Studio's folder, else
~/.quanteeze/models, both listed by the page), the GB it writes (the weights at bits + 0.5 a weight for the
scales and biases of groups of 64), the GB free there, and why not where it cannot: a model loaded (the converter
reads and quantizes on the GPU; unload it first), a conversion running, the folder there already, too little room
(DISK_LEAVES_GB left free), or a model type neither knows.

start() runs this file as a process of its own (python -I convert.py ...: the standard library alone, no threads in
the server, alive across a restart of the server), which
- makes a staging folder of links to the checkpoint's files beside the destination, its config.json and
  tokenizer_config.json written again without auto_map and model_file (what would run the repository's own code:
  none is run) and with a {"__float__": "Infinity"} read as the number (which some exporters write and mlx-lm
  cannot read); the repository's .py files and folders are not linked;
- runs `python -I -m mlx_lm convert` (or mlx_vlm) on it, -q at the bits, groups of 64, the rest bfloat16;
- watches it every 2 s: stopped where the swap grows SWAP_CUT_GB or the folder grows past CAP times the GB told
  (no unbounded writes); the state in STATE_DIR/state.json, its output in log.txt;
- copies the checkpoint's other small files the converter left out (a chat template, a manifest), notes what it was
  made from (MARKER), and removes the staging folder; a failed or stopped one removes the folder it made.
running() tells whether one runs (control.switch_model refuses a load meanwhile); state() what the page shows.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

BITS = (4, 8)
GROUP = 64
SCALES_BITS = 0.5                       # a weight's share of its group's bfloat16 scale and bias (2 x 16 / 64)
DISK_LEAVES_GB = 20.0                   # GB of the destination's drive a conversion is never to take
CAP = 1.5                               # the folder written at most this many times the GB told, and SLACK more
SLACK = 2e9
SWAP_CUT_GB = 8.0                       # the converter stopped where the swap grows by this much
EXTRA_MOST_BYTES = 64 << 20             # a file of the checkpoint's copied beside the converted weights at most so big
MARKER = 'quanteeze_converted.json'     # what a converted folder was made from
QUANTIZED = ('quantization', 'quantization_config')
FLOATS = ('F32', 'BF16', 'F16', 'F64')
STATE_DIR = Path.home() / '.quanteeze' / 'convert'
_RUNNER = [None]                        # the runner's Popen, polled as the state is read (reaped)


# ---------------------------------------------------------------- what a checkpoint is

def _header(file):
    with open(file, 'rb') as f:
        n = int.from_bytes(f.read(8), 'little')
        return json.loads(f.read(n))


def _tensors(path):
    """{dtype: elements} over the checkpoint's safetensors, from their headers alone."""
    out = {}
    for file in sorted(Path(path).glob('*.safetensors')):
        for name, t in _header(file).items():
            if name == '__metadata__' or not isinstance(t, dict):
                continue
            n = 1
            for d in t.get('shape') or ():
                n *= int(d)
            out[t.get('dtype')] = out.get(t.get('dtype'), 0) + n
    return out


def _configs(path):
    whole = json.loads((Path(path) / 'config.json').read_text())
    return whole, whole.get('text_config') or {}


def unquantized(path):
    """{'dtype', 'params'} of a checkpoint whose weights are float -- no quantization in its config (MLX's, or the
    makers' own: FP8 and such are not converted here) and most of its elements float -- else None."""
    whole, text = _configs(path)
    if any(k in whole or k in text for k in QUANTIZED):
        return None
    counts = _tensors(path)
    total = sum(counts.values())
    if not total:
        return None
    dtype = max(counts, key=counts.get)
    if dtype not in FLOATS:
        return None
    return {'dtype': dtype.lower(), 'params': total}


def tool(path):
    """('mlx_lm' or 'mlx_vlm', None), or (None, why): mlx-vlm for a model that reads images or audio, else mlx-lm
    where it knows the model type, else mlx-vlm where it does."""
    import importlib.util
    whole, _ = _configs(path)
    kind = whole.get('model_type')
    if not kind:
        return None, 'its config.json names no model_type'

    def vlm():
        try:
            from mlx_vlm.utils import get_model_type
            name = get_model_type(whole)
            return bool(name) and importlib.util.find_spec(f'mlx_vlm.models.{name}') is not None
        except (ImportError, ValueError, KeyError):
            return False

    def lm():
        try:
            from mlx_lm.utils import _get_classes
            _get_classes(whole)
            return True
        except (ImportError, ValueError, KeyError):
            return False

    if any(k in whole for k in ('vision_config', 'audio_config')) and vlm():
        return 'mlx_vlm', None
    if lm():
        return 'mlx_lm', None
    if vlm():
        return 'mlx_vlm', None
    return None, f'neither mlx-lm nor mlx-vlm knows model type {kind!r}'


def destination(path, bits, roots, repo=None):
    """Where the converted checkpoint goes: <name>-MLX-<bits>bit beside the checkpoint where it sits in one of the
    model folders `roots` as <org>/<name>; a Hugging Face snapshot's (its `repo` <org>/<name>) into the first of
    LM Studio's folder (where it is) and ~/.quanteeze/models."""
    path = Path(path).resolve()
    for root in roots:
        root = Path(root).resolve()
        if path.parent.parent == root or path.parent == root:
            return path.parent / f'{path.name}-MLX-{bits}bit'
    org, _, name = (repo or f'local/{path.name}').partition('/')
    lmstudio = Path.home() / '.lmstudio' / 'models'
    base = lmstudio if lmstudio.is_dir() else fallback_root()
    return base / org / f'{name}-MLX-{bits}bit'


def fallback_root():
    """Where a snapshot is converted to with no LM Studio folder: a model folder the page lists (home.model_roots)."""
    return Path.home() / '.quanteeze' / 'models'


def _free_gb(folder):
    folder = Path(folder)
    while not folder.exists():
        folder = folder.parent
    return shutil.disk_usage(folder).free / 1e9


def _writable(folder):
    folder = Path(folder)
    while not folder.exists():
        folder = folder.parent
    return os.access(folder, os.W_OK | os.X_OK)


def plan(path, bits, roots, repo=None, loaded=None):
    """What converting the checkpoint at `path` to `bits` would do, for the page: {'path', 'bits', 'tool', 'dest',
    'gb' (written, told ahead), 'free_gb', 'dtype', 'params', 'why' (None where it may start)}.  loaded: the model
    the server holds now, or None."""
    bits = int(bits)
    if bits not in BITS:
        raise ValueError(f'bits: {", ".join(map(str, BITS))}, not {bits}')
    path = Path(path).expanduser()
    if not (path / 'config.json').is_file():
        raise ValueError(f'{path}: no config.json there')
    found = unquantized(path)
    out = {'path': str(path), 'bits': bits, 'tool': None, 'dest': None, 'gb': None, 'free_gb': None, 'why': None,
           'dtype': found and found['dtype'], 'params': found and found['params']}
    if found is None:
        out['why'] = 'its weights are quantized already'
        return out
    out['tool'], why = tool(path)
    dest = destination(path, bits, roots, repo)
    out.update(dest=str(dest), gb=round(found['params'] * (bits + SCALES_BITS) / 8 / 1e9, 1),
               free_gb=round(_free_gb(dest.parent), 1))
    if why:
        out['why'] = why
    elif running():
        out['why'] = 'a conversion is running'
    elif loaded:
        out['why'] = (f'a model is loaded ({Path(loaded).name}): unload it first -- the converter reads and quantizes '
                      'on the GPU, and needs the memory')
    elif dest.exists():
        out['why'] = f'{dest} is there already'
    elif not _writable(dest.parent):
        out['why'] = f'{dest.parent} may not be written'
    elif out['free_gb'] - out['gb'] < DISK_LEAVES_GB:
        out['why'] = (f'{out["free_gb"]:.0f} GB free there: {out["gb"]:.0f} GB would leave less than '
                      f'{DISK_LEAVES_GB:.0f} GB')
    return out


# ---------------------------------------------------------------- the runner, from the server

def _read_state():
    try:
        return json.loads((STATE_DIR / 'state.json').read_text())
    except (OSError, ValueError):
        return None


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def state():
    """The last conversion's state ({'state': 'idle'} where none ran): its source, destination, bits, tool, GB
    told, the bytes its folder holds now (while converting), its error, the end of its log."""
    if _RUNNER[0] is not None and _RUNNER[0].poll() is not None:
        _RUNNER[0] = None                               # reaped
    s = _read_state()
    if s is None:
        return {'state': 'idle'}
    if s.get('state') == 'converting':
        if not _alive(s.get('pid')):
            s.update(state='failed', error='the converter stopped without a word (the machine or its process)')
        else:
            s['bytes'] = _bytes(s.get('dest'))
    return s


def running():
    return state().get('state') == 'converting'


def start(path, bits, roots, repo=None, loaded=None):
    """Convert the checkpoint at `path` to `bits` in a process of its own, where plan() lets it; its state()."""
    p = plan(path, bits, roots, repo, loaded)
    if p['why']:
        raise ValueError(p['why'])
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    told = int(p['gb'] * 1e9)
    with open(STATE_DIR / 'log.txt', 'w') as log:
        _RUNNER[0] = subprocess.Popen(
            [sys.executable, '-I', str(Path(__file__).resolve()), p['path'], p['dest'], str(p['bits']), p['tool'],
             str(told), str(STATE_DIR)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(50):                                 # its first state written
        s = _read_state()
        if s and s.get('pid') == _RUNNER[0].pid:
            break
        time.sleep(0.1)
    return state()


# ---------------------------------------------------------------- the runner itself (python -I convert.py ...)

def _bytes(folder):
    total = 0
    try:
        for root, _, files in os.walk(folder):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except (OSError, TypeError):
        pass
    return total


def _swap_gb():
    """GB of swap in use, or None where it cannot be told."""
    try:
        if sys.platform == 'darwin':
            out = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True, timeout=5).stdout
            used = out.split('used =')[1].split()[0]
            scale = {'K': 1e-6, 'M': 1e-3, 'G': 1.0}.get(used[-1], 1e-9)
            return float(used.rstrip('KMG')) * scale
        info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line)
        return (int(info['SwapTotal'].split()[0]) - int(info['SwapFree'].split()[0])) / 1e6
    except Exception:                                   # noqa: BLE001 -- not watched then
        return None


def _floats(value):
    """`value` with {"__float__": "Infinity"} and the like read as numbers (a config some exporters write)."""
    if isinstance(value, dict):
        if set(value) == {'__float__'}:
            return float(value['__float__'])
        return {k: _floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_floats(v) for v in value]
    return value


CODE_KEYS = ('auto_map', 'model_file')        # what has mlx-lm, mlx-vlm or transformers run the repository's code


def stage(source, folder):
    """A folder of links to `source`'s files (not its folders or .py files), its config.json and
    tokenizer_config.json written again without CODE_KEYS and with _floats()."""
    source, folder = Path(source), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for f in source.iterdir():
        if f.is_dir() or f.suffix == '.py' or f.name in ('config.json', 'tokenizer_config.json'):
            continue
        link = folder / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    for name in ('config.json', 'tokenizer_config.json'):
        if not (source / name).is_file():
            continue
        doc = _floats(json.loads((source / name).read_text()))
        for key in CODE_KEYS:
            doc.pop(key, None)
            if isinstance(doc.get('text_config'), dict):
                doc['text_config'].pop(key, None)
        (folder / name).write_text(json.dumps(doc, indent=2))
    return folder


def _extras(source, dest):
    """The checkpoint's small files the converter did not write (a chat template, a manifest), copied beside."""
    copied = []
    for f in Path(source).iterdir():
        if (f.is_dir() or f.suffix in ('.py', '.safetensors') or f.name.startswith('.')
                or f.name == 'model.safetensors.index.json' or (Path(dest) / f.name).exists()
                or f.stat().st_size > EXTRA_MOST_BYTES):
            continue
        shutil.copy2(f, Path(dest) / f.name)
        copied.append(f.name)
    return copied


def run(source, dest, bits, which, told, state_dir, python=sys.executable, command=None, every=2.0):
    """Convert, watch, finish (the module's docstring); the state written to state_dir/state.json throughout.
    command: the converter's argument list in place of mlx-lm's or mlx-vlm's (tests).  Returns the exit code."""
    source, dest, state_dir = Path(source), Path(dest), Path(state_dir)
    staging = dest.parent / f'.{dest.name}.staging'
    s = {'state': 'converting', 'pid': os.getpid(), 'source': str(source), 'dest': str(dest), 'bits': int(bits),
         'tool': which, 'told': int(told), 'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'error': None}

    def write():
        tmp = state_dir / f'state.json.{os.getpid()}.tmp'
        tmp.write_text(json.dumps(s))
        tmp.replace(state_dir / 'state.json')

    stopped = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.append('stopped (SIGTERM)'))
    write()
    child, made = None, False
    try:
        if dest.exists():
            raise RuntimeError(f'{dest} is there already')
        stage(source, staging)
        made = True
        cmd = command or [python, '-I', '-m', which, 'convert', '--hf-path', str(staging), '--mlx-path', str(dest),
                          '-q', '--q-bits', str(bits), '--q-group-size', str(GROUP), '--dtype', 'bfloat16']
        swap0 = _swap_gb()
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr)
        while child.poll() is None:
            time.sleep(every)
            swap = _swap_gb()
            if stopped:
                why = stopped[0]
            elif swap is not None and swap0 is not None and swap - swap0 > SWAP_CUT_GB:
                why = f'stopped: the swap grew {swap - swap0:.1f} GB'
            elif _bytes(dest) > CAP * told + SLACK:
                why = f'stopped: it wrote {_bytes(dest) / 1e9:.1f} GB, past {CAP:g} times the {told / 1e9:.1f} GB told'
            else:
                continue
            child.terminate()
            try:
                child.wait(30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise RuntimeError(why)
        if child.returncode:
            raise RuntimeError(f'the converter exited with {child.returncode}')
        copied = _extras(source, dest)
        (dest / MARKER).write_text(json.dumps({'source': str(source), 'bits': int(bits), 'group_size': GROUP,
                                               'tool': which, 'made': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                               'copied': copied}, indent=2))
        s.update(state='done', bytes=_bytes(dest), ended=time.strftime('%Y-%m-%dT%H:%M:%S'))
        write()
        return 0
    except Exception as e:                              # noqa: BLE001 -- told on the page
        s.update(state='failed', error=str(e), ended=time.strftime('%Y-%m-%dT%H:%M:%S'))
        if made and dest.exists():                      # (a folder there before: not ours, left)
            shutil.rmtree(dest, ignore_errors=True)
        write()
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    sys.exit(run(a[0], a[1], int(a[2]), a[3], int(a[4]), a[5]))
