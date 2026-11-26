# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""One model served over several GPUs, a process each, its decoder layers split among them
(QUANTEEZE_PARALLEL; MLX's NCCL backend).

A model too large for one GPU's memory was served from the host's memory or the disk, its experts
read as they were routed to: Flash-Next on one 40 GB A100 decoded 0.6 to 0.9 tokens a second.
Split over four, each holding a quarter of its layers, it needs no expert read at all (a 42-token
prompt's second prefill took 0.5 s).  What the job's
hardware calls for is chosen by itself, and the page can choose too.

- launch(), at the server's start: with QUANTEEZE_PARALLEL auto, where the job sees two GPUs or
  more and the checkpoint's weights (--model) do not fit one's room but fit theirs, or with
  layers, the process starts one server a GPU (QUANTEEZE_GPUS of them, all by default) and waits
  on them, stopping them all when one stops.  Rank 0 serves on the given host and port; the
  others on a port of their own, no one asking, their output in QUANTEEZE_PARALLEL_LOGS.
- init(), in each: the NCCL group, and a connection from each rank to rank 0.
- split(model), as the model loads: the layers in runs of about equal bytes, a run a rank; the
  others' layers Absent, holding nothing.  Each run's last layer hands its state on: an all_sum
  every rank joins, the owner adding its state and the rest zeros, so every rank goes on with
  the same state and ends with the same logits.  An absent layer's cache is an empty ArraysCache,
  which every batch operation passes over.  A model type needs its own few lines (_ADAPTERS:
  which layers its masks are made from, and a forward that calls a layer's parts itself).
- install(generator_cls): every rank runs the server's generation loop (mlx-vlm's
  ResponseGenerator._run_impl), and each takes the same requests at the same step: rank 0's.  At
  each pass rank 0 sends what it took (each request pickled, its queue left behind), the
  requests cancelled, the QUANTEEZE_* and MLX_VLM_* variables changed, and a seed for the
  sampling when requests come; the others take that in place of their own queue.  The rest of
  the loop computes the same on every rank from the same logits.
- least(n): what depends on a rank's own memory (quanteeze.admission's rows) is the least of
  every rank's.

Not yet over several GPUs: another model than the one started with (a request or the page naming
one), a model in parts (its experts in the host's memory or on the disk), the prompt cache (APC,
whose eviction weighs each rank's bytes).  A request whose arguments do not
pickle (a structured response's processors) is answered with an error.
"""
import io
import os
import pickle
import queue
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from . import fitting, guards, per_load, per_thread, q2_experts, variables

GROUP = [None]       # the NCCL group of the ranks, once init() made it
_OWNED = [None]      # this rank's decoder layer range, once split() made it
_PEERS = []          # rank 0: a connection to each other rank, in order; the others: the one to rank 0
_SENT = {}           # rank 0: the variables as last sent
_LAST = per_thread.Field('collective')   # the last collective this thread issued (_collective)
_ON_HOST = [None]    # how many ranks run on this rank's host, once init() asked them all
_HOSTS = [None]      # every rank's host (its _host_id), once init() asked them all
_CUT = [None]        # the runs' first layers and the end, once split() made them
_LAYER = re.compile(r'^(?:language_model\.)?(?:model\.)?layers\.(\d+)\.')
_STACKED_EXPERT = re.compile(
    r'^(?:language_model\.)?(?:model\.)?layers\.(\d+)\.(.+)\.(gate_proj|up_proj|down_proj)\.(weight|scales?|biases)$'
)
_VISION_PREFIXES = ('vision.', 'aligner.', 'vision_tower.', 'model.visual.', 'visual.', 'model.vision_tower.')
_EMBED_KEYS = ('embed.weight', 'model.embed_tokens.weight', 'language_model.model.embed_tokens.weight')
SHARED = ('QUANTEEZE_', 'MLX_VLM_')         # the variables rank 0 hands on as they change
OWN = ('QUANTEEZE_PARALLEL', 'QUANTEEZE_CONTROL_TOKEN')    # but these, each rank's


def size():
    return GROUP[0].size() if GROUP[0] is not None else 1


def rank():
    return GROUP[0].rank() if GROUP[0] is not None else 0


def owned():
    """This rank's run of the decoder layers (a range), once the model is split; None in one process."""
    return _OWNED[0]


# -- the split's rule: a rank's work within its own layers ------------------------------------------
# Over several GPUs each rank works within its own run of the layers alone (owned()); what crosses
# between ranks is the state handed on at a run's end and the collectives every rank joins. A layer
# of another rank asked for is a crossing: refused, counted (crossings()), and logged once -- or
# raised as Crossing under QUANTEEZE_PARALLEL_CROSSING=raise.

_CROSSED = []        # each thread's own {what: how many times a layer of another rank was asked for it}
_MINE = threading.local()


class Crossing(RuntimeError):
    """A rank asked to work on a layer of another rank's run."""


def own_layer(j):
    """Whether decoder layer j belongs to this rank (every layer in one process)."""
    own = _OWNED[0]
    return own is None or int(j) in own


def keep(j, what):
    """Whether this rank may do `what` with layer j: its own, or one process; a layer of another rank refused
    (False), counted and logged the first time -- or raises Crossing under QUANTEEZE_PARALLEL_CROSSING=raise."""
    if own_layer(j):
        return True
    refuse(what, f"layer {j} is not this rank's (layers {_OWNED[0].start}-{_OWNED[0].stop - 1})")
    return False


def require(j, what):
    """As keep(), for work that cannot be skipped (e.g., reading an expert's arrays): a layer of another rank raises
    regardless of QUANTEEZE_PARALLEL_CROSSING."""
    if not keep(j, what):
        raise Crossing(f"quanteeze: rank {rank()} asked for layer {j}'s {what}, another rank's")


def refuse(what, why):
    """Record a split crossing in crossings(), log once, or raise Crossing under QUANTEEZE_PARALLEL_CROSSING=raise."""
    own = getattr(_MINE, 'counts', None)
    if own is None or not any(own is c for c in _CROSSED):
        own = _MINE.counts = {}
        _CROSSED.append(own)
    own[what] = own.get(what, 0) + 1
    n = crossings()[what]
    if (variables.get('QUANTEEZE_PARALLEL_CROSSING', variables.at_start()) or 'refuse').lower() == 'raise':
        raise Crossing(f'quanteeze: rank {rank()}: {what} crosses the split: {why}')
    if n == 1:
        print(f'quanteeze: rank {rank()}: {what} refused, it crosses the split: {why} (counted; '
              f'QUANTEEZE_PARALLEL_CROSSING=raise stops at it)', file=sys.stderr, flush=True)


def crossings():
    """`{what: times refused}` in this process (empty when nothing crossed)."""
    out = {}
    for counts in list(_CROSSED):
        for what, n in list(counts.items()):
            out[what] = out.get(what, 0) + n
    return out


def runs_here():
    """Layer ranges (runs) for ranks sharing this host, including this rank's, once split(); None in one process."""
    cut, hosts = _CUT[0], _HOSTS[0]
    if cut is None or _OWNED[0] is None:
        return None
    mine = hosts[rank()] if hosts else None
    return [range(cut[r], cut[r + 1]) for r in range(len(cut) - 1) if hosts is None or hosts[r] == mine]


def on_this_host():
    """How many ranks share this host's memory (populated in init(); defaults to size() before init)."""
    return _ON_HOST[0] or size()


def _host_id(name=None):
    import zlib
    return zlib.crc32((name or socket.gethostname()).encode()) & 0x7fffffff


# -- the launch ------------------------------------------------------------------------------------

def _gpus():
    """[(name, bytes)] of the GPUs this process may use: CUDA_VISIBLE_DEVICES's names where it is
    set, else nvidia-smi's UUIDs; none where nvidia-smi cannot say."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = [[f.strip() for f in line.split(',')] for line in out.splitlines() if line.count(',') == 2]
    gpus = [(uuid, int(mib) << 20, index) for index, uuid, mib in found]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return [(uuid, total) for uuid, total, _ in gpus]
    out = []
    for name in (v.strip() for v in visible.split(',') if v.strip()):
        match = [g for g in gpus if name in (g[2], g[0]) or (name.startswith('GPU-') and g[0].startswith(name))]
        if match:
            out.append((name, match[0][1]))
    return out


def _gpus_with_free():
    """[(name, total_bytes, free_bytes)] for GPUs this process may use, or `[]` when unavailable."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = [[f.strip() for f in line.split(',')] for line in out.splitlines() if line.count(',') == 3]
    gpus = [(uuid, int(total) << 20, int(free) << 20, index) for index, uuid, total, free in found]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return [(uuid, total, free) for uuid, total, free, _ in gpus]
    out = []
    for name in (v.strip() for v in visible.split(',') if v.strip()):
        match = [g for g in gpus if name in (g[3], g[0]) or (name.startswith('GPU-') and g[0].startswith(name))]
        if match:
            out.append((name, match[0][1], match[0][2]))
    return out


def _bits_target():
    raw = (variables.get('QUANTEEZE_BITS') or '').strip().lower()
    if raw in ('', 'off'):
        return None
    if raw in ('mxfp4', *fitting.GGML_MODES):
        return raw
    if raw.isdigit():
        return int(raw)
    return None


def _flag_value(argv, name):
    for i, a in enumerate(argv or ()):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + '='):
            return a.partition('=')[2]
    return None


def _experts_gguf_path(argv=None):
    raw = _flag_value(argv, '--experts-gguf') or variables.get('QUANTEEZE_EXPERTS_GGUF')
    if not raw or raw.strip().lower() == 'off':
        return None
    return Path(raw).expanduser()


def _config_meta(path):
    import json
    cfg_path = Path(path) / 'config.json'
    if not cfg_path.is_file():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return {}


def _tensor_elements(dtype, shape, scales_shape, q_bits, q_group):
    import math
    lead = math.prod(shape[:-1]) if len(shape) > 1 else 1
    cols = shape[-1]
    if dtype in ('BF16', 'F16', 'F32'):
        return lead * cols
    if dtype in ('I8', 'U8'):
        if scales_shape and cols * 2 == scales_shape[-1] * 32:
            return lead * cols * 2
        return lead * cols * (2 if q_bits == 4 else 1)
    if dtype == 'U32':
        if scales_shape and q_group:
            return lead * scales_shape[-1] * int(q_group)
        return lead * cols * (32 // int(q_bits or 4))
    return lead * cols


def sizes(path, experts_gguf=None, include_vision=False):
    """({decoder layer: bytes}, bytes of the rest) of the language model in the checkpoint at
    `path`, from its safetensors headers, accounting for off-GPU tables, `QUANTEEZE_BITS`, and
    `--experts-gguf` / `QUANTEEZE_EXPERTS_GGUF`."""
    import json
    import struct
    config = _config_meta(path)
    text_cfg = config.get('text_config') or {}
    model_type = config.get('model_type') or text_cfg.get('model_type')
    off_gpu = bool(text_cfg.get('ple_storage') or config.get('ple_storage'))
    quant_cfg = config.get('quantization') or text_cfg.get('quantization') or {}
    q_bits = quant_cfg.get('bits')
    q_group = quant_cfg.get('group_size')
    v41_img = (variables.get('QUANTEEZE_V41_READS_IMAGES') or 'on').strip().lower()
    skip_embed = model_type == 'deepseek_v41' or (variables.get('QUANTEEZE_EMBED_SSD') or '').strip().lower() == 'on'
    if include_vision:
        skip_vision = model_type == 'deepseek_v41' and v41_img == 'off'
    else:
        from . import vision_ssd
        skip_vision = (
            (model_type == 'deepseek_v41' and v41_img in ('off', 'ssd'))
            or vision_ssd.enabled(path)
        )
    gguf_path = Path(experts_gguf).expanduser() if experts_gguf is not None else _experts_gguf_path()
    target_bits = None if gguf_path is not None else _bits_target()

    layers, rest = {}, 0
    headers = []
    all_shapes = {}
    for file in sorted(Path(path).glob('*.safetensors')):
        with open(file, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(n))
        headers.append(header)
        for key, tensor in header.items():
            if key != '__metadata__':
                all_shapes[key] = tuple(tensor['shape'])

    for header in headers:
        for key, tensor in header.items():
            if key == '__metadata__' or 'mtp' in key.split('.') or '.engram.embed.' in key:
                continue
            if off_gpu and '.ple_embedding.' in key:
                continue
            if skip_embed and key in _EMBED_KEYS:
                continue
            if skip_vision and key.startswith(_VISION_PREFIXES):
                continue

            shape = tuple(tensor['shape'])
            pm = q2_experts.PER_EXPERT_RE.match(key)
            sm = None if pm else (_STACKED_EXPERT.match(key) if len(shape) >= 3 else None)
            if pm or sm:
                layer_idx = int((pm or sm).group(2 if pm else 1))
                part = (pm.group(6) if pm else sm.group(4))
                if gguf_path is not None:
                    continue
                if target_bits is not None:
                    if part == 'weight':
                        if pm:
                            skey_s = key.rsplit('.', 1)[0] + '.scales'
                            skey_1 = key.rsplit('.', 1)[0] + '.scale'
                            scales_shape = all_shapes.get(skey_s) or all_shapes.get(skey_1)
                        else:
                            scales_shape = all_shapes.get(key.rsplit('.', 1)[0] + '.scales')
                        elems = _tensor_elements(tensor['dtype'], shape, scales_shape, q_bits, q_group)
                        layers[layer_idx] = layers.get(layer_idx, 0) + int(fitting.quantized_bytes(elems, target_bits))
                    continue

            nbytes = tensor['data_offsets'][1] - tensor['data_offsets'][0]
            m = _LAYER.search(key)
            if m:
                layers[int(m.group(1))] = layers.get(int(m.group(1)), 0) + nbytes
            else:
                rest += nbytes

    if gguf_path is not None:
        from . import gguf_experts
        eg = gguf_experts.ExpertGGUF(gguf_path)
        for layer_idx, name in eg.routed(layers=True):
            layers[layer_idx] = layers.get(layer_idx, 0) + eg.nbytes(name)
    return layers, rest


def bounds(layer_bytes, n):
    """The first layer of each of n runs of about equal bytes, and the end: [0, ..., len]."""
    total, out, run = sum(layer_bytes), [0], 0
    for i, b in enumerate(layer_bytes):
        run += b
        if len(out) < n and run >= total * len(out) / n and i + 1 < len(layer_bytes):
            out.append(i + 1)
    while len(out) < n:                          # fewer layers than ranks: never, but not empty runs
        out.append(len(layer_bytes))
    return out + [len(layer_bytes)]


def _model_argument(argv):
    return _flag_value(argv, '--model')


def plan(argv, gpus=None):
    """The GPUs to split the model over, [(name, bytes)], or None for one process; and why, a line."""
    mode = (variables.get('QUANTEEZE_PARALLEL', variables.at_start()) or 'auto').strip().lower()
    if mode == 'off':
        return None, 'QUANTEEZE_PARALLEL=off'
    if mode not in ('auto', 'layers'):
        raise SystemExit(f'quanteeze: QUANTEEZE_PARALLEL is auto, layers or off, not {mode!r} '
                         '(experts apart over the GPUs is not yet)')
    want = (variables.get('QUANTEEZE_GPUS', variables.at_start()) or 'auto').strip().lower()
    if gpus is None:
        entries = _gpus_with_free() or [(n, t, t) for n, t in _gpus()]
    else:
        entries = [(g[0], g[1], g[2] if len(g) > 2 else g[1]) for g in gpus]
    if want != 'auto':
        entries = entries[:int(want)]
    gpus = [(n, t) for n, t, _ in entries]
    model = _model_argument(argv)
    why = None
    if len(gpus) < 2:
        why = f'{len(gpus)} GPU here'
    elif model is None or not Path(model).expanduser().is_dir():
        why = 'no checkpoint directory given (--model)'
    if why:
        if mode == 'layers':
            raise SystemExit(f'quanteeze: QUANTEEZE_PARALLEL=layers: {why}')
        return None, why
    model_path = Path(model).expanduser()
    config = _config_meta(model_path)
    model_type = config.get('model_type') or (config.get('text_config') or {}).get('model_type')
    if model_type and model_type not in _ADAPTERS:
        msg = f'{model_type!r} cannot be split over GPUs yet (quanteeze.parallel knows {sorted(_ADAPTERS)})'
        if mode == 'layers':
            raise SystemExit(f'quanteeze: {msg}')
        return None, msg
    share = guards.working_set_share()
    if not variables.is_set('QUANTEEZE_MEMORY_RESERVE') and model_type in fitting.RESERVE_ON_CUDA:
        reserve = fitting.RESERVE_ON_CUDA[model_type]
    else:
        reserve = fitting.reserve()
    usable = min((free if share is None else total * share) for _, total, free in entries)
    room = usable * (1 - reserve)
    layer_bytes, rest = sizes(model_path, experts_gguf=_experts_gguf_path(argv))
    whole = sum(layer_bytes.values()) + rest
    order = [layer_bytes.get(i, 0) for i in range(max(layer_bytes, default=-1) + 1)]
    cut = bounds(order, len(gpus))
    most = max(sum(order[cut[r]:cut[r + 1]]) for r in range(len(gpus))) + rest
    share_label = 'auto' if share is None else f'{share:g}'
    said = (f'{whole / 1e9:.1f} GB of weights, {room / 1e9:.1f} GB of room a GPU (QUANTEEZE_WORKING_SET_SHARE '
            f'{share_label}, QUANTEEZE_MEMORY_RESERVE {reserve:g})')
    if mode == 'auto' and whole <= room:
        return None, f'{said}: one GPU holds it'
    if most > room:
        if mode == 'layers':
            print(f'quanteeze: WARNING QUANTEEZE_PARALLEL=layers: {said}; split over {len(gpus)} GPUs one holds '
                  f'{most / 1e9:.1f} GB', file=sys.stderr, flush=True)
        else:
            return None, f'{said}: split over {len(gpus)} GPUs one would hold {most / 1e9:.1f} GB'
    return gpus, f'{said}: layers {cut} over {len(gpus)} GPUs, the most one holds {most / 1e9:.1f} GB'


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _own_port(argv, port):
    """argv with --host 127.0.0.1 and --port `port`: a rank that serves no one."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in ('--port', '--host'):
            skip = True
            continue
        if a.startswith(('--port=', '--host=')):
            continue
        out.append(a)
    return out + ['--host', '127.0.0.1', '--port', str(port)]


def launch(argv=None):
    """Where plan() splits the model: start a server a GPU and wait on them (never returns; exits
    with rank 0's status).  A rank, or one process: returns -- one process where the job sees
    several GPUs after running again on the first alone (_alone)."""
    if os.environ.get('MLX_WORLD_SIZE'):
        return
    argv = sys.argv[1:] if argv is None else argv
    seen = _gpus()
    gpus, why = plan(argv, seen)
    if not gpus:
        print(f'quanteeze: one GPU process: {why}', file=sys.stderr, flush=True)
        if len(seen) > 1:
            _alone(seen[0][0], argv)
        return
    logs = variables.path('QUANTEEZE_PARALLEL_LOGS', variables.at_start())
    logs.mkdir(parents=True, exist_ok=True)
    common = dict(os.environ, MLX_WORLD_SIZE=str(len(gpus)), NCCL_HOST_IP='127.0.0.1', NCCL_PORT=str(_free_port()),
                  QUANTEEZE_PARALLEL='layers', QUANTEEZE_PARALLEL_PORT=str(_free_port()),
                  QUANTEEZE_PARALLEL_KEY=secrets.token_hex(16))
    print(f'quanteeze: {why}: a server a GPU, the others\' output in {logs}', file=sys.stderr, flush=True)
    children = []
    for r, (name, _) in enumerate(gpus):
        env = dict(common, MLX_RANK=str(r), CUDA_VISIBLE_DEVICES=name)
        out = None
        if r:
            env.setdefault('QUANTEEZE_CONTROL_TOKEN', secrets.token_hex(16))    # rank 0's token file stays
            out = open(logs / f'rank{r}.log', 'w')
        command = [sys.executable, '-c', 'from quanteeze.serve import main; main()',
                   *(argv if r == 0 else _own_port(argv, _free_port()))]
        children.append(subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT if out else None))
    sys.exit(_supervise(children))


def _alone(name, argv):
    """Run this server again seeing GPU `name` alone (exec: never returns).  One process uses one
    GPU, but MLX's CUDA backend, seeing several, broke it: mx.clear_cache() makes each GPU current
    in turn and leaves the last, and an array read on the host is then copied off GPU 0 on that one
    -- "cudaMemcpy(data, buf.data, buf.size, kind) failed: invalid argument", the server gone
    (Flash-Next in parts on four A100s seen; clear_cache() then np.asarray() alone aborts so with four seen, never with one).  CUDA reads CUDA_VISIBLE_DEVICES once, as MLX starts it, hence a new process."""
    print(f'quanteeze: the server again on {name} alone (CUDA_VISIBLE_DEVICES)', file=sys.stderr, flush=True)
    os.execve(sys.executable, [sys.executable, '-c', 'from quanteeze.serve import main; main()', *argv],
              dict(os.environ, CUDA_VISIBLE_DEVICES=name))


def _supervise(children, poll=1.0, grace=60.0):
    """Wait until one of `children` ends, then stop the rest (SIGTERM, SIGKILL after `grace`
    seconds); a SIGTERM or SIGINT here is passed on to each.  Returns the first's status."""
    def forward(sig, frame):
        for c in children:
            if c.poll() is None:
                c.send_signal(sig)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    while all(c.poll() is None for c in children):
        time.sleep(poll)
    first = children[0] if children[0].poll() is not None else next(c for c in children if c.poll() is not None)
    if any(c.poll() is None for c in children):
        print(f'quanteeze: rank {children.index(first)} ended ({first.returncode}): stopping the others',
              file=sys.stderr, flush=True)
    for c in children:
        if c.poll() is None:
            c.terminate()
    end = time.monotonic() + grace
    for c in children:
        try:
            c.wait(max(0.0, end - time.monotonic()))
        except subprocess.TimeoutExpired:
            c.kill()
            c.wait()
    return first.returncode


# -- each rank -------------------------------------------------------------------------------------

def no_delay(conn):
    """Enable TCP_NODELAY on a connection's socket to avoid Nagle/delayed-ACK stalls on multi-write messages."""
    try:
        sock = socket.socket(fileno=os.dup(conn.fileno()))
    except (OSError, ValueError):
        return conn
    try:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    finally:
        sock.close()
    return conn


def init(backend='nccl'):
    """In a rank (MLX_WORLD_SIZE set by launch()): the group (NCCL's; the tests' on the CPU, ring's),
    and the connections over which rank 0 hands on its requests.  Returns the group, or None in
    one process."""
    if int(os.environ.get('MLX_WORLD_SIZE', '1')) < 2:
        return None
    group = mx.distributed.init(backend=backend, strict=True)
    host = os.environ.get('NCCL_HOST_IP', '127.0.0.1')
    address = (host, variables.integer('QUANTEEZE_PARALLEL_PORT', variables.at_start()))
    key = (variables.get('QUANTEEZE_PARALLEL_KEY', variables.at_start()) or '').encode()
    if group.rank() == 0:
        peers = {}
        with Listener(address, authkey=key) as listener:
            while len(peers) < group.size() - 1:
                conn = no_delay(listener.accept())
                peers[conn.recv()] = conn
        _PEERS[:] = [peers[r] for r in range(1, group.size())]
    else:
        for attempt in range(600):
            try:
                conn = no_delay(Client(address, authkey=key))
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise SystemExit(f'quanteeze: rank {group.rank()}: rank 0 never listened at {address}')
        conn.send(group.rank())
        _PEERS[:] = [conn]
    GROUP[0] = group
    mine = _host_id()
    hosts = mx.zeros((group.size(),), mx.int32).at[group.rank()].add(mine)
    hosts = _collective(hosts, lambda x: mx.distributed.all_sum(x, group=group)).tolist()
    _ON_HOST[0] = hosts.count(mine)
    _HOSTS[0] = hosts
    print(f'quanteeze: rank {group.rank()} of {group.size()} on {socket.gethostname()}, {_ON_HOST[0]} rank(s) '
          f'on this host, {len(set(hosts))} host(s)', file=sys.stderr, flush=True)
    return group


def _collective(x, op):
    """op(x), a collective, made to wait on the one before: MLX evaluates a graph's independent
    branches in an order of its own -- two rows' prefills, say -- and a rank whose branches differ
    (its layers absent) may take them in another; ring's all_sums then waited on each other for
    good.  Chained, every rank issues them in its program's order, the same."""
    if _LAST[0] is not None:
        x = mx.depends(x, _LAST[0])
    _LAST[0] = out = op(x)
    return out


@per_load.on_let_go
def forget_chain():
    """Forget the calling thread's collective chain (_collective) on pass failure or model unload."""
    _LAST[0] = None


def _forgetting(call):
    """Wrap `call` so an exception forgets the calling thread's collective chain before propagating."""
    def forgetting(*args, **kwargs):
        try:
            return call(*args, **kwargs)
        except BaseException:
            forget_chain()
            raise
    forgetting.__wrapped__ = call
    return forgetting


def least(value):
    """The least of the ranks' `value`, an int each asks for at the same step (one process: it)."""
    if GROUP[0] is None:
        return value
    mine = mx.zeros((size(),), mx.int32).at[rank()].add(int(value))     # all_sum: what NCCL was seen to do
    return int(_collective(mine, lambda x: mx.distributed.all_sum(x, group=GROUP[0])).min().item())


def hand_on(h, owner):
    """The state every rank goes on with at the end of a rank's run of layers: an all_sum to which
    the `owner` adds `h` and the rest zeros."""
    return _collective(h if owner else mx.zeros_like(h), lambda x: mx.distributed.all_sum(x, group=GROUP[0]))


class Absent(nn.Module):
    """A decoder layer another rank runs: the state passes, and where that rank's run ends, the
    state it hands on is taken."""

    def __init__(self, is_linear, hands_on):
        super().__init__()
        self.is_linear, self.hands_on = is_linear, hands_on

    def __call__(self, h, *args, **kwargs):
        return hand_on(h, False) if self.hands_on else h


def _handing(cls):
    """`cls` whose call hands the state on after it (the last layer of a rank's run)."""
    def __call__(self, *args, **kwargs):
        return hand_on(cls.__call__(self, *args, **kwargs), True)
    return type(cls.__name__, (cls,), {'__call__': __call__, 'quanteeze_hands_on': True})


def _qwen4_exp(model, holder, layers, cut, own, ends):
    """Flash-Next: the masks made from this rank's first full-attention and linear layers'
    caches (the model's first of each may be another rank's), and the batch-invariant forward,
    which calls a layer's parts itself, passing over an absent layer and handing on after a
    run's last."""
    from mlx_vlm.models.qwen4_exp import language as q4
    for i, layer in enumerate(layers):
        if i not in own:
            layers[i] = Absent(layer.is_linear, i in ends)
        elif i in ends:
            layer.__class__ = _handing(type(layer))
    holder.layers = layers
    try:
        holder.fa_idx = next(i for i in own if not holder.layers[i].is_linear)
        holder.ssm_idx = next(i for i in own if holder.layers[i].is_linear)
    except StopIteration:
        raise SystemExit(f'quanteeze: rank {rank()}\'s layers {own.start}-{own.stop - 1} lack a full-attention or '
                         'a linear layer, whose caches the masks are made from: fewer GPUs (QUANTEEZE_GPUS)')
    forward = q4.Qwen4ExpBatchInvariantForward
    if getattr(forward._layer, 'quanteeze', False):
        return
    layer_of = forward._layer

    def _layer(self, layer, hidden, *args):
        if isinstance(layer, Absent):
            return layer(hidden)
        out = layer_of(self, layer, hidden, *args)
        return hand_on(out, True) if getattr(layer, 'quanteeze_hands_on', False) else out
    _layer.quanteeze = True
    forward._layer = _layer
    forward.__call__ = _forgetting(forward.__call__)


class _AbsentAttn:
    indexer = None


class _GlmAbsent(Absent):
    """Absent `glm5_next` decoder layer forwarding `(hidden, topk)` tuple."""

    def __init__(self, hands_on):
        super().__init__(False, hands_on)
        self.block_type = 'absent'
        self.self_attn = _AbsentAttn()
        self.mlp = None
        self.engram = None

    def __call__(self, h, mask=None, cache=None, prev_topk_indices=None, *args, **kwargs):
        return (hand_on(h, False), None) if self.hands_on else (h, prev_topk_indices)


def _glm_handing(cls):
    """`glm5_next` decoder layer subclass handing its hidden output across ranks."""
    def __call__(self, *args, **kwargs):
        h, topk = cls.__call__(self, *args, **kwargs)
        return hand_on(h, True), topk
    return type(cls.__name__, (cls,), {'__call__': __call__, 'quanteeze_hands_on': True})


def _glm5_next(model, holder, layers, cut, own, ends):
    """GLM-5.3: verify the rank's first full-attention layer owns an indexer and wrap layers."""
    first_attn = next((i for i in own if not getattr(layers[i], 'is_linear', False)), None)
    if first_attn is not None and getattr(getattr(layers[first_attn], 'self_attn', None), 'indexer', None) is None:
        raise SystemExit(
            f'quanteeze: rank {rank()}\'s layers {own.start}-{own.stop - 1} start attention at layer {first_attn}, '
            'which shares top-k indices from an earlier rank: fewer GPUs (QUANTEEZE_GPUS)'
        )
    for i, layer in enumerate(layers):
        if i not in own:
            layers[i] = _GlmAbsent(i in ends)
        elif i in ends:
            layer.__class__ = _glm_handing(type(layer))
    holder.layers = layers


def _v41_hand_on(h, pre_mix, entry, next_cut, run_start, config, n_layers, owner):
    """Hand `deepseek_v41` hidden states, pre-mix, and any cross-rank KV/indexer state across ranks."""
    h = hand_on(h, owner)
    pre_mix = hand_on(pre_mix, owner)
    if (
        0 < next_cut < n_layers
        and entry is not None
        and config.compress_ratios[next_cut] > 0
        and next_cut not in config.kv_source_layer_ids
    ):
        s = max((k for k in config.kv_source_layer_ids if k < next_cut), default=None)
        if s is not None:
            rows = entry.rows if hasattr(entry, 'rows') else [entry]
            b = 1 if hasattr(entry, 'rows') else h.shape[0]
            width = h.shape[1]
            ratio = config.compress_ratios[s]
            need_kv = run_start <= s < next_cut
            cand_id = config.candidate_source_layer_id
            need_cand = 0 <= cand_id < next_cut and run_start <= cand_id < next_cut
            idx_s = (
                None if next_cut in config.index_source_layer_ids
                else max((k for k in config.index_source_layer_ids if k < next_cut), default=None)
            )
            need_topk = idx_s is not None and run_start <= idx_s < next_cut
            if need_kv or need_cand or need_topk:
                for r_idx, row in enumerate(rows):
                    start, stop = entry._bounds(r_idx, width) if hasattr(entry, '_bounds') else (0, width)
                    if start == stop:
                        continue
                    seqlen = stop - start
                    start_pos, end_pos = row.offset, row.offset + seqlen
                    comp_len = end_pos // ratio
                    n_latent = comp_len - (start_pos // ratio)
                    if need_kv:
                        if n_latent > 0:
                            pkv = hand_on(
                                row.compress[s][:, -n_latent:].astype(h.dtype)
                                if owner else mx.zeros((b, n_latent, config.head_dim), dtype=h.dtype),
                                owner,
                            )
                            ik = hand_on(
                                row.keys[s][:, -n_latent:].astype(h.dtype)
                                if owner else mx.zeros((b, n_latent, config.index_head_dim), dtype=h.dtype),
                                owner,
                            )
                            if not owner:
                                row.append_compressed(s, pkv)
                                row.append_index_keys(s, ik)
                        if not owner:
                            row.compress_kv = row.compress[s]
                            row.index_k = row.keys[s]
                    if need_cand:
                        cand_len = end_pos // config.compress_ratios[cand_id]
                        if cand_len > 0:
                            cand = hand_on(
                                row.candidates.astype(mx.int32)
                                if owner else mx.zeros((b, seqlen, cand_len), dtype=mx.int32),
                                owner,
                            )
                            if not owner:
                                row.candidates = cand.astype(mx.bool_)
                    if need_topk:
                        idx_len = end_pos // config.compress_ratios[idx_s]
                        if idx_len > 0:
                            tk = min(config.index_topk, idx_len)
                            t_idxs = hand_on(
                                row.topk_idxs if owner else mx.zeros((b, seqlen, tk), dtype=mx.int32),
                                owner,
                            )
                            if not owner:
                                row.topk_idxs = t_idxs
    return h, pre_mix


class _V41Absent(Absent):
    """Absent `deepseek_v41` block forwarding `(h, pre_mix)` and receiving cross-rank cache state."""

    def __init__(self, hands_on, next_cut=0, run_start=0, config=None, n_layers=0):
        super().__init__(False, hands_on)
        self.engram = None
        self.next_cut = next_cut
        self.run_start = run_start
        self.config = config
        self.n_layers = n_layers

    def __call__(self, h, pre_mix, image_mask=None, cache=None, *args, **kwargs):
        if not self.hands_on:
            return h, pre_mix
        return _v41_hand_on(h, pre_mix, cache, self.next_cut, self.run_start, self.config, self.n_layers, False)


def _v41_handing(cls, next_cut, run_start, config, n_layers):
    """`deepseek_v41` block subclass handing `h`, `pre_mix`, and shared cache state to later ranks."""
    def __call__(self, h, pre_mix, image_mask=None, cache=None, *args, **kwargs):
        out_h, out_mix = cls.__call__(self, h, pre_mix, image_mask, cache, *args, **kwargs)
        return _v41_hand_on(out_h, out_mix, cache, next_cut, run_start, config, n_layers, True)
    return type(cls.__name__, (cls,), {'__call__': __call__, 'quanteeze_hands_on': True})


def _deepseek_v41(model, holder, layers, cut, own, ends):
    """DeepSeek V4.1: wrap absent/boundary blocks with cross-rank state and compressed-KV handoffs."""
    config = getattr(holder, 'config', None) or getattr(model, 'config', None)
    n_layers = len(layers)
    runs = {cut[r + 1] - 1: (cut[r], cut[r + 1]) for r in range(len(cut) - 1) if cut[r + 1] > cut[r]}
    for i, layer in enumerate(layers):
        if i not in own:
            if i in ends:
                run_start, next_cut = runs[i]
                layers[i] = _V41Absent(True, next_cut, run_start, config, n_layers)
            else:
                layers[i] = _V41Absent(False)
        elif i in ends:
            run_start, next_cut = runs[i]
            layer.__class__ = _v41_handing(type(layer), next_cut, run_start, config, n_layers)
    holder.layers = layers


_ADAPTERS = {'qwen4_exp': _qwen4_exp, 'glm5_next': _glm5_next, 'deepseek_v41': _deepseek_v41}


def _passes_forgetting(lm):
    """Wrap `lm`'s class so an exception in a pass forgets the thread's collective chain."""
    cls = type(lm)
    if not getattr(cls, '_quanteeze_forgetting', False):
        lm.__class__ = type(cls.__name__, (cls,), {'__call__': _forgetting(cls.__call__), '_quanteeze_forgetting': True})


def split(model):
    """This rank's run of `model`'s decoder layers, the others' absent (nothing of them read);
    its range, or None in one process."""
    if GROUP[0] is None:
        return None
    from mlx.utils import tree_flatten
    from mlx_vlm.models.cache import ArraysCache

    from . import arch
    config = getattr(model, 'config', None) or getattr(model, 'args', None)
    kind = getattr(config, 'model_type', None) if config is not None else None
    if kind not in _ADAPTERS:
        raise SystemExit(f'quanteeze: {kind!r} cannot be split over GPUs yet (quanteeze.parallel knows '
                         f'{sorted(_ADAPTERS)})')
    lm = arch.language_model(model)
    holder = lm.model if getattr(lm, 'model', None) is not None and hasattr(lm.model, 'layers') else lm
    layers = list(holder.layers)
    cut = bounds(
        [sum(v.nbytes for k, v in tree_flatten(layer.parameters()) if arch.ROW_BY_ROW not in k) for layer in layers],
        size(),
    )
    own = range(cut[rank()], cut[rank() + 1])
    ends = {b - 1 for b in cut[1:] if b > 0}
    _ADAPTERS[kind](model, holder, layers, cut, own, ends)
    make_cache = lm.make_cache

    def own_caches():
        caches = make_cache()
        if len(caches) == len(layers) and kind != 'deepseek_v41':
            for i, layer in enumerate(layers):
                if isinstance(layer, Absent):
                    caches[i] = ArraysCache(0)
        return caches
    lm.make_cache = own_caches
    _passes_forgetting(lm)
    print(f'quanteeze: rank {rank()} of {size()}: layers {own.start}-{own.stop - 1} of {len(layers)} '
          f'(runs from {cut[:-1]})', file=sys.stderr, flush=True)
    _CROSSED.clear()
    _CUT[0] = cut
    _OWNED[0] = own
    return own


# -- the generation loop in step -------------------------------------------------------------------

class _Sink:
    """A rank's stand-in for a request's queue: what its generation hands back goes nowhere."""

    def put(self, item, *args, **kwargs):
        pass

    put_nowait = put


class _Pickler(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, (queue.Queue, _Sink)):
            return _Sink, ()
        return NotImplemented


def _dumps(obj):
    out = io.BytesIO()
    _Pickler(out).dump(obj)
    return out.getvalue()


def _changed():
    """The shared variables changed since last sent: ({name: value}, [names unset]), or None."""
    now = {k: v for k, v in os.environ.items() if k.startswith(SHARED) and not k.startswith(OWN)}
    set_ = {k: v for k, v in now.items() if _SENT.get(k) != v}
    unset = [k for k in _SENT if k not in now]
    _SENT.clear()
    _SENT.update(now)
    return (set_, unset) if set_ or unset else None


def _send(message):
    data = _dumps(message)
    for conn in _PEERS:
        conn.send_bytes(data)


def install(generator_cls):
    """Every rank's generation loop taking rank 0's requests and cancellations, pass by pass."""
    if GROUP[0] is None or getattr(generator_cls, '_quanteeze_parallel', False):
        return
    collect, drain = generator_cls._collect_pending_requests, generator_cls._drain_cancellations

    def _collect_lead(self, *args, **kwargs):
        pending, stop = collect(self, *args, **kwargs)
        taken, shared = [], []
        for request in pending:
            try:
                shared.append(_dumps(request))
            except Exception as e:                 # a structured response's processors, say
                request.rqueue.put(ValueError(f'quanteeze: this request cannot be served over several GPUs: {e}'))
                continue
            taken.append(request)
        cancelled = drain(self)
        self.__dict__['_quanteeze_cancelled'] = cancelled
        seed = int.from_bytes(os.urandom(4), 'little') if taken else None
        _send((shared, stop, cancelled, _changed(), seed))
        if seed is not None:
            mx.random.seed(seed)
        return taken, stop

    def _collect_follow(self, *args, **kwargs):
        try:
            shared, stop, cancelled, changed, seed = pickle.loads(_PEERS[0].recv_bytes())
        except (EOFError, OSError):                # rank 0 has gone: so does this one
            shared, stop, cancelled, changed, seed = [], True, set(), None, None
            os.kill(os.getpid(), signal.SIGTERM)
        if changed is not None:
            os.environ.update(changed[0])
            for name in changed[1]:
                os.environ.pop(name, None)
        self.__dict__['_quanteeze_cancelled'] = cancelled
        if stop:
            self._stop = True
        if seed is not None:
            mx.random.seed(seed)
        return [pickle.loads(blob) for blob in shared], stop

    def _drain(self):
        """The cancellations rank 0 took at this pass (the loop drains them right after it takes
        requests); a rank's own are none."""
        return self.__dict__.pop('_quanteeze_cancelled', None) or set()

    generator_cls._collect_pending_requests = _collect_lead if rank() == 0 else _collect_follow
    generator_cls._drain_cancellations = _drain
    generator_cls._quanteeze_parallel = True
