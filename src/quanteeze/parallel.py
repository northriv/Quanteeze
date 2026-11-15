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
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from . import variables

GROUP = [None]       # the NCCL group of the ranks, once init() made it
_OWNED = [None]      # this rank's decoder layer range, once split() made it
_PEERS = []          # rank 0: a connection to each other rank, in order; the others: the one to rank 0
_SENT = {}           # rank 0: the variables as last sent
_LAST = [None]       # the last collective this rank issued (_collective)
_LAYER = re.compile(r'(?:^|\.)(?:language_model\.)?model\.layers\.(\d+)\.')
SHARED = ('QUANTEEZE_', 'MLX_VLM_')         # the variables rank 0 hands on as they change
OWN = ('QUANTEEZE_PARALLEL', 'QUANTEEZE_CONTROL_TOKEN')    # but these, each rank's


def size():
    return GROUP[0].size() if GROUP[0] is not None else 1


def rank():
    return GROUP[0].rank() if GROUP[0] is not None else 0


def owned():
    return _OWNED[0]


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


def sizes(path):
    """({decoder layer: bytes}, bytes of the rest) of the language model in the checkpoint at
    `path`, from its safetensors headers (an MTP head's left out: it is the drafter's; so is the
    n-gram table of a view whose config names a ple_storage manifest, read from the host's memory
    or the disk, never the GPU's: 5.75 GB of Flash-Next's layer 1)."""
    import json
    import struct
    config = json.loads((Path(path) / 'config.json').read_text()) if (Path(path) / 'config.json').is_file() else {}
    off_gpu = bool((config.get('text_config') or {}).get('ple_storage') or config.get('ple_storage'))
    layers, rest = {}, 0
    for file in sorted(Path(path).glob('*.safetensors')):
        with open(file, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(n))
        for key, tensor in header.items():
            if key == '__metadata__' or 'mtp' in key.split('.') or (off_gpu and '.ple_embedding.' in key):
                continue
            nbytes = tensor['data_offsets'][1] - tensor['data_offsets'][0]
            m = _LAYER.search(key)
            if m:
                layers[int(m.group(1))] = layers.get(int(m.group(1)), 0) + nbytes
            else:
                rest += nbytes
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
    for i, a in enumerate(argv):
        if a == '--model' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--model='):
            return a.partition('=')[2]
    return None


def plan(argv, gpus=None):
    """The GPUs to split the model over, [(name, bytes)], or None for one process; and why, a line."""
    mode = (variables.get('QUANTEEZE_PARALLEL') or 'auto').strip().lower()
    if mode == 'off':
        return None, 'QUANTEEZE_PARALLEL=off'
    if mode not in ('auto', 'layers'):
        raise SystemExit(f'quanteeze: QUANTEEZE_PARALLEL is auto, layers or off, not {mode!r} '
                         '(experts apart over the GPUs is not yet)')
    gpus = _gpus() if gpus is None else gpus
    want = (variables.get('QUANTEEZE_GPUS') or 'auto').strip().lower()
    if want != 'auto':
        gpus = gpus[:int(want)]
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
    from . import fitting
    share = variables.number('QUANTEEZE_WORKING_SET_SHARE')
    room = min(total for _, total in gpus) * share * (1 - fitting.reserve())
    layer_bytes, rest = sizes(Path(model).expanduser())
    whole = sum(layer_bytes.values()) + rest
    order = [layer_bytes.get(i, 0) for i in range(max(layer_bytes, default=-1) + 1)]
    cut = bounds(order, len(gpus))
    most = max(sum(order[cut[r]:cut[r + 1]]) for r in range(len(gpus))) + rest
    said = (f'{whole / 1e9:.1f} GB of weights, {room / 1e9:.1f} GB of room a GPU (QUANTEEZE_WORKING_SET_SHARE '
            f'{share:g}, QUANTEEZE_MEMORY_RESERVE {fitting.reserve():g})')
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
    logs = variables.path('QUANTEEZE_PARALLEL_LOGS')
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

def init(backend='nccl'):
    """In a rank (MLX_WORLD_SIZE set by launch()): the group (NCCL's; the tests' on the CPU, ring's),
    and the connections over which rank 0 hands on its requests.  Returns the group, or None in
    one process."""
    if int(os.environ.get('MLX_WORLD_SIZE', '1')) < 2:
        return None
    group = mx.distributed.init(backend=backend, strict=True)
    host = os.environ.get('NCCL_HOST_IP', '127.0.0.1')
    address = (host, variables.integer('QUANTEEZE_PARALLEL_PORT'))
    key = (variables.get('QUANTEEZE_PARALLEL_KEY') or '').encode()
    if group.rank() == 0:
        peers = {}
        with Listener(address, authkey=key) as listener:
            while len(peers) < group.size() - 1:
                conn = listener.accept()
                peers[conn.recv()] = conn
        _PEERS[:] = [peers[r] for r in range(1, group.size())]
    else:
        for attempt in range(600):
            try:
                conn = Client(address, authkey=key)
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise SystemExit(f'quanteeze: rank {group.rank()}: rank 0 never listened at {address}')
        conn.send(group.rank())
        _PEERS[:] = [conn]
    GROUP[0] = group
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


def _qwen4_exp(model, own):
    """Flash-Next: the masks made from this rank's first full-attention and linear layers'
    caches (the model's first of each may be another rank's), and the batch-invariant forward,
    which calls a layer's parts itself, passing over an absent layer and handing on after a
    run's last."""
    from mlx_vlm.models.qwen4_exp import language as q4
    inner = model.language_model.model
    try:
        inner.fa_idx = next(i for i in own if not inner.layers[i].is_linear)
        inner.ssm_idx = next(i for i in own if inner.layers[i].is_linear)
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


_ADAPTERS = {'qwen4_exp': _qwen4_exp}


def split(model):
    """This rank's run of `model`'s decoder layers, the others' absent (nothing of them read);
    its range, or None in one process."""
    if GROUP[0] is None:
        return None
    from mlx.utils import tree_flatten
    from mlx_vlm.models.cache import ArraysCache

    from . import arch
    kind = getattr(model.config, 'model_type', None)
    if kind not in _ADAPTERS:
        raise SystemExit(f'quanteeze: {kind!r} cannot be split over GPUs yet (quanteeze.parallel knows '
                         f'{sorted(_ADAPTERS)})')
    lm = arch.language_model(model)
    inner = lm.model
    layers = list(inner.layers)
    cut = bounds([sum(v.nbytes for _, v in tree_flatten(layer.parameters())) for layer in layers], size())
    own = range(cut[rank()], cut[rank() + 1])
    ends = {b - 1 for b in cut[1:]}
    for i, layer in enumerate(layers):
        if i not in own:
            layers[i] = Absent(layer.is_linear, i in ends)
        elif i in ends:
            layer.__class__ = _handing(type(layer))
    inner.layers = layers
    _ADAPTERS[kind](model, own)
    make_cache = lm.make_cache

    def own_caches():
        caches = make_cache()
        for i, layer in enumerate(layers):
            if isinstance(layer, Absent):
                caches[i] = ArraysCache(0)
        return caches
    lm.make_cache = own_caches
    print(f'quanteeze: rank {rank()} of {size()}: layers {own.start}-{own.stop - 1} of {len(layers)} '
          f'(runs from {cut[:-1]})', file=sys.stderr, flush=True)
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
