# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Multi-Token Prediction (MTP) and DSpark speculative decoding (`quanteeze.mtp`).

mlx-vlm drafts with a checkpoint's own next-token-prediction layer -- loaded as a drafter model of
its own (`load_drafter`: its attention, its MoE of every expert and its shared experts; the `lm_head`
is the target's), and verifies the draft tokens with the target in one short pass, rolling the
caches back past a rejected one.  The target's output is unchanged as far as the verify pass
computes what one-token decodes would: sampling keeps its distribution, and greedy decoding its
tokens where the pass is exact (`quanteeze.verify_qmv`).

- `QUANTEEZE_MTP=auto` (the default): draft where the checkpoint bundles a head mlx-vlm splits out
  (`head()`) and its model type's verify pass is verified (`VERIFIED`: GLM-5.3, Flash-Next, and
  DeepSeek V4.1 with `QUANTEEZE_MTP_EXACT=bits`); `on`: wherever a head loads; `off`: never.
- `load()`: the drafter read from the served checkpoint, its routed experts lowered by least squares
  to `bits` (`QUANTEEZE_MTP_BITS`, default 2).
- `enable()`: from the next request on, the server's generator drafts with it.
- `set_block()`: the tokens a round verifies, `QUANTEEZE_MTP_BLOCK`.
- `summary()`: rounds, tokens drafted and accepted, and the acceptance rate.
- DeepSeek V4.1 drafts with its own DSpark head (`quanteeze.dspark`: mlx-vlm's DFlash rounds, the
  target's taps and rollback in `quanteeze.arch.deepseek_v41`); `head()` and `load()` find it first.
- `filter_caches()`: when a request of a batch finishes, cuts all per-row arrays in the drafter's
  caches to the remaining rows.
"""
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx

from . import requant, variables

ACTIVE = [None]                       # (drafter, kind) once enabled
CHUNK = 32                            # experts lowered at a time
BLOCKS = {'glm5_next': 2,             # QUANTEEZE_MTP_BLOCK=auto: the block a round, by the target's model type;
          'deepseek_v41': 6}          # Flash-Next (qwen4_exp) leaves runtime_block_size unset for adaptive depth (4 / DFlash2: 5)
OVERRIDE_BLOCK = [None]               # runtime draft_block_size override when requested block > configured depth
DFLASH2_REPOS = {
    'qwen4_exp': 'incoai/Qwen3.8-Flash-Next-DFlash2',
    'qwen3_5': 'z-lab/Qwen3.8-27B-DFlash2',
}
VERIFIED = ('glm5_next', 'qwen4_exp',
            'deepseek_v41')
SLOWER_IN_MEMORY = ()                 # not enabled under auto once every expert is in memory
SLOWER_IN_MEMORY_CUDA = ('qwen4_exp',)   # not enabled on CUDA once every expert is in memory
SLOWER_IN_MEMORY_WHY = ('its rounds cost more than the tokens they win with every expert in memory; '
                        'on demand from SSD, where the reads wait, it drafts')
GATHER_MS = 2000                      # requests gathered before a drafting batch starts (MLX_VLM_SPEC_BATCH_COALESCE_MS
                                      # unset): bursts can arrive over a second, and a batch takes none later
BURST_S = 300                         # ... only within this of two requests coming GATHER_MS apart or less: one
                                      # asked alone, as in a chat, starts at once
WAIT_S = 3600                         # a request waiting for a drafting batch to end: given up after this with no
                                      # token (mlx-vlm's 600 s otherwise)
_COMING = {'last': None, 'burst': None}    # when the last request came, and the last two together (monotonic)


def head(path):
    """The tensors of an MTP head the checkpoint at `path` bundles, as mlx-vlm splits one out for a
    drafter (0: none; a checkpoint mlx-vlm splits no head from, or one not readable, has none)."""
    from mlx_vlm.speculative.drafters.mtp_split import get_mtp_splitter
    from . import dspark
    if dspark.available(path):
        return dspark.heads(path)
    try:
        config = json.loads((Path(path) / 'config.json').read_text())
        splitter = get_mtp_splitter(source_config=config)
        if splitter is None:
            return 0
        text = splitter.read_text_config(config)
        found = 0
        for file in Path(path).glob('*.safetensors'):
            with open(file, 'rb') as f:
                header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
            found += sum(1 for key in header if key != '__metadata__' and splitter.select_keys(key, text))
        return found
    except (OSError, ValueError, KeyError, NotImplementedError):
        return 0


def auto(path, kind):
    """(whether QUANTEEZE_MTP=auto drafts for the checkpoint at `path` of model type `kind`, why)."""
    if not head(path):
        return False, 'the checkpoint bundles no MTP head'
    if kind not in VERIFIED:
        return False, f'{kind}\'s verify pass is not checked exact here (QUANTEEZE_MTP=on drafts all the same)'
    return True, 'the checkpoint bundles an MTP head'


def pays_in_memory(kind):
    """Whether QUANTEEZE_MTP=auto drafts for model type `kind` once every expert is in memory:
    not those of SLOWER_IN_MEMORY, nor on CUDA those of SLOWER_IN_MEMORY_CUDA."""
    if kind in SLOWER_IN_MEMORY:
        return False
    return mx.metal.is_available() or kind not in SLOWER_IN_MEMORY_CUDA


def _local_external(repo):
    """Return local checkpoint path for `repo` if present in model roots or HF cache, else `repo`."""
    from . import home
    expanded = Path(repo).expanduser()
    if expanded.is_dir():
        return str(expanded.resolve())
    for root in home.model_roots():
        candidate = root / repo
        if candidate.is_dir() and (candidate / 'config.json').is_file() and any(candidate.glob('*.safetensors')):
            return str(candidate)
    snap = home.hub() / f'models--{repo.replace("/", "--")}' / 'snapshots'
    if snap.is_dir():
        snaps = sorted(
            (p.parent for p in snap.glob('*/config.json') if any(p.parent.glob('*.safetensors'))),
            key=lambda p: p.stat().st_mtime,
        )
        if snaps:
            return str(snaps[-1])
    return repo


def resolve_external(path, kind):
    """Return external drafter source (`repo` or path) for `QUANTEEZE_DRAFTER` / `QUANTEEZE_MTP=dflash2`, else None."""
    del path
    explicit = (variables.get('QUANTEEZE_DRAFTER') or '').strip()
    if explicit and explicit.lower() not in ('off', 'none'):
        target = DFLASH2_REPOS.get(kind) if explicit.lower() == 'dflash2' else explicit
        return None if not target else _local_external(target)
    mode = (variables.get('QUANTEEZE_MTP') or 'auto').strip().lower()
    if mode == 'dflash2':
        target = DFLASH2_REPOS.get(kind)
        return None if not target else _local_external(target)
    return None


def _quantize_dense_drafter(drafter, bits):
    """Quantize dense Linear/Embedding projections of a standalone drafter to `max(4, bits)` when unquantized."""
    import mlx.nn as nn
    if bits is None or switches(drafter):
        return
    if any(getattr(m, 'bits', None) is not None for _, m in drafter.named_modules()):
        return
    eff_bits = max(4, int(bits))
    nn.quantize(
        drafter,
        group_size=64,
        bits=eff_bits,
        class_predicate=lambda p, m: (
            hasattr(m, 'to_quantized')
            and getattr(m, 'weight', None) is not None
            and m.weight.ndim == 2
            and m.weight.shape[-1] % 64 == 0
        ),
    )


def load_external(source, bits=2, target_model=None):
    """Load an external DFlash/DFlash2/MTP drafter checkpoint `source` and pair it for speculative serving."""
    from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
    drafter, kind = load_drafter(str(source))
    if hasattr(drafter, 'candidate_selector'):
        # DFlash2DraftModel inherits DFlashDraftModel.draft_block_greedy, which bypasses CandidateSelector.select.
        object.__setattr__(drafter, 'draft_block_greedy', drafter.draft_block)
    if target_model is not None:
        validate_drafter_compatibility(target_model, drafter, kind)
    if switches(drafter) and bits is not None:
        lower(drafter, bits)
    else:
        _quantize_dense_drafter(drafter, bits)
    mx.eval(drafter.parameters())
    filter_caches(drafter)
    return drafter, kind


def load(path, bits=2):
    """The served checkpoint's MTP drafter (mlx-vlm's load_drafter), its routed experts at `bits`
    (None: as the checkpoint stores them).  Returns (drafter, kind)."""
    from mlx_vlm.speculative.drafters import load_drafter
    from mlx_vlm.speculative.drafters.mtp_split import get_mtp_splitter
    from . import dspark
    if dspark.available(path):                         # DeepSeek V4.1's DSpark, mlx-vlm's DFlash rounds
        return dspark.load(path, bits)
    config = json.loads((Path(path) / 'config.json').read_text())
    splitter = get_mtp_splitter(source_config=config)
    if splitter is not None:
        source_quantization(type(splitter))
    drafter, kind = load_drafter(str(path), kind='mtp')
    if bits is not None:
        lower(drafter, bits)
    mx.eval(drafter.parameters())
    filter_caches(drafter)
    return drafter, kind


# What a drafter's cache may keep a row of: a KVCache's keys and values, and a QSAKVCache's
# indexer keys, block keys and positions ([rows, T], or MRoPE's [3, rows, T]).
ROWS = ('keys', 'values', 'index_keys', 'index_block_keys', 'index_position_ids')


def _source_names(module):
    """Where a drafter module's quantization may stand in the checkpoint's config."""
    names = [module, f'mtp.{module}', f'language_model.mtp.{module}', f'model.mtp.{module}']
    if module.startswith('stages.'):
        names.append('mtp.' + module[len('stages.'):])          # DeepSeek V4's DSpark: mtp.<stage>.* in the checkpoint
    return names


def source_quantization(splitter_cls):
    """A splitter's drafter quantized as the checkpoint stores its head.  The splitter's choice
    stands where the checkpoint's config says nothing of a module; where it names one, that is
    taken; a module stored with biases (affine) that it does not name takes the config's own."""
    own = splitter_cls.quantization_from_source
    if getattr(own, '_quanteeze', False):
        return

    def quantization_from_source(self, tensors, source_config):
        chosen = own(self, tensors, source_config)
        config = source_config or {}
        source = config.get('quantization') or config.get('quantization_config') or {}
        base = {k: source[k] for k in ('group_size', 'bits', 'mode') if k in source}
        found = {}
        for key in tensors:
            if not key.endswith('.scales'):
                continue
            module = key[:-len('.scales')]
            own_q = next((source[n] for n in _source_names(module) if isinstance(source.get(n), dict)), None)
            if own_q is None and f'{module}.biases' in tensors and base.get('bits'):
                own_q = {'mode': 'affine', **base}
            if own_q is not None:
                found[module] = {k: own_q[k] for k in ('group_size', 'bits', 'mode') if k in own_q}
        if not found:
            return chosen
        out = dict(chosen) if chosen else {'mode': 'affine', **base}
        out.update(found)
        return out

    quantization_from_source._quanteeze = True
    splitter_cls.quantization_from_source = quantization_from_source


def filter_caches(drafter):
    """A drafter whose `filter_batch` is `AutoregressiveMTPDraftModel`'s gets one that cuts all
    per-row arrays in its caches when a request in a batch finishes."""
    from mlx_vlm.speculative.drafters.mtp_base import AutoregressiveMTPDraftModel
    if getattr(type(drafter), 'filter_batch', None) is not AutoregressiveMTPDraftModel.filter_batch:
        return False                                   # its own, or none (DeepSeek V4's DSpark drafter)

    def filter_batch(keep):
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)
        for cache in drafter._cache:
            if cache.keys is None:
                continue
            rows = cache.keys.shape[0]
            for name in ROWS:
                value = getattr(cache, name, None)
                axis = 1 if name == 'index_position_ids' and getattr(value, 'ndim', 0) == 3 else 0
                if isinstance(value, mx.array) and value.ndim > axis and value.shape[axis] == rows:
                    setattr(cache, name, value[keep] if axis == 0 else value[:, keep])
        if drafter._seed_token is not None:
            drafter._seed_token = drafter._seed_token[keep]
        if drafter._seed_hidden is not None:
            drafter._seed_hidden = drafter._seed_hidden[keep]
        value = drafter._next_position
        if isinstance(value, mx.array) and value.ndim > 0 and value.size > 1:
            drafter._next_position = value[keep]

    object.__setattr__(drafter, 'filter_batch', filter_batch)
    return True


def set_block(drafter, block):
    """Set target verify block length for `drafter`, returning the effective block length used at runtime.

    For `DFlash`/`DFlash2`/`DSpark` drafters, clamps `runtime_block_size` to `[2, config.block_size]`.
    For autoregressive MTP drafters whose `block` exceeds `config.block_size` (e.g., `block=4` on 1-layer
    `Qwen4ExpMTPDraftModel` whose `config.block_size == 2`), keeps `runtime_block_size = None` and passes
    `block` through `OVERRIDE_BLOCK` (`_get_draft_block_size_from_env`) so `mlx_vlm` uses it as an upper
    bound instead of clamping to `min(config.block_size, runtime_block_size)`.
    """
    want = int(block)
    cfg = getattr(drafter, 'config', None)
    configured = getattr(cfg, 'block_size', None)
    mtype = str(getattr(cfg, 'model_type', '')).lower()
    is_block_drafter = (
        'dflash' in mtype
        or 'dspark' in mtype
        or hasattr(drafter, 'candidate_selector')
        or hasattr(cfg, 'target_layer_ids')
    )
    if is_block_drafter:
        eff = max(2, want) if configured is None else max(2, min(want, int(configured)))
        cfg.runtime_block_size = eff
        OVERRIDE_BLOCK[0] = None
        return eff
    if configured is not None and want > int(configured):
        OVERRIDE_BLOCK[0] = want
        cfg.runtime_block_size = None
        return want
    OVERRIDE_BLOCK[0] = None
    cfg.runtime_block_size = want
    return want


def block_for(setting, model_type):
    """QUANTEEZE_MTP_BLOCK's block for the target's model type: a number as given, 'auto' from
    BLOCKS, None ('off', or 'auto' for a type not in BLOCKS) for mlx-vlm's choice."""
    if setting is None or setting.lower() == 'off':
        return None
    if setting.lower() == 'auto':
        return BLOCKS.get(model_type)
    return int(setting)


def switches(drafter):
    """The drafter's routed experts' switch modules (SwitchGLU's projections, quantized)."""
    out = []
    for name, module in drafter.named_modules():
        if name.endswith('switch_mlp'):
            out.append(module)
    return out


def lower(drafter, bits, group_size=requant.GROUP_SIZE):
    """Each routed expert of the drafter at `bits`, by least squares (quanteeze.requant), in groups
    of `group_size` (a projection whose width it does not divide keeps its own), CHUNK experts at
    a time."""
    if not requant.native(bits):                       # mlx-vlm's SwitchGLU computes them by gather_qmm
        raise ValueError(f'the drafter\'s experts at {bits} bits: MLX computes none here (its CUDA backend)')
    for switch in switches(drafter):
        for p in ('gate_proj', 'up_proj', 'down_proj'):
            e = getattr(switch, p)
            gs = group_size if e.input_dims % group_size == 0 else e.group_size     # its own where it does not divide
            if e.bits == bits and e.group_size == gs and e.mode == 'affine':
                continue
            parts = []
            for a in range(0, e.weight.shape[0], CHUNK):
                w = requant.dequantize(e.weight[a:a + CHUNK], e.scales[a:a + CHUNK], e.get('biases')[a:a + CHUNK]
                                       if e.get('biases') is not None else None, group_size=e.group_size,
                                       bits=e.bits, mode=e.mode)
                part = requant.quantize(w, bits, gs)
                mx.eval(part)
                parts.append(part)
            e.weight, e.scales, e.biases = (mx.concatenate([q[k] for q in parts]) for k in range(3))
            e.group_size, e.bits, e.mode = gs, bits, 'affine'
            mx.eval(e.weight, e.scales, e.biases)


def nbytes(drafter):
    return sum(v.nbytes for _, v in _leaves(drafter.parameters()))


def _leaves(tree, prefix=''):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _leaves(v, f'{prefix}{k}.')
    elif isinstance(tree, list):
        for k, v in enumerate(tree):
            yield from _leaves(v, f'{prefix}{k}.')
    elif isinstance(tree, mx.array):
        yield prefix[:-1], tree


def install(generator_cls):
    """The server's generator (mlx-vlm's ResponseGenerator) drafts with the drafter once enabled,
    and a request's thinking budget is taken off while it does.

    mlx-vlm extends no batch that drafts: a request coming while one decodes waits for it to end,
    and a batch drafts alone (install_batches).  So while drafting and requests come together (two
    within GATHER_MS of each other in the last BURST_S), the requests arriving within GATHER_MS of
    the first start together, undrafted, rather than one drafting while the others wait; a request
    asked alone, as in a chat, starts at once (mlx-vlm waits the gathering time for every first
    request: 2 s a turn).  Setting MLX_VLM_SPEC_BATCH_COALESCE_MS gathers the first batch too.
    A request waiting for a drafting batch to end is given up after WAIT_S with no token, not
    mlx-vlm's 600 s, nor never, which a generation stuck for any other cause would wait."""
    import os
    import time

    from mlx_vlm import server as _server
    from mlx_vlm.server import generation

    from . import stream
    timeout, coalesce = generation.get_token_queue_timeout, generation.get_speculative_batch_coalesce_s
    block_fn = getattr(generation, '_get_draft_block_size_from_env', None)

    def token_queue_timeout():
        return WAIT_S if ACTIVE[0] is not None else timeout()

    def speculative_batch_coalesce_s():
        if os.environ.get('MLX_VLM_SPEC_BATCH_COALESCE_MS'):
            return coalesce()
        burst = _COMING['burst']
        return GATHER_MS / 1000 if burst is not None and time.monotonic() - burst <= BURST_S else 0.0

    def draft_block_size():
        env = block_fn() if callable(block_fn) else None
        if env is not None:
            return env
        return OVERRIDE_BLOCK[0] if ACTIVE[0] is not None else None

    if not getattr(timeout, 'quanteeze', False):                  # once, however often installed
        token_queue_timeout.quanteeze = True
        generation.get_token_queue_timeout = token_queue_timeout
        generation.get_speculative_batch_coalesce_s = speculative_batch_coalesce_s
    if callable(block_fn) and not getattr(block_fn, 'quanteeze', False):
        draft_block_size.quanteeze = True
        generation._get_draft_block_size_from_env = draft_block_size
        if hasattr(_server, '_get_draft_block_size_from_env'):
            _server._get_draft_block_size_from_env = draft_block_size
    for name, at in (('draft_model', 0), ('draft_kind', 1)):
        own = '_quanteeze_' + name

        def get(self, own=own, at=at):
            return ACTIVE[0][at] if ACTIVE[0] is not None else self.__dict__.get(own)

        def set_(self, value, own=own):
            self.__dict__[own] = value
        setattr(generator_cls, name, property(get, set_))

    said = []

    def before(request):
        now = time.monotonic()
        if _COMING['last'] is not None and now - _COMING['last'] <= GATHER_MS / 1000:
            _COMING['burst'] = now
        _COMING['last'] = now
        args = request.args
        if ACTIVE[0] is not None and args is not None and getattr(args, 'thinking_budget', None) is not None:
            args.thinking_budget = None
            if not said:
                said.append(True)
                print('quanteeze: MTP drafting: a request\'s thinking budget is taken off (mlx-vlm refuses it '
                      'with a drafter)', file=sys.stderr, flush=True)

    stream.add(generator_cls, 'mtp', before=before)


def install_batches(batch_cls):
    """`mlx-vlm`'s `BatchGenerator` drafts only for a batch that starts with one request; when
    multiple requests are batched together, they decode without the drafter."""
    step = batch_cls._next

    def _next(self, **kwargs):
        own = self.__dict__.setdefault('_quanteeze_drafter', (self.draft_model, self.draft_kind))
        if own[0] is not None and self._prompt_batch is None and self._unprocessed_sequences \
                and not len(self._generation_batch):
            alone = min(self.prefill_batch_size, len(self._unprocessed_sequences)) == 1
            self.draft_model, self.draft_kind = own if alone else (None, None)
        return step(self, **kwargs)
    batch_cls._next = _next


def enable(drafter, kind='mtp'):
    """Draft with `drafter` from the next request on."""
    ACTIVE[0] = (drafter, kind)


def disable():
    ACTIVE[0] = None
    OVERRIDE_BLOCK[0] = None


def label_of(drafter):
    """Short UI/log label for `drafter` ('DFlash2', 'DSpark', or 'MTP')."""
    from . import dspark
    if isinstance(drafter, dspark.Drafter):
        return 'DSpark'
    name = type(drafter).__name__
    if 'DFlash2' in name:
        return 'DFlash2'
    if 'DFlash' in name:
        return 'DFlash'
    return 'MTP'


def summary(drafter):
    """Rounds, tokens drafted and accepted, and the acceptance rate, as mlx-vlm counts them."""
    rounds = int(getattr(drafter, 'speculative_total_rounds', 0))
    drafted = int(getattr(drafter, 'speculative_total_drafted', 0))
    accepted = float(getattr(drafter, 'speculative_total_accepted', 0.0))
    return dict(label=label_of(drafter), rounds=rounds, drafted=drafted, accepted=int(round(accepted)),
                acceptance=round(accepted / drafted, 4) if drafted else None,
                tokens_a_round=round(1 + accepted / rounds, 3) if rounds else None)


TIMES = {}                            # stage: [calls, seconds], with QUANTEEZE_MTP_TIMING
PROFILE = [None]                      # QUANTEEZE_MTP_TIMING=profile: the serving thread sampled between tokens
STAGES = (('_mtp_draft_block_active', 'draft'), ('_mtp_verify_target', 'verify'),
          ('_speculative_walk_batch', 'walk'), ('_speculative_walk_batch_deferred_greedy', 'walk'),
          ('_speculative_walk_batch_deferred_uniform', 'walk'), ('_mtp_draft_hidden', 'hidden'))


def _arrays(value, out):
    if isinstance(value, mx.array):
        out.append(value)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _arrays(v, out)
    elif isinstance(value, dict):
        for v in value.values():
            _arrays(v, out)
    elif hasattr(value, '__dataclass_fields__'):
        for name in value.__dataclass_fields__:
            _arrays(getattr(value, name), out)
    return out


def time_rounds(drafter, mode):
    """Each stage of mlx-vlm's MTP rounds (_mtp_rounds_batch, the server's) timed into TIMES: the
    draft, the verify pass, the walk (its .tolist() waits on the GPU), the drafter's replay of
    what was kept (accept), the target's commit, the drafter's rebinding (rebind), the hidden
    state for the next draft; and the round's time (inside: from resuming the rounds to their
    next token) and the server's between tokens (outside; outside_cpu: its thread's CPU).  mode 'wall' times the Python that
    builds each stage, the GPU's work landing where something waits on it (the walk); 'sync'
    evaluates each stage's outputs after it, so each holds its own GPU work (and building no
    longer overlaps computing: the rounds run slower); 'profile' as 'wall', and the serving
    thread's stack sampled between tokens (_Sampler: what it does with a token; times())."""
    from . import dspark
    if isinstance(drafter, dspark.Drafter):
        print('quanteeze: QUANTEEZE_MTP_TIMING times MTP rounds, not DSpark\'s (DFlash rounds)', file=sys.stderr,
              flush=True)
        return
    import time
    from mlx_vlm.speculative import mtp as spec, utils

    sync = mode == 'sync'

    def add(stage, dt):
        entry = TIMES.setdefault(stage, [0, 0.0])
        entry[0] += 1
        entry[1] += dt

    def timed(fn, stage, outputs=None):
        def call(*args, **kwargs):
            t = time.perf_counter()
            out = fn(*args, **kwargs)
            if sync:
                arrays = _arrays(out if outputs is None else outputs(), [])
                if arrays:
                    mx.eval(arrays)
                mx.synchronize()
            add(stage, time.perf_counter() - t)
            return out
        return call

    for name, stage in STAGES:
        if hasattr(spec, name):
            setattr(spec, name, timed(getattr(spec, name), stage))
    result = spec._MTPVerifyResult
    result.commit = timed(result.commit, 'commit')
    drafter.accept_verified_tokens_batch = timed(
        drafter.accept_verified_tokens_batch, 'accept',
        lambda: (drafter._seed_token, drafter._seed_hidden))
    drafter.set_shared_kv = timed(drafter.set_shared_kv, 'rebind')
    rounds = utils._mtp_rounds_batch
    if mode == 'profile':
        PROFILE[0] = _Sampler()

    def timed_rounds(*args, **kwargs):
        gen = rounds(*args, **kwargs)
        try:
            while True:
                t = time.perf_counter()
                try:
                    item = next(gen)
                except StopIteration:
                    return
                add('inside', time.perf_counter() - t)
                t, cpu = time.perf_counter(), time.thread_time()
                if PROFILE[0] is not None:
                    PROFILE[0].watch()
                try:
                    yield item
                finally:
                    if PROFILE[0] is not None:
                        PROFILE[0].watch(False)
                add('outside', time.perf_counter() - t)
                add('outside_cpu', time.thread_time() - cpu)
        finally:
            gen.close()
    utils._mtp_rounds_batch = timed_rounds


def times(drafter):
    """TIMES as ms a round (the rounds mlx-vlm counted since the start), and the round's rest:
    inside less its timed stages."""
    rounds = int(getattr(drafter, 'speculative_total_rounds', 0))
    if not rounds or not TIMES:
        return {}
    ms = {k: round(v[1] * 1e3 / rounds, 2) for k, v in TIMES.items()}
    staged = sum(v for k, v in ms.items() if k not in ('inside', 'outside', 'outside_cpu'))
    if 'inside' in ms:
        ms['rest'] = round(ms['inside'] - staged, 2)
    out = dict(rounds=rounds, ms_a_round=ms)
    if PROFILE[0] is not None:
        out['outside_samples'] = PROFILE[0].top()
    return out


class _Sampler:
    """The serving thread's stack between tokens, sampled every 0.2 ms by a thread of its own
    (cProfile, on Python 3.12 on, sees every thread at once): the share of samples each
    function was on the stack (where), and the innermost frames (own)."""

    def __init__(self, every=2e-4):
        import threading
        self.on, self.ident = False, None
        self.where, self.own, self.samples = {}, {}, 0
        self.every = every
        threading.Thread(target=self._run, daemon=True, name='mtp-sampler').start()

    def watch(self, on=True):
        import threading
        self.ident = threading.get_ident()
        self.on = on

    def _run(self):
        import time
        while True:
            time.sleep(self.every)
            if not self.on:
                continue
            frame = sys._current_frames().get(self.ident)
            if frame is None:
                continue
            self.samples += 1
            seen = set()
            first = True
            while frame is not None:
                code = frame.f_code
                name = f'{code.co_filename.rsplit("/", 1)[-1]}:{code.co_firstlineno} {code.co_name}'
                if first:
                    self.own[name] = self.own.get(name, 0) + 1
                    first = False
                if name not in seen:
                    seen.add(name)
                    self.where[name] = self.where.get(name, 0) + 1
                frame = frame.f_back

    def top(self, n=25):
        if not self.samples:
            return {}
        share = lambda d: [(k, round(v / self.samples, 3)) for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n]]
        return dict(samples=self.samples, own=share(self.own), where=share(self.where))


def gate_said(drafter):
    """What the load's line adds for DSpark's drafter, gated by its confidence head; '' for others."""
    from . import dspark
    if not isinstance(drafter, dspark.Drafter):
        return ''
    gate = dspark.gate_threshold()
    return (f' at most, the drafts kept while their confidence stays at {gate:g} or more (QUANTEEZE_DSPARK_GATE)'
            if gate else '')


def bits_of(drafter):
    """The routed experts' bits (or dense quantized bits / floating dtype when dense)."""
    bits = sorted({getattr(s, p).bits for s in switches(drafter) for p in ('gate_proj', 'up_proj', 'down_proj')})
    if bits:
        return str(bits[0])
    qbits = sorted({int(m.bits) for _, m in drafter.named_modules() if getattr(m, 'bits', None) is not None})
    if qbits:
        return '/'.join(str(b) for b in qbits)
    for _, arr in _leaves(drafter.parameters()):
        if mx.issubdtype(arr.dtype, mx.floating):
            return {mx.bfloat16: 'bf16', mx.float16: 'fp16', mx.float32: 'fp32'}.get(arr.dtype, str(arr.dtype))
    return '?'

