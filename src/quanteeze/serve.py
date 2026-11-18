# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Quanteeze MLX inference server (`quanteeze.serve`).

Serves MoE and dense models with:
- Uniform least-squares expert quantization (`QUANTEEZE_BITS`: 4, 3, 2, or 1 bit, or `off` for the
  checkpoint's original precision), lowering routed experts in memory when they fit `fitting.room`,
  or streaming them on demand from the SSD when they do not (`QUANTEEZE_FROM_SSD_GB`).
- Custom Metal and CUDA decode/verify kernels (`verify_qmv`, `expert_qmv`, `hyper`, `linear_attn`,
  `glm_hyper`, `glm_linear_attn`, `glm_indexer`, `sigmoid_router`, `cuda_qmv`, `cuda_experts`,
  `cuda_router`), each shape verified bit for bit against stock before serving.
- Multi-Token Prediction (MTP) and DSpark speculative decoding (`quanteeze.mtp`, `quanteeze.dspark`).
- PLE / Engram SSD streaming and multi-threaded read-ahead (`quanteeze.ple_store`,
  `quanteeze.ngram_prefetch`).
- Memory-aware prompt admission (`quanteeze.admission`), layer-at-a-time prefill (`quanteeze.prefill`),
  low-RSS lock-free host pool allocation (`quanteeze.allocator`, `kamepoolalloc`), and macOS thermal
  throttling (`quanteeze.guards`).
- Browser UI at `/quanteeze/` (`quanteeze.home`) and `/quanteeze/settings` (`quanteeze.control`).
"""
import atexit
import gc
import importlib
import os
import signal
import sys
import threading
import time
import weakref
from pathlib import Path

import mlx.core as mx
import mlx_vlm.server as _server
import uvicorn.server
from mlx.utils import tree_flatten
from mlx_vlm.apc import APCManager
from mlx_vlm.generate.ar import BatchGenerator, PromptProcessingBatch
from mlx_vlm.server import generation
from mlx_vlm.server.cli import main as _serve

from . import (
    admission,
    allocator,
    apc_disk,
    apc_images,
    arch,
    control,
    cuda_attn,
    cuda_experts,
    cuda_head,
    cuda_kernels,
    cuda_moe,
    cuda_ple,
    cuda_ple_layer,
    cuda_qmv,
    cuda_router,
    cuda_v41,
    demand,
    dsml,
    efforts,
    embed_ssd,
    expert_qmv,
    fitting,
    gguf_experts,
    gguf_qmv,
    glm_hyper,
    glm_indexer,
    glm_linear_attn,
    guards,
    home,
    hyper,
    kquant_qmv,
    linear_attn,
    loops,
    mtp,
    ngram_prefetch,
    parallel,
    phases,
    prefill,
    progress,
    q2_experts,
    reask,
    requant,
    shutdown,
    sigmoid_router,
    stream,
    toolcalls,
    unindexed,
    variables,
    verify_qmv,
    vision_ssd,
)

_load = generation.load
UNIFORM_BITS = (1, 2, 3, 4, 8)
_SERVING = {}                # active model metadata & Demand instance for status/cleanup
_AT_EXIT = {}
_LOADED = []
_SPLIT = []
_ADMISSION = [None]
_KEEPER = [None]
_IN_MEMORY = [True]


def serving_info():
    """Snapshot of the currently loaded model's precision and serving mode for the Browser UI."""
    info = dict(_SERVING.get('info') or {})
    dem = _SERVING.get('demand')
    if dem is not None:
        info['demand'] = dem.summary()
    return info


def _precision():
    """The served model's precision settings that change what a prompt computes, '' when none."""
    info = _SERVING.get('info') or {}
    return f'{info.get("mode", "")}:{info.get("bits", "")}' if info else ''


WARM_TEXT = ('A model warms up before its first request: the kernels it compiles at their first use are '
             'compiled now, not while someone waits.  モデルは最初のリクエストの前に一度動かしておく。'
             'Each sentence here only makes tokens; what they say does not matter.  ') * 12
WARM_TOKENS = (240, 40, 4)       # a prefill tiled over 512 experts' layers, one that is not, decode steps


def _warm_up(model, processor):
    """On CUDA, the model run once on WARM_TEXT before the server takes requests (QUANTEEZE_WARM_UP):
    quanteeze's own kernels (mx.fast.cuda_kernel) are compiled at their first use in each process, MLX's
    disk cache keeping only its own, and the first request had waited for them; not where experts are read
    on demand (the prompt would fill the banks)."""
    if variables.off('QUANTEEZE_WARM_UP') or not cuda_kernels.available() or parallel.size() > 1:
        return
    known = arch.resolve(model)
    switches = [arch.switch_of(layer) for layer in arch.decoder_layers(model)] if known else []
    if any(type(s).__name__ == 'DemandSwitchGLU' for s in switches):
        return
    lm = getattr(model, 'language_model', model)
    tokenizer = getattr(processor, 'tokenizer', processor)
    if not hasattr(lm, 'make_cache') or not hasattr(tokenizer, 'encode'):
        return
    start = time.time()
    try:
        ids = tokenizer.encode(WARM_TEXT, add_special_tokens=False)
        prefill_n, more, steps = WARM_TOKENS
        cache = lm.make_cache()

        def last(out):
            return mx.argmax((out.logits if hasattr(out, 'logits') else out)[:, -1], axis=-1)
        y = None
        for part in (ids[:prefill_n], ids[prefill_n:prefill_n + more]):
            y = last(lm(mx.array([part]), cache=cache))
            mx.eval(y)
        for _ in range(steps):
            y = last(lm(y[:, None], cache=cache))
            mx.eval(y)
        del cache
        print(f'quanteeze: warmed up in {time.time() - start:.0f} s, its kernels compiled before the first request '
              '(QUANTEEZE_WARM_UP)', file=sys.stderr, flush=True)
    except Exception as e:                                # noqa: BLE001 -- a warm-up that fails leaves the first request to it
        print(f'quanteeze: the warm-up did not run ({type(e).__name__}: {e}); the first request compiles the '
              'kernels', file=sys.stderr, flush=True)
    finally:
        mx.clear_cache()


def _switch_nbytes(switch):
    seen, total = set(), 0
    for _, a in tree_flatten(switch.parameters()):
        if id(a) not in seen:
            seen.add(id(a))
            total += a.nbytes
    return total


def _uniform_bits():
    """`(bits, group_size)` when `QUANTEEZE_BITS` requests uniform expert quantization, else `None`."""
    value = (variables.get('QUANTEEZE_BITS') or '').strip().lower()
    if value in ('', 'off'):
        return None
    if value == 'mxfp4':
        return value, 32
    if value in requant.GGML_MODES:
        return value, 256
    if not value.isdigit() or int(value) not in UNIFORM_BITS:
        choices = [*map(str, UNIFORM_BITS), *requant.UNIFORM_MODES]
        raise SystemExit(f'quanteeze: QUANTEEZE_BITS={value}: one of {", ".join(choices)}, or off')
    b = int(value)
    return b, fitting.group_size(b)


def _uniform_label(uniform):
    if uniform[0] == 'mxfp4':
        return 'uniform MXFP4 (group size 32)'
    if uniform[0] in requant.GGML_MODES:
        return f'uniform {uniform[0].upper()} (block size 256)'
    return f'uniform {uniform[0]} bits (group size {uniform[1]}, least squares)'


def _demand_variant():
    """Budget in GB (or `'auto'`) when `QUANTEEZE_FROM_SSD_GB` forces SSD on-demand streaming, else `None`."""
    gb = variables.get('QUANTEEZE_FROM_SSD_GB')
    if not gb:
        return None
    return 'auto' if gb.strip().lower() == 'auto' else float(gb)


def _experts_gguf(uniform=None):
    """`gguf_experts.ExpertGGUF` when `QUANTEEZE_EXPERTS_GGUF` (`--experts-gguf`) is set, else `None`."""
    value = (variables.get('QUANTEEZE_EXPERTS_GGUF') or '').strip()
    if not value or value.lower() == 'off':
        return None
    if uniform is not None:
        raise SystemExit(
            'quanteeze: set either QUANTEEZE_BITS or QUANTEEZE_EXPERTS_GGUF (--experts-gguf), not both'
        )
    try:
        return gguf_experts.ExpertGGUF(value)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f'quanteeze: QUANTEEZE_EXPERTS_GGUF={value}: {e}')


def _load_whole():
    value = (variables.get('QUANTEEZE_LOAD_WHOLE') or '').lower()
    return value in ('on', '1', 'true', 'yes')


def _load_hooked(*args, **kwargs):
    """`mlx_vlm`'s load with Quanteeze's uniform quantization, kernel hooks, and progress reporting."""
    guards.note_others()
    path = _checkpoint_path(args, kwargs)
    if path and os.path.isdir(path):
        try:
            chosen, why = home.ngram_source(path)
        except ValueError as e:
            chosen, why = path, f'its n-gram table held in memory: no view of it could be made ({e})'
        if why is not None:
            print(f'quanteeze: {why}' + (f' ({chosen})' if str(chosen) != path else ''), file=sys.stderr, flush=True)
            args, kwargs = (
                ((str(chosen), *args[1:]), kwargs) if args else (args, dict(kwargs, path_or_hf_repo=str(chosen)))
            )
    progress.begin(_checkpoint_path(args, kwargs))
    if _ADMISSION[0] is not None:
        _ADMISSION[0].fixed, _ADMISSION[0].per_token = 0, None
    try:
        out = _load_serving(*args, **kwargs)
        _warm_up(*out)
    except BaseException as e:
        progress.end(f'{type(e).__name__}: {e}')
        raise
    progress.end()
    return out


def _load_serving(*args, **kwargs):
    if parallel.size() > 1 and _SPLIT:
        raise RuntimeError(
            'quanteeze: served over several GPUs, the model is the one the server started with; '
            'restart it to serve another'
        )
    control.loaded()
    uniform = _uniform_bits()
    experts_gguf = _experts_gguf(uniform)
    on_demand = _demand_variant()
    _let_go()
    mtp.disable()
    progress.phase('opening')
    model, processor = _load(*args, **{**kwargs, 'lazy': True})
    _LOADED.append(weakref.ref(model))
    if parallel.split(model) is not None:
        _SPLIT.append(True)
    kind = getattr(model.config, 'model_type', None)
    fitting.MODEL[0] = kind
    known = arch.resolve(model)
    layers = arch.decoder_layers(model) if known else []
    path = _checkpoint_path(args, kwargs)
    if known is not None and known.prepare(model, path):
        print(f'quanteeze: {kind}\'s routed experts and projections quantized as the checkpoint stores them',
              file=sys.stderr, flush=True)
    if path and os.path.isdir(path):
        if vision_ssd.install(model, path) + embed_ssd.install(model, path):
            gc.collect()
            mx.clear_cache()
    switches = [arch.switch_of(layer) for layer in layers if arch.switch_of(layer) is not None]
    has_experts = bool(switches)
    _hold_tables(model, path)

    if known:
        blocks = known.install(model)
        print(f'quanteeze: hooks on {len(blocks)} MoE blocks ({kind})', file=sys.stderr, flush=True)
        if kind == 'qwen4_exp' and hyper.ON[0] and hyper.install():
            print('quanteeze: Flash-Next\'s hyper-connections a decode token in fewer kernels (quanteeze.hyper), '
                  'each shape checked bit for bit first', file=sys.stderr, flush=True)
        if kind == 'glm5_next' and sigmoid_router.ON[0]:
            print('quanteeze: GLM-5.3\'s routing a decode token in two kernels (quanteeze.sigmoid_router), '
                  'each shape checked bit for bit first', file=sys.stderr, flush=True)
        if kind == 'glm5_next' and linear_attn.ON[0]:
            print('quanteeze: GLM-5.3\'s linear attention a decode step in three kernels '
                  '(quanteeze.glm_linear_attn), each shape checked bit for bit first', file=sys.stderr, flush=True)
        if kind == 'glm5_next' and glm_hyper.ON[0]:
            print('quanteeze: GLM-5.3\'s hyper-connections a decode token in two kernels (quanteeze.glm_hyper), '
                  'each shape checked bit for bit first', file=sys.stderr, flush=True)
        if kind == 'glm5_next' and glm_indexer.ON[0]:
            print('quanteeze: GLM-5.3\'s indexer choosing a decode token\'s positions in one kernel '
                  '(quanteeze.glm_indexer), each path checked bit for bit first', file=sys.stderr, flush=True)
    else:
        print(f'quanteeze: serving dense / standard model {kind!r}', file=sys.stderr, flush=True)

    readings = efforts.use_model(kind)
    if readings:
        print('quanteeze: reasoning effort ' + ', '.join(f'{a} read as {b}' for a, b in readings.items()),
              file=sys.stderr, flush=True)
    apc_images.use_model(model)
    chat_format = dsml.use_model(model, processor)
    if chat_format:
        print(f'quanteeze: {chat_format}', file=sys.stderr, flush=True)
    throttle = guards.Throttle.from_env()
    if throttle is not None:
        guards.install_throttle(arch.language_model(model), throttle)
        _SERVING['throttle'] = throttle
        shares = (', '.join(f'{v:.0%} at {k}' for k, v in sorted(throttle.duty.items()) if k >= throttle.at)
                  if isinstance(throttle.duty, dict) else f'{throttle.duty:.0%}')
        print(f'quanteeze: thermal throttle from pressure {throttle.at:g}: the GPU works {shares} of the time',
              file=sys.stderr, flush=True)

    drafter = _drafter(path, kind)

    if not has_experts:
        if on_demand is not None or uniform or experts_gguf is not None:
            why = ('QUANTEEZE_BITS' if uniform else
                   'QUANTEEZE_EXPERTS_GGUF' if experts_gguf is not None else 'QUANTEEZE_FROM_SSD_GB')
            raise SystemExit(
                f'quanteeze: {why} is for MoE models; '
                f'no routed experts found in {kind!r}'
            )
        _evaluate(model)
        _enable(drafter, 'from the start', in_memory=True)
        _SERVING['info'] = {
            'model': path, 'model_type': kind, 'mode': 'in_memory',
            'bits': None, 'stock_bits': None, 'active_gb': round(mx.get_active_memory() / 1e9, 2),
        }
        return model, processor

    stock_bits = fitting.switch_bits(switches[0])
    target_bits = uniform[0] if uniform else None
    gguf_label = None
    if experts_gguf is not None:
        try:
            experts_gguf.validate(layers)
        except (KeyError, ValueError) as e:
            raise SystemExit(f'quanteeze: {e}')
        moe_indices = [i for i, layer in enumerate(layers) if arch.switch_of(layer) is not None]
        gguf_label = experts_gguf.format_label(moe_indices)
    effective_bits = gguf_label or target_bits or stock_bits
    resident = _resident(model, switches)
    space = fitting.room(resident)
    if experts_gguf is not None:
        experts_bytes = experts_gguf.total_bytes(moe_indices)
    elif target_bits is not None:
        experts_bytes = sum(fitting.quantized_bytes(fitting.weights(s), target_bits) for s in switches)
    else:
        experts_bytes = sum(_switch_nbytes(s) for s in switches)

    fits_in_memory = _load_whole() or space is None or experts_bytes <= space
    if not fits_in_memory or on_demand is not None:
        _apc_off_the_gpu()
        _prefill_short()
    if on_demand is None and not fits_in_memory:
        if parallel.size() > 1:
            _parallel_failed(
                'this rank\'s layers do not fit its GPU whole: more GPUs (QUANTEEZE_GPUS), '
                'lower QUANTEEZE_BITS, or a smaller QUANTEEZE_MEMORY_RESERVE'
            )
        on_demand = 'auto'
        how_bits = gguf_label or (effective_bits.upper() if isinstance(effective_bits, str) else f'{effective_bits} bits')
        print(
            f'quanteeze: the routed experts at {how_bits} take {experts_bytes / 1e9:.1f} GB, '
            f'and {space / 1e9:.1f} GB is left beside the rest ({resident / 1e9:.1f} GB) and the reserve '
            f'(QUANTEEZE_MEMORY_RESERVE, {fitting.reserve():g}): serving on demand from SSD',
            file=sys.stderr, flush=True,
        )

    if on_demand is not None:
        if parallel.size() > 1:
            _parallel_failed('over several GPUs the model must fit in memory whole')
        if stock_bits == fitting.FLOAT_BITS and target_bits is None and experts_gguf is None:
            raise SystemExit(
                'quanteeze: on-demand SSD streaming requires quantized experts, QUANTEEZE_EXPERTS_GGUF, or '
                'QUANTEEZE_BITS=4/3/2/1/mxfp4/q3_k/q2_k/iq2_xs'
            )
        gb = on_demand
        if gb == 'auto':
            gb = _auto_gb(resident, space)
        n_experts = switches[0].gate_proj.weight.shape[0]
        expert_bytes = experts_bytes / (len(switches) * n_experts)
        if experts_gguf is not None:
            source = experts_gguf
        elif uniform:
            t0 = time.time()
            progress.phase('experts', len(switches), 'MoE layers')
            source = requant.cached_checkpoint(
                layers, path, *uniform,
                on_layer=lambda done, _total: progress.advance(done),
            )
            if source is not None:
                mx.clear_cache()
                print(
                    f'quanteeze: lowered experts for {len(switches)} MoE layers of {kind} at '
                    f'{_uniform_label(uniform)} cached in {source.path} ({time.time() - t0:.1f} s)',
                    file=sys.stderr, flush=True,
                )
            else:
                source = requant.Flat(path, *uniform)
        else:
            source = q2_experts.Checkpoint(path)
        prompt_room = (None if variables.off('QUANTEEZE_DEMAND_PROMPT_GB')
                       else variables.number('QUANTEEZE_DEMAND_PROMPT_GB') * 1e9)
        served = demand.Demand(
            model, source, gb * 1e9,
            workers=q2_experts.read_threads(str(source.path) if hasattr(source, 'path') else path),
            prompt_room=prompt_room,
        )
        if _ADMISSION[0] is not None:
            cap = served._prefill_cap()
            _ADMISSION[0].fixed = min(cap, n_experts) * expert_bytes
            _ADMISSION[0].batch_fixed = min(cap, max(0, n_experts - served.slots)) * expert_bytes
        _SERVING.update(demand=served, info={
            'model': path, 'model_type': kind, 'mode': 'on_demand',
            'bits': effective_bits, 'stock_bits': stock_bits,
            'experts_gb': round(served.slots * expert_bytes * len(switches) / 1e9, 2),
            'total_experts_gb': round(experts_bytes / 1e9, 2),
            'resident_gb': round(resident / 1e9, 2),
            'room_gb': round(space / 1e9, 2) if space is not None else None,
        })
        _enable(drafter, 'on demand, from the start', in_memory=False)
        layered = None if variables.off('QUANTEEZE_PREFILL_LAYERS') else variables.integer('QUANTEEZE_PREFILL_LAYERS')
        if layered and not prefill.supported(arch.language_model(model)):
            print(f'quanteeze: {kind}\'s caches are shared or windowed: prompts go through every layer a chunk at a time',
                  file=sys.stderr, flush=True)
            layered = None
        if layered:
            prefill.install(layers, enabled=lambda: True)
            prefill.widen(PromptProcessingBatch, layered)
            print(f'quanteeze: on demand, a prompt goes through the model a layer at a time, up to '
                  f'{layered} tokens at once', file=sys.stderr, flush=True)
        if isinstance(source, (q2_experts.Checkpoint, gguf_experts.ExpertGGUF)):
            _hold_experts(source, 2 * expert_bytes * n_experts)
        mx.clear_cache()
        print(
            f'quanteeze: serving {kind} '
            + (f'from {gguf_label} ({experts_gguf.path.name})'
               if experts_gguf is not None
               else f'at {_uniform_label(uniform)}' if uniform
               else f'at checkpoint precision ({stock_bits} bits)')
            + f' on demand from SSD: {served.slots} of {n_experts} experts a layer kept in memory '
            f'({served.slots * expert_bytes * len(switches) / 1e9:.1f} GB); '
            f'MLX holds {mx.get_active_memory() / 1e9:.1f} GB',
            file=sys.stderr, flush=True,
        )
        _at_exit('demand', lambda: print(f'quanteeze: on demand, {served.summary()}', file=sys.stderr, flush=True))
        return model, processor

    # Whole in memory (pre-distributed GGUF, lowered uniformly to `target_bits`, or at checkpoint precision)
    if experts_gguf is not None:
        t0 = time.time()
        progress.phase('experts', len(switches), 'MoE layers')
        gguf_experts.load_model(
            layers, experts_gguf,
            on_layer=lambda done, _total: progress.advance(done),
        )
        mx.clear_cache()
        print(
            f'quanteeze: loaded all {len(switches)} MoE layers of {kind} from {gguf_label} '
            f'({experts_gguf.path.name}) in {time.time() - t0:.1f} s ({experts_bytes / 1e9:.1f} GB of experts); '
            f'MLX holds {mx.get_active_memory() / 1e9:.1f} GB',
            file=sys.stderr, flush=True,
        )
    elif uniform is not None:
        t0 = time.time()
        progress.phase('experts', len(switches), 'MoE layers')
        requant.lower_model(
            layers, path, uniform[0], uniform[1],
            on_layer=lambda done, _total: progress.advance(done),
        )
        for layer in layers:
            sw = arch.switch_of(layer)
            if sw is not None:
                expert_qmv.install_switch(sw)
        mx.clear_cache()
        print(
            f'quanteeze: lowered all {len(switches)} MoE layers of {kind} to {_uniform_label(uniform)} '
            f'in {time.time() - t0:.1f} s ({experts_bytes / 1e9:.1f} GB of experts); '
            f'MLX holds {mx.get_active_memory() / 1e9:.1f} GB',
            file=sys.stderr, flush=True,
        )
    else:
        _evaluate(model)
        mx.clear_cache()
        print(
            f'quanteeze: serving {kind} whole in memory at checkpoint precision ({stock_bits} bits, '
            f'{experts_bytes / 1e9:.1f} GB of experts); MLX holds {mx.get_active_memory() / 1e9:.1f} GB',
            file=sys.stderr, flush=True,
        )

    _SERVING['info'] = {
        'model': path, 'model_type': kind, 'mode': 'in_memory',
        'bits': effective_bits, 'stock_bits': stock_bits,
        'experts_gb': round(experts_bytes / 1e9, 2),
        'total_experts_gb': round(experts_bytes / 1e9, 2),
        'resident_gb': round(resident / 1e9, 2),
        'room_gb': round(space / 1e9, 2) if space is not None else None,
        'active_gb': round(mx.get_active_memory() / 1e9, 2),
    }
    _enable(drafter, 'every expert in memory', in_memory=True)
    return model, processor


def _parallel_failed(why):
    print(f'quanteeze: {why}', file=sys.stderr, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
    raise RuntimeError(f'quanteeze: {why}')


def _drafter(path, kind):
    mode = variables.get('QUANTEEZE_MTP').lower()
    if variables.compact():
        kept = variables.get('QUANTEEZE_COMPACT_KEEP')
        print('quanteeze: QUANTEEZE_COMPACT=on: MTP off, vision towers and token embeddings streamed from SSD '
              '(images stay supported), and the n-gram table read from SSD'
              + (f' (except {kept})' if kept else '')
              + ': the memory left to the experts', file=sys.stderr, flush=True)
    if mode == 'off':
        return None
    if parallel.size() > 1 and kind == 'deepseek_v41':
        print('quanteeze: DeepSeek V4.1 split over GPUs: no drafting yet (its drafter reads what enters layers '
              'another rank holds)', file=sys.stderr, flush=True)
        return None
    _IN_MEMORY[0] = True
    if mode != 'on':
        drafts, why = mtp.auto(path, kind)
        print(f'quanteeze: QUANTEEZE_MTP=auto: {"drafting" if drafts else "no drafting"}: {why}',
              file=sys.stderr, flush=True)
        if not drafts:
            return None
        _IN_MEMORY[0] = mtp.pays_in_memory(kind)
    from .arch import deepseek_v41
    exact = (variables.get('QUANTEEZE_MTP_EXACT') or 'bits').strip().lower()
    deepseek_v41.EXACT[0] = exact != 'distribution'
    from . import dspark
    dspark.GATE[0] = None
    if exact == 'distribution':
        print('quanteeze: QUANTEEZE_MTP_EXACT=distribution: verify pass computes tokens together',
              file=sys.stderr, flush=True)
    bits = None if variables.off('QUANTEEZE_MTP_BITS') else variables.integer('QUANTEEZE_MTP_BITS')
    if bits is not None and not requant.native(bits):
        print(f'quanteeze: QUANTEEZE_MTP_BITS={bits}: MLX computes no {bits}-bit experts on CUDA; using 2 bits',
              file=sys.stderr, flush=True)
        bits = 2
    t = time.time()
    try:
        drafter = mtp.load(path, bits)
    except Exception as e:
        print(f'quanteeze: QUANTEEZE_MTP: no MTP drafter in {path} ({type(e).__name__}: {e}); decoding 1 token/pass',
              file=sys.stderr, flush=True)
        return None
    timing = variables.get('QUANTEEZE_MTP_TIMING')
    if timing is not None and timing.lower() in ('wall', 'sync', 'profile'):
        mtp.time_rounds(drafter[0], timing.lower())
        _at_exit('mtp_timing', lambda: print(f'quanteeze: MTP rounds timed ({timing}), {mtp.times(drafter[0])}',
                                             file=sys.stderr, flush=True))
    block = mtp.block_for(variables.get('QUANTEEZE_MTP_BLOCK'), kind)
    if block is not None:
        mtp.set_block(drafter[0], block)
    print(f'quanteeze: MTP drafter of {kind} read in {time.time() - t:.0f} s: {mtp.nbytes(drafter[0]) / 1e9:.1f} GB, '
          f'its experts at {mtp.bits_of(drafter[0])} bits, '
          + (f'{block} tokens a round' if block is not None else 'mlx-vlm\'s tokens a round')
          + mtp.gate_said(drafter[0]),
          file=sys.stderr, flush=True)
    _at_exit('mtp', lambda: print(f'quanteeze: MTP, {mtp.summary(drafter[0])}', file=sys.stderr, flush=True))
    return drafter


def _enable(drafter, when, in_memory=True):
    if drafter is None:
        return
    if in_memory and not _IN_MEMORY[0]:
        print(f'quanteeze: QUANTEEZE_MTP=auto: no drafting ({when}): {mtp.SLOWER_IN_MEMORY_WHY}',
              file=sys.stderr, flush=True)
        return
    mtp.enable(*drafter)
    print(f'quanteeze: MTP drafting on ({when}), from the next request', file=sys.stderr, flush=True)


def _at_exit(name, fn):
    _AT_EXIT[name] = fn


def _let_go():
    if not _AT_EXIT and not _SERVING and not _LOADED:
        return
    held = []
    if _SERVING.get('demand') is not None:
        held += list(getattr(_SERVING['demand'], 'switches', {}).values())
        _SERVING['demand'].close()
    prefill.release()
    q2_experts.let_go()
    ngram_prefetch.let_go()
    vision_ssd.IMAGES.clear()
    fitting.MODEL[0] = None
    _AT_EXIT.clear()
    _SERVING.clear()
    gc.collect()
    kept = set()
    for ref in _LOADED:
        model = ref()
        if model is not None and not _served(model):
            for _, module in model.named_modules():
                module.clear()
        elif model is not None:
            kept.update(id(module) for _, module in model.named_modules())
        del model
    for top in held:
        for _, module in top.named_modules():
            if id(module) not in kept:
                module.clear()
    del held
    _LOADED[:] = [ref for ref in _LOADED if ref() is not None]
    gc.collect()
    mx.clear_cache()
    allocator.trim()


def _served(model):
    from mlx_vlm.server.runtime import runtime
    cache = runtime.model_cache
    caches = [cache] if isinstance(cache, dict) else list(cache.values()) if cache is not None else []
    return any(c.get('model') is model for c in caches)


def _run_exit_summaries():
    fns = list(_AT_EXIT.values())
    _AT_EXIT.clear()
    for fn in fns:
        fn()


def _shutdown_serving():
    _run_exit_summaries()
    if _KEEPER[0] is not None:
        _KEEPER[0].close()
        _KEEPER[0] = None
    try:
        vlm = importlib.import_module('mlx_vlm.server.app')
        vlm.unload_model_sync()
    except Exception:
        pass
    _let_go()
    ngram_prefetch.close_all()
    try:
        mx.synchronize()
    except Exception:
        pass
    clear = getattr(mx, 'clear_streams', None)
    if clear is not None:
        try:
            clear()
        except Exception:
            pass


atexit.register(_shutdown_serving)


def _failures_let_go(gen):
    import traceback
    notify = gen._notify_queues
    if getattr(notify, 'quanteeze', False):
        return

    def _notify_queues(queues, *items):
        for item in items:
            if isinstance(item, BaseException) and item.__traceback__ is not None:
                traceback.clear_frames(item.__traceback__)
        return notify(queues, *items)
    _notify_queues.quanteeze = True
    gen._notify_queues = _notify_queues


def _checkpoint_path(args, kwargs):
    path = args[0] if args else kwargs.get('path_or_hf_repo')
    return os.path.abspath(os.path.expanduser(path)) if path and os.path.exists(os.path.expanduser(path)) else path


def _in_memory(model):
    return arch.in_memory(model)


def _evaluate(model):
    progress.evaluate(a for _, a in _in_memory(model))


def _resident(model, switches):
    experts = {id(a) for s in switches for _, a in tree_flatten(s.parameters())}
    rest = [a for _, a in _in_memory(model) if id(a) not in experts]
    progress.evaluate(rest)
    return sum(a.nbytes for a in rest)


HELD_AFTER = 3e9


def _host_by_rows(path, by=None, budget=None, total=None):
    """Whether the host's copy holds checkpoint `path`'s experts by rows (`QUANTEEZE_HOST_EXPERTS_BY`):
    `False` for whole tensors from the first layer on; else `True` (as many of every layer by rows)."""
    value = (by or variables.get('QUANTEEZE_HOST_EXPERTS_BY') or 'auto').strip().lower()
    if value == 'layers':
        return False
    if value == 'rows' or not q2_experts.on_network(path):
        return True
    return bool(budget is not None and total is not None and budget >= total)


def _hold_experts(source_or_path, passing=0):
    ckpt = (
        source_or_path
        if isinstance(source_or_path, (q2_experts.Checkpoint, gguf_experts.ExpertGGUF))
        else None
    )
    check_path = str(ckpt.path) if ckpt is not None else source_or_path
    value = (variables.get('QUANTEEZE_HOST_EXPERTS_GB') or 'auto').strip().lower()
    if value == 'off':
        return None
    if value == 'auto':
        if (mx.metal.is_available() or mx.default_device() != mx.gpu or cuda_ple.integrated()
                or q2_experts.in_memory(check_path)):
            return None
        budget = (fitting.host_room() or 0) - ngram_prefetch.unread() - passing - HELD_AFTER
        if budget <= 0:
            return None
    else:
        budget = float(value) * 1e9
    t = time.time()
    owned = parallel.owned()
    routed_total = None
    by_mode = (variables.get('QUANTEEZE_HOST_EXPERTS_BY') or 'auto').strip().lower()
    if by_mode == 'auto' and q2_experts.on_network(check_path):
        ckpt_obj = ckpt
        if ckpt_obj is None:
            resolved = Path(check_path).expanduser().resolve()
            if resolved.suffix.lower() == '.gguf' or (
                resolved.is_dir() and not any(resolved.glob('*.safetensors')) and any(resolved.glob('*.gguf'))
            ):
                ckpt_obj = gguf_experts.ExpertGGUF(resolved)
            else:
                ckpt_obj = q2_experts.Checkpoint(resolved)
            ckpt = ckpt_obj
        routed_total = sum(
            ckpt_obj.nbytes(n) for layer, n in ckpt_obj.routed(layers=True) if owned is None or layer in owned
        )
    by_rows = _host_by_rows(check_path, by=by_mode, budget=budget, total=routed_total)
    held = q2_experts.hold(ckpt if ckpt is not None else check_path, budget,
                           q2_experts.read_threads(check_path), owned, by_rows)
    if not held.bytes:
        return None
    if held.by_rows:
        counts = [ids.size for ids in held.expert_in.values()]
        how = (f'by rows, {min(counts)}-{max(counts)} of every layer\'s {max(held.experts.values())} experts '
               '(QUANTEEZE_HOST_EXPERTS_BY)')
    else:
        how = 'whole tensors from the first layer on (QUANTEEZE_HOST_EXPERTS_BY)'
    print(f'quanteeze: reading {held.bytes / 1e9:.1f} GB of the routed experts\' {held.total / 1e9:.1f} into '
          f'host memory (QUANTEEZE_HOST_EXPERTS_GB {value}), {how}',
          file=sys.stderr, flush=True)

    def done():
        try:
            held.wait()
        except Exception as e:
            print(f'quanteeze: host copy of experts stopped: {e}', file=sys.stderr, flush=True)
            return
        if held.arrays:
            print(f'quanteeze: {held.bytes / 1e9:.1f} GB of experts held in host memory, read in '
                  f'{time.time() - t:.0f} s', file=sys.stderr, flush=True)

    threading.Thread(target=done, name='host-copy-done', daemon=True).start()
    return held


def _hold_tables(model, path=None):
    value = (variables.get('QUANTEEZE_HOST_NGRAM_GB') or 'auto').strip().lower()
    if value == 'off':
        return None
    if value == 'auto':
        if mx.metal.is_available() or mx.default_device() != mx.gpu or cuda_ple.integrated():
            return None
        budget = fitting.host_room()
        if not budget:
            return None
    else:
        budget = float(value) * 1e9
    t = time.time()
    held = ngram_prefetch.hold(model, budget, q2_experts.read_threads(path), q2_experts.in_memory)
    if held is None or not held.bytes:
        return None
    print(f'quanteeze: reading {held.bytes / 1e9:.1f} GB of the n-gram tables\' {held.total / 1e9:.1f} into '
          f'host memory (QUANTEEZE_HOST_NGRAM_GB {value})', file=sys.stderr, flush=True)

    def done():
        try:
            read = held.wait()
        except Exception as e:
            print(f'quanteeze: host copy of n-gram tables stopped: {e}', file=sys.stderr, flush=True)
            return
        if read:
            print(f'quanteeze: {read / 1e9:.1f} GB of n-gram tables held in host memory, read in '
                  f'{time.time() - t:.0f} s', file=sys.stderr, flush=True)

    threading.Thread(target=done, name='ngram-copy-done', daemon=True).start()
    return held


def _auto_gb(resident, space):
    if space is None:
        raise SystemExit('quanteeze: GB auto: no working set is known here; give QUANTEEZE_FROM_SSD_GB')
    if space <= 0:
        raise SystemExit(
            f'quanteeze: the model without its routed experts holds {resident / 1e9:.1f} GB, and with '
            f'QUANTEEZE_MEMORY_RESERVE ({fitting.reserve():g} of the working set) no room is left for experts'
        )
    return round(space / 1e9, 1)


def _check_load(path):
    uniform = _uniform_bits()
    _experts_gguf(uniform)
    _demand_variant()


def _load_from_page(path):
    vlm = importlib.import_module('mlx_vlm.server.app')
    unload = getattr(vlm, '_unload_model_cache_group', None)
    if unload is not None:
        unload('text_generation')
    else:
        vlm.unload_model_sync()
    vlm.get_cached_model(path, model_kind='text_generation')


CACHE_SHARE = 0.03


def _cache_limit():
    """MLX's limit on freed buffers it keeps, in bytes: QUANTEEZE_CACHE_GB (auto: 2 GB, and on a GPU
    no more than CACHE_SHARE of its working set)."""
    text = (variables.get('QUANTEEZE_CACHE_GB') or 'auto').strip().lower()
    if text != 'auto':
        return int(variables.number('QUANTEEZE_CACHE_GB') * 1e9)
    limit = int(2e9)
    if mx.default_device() == mx.gpu:
        budget = guards.recommended_budget()
        if budget:
            limit = min(limit, int(CACHE_SHARE * budget))
    return limit


def _checkpoints_a_prompt(coordinator_cls=None):
    """mlx-vlm's checkpoints of a prompt (APCCoordinator.checkpoint_lengths) as with two entries
    kept: its end and the last interval before it, however many APC_CHECKPOINT_ENTRIES keeps."""
    from mlx_vlm.apc import adjust_prefix_to_text_suffix_boundary
    if coordinator_cls is None:
        from mlx_vlm.apc_coordinator import APCCoordinator as coordinator_cls
    if getattr(coordinator_cls, '_quanteeze_two', False):
        return

    def checkpoint_lengths(self, token_ids, media_token_ids):
        final = self.checkpoint_len(token_ids, media_token_ids)
        if final <= 0:
            return []
        m = self.manager
        interval = m.checkpoint_interval_tokens
        if interval <= 0 or (m._exact_cache_max or (2 if m.disk else 1)) <= 1:
            return [final]
        interval = ((interval + m.block_size - 1) // m.block_size) * m.block_size
        last = ((final - 1) // interval) * interval
        lengths = {final}
        if last >= interval:
            boundary = adjust_prefix_to_text_suffix_boundary(token_ids, last, media_token_ids, max_prefix_tokens=final)
            if m.exact_cache_min_tokens <= boundary < final:
                lengths.add(boundary)
        return sorted(lengths)
    coordinator_cls.checkpoint_lengths = checkpoint_lengths
    coordinator_cls._quanteeze_two = True


def _out_of_memory(error):
    text = str(error)
    return 'Insufficient Memory' in text or 'OutOfMemory' in text


_APC = dict(given=False, off_gpu=False)
_APC_MANAGERS = weakref.WeakSet()


def _apc_seen(manager_cls=APCManager):
    if getattr(manager_cls, '_quanteeze_seen', False):
        return
    init = manager_cls.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        _APC_MANAGERS.add(self)
        if _APC['off_gpu']:
            self.memory_max_bytes = 0
    manager_cls.__init__ = __init__
    manager_cls._quanteeze_seen = True


def _apc_reads_free_memory(module=None, readings=None):
    if module is None:
        from mlx_vlm import apc as module
    if getattr(module, '_quanteeze_free', False) or module._free_ram_bytes() is not None:
        return
    readings = guards.Readings() if readings is None else readings
    if (readings._meminfo() or {}).get('MemAvailable') is None:
        return

    def _free_ram_bytes():
        free = (readings._meminfo() or {}).get('MemAvailable')
        limited = readings.cgroup_memory()
        if free is not None and limited is not None:
            free = min(free, limited[0] - limited[1])
        return None if free is None else int(free)
    module._free_ram_bytes = _free_ram_bytes
    module._quanteeze_free = True


FULL_GPU_PREFILL = 512
ON_DEMAND_COALESCE_S = 0.012
_PREFILL = dict(given=False, short=False)


def _coalesce_on_demand(generator_cls):
    if getattr(generator_cls, '_quanteeze_coalesce', False):
        return
    own = generator_cls._collect_pending_requests

    def _collect_pending_requests(self, *, active, capacity=None, idle_timeout=0.1, coalesce_s=0.0):
        if (not active and coalesce_s <= 0.0 and _SERVING.get('demand') is not None
                and (capacity is None or capacity > 1)):
            coalesce_s = ON_DEMAND_COALESCE_S
        return own(self, active=active, capacity=capacity, idle_timeout=idle_timeout, coalesce_s=coalesce_s)

    generator_cls._collect_pending_requests = _collect_pending_requests
    generator_cls._quanteeze_coalesce = True


def _prefill_steps(generator_cls):
    if getattr(generator_cls, '_quanteeze_steps', False):
        return
    own = generator_cls._effective_prefill_step_size

    def _effective_prefill_step_size(self):
        n = own(self)
        return min(n, FULL_GPU_PREFILL) if _PREFILL['short'] and n > 0 else n
    generator_cls._effective_prefill_step_size = _effective_prefill_step_size
    generator_cls._quanteeze_steps = True


def _prefill_short():
    if _PREFILL['given'] or mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    _PREFILL['short'] = True
    print(f'quanteeze: a prompt goes through {FULL_GPU_PREFILL} tokens a step, not 2048: a longer step\'s work '
          'would overflow the GPU the experts fill (--prefill-step-size or PREFILL_STEP_SIZE sets it)',
          file=sys.stderr, flush=True)
    return True


def _apc_off_the_gpu():
    if _APC['given'] or mx.default_device() != mx.gpu:
        return False
    _APC['off_gpu'] = True
    os.environ['APC_MEMORY_MAX_GB'] = '0'
    for m in list(_APC_MANAGERS):
        m.memory_max_bytes = 0
    if os.environ.get('APC_ENABLED', '0').lower() in ('1', 'true', 'yes'):
        disk = os.environ.get('APC_DISK_ENABLED', '1').lower() in ('1', 'true', 'yes')
        print('quanteeze: the prompt cache (APC) keeps none in the GPU, whose memory the experts fill: '
              + ('its disk tier alone, the host\'s page cache holding what was written lately' if disk
                 else 'none at all, its disk tier off (APC_DISK_ENABLED)')
              + ' (APC_MEMORY_MAX_GB keeps them there)', file=sys.stderr, flush=True)
    return True


def _image_embeddings_now(generator_cls):
    embed = generator_cls._gpu_embed

    def _gpu_embed(self, raw_inputs, *args, **kwargs):
        images = raw_inputs.get('pixel_values') is not None
        if images:
            mx.clear_cache()
        for attempt in range(2):
            try:
                input_ids, gen_kwargs = embed(self, raw_inputs, *args, **kwargs)
                if images and gen_kwargs.get('inputs_embeds') is not None:
                    mx.eval(gen_kwargs['inputs_embeds'])
                    mx.clear_cache()
                return input_ids, gen_kwargs
            except RuntimeError as error:
                if not images or attempt or not _out_of_memory(error):
                    raise
                gen_kwargs = None
                gc.collect()
                mx.clear_cache()
                print('quanteeze: Metal ran out of memory embedding a prompt\'s images: its freed buffers given '
                      'back, made once more', file=sys.stderr, flush=True)

    generator_cls._gpu_embed = _gpu_embed


def _prompt_steps_waited(batch_cls):
    step = batch_cls.prompt_step

    def prompt_step(self):
        n = step(self)
        if n:
            mx.synchronize()
            if not getattr(self, 'entries', True) and (least := cuda_ple.free_least()) is not None:
                cuda_ple.room_back(least)
        return n

    batch_cls.prompt_step = prompt_step


def _wired_limit():
    return int(variables.number('QUANTEEZE_WIRED_GB') * 1e9)


def _cache_eval_when_batched(generator_cls):
    def get(self):
        batch = getattr(self, '_generation_batch', None)
        return self.__dict__.get('_quanteeze_interval', 0) if batch is not None and len(batch) > 1 else 0

    def set_(self, interval):
        self.__dict__['_quanteeze_interval'] = interval

    generator_cls._cache_eval_interval = property(get, set_)


def _idle():
    return (
        control._SWITCH['inflight'] == 0
        and not control._SWITCH['busy']
        and not progress.state()['loading']
        and not q2_experts.reading()
        and not ngram_prefetch.unread()
    )


def _shed():
    gc.collect()
    mx.clear_cache()
    allocator.trim()


def _exit_now():
    sys.stderr.flush()
    os._exit(1)


def _keep_memory():
    """The server's footprint kept to (quanteeze.guards.Keeper), or None with QUANTEEZE_KEEP_GB=off."""
    import faulthandler
    keeper = guards.Keeper.from_env(idle=_idle, shed=_shed, stop=lambda: os.kill(os.getpid(), signal.SIGTERM),
                                    exit=_exit_now, stacks=lambda: faulthandler.dump_traceback(all_threads=True))
    if keeper is None:
        print('quanteeze: the memory keeper off (QUANTEEZE_KEEP_GB)', file=sys.stderr, flush=True)
        return None
    print(f'quanteeze: the server keeps to {keeper.limit / 1e9:.1f} GB (QUANTEEZE_KEEP_GB), MLX\'s freed buffers '
          'given back over it; it stops itself '
          + (f'at {keeper.stop_at / 1e9:.1f} GB, ' if keeper.stop_at is not None else '')
          + f'or over the limit with the swap grown {keeper.swap_growth / 1e9:g} GB or under '
          f'{keeper.disk_floor / 1e9:g} GB left on the swap\'s disk', file=sys.stderr, flush=True)
    _KEEPER[0] = keeper.start()
    atexit.register(lambda: print(f'quanteeze: memory, {keeper.summary()}', file=sys.stderr, flush=True))
    return keeper


def _terminated(sig, frame):
    raise SystemExit(0 if shutdown._STATE['signalled'] else 128 + sig)


def _numpy_huge_pages():
    """Disable NumPy transparent huge pages on Linux by default (`QUANTEEZE_NUMPY_HUGE_PAGES=off`)."""
    if (not sys.platform.startswith('linux') or os.environ.get('NUMPY_MADVISE_HUGEPAGE')
            or not variables.off('QUANTEEZE_NUMPY_HUGE_PAGES')):
        return False
    try:
        from numpy._core import multiarray
    except ImportError:
        from numpy.core import multiarray
    switch = getattr(multiarray, '_set_madvise_hugepage', None)
    if switch is None:
        return False
    switch(False)
    print('quanteeze: numpy asks for no huge pages for its large arrays (QUANTEEZE_NUMPY_HUGE_PAGES)',
          file=sys.stderr, flush=True)
    return True


def _warn_other_servers():
    """Warn on stderr if another MLX server (`quanteeze`, `mlx_vlm.server`, or `mlx_lm.server`) is
    already running on this machine."""
    if parallel.rank():
        return
    import subprocess
    try:
        out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return
    skip = {os.getpid(), os.getppid()}
    found = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid in skip or 'grep' in cmd:
            continue
        if any(tag in cmd for tag in ('quanteeze', 'mlx_vlm.server', 'mlx_lm.server')):
            found.append((pid, cmd[:100]))
    if found:
        listed = '; '.join(f'pid {pid} ({cmd})' for pid, cmd in found[:3])
        print(f'quanteeze: WARNING another MLX server process is already running: {listed}',
              file=sys.stderr, flush=True)


def _extract_cli_flags(argv):
    """Strip Quanteeze-only CLI flags from `argv` and mirror them into environment variables."""
    kept = list(argv[:1])
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == '--experts-gguf':
            if i + 1 >= len(argv):
                raise SystemExit('quanteeze: --experts-gguf requires a path argument')
            os.environ['QUANTEEZE_EXPERTS_GGUF'] = argv[i + 1]
            i += 2
        elif arg.startswith('--experts-gguf='):
            os.environ['QUANTEEZE_EXPERTS_GGUF'] = arg.split('=', 1)[1]
            i += 1
        else:
            kept.append(arg)
            i += 1
    return kept


def main():
    sys.argv = _extract_cli_flags(sys.argv)
    alloc_msg = allocator.ensure_active(sys.argv[1:])
    if sys.platform.startswith('linux'):
        try:
            import ctypes
            ctypes.CDLL(None).prctl(15, b'quanteeze', 0, 0, 0)
        except Exception:
            pass
    signal.signal(signal.SIGTERM, _terminated)
    import faulthandler
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    if alloc_msg:
        print(f'quanteeze: {alloc_msg}', file=sys.stderr, flush=True)
    _PREFILL['given'] = bool(os.environ.get('PREFILL_STEP_SIZE')) or any(
        a == '--prefill-step-size' or a.startswith('--prefill-step-size=') for a in sys.argv[1:])
    if os.environ.get('PREFILL_STEP_SIZE') and '--prefill-step-size' not in sys.argv:
        sys.argv += ['--prefill-step-size', os.environ['PREFILL_STEP_SIZE']]
    for line in unindexed.unindex_all():
        print(f'quanteeze: {line}', file=sys.stderr, flush=True)
    for line in control.start():
        print(f'quanteeze: {line}', file=sys.stderr, flush=True)
    _numpy_huge_pages()
    _warn_other_servers()
    parallel.launch()
    if parallel.init() is not None:
        os.environ['APC_ENABLED'] = '0'
        print(f'quanteeze: rank {parallel.rank()} of {parallel.size()}; prompt cache (APC) off over several GPUs',
              file=sys.stderr, flush=True)
    for name in variables.unknown():
        print(f'quanteeze: WARNING {name} is set, but quanteeze reads no such variable '
              '(python -m quanteeze.variables lists them)', file=sys.stderr, flush=True)
    for old, new in variables.renamed():
        print(f'quanteeze: WARNING {old} is an old name, read as {" and ".join(new)}',
              file=sys.stderr, flush=True)
    os.environ.setdefault('MLX_VLM_MAX_TOKENS', '32768')
    os.environ.setdefault('APC_ENABLED', '1')
    os.environ.setdefault('APC_DISK_ENABLED', '1')
    os.environ.setdefault('APC_HASH', 'sha256')
    _APC['given'] = 'APC_MEMORY_MAX_GB' in os.environ
    os.environ.setdefault('APC_MEMORY_MAX_GB', '4')
    os.environ.setdefault('APC_CHECKPOINT_ENTRIES', '32')
    apc_disk.ON[0] = not variables.off('QUANTEEZE_APC_ENCRYPT')
    apc_disk.use_precision(_precision)
    if apc_disk.ON[0]:
        apc_disk.install()
        atexit.register(
            lambda: print(f'quanteeze: APC disk sealing, {apc_disk.summary()}', file=sys.stderr, flush=True)
            if any(apc_disk.STATS.values()) else None
        )
    _checkpoints_a_prompt()
    _apc_seen()
    _apc_reads_free_memory()
    if not any(a == '--host' or a.startswith('--host=') for a in sys.argv[1:]):
        sys.argv += ['--host', '127.0.0.1']
    if not any(a == '--port' or a.startswith('--port=') for a in sys.argv[1:]):
        sys.argv += ['--port', control.DEFAULT_PORT]
    if 'MLX_VLM_BATCH_CACHE_EVAL_INTERVAL' not in os.environ:
        _cache_eval_when_batched(BatchGenerator)
    mx.set_cache_limit(_cache_limit())
    print(f'quanteeze: MLX keeps at most {_cache_limit() / 1e9:g} GB of freed buffers; '
          f'APC at most {os.environ["APC_MEMORY_MAX_GB"]} GiB of cached prompts '
          f'({"AES-256-GCM disk sealing on" if apc_disk.ON[0] else "disk sealing off"})',
          file=sys.stderr, flush=True)
    _keep_memory()
    if _wired_limit():
        mx.set_wired_limit(_wired_limit())
        print(f'quanteeze: up to {_wired_limit() / 1e9:g} GB wired between requests', file=sys.stderr, flush=True)
    generation.load = _load_hooked
    efforts.install(generation.GenerationArguments)
    prefetcher = ngram_prefetch.use()
    print(f'quanteeze: n-gram pages read ahead by {prefetcher.threads} threads' if prefetcher.threads
          else 'quanteeze: n-gram read-ahead off', file=sys.stderr, flush=True)
    apc_images.install(generation.ResponseGenerator, BatchGenerator, APCManager)
    apc_images.use_precision(_precision)
    _image_embeddings_now(generation.ResponseGenerator)
    _prefill_steps(generation.ResponseGenerator)
    _coalesce_on_demand(generation.ResponseGenerator)
    _prompt_steps_waited(PromptProcessingBatch)
    mtp.install_batches(BatchGenerator)
    _ADMISSION[0] = admission.Admission.from_env()
    if _ADMISSION[0] is not None:
        admission.install(BatchGenerator, _ADMISSION[0])
        print(f'quanteeze: waiting prompts taken into a prefill as memory holds them, MLX at most '
              f'{_ADMISSION[0].limit / 1e9:.1f} GB (QUANTEEZE_ADMIT_SHARE), '
              + (f'{_ADMISSION[0].ahead} tokens of cache growth kept free per decoding row'
                 if _ADMISSION[0].ahead else 'no cache growth kept free'), file=sys.stderr, flush=True)
        _at_exit('admission', lambda: print(f'quanteeze: admission, {_ADMISSION[0].summary()}', file=sys.stderr,
                                            flush=True))
    dsml.install(_server)
    toolcalls.install(generation.ResponseGenerator)
    reask.install(generation.ResponseGenerator)
    shutdown.install(generation.ResponseGenerator)
    phases.install(generation.ResponseGenerator)
    shutdown.patch(uvicorn.server.Server, variables.number('QUANTEEZE_SHUTDOWN_S'))
    mtp.install(generation.ResponseGenerator)
    _failures_let_go(generation)
    verify = variables.get('QUANTEEZE_VERIFY_QMV').lower()
    if verify != 'off':
        if verify.isdigit() and int(verify) > 0:
            verify_qmv.PER = min(int(verify), verify_qmv.MAX_TOKENS)
        verify_qmv.install()
        atexit.register(lambda: print(f'quanteeze: verify projections, {verify_qmv.summary()}', file=sys.stderr,
                                      flush=True) if verify_qmv.CHOICE or verify_qmv.HC_CHOICE else None)
        from .arch import deepseek_v41
        atexit.register(lambda: print(f'quanteeze: V4.1 verify indexer, {deepseek_v41.GROUP_STATS}',
                                      file=sys.stderr, flush=True) if deepseek_v41.GROUP_CHOICE else None)
    expert_qmv.ON[0] = not variables.off('QUANTEEZE_EXPERT_QMV')
    kquant_qmv.ON[0] = not variables.off('QUANTEEZE_KQUANT_QMV')
    gguf_qmv.ON[0] = kquant_qmv.ON[0]
    hyper.ON[0] = not variables.off('QUANTEEZE_HC_KERNELS')
    glm_hyper.ON[0] = hyper.ON[0]
    cuda_v41.ON[0] = hyper.ON[0]
    from .arch import qwen4_exp
    qwen4_exp.COMPILE[0] = not variables.off('QUANTEEZE_COMPILE_DECODE')
    qwen4_exp.EVAL_LAYERS[0] = variables.integer('QUANTEEZE_EVAL_LAYERS')
    linear_attn.ON[0] = not variables.off('QUANTEEZE_LINEAR_ATTN_KERNELS')
    try:
        linear_attn.share_fused_linears()
    except (ImportError, AttributeError) as error:
        print(f'quanteeze: gated delta projections kept apart from their fused copy ({error})',
              file=sys.stderr, flush=True)
    sigmoid_router.ON[0] = not variables.off('QUANTEEZE_SIGMOID_ROUTER_KERNELS')
    glm_indexer.ON[0] = not variables.off('QUANTEEZE_INDEXER_KERNEL')
    linear_attn.DELTA[0] = not variables.off('QUANTEEZE_CUDA_DELTA')
    linear_attn.few()
    cuda_qmv.ON[0] = not variables.off('QUANTEEZE_CUDA_QMV')
    cuda_router.ON[0] = not variables.off('QUANTEEZE_CUDA_ROUTER')
    cuda_v41.ROUTE_ON[0] = cuda_router.ON[0]
    cuda_experts.ON[0] = not variables.off('QUANTEEZE_CUDA_EXPERTS')
    cuda_kernels.FUSE[0] = not variables.off('QUANTEEZE_CUDA_FUSE')
    cuda_attn.ON[0] = not variables.off('QUANTEEZE_CUDA_ATTN')
    atexit.register(lambda: print(f'quanteeze: the PLE layer\'s kernels, {cuda_ple_layer.summary()}', file=sys.stderr,
                                  flush=True) if cuda_ple_layer.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: the full attention\'s kernels, {cuda_attn.summary()}', file=sys.stderr,
                                  flush=True) if cuda_attn.CHOICE or cuda_attn.DECLINED else None)
    cuda_head.ON[0] = not variables.off('QUANTEEZE_CUDA_HEAD')
    if cuda_head.ON[0] and cuda_kernels.available():
        cuda_head.install()
        atexit.register(lambda: print(f'quanteeze: the head\'s kernels, {cuda_head.summary()}', file=sys.stderr,
                                      flush=True) if cuda_head.CHOICE else None)
    cuda_ple.ON[0] = not variables.off('QUANTEEZE_CUDA_PLE')
    cuda_ple.HOST[0] = not variables.off('QUANTEEZE_HOST_NGRAM')
    if cuda_ple.HOST[0]:
        cuda_ple.track_tokens()
    if (cuda_ple.ON[0] and cuda_kernels.available()) or cuda_ple.HOST[0]:
        cuda_ple.install()
        atexit.register(lambda: print(f'quanteeze: the n-gram rows\' kernel, {cuda_ple.summary()}', file=sys.stderr,
                                      flush=True) if cuda_ple.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: DeepSeek V4.1 CUDA kernels, {cuda_v41.summary()}', file=sys.stderr,
                                  flush=True) if cuda_v41.STATS['checked'] or cuda_v41.ROUTE[0] or cuda_v41.FP8_CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: the MoE blocks\' kernels, {cuda_moe.summary()}', file=sys.stderr,
                                  flush=True) if cuda_moe.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: CUDA experts, {cuda_experts.summary()}', file=sys.stderr,
                                  flush=True) if cuda_experts.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: CUDA router, {cuda_router.summary()}', file=sys.stderr,
                                  flush=True) if cuda_router.CHOICE else None)
    if cuda_qmv.ON[0] and cuda_qmv.install():
        atexit.register(lambda: print(f'quanteeze: CUDA decode qmv, {cuda_qmv.summary()}', file=sys.stderr,
                                      flush=True) if cuda_qmv.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: linear attention kernels, {linear_attn.summary()}', file=sys.stderr,
                                  flush=True) if linear_attn.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: hyper-connection kernels, {hyper.summary()}', file=sys.stderr,
                                  flush=True) if hyper.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: sigmoid router kernels, {sigmoid_router.summary()}', file=sys.stderr,
                                  flush=True) if sigmoid_router.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: GLM-5.3 linear attention kernels, {glm_linear_attn.summary()}',
                                  file=sys.stderr, flush=True) if glm_linear_attn.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: GLM-5.3 hyper-connection kernels, {glm_hyper.summary()}',
                                  file=sys.stderr, flush=True) if glm_hyper.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: GLM-5.3 indexer kernel, {glm_indexer.summary()}',
                                  file=sys.stderr, flush=True) if glm_indexer.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: expert_qmv kernels, {expert_qmv.summary()}', file=sys.stderr,
                                  flush=True) if expert_qmv.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: kquant_qmv kernels, {kquant_qmv.summary()}', file=sys.stderr,
                                  flush=True) if kquant_qmv.CHOICE else None)
    atexit.register(lambda: print(f'quanteeze: gguf_qmv kernels, {gguf_qmv.summary()}', file=sys.stderr,
                                  flush=True) if gguf_qmv.CHOICE else None)
    watch = variables.get('QUANTEEZE_LOOPS').lower()
    if watch != 'off':
        loops.install(generation.ResponseGenerator)
    _experts_gguf(_uniform_bits()), _demand_variant()
    print('quanteeze: each request\'s tokens through ' + ', '.join(stream.stages(generation.ResponseGenerator))
          + ' (innermost first)', file=sys.stderr, flush=True)
    page = control.install(_server.app, loader=_load_from_page, check=_check_load)
    parallel.install(generation.ResponseGenerator)
    print(f'quanteeze: control routes ask for a token; Browser UI at '
          f'{page.replace(control.PAGE_PATH, control.HOME_PATH, 1)}, all settings at {page}',
          file=sys.stderr, flush=True)
    try:
        _serve()
    finally:
        _shutdown_serving()


if __name__ == '__main__':
    main()
