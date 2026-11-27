# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Run concurrent inference requests each on a dedicated worker thread and MLX stream.

mlx-vlm's ResponseGenerator normally runs one generation loop interleaving every active request,
which prevents batch=1 speculative drafting while another prompt runs and serializes synchronous waits.
With QUANTEEZE_REQUEST_THREADS above 1, each request runs to completion on a dedicated thread
with its own MLX stream, KV cache, sampler PRNG state, and thread-local pass storage
(`quanteeze.per_thread`), provided active on-demand streaming (`Demand`) is coordinated by
the C++ STM driver (`QUANTEEZE_DEMAND_STM=on`), or `QUANTEEZE_REQUEST_THREADS_ANY=on` is set when
all model weights reside in memory.
"""
import contextlib
import copy
import itertools
import secrets
import sys
import threading
import traceback

import mlx.core as mx

from . import leases, per_load, per_thread, variables

_PROMPT = threading.Lock()
_BUILD = threading.Lock()
_RNG = threading.Lock()
_UIDS = itertools.count(1)
_RUNNING = {}
_RUNNING_LOCK = threading.Lock()
_ROUNDS = threading.Condition(_RUNNING_LOCK)
_IN_ROUND = [0]
_PROMPTING = [0]
_WORKER = threading.local()
_SERVING = set()
_SERVING_LOCK = threading.RLock()
_SAID_ALONE = [False]
WHOLE = [False]
_WIRING = dict(generating=0, before=None)
_WIRING_LOCK = threading.Lock()

STATS = dict(
    first_here=0,
    threads_started=0,
    alone=0,
    together=0,
    held_for_room=0,
    most_at_once=0,
    mtp_rounds=0,
    plain_steps=0,
    cpu_blocks=0,
    prompt_turn_waits=0,
    turn_timeouts=0,
    wired=0,
)

PER_REQUEST = ('_rope_deltas', '_position_ids')
SKIP_EVALUATING = ('engram', 'vision')
EVALUATE_AT_MOST = 8e9


_LOADED = dict(threads=None, any=False)
_LOOP_THREAD = [None]


def _count(key, n=1):
    with _RUNNING_LOCK:
        STATS[key] += n


def running():
    """Return the number of active request worker executions currently running."""
    with _RUNNING_LOCK:
        return len(_RUNNING)


def _load_settings(environ=None):
    """Snapshot QUANTEEZE_REQUEST_THREADS and _ANY into `_LOADED` (live env when `environ is None`)."""
    _LOADED.update(
        threads=max(1, variables.integer('QUANTEEZE_REQUEST_THREADS', environ)),
        any=(variables.get('QUANTEEZE_REQUEST_THREADS_ANY', environ) or 'off').strip().lower() == 'on',
    )
    return _LOADED['threads']


def threads():
    """Return QUANTEEZE_REQUEST_THREADS clamped to at least 1."""
    n = _LOADED['threads']
    return max(1, variables.integer('QUANTEEZE_REQUEST_THREADS', variables.at_load())) if n is None else n


def _any():
    if _LOADED['threads'] is not None:
        return _LOADED['any']
    return (variables.get('QUANTEEZE_REQUEST_THREADS_ANY', variables.at_load()) or 'off').strip().lower() == 'on'


def request_lanes():
    """Number of host mailbox lanes needed for concurrent CPU expert serving."""
    n = threads()
    return 1 if n <= 1 else n + 1


@per_load.on_let_go
def reset():
    """Reset per-load request-thread counters (and `_LOADED` when not inside the active loop's own load)."""
    with _RUNNING_LOCK:
        for k in STATS:
            STATS[k] = 0
    _SAID_ALONE[0] = False
    if threading.current_thread() is not _LOOP_THREAD[0]:
        _LOADED.update(threads=None, any=False)


def _turns():
    """Return the active C++ STM driver when prompt/decode GPU turn arbitration is enabled, else None."""
    if threads() < 2 or variables.off('QUANTEEZE_PROMPT_DECODE_TURNS'):
        return None
    from . import demand
    for d in list(demand._LIVE):
        stm = getattr(d, '_stm', None)
        if stm is not None and stm.started:
            return stm
    return None


class _PromptTurns:
    """Prompt unit-of-work GPU arbiter hook installed on `quanteeze.prefill.TURNS`."""

    def active(self):
        return _turns() is not None

    def contended(self):
        return running() > 1

    def tokens(self):
        return max(1, variables.integer('QUANTEEZE_PROMPT_CONTENDED_TOKENS'))

    @contextlib.contextmanager
    def turn(self, layer):
        from . import arch, stm_core
        d = _turns() if running() > 1 else None
        if d is None:
            yield
            return
        birth = d.now_ns()
        if not d.turn_try(stm_core.TURN_PROMPT, birth):
            _count('prompt_turn_waits')
            sw = arch.switch_of(layer) if layer is not None else None
            dem = getattr(sw, 'demand', None) if sw is not None else None
            stm = getattr(dem, '_stm', None) if dem is not None else None
            if stm is not None and stm.started:
                stm.layer_done(sw.layer)
            if d.turn_take(stm_core.TURN_PROMPT, birth) < 0:
                _count('turn_timeouts')
        try:
            yield
        finally:
            d.turn_give()


@contextlib.contextmanager
def _decode_turn(prompting):
    """Acquire a decode GPU turn (oldest-birth-wins against contended prompt units) around one decode `_step`."""
    from . import stm_core
    d = None if prompting else _turns()
    if d is None:
        yield
        return
    per_thread.PASS.birth = birth = d.now_ns()
    if d.turn_take(stm_core.TURN_DECODE, birth) < 0:
        _count('turn_timeouts')
    try:
        yield
    finally:
        d.turn_give()
        per_thread.PASS.birth = 0


def together_ok():
    """Return True if concurrent request execution is currently safe to overlap (why_alone found nothing)."""
    return why_alone() is None


def why_alone():
    """Why requests may not run at once now (together_ok's conditions, in words for the home page: state), or None."""
    from . import demand, kernel_checks, parallel
    if parallel.size() > 1:
        return 'the model is split over several GPUs'
    if not kernel_checks.shared():
        return 'the C++ extension is not in use (QUANTEEZE_NATIVE=off, or it did not build)'
    live = list(demand._LIVE)
    if not live:
        if WHOLE[0] or _any():
            return None
        return "the model is not loaded whole (its experts are a GGUF's)"
    for d in live:
        if d._stm is None:
            why = getattr(d, '_stm_why', None)
            return 'the C++ STM does not serve the on-demand reads' + (f' ({why})' if why else '')
        if not d._stm.started:
            return "the C++ STM starts at the first request's pass"
    return None


def state():
    """For the home page (home.live): requests at once as set now ('set') and as the model's load read them
    ('loaded': None before a load's loop began), why they go one at a time where they do (why_alone, or a drafter
    that drafts with its request alone: drafts_alone), and the counts so far (STATS)."""
    from . import mtp
    set_now = max(1, variables.integer('QUANTEEZE_REQUEST_THREADS'))
    loaded = _LOADED['threads']
    why = None
    if (loaded or set_now) > 1:
        why = why_alone()
        drafter = mtp.ACTIVE[0]
        if why is None and drafter is not None and drafter[1] != 'dflash':
            why = f'a drafter other than DFlash ({mtp.name(drafter[0])}) drafts with its request alone'
    return {'set': set_now, 'loaded': loaded, 'why_alone': why, 'running': running(),
            **{k: STATS[k] for k in ('together', 'alone', 'most_at_once')}}


def drafts_alone(generator):
    """Return True if the active drafter requires solo execution (non-DFlash speculative heads)."""
    if getattr(generator, 'draft_model', None) is None or getattr(generator, 'draft_kind', None) == 'dflash':
        return False
    if not _SAID_ALONE[0]:
        _SAID_ALONE[0] = True
        print(
            f'quanteeze: request threads: a {generator.draft_kind} drafter runs requests one at a time '
            '(only DFlash rounds synchronize cleanly beside concurrent requests)',
            file=sys.stderr,
            flush=True,
        )
    return True


def while_alone(fn):
    """Execute `fn()` only when no other thread is currently serving a request."""
    with _SERVING_LOCK:
        if others_serving():
            return False
        fn()
        return True


def others_serving():
    """Return True if any thread other than the calling thread is actively serving a request."""
    me = threading.get_ident()
    return any(t != me for t in list(_SERVING))


class _OwnRandom:
    """Wrap a sampler with an isolated MLX PRNG key swapped safely under `_RNG`."""

    def __init__(self, sampler, seed):
        self.sampler = sampler
        self.key = mx.random.key(int(seed) & 0xFFFFFFFF)

    def __call__(self, logprobs):
        with _RNG:
            saved = [mx.array(s) for s in mx.random.state]
            mx.random.state[0][:] = self.key
            try:
                out = self.sampler(logprobs)
            finally:
                self.key = mx.array(mx.random.state[0])
                for i, value in enumerate(saved):
                    mx.random.state[i][:] = value
        return out

    def __getattr__(self, name):
        return getattr(self.sampler, name)


def _sampler(generator, args):
    made = generator._make_sampler(args)
    if made is None:
        return None
    seed = args.seed if getattr(args, 'seed', None) is not None else secrets.randbits(32)
    return _OwnRandom(made, seed)


def _drafter(generator):
    drafter = getattr(generator, 'draft_model', None)
    if drafter is None:
        return None
    mine = copy.copy(drafter)
    if 'filter_batch' in vars(drafter):
        from . import mtp
        if hasattr(mtp, 'filter_caches'):
            vars(mine).pop('filter_batch', None)
            mtp.filter_caches(mine)
    return mine


def _room_for(request, workers=()):
    from . import serve
    adm = serve._ADMISSION[0]
    with _RUNNING_LOCK:
        active = list(_RUNNING.values())
    if len(active) < sum(1 for w in workers if w.is_alive()):
        return False
    if not active or adm is None:
        return True
    remaining = [max(0, m - n) for m, n in active] + [int(getattr(request.args, 'max_tokens', 0) or 0)]
    if adm.kv_token:
        needed = adm.reserved(remaining)
    else:
        per = leases.sizes()['kv_token']
        needed = per * sum(min(r, adm.ahead) for r in remaining) if per and adm.ahead else 0.0
    return needed <= adm.limit - mx.get_active_memory()


def _batch_generator(generator, args, drafter):
    from mlx_vlm.generate import ar
    from mlx_vlm.server import generation as G
    with _BUILD:
        was = ar.wired_limit
        ar.wired_limit = lambda *a, **k: contextlib.nullcontext()
        try:
            batch_gen = G.BatchGenerator(
                generator.model.language_model,
                generator.processor,
                stop_tokens=generator.stop_tokens,
                sampler=_sampler(generator, args),
                kv_bits=generator.kv_bits,
                kv_key_bits=getattr(generator, 'kv_key_bits', None),
                kv_value_bits=getattr(generator, 'kv_value_bits', None),
                kv_key_scheme=getattr(generator, 'kv_key_scheme', None),
                kv_value_scheme=getattr(generator, 'kv_value_scheme', None),
                kv_group_size=generator.kv_group_size,
                kv_quant_scheme=generator.kv_quant_scheme,
                quantized_kv_start=generator.quantized_kv_start,
                compute_logprobs=bool(args.logprobs),
                top_logprobs_k=generator.top_logprobs_k if args.logprobs else 0,
                stream=mx.default_stream(mx.default_device()),
                apc_manager=generator.apc_manager,
                draft_model=drafter,
                draft_kind=generator.draft_kind if drafter is not None else None,
                draft_block_size=G._get_draft_block_size_from_env(),
                greedy_sampling=args.temperature == 0,
                prefill_step_size=generator._effective_prefill_step_size(),
            )
        finally:
            ar.wired_limit = was
    batch_gen._quanteeze_worker = True
    return batch_gen


def _phase_ended(info):
    from . import phases
    entry = info.get(phases.KEY) if isinstance(info, dict) else None
    if entry is not None:
        phases.end(entry, 'cancelled')


def _own_cancellations(generator, batch_gen, active):
    with generator._cancel_lock:
        mine = generator._cancelled & set(active)
        generator._cancelled -= mine
    for uid in mine:
        if not batch_gen.remove(uid):
            generator._cancel(uid)
            continue
        info = active.pop(uid)
        _phase_ended(info)
        try:
            info['rqueue'].put(None)
        except Exception:
            pass


def _step(generator, batch_gen, active, drafter):
    from mlx_vlm.server import generation as G
    prompt_responses, responses = batch_gen.next()
    generator._log_prefill_progress(batch_gen, active)
    for prompt_response in prompt_responses:
        if prompt_response.uid in active:
            info = active[prompt_response.uid]
            info['prompt_tps'] = prompt_response.prompt_tps
            info['cached_tokens'] = getattr(prompt_response, 'cached_tokens', 0)
            generator._log_prefill_completed(prompt_response.uid, info, prompt_response)
    for r in responses or ():
        if r.uid not in active:
            continue
        info = active[r.uid]
        tok = r.token
        token_count = 0 if tok is None else 1
        if tok is None:
            text = info['streamer'].finalize()
            tok = 0
        elif hasattr(tok, 'item'):
            tok = tok.item()
            text = generator._stream_text(info, tok, r.finish_reason)
        else:
            text = generator._stream_text(info, tok, r.finish_reason)
        emitted_at = generator._log_decode_progress(
            r.uid, info, token=tok, text=text, finish_reason=r.finish_reason, token_count=token_count
        )
        draft_rounds = draft_accepted = draft_total = None
        kind = None
        if r.finish_reason is not None and info.get('spec_snapshot') is not None:
            draft_rounds, draft_accepted, draft_total = G.speculative_stats_since(drafter, info['spec_snapshot'])
            if draft_rounds is not None:
                kind = generator.draft_kind
        with _RUNNING_LOCK:
            if r.uid in _RUNNING:
                most, n = _RUNNING[r.uid]
                _RUNNING[r.uid] = (most, n + token_count)
        info['rqueue'].put(G.StreamingToken(
            text=text,
            token=tok,
            logprobs=r.token_logprob,
            finish_reason=r.finish_reason,
            peak_memory=mx.get_peak_memory() / 1e9 if r.finish_reason else 0,
            prompt_tps=info.get('prompt_tps'),
            draft_kind=kind,
            draft_rounds=draft_rounds,
            draft_n_accepted=draft_accepted,
            draft_n=draft_total,
            top_logprobs=getattr(r, 'top_logprobs', None),
            cached_tokens=info.get('cached_tokens', 0),
            token_count=token_count,
            emitted_at=emitted_at,
        ))
        if r.finish_reason is not None:
            info['rqueue'].put(None)
            del active[r.uid]


def _passes_done():
    """Synchronize the calling thread's stream and release any STM leases it held."""
    from . import demand
    try:
        mx.synchronize()
    except Exception as e:
        print(f'quanteeze: request threads: synchronizing stream at request end failed: {type(e).__name__}: {e}',
              file=sys.stderr, flush=True)
    for d in list(demand._LIVE):
        try:
            d.thread_done()
        except Exception as e:
            print(f'quanteeze: request threads: releasing thread passes failed: {type(e).__name__}: {e}',
                  file=sys.stderr, flush=True)


def passes_failed():
    """Undo any per-thread marks and holds left behind by a pass that raised."""
    from . import demand
    for d in list(demand._LIVE):
        try:
            d.pass_failed()
        except Exception as e:
            print(f'quanteeze: request threads: resetting failed pass state failed: {type(e).__name__}: {e}',
                  file=sys.stderr, flush=True)


def _draft_together():
    return (variables.get('QUANTEEZE_REQUEST_THREADS_DRAFT_TOGETHER', variables.at_load()) or 'off').lower() == 'on'


def _prompt_done(counted):
    if counted:
        with _ROUNDS:
            _PROMPTING[0] -= 1
            _ROUNDS.notify_all()
    _PROMPT.release()


def _begin_round(together):
    with _ROUNDS:
        if _PROMPTING[0] > 0 or (len(_RUNNING) > 1 and not together):
            return False
        _IN_ROUND[0] += 1
        return True


def _end_round():
    with _ROUNDS:
        _IN_ROUND[0] -= 1
        _ROUNDS.notify_all()


def _install_rounds():
    """Coordinate single-row DFlash rounds across concurrent requests."""
    from mlx_vlm.generate import ar
    cls = ar.SpeculativeGenerationBatch
    if cls.__dict__.get('_quanteeze_rounds'):
        return
    original = cls._start_rounds

    def _start_rounds(self):
        if (self._rounds_iter is not None or not getattr(_WORKER, 'on', False)
                or self.draft_kind != 'dflash' or len(self._all_uids) != 1):
            return original(self)

        def stop_check(token_id):
            return (self._finished[0] or self.stop_criteria(token_id)
                    or self._num_tokens[0] >= self.max_tokens[0])

        self._rounds_iter = _rounds(
            self.model,
            self.draft_model,
            self.prompt_cache,
            self.hidden,
            first_bonus=int(self.first_tokens.reshape(-1).item()),
            max_tokens=max(self.max_tokens),
            sampler=self.sampler,
            draft_block_size=self.draft_block_size,
            token_dtype=self.token_dtype,
            greedy_sampling=self.greedy_sampling,
            stop_check=stop_check,
        )

    cls._start_rounds = _start_rounds
    cls._quanteeze_rounds = True


def _rounds(model, draft_model, prompt_cache, hidden, *, first_bonus, max_tokens, sampler, draft_block_size,
            token_dtype, greedy_sampling, stop_check):
    from mlx_vlm.speculative import dflash as D
    from mlx_vlm.speculative.cache_state import abort_speculative_round, commit_speculative_round

    lm = model.language_model if hasattr(model, 'language_model') else model
    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = D._dflash_block_total(draft_model, draft_block_size)
    choose_block_ceiling = getattr(draft_model, 'choose_block_ceiling', None)
    if callable(choose_block_ceiling):
        block_total = choose_block_ceiling(int(hidden.shape[1]), block_total)
    initial_block_size = getattr(draft_model, 'dflash_initial_block_size', None)
    choose_initial_block_size = getattr(draft_model, 'choose_initial_block_size', None)
    if callable(choose_initial_block_size):
        initial_block_size = choose_initial_block_size(int(hidden.shape[1]), block_total)
    draft_cache = draft_model.reset(model)
    D._reserve_dflash_target_cache(prompt_cache, block_total)
    positioned = D._supports_positioned_target_sampling(sampler)
    sampler_rng = D._SpeculativeSamplerRNG(draft_model, enabled=not greedy_sampling and not positioned)
    prepare = getattr(draft_model, 'prepare_target_hidden', None)
    window = getattr(getattr(draft_model, 'cfg', None), 'sliding_window', None)
    together = _draft_together()

    b, emitted = first_bonus, 1
    while emitted < max_tokens:
        if not _begin_round(together):
            out = lm(mx.array([[b]], dtype=token_dtype), cache=prompt_cache, capture_layer_ids=target_layer_ids)
            step = mx.concatenate(out.hidden_states, axis=-1)
            hidden = mx.concatenate([hidden, step], axis=1)
            if window and hidden.shape[1] > window:
                hidden = hidden[:, -int(window):]
            if greedy_sampling:
                tok = sampler(out.logits)
            elif positioned:
                tok = D._sample_dflash_target_block(out.logits[:, -1:], sampler, row_ids=[0], base_positions=[emitted])
            else:
                tok = sampler(D._dflash_target_logprobs(out.logits[:, -1:])[:, 0, :])
                sampler_rng.target_sampled(sync_draft=True)
            mx.async_eval(tok, hidden)
            b = int(tok.reshape(-1)[-1].item())
            _count('plain_steps')
            yield [b], None
            emitted += 1
            if stop_check(b):
                return
            continue
        try:
            bs = D._dflash_next_block_size(draft_model, block_total, max_tokens - emitted + 1, initial_block_size)
            if bs <= 1:
                return
            drafted_hidden = prepare(hidden) if callable(prepare) else hidden
            draft_kwargs = {'target_hidden_prepared': True} if callable(prepare) else {}
            draft_sampler = (
                D._PositionedDraftSampler(sampler, row_ids=[0], positions=[emitted])
                if not greedy_sampling and positioned else sampler
            )
            draft_fn = (
                getattr(draft_model, 'draft_block_greedy', draft_model.draft_block)
                if greedy_sampling else draft_model.draft_block
            )
            draft_tokens = sampler_rng.draft_tokens(
                draft_fn, b, drafted_hidden, draft_cache, bs, draft_sampler, token_dtype, **draft_kwargs
            )
            mx.async_eval(draft_tokens)
            gdn_states = None
            try:
                verify_input = mx.concatenate([mx.array([[b]], dtype=token_dtype), draft_tokens], axis=1)
                if greedy_sampling:
                    captured, gdn_states, target_tokens = D._dflash_verify_greedy(
                        lm, verify_input, prompt_cache, target_layer_ids, sampler
                    )
                    hidden = mx.concatenate(captured, axis=-1)
                    mx.async_eval(target_tokens, hidden)
                    accepted, new_tokens = D._speculative_walk(draft_tokens, target_tokens, max_tokens - emitted)
                else:
                    verify_out, gdn_states = D._dflash_verify(lm, verify_input, prompt_cache, target_layer_ids)
                    hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
                    mx.async_eval(hidden)
                    accepted_list, new_tokens_list = D._sample_dflash_target_walk(
                        verify_out.logits, draft_tokens, sampler, [max_tokens - emitted],
                        row_ids=[0], base_positions=[emitted]
                    )
                    accepted, new_tokens = accepted_list[0], new_tokens_list[0]
                    sampler_rng.target_sampled(sync_draft=not positioned)
                    verify_out = None
                D._record_speculative_round(draft_model, accepted, bs - 1)
                if accepted < bs - 1:
                    hidden = hidden[:, :accepted + 1, :]
                b = new_tokens[-1] if new_tokens else b
                commit_speculative_round(lm, prompt_cache, gdn_states, accepted, bs)
            except BaseException:
                abort_speculative_round(gdn_states)
                raise
        finally:
            _end_round()
        _count('mtp_rounds')
        for tok in new_tokens:
            yield [tok], None
            emitted += 1
            if emitted >= max_tokens or stop_check(tok):
                return


def _own_request_state(generator):
    """Move shared per-request model module attributes into thread-local storage."""
    for root in (getattr(generator, 'model', None), getattr(generator, 'draft_model', None)):
        if root is None or not hasattr(root, 'named_modules'):
            continue
        for _, module in root.named_modules():
            per_thread.own_attributes(module, PER_REQUEST)


def _is_cpu(s):
    device = getattr(s, 'device', s)
    return getattr(device, 'type', device) == mx.cpu


def _no_cpu_blocks():
    """Avoid switching the process-wide default MLX stream to CPU while multiple threads run."""
    try:
        from mlx_vlm.models.deepseek_v41.engram import QuantizedEngramEmbedding
    except ImportError:
        QuantizedEngramEmbedding = None
    if QuantizedEngramEmbedding is not None and not QuantizedEngramEmbedding.__dict__.get('_quanteeze_cpu_ops'):
        def __call__(self, indices):
            cpu = mx.cpu
            rows = mx.take(self.weight, indices, axis=0, stream=cpu)
            scales = mx.take(self.scales, indices, axis=0, stream=cpu)
            biases = mx.take(self.biases, indices, axis=0, stream=cpu) if self.biases is not None else None
            mx.eval((rows, scales) if biases is None else (rows, scales, biases))
            return mx.dequantize(
                rows, scales=scales, biases=biases,
                group_size=self.group_size, bits=self.bits, mode=self.mode, dtype=mx.bfloat16,
            )

        QuantizedEngramEmbedding.__call__ = __call__
        QuantizedEngramEmbedding._quanteeze_cpu_ops = True
    if getattr(mx.stream, '_quanteeze_watch', False):
        return
    original, said = mx.stream, set()

    def stream(s):
        if _is_cpu(s) and running() > 1:
            frame = sys._getframe(1)
            where = (frame.f_code.co_filename, frame.f_lineno)
            if where not in said:
                said.add(where)
                _count('cpu_blocks')
                print(
                    f'quanteeze: request threads: mx.stream(cpu) at {where[0]}:{where[1]} while {running()} requests run',
                    file=sys.stderr,
                    flush=True,
                )
        return original(s)

    stream._quanteeze_watch = True
    mx.stream = stream


def _one_stream():
    """Ensure mlx-vlm speculative helpers run on each thread's default stream."""
    import importlib
    for name in ('common', 'dflash', 'mtp', 'eagle3'):
        try:
            module = importlib.import_module(f'mlx_vlm.speculative.{name}')
        except ImportError:
            continue
        if hasattr(module, 'generation_stream'):
            module.generation_stream = mx.default_device()


def _evaluated(generator):
    """Evaluate lazy model parameters on the loading thread before spawning workers."""
    from mlx.utils import tree_flatten
    arrays = []
    for module in (getattr(generator, 'model', None), getattr(generator, 'draft_model', None)):
        if module is None or not hasattr(module, 'parameters'):
            continue
        for name, a in tree_flatten(module.parameters()):
            if not isinstance(a, mx.array):
                continue
            if any(k in name.lower() for k in SKIP_EVALUATING) or a.nbytes > EVALUATE_AT_MOST:
                continue
            arrays.append(a)
    if arrays:
        mx.eval(arrays)
        mx.synchronize()


def _prompt_count(request):
    n = getattr(request, 'prompt_tokens', 0) or 0
    return int(n) if isinstance(n, (int, float)) else len(n)


def _caches_bytes(batch_gen):
    from .admission import _cache_bytes
    return _cache_bytes(getattr(batch_gen, '_prompt_batch', None)) + _cache_bytes(
        getattr(batch_gen, '_generation_batch', None)
    )


def _oldest(uid):
    with _RUNNING_LOCK:
        return bool(_RUNNING) and uid == min(_RUNNING)


@contextlib.contextmanager
def _wired_while_generating():
    """Keep Metal wired up to working set (or QUANTEEZE_WIRED_GB if higher) across concurrent request threads."""
    if not mx.metal.is_available():
        yield
        return
    with _WIRING_LOCK:
        if _WIRING['generating'] == 0:
            most = int(mx.device_info().get('max_recommended_working_set_size', 0))
            serve = sys.modules.get(__package__ + '.serve')
            limit = max(most, int(getattr(serve, '_WIRED', [0])[0] or 0))
            _WIRING['before'] = mx.set_wired_limit(limit) if limit > 0 else None
            STATS['wired'] += 1
        _WIRING['generating'] += 1
    try:
        yield
    finally:
        with _WIRING_LOCK:
            _WIRING['generating'] -= 1
            if _WIRING['generating'] == 0 and _WIRING['before'] is not None:
                mx.set_wired_limit(_WIRING['before'])
                _WIRING['before'] = None


def _serve(generator, request, uid, slots, own=True, lease=None):
    with lease if lease is not None else contextlib.nullcontext(), _wired_while_generating():
        _served(generator, request, uid, slots, own, lease)


def _served_ends(uid, batch_gen, active, prompting, counted):
    """Per-step cleanup for `_served`, isolating failures so every subsequent resource still releases."""
    from . import cuda_ple

    def removed():
        with _RUNNING_LOCK:
            for k in [k for k in _RUNNING if k == uid or k in active]:
                _RUNNING.pop(k, None)

    def unmarked():
        with _SERVING_LOCK:
            _SERVING.discard(threading.get_ident())

    steps = (
        ('prompt turn', (lambda: _prompt_done(counted)) if prompting else None),
        ('batch close', batch_gen.close if batch_gen is not None and callable(getattr(batch_gen, 'close', None)) else None),
        ('running marks', removed),
        ('passes', _passes_done),
        ('serving mark', unmarked),
        ('deferred frees', cuda_ple.free_later),
    )
    for what, step in steps:
        if step is None:
            continue
        try:
            step()
        except Exception as e:
            print(f'quanteeze: request {uid}: {what}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)


def _served(generator, request, uid, slots, own=True, lease=None):
    from mlx_vlm._stream_cleanup import clear_mlx_streams
    from mlx_vlm.server import generation as G

    rqueue, active, batch_gen, prompting, counted = request.rqueue, {}, None, False, False
    per_thread.PASS.__init__()
    per_thread.PASS.request_uid = uid
    _WORKER.on = True
    with _SERVING_LOCK:
        _SERVING.add(threading.get_ident())
    try:
        args = request.args
        log_state = generator._log_prefill_started(request, backend='request_threads')
        drafter = _drafter(generator)
        batch_gen = _batch_generator(generator, args, drafter)
        _PROMPT.acquire()
        prompting = True
        coordinator = getattr(batch_gen, 'apc', None)
        if coordinator is not None:
            coordinator.prepare_prefill(
                request.prompt_tokens, prefill_step_size=generator._effective_prefill_step_size()
            )
        input_ids, gen_kwargs = generator._gpu_embed(
            request.raw_inputs, request.images, apc_semantic_hash=request.apc_semantic_hash
        )
        has_embeds = bool(gen_kwargs.get('inputs_embeds') is not None)
        if request.apc_semantic_hash is None and getattr(args, 'tenant_id', None):
            gen_kwargs['_apc_tenant'] = args.tenant_id
        batch_gen.uid_count = uid
        (got,) = batch_gen.insert(
            [input_ids.squeeze(0).tolist()],
            max_tokens=args.max_tokens,
            prompt_kwargs=[gen_kwargs],
            logits_processors=[generator._make_logits_processors(args, input_ids)],
            thinking_budget_criteria=[getattr(request, 'thinking_budget_criteria', None)],
        )
        with _ROUNDS:
            _RUNNING[got] = (int(getattr(args, 'max_tokens', 0) or 0), 0)
            _PROMPTING[0] += 1
            counted = True
            _ROUNDS.wait_for(lambda: _IN_ROUND[0] == 0)
        rqueue.put(G.GenerationContext(uid=got, prompt_tokens=request.prompt_tokens))
        active[got] = {
            'rqueue': rqueue,
            'streamer': G._ServerTokenStreamer(generator.tokenizer, G.make_streaming_detokenizer(generator.processor)),
            'gen_kwargs': gen_kwargs if has_embeds else None,
            'prompt_tps': None,
            'cached_tokens': 0,
            'spec_snapshot': G.speculative_stats_snapshot(drafter) if drafter is not None else None,
            **log_state,
        }
        measuring, measured = leases.enabled() and threads() > 1, 0
        while active:
            _own_cancellations(generator, batch_gen, active)
            if not active:
                break
            if lease is not None and not prompting:
                with _RUNNING_LOCK:
                    made = _RUNNING.get(got, (0, 0))[1]
                lease.more(made, oldest=lambda: _oldest(got), cancelled=lambda: got in generator._cancelled)
            if measuring and running() == 1 and (prompting or 1 <= measured <= leases.MEASURED_STEPS):
                def _run_step():
                    with _decode_turn(prompting):
                        return _step(generator, batch_gen, active, drafter)

                leases.measure(
                    _run_step,
                    prompting,
                    min(_prompt_count(request), generator._effective_prefill_step_size() or (1 << 30)),
                    lease,
                    lambda: _caches_bytes(batch_gen),
                )
            else:
                with _decode_turn(prompting):
                    _step(generator, batch_gen, active, drafter)
            measured += not prompting
            if prompting and batch_gen._prompt_batch is None and not batch_gen._unprocessed_sequences:
                prompting = False
                _prompt_done(counted)
                if measuring:
                    caches = getattr(getattr(batch_gen, '_generation_batch', None), 'prompt_cache', None)
                    if caches is not None:
                        leases.note_kv(caches)
                if lease is not None:
                    lease.prompt_done()
                    lease.held(_caches_bytes(batch_gen))
    except Exception as e:
        print(f'quanteeze: a request thread failed: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
        for info in list(active.values()):
            _phase_ended(info)
        traceback.print_exc(file=sys.stderr)
        passes_failed()
        if e.__traceback__ is not None:
            traceback.clear_frames(e.__traceback__)
        try:
            rqueue.put(e)
        except Exception:
            pass
    finally:
        try:
            _served_ends(uid, batch_gen, active, prompting, counted)
        finally:
            with _SERVING_LOCK:
                _SERVING.discard(threading.get_ident())
            per_thread.PASS.request_uid = None
            _WORKER.on = False
            slots.release()
            if own:
                clear_mlx_streams()


def _dispatch(generator, n):
    """Serve requests from `generator.requests` on up to `n` concurrent worker threads."""
    from queue import Empty
    from mlx_vlm.server import generation as G
    try:
        generator._initialize_model()
    except Exception as e:
        generator._load_error = e
        generator._ready.set()
        return
    if G.is_diffusion_model(generator.model):
        generator._ready.set()
        generator._run_diffusion()
        return
    _own_request_state(generator)
    _no_cpu_blocks()
    _evaluated(generator)
    generator._ready.set()
    _one_stream()
    slots = threading.BoundedSemaphore(n)
    workers = []
    while True:
        workers = [w for w in workers if w.is_alive()]
        if generator._stop and not workers and generator.requests.empty():
            break
        try:
            request = generator.requests.get(timeout=0.1)
        except Empty:
            continue
        if request is None:
            continue
        lease = None
        try:
            while workers and (not together_ok() or drafts_alone(generator)):
                workers = [w for w in workers if w.is_alive()]
                if workers:
                    workers[0].join(0.05)
            held = False
            if n > 1 and not workers and not leases.seeded() and leases.enabled() and together_ok():
                leases.seed()
            uid = next(_UIDS)
            if leases.seeded():
                lease = leases.Lease(
                    uid, _prompt_count(request), getattr(request.args, 'max_tokens', 0),
                    generator._effective_prefill_step_size(),
                )
                while not lease.take(alone=not workers):
                    held = True
                    workers = [w for w in workers if w.is_alive()]
                    if workers:
                        workers[0].join(0.05)
            else:
                while workers and not _room_for(request, workers):
                    held = True
                    workers = [w for w in workers if w.is_alive()]
                    if workers:
                        workers[0].join(0.05)
            if held:
                _count('held_for_room')
            handed = False
            try:
                while not slots.acquire(timeout=0.1):
                    workers = [w for w in workers if w.is_alive()]
                workers = [w for w in workers if w.is_alive()]
                if not STATS['threads_started'] and not STATS['first_here']:
                    STATS['first_here'] = 1
                    handed = True
                    _serve(generator, request, uid, slots, own=False, **({} if lease is None else dict(lease=lease)))
                    continue
                STATS['threads_started'] += 1
                STATS['together' if workers else 'alone'] += 1
                try:
                    w = threading.Thread(
                        target=_serve,
                        args=(generator, request, uid, slots),
                        kwargs=dict(own=True, **({} if lease is None else dict(lease=lease))),
                        daemon=True,
                        name=f'quanteeze-request-{uid}',
                    )
                    w.start()
                except BaseException:
                    slots.release()
                    raise
                handed = True
            finally:
                if lease is not None and not handed:
                    lease.give()
            workers.append(w)
            STATS['most_at_once'] = max(STATS['most_at_once'], len(workers))
        except Exception as e:
            print(f'quanteeze: request threads: a request could not be dispatched: {type(e).__name__}: {e}',
                  file=sys.stderr, flush=True)
            if lease is not None:
                try:
                    lease.give()
                except Exception:
                    pass
            try:
                request.rqueue.put(e)
            except Exception:
                pass
    for w in workers:
        w.join()


def install(generator_cls):
    """Wrap `ResponseGenerator._run_impl` to dispatch across worker threads when enabled."""
    from . import prefill
    if prefill.TURNS is None:
        prefill.TURNS = _PromptTurns()
    if generator_cls.__dict__.get('_quanteeze_request_threads'):
        return
    original = generator_cls._run_impl

    def _run_impl(self):
        n = _load_settings()
        me = threading.current_thread()
        _LOOP_THREAD[0] = me
        try:
            if n <= 1:
                return original(self)
            return _dispatch(self, n)
        finally:
            if _LOOP_THREAD[0] is me:
                _LOOP_THREAD[0] = None
                _LOADED.update(threads=None, any=False)

    generator_cls._run_impl = _run_impl
    generator_cls._quanteeze_request_threads = True
    _install_rounds()
