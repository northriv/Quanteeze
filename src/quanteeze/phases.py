# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""What each request is doing now, and how the last ones ended, for the page (quanteeze.home's
live).  A chat client shows nothing while a long prompt is read,
so a client's retry and the end of a conversation looked the same.

install() wraps mlx-vlm's ResponseGenerator logs of a request's prefill and decode, which every
request passes whatever its backend.  A request is:
- reading: its prompt, `read` of `prompt` tokens (the prompt cache's `cached` among them once
  read), with the reasoning effort the template gets (`effort`; None: none named, the template's
  own);
- thinking, answering, or calling a tool, as its text goes: thinking where the prompt opened a
  <think> until the model closes it, then answering, and calling a tool from a <tool_call> on;
- then ended: 'answered', 'tool' (called a tool), 'length' (cut at its token limit),
  'thinking' (stopped before closing its thinking: no answer and no call, so a client such as
  Pydantic AI asks again), 'cancelled' (the client went away, or a stop), or 'stopped' (no
  request runs at the server any more, and this one never said it ended: an error).
"""
import threading
import time
from collections import deque

RECENT = 8                      # the requests that ended, kept for the page
KEY = 'quanteeze_phase'        # the entry's place in mlx-vlm's per-request log state
THINK, THINK_END, CALL = '<think>', '</think>', '<tool_call>'
_LOCK = threading.Lock()
_NOW = {}                       # request id -> its entry while it runs
_ENDED = deque(maxlen=RECENT)   # the newest last
_COUNTS = {'requests': 0, 'answered': 0, 'tool': 0, 'length': 0, 'thinking': 0, 'cancelled': 0, 'stopped': 0}
_INTERNAL = ('tail', 'read_at', 'first_at', 'last_at')


def _opens_thinking(generator, request):
    """Whether the request's prompt ends inside a <think> the template opened (its last tokens),
    or continues an unclosed thinking block (`reask`); where they cannot be read, whether its
    arguments leave thinking on."""
    args = getattr(request, 'args', None)
    if getattr(args, '_quanteeze_in_thinking', False):
        return True
    try:
        import numpy as np
        ids = np.array(request.raw_inputs['input_ids']).reshape(-1)[-4:]
        tokenizer = getattr(generator.tokenizer, 'tokenizer', generator.tokenizer)
        tail = tokenizer.decode([int(i) for i in ids])
        return THINK in tail and THINK_END not in tail.rsplit(THINK, 1)[-1]
    except Exception:                                  # noqa: BLE001 -- a guess is enough for the page
        return getattr(args, 'enable_thinking', None) is not False and getattr(args, 'reasoning_effort', None) != 'none'


def begin(request_id, prompt, thinking, effort=None):
    """A request's entry, kept while it runs."""
    entry = {'id': str(request_id), 'started': time.time(), 'prompt': int(prompt), 'read': 0, 'cached': None,
             'phase': 'reading', 'thinking': bool(thinking), 'effort': effort, 'thought': 0, 'written': 0, 'tail': '',
             'read_at': None, 'first_at': None, 'last_at': None}
    with _LOCK:
        _NOW[entry['id']] = entry
    return entry


def read(entry, processed):
    if entry['phase'] == 'reading':
        entry['read'] = max(entry['read'], int(processed))


def prompt_read(entry, cached):
    entry.update(read=entry['prompt'], cached=int(cached), read_at=time.time(),
                 phase='thinking' if entry['thinking'] else 'answering')


def wrote(entry, text, tokens, finish_reason):
    """`tokens` more of the request's tokens, their `text`; its end where `finish_reason` is set."""
    if entry['phase'] == 'reading':                    # a backend that reads without saying it has read
        prompt_read(entry, entry['cached'] or 0)
    n = int(tokens)
    if n > 0:
        now = time.time()
        if entry.get('first_at') is None:
            entry['first_at'] = now
        entry['last_at'] = now
    entry['thought' if entry['phase'] == 'thinking' else 'written'] += n
    tail = entry['tail'] + (text or '')
    if entry['phase'] == 'thinking' and THINK_END in tail:
        entry['phase'], tail = 'answering', tail.split(THINK_END, 1)[1]
    if entry['phase'] in ('thinking', 'answering') and CALL in tail:
        entry['phase'] = 'tool'
    entry['tail'] = tail[-len(CALL) - len(THINK_END):]
    if finish_reason is not None:
        end(entry, finish_reason)


def _rates(e, now):
    """Per-request timing & throughput fields (`ttft_s`, `decode_tps`, `prefill_tps`)."""
    first_at, last_at, read_at = e.get('first_at'), e.get('last_at'), e.get('read_at')
    ttft_s = round(first_at - e['started'], 2) if first_at is not None else None
    gen = e['thought'] + e['written']
    decode_tps = None
    if gen >= 2 and first_at is not None and last_at is not None and last_at > first_at:
        decode_tps = round((gen - 1) / (last_at - first_at), 1)
    elif gen >= 1 and read_at is not None and last_at is not None and last_at > read_at:
        decode_tps = round(gen / (last_at - read_at), 1)
    prefill_tps = None
    if read_at is not None and read_at > e['started']:
        fresh = max(0, e['prompt'] - (e['cached'] or 0))
        if fresh > 0:
            prefill_tps = round(fresh / (read_at - e['started']), 0)
    elif e.get('phase') == 'reading' and e['read'] > 0 and now - e['started'] >= 0.2:
        prefill_tps = round(e['read'] / (now - e['started']), 0)
    return {'ttft_s': ttft_s, 'decode_tps': decode_tps, 'prefill_tps': prefill_tps}


def end(entry, finish_reason):
    how = ('cancelled' if finish_reason == 'cancelled' else 'stopped' if finish_reason == 'stopped'
           else 'length' if finish_reason == 'length' else 'thinking' if entry['phase'] == 'thinking'
           else 'tool' if entry['phase'] == 'tool' or finish_reason == 'tool_calls' else 'answered')
    with _LOCK:
        if _NOW.pop(entry['id'], None) is None:
            return
        _COUNTS['requests'] += 1
        _COUNTS[how] = _COUNTS.get(how, 0) + 1
        _ENDED.append({**{k: v for k, v in entry.items() if k not in ('tail', 'phase')}, 'ended': time.time(),
                       'how': how})


def state(running=None):
    """{'now': the requests running, oldest first, each with its seconds so far; 'ended': the last
    that ended, newest first; 'counts': cumulative request totals}.  running: how many requests the
    server counts as running (quanteeze.control's); where none, an entry left is ended 'stopped'."""
    if running == 0:
        with _LOCK:
            left = list(_NOW.values())
        for entry in left:
            end(entry, 'stopped')
    now = time.time()
    with _LOCK:
        running_now = [{**{k: v for k, v in e.items() if k not in _INTERNAL}, **_rates(e, now),
                        'seconds': round(now - e['started'], 1)}
                       for e in sorted(_NOW.values(), key=lambda e: e['started'])]
        ended = [{**{k: v for k, v in e.items() if k not in _INTERNAL}, **_rates(e, e['ended']),
                  'seconds': round(e['ended'] - e['started'], 1)}
                 for e in reversed(_ENDED)]
        counts = dict(_COUNTS)
    return {'now': running_now, 'ended': ended, 'counts': counts}


def install(generator_cls):
    """Wrap `generator_cls`'s (mlx-vlm's ResponseGenerator) prefill and decode logs."""
    if getattr(generator_cls, '_quanteeze_phases', False):
        return
    started, progress = generator_cls._log_prefill_started, generator_cls._log_prefill_progress
    completed, decoded = generator_cls._log_prefill_completed, generator_cls._log_decode_progress
    cancellations = generator_cls._apply_cancellations

    def _log_prefill_started(self, request, *, backend):
        info = started(self, request, backend=backend)
        info[KEY] = begin(info.get('request_id'), request.prompt_tokens or 0, _opens_thinking(self, request),
                          getattr(getattr(request, 'args', None), 'reasoning_effort', None))
        return info

    def _log_prefill_progress(self, batch_gen, active):
        progress(self, batch_gen, active)
        for info in list(active.values()):
            if KEY in info:
                read(info[KEY], info.get('prefill_processed') or 0)

    def _log_prefill_completed(uid, info, prompt_response):
        completed(uid, info, prompt_response)
        if KEY in info:
            prompt_read(info[KEY], getattr(prompt_response, 'cached_tokens', 0) or 0)

    def _log_decode_progress(uid, info, *, token, text, finish_reason, token_count=1):
        out = decoded(uid, info, token=token, text=text, finish_reason=finish_reason, token_count=token_count)
        if KEY in info:
            wrote(info[KEY], text, token_count, finish_reason)
        return out

    def _apply_cancellations(self, batch_gen, active):
        before = dict(active)
        cancellations(self, batch_gen, active)
        for uid, info in before.items():
            if uid not in active and KEY in info:
                end(info[KEY], 'cancelled')

    generator_cls._log_prefill_started = _log_prefill_started
    generator_cls._log_prefill_progress = _log_prefill_progress
    generator_cls._log_prefill_completed = staticmethod(_log_prefill_completed)
    generator_cls._log_decode_progress = staticmethod(_log_decode_progress)
    generator_cls._apply_cancellations = _apply_cancellations
    generator_cls._quanteeze_phases = True
