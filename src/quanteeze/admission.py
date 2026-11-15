# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""How many waiting prompts the server takes into one prefill: as many as the memory holds.

mlx-vlm's BatchGenerator takes up to prefill_batch_size (8) waiting prompts into one prefill, and
up to completion_batch_size (32) rows into decoding, whatever the memory.  A model served on
demand (quanteeze.demand) reads nearly every expert of a layer for a prompt's pass however few
its tokens -- DeepSeek V4.1's 1,700 new tokens read some 250 GB -- so prompts
taken in together share one read, and the more the better; but a pass's arrays grow with its
tokens (V4.1's sparse attention gathers 512 compressed positions of 512 values for each, some
1 MiB a token at its peak).  So the number is the memory's (
within the budget, and as many as speed benefits from):

- the limit is the GPU's working set (guards.recommended_budget) times QUANTEEZE_ADMIT_SHARE
  (0.9: V4.1 on demand slowed tenfold with MLX at 78.5 GB of 83.5);
- a prefill of r rows takes about fixed + r x step x per_token: `fixed`, what a pass holds
  whatever its tokens (on demand, the experts of the layer it computes and of the next, read for
  it), and per_token, measured on the passes so far (the most seen: what MLX's peak rose by, less
  `fixed`, over the step's tokens); `step` the prefill step a row goes in;
- the rows taken are as many as fit beside what MLX holds now, at most mlx-vlm's own
  prefill_batch_size; none while others decode and none fit, one when nothing else runs.
  Until a pass has been measured, as many of the first come as one prefill step holds (their
  prompts, each at most a step, within a step's tokens): a pass no larger than one prompt's --
  one at a time for long prompts, and short ones together, which a prefill of fewer than
  MIN_TOKENS never measures (Flash-Next's 200-token prompts went one at a time).

What the rows decoding will take is kept free too (the caches' growth, which a model of large
KV caches -- GLM-5.3's -- runs out of memory on when long requests decode together): each row
decoding, and each row taken in, is reserved kv_token x the tokens it may still generate, at most
QUANTEEZE_ADMIT_AHEAD (4096; off: nothing reserved).  kv_token is measured on the decoding
itself: what the rows' caches hold grew by (their arrays' bytes, _cache_bytes), over DECODE_SPAN
tokens or more decoded by the same rows, the most seen; until then nothing is reserved.  Not what
MLX holds overall: on-demand reads while rows decode would look like cache growth, and with
one row decoding nothing more would be let in.

Taking fewer than are waiting, it takes them in the order they came: mlx-vlm keeps them shortest
first, and a long prompt was passed over by the shorter ones arriving after it until mlx-vlm gave
it up, 600 s with no token (DeepSeek V4.1 on demand, 5,901-token prompt
among five tasks).

A model in memory has the same eight as before wherever the memory holds them.  install() wraps
BatchGenerator._next: the rows, set as prefill_batch_size before it looks, and the measuring, of
the prompt step it takes (the prompt batch's columns processed, times its rows) -- whatever
prompt_step the batch has (quanteeze.prefill widens and batches it after this is installed).
"""
import mlx.core as mx
import mlx.nn as nn

from . import guards, parallel, variables

MIN_TOKENS = 256        # a step of fewer tokens is not measured: its cost is mostly not theirs
DECODE_SPAN = 1024      # tokens the same rows decode before their caches are read for their growth


class Admission:
    """The rows a prefill may take: within `limit` bytes of what MLX holds, a pass costing `fixed`
    besides its tokens; at most `most` rows."""

    def __init__(self, limit, fixed=0, most=8, ahead=0):
        self.limit, self.fixed, self.most, self.ahead = limit, fixed, most, ahead
        self.per_token = self.kv_token = None
        self.stats = {'measured': 0, 'admitted': 0, 'held_back': 0, 'rows_most': 0, 'decode_measured': 0}

    @classmethod
    def from_env(cls, fixed=0, most=8):
        """The limit from the working set and QUANTEEZE_ADMIT_SHARE; None where no working set is
        known or the share is 'off'."""
        share = variables.get('QUANTEEZE_ADMIT_SHARE')
        working = guards.recommended_budget()
        if not working or str(share).lower() == 'off':
            return None
        ahead = variables.get('QUANTEEZE_ADMIT_AHEAD')
        return cls(working * float(share), fixed, most, 0 if str(ahead).lower() == 'off' else int(ahead))

    def reserved(self, remaining):
        """Bytes kept for the rows decoding to grow their caches: `remaining`, the tokens each may
        still generate, each at most `ahead`."""
        if not self.kv_token or not self.ahead:
            return 0.0
        return self.kv_token * sum(min(max(0, int(r)), self.ahead) for r in remaining)

    def rows(self, pending, free_rows, step, busy, held=None, remaining=(), lengths=None):
        """How many of `pending` waiting prompts to take: within `free_rows` places to decode, each
        prefilled `step` tokens at a time and then decoding; `busy`, whether others decode
        meanwhile, `remaining` the tokens each of them may still generate; `held`, what MLX holds
        now (bytes); `lengths`, the waiting prompts' tokens, the first come first."""
        if pending <= 0 or free_rows <= 0:
            return 0
        most = min(self.most, pending, free_rows)
        if self.per_token is None:
            n, tokens = 0, 0
            for length in (lengths or [step])[:most]:
                tokens += min(int(length), step)
                if n and tokens > step:
                    break
                n += 1
            return max(1, n)
        held = mx.get_active_memory() if held is None else held
        room = self.limit - held - self.fixed - self.reserved(remaining)
        fit = int(room // max(1.0, self.per_token * step + self.reserved([self.ahead])))
        if fit >= 1:
            return min(most, fit)
        return 0 if busy else 1

    def measured(self, before, peak, tokens):
        """A pass of `tokens` tokens that took MLX from `before` to `peak` bytes."""
        if tokens < MIN_TOKENS:
            return
        cost = max(0.0, (peak - before - self.fixed) / tokens)
        self.per_token = cost if self.per_token is None else max(self.per_token, cost)
        self.stats['measured'] += 1

    def decoded(self, before, after, tokens):
        """The same rows decoded `tokens` tokens (all rows together) while what MLX holds went from
        `before` to `after` bytes: their caches' growth a token, the most seen."""
        if tokens < DECODE_SPAN:
            return
        grown = max(0.0, (after - before) / tokens)
        self.kv_token = grown if self.kv_token is None else max(self.kv_token, grown)
        self.stats['decode_measured'] += 1

    def summary(self):
        return dict(self.stats, limit_gb=self.limit / 1e9, fixed_gb=self.fixed / 1e9, ahead=self.ahead,
                    per_token_kb=None if self.per_token is None else self.per_token / 1e3,
                    kv_token_kb=None if self.kv_token is None else self.kv_token / 1e3)


def _columns(batch):
    return None if batch is None else (id(batch), getattr(batch, '_processed_prompt_columns', 0), len(batch.uids))


def _remaining(batch):
    """The tokens each row of a generation batch may still generate (mlx-vlm's GenerationBatch:
    max_tokens and _num_tokens a row)."""
    most, done = getattr(batch, 'max_tokens', None) or [], getattr(batch, '_num_tokens', None) or []
    return [m - n for m, n in zip(most, done)]


def _cache_bytes(batch):
    """The bytes of the arrays a generation batch's caches hold (mlx-vlm's GenerationBatch:
    prompt_cache, a cache a layer, whatever their kind: each array once)."""
    seen, total, todo = set(), 0, [(getattr(batch, 'prompt_cache', None), 0)]
    while todo:
        obj, depth = todo.pop()
        if obj is None or id(obj) in seen or depth > 8 or isinstance(obj, nn.Module):    # a cache's, not a model's
            continue
        seen.add(id(obj))
        if isinstance(obj, mx.array):
            total += obj.nbytes
        elif isinstance(obj, dict):
            todo.extend((v, depth + 1) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            todo.extend((v, depth + 1) for v in obj)
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            todo.extend((v, depth + 1) for v in vars(obj).values())
    return total


def _decoding(batch):
    """(the rows, the tokens they decoded) of a generation batch, or None."""
    uids, done = getattr(batch, 'uids', None), getattr(batch, '_num_tokens', None)
    if not uids or done is None:
        return None
    return tuple(uids), sum(done)


def install(generator_cls, admission):
    """Have `generator_cls` (mlx-vlm's BatchGenerator) take as many waiting prompts as `admission`
    lets it, and measure the prompt steps it takes and the caches' growth as it decodes."""
    step_next = generator_cls._next

    def _next(self, **kwargs):
        own = self.__dict__.setdefault('_quanteeze_rows', self.prefill_batch_size)
        pending = len(self._unprocessed_sequences)
        now = _decoding(self._generation_batch) if self._prompt_batch is None else None
        mark = self.__dict__.get('_quanteeze_decoded')
        if now is None or mark is None or mark[0] != now[0]:            # other rows, or a prompt between
            self.__dict__['_quanteeze_decoded'] = (None if now is None else
                                                    (now[0], now[1], _cache_bytes(self._generation_batch)))
        elif now[1] - mark[1] >= DECODE_SPAN:
            held = _cache_bytes(self._generation_batch)
            admission.decoded(mark[2], held, now[1] - mark[1])
            self.__dict__['_quanteeze_decoded'] = (now[0], now[1], held)
        if pending and self._prompt_batch is None:
            free = self.completion_batch_size - len(self._generation_batch)
            admission.most = own
            first = sorted(self._unprocessed_sequences, key=lambda sequence: sequence[0])
            rows = admission.rows(pending, free, self.prefill_step_size or 2048, len(self._generation_batch) > 0,
                                  remaining=_remaining(self._generation_batch), lengths=[len(q[1]) for q in first])
            rows = parallel.least(rows)             # over several GPUs, what every one's memory holds
            if rows:
                admission.stats['admitted'] += rows
                admission.stats['rows_most'] = max(admission.stats['rows_most'], rows)
                if rows < pending:                   # the first come first, not mlx-vlm's shortest
                    self._unprocessed_sequences.sort(key=lambda sequence: sequence[0])
            else:
                admission.stats['held_back'] += 1
            # mlx-vlm takes min(prefill_batch_size, pending) when the places free are at least
            # prefill_batch_size: more than the places free takes none
            self.prefill_batch_size = rows if rows else self.completion_batch_size + 1
        was, held = _columns(self._prompt_batch), mx.get_active_memory()
        mx.reset_peak_memory()
        try:
            return step_next(self, **kwargs)
        finally:
            self.prefill_batch_size = own
            now = _columns(self._prompt_batch)
            if now is not None:                      # a prompt step taken: its columns, all its rows
                tokens = (now[1] - (was[1] if was is not None and was[0] == now[0] else 0)) * now[2]
                if tokens > 0:
                    mx.synchronize()
                    admission.measured(held, mx.get_peak_memory(), tokens)

    generator_cls._next = _next
    return admission
