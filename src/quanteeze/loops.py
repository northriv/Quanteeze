# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Detect and stop repetitive generation loops on the token stream (`quanteeze.loops`)."""
import copy
import sys

from . import stream, variables

MIN_PERIOD = 4
MAX_PERIOD = 256
STREAK = 20                     # repeats of a period before stopping


def repeated_period(ids, min_period=MIN_PERIOD, max_period=MAX_PERIOD, streak=STREAK):
    """If the tail of `ids` repeats a period `p` (`min_period <= p <= max_period`) at least `streak`
    times in a row (and the period is not a single repeated token when `p > 1`), return `p`; else 0."""
    n = len(ids)
    if streak < 2 or n < min_period * streak:
        return 0
    limit = min(max_period, n // streak)
    for p in range(min_period, limit + 1):
        span = ids[n - p:]
        if len(set(span)) <= 1:
            continue
        ok = True
        for k in range(2, streak + 1):
            if ids[n - k * p:n - (k - 1) * p] != span:
                ok = False
                break
        if ok:
            return p
    return 0


class LoopGuard:
    """Wraps a request's token iterator and stops generation if a repetition loop is detected."""

    def __init__(self, tokens, uid=None):
        self._tokens = tokens
        self._uid = uid
        self._ids = []
        self._stopped = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._stopped:
            raise StopIteration
        item = next(self._tokens)
        tid = getattr(item, 'token', None)
        if tid is not None:
            self._ids.append(int(tid))
            if len(self._ids) % 16 == 0:
                p = repeated_period(self._ids)
                if p:
                    self._stopped = True
                    print(f'quanteeze: request {self._uid}: stopped repetition loop (period {p} tokens)',
                          file=sys.stderr, flush=True)
                    item = copy.copy(item)
                    item.finish_reason = 'stop'
                    close = getattr(self._tokens, 'close', None)
                    if callable(close):
                        close()
        return item

    def __getattr__(self, name):
        return getattr(self._tokens, name)


def install(generator_cls):
    """Install repetition loop detection on `generator_cls` according to `QUANTEEZE_LOOPS`."""
    mode = (variables.get('QUANTEEZE_LOOPS') or 'on').lower()
    if mode == 'off':
        return
    stream.add(generator_cls, 'loops',
               lambda req, tokens: LoopGuard(tokens, uid=req.uid))
