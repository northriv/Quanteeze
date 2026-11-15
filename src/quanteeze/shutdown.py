# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Stopping the server while it generates (neither SIGTERM nor SIGINT stopped
a server whose generation had gone round and would not end).

uvicorn stops on the first signal by waiting, without a limit, for the responses in flight to
end -- a streamed generation goes on to its max tokens (32,768) -- and a second SIGTERM only asks
the same again (a second SIGINT forces).  Here the first signal also cancels every generation in
flight, through the server's own cancellation (ResponseGenerator._cancel, as a client that goes
away does): each request's tokens end between two steps, the responses close, and the server
shuts down, its atexit work done (the summaries printed).  A step
under way finishes first -- a prompt's can take a minute -- so if the process is still there
GRACE seconds after the signal, the atexit work is run from here and the process exits.

install(generator_cls) keeps the uid of every request generating (the stream stage 'live');
patch(server_cls) has uvicorn's Server.handle_exit cancel them first.
"""
import atexit
import os
import sys
import threading

GRACE = 60.0                      # seconds after a signal before the process is made to exit
LIVE = {}                         # uid: the generator generating it
_STATE = {'signalled': False}


class _Live:
    """A request's token iterator that keeps its uid in LIVE until it ends."""

    def __init__(self, tokens, uid):
        self._tokens, self._uid = tokens, uid

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._tokens)
        except BaseException:
            LIVE.pop(self._uid, None)
            raise

    def close(self):
        LIVE.pop(self._uid, None)
        return self._tokens.close()

    def __getattr__(self, name):            # anything else the server asks of the iterator
        return getattr(self._tokens, name)


def install(generator_cls):
    """Keep the uid of every request `generator_cls` (mlx-vlm's ResponseGenerator) generates."""
    from . import stream

    def wrap(request, tokens):
        if request.uid is None:
            return tokens
        LIVE[request.uid] = request.generator
        return _Live(tokens, request.uid)

    stream.add(generator_cls, 'live', wrap)


def cancel_all():
    """Cancel every generation in flight; returns how many."""
    n = 0
    for uid, generator in list(LIVE.items()):
        cancel = getattr(generator, '_cancel', None)
        if cancel is not None:
            cancel(uid)
            n += 1
    return n


def stop(sig, grace=GRACE, exit=os._exit):
    """The first signal: every generation cancelled, and the process made to exit `grace` seconds
    later if it is still there."""
    if _STATE['signalled']:
        return
    _STATE['signalled'] = True
    n = cancel_all()
    print(f'quanteeze: stopping: {n} generation(s) cancelled; exiting within {grace:g} s',
          file=sys.stderr, flush=True)

    def later():
        print(f'quanteeze: still running {grace:g} s after the signal: exiting',
              file=sys.stderr, flush=True)
        try:
            atexit._run_exitfuncs()
        finally:
            exit(128 + int(sig))
    timer = threading.Timer(grace, later)
    timer.daemon = True
    timer.start()
    return timer


def patch(server_cls, grace=GRACE):
    """uvicorn's Server (server_cls) cancels the generations in flight on the first signal."""
    handle = server_cls.handle_exit

    def handle_exit(self, sig, frame):
        stop(sig, grace)
        return handle(self, sig, frame)

    server_cls.handle_exit = handle_exit
