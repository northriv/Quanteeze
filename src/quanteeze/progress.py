# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""How far the server is in loading a model, for the control page's percent (quanteeze.control).

A load goes through phases -- the checkpoint opened (mlx-vlm's lazy load), its weights read
(the arrays evaluated, counted in bytes), the experts lowered to the target bit-width (counted in MoE
layers) -- each with a total and how much is done; percent() weighs them by WEIGHTS, so the
bar moves on through a load whose phases take very different times.  Written from the loading
thread, read from any: each update replaces a value whole.
"""
import threading
import time

WEIGHTS = {'opening': 0.05, 'weights': 0.55, 'experts': 0.35, 'starting': 0.05}
ORDER = tuple(WEIGHTS)

_STATE = {'model': None, 'phase': None, 'done': 0.0, 'total': 0.0, 'unit': '', 'started': None, 'phases': {},
          'error': None, 'finished': None}
_LOCK = threading.Lock()
_COUNTED = set()                # ids of the arrays this load's evaluate() has read


def begin(model):
    """A load of `model` starts."""
    _COUNTED.clear()
    with _LOCK:
        _STATE.update(model=str(model), phase=None, done=0.0, total=0.0, unit='', started=time.time(), phases={},
                      error=None, finished=None)


def phase(name, total=1.0, unit=''):
    """A phase starts, of `total` units (bytes, layers); the ones before it are done."""
    with _LOCK:
        for earlier in ORDER[:ORDER.index(name)] if name in ORDER else ():
            _STATE['phases'].setdefault(earlier, 1.0)
        _STATE.update(phase=name, done=0.0, total=float(total) or 1.0, unit=unit)
        _STATE['phases'][name] = 0.0


def advance(amount=1.0):
    """`amount` more of the phase done."""
    with _LOCK:
        _STATE['done'] = min(_STATE['total'], _STATE['done'] + amount)
        if _STATE['phase'] is not None:
            _STATE['phases'][_STATE['phase']] = _STATE['done'] / _STATE['total']


def end(error=None):
    """The load is over: done, or failed with `error`."""
    _COUNTED.clear()
    with _LOCK:
        if error is None:
            _STATE['phases'] = {name: 1.0 for name in ORDER}
        _STATE.update(error=None if error is None else str(error), finished=time.time())


def percent():
    """0 to 100: the phases' shares done, by WEIGHTS; a phase a load skips counts done once a later
    one starts."""
    with _LOCK:
        phases = dict(_STATE['phases'])
    return round(100 * sum(WEIGHTS[n] * min(1.0, phases.get(n, 0.0)) for n in ORDER), 1)


def state():
    """What the page shows: the model, the phase, how much of it, the percent, the seconds so far."""
    with _LOCK:
        out = {k: v for k, v in _STATE.items() if k != 'phases'}
    out['percent'] = percent()
    out['seconds'] = round((out['finished'] or time.time()) - out['started'], 1) if out['started'] else None
    out['loading'] = out['started'] is not None and out['finished'] is None
    return out


def evaluate(arrays, chunk=2 ** 30):
    """mx.eval `arrays` a chunk of about `chunk` bytes at a time, the 'weights' phase advanced by
    their bytes as they are read -- the same arrays evaluated as at once.  Arrays this load has
    evaluated here already are left out (the rest of a model is read before its experts are
    weighed against the room left), and a second call goes on with the phase."""
    import mlx.core as mx
    arrays = [a for a in arrays if id(a) not in _COUNTED]
    _COUNTED.update(id(a) for a in arrays)
    size_all = sum(a.nbytes for a in arrays)
    if _STATE['phase'] == 'weights':
        with _LOCK:
            _STATE['total'] += size_all
    else:
        phase('weights', size_all, 'bytes')
    batch, size = [], 0
    for a in arrays:
        batch.append(a)
        size += a.nbytes
        if size >= chunk:
            mx.eval(batch)
            advance(size)
            batch, size = [], 0
    if batch:
        mx.eval(batch)
        advance(size)
