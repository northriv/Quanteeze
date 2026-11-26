# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""What a model's load leaves at module level, released when the model is let go.

A module that keeps state derived from the served model -- cached module maps, native handles,
or per-load decisions -- registers a cleanup handler (`on_let_go`), and `serve._let_go()` calls
every registered handler (`let_go()`) at page unload and before loading the next model.
"""
import sys

_LET_GO = []        # [(qualname, fn)], in registration order


def on_let_go(fn):
    """Register `fn()` to run when `serve._let_go()` releases a model; idempotent by name."""
    name = f'{fn.__module__}.{fn.__qualname__}'
    _LET_GO[:] = [(n, f) for n, f in _LET_GO if n != name] + [(name, fn)]
    return fn


def let_go():
    """Call every registered handler in order, isolating exceptions so remaining cleanups still run."""
    for name, fn in list(_LET_GO):
        try:
            fn()
        except Exception as error:                           # noqa: BLE001 -- next model still loads
            print(f'quanteeze: letting go of {name} failed: {type(error).__name__}: {error}', file=sys.stderr,
                  flush=True)


def registered():
    """Return registered cleanup handler names in registration order."""
    return [n for n, _ in _LET_GO]
