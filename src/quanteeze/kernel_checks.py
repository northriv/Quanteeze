# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Shared per-shape kernel equivalence checks and atomic check tables (`quanteeze.kernel_checks`).

Compares the first `CHECKS` invocations of each kernel shape bit-for-bit against its Python/MLX
reference (`same_bits`), then commits to the kernel or sticks permanently to the reference if any
check mismatched. `tables()` returns `(CHOICE, CHECKED)` dict-like views backed by C++ atomic words
(`vendor/checks/checks.cpp`) when native extensions are available, or plain dicts under
`QUANTEEZE_NATIVE=off` (where `request_threads.together_ok()` serializes requests).
"""
import sys

import mlx.core as mx

from . import native_ext, per_thread

_UNSIGNED = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}
_EXT = {}
_CODES = {False: 0, True: 1}
_VALUES = {0: False, 1: True}
_NONE = object()


def _ext():
    """Return `native_ext`'s `_checks` module unless QUANTEEZE_NATIVE is off or build fails."""
    key = native_ext.mode()
    if key not in _EXT:
        _EXT[key] = (
            native_ext.extension('checks', with_mlx=False, with_stm=False, quiet=(key == 'auto'))
            if native_ext.allowed(True)
            else None
        )
    return _EXT[key]


def shared():
    """Whether check tables use C++ atomics so concurrent request threads can verify safely."""
    return _ext() is not None


def _code(value):
    code = _CODES.get(value)
    if code is None:
        code = _ext().code(repr(value))
        if code < 0:
            raise ValueError(f'kernel_checks: too many choices to number ({value!r})')
        _VALUES[code] = value
        _CODES[value] = code
    return code


SHAPES_MOST = 4096

# Word layout bits matching vendor/checks/checks.cpp (validated when loaded).
_HAS, _COUNT, _SHIFT = 1 << 61, (1 << 32) - 1, 32
_CODE = ((1 << 29) - 1) << _SHIFT


class _Table:
    """Per-module shape table holding C++ atomic cells and fallback plain dicts."""

    __slots__ = ('name', 'cells', 'plain', 'said', 'reference')

    def __init__(self, name, reference=False):
        self.name, self.cells, self.plain, self.said, self.reference = name, {}, ({}, {}), False, reference

    def full(self, key):
        """Return True once the table already holds `SHAPES_MOST` distinct shapes (warned once)."""
        if key in self.cells or key in self.plain[0] or key in self.plain[1]:
            return False
        if max(len(self.cells), len(self.plain[0]), len(self.plain[1])) < SHAPES_MOST:
            return False
        if not self.said:
            self.said = True
            print(
                f'quanteeze: kernel_checks: {self.name} holds {SHAPES_MOST} shapes; new shapes use the reference '
                '(check if key grows dynamically)',
                file=sys.stderr,
                flush=True,
            )
        return True

    def cell(self, key):
        """Return the C++ `Cell` for `key`, creating it if absent, or None without native checks."""
        cell = self.cells.get(key)
        if cell is None:
            ext = _ext()
            if ext is None:
                return None
            if (ext.HAS, ext.COUNT, ext.CODE, ext.CODE_SHIFT) != (_HAS, _COUNT, _CODE, _SHIFT):
                raise RuntimeError("kernel_checks: C++ word layout does not match Python constants")
            cell = self.cells[key] = ext.cell(f'{self.name}|{key!r}')
        return cell


class _View:
    """Dict-like view over a table's choices (part=0) or check counts (part=1)."""

    __slots__ = ('table',)
    part = 0

    def __init__(self, table):
        self.table = table

    def __getitem__(self, key):
        got = self.get(key, _NONE)
        if got is _NONE:
            raise KeyError(key)
        return got

    def __contains__(self, key):
        return self.get(key, _NONE) is not _NONE

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        cell = self.table.cells.get(key)
        if cell is None:
            del self.table.plain[self.part][key]
        else:
            cell.put(_COUNT if self.part else _HAS | _CODE, 0)

    def items(self):
        got = [(key, self.get(key, _NONE)) for key in list(self.table.cells) + list(self.table.plain[self.part])]
        return [(key, value) for key, value in got if value is not _NONE]

    def keys(self):
        return [key for key, _ in self.items()]

    def values(self):
        return [value for _, value in self.items()]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.items())

    def __eq__(self, other):
        return dict(self.items()) == (dict(other.items()) if isinstance(other, _View) else other)

    __hash__ = None

    def pop(self, key, *default):
        got = self.get(key, _NONE)
        if got is _NONE:
            if default:
                return default[0]
            raise KeyError(key)
        del self[key]
        return got

    def clear(self):
        for key in self.keys():
            del self[key]

    def __repr__(self):
        return f'kernel_checks.{("CHOICE", "CHECKED")[self.part]}({self.table.name}: {dict(self.items())!r})'


class _Choices(_View):
    __slots__ = ()

    def get(self, key, default=None):
        cell = self.table.cells.get(key)
        if cell is None:
            return self.table.plain[0].get(key, default)
        word = cell.load()
        return _VALUES[(word & _CODE) >> _SHIFT] if word & _HAS else default

    def __setitem__(self, key, value):
        cell = self.table.cell(key)
        if cell is None:
            self.table.plain[0][key] = value
        else:
            cell.put(_HAS | _CODE, _HAS | (_code(value) << _SHIFT))

    def count(self, key, found, checks, reference=False):
        """Record one check for `key` via atomic CAS and return the resulting choice."""
        if self.table.full(key):
            return reference
        cell = self.table.cell(key)
        if cell is None:
            return per_thread.counted(*self.table.plain, key, found, checks, reference)
        word = cell.count(_code(found) if found else 0, _code(reference), int(checks), not found)
        return _VALUES[(word & _CODE) >> _SHIFT]


class _Counts(_View):
    __slots__ = ()
    part = 1

    def get(self, key, default=None):
        cell = self.table.cells.get(key)
        if cell is None:
            return self.table.plain[1].get(key, default)
        seen = cell.load() & _COUNT
        return seen if seen else default

    def __setitem__(self, key, value):
        cell = self.table.cell(key)
        if cell is None:
            self.table.plain[1][key] = value
        else:
            cell.put(_COUNT, int(value))


def tables(name, reference=False):
    """Return `(CHOICE, CHECKED)` views backed by one atomic check table named `name`."""
    table = _Table(name, reference)
    return _Choices(table), _Counts(table)


def read(choices, counts, key):
    """Read `(choice, seen)` for `key` from a single atomic load when backed by `_Table`."""
    table = getattr(choices, 'table', None)
    if isinstance(choices, _Choices) and isinstance(counts, _Counts) and counts.table is table:
        cell = table.cells.get(key)
        if cell is not None:
            word = cell.load()
            return (_VALUES[(word & _CODE) >> _SHIFT] if word & _HAS else None), word & _COUNT
        if table.full(key):
            return table.reference, _COUNT
    return choices.get(key), counts.get(key, 0)


def counted(choices, counts, key, found, checks, reference=False):
    """Record one check and return `(choice, seen)` from the resulting CAS word."""
    table = getattr(choices, 'table', None)
    if isinstance(choices, _Choices) and isinstance(counts, _Counts) and counts.table is table:
        if table.full(key):
            return reference, _COUNT
        cell = table.cell(key)
        if cell is not None:
            word = cell.count(_code(found) if found else 0, _code(reference), int(checks), not found)
            return _VALUES[(word & _CODE) >> _SHIFT], word & _COUNT
    per_thread.checked(choices, counts, key, found, checks, reference)
    return choices.get(key), counts.get(key, 0)


def settled(choices, counts, key, checks):
    """Return `choice` for `key` once `seen >= checks` (via single atomic `read()`), else None."""
    choice, seen = read(choices, counts, key)
    return choice if seen >= checks else None


def same_bits(a, b):
    """Return True iff arrays (or nested tuples/lists of arrays) have identical dtype, shape, and bit patterns."""
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return (
            isinstance(a, (tuple, list))
            and isinstance(b, (tuple, list))
            and len(a) == len(b)
            and all(same_bits(x, y) for x, y in zip(a, b))
        )
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    unsigned = _UNSIGNED.get(a.dtype.size)
    if unsigned is None:
        return bool(mx.array_equal(a, b).item())
    return bool(mx.array_equal(a.view(unsigned), b.view(unsigned)).item())


def same_bits_each(pairs):
    """Batch `same_bits(a, b)` over `(a, b)` array pairs with a single GPU synchronization."""
    out, asked = [False] * len(pairs), []
    for n, (a, b) in enumerate(pairs):
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            continue
        unsigned = _UNSIGNED.get(a.dtype.size)
        asked.append((
            n,
            mx.array_equal(a, b) if unsigned is None else mx.array_equal(a.view(unsigned), b.view(unsigned)),
        ))
    if asked:
        for (n, _), same in zip(asked, mx.stack([e for _, e in asked]).tolist()):
            out[n] = bool(same)
    return out


def served(module, key, mine, theirs, what, stat='served', same=same_bits):
    """Run `mine()` when shape `key` passed `module.CHECKS` checks, else verify or fall back to `theirs()`."""
    choice, seen = read(module.CHOICE, module.CHECKED, key)
    if choice is not None and seen >= module.CHECKS:
        if choice:
            module.STATS[stat] += 1
            return mine()
        return theirs()
    want = theirs()
    module.STATS['checked'] += 1
    try:
        found = bool(same(mine(), want))
    except Exception as error:
        if not module.SAID:
            module.SAID.append(True)
            print(f'quanteeze: {what} unavailable ({type(error).__name__}: {error})', file=sys.stderr, flush=True)
        found = False
    module.STATS['mismatched'] += not found
    per_thread.checked(module.CHOICE, module.CHECKED, key, found, module.CHECKS)
    return want
