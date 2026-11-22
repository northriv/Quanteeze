# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Thread-local per-pass state and per-thread request attributes (`quanteeze.per_thread`).

Replaces process-global mutable lists (`dspark.PENDING`, `dspark.DRAFTED`, `dspark.ROWS`,
`cuda_ple.TOKENS`, `cuda_ple.PENDING`, `deepseek_v41.VERIFYING`, `_WINDOWS`) with views of one
thread-local `PASS` object (`Field`, `Rows`), plus lock-free per-thread counters (`Counts`) and
per-thread descriptors (`own_attributes`) for shared `mlx.nn.Module` state such as `_rope_deltas`
and `_position_ids`.
"""
import copy
import dataclasses
import threading
import weakref
from collections import defaultdict
from itertools import count

import numpy as np


class _Pass(threading.local):
    """Per-thread pass state initialized fresh on every thread."""

    def __init__(self):
        self.verify = False
        self.exact = False
        self.batch = False
        self.caps = None
        self.round = None
        self.drafted = None
        self.rows = []
        self.sorted_rows = False
        self.ple_pending = None
        self.ple_tokens = None
        self.entries = {}
        self.request_uid = None
        self.birth = 0
        self.memo = {}


PASS = _Pass()
FIELDS = tuple(vars(_Pass()))


class Field:
    """One-element list view (`X[0]`) backed by `PASS.<name>` on the calling thread."""

    def __init__(self, name):
        if name not in FIELDS:
            raise ValueError(f'no pass field {name}')
        self.name = name

    def __getitem__(self, i):
        if i not in (0, -1):
            raise IndexError(i)
        return getattr(PASS, self.name)

    def __setitem__(self, i, value):
        if i not in (0, -1):
            raise IndexError(i)
        setattr(PASS, self.name, value)

    def __len__(self):
        return 1

    def __iter__(self):
        return iter((getattr(PASS, self.name),))

    def __repr__(self):
        return f'per_thread.Field({self.name!r}: {getattr(PASS, self.name)!r})'


class Rows:
    """List view backed by `PASS.<name>` on the calling thread."""

    def __init__(self, name):
        if name not in FIELDS:
            raise ValueError(f'no pass field {name}')
        self.name = name

    def _list(self):
        return getattr(PASS, self.name)

    def append(self, value):
        self._list().append(value)

    def clear(self):
        self._list().clear()

    def __iter__(self):
        return iter(self._list())

    def __len__(self):
        return len(self._list())

    def __getitem__(self, i):
        return self._list()[i]

    def __repr__(self):
        return f'per_thread.Rows({self.name!r}: {self._list()!r})'


class Counts:
    """Per-thread counter map whose reads sum all threads and writes record per-thread deltas."""

    def __init__(self, values=None):
        self._template = dict(values or {})
        self._local = threading.local()
        self._owns = []
        self._lock = threading.Lock()

    def _mine(self):
        own = getattr(self._local, 'own', None)
        if own is None:
            own = self._local.own = {}
            with self._lock:
                self._owns.append(own)
        return own

    def _sum(self, key):
        found, out = False, None
        for own in list(self._owns):
            if key in own:
                v = own[key]
                if not found:
                    out, found = (list(v) if isinstance(v, list) else v), True
                elif _number(v) and _number(out):
                    out = out + v
                elif isinstance(v, list) and isinstance(out, list):
                    n = max(len(v), len(out))
                    out = [(out[i] if i < len(out) else 0) + (v[i] if i < len(v) else 0) for i in range(n)]
        if not found:
            if key not in self._template:
                raise KeyError(key)
            out = copy.deepcopy(self._template[key])
        return out

    def _alone(self):
        owns = self._owns
        if len(owns) == 1:
            own = getattr(self._local, 'own', None)
            if own is owns[0]:
                return own
        return None

    def __getitem__(self, key):
        own = self._alone()
        if own is not None:
            try:
                v = own[key]
            except KeyError:
                if key not in self._template:
                    raise
                v = own[key] = copy.deepcopy(self._template[key])
            self._local.read = (key, v)
            return v
        out = self._sum(key)
        if isinstance(out, bool) or not (_number(out) or isinstance(out, list)):
            return out
        if isinstance(out, list):
            return _ReadList(self, key, out) if all(map(_number, out)) else out
        return (_ReadInt if isinstance(out, int) else _ReadFloat)(out, key, out)

    def __setitem__(self, key, value):
        alone = self._alone()
        if alone is not None:
            alone[key] = _plain(value)
            return
        own = self._mine()
        if isinstance(value, bool) or not (_number(value) or isinstance(value, list)):
            own[key] = value
            return
        if isinstance(value, list):
            own[key] = list(value)
            return
        read = getattr(self._local, 'read', None)
        if isinstance(value, (_ReadInt, _ReadFloat)) and value.key == key:
            base = value.base
        elif read is not None and read[0] == key and _number(read[1]):
            base = read[1]
        else:
            base = self._sum_or(key, 0)
        self._local.read = None
        mine = own.get(key)
        if mine is None:
            mine = _zero_like(self._template[key]) if key in self._template else type(value)(0)
            if not self._owns or own is self._owns[0]:
                mine = copy.deepcopy(self._template.get(key, mine))
        own[key] = _plain(mine + (value - base))

    def _sum_or(self, key, default):
        try:
            return self._sum(key)
        except KeyError:
            return default

    def _add_item(self, key, i, change):
        own = self._mine()
        mine = own.get(key)
        if mine is None:
            mine = own[key] = _zero_like(self._sum(key))
        while len(mine) <= i:
            mine.append(0)
        mine[i] = _plain(mine[i] + change)

    def __contains__(self, key):
        return key in self._template or any(key in own for own in list(self._owns))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key, value):
        with self._lock:
            if key not in self:
                self._template[key] = value
        return self[key]

    def total(self):
        """Return summed counts across all threads as a plain dictionary."""
        with self._lock:
            keys = dict.fromkeys(self._template)
        for own in list(self._owns):
            keys.update(dict.fromkeys(own))
        return {k: self._sum(k) for k in keys}

    def keys(self):
        return self.total().keys()

    def items(self):
        return self.total().items()

    def values(self):
        return self.total().values()

    def __len__(self):
        return len(self.total())

    def __iter__(self):
        return iter(self.total())

    def __repr__(self):
        return f'per_thread.Counts({self.total()!r})'


class _ReadInt(int):
    """Integer count carrying its read baseline so `c[k] = c[k] + n` commits only the delta."""

    def __new__(cls, value, key, base):
        out = super().__new__(cls, value)
        out.key, out.base = key, base
        return out

    def _again(self, value):
        if _number(value) and not isinstance(value, bool):
            return (_ReadInt if isinstance(value, int) else _ReadFloat)(value, self.key, self.base)
        return value

    def __add__(self, other):
        return self._again(int(self) + _plain(other))

    def __sub__(self, other):
        return self._again(int(self) - _plain(other))

    __iadd__, __isub__ = __add__, __sub__


class _ReadFloat(float):
    """Float count carrying its read baseline."""

    def __new__(cls, value, key, base):
        out = super().__new__(cls, value)
        out.key, out.base = key, base
        return out

    def _again(self, value):
        if _number(value) and not isinstance(value, bool):
            return _ReadFloat(float(value), self.key, self.base)
        return value

    def __add__(self, other):
        return self._again(float(self) + _plain(other))

    def __sub__(self, other):
        return self._again(float(self) - _plain(other))

    __iadd__, __isub__ = __add__, __sub__


class _ReadList(list):
    """List count whose item assignments forward deltas to `Counts`."""

    def __init__(self, counts, key, values):
        super().__init__(values)
        self._counts, self._key = counts, key

    def __setitem__(self, i, value):
        change = _plain(value) - _plain(self[i])
        super().__setitem__(i, _plain(value))
        self._counts._add_item(self._key, i, change)


def _plain(v):
    if isinstance(v, _ReadInt):
        return int(v)
    if isinstance(v, _ReadFloat):
        return float(v)
    return v


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _zero_like(v):
    if _number(v):
        return type(v)(0)
    if isinstance(v, list) and all(map(_number, v)):
        return [type(x)(0) for x in v]
    return copy.deepcopy(v)


KIND_PROMPT = 0
KIND_DECODE = 1
KIND_VERIFY = 2
KIND_DRAFT = 3
PASS_KINDS = ('prompt', 'decode', 'verify', 'draft')
KIND_CODES = {k: n for n, k in enumerate(PASS_KINDS)}


@dataclasses.dataclass(frozen=True)
class PassMark:
    """Descriptor identifying a pass by id, request number, cache serial, and pass kind."""

    number: int
    request: int = -1
    serial: int = -1
    kind: str = 'decode'

    @property
    def kind_code(self):
        return KIND_CODES[self.kind]


_PASSING = threading.local()
_REQUESTS = weakref.WeakKeyDictionary()
_REQUESTS_LOCK = threading.Lock()
_REQUEST_NUMBERS = count()
_SERIALS = defaultdict(count)


def request_number(entry=None):
    """Return a monotonically assigned integer id for KV cache `entry`, or `-1`."""
    if entry is None:
        return -1
    try:
        with _REQUESTS_LOCK:
            n = _REQUESTS.get(entry)
            if n is None:
                n = _REQUESTS[entry] = next(_REQUEST_NUMBERS)
        return n
    except TypeError:
        return -1


def next_serial(req_id):
    """Return the next per-request monotonic pass serial integer for `req_id`."""
    with _REQUESTS_LOCK:
        return next(_SERIALS[int(req_id)])


def passing(entry, verify=None):
    """Record request number, cache offset serial, and verify flag for the thread's next pass."""
    offset = getattr(entry, 'offset', None)
    serial = int(offset) if isinstance(offset, (int, np.integer)) else -1
    _PASSING.next = (request_number(entry), serial, bool(PASS.verify if verify is None else verify))


class _Own:
    """Descriptor storing `obj.<name>` in per-thread storage attached to `obj`."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return getattr(obj.__dict__['_quanteeze_own'], self.name, None)

    def __set__(self, obj, value):
        setattr(obj.__dict__['_quanteeze_own'], self.name, value)

    def __delete__(self, obj):
        setattr(obj.__dict__['_quanteeze_own'], self.name, None)


_OWN_CLASSES = {}
_OWN_LOCK = threading.Lock()


def own_attributes(obj, names):
    """Make attributes `names` on `obj` thread-local without registering arrays as MLX parameters."""
    if type(obj).__dict__.get('_quanteeze_own_names') is not None:
        return ()
    names = tuple(n for n in names if n in vars(obj) or (isinstance(obj, dict) and n in obj))
    if not names:
        return ()
    base = type(obj)
    with _OWN_LOCK:
        sub = _OWN_CLASSES.get((base, names))
        if sub is None:
            def __setattr__(self, k, v, _names=frozenset(names), _base=base):
                if k in _names:
                    object.__setattr__(self, k, v)
                else:
                    _base.__setattr__(self, k, v)

            body = {n: _Own(n) for n in names}
            body.update(
                __setattr__=__setattr__,
                _quanteeze_own_names=names,
                __module__=base.__module__,
                __qualname__=base.__qualname__,
            )
            sub = _OWN_CLASSES[(base, names)] = type(base.__name__, (base,), body)
    own = threading.local()
    for n in names:
        value = vars(obj).pop(n, None)
        if isinstance(obj, dict) and n in obj:
            value = dict.pop(obj, n)
        setattr(own, n, value)
    object.__setattr__(obj, '_quanteeze_own', own)
    obj.__class__ = sub
    return names


def slots(switch):
    """Return weakref-keyed per-thread scratch dictionary for `switch`."""
    mine = PASS.memo.get('switch_slots')
    if mine is None:
        mine = PASS.memo['switch_slots'] = {}
    got = mine.get(id(switch))
    if got is None or got[0]() is not switch:
        got = mine[id(switch)] = (weakref.ref(switch), {})
    return got[1]
