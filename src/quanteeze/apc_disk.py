# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Convergent AES-256-GCM sealing for mlx-vlm's on-disk Prefix Cache (APC).

mlx-vlm's `DiskBlockStore` writes `.safetensors` files whose plaintext JSON headers carry
the prompt's `token_ids` and whose data sections hold unencrypted KV / recurrent state
tensors.  `install()` wraps `mlx_vlm.apc` so that when `QUANTEEZE_APC_ENCRYPT` is `on`:

1. Every disk shard (`shard_<32-hex>.safetensors`) and exact checkpoint
   (`exact_<32-hex>.safetensors`) is streamed in `PIECE`-sized chunks directly through a
   streaming `AES-256-GCM` cipher (never materializing full checkpoints in memory or hitting
   one-shot 2 GiB AEAD limits) under a 256-bit convergent key derived from the prompt's own
   token chain, `extra_hash`, and the active quantization / precision configuration
   (`QUANTEEZE_BITS`, and `serve._precision()`).
2. Block shards and trimmable dense checkpoints seal each `block_size` slice under its
   own chained prefix key `K_i`, so a request sharing only an earlier prefix derives keys
   for and decrypts only that shared prefix without being able to decrypt later blocks.
3. On-disk filenames and outer envelopes store only one-way HMAC-SHA-256 locators keyed by
   this machine's secret (`QUANTEEZE_APC_DISK_SECRET`, 32 random bytes stored 0600 and excluded
   from Time Machine on macOS; `off` falls back to plain SHA-256) and record offsets
   (authenticated as GCM Associated Data), exposing zero token IDs, tensor shapes, or
   unencrypted KV values at rest.  Unsealed legacy files or tampered ciphertexts are
   rejected cleanly on startup or lookup without crashing the server.
4. Multi-length exact-prefix lookup runs in a single incremental hash pass (`h.copy()`),
   and saving a longer conversation checkpoint prunes intermediate extended checkpoints
   while preserving the shortest shared prefix and the previous longest checkpoint.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import stat
import struct
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from mlx_vlm import apc as _apc

from . import variables

logger = logging.getLogger(__name__)

MAGIC = b'QZAPC01\x00'
NONCE_BYTES = 12
TAG_BYTES = 16
PIECE = 64 << 20
DELTA_MIN_BYTES = 1 << 20
DELTA_DEPTH = 32
_SYNTHETIC = 1 << 52
MAX_ENVELOPE_BYTES = 4 * 1024 * 1024
CHAIN_KEY_CAP = 65536
UNLOCK_CAP = 512
SENTINEL_TENSOR = '__qz_empty_sentinel__'
_DTYPES = {
    mx.bfloat16: 'BF16',
    mx.float16: 'F16',
    mx.float32: 'F32',
    mx.int64: 'I64',
    mx.int32: 'I32',
    mx.int16: 'I16',
    mx.int8: 'I8',
    mx.uint64: 'U64',
    mx.uint32: 'U32',
    mx.uint16: 'U16',
    mx.uint8: 'U8',
    mx.bool_: 'BOOL',
}

ON = [True]
PRECISION = [None]
STATS = {
    'writes': 0,
    'hits': 0,
    'rejected': 0,
    'purged_plaintext': 0,
    'pruned': 0,
}

_CHAIN_LOCK = threading.Lock()
_CHAIN_KEYS: OrderedDict[int, bytes] = OrderedDict()

_UNLOCK_LOCK = threading.Lock()
_UNLOCK: OrderedDict[Tuple[int, int], tuple] = OrderedDict()

_MEM_SAFES = threading.local()

SECRET_BYTES = 32
_SECRET = {'value': None, 'read': False}
_SECRET_LOCK = threading.Lock()


def use_secret(value):
    """Set the in-process APC disk secret (`bytes`, or `None` for plain SHA-256), bypassing disk (tests)."""
    with _SECRET_LOCK:
        _SECRET['value'] = None if value is None else bytes(value)
        _SECRET['read'] = True


def secret():
    """Return this machine's secret (`SECRET_BYTES`) for HMAC-SHA-256 disk locators/keys, or None when off."""
    with _SECRET_LOCK:
        if not _SECRET['read']:
            if variables.off('QUANTEEZE_APC_DISK_SECRET'):
                _SECRET['value'] = None
            else:
                path = variables.path('QUANTEEZE_APC_DISK_SECRET')
                try:
                    _SECRET['value'] = _secret_in(path)
                except OSError as e:
                    _SECRET['value'] = secrets.token_bytes(SECRET_BYTES)
                    logger.error(
                        'APC disk: no secret kept at %s (%s): this process seals under an ephemeral secret',
                        path, e,
                    )
            _SECRET['read'] = True
        return _SECRET['value']


def _kept_secret(path):
    """Return the secret bytes in `path` if regular (non-symlink), owned by uid, mode 0600-safe, and `SECRET_BYTES` hex."""
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            return None
        value = bytes.fromhex(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return value if len(value) == SECRET_BYTES else None


def _secret_in(path):
    """Return or create the secret at `path` (mode 0600 via atomic link/replace) and exclude from Time Machine."""
    kept = _kept_secret(path)
    if kept is not None:
        return kept
    path.parent.mkdir(parents=True, exist_ok=True)
    unsafe = path.exists() or path.is_symlink()
    made = secrets.token_bytes(SECRET_BYTES)
    part = path.with_name(f'.{path.name}.{os.getpid()}.{secrets.token_hex(4)}')
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w') as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(made.hex() + '\n')
            f.flush()
            os.fsync(f.fileno())
        if unsafe:
            os.replace(part, path)
            logger.warning(
                'APC disk: %s was not readable by this user alone; a new secret is active and prior files age out',
                path,
            )
        else:
            try:
                os.link(part, path)
            except FileExistsError:
                kept = _kept_secret(path)
                if kept is not None:
                    return kept
                os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()
    _exclude_from_backups(path)
    return made


def _exclude_from_backups(path):
    """On macOS, exclude `path` from Time Machine backups."""
    if sys.platform != 'darwin':
        return
    try:
        subprocess.run(['tmutil', 'addexclusion', str(path)], check=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning('APC disk: %s not excluded from Time Machine (%s): exclude it manually', path, e)


def _hash(data: bytes = b''):
    """Incremental HMAC-SHA-256 under `secret()`, or plain SHA-256 when `QUANTEEZE_APC_DISK_SECRET=off`."""
    key = secret()
    return hashlib.sha256(data) if key is None else hmac.new(key, data, hashlib.sha256)

_GATHERS: dict[int, tuple] = {}
_GATHERS_LOCK = threading.Lock()
_MINE = threading.local()

WROTE = {'bytes': 0, 'refused': 0, 'said': False}
_WROTE_LOCK = threading.Lock()


def write_cap() -> Optional[float]:
    """Return `QUANTEEZE_APC_DISK_WRITE_GB` in bytes, or None when unbounded."""
    raw = variables.get('QUANTEEZE_APC_DISK_WRITE_GB', variables.at_start())
    return None if raw in (None, '', 'off') else float(raw) * 1e9


def over_cap() -> bool:
    """Return True once bytes written by this process reach `write_cap()`."""
    cap = write_cap()
    with _WROTE_LOCK:
        if cap is None or WROTE['bytes'] < cap:
            return False
        WROTE['refused'] += 1
        said, WROTE['said'] = WROTE['said'], True
    if not said:
        logger.warning(
            'APC disk: %.1f GB written, QUANTEEZE_APC_DISK_WRITE_GB=%g: no new checkpoints/shards written',
            WROTE['bytes'] / 1e9,
            cap / 1e9,
        )
        print(
            f'quanteeze: APC disk wrote {WROTE["bytes"] / 1e9:.1f} GB '
            f'(QUANTEEZE_APC_DISK_WRITE_GB={cap / 1e9:g}): skipping further writes',
            file=sys.stderr,
            flush=True,
        )
    return True


def _record_wrote(path: Path) -> int:
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if size:
        with _WROTE_LOCK:
            WROTE['bytes'] += size
    return size


def gathered(array, source, indices, axis: int) -> None:
    """Register lazy `mx.take(source, indices, axis)` view `array` for streaming serialization."""
    with _GATHERS_LOCK:
        _GATHERS[id(array)] = (array, source, np.asarray(indices, np.int32), int(axis))
    mine = getattr(_MINE, 'ids', None)
    if mine is None:
        mine = _MINE.ids = set()
    mine.add(id(array))


def gather_of(array):
    """Return `(source, indices, axis)` if `array` is a registered lazy gather view, else None."""
    with _GATHERS_LOCK:
        got = _GATHERS.get(id(array))
    return got[1:] if got is not None and got[0] is array else None


def forget(arrays=None, everyone: bool = False) -> None:
    """Drop lazy gather registrations for `arrays`, current thread (`arrays is None`), or all threads."""
    with _GATHERS_LOCK:
        if everyone:
            _GATHERS.clear()
            _MINE.ids = set()
        elif arrays is None:
            for k in getattr(_MINE, 'ids', ()):
                _GATHERS.pop(k, None)
            _MINE.ids = set()
        else:
            for a in arrays:
                if a is not None and _GATHERS.get(id(a), (None,))[0] is a:
                    del _GATHERS[id(a)]


def use_precision(precision) -> None:
    """Register `precision()`, text describing the active model precision, mixed into every key."""
    PRECISION[0] = precision


def clear_runtime_keys() -> None:
    """Drop ephemeral in-memory derived keys (for cold-restart testing or explicit cache resets)."""
    with _CHAIN_LOCK:
        _CHAIN_KEYS.clear()
    with _UNLOCK_LOCK:
        _UNLOCK.clear()


def summary() -> str:
    extra = (
        f", {WROTE['bytes'] / 1e9:.1f} GB written ({WROTE['refused']} refused)"
        if WROTE['refused'] else ''
    )
    return (
        f"{STATS['writes']} sealed writes, {STATS['hits']} decrypted hits, "
        f"{STATS['rejected']} rejected, {STATS['pruned']} pruned, "
        f"{STATS['purged_plaintext']} unsealed purged{extra}"
    )


def _precision_salt() -> bytes:
    own = PRECISION[0]() if PRECISION[0] is not None else ''
    bits = (variables.get('QUANTEEZE_BITS', variables.at_load()) or 'off').strip().lower()
    return f'{own}|bits={bits}'.encode('utf-8')


def _root_chain_key(salt: bytes) -> bytes:
    h = _hash()
    h.update(b'quanteeze-apc-root-v1\x00')
    h.update(len(salt).to_bytes(4, 'little'))
    h.update(salt)
    return h.digest()


def _step_chain_key(parent_key: bytes, tokens: Sequence[int], extra_hash: int) -> Tuple[bytes, int]:
    h = _hash()
    h.update(b'quanteeze-apc-blk-key-v1\x00')
    h.update(parent_key)
    h.update(int(extra_hash & ((1 << 64) - 1)).to_bytes(8, 'little'))
    arr = np.asarray(tokens, dtype='<i4')
    h.update(int(arr.size).to_bytes(4, 'little'))
    h.update(arr.tobytes())
    key = h.digest()
    loc_digest = _hash(b'quanteeze-apc-blk-loc-v1\x00' + key).digest()
    loc = int.from_bytes(loc_digest[:8], 'little', signed=True)
    if loc == _apc.SEED_PARENT_HASH:
        loc ^= 1
    return key, loc


def _put_chain_key(loc: int, key: bytes) -> None:
    with _CHAIN_LOCK:
        _CHAIN_KEYS[int(loc)] = key
        _CHAIN_KEYS.move_to_end(int(loc))
        while len(_CHAIN_KEYS) > CHAIN_KEY_CAP:
            _CHAIN_KEYS.popitem(last=False)


def _get_chain_key(loc: int) -> Optional[bytes]:
    with _CHAIN_LOCK:
        key = _CHAIN_KEYS.get(int(loc))
        if key is not None:
            _CHAIN_KEYS.move_to_end(int(loc))
        return key


def _derive_exact_locator(key: bytes) -> Tuple[int, str]:
    d = _hash(b'quanteeze-apc-exact-loc-v1\x00' + key).digest()
    loc_int = int.from_bytes(d[:8], 'little', signed=True)
    return loc_int, d[:16].hex()


def _derive_exact_digests(
    tokens: Sequence[int],
    extra_hash: int,
    lengths: Iterable[int],
    salt: bytes,
) -> dict[int, Tuple[bytes, int, str]]:
    """Derive convergent keys and locators for multiple prefix lengths in one SHA-256 pass."""
    n_toks = len(tokens)
    targets = sorted({int(n) for n in lengths if 0 < int(n) <= n_toks})
    if not targets:
        return {}
    max_n = targets[-1]
    arr = np.ascontiguousarray(tokens[:max_n], dtype='<i4')
    raw = memoryview(arr).cast('B')

    h = _hash()
    h.update(b'quanteeze-apc-exact-key-v1\x00')
    h.update(len(salt).to_bytes(4, 'little'))
    h.update(salt)
    h.update(int(extra_hash & ((1 << 64) - 1)).to_bytes(8, 'little'))

    out: dict[int, Tuple[bytes, int, str]] = {}
    done = 0
    for n in targets:
        if n > done:
            h.update(raw[4 * done : 4 * n])
            done = n
        snap = h.copy()
        snap.update(int(n).to_bytes(8, 'little'))
        key = snap.digest()
        loc_int, loc_hex = _derive_exact_locator(key)
        out[n] = (key, loc_int, loc_hex)
    return out


def _derive_exact_key(tokens: Sequence[int], extra_hash: int, salt: bytes) -> bytes:
    n = len(tokens)
    if n <= 0:
        h = _hash()
        h.update(b'quanteeze-apc-exact-key-v1\x00')
        h.update(len(salt).to_bytes(4, 'little'))
        h.update(salt)
        h.update(int(extra_hash & ((1 << 64) - 1)).to_bytes(8, 'little'))
        h.update((0).to_bytes(8, 'little'))
        return h.digest()
    return _derive_exact_digests(tokens, extra_hash, (n,), salt)[n][0]


def _put_unlock(slot: Tuple[int, int], grant: tuple) -> None:
    with _UNLOCK_LOCK:
        _UNLOCK[slot] = grant
        _UNLOCK.move_to_end(slot)
        while len(_UNLOCK) > UNLOCK_CAP:
            _UNLOCK.popitem(last=False)


def _get_unlock(slot: Tuple[int, int]) -> Optional[tuple]:
    with _UNLOCK_LOCK:
        grant = _UNLOCK.get(slot)
        if grant is not None:
            _UNLOCK.move_to_end(slot)
        return grant


def _safetensors_dumps(arrays: dict[str, mx.array], metadata: dict[str, str]) -> bytes:
    work_arrays = dict(arrays)
    work_meta = dict(metadata)
    empty_specs = _apc._extract_empty_tensor_specs(work_arrays)
    if empty_specs:
        work_meta[_apc._EMPTY_TENSORS_METADATA] = json.dumps(
            empty_specs, separators=(',', ':'), sort_keys=True
        )
    if not work_arrays:
        work_arrays[SENTINEL_TENSOR] = mx.zeros((1,), dtype=mx.uint8)
    mx.eval(list(work_arrays.values()))
    buf = io.BytesIO()
    mx.save_safetensors(buf, work_arrays, metadata=work_meta)
    return buf.getvalue()


def _parse_safetensors_bytes(raw: bytes) -> Optional[Tuple[dict, dict, int]]:
    if len(raw) < 8:
        return None
    header_size = int.from_bytes(raw[:8], 'little')
    if header_size <= 0 or header_size > 64 * 1024 * 1024 or 8 + header_size > len(raw):
        return None
    try:
        header = json.loads(raw[8 : 8 + header_size])
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(header, dict):
        return None
    metadata = dict(header.pop('__metadata__', {}) or {})
    header.pop(SENTINEL_TENSOR, None)
    return header, metadata, 8 + header_size


def _load_arrays_from_safetensors_bytes(raw: bytes) -> Optional[Tuple[dict[str, mx.array], dict]]:
    parsed = _parse_safetensors_bytes(raw)
    if parsed is None:
        return None
    tensor_entries, metadata, data_start = parsed
    tensor_entries = _apc._restore_empty_tensor_entries(tensor_entries, metadata)
    if tensor_entries is None:
        return None
    view = memoryview(raw)
    out: dict[str, mx.array] = {}
    for name, entry in tensor_entries.items():
        if name == SENTINEL_TENSOR:
            continue
        if entry.get(_apc._EMPTY_TENSOR_MARKER) is True:
            try:
                out[name] = mx.zeros(tuple(entry['shape']), dtype=entry['mlx_dtype'])
            except (KeyError, TypeError, ValueError):
                return None
            continue
        bounds = _apc._safetensors_tensor_bounds(entry)
        if bounds is None:
            return None
        start, end, _ = bounds
        if data_start + end > len(raw):
            return None
        arr = _apc._mlx_array_from_safetensors_bytes(
            view[data_start + start : data_start + end], entry
        )
        if arr is None:
            return None
        out[name] = arr
    return out, metadata


def _bytes(a: mx.array) -> memoryview:
    """Return `a`'s C-order bytes as a flat `uint8` `memoryview`."""
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.view(mx.uint16)
    return memoryview(np.ascontiguousarray(np.array(a)).reshape(-1).view(np.uint8))


def _pieces(a: mx.array, limit: int = PIECE):
    """Yield `a`'s C-order bytes in slices of at most `limit` bytes without materializing large copies."""
    gather = gather_of(a)
    if gather is not None:
        source, indices, axis = gather
        per = int(np.prod(source.shape[axis + 1 :], dtype=np.int64)) * source.itemsize
        step = max(1, limit // max(per, 1))
        for lead in np.ndindex(*source.shape[:axis]):
            part = source[lead] if lead else source
            for i in range(0, indices.size, step):
                yield _bytes(mx.take(part, mx.array(indices[i : i + step]), axis=0))
        return
    if a.ndim == 0 or a.size == 0 or a.nbytes <= limit:
        yield _bytes(a)
        return
    row = a.nbytes // a.shape[0]
    if row > limit and a.ndim > 1:
        for i in range(a.shape[0]):
            yield from _pieces(a[i], limit)
        return
    step = max(1, limit // max(row, 1))
    for i in range(0, a.shape[0], step):
        yield _bytes(a[i : i + step])


def _prepare_safetensors_record(
    arrays: dict[str, mx.array],
    metadata: dict[str, str],
) -> Tuple[list[str], dict[str, mx.array], dict, bytes, int]:
    """Prepare safetensors header preamble and tensor byte layout without building a monolithic buffer."""
    work_arrays = dict(arrays)
    work_meta = {str(k): str(v) for k, v in metadata.items()}
    empty_specs = _apc._extract_empty_tensor_specs(work_arrays)
    if empty_specs:
        work_meta[_apc._EMPTY_TENSORS_METADATA] = json.dumps(
            empty_specs, separators=(',', ':'), sort_keys=True
        )
    if not work_arrays:
        work_arrays[SENTINEL_TENSOR] = mx.zeros((1,), dtype=mx.uint8)

    names = list(work_arrays)
    header: dict[str, Any] = {}
    offset = 0
    for name in names:
        arr = work_arrays[name]
        nbytes = int(arr.nbytes)
        header[name] = {
            'dtype': _DTYPES[arr.dtype],
            'shape': [int(d) for d in arr.shape],
            'data_offsets': [offset, offset + nbytes],
        }
        offset += nbytes
    header['__metadata__'] = work_meta
    text = json.dumps(header, separators=(',', ':')).encode('utf-8')
    text += b' ' * (-len(text) % 8)
    preamble = struct.pack('<Q', len(text)) + text
    return names, work_arrays, header, preamble, len(preamble) + offset


def _write_atomic(path: Path, write_fn) -> None:
    """Create parent dirs (`0o700`), stream to a `0o600` temp file, and atomically replace `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass

    tag = f'{os.getpid()}-{threading.get_ident()}'
    tmp = path.parent / f'{path.stem}.{tag}{_apc.DiskBlockStore.SUFFIX}'
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            f = os.fdopen(fd, 'wb')
        except BaseException:
            os.close(fd)
            raise
        with f:
            write_fn(f)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _write_sealed_records(
    path: Path,
    envelope: dict,
    record_specs: Sequence[Tuple[dict[str, mx.array], dict[str, str], bytes]],
    *,
    limit: int = PIECE,
) -> dict:
    """Stream safetensors records through AES-256-GCM in `limit`-byte chunks and atomically seal `path`."""
    prepared = []
    records = []
    cursor = 0
    for arrays, metadata, key in record_specs:
        names, work_arrays, header, preamble, plain_len = _prepare_safetensors_record(arrays, metadata)
        rec_len = NONCE_BYTES + plain_len + TAG_BYTES
        records.append([cursor, rec_len])
        cursor += rec_len
        prepared.append((names, work_arrays, header, preamble, key))

    env = dict(envelope)
    env['records'] = records
    header_bytes = json.dumps(env, separators=(',', ':'), sort_keys=True).encode('utf-8')
    prefix_bytes = MAGIC + len(header_bytes).to_bytes(4, 'little') + header_bytes

    def _stream(f) -> None:
        f.write(prefix_bytes)
        for idx, (names, work_arrays, header, preamble, key) in enumerate(prepared):
            nonce = os.urandom(NONCE_BYTES)
            enc = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
            enc.authenticate_additional_data(prefix_bytes + f':rec:{idx}'.encode('ascii'))
            f.write(nonce)
            f.write(enc.update(preamble))
            for name in names:
                expected = header[name]['data_offsets'][1] - header[name]['data_offsets'][0]
                written = 0
                for piece in _pieces(work_arrays[name], limit):
                    f.write(enc.update(piece))
                    written += piece.nbytes
                if written != expected:
                    raise ValueError(f'{name}: {written} bytes written for {header[name]["shape"]}')
            f.write(enc.finalize())
            f.write(enc.tag)

    _write_atomic(path, _stream)
    return env


def _write_sealed_file(
    path: Path,
    envelope: dict,
    plaintexts_and_keys: Sequence[Tuple[bytes, bytes]],
) -> dict:
    records = []
    cursor = 0
    for plaintext, _ in plaintexts_and_keys:
        rec_len = NONCE_BYTES + len(plaintext) + TAG_BYTES
        records.append([cursor, rec_len])
        cursor += rec_len
    env = dict(envelope)
    env['records'] = records
    header_bytes = json.dumps(env, separators=(',', ':'), sort_keys=True).encode('utf-8')
    prefix_bytes = MAGIC + len(header_bytes).to_bytes(4, 'little') + header_bytes

    blobs: list[bytes] = []
    for idx, (plaintext, key) in enumerate(plaintexts_and_keys):
        nonce = os.urandom(NONCE_BYTES)
        aad = prefix_bytes + f':rec:{idx}'.encode('ascii')
        ct = AESGCM(key).encrypt(nonce, plaintext, aad)
        blobs.append(nonce + ct)

    def _write(f) -> None:
        f.write(prefix_bytes)
        for blob in blobs:
            f.write(blob)

    _write_atomic(path, _write)
    return env


def _read_envelope(path: Path) -> Optional[Tuple[dict, bytes, int]]:
    try:
        with open(path, 'rb') as f:
            head = f.read(12)
            if len(head) < 12 or head[:8] != MAGIC:
                return None
            header_len = int.from_bytes(head[8:12], 'little')
            if header_len <= 0 or header_len > MAX_ENVELOPE_BYTES:
                return None
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                return None
        env = json.loads(header_bytes)
        if not isinstance(env, dict) or not isinstance(env.get('records'), list):
            return None
        return env, head + header_bytes, 12 + header_len
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def _decrypt_record(
    f,
    prefix_bytes: bytes,
    data_start: int,
    records: list,
    idx: int,
    key: bytes,
) -> Optional[bytes]:
    if not 0 <= idx < len(records):
        return None
    try:
        rel_off, rec_len = int(records[idx][0]), int(records[idx][1])
    except (IndexError, TypeError, ValueError):
        return None
    if rel_off < 0 or rec_len < NONCE_BYTES + TAG_BYTES:
        return None
    try:
        f.seek(data_start + rel_off)
        blob = f.read(rec_len)
    except OSError:
        return None
    if len(blob) != rec_len:
        STATS['rejected'] += 1
        return None
    nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
    aad = prefix_bytes + f':rec:{idx}'.encode('ascii')
    try:
        return AESGCM(key).decrypt(nonce, ct, aad)
    except (InvalidTag, ValueError):
        STATS['rejected'] += 1
        return None


def _decrypt_record_tensors(
    f,
    prefix_bytes: bytes,
    data_start: int,
    records: list,
    idx: int,
    key: bytes,
    *,
    limit: int = PIECE,
) -> Optional[Tuple[dict, dict, int, dict[Tuple[int, int], mx.array]]]:
    """Stream-decrypt one record in `limit`-byte chunks and materialize each tensor before verifying the GCM tag."""
    if not 0 <= idx < len(records):
        return None
    try:
        rel_off, rec_len = int(records[idx][0]), int(records[idx][1])
    except (IndexError, TypeError, ValueError):
        return None
    if rel_off < 0 or rec_len < NONCE_BYTES + TAG_BYTES:
        return None
    ct_len = rec_len - NONCE_BYTES - TAG_BYTES
    try:
        f.seek(data_start + rel_off)
        nonce = f.read(NONCE_BYTES)
        if len(nonce) != NONCE_BYTES:
            STATS['rejected'] += 1
            return None
        f.seek(data_start + rel_off + NONCE_BYTES + ct_len)
        tag = f.read(TAG_BYTES)
        if len(tag) != TAG_BYTES:
            STATS['rejected'] += 1
            return None
        f.seek(data_start + rel_off + NONCE_BYTES)
    except OSError:
        return None

    dec = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
    dec.authenticate_additional_data(prefix_bytes + f':rec:{idx}'.encode('ascii'))
    left = ct_len
    pending = bytearray()

    def take(n: int) -> np.ndarray:
        nonlocal left
        out = np.empty(n, np.uint8)
        have = min(n, len(pending))
        if have:
            out[:have] = np.frombuffer(bytes(pending[:have]), np.uint8)
            del pending[:have]
        while have < n:
            if left <= 0:
                raise ValueError('short ciphertext')
            chunk = f.read(min(limit, left))
            if not chunk:
                raise ValueError('short ciphertext')
            left -= len(chunk)
            piece = dec.update(chunk)
            use = min(len(piece), n - have)
            if use:
                out[have : have + use] = np.frombuffer(piece, np.uint8, count=use)
            if use < len(piece):
                pending.extend(piece[use:])
            have += use
        return out

    try:
        header_size = int.from_bytes(take(8).tobytes(), 'little')
        if header_size <= 0 or header_size > min(64 * 1024 * 1024, left + len(pending)):
            raise ValueError('invalid safetensors header size')
        header = json.loads(take(header_size).tobytes())
        if not isinstance(header, dict):
            raise ValueError('invalid safetensors header')
        metadata = dict(header.pop('__metadata__', {}) or {})
        spans = sorted(
            (tuple(int(x) for x in entry['data_offsets']), name)
            for name, entry in header.items()
            if isinstance(entry, dict) and 'data_offsets' in entry
        )
        tensors: dict[Tuple[int, int], mx.array] = {}
        at = 0
        for (begin, end), name in spans:
            if begin < at or end < begin:
                raise ValueError('invalid tensor data_offsets')
            if begin > at:
                take(begin - at)
            data = take(end - begin)
            if name != SENTINEL_TENSOR and end > begin:
                arr = _apc._mlx_array_from_safetensors_bytes(data, header[name])
                if arr is None:
                    raise ValueError(f'failed to decode tensor {name}')
                tensors[(begin, end)] = arr
            del data
            at = end
        header.pop(SENTINEL_TENSOR, None)
        while left > 0:
            chunk = f.read(min(limit, left))
            if not chunk:
                raise ValueError('short ciphertext')
            left -= len(chunk)
            dec.update(chunk)
        dec.finalize()
    except OSError:
        return None
    except (InvalidTag, UnicodeDecodeError, ValueError, KeyError, TypeError):
        STATS['rejected'] += 1
        return None

    return header, metadata, 8 + header_size, tensors


def _decrypt_record_arrays(
    f,
    prefix_bytes: bytes,
    data_start: int,
    records: list,
    idx: int,
    key: bytes,
    *,
    limit: int = PIECE,
) -> Optional[Tuple[dict[str, mx.array], dict]]:
    """Stream-decrypt one record into restored `{name: mx.array}` and `metadata`."""
    got = _decrypt_record_tensors(f, prefix_bytes, data_start, records, idx, key, limit=limit)
    if got is None:
        return None
    header, metadata, _st_data_start, tensors = got
    tensor_entries = _apc._restore_empty_tensor_entries(dict(header), metadata)
    if tensor_entries is None:
        STATS['rejected'] += 1
        return None
    out: dict[str, mx.array] = {}
    for name, entry in tensor_entries.items():
        if name == SENTINEL_TENSOR:
            continue
        if entry.get(_apc._EMPTY_TENSOR_MARKER) is True:
            try:
                out[name] = mx.zeros(tuple(entry['shape']), dtype=entry['mlx_dtype'])
            except (KeyError, TypeError, ValueError):
                STATS['rejected'] += 1
                return None
            continue
        bounds = _apc._safetensors_tensor_bounds(entry)
        if bounds is None:
            STATS['rejected'] += 1
            return None
        start, end, _ = bounds
        arr = (
            _apc._mlx_array_from_safetensors_bytes(b'', entry)
            if start == end
            else tensors.get((start, end))
        )
        if arr is None:
            STATS['rejected'] += 1
            return None
        out[name] = arr
    return out, metadata


def _peek_record_metadata(path: Path, key: bytes) -> Optional[dict]:
    """Decrypt only record 0's safetensors JSON metadata (unauthenticated; used solely on the write
    path to inspect candidate delta parents — reads always verify full GCM tags across the chain)."""
    parsed = _read_envelope(path)
    if parsed is None:
        return None
    env, prefix_bytes, data_start = parsed
    records = env.get('records') or []
    if not records:
        return None
    try:
        rel_off, rec_len = int(records[0][0]), int(records[0][1])
    except (IndexError, TypeError, ValueError):
        return None
    ct_len = rec_len - NONCE_BYTES - TAG_BYTES
    if rel_off < 0 or ct_len < 8:
        return None
    try:
        with open(path, 'rb') as f:
            f.seek(data_start + rel_off)
            nonce = f.read(NONCE_BYTES)
            if len(nonce) != NONCE_BYTES:
                return None
            f.seek(data_start + rel_off + NONCE_BYTES + ct_len)
            tag = f.read(TAG_BYTES)
            if len(tag) != TAG_BYTES:
                return None
            f.seek(data_start + rel_off + NONCE_BYTES)
            dec = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
            dec.authenticate_additional_data(prefix_bytes + b':rec:0')
            header_size = int.from_bytes(dec.update(f.read(8)), 'little')
            if header_size <= 0 or header_size > min(64 * 1024 * 1024, ct_len - 8):
                return None
            header = json.loads(dec.update(f.read(header_size)))
            if not isinstance(header, dict):
                return None
            return dict(header.get('__metadata__', {}) or {})
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def _digest(a: mx.array, upto: Optional[int] = None) -> Tuple[str, Optional[str]]:
    """Return `(sha256_whole_hex, sha256_first_upto_bytes_hex_or_None)` streamed over `_pieces(a)`."""
    h = hashlib.sha256()
    done = 0
    head: Optional[str] = h.hexdigest() if upto == 0 else None
    for piece in _pieces(a):
        if head is None and upto is not None and done + piece.nbytes >= upto:
            cut = upto - done
            h.update(piece[:cut])
            head = h.hexdigest()
            h.update(piece[cut:])
        else:
            h.update(piece)
        done += piece.nbytes
    return h.hexdigest(), head


def _grown(shape: Tuple[int, ...], before: Tuple[int, ...]) -> Optional[int]:
    """Return the single axis along which `shape` extends `before` (`0` when identical), else `None`."""
    if len(shape) != len(before):
        return None
    axes = [i for i, (s, b) in enumerate(zip(shape, before)) if s != b]
    if len(axes) > 1 or (axes and shape[axes[0]] < before[axes[0]]):
        return None
    return axes[0] if axes else 0


def _rows(axis: int, start: int, stop: Optional[int] = None):
    """Slice index selecting `[start:stop]` along `axis`."""
    return (slice(None),) * axis + (slice(start, stop),)


def _delta_exact(
    store,
    token_ids: Sequence[int],
    extra_hash: int,
    salt: bytes,
    arrays: dict[str, mx.array],
    metadata: dict[str, str],
) -> Optional[Tuple[int, int]]:
    """Rewrite `exact_v1` `arrays` and `metadata` in place as a delta of the longest stored `exact_v1`
    prefix of `token_ids` (up to `DELTA_DEPTH` deep). Returns `(parent_token_count, parent_locator)`
    when written as a delta, or `None` when written whole."""
    n_cur = len(token_ids)
    parent = None
    depth = 0
    envs = getattr(store, '_quanteeze_envelopes', None)
    with store._index_lock:
        entries = list(store._exact_index.items())
    by_len: dict[int, list[Tuple[int, Path]]] = {}
    for loc_int, p in entries:
        if not p.exists():
            continue
        env = envs.get(p) if isinstance(envs, dict) else None
        if env is None:
            parsed = _read_envelope(p)
            env = parsed[0] if parsed is not None else None
        if env is None or env.get('kind') != 'exact_v1':
            continue
        try:
            tc = int(env.get('token_count', 0))
        except (TypeError, ValueError):
            continue
        if 0 < tc < n_cur:
            by_len.setdefault(tc, []).append((int(loc_int), p))

    if by_len:
        derived = _derive_exact_digests(token_ids, int(extra_hash), by_len.keys(), salt)
        for tc in sorted(derived, reverse=True):
            ekey, expected_loc, _ = derived[tc]
            hit = next((p for loc_int, p in by_len.get(tc, ()) if loc_int == expected_loc), None)
            if hit is not None:
                seen = _peek_record_metadata(hit, ekey)
                try:
                    seen_depth = int((seen or {}).get('qz_depth', 0))
                    seen_digests = json.loads(seen['qz_digests']) if seen and 'qz_digests' in seen else None
                except (TypeError, ValueError):
                    seen_depth, seen_digests = DELTA_DEPTH, None
                if seen_digests is not None and seen_depth < DELTA_DEPTH:
                    parent = (tc, int(expected_loc), seen_digests)
                    depth = seen_depth
                break

    held: dict[str, list] = {}
    plan: dict[str, list] = {}
    for k, a in arrays.items():
        if a.nbytes < DELTA_MIN_BYTES or gather_of(a) is not None:
            continue
        before = parent[2].get(k) if parent is not None else None
        axis = None
        if isinstance(before, list) and len(before) == 3 and before[0] == str(a.dtype):
            axis = _grown(tuple(int(d) for d in a.shape), tuple(int(d) for d in before[1]))
        leading = axis is not None and all(d == 1 for d in a.shape[:axis])
        upto = int(np.prod(before[1], dtype=np.int64)) * a.itemsize if leading else None
        whole, head = _digest(a, upto)
        if axis is not None and not leading:
            head = _digest(a[_rows(axis, 0, int(before[1][axis]))])[0]
        held[k] = [str(a.dtype), [int(d) for d in a.shape], whole]
        if axis is not None and head == before[2]:
            plan[k] = [axis, int(before[1][axis]), before[2], int(a.shape[axis]) == int(before[1][axis])]

    metadata['qz_digests'] = json.dumps(held, separators=(',', ':'))
    metadata['qz_depth'] = str(depth + 1 if plan else 0)
    if not plan or parent is None:
        return None
    for k, (axis, rows, _digest_hex, same) in plan.items():
        if same:
            del arrays[k]
        else:
            arrays[k] = arrays[k][_rows(axis, int(rows))]
    metadata['qz_parent'] = str(parent[0])
    metadata['qz_delta'] = json.dumps(plan, separators=(',', ':'))
    return parent[0], parent[1]


def _parent_paths(store) -> Tuple[set[int], set[Path]]:
    """Return `(parent_locators, parent_paths)` referenced by any existing delta checkpoint in `store`."""
    envs = getattr(store, '_quanteeze_envelopes', None)
    with store._index_lock:
        exact_map = dict(store._exact_index)
    parent_locs: set[int] = set()
    for p in exact_map.values():
        if not p.exists():
            continue
        env = envs.get(p) if isinstance(envs, dict) else None
        if env is None:
            parsed = _read_envelope(p)
            env = parsed[0] if parsed is not None else None
        if env is not None and env.get('parent_locator') is not None:
            try:
                parent_locs.add(int(env['parent_locator']))
            except (TypeError, ValueError):
                pass
    parent_paths = {exact_map[loc] for loc in parent_locs if loc in exact_map and exact_map[loc].exists()}
    return parent_locs, parent_paths


def evict(store, keep: Optional[Path] = None) -> int:
    """Evict least-recently-used shards until `store._disk_bytes <= target`, evicting delta children
    before their parents so retained prefix chains remain whole."""
    if not store._ensure_dir():
        return 0
    if store.max_bytes is None:
        return 0
    with store._in_flight_lock:
        in_flight_hashes = set(store._in_flight.keys())
    with store._index_lock:
        in_flight_paths = {store._index[h][0] for h in in_flight_hashes if h in store._index}
        in_flight_paths.update(
            store._exact_index[h] for h in in_flight_hashes if h in store._exact_index
        )

    candidates: list[dict[str, Any]] = []
    disk_total = 0
    for p in store.dir.glob(f'*{store.SUFFIX}'):
        if not store._is_canonical_store_file(p):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        disk_total += st.st_size
        if p in in_flight_paths or p == keep:
            continue
        store_id = str(p)
        segment_index = 0
        meta = _apc._read_safetensors_metadata(p)
        if meta is not None:
            store_id = meta.get('store_id', store_id)
            try:
                segment_index = int(meta.get('segment_index', '0'))
            except (TypeError, ValueError):
                segment_index = 0
        candidates.append({
            'last_used': st.st_mtime,
            'size': st.st_size,
            'path': p,
            'store_id': store_id,
            'segment_index': segment_index,
        })

    total = max(int(store._disk_bytes), disk_total)
    store._disk_bytes = total
    if total <= store.max_bytes:
        return 0
    target = int(store.max_bytes * store._EVICT_LOW_WATERMARK)

    store_last_used: dict[str, float] = {}
    for c in candidates:
        sid = c['store_id']
        store_last_used[sid] = min(c['last_used'], store_last_used.get(sid, c['last_used']))
    candidates.sort(key=lambda c: (store_last_used[c['store_id']], -int(c['segment_index']), c['last_used']))

    evicted = 0
    while store._disk_bytes > target and candidates:
        _, protected_paths = _parent_paths(store)
        went = 0
        remaining: list[dict[str, Any]] = []
        for idx, c in enumerate(candidates):
            if store._disk_bytes <= target:
                remaining.extend(candidates[idx:])
                break
            p = c['path']
            if p in protected_paths or not p.exists():
                if p.exists():
                    remaining.append(c)
                continue
            try:
                p.unlink()
            except OSError as e:
                logger.warning('APC disk: failed to evict %s: %s', p, e)
                remaining.append(c)
                continue
            store._disk_bytes = max(0, store._disk_bytes - int(c['size']))
            evicted += 1
            went += 1
            store._drop_index_for_path(p)
        if not went:
            break
        candidates = remaining

    if evicted:
        store.evictions += evicted
        logger.info(
            'APC disk: evicted %d shard(s); now %.1f MB / %.1f MB cap',
            evicted,
            store._disk_bytes / 1e6,
            store.max_bytes / 1e6,
        )
    return evicted


def _query_block_chain(
    token_ids: Sequence[int],
    extra_hash: int,
    block_size: int,
    max_blocks: int,
    salt: bytes,
) -> list[Tuple[bytes, int]]:
    bs = int(block_size)
    if bs <= 0 or max_blocks <= 0:
        return []
    n_blocks = min(int(max_blocks), len(token_ids) // bs)
    if n_blocks <= 0:
        return []
    stride = bs * 4
    arr = np.ascontiguousarray(token_ids[: n_blocks * bs], dtype='<i4')
    raw = memoryview(arr).cast('B')
    extra_bytes = int(extra_hash & ((1 << 64) - 1)).to_bytes(8, 'little')
    size_bytes = bs.to_bytes(4, 'little')

    out: list[Tuple[bytes, int]] = []
    parent_key = _root_chain_key(salt)
    for i in range(n_blocks):
        h = _hash()
        h.update(b'quanteeze-apc-blk-key-v1\x00')
        h.update(parent_key)
        h.update(extra_bytes)
        h.update(size_bytes)
        h.update(raw[i * stride : (i + 1) * stride])
        key = h.digest()
        loc_digest = _hash(b'quanteeze-apc-blk-loc-v1\x00' + key).digest()
        loc = int.from_bytes(loc_digest[:8], 'little', signed=True)
        if loc == _apc.SEED_PARENT_HASH:
            loc ^= 1
        _put_chain_key(loc, key)
        out.append((key, loc))
        parent_key = key
    return out


def install(store_cls=None, manager_cls=None) -> bool:
    """Install convergent AES-256-GCM sealing on `mlx_vlm.apc.DiskBlockStore`."""
    store_cls = _apc.DiskBlockStore if store_cls is None else store_cls
    manager_cls = _apc.APCManager if manager_cls is None else manager_cls
    if getattr(store_cls, '_quanteeze_sealed', False):
        return True

    orig_hash_tokens = _apc._hash_tokens
    orig_read_metadata = _apc._read_safetensors_metadata
    orig_read_tensor = _apc._read_safetensors_tensor

    orig_store_init = store_cls.__init__
    orig_ensure_dir = store_cls._ensure_dir
    orig_rebuild_index = store_cls._rebuild_index
    orig_drop_index = store_cls._drop_index_for_path
    orig_clear_state = store_cls._clear_disk_state_after_external_delete
    orig_open_header = store_cls._open_shard_header
    orig_maybe_evict = store_cls._maybe_evict
    orig_exact_bytes = store_cls.exact_cache_bytes
    orig_find_exact = store_cls.find_exact_prefix
    orig_load_exact = store_cls.load_exact_cache
    orig_save_exact = store_cls.save_exact_cache
    orig_write_exact = store_cls._write_exact_cache_snapshot
    orig_save_blocks = store_cls.save_layer_major_blocks
    orig_write_blocks = store_cls._write_layer_major_snapshot
    orig_load_blocks = store_cls.load_layer_major_prefix

    orig_manager_init = manager_cls.__init__

    def _hash_tokens(parent: int, tokens: Tuple[int, ...], extra: int) -> int:
        if not ON[0]:
            return orig_hash_tokens(parent, tokens, extra)
        if parent == _apc.SEED_PARENT_HASH:
            parent_key = _root_chain_key(_precision_salt())
        else:
            parent_key = _get_chain_key(parent)
            if parent_key is None:
                parent_key = _hash(
                    b'quanteeze-apc-fallback-v1\x00'
                    + _precision_salt()
                    + int(parent & ((1 << 64) - 1)).to_bytes(8, 'little')
                ).digest()
        key, loc = _step_chain_key(parent_key, tokens, extra)
        _put_chain_key(loc, key)
        return loc

    def _read_safetensors_metadata(path: Path) -> Optional[dict]:
        if not ON[0]:
            return orig_read_metadata(path)
        parsed = _read_envelope(path)
        if parsed is None:
            return None
        env, _, _ = parsed
        return {
            'store_id': str(env.get('store_id', path)),
            'segment_index': str(int(env.get('segment_index', 0))),
        }

    def _read_safetensors_tensor(path: Path, data_start: int, entry: dict) -> Optional[mx.array]:
        mem_map = getattr(_MEM_SAFES, 'active', None)
        if mem_map is not None and path in mem_map:
            if entry.get(_apc._EMPTY_TENSOR_MARKER) is True:
                try:
                    return mx.zeros(tuple(entry['shape']), dtype=entry['mlx_dtype'])
                except (KeyError, TypeError, ValueError):
                    return None
            bounds = _apc._safetensors_tensor_bounds(entry)
            if bounds is None:
                return None
            start, end, _ = bounds
            raw = mem_map[path][3]
            if isinstance(raw, dict):
                if start == end:
                    return _apc._mlx_array_from_safetensors_bytes(b'', entry)
                return raw.get((start, end))
            if data_start + end > len(raw):
                return None
            return _apc._mlx_array_from_safetensors_bytes(
                memoryview(raw)[data_start + start : data_start + end], entry
            )
        return orig_read_tensor(path, data_start, entry)

    def _store_envelopes(self) -> dict[Path, dict]:
        envs = getattr(self, '_quanteeze_envelopes', None)
        if envs is None:
            envs = {}
            self._quanteeze_envelopes = envs
        return envs

    def _get_store_envelope(self, path: Path) -> Optional[dict]:
        envs = _store_envelopes(self)
        with self._index_lock:
            cached = envs.get(path)
        if cached is not None:
            return cached
        parsed = _read_envelope(path)
        if parsed is None:
            return None
        env, _, _ = parsed
        with self._index_lock:
            envs[path] = env
        return env

    def _ensure_dir(self) -> bool:
        ok = orig_ensure_dir(self)
        if ok and ON[0]:
            try:
                os.chmod(self.dir, 0o700)
            except OSError:
                pass
        return ok

    def __store_init__(self, root: Path, namespace: str = 'default', num_workers: int = 1,
                       max_bytes: Optional[int] = None, *, overrides: Optional[dict] = None):
        self._quanteeze_envelopes = {}
        self._quanteeze_pending_exact = OrderedDict()
        self._quanteeze_pending_blocks = OrderedDict()
        self._quanteeze_block_size = int(
            _apc._setting(overrides, 'block_size', 'APC_BLOCK_SIZE', _apc.DEFAULT_BLOCK_SIZE)
        )
        orig_store_init(
            self,
            root,
            namespace=namespace,
            num_workers=num_workers,
            max_bytes=max_bytes,
            overrides=overrides,
        )

    def __manager_init__(self, *args, **kwargs):
        orig_manager_init(self, *args, **kwargs)
        if self.disk is not None:
            self.disk._quanteeze_block_size = int(self.block_size)

    def _rebuild_index(self) -> int:
        if not ON[0]:
            return orig_rebuild_index(self)
        if not self._ensure_dir():
            return 0
        total = 0
        envs = _store_envelopes(self)
        with self._index_lock:
            self._index.clear()
            self._exact_index.clear()
            envs.clear()
            for p in self.dir.glob(f'*{self.SUFFIX}'):
                if not self._is_canonical_store_file(p):
                    continue
                parsed = _read_envelope(p)
                if parsed is None:
                    STATS['purged_plaintext'] += 1
                    logger.warning('APC disk: dropping unsealed or unreadable shard %s', p)
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue
                env, _, _ = parsed
                kind = env.get('kind')
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if self._is_canonical_exact(p) and kind in ('exact_v1', 'exact_trim_v1'):
                    try:
                        loc = int(env['locator'])
                    except (KeyError, TypeError, ValueError):
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass
                        continue
                    self._exact_index[loc] = p
                    envs[p] = env
                    total += size
                elif self._is_canonical_shard(p) and kind == 'blocks_v1':
                    try:
                        locs = [int(x) for x in env.get('locators', ())]
                    except (TypeError, ValueError):
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass
                        continue
                    if not locs:
                        continue
                    for idx, bh in enumerate(locs):
                        self._index[bh] = (p, idx)
                    envs[p] = env
                    total += size
                else:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
        return total

    def _drop_index_for_path(self, path: Path) -> None:
        orig_drop_index(self, path)
        envs = getattr(self, '_quanteeze_envelopes', None)
        if envs is not None:
            with self._index_lock:
                envs.pop(path, None)

    def _clear_disk_state_after_external_delete(self) -> None:
        orig_clear_state(self)
        envs = getattr(self, '_quanteeze_envelopes', None)
        if envs is not None:
            with self._index_lock:
                envs.clear()

    def _open_shard_header(self, shard_path: Path):
        mem_map = getattr(_MEM_SAFES, 'active', None)
        if mem_map is not None and shard_path in mem_map:
            header, metadata, data_start, _ = mem_map[shard_path]
            return header, metadata, data_start
        if ON[0]:
            return None
        return orig_open_header(self, shard_path)

    def _maybe_evict(self) -> int:
        if not ON[0]:
            return orig_maybe_evict(self)
        return evict(self)

    def exact_cache_bytes(self, cache_hash: int) -> int:
        if not ON[0]:
            return orig_exact_bytes(self, cache_hash)
        with self._index_lock:
            cur_path = self._exact_index.get(int(cache_hash))
        total = 0
        seen: set[Path] = set()
        for _ in range(DELTA_DEPTH + 2):
            if cur_path is None or cur_path in seen:
                break
            seen.add(cur_path)
            try:
                total += cur_path.stat().st_size
            except OSError:
                break
            env = _get_store_envelope(self, cur_path)
            p_loc = env.get('parent_locator') if env is not None else None
            if p_loc is None:
                break
            try:
                p_loc_int = int(p_loc)
            except (TypeError, ValueError):
                break
            with self._index_lock:
                cur_path = self._exact_index.get(p_loc_int)
        return total

    def _chain_on_disk(self, path: Path) -> bool:
        cur_path = path
        for _ in range(DELTA_DEPTH + 2):
            if not cur_path.exists():
                return False
            env = _get_store_envelope(self, cur_path)
            if env is None:
                return False
            p_loc = env.get('parent_locator')
            if p_loc is None:
                return True
            try:
                p_loc_int = int(p_loc)
            except (TypeError, ValueError):
                return False
            with self._index_lock:
                next_path = self._exact_index.get(p_loc_int)
            if next_path is None or next_path == cur_path:
                return False
            cur_path = next_path
        return False

    def _prune_extended_exact(
        self,
        written_path: Path,
        token_ids: Sequence[int],
        extra_hash: int,
        salt: bytes,
    ) -> int:
        n_cur = len(token_ids)
        if n_cur <= 1:
            return 0
        with self._index_lock:
            entries = [
                (int(loc_int), p)
                for loc_int, p in self._exact_index.items()
                if p != written_path
            ]
        if len(entries) < 3:
            return 0

        by_len: dict[int, list[Tuple[int, Path]]] = {}
        for loc_int, p in entries:
            if not p.exists():
                self._drop_index_for_path(p)
                continue
            env = _get_store_envelope(self, p)
            if env is None:
                continue
            try:
                tc = int(env.get('token_count', 0))
            except (TypeError, ValueError):
                continue
            if 0 < tc < n_cur:
                by_len.setdefault(tc, []).append((loc_int, p))
        if len(by_len) < 3:
            return 0

        digests = _derive_exact_digests(token_ids, int(extra_hash), by_len.keys(), salt)
        matched: list[Tuple[int, int, Path]] = []
        for tc in sorted(digests):
            _, expected_loc, _ = digests[tc]
            for loc_int, p in by_len.get(tc, ()):
                if loc_int == expected_loc:
                    matched.append((tc, loc_int, p))
        if len(matched) < 3:
            return 0

        with self._in_flight_lock:
            in_flight = set(self._in_flight)
            with self._index_lock:
                in_flight_paths = {
                    self._exact_index[h] for h in in_flight if h in self._exact_index
                }

        gone = 0
        middle = list(reversed(matched[1:-1]))
        while middle:
            parent_locs, parent_paths = _parent_paths(self)
            went = 0
            remaining: list[Tuple[int, int, Path]] = []
            for tc, loc_int, p in middle:
                if (
                    loc_int in in_flight
                    or p in in_flight_paths
                    or loc_int in parent_locs
                    or p in parent_paths
                ):
                    remaining.append((tc, loc_int, p))
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    remaining.append((tc, loc_int, p))
                    continue
                if size:
                    self._disk_bytes = max(0, self._disk_bytes - size)
                self._drop_index_for_path(p)
                gone += 1
                went += 1
            if not went:
                break
            middle = remaining

        if gone:
            STATS['pruned'] += gone
            logger.info(
                'APC disk: pruned %d intermediate checkpoints (kept %d..%d + %d tokens)',
                gone,
                matched[0][0],
                matched[-1][0],
                n_cur,
            )
        return gone

    def save_exact_cache(
        self,
        cache_hash: int,
        token_ids: Sequence[int],
        extra_hash: int,
        prompt_cache: Sequence[Any],
        *,
        synchronous: bool = False,
    ) -> bool:
        if not ON[0]:
            if over_cap():
                return False
            return orig_save_exact(
                self,
                cache_hash,
                token_ids,
                extra_hash,
                prompt_cache,
                synchronous=synchronous,
            )
        token_tuple = tuple(int(t) for t in token_ids)
        if not token_tuple or not prompt_cache:
            return False
        salt = _precision_salt()
        n_toks = len(token_tuple)
        exact_key, loc_int, loc_hex = _derive_exact_digests(
            token_tuple, int(extra_hash), (n_toks,), salt
        )[n_toks]
        shard_id = f'{self.EXACT_PREFIX}{loc_hex}'
        existing_path = self._shard_path(shard_id)
        if existing_path.exists():
            env = _get_store_envelope(self, existing_path)
            kind = env.get('kind') if env is not None else None
            intact = kind == 'exact_trim_v1' or (kind == 'exact_v1' and _chain_on_disk(self, existing_path))
            if intact:
                try:
                    os.utime(existing_path, None)
                except OSError:
                    pass
                with self._index_lock:
                    self._exact_index[int(loc_int)] = existing_path
                return True
            with self._write_lock:
                _drop_corrupt_shard(self, existing_path)
        else:
            self._drop_index_for_path(existing_path)
        if over_cap():
            return False
        bs = max(1, int(getattr(self, '_quanteeze_block_size', _apc.DEFAULT_BLOCK_SIZE)))
        trimmable = (
            _apc._dense_checkpoint_trimmable(prompt_cache, n_toks)
            and n_toks >= bs
        )
        block_keys: list[bytes] = []
        block_locs: list[int] = []
        if trimmable:
            chain = _query_block_chain(
                token_tuple, int(extra_hash), bs, n_toks // bs, salt
            )
            block_keys = [k for k, _ in chain]
            block_locs = [loc for _, loc in chain]
        snapshot = _apc._DiskExactCacheSnapshot(
            cache_hash=int(loc_int),
            token_ids=token_tuple,
            extra_hash=int(extra_hash),
            prompt_cache=list(prompt_cache),
        )
        seal_info = (
            exact_key,
            int(loc_int),
            loc_hex,
            bs,
            trimmable,
            block_keys,
            block_locs,
            salt,
        )
        with self._index_lock:
            self._quanteeze_pending_exact[id(snapshot)] = seal_info
            while len(self._quanteeze_pending_exact) > 4096:
                self._quanteeze_pending_exact.popitem(last=False)
        return self._enqueue_shard(shard_id, [int(loc_int)], snapshot, synchronous=synchronous)

    def _write_exact_cache_snapshot(self, path: Path, snapshot) -> list[int]:
        with self._index_lock:
            seal_info = self._quanteeze_pending_exact.pop(id(snapshot), None)
        if not ON[0] or seal_info is None:
            if over_cap():
                return []
            res = orig_write_exact(self, path, snapshot)
            _record_wrote(path)
            return res
        if over_cap():
            return []
        if not self._ensure_dir():
            raise OSError(f'APC disk cache directory unavailable: {self.dir}')
        (
            exact_key,
            loc_int,
            loc_hex,
            bs,
            trimmable,
            block_keys,
            block_locs,
            salt,
        ) = seal_info
        store_id = f'{self.EXACT_PREFIX}{loc_hex}'
        try:
            prior_size = path.stat().st_size if path.exists() else 0
        except OSError:
            prior_size = 0

        if not trimmable:
            metadata: dict[str, str] = {
                'layout': 'exact_cache_v1',
                'cache_hash': str(int(loc_int)),
                'extra_hash': str(int(snapshot.extra_hash)),
                'token_ids': ','.join(str(int(t)) for t in snapshot.token_ids),
                'num_entries': str(len(snapshot.prompt_cache)),
                'prefix_trimmable': '0',
                'store_id': store_id,
            }
            arrays: dict[str, mx.array] = {}
            for i, c in enumerate(snapshot.prompt_cache):
                if not self._snapshot_exact_cache_entry(c, f'c{i}', arrays, metadata):
                    raise ValueError(f'unsupported exact-cache entry at index {i}')
            parent_info = _delta_exact(
                self,
                snapshot.token_ids,
                int(snapshot.extra_hash),
                salt,
                arrays,
                metadata,
            )
            envelope_dict: dict[str, Any] = {
                'kind': 'exact_v1',
                'store_id': store_id,
                'locator': int(loc_int),
                'token_count': len(snapshot.token_ids),
                'segment_index': 0,
            }
            if parent_info is not None:
                envelope_dict['parent_token_count'] = int(parent_info[0])
                envelope_dict['parent_locator'] = int(parent_info[1])
            env = _write_sealed_records(
                path,
                envelope_dict,
                [(arrays, metadata, exact_key)],
            )
        else:
            n_full = len(snapshot.token_ids) // bs
            rem = len(snapshot.token_ids) - n_full * bs
            n_entries = len(snapshot.prompt_cache)
            steps = [
                int(getattr(c, 'step', getattr(type(c), 'step', 256)) or 256)
                for c in snapshot.prompt_cache
            ]
            steps_csv = ','.join(str(st) for st in steps)
            records_to_seal: list[Tuple[dict[str, mx.array], dict[str, str], bytes]] = []
            for b_idx in range(n_full):
                s, e = b_idx * bs, (b_idx + 1) * bs
                chunk = snapshot.token_ids[s:e]
                blk_arrays: dict[str, mx.array] = {}
                for i, c in enumerate(snapshot.prompt_cache):
                    blk_arrays[f'c{i}_k'] = c.keys[..., s:e, :]
                    blk_arrays[f'c{i}_v'] = c.values[..., s:e, :]
                blk_meta = {
                    'token_ids': ','.join(str(int(t)) for t in chunk),
                    'extra_hash': str(int(snapshot.extra_hash)),
                    'num_entries': str(n_entries),
                    'steps': steps_csv,
                }
                records_to_seal.append((blk_arrays, blk_meta, block_keys[b_idx]))
            if rem > 0:
                s, e = n_full * bs, len(snapshot.token_ids)
                tail_tokens = snapshot.token_ids[s:e]
                tail_arrays: dict[str, mx.array] = {}
                for i, c in enumerate(snapshot.prompt_cache):
                    tail_arrays[f'c{i}_k'] = c.keys[..., s:e, :]
                    tail_arrays[f'c{i}_v'] = c.values[..., s:e, :]
                tail_meta = {
                    'token_ids': ','.join(str(int(t)) for t in tail_tokens),
                    'extra_hash': str(int(snapshot.extra_hash)),
                    'num_entries': str(n_entries),
                    'steps': steps_csv,
                }
                records_to_seal.append((tail_arrays, tail_meta, exact_key))
            env = _write_sealed_records(
                path,
                {
                    'kind': 'exact_trim_v1',
                    'store_id': store_id,
                    'locator': int(loc_int),
                    'token_count': len(snapshot.token_ids),
                    'block_size': int(bs),
                    'block_locators': [int(x) for x in block_locs],
                    'has_tail': bool(rem > 0),
                    'segment_index': 0,
                },
                records_to_seal,
            )

        written_size = _record_wrote(path)
        if written_size or prior_size:
            self._disk_bytes = max(0, self._disk_bytes - prior_size + written_size)
        with self._index_lock:
            self._exact_index[int(loc_int)] = path
            _store_envelopes(self)[path] = env
        STATS['writes'] += 1
        _prune_extended_exact(self, path, snapshot.token_ids, int(snapshot.extra_hash), salt)
        evict(self, keep=path)
        return [int(loc_int)]

    def find_exact_prefix(
        self,
        token_ids: Sequence[int],
        *,
        extra_hash: int = 0,
        max_prefix_tokens: Optional[int] = None,
        min_prefix_tokens: int = 0,
        block_size: int = _apc.DEFAULT_BLOCK_SIZE,
    ) -> Optional[Tuple[int, int]]:
        if not ON[0]:
            return orig_find_exact(
                self,
                token_ids,
                extra_hash=extra_hash,
                max_prefix_tokens=max_prefix_tokens,
                min_prefix_tokens=min_prefix_tokens,
                block_size=block_size,
            )
        token_tuple = tuple(int(t) for t in token_ids)
        max_len = len(token_tuple) - 1
        if max_prefix_tokens is not None and max_prefix_tokens > 0:
            max_len = min(max_len, int(max_prefix_tokens))
        if max_len <= min_prefix_tokens:
            return None

        with self._index_lock:
            entries = list(self._exact_index.items())
        if not entries:
            return None

        salt = _precision_salt()
        indexed_envs: list[Tuple[int, Path, dict]] = []
        by_loc: dict[int, Tuple[Path, dict]] = {}
        candidate_lengths: set[int] = set()
        trim_block_sizes: set[int] = set()
        for loc_int, path in entries:
            if not path.exists():
                self._drop_index_for_path(path)
                continue
            env = _get_store_envelope(self, path)
            if env is None:
                self._drop_index_for_path(path)
                continue
            loc_i = int(loc_int)
            indexed_envs.append((loc_i, path, env))
            by_loc[loc_i] = (path, env)
            if env.get('kind') == 'exact_trim_v1':
                try:
                    tc_trim = int(env.get('token_count', 0))
                except (TypeError, ValueError):
                    tc_trim = 0
                if min_prefix_tokens < tc_trim <= max_len:
                    candidate_lengths.add(tc_trim)
                try:
                    ebs = int(env.get('block_size', block_size))
                except (TypeError, ValueError):
                    ebs = int(block_size)
                if ebs > 0 and max_len >= ebs:
                    trim_block_sizes.add(ebs)

        exact_v1_chains: dict[int, list[Tuple[int, Path, int]]] = {}
        for loc_i, path, env in indexed_envs:
            if env.get('kind') != 'exact_v1':
                continue
            try:
                tc = int(env.get('token_count', 0))
            except (TypeError, ValueError):
                continue
            if not (min_prefix_tokens < tc <= max_len):
                continue
            chain: list[Tuple[int, Path, int]] = [(loc_i, path, tc)]
            cur_env, cur_tc = env, tc
            ok = True
            for _ in range(DELTA_DEPTH + 1):
                p_loc_raw = cur_env.get('parent_locator')
                if p_loc_raw is None:
                    break
                try:
                    p_loc = int(p_loc_raw)
                    p_tc = int(cur_env['parent_token_count'])
                except (KeyError, TypeError, ValueError):
                    ok = False
                    break
                if not 0 < p_tc < cur_tc or p_loc not in by_loc:
                    ok = False
                    break
                p_path, p_env = by_loc[p_loc]
                try:
                    actual_tc = int(p_env.get('token_count', 0))
                except (TypeError, ValueError):
                    actual_tc = 0
                if p_env.get('kind') != 'exact_v1' or actual_tc != p_tc or not p_path.exists():
                    ok = False
                    break
                chain.append((p_loc, p_path, p_tc))
                cur_env, cur_tc = p_env, p_tc
            else:
                ok = False
            if ok:
                chain.reverse()
                exact_v1_chains[loc_i] = chain
                for _c_loc, _c_path, c_tc in chain:
                    candidate_lengths.add(c_tc)

        exact_derived = _derive_exact_digests(
            token_tuple, int(extra_hash), candidate_lengths, salt
        )

        chains_by_bs: dict[int, list[Tuple[bytes, int]]] = {}
        for ebs in trim_block_sizes:
            chains_by_bs[ebs] = _query_block_chain(
                token_tuple, int(extra_hash), ebs, max_len // ebs, salt
            )

        best: Optional[Tuple[int, int]] = None
        best_grant: Optional[tuple] = None

        for loc_int, _path, env in indexed_envs:
            kind = env.get('kind')
            try:
                tc = int(env.get('token_count', 0))
            except (TypeError, ValueError):
                continue

            if tc in exact_derived:
                ekey, dloc, _ = exact_derived[tc]
                if dloc == loc_int and (best is None or tc > best[1]):
                    if kind == 'exact_v1':
                        v1_chain = exact_v1_chains.get(loc_int)
                        if v1_chain is not None and all(
                            c_tc in exact_derived and exact_derived[c_tc][1] == c_loc
                            for c_loc, _c_path, c_tc in v1_chain
                        ):
                            best = (loc_int, tc)
                            best_grant = (
                                'exact_v1',
                                [(c_path, exact_derived[c_tc][0]) for _c_loc, c_path, c_tc in v1_chain],
                            )
                            continue
                    if kind == 'exact_trim_v1':
                        ebs = int(env.get('block_size', block_size))
                        chain = chains_by_bs.get(ebs)
                        if chain is None:
                            chain = _query_block_chain(
                                token_tuple, int(extra_hash), ebs, tc // ebs, salt
                            )
                            chains_by_bs[ebs] = chain
                        n_full = tc // ebs
                        stored_locs = [int(x) for x in env.get('block_locators', ())]
                        if len(stored_locs) == n_full and [loc for _, loc in chain[:n_full]] == stored_locs:
                            best = (loc_int, tc)
                            best_grant = ('exact_trim_full', [k for k, _ in chain[:n_full]], ekey)
                            continue

            if kind == 'exact_trim_v1':
                ebs = int(env.get('block_size', block_size))
                chain = chains_by_bs.get(ebs, ())
                stored_locs = [int(x) for x in env.get('block_locators', ())]
                limit = min(len(chain), len(stored_locs), max_len // ebs)
                matched_blocks = 0
                while matched_blocks < limit and chain[matched_blocks][1] == stored_locs[matched_blocks]:
                    matched_blocks += 1
                trimmed_len = matched_blocks * ebs
                if trimmed_len > min_prefix_tokens and (best is None or trimmed_len > best[1]):
                    best = (loc_int, trimmed_len)
                    best_grant = ('exact_trim_blocks', [k for k, _ in chain[:matched_blocks]])

        if best is not None and best_grant is not None:
            _put_unlock(best, best_grant)
        return best

    def _drop_corrupt_shard(self, path: Path) -> None:
        try:
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            self._disk_bytes = max(0, self._disk_bytes - size)
        except OSError:
            pass
        self._drop_index_for_path(path)

    def _rebuilt_exact(
        self,
        chain_links: Sequence[Tuple[Path, bytes]],
    ) -> Optional[Tuple[dict, dict, int, dict[Tuple[int, int], mx.array]]]:
        entries: dict = {}
        metadata: dict = {}
        st_data_start = 0
        tensors: dict[Tuple[int, int], mx.array] = {}
        parts: dict[str, Tuple[int, str, list[mx.array]]] = {}

        for idx, (c_path, c_key) in enumerate(chain_links):
            c_parsed = _read_envelope(c_path)
            if c_parsed is None:
                _drop_corrupt_shard(self, c_path)
                return None
            c_env, c_prefix, c_start = c_parsed
            c_records = c_env.get('records', [])
            if c_env.get('kind') != 'exact_v1' or len(c_records) != 1:
                _drop_corrupt_shard(self, c_path)
                return None
            try:
                with open(c_path, 'rb') as f:
                    got = _decrypt_record_tensors(f, c_prefix, c_start, c_records, 0, c_key)
            except OSError:
                return None
            if got is None:
                _drop_corrupt_shard(self, c_path)
                return None
            c_entries, c_meta, c_st_start, c_tensors = got
            if idx == 0:
                if 'qz_delta' in c_meta:
                    _drop_corrupt_shard(self, c_path)
                    return None
                entries, metadata, st_data_start, tensors = c_entries, c_meta, c_st_start, c_tensors
                continue
            try:
                plan = json.loads(c_meta['qz_delta'])
                recorded = json.loads(metadata.get('qz_digests', '{}'))
            except (KeyError, TypeError, ValueError):
                for stale_p, _ in chain_links[idx:]:
                    _drop_corrupt_shard(self, stale_p)
                return None
            next_parts: dict[str, Tuple[int, str, list[mx.array]]] = {}
            for k, spec in plan.items():
                try:
                    axis, rows, digest_hex, same = int(spec[0]), int(spec[1]), spec[2], bool(spec[3])
                except (IndexError, TypeError, ValueError):
                    for stale_p, _ in chain_links[idx:]:
                        _drop_corrupt_shard(self, stale_p)
                    return None
                if (recorded.get(k) or [None, None, None])[2] != digest_hex:
                    for stale_p, _ in chain_links[idx:]:
                        _drop_corrupt_shard(self, stale_p)
                    return None
                if k in parts:
                    p_axis, dtype, before = parts[k]
                    if p_axis != axis:
                        for stale_p, _ in chain_links[idx:]:
                            _drop_corrupt_shard(self, stale_p)
                        return None
                else:
                    bounds = _apc._safetensors_tensor_bounds(entries.get(k) or {})
                    whole = None if bounds is None else tensors.get(bounds[:2])
                    if whole is None:
                        for stale_p, _ in chain_links[idx:]:
                            _drop_corrupt_shard(self, stale_p)
                        return None
                    dtype, before = entries[k]['dtype'], [whole]
                if sum(int(x.shape[axis]) for x in before) != rows:
                    for stale_p, _ in chain_links[idx:]:
                        _drop_corrupt_shard(self, stale_p)
                    return None
                if not same:
                    bounds = _apc._safetensors_tensor_bounds(c_entries.get(k) or {})
                    tail = None if bounds is None else c_tensors.get(bounds[:2])
                    if tail is None:
                        for stale_p, _ in chain_links[idx:]:
                            _drop_corrupt_shard(self, stale_p)
                        return None
                    before = before + [tail]
                next_parts[k] = (axis, dtype, before)
            entries, metadata, st_data_start, tensors, parts = (
                c_entries,
                c_meta,
                c_st_start,
                c_tensors,
                next_parts,
            )

        at = _SYNTHETIC
        for k, (axis, dtype, pieces) in parts.items():
            whole = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=axis)
            nbytes = int(whole.nbytes)
            entries[k] = {
                'dtype': dtype,
                'shape': [int(d) for d in whole.shape],
                'data_offsets': [at, at + nbytes],
            }
            tensors[(at, at + nbytes)] = whole
            at += nbytes
        return entries, metadata, st_data_start, tensors

    def load_exact_cache(
        self,
        cache_hash: int,
        *,
        wait_in_flight_ms: float = 0.0,
        min_capacity_tokens: Optional[int] = None,
        prefix_len: Optional[int] = None,
    ) -> Optional[Tuple[Tuple[int, ...], int, list[Any]]]:
        if not ON[0]:
            return orig_load_exact(
                self,
                cache_hash,
                wait_in_flight_ms=wait_in_flight_ms,
                min_capacity_tokens=min_capacity_tokens,
                prefix_len=prefix_len,
            )
        with self._index_lock:
            path = self._exact_index.get(int(cache_hash))
        if path is None:
            if wait_in_flight_ms > 0:
                with self._in_flight_lock:
                    ev = self._in_flight.get(int(cache_hash))
                if ev is not None and ev.wait(wait_in_flight_ms / 1000.0):
                    with self._index_lock:
                        path = self._exact_index.get(int(cache_hash))
            if path is None:
                return None

        parsed = _read_envelope(path)
        if parsed is None:
            _drop_corrupt_shard(self, path)
            return None
        env, prefix_bytes, data_start = parsed
        try:
            total_tokens = int(env['token_count'])
        except (KeyError, TypeError, ValueError):
            _drop_corrupt_shard(self, path)
            return None

        target_len = total_tokens if prefix_len is None else int(prefix_len)
        grant = _get_unlock((int(cache_hash), target_len))
        if grant is None:
            return None

        records = env.get('records', [])
        mode = grant[0]

        if mode == 'exact_v1':
            if env.get('kind') != 'exact_v1' or len(records) != 1:
                return None
            chain_links = (
                [(path, bytes(grant[1]))]
                if isinstance(grant[1], (bytes, bytearray))
                else list(grant[1])
            )
            if not chain_links or chain_links[-1][0] != path:
                return None
            rebuilt = _rebuilt_exact(self, chain_links)
            if rebuilt is None:
                return None
            header, metadata, st_data_start, tensors = rebuilt
            active = getattr(_MEM_SAFES, 'active', None)
            if active is None:
                active = {}
                _MEM_SAFES.active = active
            active[path] = (header, metadata, st_data_start, tensors)
            try:
                loaded = self._load_exact_cache_file(
                    path, min_capacity_tokens=min_capacity_tokens, prefix_len=prefix_len
                )
            finally:
                active.pop(path, None)
            if loaded is not None:
                for c_path, _ in chain_links[:-1]:
                    try:
                        os.utime(c_path, None)
                    except OSError:
                        pass
                STATS['hits'] += 1
            return loaded

        if mode in ('exact_trim_full', 'exact_trim_blocks'):
            if env.get('kind') != 'exact_trim_v1':
                return None
            from mlx_vlm.models import cache as lm_cache

            block_keys: list[bytes] = grant[1]
            tail_key: Optional[bytes] = grant[2] if mode == 'exact_trim_full' else None
            has_tail = bool(env.get('has_tail', False))
            needed_keys = list(block_keys) + ([tail_key] if (mode == 'exact_trim_full' and has_tail and tail_key is not None) else [])
            if len(needed_keys) > len(records) or not needed_keys:
                return None

            decrypted_parts: list[Tuple[dict[str, mx.array], dict]] = []
            try:
                with open(path, 'rb') as f:
                    for r_idx, r_key in enumerate(needed_keys):
                        part = _decrypt_record_arrays(f, prefix_bytes, data_start, records, r_idx, r_key)
                        if part is None:
                            _drop_corrupt_shard(self, path)
                            return None
                        decrypted_parts.append(part)
            except OSError:
                return None

            tokens_acc: list[int] = []
            extra_ref: Optional[int] = None
            n_entries_ref: Optional[int] = None
            steps_ref: Optional[list[int]] = None
            for _arrays, meta in decrypted_parts:
                try:
                    part_tokens = [int(x) for x in meta.get('token_ids', '').split(',') if x]
                    part_extra = int(meta.get('extra_hash', '0'))
                    part_entries = int(meta.get('num_entries', '0'))
                    part_steps = [int(x) for x in meta.get('steps', '').split(',') if x]
                except (TypeError, ValueError):
                    return None
                if extra_ref is None:
                    extra_ref = part_extra
                    n_entries_ref = part_entries
                    steps_ref = part_steps
                elif part_extra != extra_ref or part_entries != n_entries_ref:
                    return None
                tokens_acc.extend(part_tokens)

            if extra_ref is None or not n_entries_ref or len(tokens_acc) != target_len:
                return None

            prompt_cache: list[Any] = []
            eval_targets: list[mx.array] = []
            for i in range(n_entries_ref):
                k_slices = [arrays.get(f'c{i}_k') for arrays, _ in decrypted_parts]
                v_slices = [arrays.get(f'c{i}_v') for arrays, _ in decrypted_parts]
                if any(x is None for x in k_slices) or any(x is None for x in v_slices):
                    return None
                k = k_slices[0] if len(k_slices) == 1 else mx.concatenate(k_slices, axis=2)
                v = v_slices[0] if len(v_slices) == 1 else mx.concatenate(v_slices, axis=2)
                step = steps_ref[i] if steps_ref and i < len(steps_ref) else 256
                k, v = _apc._pad_kv_for_capacity(
                    k,
                    v,
                    offset=target_len,
                    min_capacity_tokens=min_capacity_tokens,
                    step=step,
                )
                c = lm_cache.KVCache()
                c.keys = k
                c.values = v
                c.offset = target_len
                c.step = step
                prompt_cache.append(c)
                eval_targets.extend([k, v])

            if eval_targets:
                mx.eval(eval_targets)
            try:
                os.utime(path, None)
            except OSError:
                pass
            STATS['hits'] += 1
            return tuple(tokens_acc), int(extra_ref), prompt_cache

        return None

    def save_layer_major_blocks(
        self,
        blocks: list[Any],
        layer_keys: Sequence[mx.array],
        layer_values: Sequence[mx.array],
        block_size: int,
    ) -> None:
        if not ON[0]:
            if over_cap():
                return None
            return orig_save_blocks(self, blocks, layer_keys, layer_values, block_size)
        if not blocks or not layer_keys or not layer_values or over_cap():
            return None
        salt = _precision_salt()
        for b in blocks:
            key = _get_chain_key(int(b.block_hash))
            if key is None:
                if int(b.parent_hash) == _apc.SEED_PARENT_HASH:
                    p_key = _root_chain_key(salt)
                else:
                    p_key = _get_chain_key(int(b.parent_hash))
                if p_key is None:
                    p_key = _hash(
                        b'quanteeze-apc-fallback-v1\x00'
                        + salt
                        + int(b.parent_hash & ((1 << 64) - 1)).to_bytes(8, 'little')
                    ).digest()
                key, loc = _step_chain_key(p_key, b.token_ids, int(b.extra_hash))
                _put_chain_key(int(b.block_hash), key)
                if loc != int(b.block_hash):
                    _put_chain_key(loc, key)
            with self._index_lock:
                self._quanteeze_pending_blocks[int(b.block_hash)] = key
                while len(self._quanteeze_pending_blocks) > CHAIN_KEY_CAP:
                    self._quanteeze_pending_blocks.popitem(last=False)
        return orig_save_blocks(self, blocks, layer_keys, layer_values, block_size)

    def _write_layer_major_snapshot(self, path: Path, snapshot) -> list[int]:
        if not ON[0]:
            if over_cap():
                return []
            res = orig_write_blocks(self, path, snapshot)
            _record_wrote(path)
            return res
        blocks = snapshot.blocks
        if not blocks or over_cap():
            return []
        if len(snapshot.layer_keys) != len(snapshot.layer_values):
            raise ValueError('layer-major disk snapshot has mismatched K/V layers')
        if not self._ensure_dir():
            raise OSError(f'APC disk cache directory unavailable: {self.dir}')

        bs = int(snapshot.block_size)
        num_layers = len(snapshot.layer_keys)
        records_to_seal: list[Tuple[dict[str, mx.array], dict[str, str], bytes]] = []
        locators: list[int] = []
        for b in blocks:
            src_idx = int(b.source_block_idx)
            s, e = src_idx * bs, (src_idx + 1) * bs
            blk_arrays: dict[str, mx.array] = {}
            for l_idx, (k_src, v_src) in enumerate(zip(snapshot.layer_keys, snapshot.layer_values)):
                blk_arrays[f'k{l_idx}'] = k_src[..., s:e, :]
                blk_arrays[f'v{l_idx}'] = v_src[..., s:e, :]
            blk_meta = {
                'block_hash': str(int(b.block_hash)),
                'parent_hash': str(int(b.parent_hash)),
                'extra_hash': str(int(b.extra_hash)),
                'token_ids': ','.join(str(int(t)) for t in b.token_ids),
                'num_layers': str(num_layers),
                'block_size': str(bs),
            }
            with self._index_lock:
                key = self._quanteeze_pending_blocks.pop(int(b.block_hash), None)
            if key is None:
                key = _get_chain_key(int(b.block_hash))
            if key is None:
                raise ValueError('missing convergent key for APC disk block')
            records_to_seal.append((blk_arrays, blk_meta, key))
            locators.append(int(b.block_hash))

        env = _write_sealed_records(
            path,
            {
                'kind': 'blocks_v1',
                'store_id': str(snapshot.store_id),
                'segment_index': int(snapshot.segment_index),
                'segment_count': int(snapshot.segment_count),
                'locators': locators,
            },
            records_to_seal,
        )
        written_size = _record_wrote(path)
        if written_size:
            self._disk_bytes += written_size
        with self._index_lock:
            for idx, loc in enumerate(locators):
                self._index[loc] = (path, idx)
            _store_envelopes(self)[path] = env
        STATS['writes'] += 1
        self._maybe_evict()
        return locators

    def load_layer_major_prefix(
        self,
        block_hashes: Sequence[int],
        *,
        preserve_capacity: bool = True,
    ) -> Optional[Tuple[list[mx.array], list[mx.array], list[dict]]]:
        if not ON[0]:
            return orig_load_blocks(self, block_hashes, preserve_capacity=preserve_capacity)
        if not block_hashes:
            return None

        plan: list[Tuple[Path, int, bytes]] = []
        for bh in block_hashes:
            with self._index_lock:
                entry = self._index.get(int(bh))
            key = _get_chain_key(int(bh))
            if entry is None or key is None:
                return None
            plan.append((entry[0], entry[1], key))

        grouped: list[Tuple[Path, list[Tuple[int, bytes]]]] = []
        for shard_path, block_idx, key in plan:
            if not grouped or grouped[-1][0] != shard_path:
                grouped.append((shard_path, [(block_idx, key)]))
            else:
                grouped[-1][1].append((block_idx, key))

        decrypted_blocks: list[Tuple[dict[str, mx.array], dict]] = []
        touched_paths: set[Path] = set()
        for shard_path, items in grouped:
            parsed = _read_envelope(shard_path)
            if parsed is None:
                _drop_corrupt_shard(self, shard_path)
                return None
            env, prefix_bytes, data_start = parsed
            if env.get('kind') != 'blocks_v1':
                return None
            records = env.get('records', [])
            try:
                with open(shard_path, 'rb') as f:
                    for block_idx, key in items:
                        part = _decrypt_record_arrays(f, prefix_bytes, data_start, records, block_idx, key)
                        if part is None:
                            _drop_corrupt_shard(self, shard_path)
                            return None
                        decrypted_blocks.append(part)
            except OSError:
                return None
            touched_paths.add(shard_path)

        num_layers: Optional[int] = None
        block_size_ref: Optional[int] = None
        metadatas: list[dict] = []
        for arrays, meta in decrypted_blocks:
            try:
                nl = int(meta.get('num_layers', '0'))
                bs = int(meta.get('block_size', '0'))
            except (TypeError, ValueError):
                return None
            if nl <= 0 or bs <= 0:
                return None
            if num_layers is None:
                num_layers = nl
                block_size_ref = bs
            elif nl != num_layers or bs != block_size_ref:
                return None
            metadatas.append(
                {
                    'block_hash': meta.get('block_hash', '0'),
                    'parent_hash': meta.get('parent_hash', '0'),
                    'extra_hash': meta.get('extra_hash', '0'),
                    'token_ids': meta.get('token_ids', ''),
                }
            )

        if num_layers is None:
            return None

        keys: list[mx.array] = []
        values: list[mx.array] = []
        for l_idx in range(num_layers):
            k_parts = [arrays.get(f'k{l_idx}') for arrays, _ in decrypted_blocks]
            v_parts = [arrays.get(f'v{l_idx}') for arrays, _ in decrypted_blocks]
            if any(x is None for x in k_parts) or any(x is None for x in v_parts):
                return None
            keys.append(k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2))
            values.append(v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2))

        if preserve_capacity:
            keys, values = store_cls._pad_layer_major_arrays(keys, values)

        mx.eval(keys + values)
        for p in touched_paths:
            try:
                os.utime(p, None)
            except OSError:
                pass
        STATS['hits'] += 1
        return keys, values, metadatas

    _apc._hash_tokens = _hash_tokens
    _apc._read_safetensors_metadata = _read_safetensors_metadata
    _apc._read_safetensors_tensor = _read_safetensors_tensor

    store_cls.__init__ = __store_init__
    store_cls._ensure_dir = _ensure_dir
    store_cls._rebuild_index = _rebuild_index
    store_cls._drop_index_for_path = _drop_index_for_path
    store_cls._clear_disk_state_after_external_delete = _clear_disk_state_after_external_delete
    store_cls._open_shard_header = _open_shard_header
    store_cls._maybe_evict = _maybe_evict
    store_cls.exact_cache_bytes = exact_cache_bytes
    store_cls.find_exact_prefix = find_exact_prefix
    store_cls.load_exact_cache = load_exact_cache
    store_cls.save_exact_cache = save_exact_cache
    store_cls._write_exact_cache_snapshot = _write_exact_cache_snapshot
    store_cls.save_layer_major_blocks = save_layer_major_blocks
    store_cls._write_layer_major_snapshot = _write_layer_major_snapshot
    store_cls.load_layer_major_prefix = load_layer_major_prefix

    manager_cls.__init__ = __manager_init__
    store_cls._quanteeze_sealed = True
    return True
