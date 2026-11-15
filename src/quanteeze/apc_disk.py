# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Convergent AES-256-GCM sealing for mlx-vlm's on-disk Prefix Cache (APC).

mlx-vlm's `DiskBlockStore` writes `.safetensors` files whose plaintext JSON headers carry
the prompt's `token_ids` and whose data sections hold unencrypted KV / recurrent state
tensors.  `install()` wraps `mlx_vlm.apc` so that when `QUANTEEZE_APC_ENCRYPT` is `on`:

1. Every disk shard (`shard_<32-hex>.safetensors`) and exact checkpoint
   (`exact_<32-hex>.safetensors`) is serialized in memory (`io.BytesIO`, never writing
   plaintext bytes to temporary files) and sealed with `AES-256-GCM` under a 256-bit
   convergent key derived from the prompt's own token chain, `extra_hash`, and the active
   quantization / precision configuration (`QUANTEEZE_BITS`, and
   `serve._precision()`).
2. Block shards and trimmable dense checkpoints seal each `block_size` slice under its
   own chained prefix key `K_i`, so a request sharing only an earlier prefix derives keys
   for and decrypts only that shared prefix without being able to decrypt later blocks.
3. On-disk filenames and outer envelopes store only one-way SHA-256 locators and record
   offsets (authenticated as GCM Associated Data), exposing zero token IDs, tensor shapes,
   or unencrypted KV values at rest.  Unsealed legacy files or tampered ciphertexts are
   rejected cleanly on startup or lookup without crashing the server.
4. Multi-length exact-prefix lookup runs in a single incremental SHA-256 pass (`h.copy()`),
   and saving a longer conversation checkpoint prunes intermediate extended checkpoints
   while preserving the shortest shared prefix and the previous longest checkpoint.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from mlx_vlm import apc as _apc

from . import variables

logger = logging.getLogger(__name__)

MAGIC = b'QZAPC01\x00'
NONCE_BYTES = 12
TAG_BYTES = 16
MAX_ENVELOPE_BYTES = 4 * 1024 * 1024
CHAIN_KEY_CAP = 65536
UNLOCK_CAP = 512
SENTINEL_TENSOR = '__qz_empty_sentinel__'

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
    return (
        f"{STATS['writes']} sealed writes, {STATS['hits']} decrypted hits, "
        f"{STATS['rejected']} rejected, {STATS['pruned']} pruned, "
        f"{STATS['purged_plaintext']} unsealed purged"
    )


def _precision_salt() -> bytes:
    own = PRECISION[0]() if PRECISION[0] is not None else ''
    bits = (variables.get('QUANTEEZE_BITS') or 'off').strip().lower()
    return f'{own}|bits={bits}'.encode('utf-8')


def _root_chain_key(salt: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(b'quanteeze-apc-root-v1\x00')
    h.update(len(salt).to_bytes(4, 'little'))
    h.update(salt)
    return h.digest()


def _step_chain_key(parent_key: bytes, tokens: Sequence[int], extra_hash: int) -> Tuple[bytes, int]:
    h = hashlib.sha256()
    h.update(b'quanteeze-apc-blk-key-v1\x00')
    h.update(parent_key)
    h.update(int(extra_hash & ((1 << 64) - 1)).to_bytes(8, 'little'))
    arr = np.asarray(tokens, dtype='<i4')
    h.update(int(arr.size).to_bytes(4, 'little'))
    h.update(arr.tobytes())
    key = h.digest()
    loc_digest = hashlib.sha256(b'quanteeze-apc-blk-loc-v1\x00' + key).digest()
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
    d = hashlib.sha256(b'quanteeze-apc-exact-loc-v1\x00' + key).digest()
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

    h = hashlib.sha256()
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
        h = hashlib.sha256()
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
            f.write(prefix_bytes)
            for blob in blobs:
                f.write(blob)
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
        h = hashlib.sha256()
        h.update(b'quanteeze-apc-blk-key-v1\x00')
        h.update(parent_key)
        h.update(extra_bytes)
        h.update(size_bytes)
        h.update(raw[i * stride : (i + 1) * stride])
        key = h.digest()
        loc_digest = hashlib.sha256(b'quanteeze-apc-blk-loc-v1\x00' + key).digest()
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
                parent_key = hashlib.sha256(
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
        for _tc, loc_int, p in matched[1:-1]:
            if loc_int in in_flight or p in in_flight_paths:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            try:
                p.unlink(missing_ok=True)
            except OSError:
                continue
            if size:
                self._disk_bytes = max(0, self._disk_bytes - size)
            self._drop_index_for_path(p)
            gone += 1

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
            if env is not None and env.get('kind') in ('exact_v1', 'exact_trim_v1'):
                try:
                    os.utime(existing_path, None)
                except OSError:
                    pass
                with self._index_lock:
                    self._exact_index[int(loc_int)] = existing_path
                return True
            with self._write_lock:
                _drop_corrupt_shard(self, existing_path)
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
            return orig_write_exact(self, path, snapshot)
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
            plain = _safetensors_dumps(arrays, metadata)
            env = _write_sealed_file(
                path,
                {
                    'kind': 'exact_v1',
                    'store_id': store_id,
                    'locator': int(loc_int),
                    'token_count': len(snapshot.token_ids),
                    'segment_index': 0,
                },
                [(plain, exact_key)],
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
            records_to_seal: list[Tuple[bytes, bytes]] = []
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
                records_to_seal.append((_safetensors_dumps(blk_arrays, blk_meta), block_keys[b_idx]))
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
                records_to_seal.append((_safetensors_dumps(tail_arrays, tail_meta), exact_key))
            env = _write_sealed_file(
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

        try:
            self._disk_bytes += path.stat().st_size
        except OSError:
            pass
        with self._index_lock:
            self._exact_index[int(loc_int)] = path
            _store_envelopes(self)[path] = env
        STATS['writes'] += 1
        _prune_extended_exact(self, path, snapshot.token_ids, int(snapshot.extra_hash), salt)
        self._maybe_evict()
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
        candidate_lengths: set[int] = set()
        trim_block_sizes: set[int] = set()
        for loc_int, path in entries:
            env = _get_store_envelope(self, path)
            if env is None:
                self._drop_index_for_path(path)
                continue
            indexed_envs.append((int(loc_int), path, env))
            try:
                tc = int(env.get('token_count', 0))
            except (TypeError, ValueError):
                continue
            if min_prefix_tokens < tc <= max_len:
                candidate_lengths.add(tc)
            if env.get('kind') == 'exact_trim_v1':
                try:
                    ebs = int(env.get('block_size', block_size))
                except (TypeError, ValueError):
                    ebs = int(block_size)
                if ebs > 0 and max_len >= ebs:
                    trim_block_sizes.add(ebs)

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
                        best = (loc_int, tc)
                        best_grant = ('exact_v1', ekey)
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
            try:
                with open(path, 'rb') as f:
                    plain = _decrypt_record(f, prefix_bytes, data_start, records, 0, grant[1])
            except OSError:
                return None
            if plain is None:
                _drop_corrupt_shard(self, path)
                return None
            st_parsed = _parse_safetensors_bytes(plain)
            if st_parsed is None:
                _drop_corrupt_shard(self, path)
                return None
            header, metadata, st_data_start = st_parsed
            active = getattr(_MEM_SAFES, 'active', None)
            if active is None:
                active = {}
                _MEM_SAFES.active = active
            active[path] = (header, metadata, st_data_start, plain)
            try:
                loaded = self._load_exact_cache_file(
                    path, min_capacity_tokens=min_capacity_tokens, prefix_len=prefix_len
                )
            finally:
                active.pop(path, None)
            if loaded is not None:
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
                        plain = _decrypt_record(f, prefix_bytes, data_start, records, r_idx, r_key)
                        if plain is None:
                            _drop_corrupt_shard(self, path)
                            return None
                        part = _load_arrays_from_safetensors_bytes(plain)
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
            return orig_save_blocks(self, blocks, layer_keys, layer_values, block_size)
        if not blocks or not layer_keys or not layer_values:
            return
        salt = _precision_salt()
        for b in blocks:
            key = _get_chain_key(int(b.block_hash))
            if key is None:
                if int(b.parent_hash) == _apc.SEED_PARENT_HASH:
                    p_key = _root_chain_key(salt)
                else:
                    p_key = _get_chain_key(int(b.parent_hash))
                if p_key is None:
                    p_key = hashlib.sha256(
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
            return orig_write_blocks(self, path, snapshot)
        blocks = snapshot.blocks
        if not blocks:
            return []
        if len(snapshot.layer_keys) != len(snapshot.layer_values):
            raise ValueError('layer-major disk snapshot has mismatched K/V layers')
        if not self._ensure_dir():
            raise OSError(f'APC disk cache directory unavailable: {self.dir}')

        bs = int(snapshot.block_size)
        num_layers = len(snapshot.layer_keys)
        records_to_seal: list[Tuple[bytes, bytes]] = []
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
            records_to_seal.append((_safetensors_dumps(blk_arrays, blk_meta), key))
            locators.append(int(b.block_hash))

        env = _write_sealed_file(
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
        try:
            self._disk_bytes += path.stat().st_size
        except OSError:
            pass
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
                        plain = _decrypt_record(f, prefix_bytes, data_start, records, block_idx, key)
                        if plain is None:
                            _drop_corrupt_shard(self, shard_path)
                            return None
                        part = _load_arrays_from_safetensors_bytes(plain)
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
