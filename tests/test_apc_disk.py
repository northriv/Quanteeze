# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for streaming AES-256-GCM sealing and unsealing in `quanteeze.apc_disk`."""
import mlx.core as mx
import numpy as np

from quanteeze import apc_disk


def _sample_arrays():
    rng = np.random.default_rng(0)
    big = mx.array(rng.standard_normal((1, 3, 50, 8)).astype(np.float32)).astype(mx.bfloat16)
    return {
        'a': big,
        'b': mx.arange(37, dtype=mx.int32),
        'c': mx.array([True, False, True]),
        'd': mx.array(rng.standard_normal((5, 7)).astype(np.float32)),
        's': big[0, 1, 3:17],
        'e': mx.array(7, dtype=mx.uint32),
        'empty': mx.zeros((0, 4), dtype=mx.float16),
    }


def _assert_arrays_equal(got: dict[str, mx.array], want: dict[str, mx.array]) -> None:
    assert set(got) == set(want)
    for name, w in want.items():
        g = got[name]
        assert g.dtype == w.dtype and g.shape == w.shape, name
        if w.size:
            assert mx.array_equal(g, w).item(), name


def test_streaming_seal_and_unseal_chunked_roundtrip_and_wire_compatibility(tmp_path):
    """Streaming `_write_sealed_records` and `_decrypt_record_*` match one-shot AESGCM wire format."""
    arrays = _sample_arrays()
    meta = {'layout': 'x', 'n': '3'}
    key_stream = b'k' * 32
    key_second = b's' * 32
    second_arrays = {'only': mx.arange(19, dtype=mx.int32)}
    second_meta = {'layout': 'second'}

    stream_path = tmp_path / 'stream_multi.safetensors'
    apc_disk._write_sealed_records(
        stream_path,
        {'kind': 'exact_trim_v1', 'token_count': 20},
        [(arrays, meta, key_stream), (second_arrays, second_meta, key_second)],
        limit=48,
    )

    env, prefix_bytes, data_start = apc_disk._read_envelope(stream_path)
    records = env['records']
    with open(stream_path, 'rb') as f:
        # Streaming decrypt across 40-byte chunks (header & tensor boundaries)
        got0_arrays, got0_meta = apc_disk._decrypt_record_arrays(
            f, prefix_bytes, data_start, records, 0, key_stream, limit=40
        )
        got1_arrays, got1_meta = apc_disk._decrypt_record_arrays(
            f, prefix_bytes, data_start, records, 1, key_second, limit=40
        )
        # One-shot decrypt on the exact same streaming-written file
        oneshot_plain = apc_disk._decrypt_record(f, prefix_bytes, data_start, records, 0, key_stream)
        assert oneshot_plain is not None
        oneshot_arrays, oneshot_meta = apc_disk._load_arrays_from_safetensors_bytes(oneshot_plain)

    from mlx_vlm import apc as _apc
    assert _apc._EMPTY_TENSORS_METADATA in got0_meta
    assert got0_meta.get('layout') == 'x' and got0_meta.get('n') == '3'
    assert got1_meta == second_meta
    assert oneshot_meta == got0_meta
    _assert_arrays_equal(got0_arrays, arrays)
    _assert_arrays_equal(got1_arrays, second_arrays)
    _assert_arrays_equal(oneshot_arrays, arrays)

    # Reverse direction: one-shot write decrypted via streaming `_decrypt_record_arrays`
    oneshot_path = tmp_path / 'oneshot_written.safetensors'
    plain_blob = apc_disk._safetensors_dumps(arrays, meta)
    apc_disk._write_sealed_file(
        oneshot_path,
        {'kind': 'exact_v1', 'token_count': 10},
        [(plain_blob, key_stream)],
    )
    env2, prefix2, start2 = apc_disk._read_envelope(oneshot_path)
    with open(oneshot_path, 'rb') as f:
        rev_arrays, rev_meta = apc_disk._decrypt_record_arrays(
            f, prefix2, start2, env2['records'], 0, key_stream, limit=40
        )
    assert rev_meta.get('layout') == 'x' and rev_meta.get('n') == '3'
    _assert_arrays_equal(rev_arrays, arrays)


def test_streaming_sentinel_only_empty_arrays_roundtrip(tmp_path):
    """Empty tensor dict seals with internal sentinel and restores empty `{}` without leaking sentinel."""
    path = tmp_path / 'empty_only.safetensors'
    key = b'z' * 32
    apc_disk._write_sealed_records(
        path,
        {'kind': 'exact_v1', 'token_count': 1},
        [({}, {'empty_meta': 'ok'}, key)],
        limit=32,
    )
    env, prefix_bytes, data_start = apc_disk._read_envelope(path)
    with open(path, 'rb') as f:
        got_arrays, got_meta = apc_disk._decrypt_record_arrays(
            f, prefix_bytes, data_start, env['records'], 0, key, limit=16
        )
    assert got_arrays == {}
    assert got_meta == {'empty_meta': 'ok'}


def test_streaming_unseal_rejects_wrong_key_and_any_bit_tamper(tmp_path):
    """Wrong keys and 1-bit modifications in envelope AAD, header, tensor payload, or tag are rejected."""
    rng = np.random.default_rng(1)
    arrays = {
        'k': mx.array(rng.standard_normal((1, 2, 64, 8)).astype(np.float32)).astype(mx.bfloat16),
        'p': mx.arange(64, dtype=mx.int32)[None],
        'f': mx.array(rng.standard_normal(9).astype(np.float32)),
        'b': mx.array([True, False]),
    }
    key = b'q' * 32
    path = tmp_path / 'sealed_tamper.safetensors'
    apc_disk._write_sealed_records(
        path,
        {'kind': 'exact_v1', 'token_count': 64},
        [(arrays, {'layout': 'y'}, key)],
        limit=48,
    )
    raw = bytearray(path.read_bytes())
    env, prefix_bytes, data_start = apc_disk._read_envelope(path)
    records = env['records']

    with open(path, 'rb') as f:
        rejected_before = apc_disk.STATS['rejected']
        assert apc_disk._decrypt_record_tensors(f, prefix_bytes, data_start, records, 0, b'r' * 32, limit=40) is None
        assert apc_disk.STATS['rejected'] == rejected_before + 1

    # Flip 1 bit in: (a) AAD envelope prefix, (b) ciphertext header, (c) tensor payload, (d) GCM tag
    positions = (
        16,                                     # envelope JSON prefix (authenticated via AAD)
        data_start + apc_disk.NONCE_BYTES + 4,  # safetensors header bytes
        len(raw) // 2,                          # tensor ciphertext bytes
        len(raw) - 1,                           # GCM tag
    )
    for pos in positions:
        touched = bytearray(raw)
        touched[pos] ^= 1
        tampered_path = tmp_path / f'tamper_{pos}.safetensors'
        tampered_path.write_bytes(bytes(touched))
        parsed = apc_disk._read_envelope(tampered_path)
        if parsed is None:
            continue
        t_env, t_prefix, t_start = parsed
        with open(tampered_path, 'rb') as f:
            rejected_before = apc_disk.STATS['rejected']
            res = apc_disk._decrypt_record_tensors(
                f, t_prefix, t_start, t_env['records'], 0, key, limit=40
            )
            assert res is None, pos
            assert apc_disk.STATS['rejected'] == rejected_before + 1


def test_disk_block_store_roundtrips_with_small_stream_piece(tmp_path, monkeypatch):
    """Full `DiskBlockStore` exact_v1, exact_trim_v1, and blocks_v1 roundtrips under small `PIECE` chunking."""
    from mlx_vlm import apc as _apc
    from mlx_vlm.models import cache as lm_cache

    apc_disk.install()
    monkeypatch.setattr(apc_disk, 'ON', [True])
    monkeypatch.setattr(apc_disk, 'PRECISION', [None])
    monkeypatch.setattr(apc_disk, 'PIECE', 64)
    monkeypatch.setenv('QUANTEEZE_BITS', '4')
    monkeypatch.setenv('APC_DISK_MIN_FREE_RAM_GB', '0')
    monkeypatch.setenv('APC_MEMORY_RESERVE_GB', '0')

    # exact_v1 (KVCache + ArraysCache)
    s_hybrid = _apc.DiskBlockStore(tmp_path / 'hybrid', namespace='ns', num_workers=1)
    m_hybrid = _apc.APCManager(
        num_blocks=4, block_size=8, disk=s_hybrid, overrides={'memory_max_gb': 0, 'checkpoint_entries': 0}
    )
    m_hybrid.exact_cache_min_tokens = 4
    toks = tuple(100 + i for i in range(12))
    kv = lm_cache.KVCache()
    kv.keys = mx.arange(1 * 2 * 12 * 4, dtype=mx.bfloat16).reshape(1, 2, 12, 4)
    kv.values = kv.keys + 1
    kv.offset = 12
    ssm = lm_cache.ArraysCache(size=1)
    ssm.cache = [mx.ones((1, 4, 8), dtype=mx.bfloat16)]
    mx.eval(kv.keys, kv.values, ssm.cache[0])
    assert m_hybrid.store_exact_cache(toks, [kv, ssm], extra_hash=1) is True
    s_hybrid.flush()
    s_hybrid.close()

    apc_disk.clear_runtime_keys()
    s_hybrid2 = _apc.DiskBlockStore(tmp_path / 'hybrid', namespace='ns', num_workers=1)
    m_hybrid2 = _apc.APCManager(
        num_blocks=4, block_size=8, disk=s_hybrid2, overrides={'memory_max_gb': 0, 'checkpoint_entries': 0}
    )
    m_hybrid2.exact_cache_min_tokens = 4
    try:
        restored, matched = m_hybrid2.lookup_exact_cache(toks + (999,), extra_hash=1)
        assert matched == 12
        assert mx.array_equal(restored[0].keys[..., :12, :], kv.keys).item()
        assert mx.array_equal(restored[0].values[..., :12, :], kv.values).item()
        assert mx.array_equal(restored[1].cache[0], ssm.cache[0]).item()
    finally:
        s_hybrid2.close()

    # exact_trim_v1 + blocks_v1
    s_blk = _apc.DiskBlockStore(tmp_path / 'blk', namespace='ns', num_workers=1)
    m_blk = _apc.APCManager(
        num_blocks=4, block_size=8, disk=s_blk, overrides={'memory_max_gb': 0, 'checkpoint_entries': 0}
    )
    m_blk.exact_cache_min_tokens = 4
    toks2 = tuple(200 + i for i in range(20))
    k0 = mx.arange(1 * 2 * 20 * 4, dtype=mx.float16).reshape(1, 2, 20, 4)
    v0 = k0 + 2
    dkv = lm_cache.KVCache()
    dkv.keys, dkv.values, dkv.offset = k0, v0, 20
    mx.eval(k0, v0)
    assert m_blk.store_exact_cache(toks2, [dkv], extra_hash=5) is True
    blocks = m_blk.store_kv_blocks(toks2[:16], [k0[..., :16, :]], [v0[..., :16, :]], extra_hash=6)
    m_blk.release(blocks)
    s_blk.flush()
    s_blk.close()

    apc_disk.clear_runtime_keys()
    s_blk2 = _apc.DiskBlockStore(tmp_path / 'blk', namespace='ns', num_workers=1)
    m_blk2 = _apc.APCManager(
        num_blocks=4, block_size=8, disk=s_blk2, overrides={'memory_max_gb': 0, 'checkpoint_entries': 0}
    )
    m_blk2.exact_cache_min_tokens = 4
    try:
        trim_cache, trim_len = m_blk2.lookup_exact_cache(toks2[:16] + (800, 801), extra_hash=5)
        assert trim_len == 16
        assert mx.array_equal(trim_cache[0].keys[..., :16, :], k0[..., :16, :]).item()
        full_cache, full_len = m_blk2.lookup_exact_cache(toks2 + (900,), extra_hash=5)
        assert full_len == 20
        assert mx.array_equal(full_cache[0].values[..., :20, :], v0).item()
        warm, wlen = m_blk2.lookup_prefix_disk_cache(toks2[:8] + (700,) * 8, extra_hash=6)
        assert wlen == 8
        assert mx.array_equal(warm[0].keys[..., :8, :], k0[..., :8, :]).item()
    finally:
        s_blk2.close()


def test_apc_disk_secret_hmac_permissions_and_rotation(tmp_path, monkeypatch):
    """`QUANTEEZE_APC_DISK_SECRET` creates a 0600 32-byte key mixed via HMAC-SHA-256 into disk names/keys,
    replaces world-readable files, calls Time Machine exclusion on macOS, and isolates prompts from guessing."""
    import hashlib
    import os
    import sys

    sec_path = tmp_path / 'sub' / 'apc-secret'
    monkeypatch.setenv('QUANTEEZE_APC_DISK_SECRET', str(sec_path))
    monkeypatch.setattr(apc_disk, '_SECRET', {'value': None, 'read': False})

    tm_calls = []
    if sys.platform == 'darwin':
        orig_run = apc_disk.subprocess.run
        monkeypatch.setattr(apc_disk.subprocess, 'run', lambda cmd, **kw: tm_calls.append(cmd) or orig_run(
            ['true'], check=True, capture_output=True))

    s1 = apc_disk.secret()
    assert isinstance(s1, bytes) and len(s1) == apc_disk.SECRET_BYTES
    assert sec_path.is_file() and (os.stat(sec_path).st_mode & 0o777) == 0o600
    if sys.platform == 'darwin':
        assert tm_calls and tm_calls[0][:2] == ['tmutil', 'addexclusion']

    # Re-reading in a new process keeps the exact same secret
    monkeypatch.setattr(apc_disk, '_SECRET', {'value': None, 'read': False})
    assert apc_disk.secret() == s1

    # Different secrets produce distinct exact locators/keys for identical tokens; off falls back to SHA-256
    toks = [1, 2, 3, 4, 5, 6]
    d1 = apc_disk._derive_exact_digests(toks, 0, (4, 6), b's')
    apc_disk.use_secret(bytes(range(1, 33)))
    d2 = apc_disk._derive_exact_digests(toks, 0, (4, 6), b's')
    assert d1[4][0] != d2[4][0] and d1[4][2] != d2[4][2]
    assert d1[6][2] != d2[6][2]

    monkeypatch.setenv('QUANTEEZE_APC_DISK_SECRET', 'off')
    monkeypatch.setattr(apc_disk, '_SECRET', {'value': None, 'read': False})
    assert apc_disk.secret() is None
    assert apc_disk._hash(b'abc').digest() == hashlib.sha256(b'abc').digest()

    # Unsafe permissions (0o644) cause replacement with a fresh 0o600 secret
    monkeypatch.setenv('QUANTEEZE_APC_DISK_SECRET', str(sec_path))
    os.chmod(sec_path, 0o644)
    monkeypatch.setattr(apc_disk, '_SECRET', {'value': None, 'read': False})
    s3 = apc_disk.secret()
    assert s3 is not None and s3 != s1
    assert (os.stat(sec_path).st_mode & 0o777) == 0o600

    # Symlinks are replaced atomically with a regular 0o600 secret without touching the symlink target
    sym_target = tmp_path / 'outside-secret'
    sym_target.write_text(bytes(range(32)).hex() + '\n')
    os.chmod(sym_target, 0o600)
    sec_path.unlink()
    sec_path.symlink_to(sym_target)
    monkeypatch.setattr(apc_disk, '_SECRET', {'value': None, 'read': False})
    s4 = apc_disk.secret()
    assert s4 is not None and not sec_path.is_symlink()
    assert (os.lstat(sec_path).st_mode & 0o777) == 0o600
    assert sym_target.read_text() == bytes(range(32)).hex() + '\n'

    # Restore fixed conftest secret
    apc_disk.use_secret(bytes(range(32)))


def _v41_cache(n: int, layers: int = 2, seed: int = 0):
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache

    def rows(layer: int, kind: int, count: int, width: int):
        return mx.concatenate(
            [
                mx.random.normal(
                    (1, 1, width),
                    key=mx.random.key(seed * 10007 + layer * 131 + kind * 17 + i),
                ).astype(mx.bfloat16)
                for i in range(count)
            ],
            axis=1,
        )

    c = DeepseekV41Cache(layers)
    c.offset = n
    for l_idx in range(layers):
        c.compress[l_idx] = rows(l_idx, 0, n // 4, 32)
        c.keys[l_idx] = rows(l_idx, 1, n // 4, 16)
        c.window[l_idx] = mx.random.normal((1, 8, 32), key=mx.random.key(n * 7 + l_idx)).astype(mx.bfloat16)
        c.kv_state[l_idx] = mx.random.normal((1, 4, 32), key=mx.random.key(n * 11 + l_idx))
        c.score_state[l_idx] = mx.random.normal((1, 4, 32), key=mx.random.key(n * 13 + l_idx))
    mx.eval(c.state)
    return c


def _same_v41(a, b) -> bool:
    sa, sb = a.state, b.state
    return (
        a.offset == b.offset
        and len(sa) == len(sb)
        and all(
            x.shape == y.shape and x.dtype == y.dtype and mx.array_equal(x, y).item()
            for x, y in zip(sa, sb)
        )
    )


def _exact_by_tokens(store) -> dict[int, tuple]:
    out = {}
    for p in store.dir.glob(f'*{store.SUFFIX}'):
        if not store._is_canonical_exact(p) or not p.exists():
            continue
        parsed = apc_disk._read_envelope(p)
        if parsed is not None:
            env = parsed[0]
            out[int(env['token_count'])] = (p, env.get('parent_token_count'))
    return out


def _load_v41(store, ids, n: int):
    hit = store.find_exact_prefix(tuple(ids) + (1,))
    assert hit is not None and hit[1] == n
    loaded = store.load_exact_cache(hit[0], prefix_len=n)
    assert loaded is not None
    return loaded[2][0]


def test_v41_delta_checkpoint_chain_depth_reset_and_cold_restart(tmp_path, monkeypatch):
    """V4.1 compressed/index keys grow along axis 1: extending prefixes seal as deltas up to `DELTA_DEPTH`,
    reset to whole past `DELTA_DEPTH`, and roundtrip across cold restarts."""
    from mlx_vlm import apc as _apc

    apc_disk.install()
    monkeypatch.setattr(apc_disk, 'ON', [True])
    monkeypatch.setattr(apc_disk, 'PRECISION', [None])
    monkeypatch.setattr(apc_disk, 'DELTA_MIN_BYTES', 256)
    monkeypatch.setattr(apc_disk, 'DELTA_DEPTH', 2)
    monkeypatch.setenv('APC_DISK_MIN_FREE_RAM_GB', '0')
    monkeypatch.setenv('APC_MEMORY_RESERVE_GB', '0')

    store = _apc.DiskBlockStore(tmp_path / 'chain', namespace='ns', num_workers=1)
    ids = list(range(1000, 1000 + 600))
    sizes = {}
    for n in (128, 256, 384, 512):
        assert store.save_exact_cache(n, ids[:n], 0, [_v41_cache(n)], synchronous=True)
        path, parent_tc = _exact_by_tokens(store)[n]
        sizes[n] = (path.stat().st_size, parent_tc)
    assert [sizes[n][1] for n in (128, 256, 384, 512)] == [None, 128, 256, None]

    alone = _apc.DiskBlockStore(tmp_path / 'alone', namespace='ns', num_workers=1)
    assert alone.save_exact_cache(384, ids[:384], 0, [_v41_cache(384)], synchronous=True)
    whole_size = _exact_by_tokens(alone)[384][0].stat().st_size
    alone.close()
    assert sizes[256][0] < 0.65 * whole_size and sizes[384][0] < 0.65 * whole_size

    # Diverged values sharing token prefix write whole rather than referencing mismatched parent arrays
    diverged_ids = ids[:256] + [5] * 344
    assert store.save_exact_cache(600, diverged_ids, 0, [_v41_cache(600, seed=3)], synchronous=True)
    assert _exact_by_tokens(store)[600][1] is None
    store.close()

    # Cold restart: verify every checkpoint in the chain and after the depth reset
    apc_disk.clear_runtime_keys()
    store2 = _apc.DiskBlockStore(tmp_path / 'chain', namespace='ns', num_workers=1)
    try:
        # Note: 256 & 384 were pruned once 512 (whole) and 600 were written if unreferenced, or check stored ones
        stored_tcs = sorted(_exact_by_tokens(store2))
        assert 128 in stored_tcs and 512 in stored_tcs and 600 in stored_tcs
        for n in (128, 512):
            assert _same_v41(_load_v41(store2, ids[:n], n), _v41_cache(n)), n
        assert _same_v41(_load_v41(store2, diverged_ids, 600), _v41_cache(600, seed=3))
    finally:
        store2.close()


def test_delta_checkpoint_missing_or_rewritten_parent_rejected(tmp_path, monkeypatch):
    """Unlinking or rewriting a delta's parent with different values rejects and drops the stale delta."""
    from mlx_vlm import apc as _apc

    apc_disk.install()
    monkeypatch.setattr(apc_disk, 'ON', [True])
    monkeypatch.setattr(apc_disk, 'PRECISION', [None])
    monkeypatch.setattr(apc_disk, 'DELTA_MIN_BYTES', 256)
    monkeypatch.setenv('APC_DISK_MIN_FREE_RAM_GB', '0')
    monkeypatch.setenv('APC_MEMORY_RESERVE_GB', '0')

    store = _apc.DiskBlockStore(tmp_path / 'stale', namespace='ns', num_workers=1)
    try:
        ids = list(range(2000, 2000 + 256))
        for n in (128, 256):
            assert store.save_exact_cache(n, ids[:n], 0, [_v41_cache(n)], synchronous=True)
        by_tc = _exact_by_tokens(store)
        parent_path = by_tc[128][0]
        child_path = by_tc[256][0]
        assert by_tc[256][1] == 128

        parent_path.unlink()
        assert store.find_exact_prefix(ids + [1]) is None

        # Rewrite parent at same token prefix with different tensor values (seed=9)
        assert store.save_exact_cache(128, ids[:128], 0, [_v41_cache(128, seed=9)], synchronous=True)
        hit = store.find_exact_prefix(ids + [1])
        assert hit is not None and hit[1] == 256
        assert store.load_exact_cache(hit[0], prefix_len=256) is None
        assert not child_path.exists()
        assert _same_v41(_load_v41(store, ids[:128], 128), _v41_cache(128, seed=9))
    finally:
        store.close()


def test_delta_checkpoint_parent_retained_by_prune_and_evicted_tail_first(tmp_path, monkeypatch):
    """Prune retains every ancestor referenced by a delta, and cap eviction drops leaf children before parents."""
    import os
    from mlx_vlm import apc as _apc

    apc_disk.install()
    monkeypatch.setattr(apc_disk, 'ON', [True])
    monkeypatch.setattr(apc_disk, 'PRECISION', [None])
    monkeypatch.setattr(apc_disk, 'DELTA_MIN_BYTES', 256)
    monkeypatch.setenv('APC_DISK_MIN_FREE_RAM_GB', '0')
    monkeypatch.setenv('APC_MEMORY_RESERVE_GB', '0')

    store = _apc.DiskBlockStore(tmp_path / 'evict', namespace='ns', num_workers=1)
    try:
        ids = list(range(3000, 3000 + 640))
        for n in (128, 256, 384, 512, 640):
            assert store.save_exact_cache(n, ids[:n], 0, [_v41_cache(n)], synchronous=True)
        by_tc = _exact_by_tokens(store)
        assert sorted(by_tc) == [128, 256, 384, 512, 640]
        assert _same_v41(_load_v41(store, ids, 640), _v41_cache(640))

        for k, (p, _) in by_tc.items():
            os.utime(p, (1000 + k, 1000 + k))
        store.max_bytes = sum(p.stat().st_size for p, _ in by_tc.values()) - 1
        assert apc_disk.evict(store, keep=None) >= 1
        left = sorted(_exact_by_tokens(store))
        assert left and left == list(range(128, 128 * (len(left) + 1), 128))
        top = left[-1]
        assert _same_v41(_load_v41(store, ids[:top], top), _v41_cache(top))
    finally:
        store.close()


def test_hybrid_kv_heads_before_tokens_delta_checkpoints(tmp_path, monkeypatch):
    """KV caches with `[1, heads, tokens, dim]` alongside recurrent `ArraysCache` seal as axis-2 deltas."""
    from mlx_vlm import apc as _apc
    from mlx_vlm.models import cache as lm_cache

    apc_disk.install()
    monkeypatch.setattr(apc_disk, 'ON', [True])
    monkeypatch.setattr(apc_disk, 'PRECISION', [None])
    monkeypatch.setattr(apc_disk, 'DELTA_MIN_BYTES', 256)
    monkeypatch.setenv('APC_DISK_MIN_FREE_RAM_GB', '0')
    monkeypatch.setenv('APC_MEMORY_RESERVE_GB', '0')

    keys = mx.random.normal((1, 4, 640, 16), key=mx.random.key(1)).astype(mx.bfloat16)
    mx.eval(keys)

    def make_hybrid(n: int):
        kv = lm_cache.KVCache()
        kv.update_and_fetch(keys[:, :, :n], keys[:, :, :n] * 3)
        ssm = lm_cache.ArraysCache(size=1)
        ssm.cache = [mx.ones((1, 4, 8), dtype=mx.bfloat16) * float(n)]
        mx.eval(kv.keys, kv.values, ssm.cache[0])
        return [kv, ssm]

    store = _apc.DiskBlockStore(tmp_path / 'kv_delta', namespace='ns', num_workers=1)
    try:
        ids = list(range(4000, 4000 + 640))
        for n in (128, 256, 384, 512, 640):
            assert store.save_exact_cache(n, ids[:n], 0, make_hybrid(n), synchronous=True)
        by_tc = _exact_by_tokens(store)
        assert [by_tc[k][1] for k in sorted(by_tc)] == [None, 128, 256, 384, 512]
        assert by_tc[640][0].stat().st_size < 0.45 * by_tc[128][0].stat().st_size * 5
        hit = store.find_exact_prefix(ids + [1])
        assert hit is not None and hit[1] == 640
        loaded = store.load_exact_cache(hit[0], prefix_len=640)
        assert loaded is not None
        _toks, _extra, (got_kv, got_ssm) = loaded
        assert got_kv.offset == 640
        assert mx.array_equal(got_kv.keys[..., :640, :], keys).item()
        assert mx.array_equal(got_kv.values[..., :640, :], keys * 3).item()
        assert mx.array_equal(got_ssm.cache[0], mx.ones((1, 4, 8), dtype=mx.bfloat16) * 640.0).item()
    finally:
        store.close()
