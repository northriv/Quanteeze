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
