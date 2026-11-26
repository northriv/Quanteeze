# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""`quanteeze.cuda_engram`: DeepSeek V4.1's Engram rows (mxfp8) and token embedding (bf16) read by the GPU from
the host's pinned copy, bit for bit as the host's lookups, with the ids left on the GPU."""
import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from quanteeze import cuda_engram, cuda_kernels, cuda_ple, ngram_prefetch
from quanteeze.arch import deepseek_v41

ROWS, DIM, VOCAB, HIDDEN = 3000, 256, 1000, 320


def _engram_source():
    """An mxfp8 Engram table as mlx-vlm's QuantizedEngramEmbedding holds it, scales from 2^-127 to 2^127 among
    them and every e4m3 byte, NaN's too, in its first rows."""
    from mlx_vlm.models.deepseek_v41.engram import QuantizedEngramEmbedding

    mx.random.seed(0)
    w, s = mx.quantize((mx.random.normal((ROWS, DIM)) * 0.05).astype(mx.bfloat16), group_size=32, bits=8, mode='mxfp8')
    w, s = np.array(w), np.array(s)
    w.view(np.uint8)[:1] = np.arange(256, dtype=np.uint8)
    s[1:3] = np.array([0, 1, 126, 127, 128, 200, 254, 254, 0, 255, 2, 3, 4, 5, 6, 7], dtype=np.uint8).reshape(2, 8)
    w.view(np.uint8)[1:3] = np.random.default_rng(3).integers(0, 256, (2, DIM), dtype=np.uint8)
    source = QuantizedEngramEmbedding(ROWS, DIM, group_size=32, bits=8, scale_dtype=mx.uint8, mode='mxfp8')
    source.weight, source.scales = mx.array(w), mx.array(s)
    return source


@pytest.fixture
def engram(tmp_path):
    source = _engram_source()
    mx.save_safetensors(
        str(tmp_path / 'model.safetensors'),
        {'layers.1.engram.embed.weight': source.weight, 'layers.1.engram.embed.scales': source.scales},
    )
    return tmp_path, source


@pytest.fixture
def embedding(tmp_path):
    emb_dir = tmp_path / 'embedding'
    emb_dir.mkdir()
    mx.random.seed(1)
    weight = mx.random.normal((VOCAB, HIDDEN)).astype(mx.bfloat16)
    mx.save_safetensors(str(emb_dir / 'emb.safetensors'), {'embed.weight': weight})
    (emb_dir / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'embed.weight': 'emb.safetensors'}})
    )
    return emb_dir, weight


@pytest.fixture
def fresh(monkeypatch):
    """Each shape checked anew, classes of their own to install on."""
    from mlx_vlm.models.deepseek_v41.engram import OffloadedEngramEmbedding

    from quanteeze.arch.deepseek_v41 import SSDEmbedding

    for name in ('CHOICE', 'CHECKED'):
        monkeypatch.setattr(cuda_engram, name, {})
    monkeypatch.setattr(cuda_engram, 'STATS', dict(engram=0, embedding=0, checked=0, mismatched=0))
    monkeypatch.setattr(cuda_ple, 'ON', [True])
    return cuda_engram.install(
        type('Engram', (OffloadedEngramEmbedding,), {}),
        type('Emb', (SSDEmbedding,), {}),
    )


def _held(emb):
    copy = ngram_prefetch.HostCopy([emb], 1 << 40, 2)
    assert copy.wait() == copy.total
    return emb


def _bits(a):
    return np.array(a.view(mx.uint16))


def _engram_ids(count, seed=1):
    edges = [0, 1, 2, ROWS - 1, 1]
    rest = np.random.default_rng(seed).integers(0, ROWS, count - len(edges))
    return np.array(edges + rest.tolist(), dtype=np.int64).reshape(1, 1, count)


def test_reads_memmaps_and_tables_order_on_all_platforms(engram, embedding):
    """`cuda_engram.reads`, `ngram_prefetch._memmaps(SSDEmbedding)`, and `tables()` ordering work on CPU/Mac."""
    from mlx_vlm.models.deepseek_v41.engram import OffloadedEngramEmbedding

    from quanteeze.arch.deepseek_v41 import SSDEmbedding

    path, source = engram
    eg = OffloadedEngramEmbedding(path, 'layers.1.engram.embed', source)
    emb = SSDEmbedding(embedding[0])

    assert cuda_engram.reads(eg)
    assert cuda_engram.reads(emb)
    assert not cuda_engram.reads(SimpleNamespace(_arrays={'weight': np.zeros((4, 8), np.uint32)},
                                                _quantization={'mode': 'affine', 'bits': 4, 'group_size': 32}))

    mmaps = ngram_prefetch._memmaps(emb)
    assert len(mmaps) == 1 and mmaps[0][1] == '_rows' and isinstance(mmaps[0][2], np.memmap)

    mod = nn.Module()
    mod.engram = eg
    mod.embed_tokens = emb
    assert ngram_prefetch.tables(mod) == [emb, eg]


def test_host_ids_and_engram_ahead_avoid_sync_when_pinned_or_in_ram(monkeypatch):
    """Pinned CUDA tables keep lazy GPU IDs, and RAM-held Engram tables skip `_engram_ahead` without `np.asarray`."""
    embed = SimpleNamespace(_arrays={'weight': np.zeros((4, 8), np.uint32), 'scales': np.zeros((4, 1), np.uint8)})
    engram_mod = SimpleNamespace(embed=embed, layer_hash_index=0)

    monkeypatch.setenv('QUANTEEZE_ENGRAM_IDS_ONCE', 'on')
    monkeypatch.setattr(cuda_ple, 'ON', [True])
    monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
    monkeypatch.setattr(cuda_engram, '_table', lambda e: ('engram', 32, None))
    assert deepseek_v41._host_ids(engram_mod) is False

    monkeypatch.setattr(cuda_engram, '_table', lambda e: None)
    assert deepseek_v41._host_ids(engram_mod) is True

    monkeypatch.setenv('QUANTEEZE_ENGRAM_READ_AHEAD', 'on')
    ahead_calls = []
    monkeypatch.setattr(ngram_prefetch, 'ahead', lambda emb, ids: ahead_calls.append((emb, ids)))

    class _SyncingHashes:
        def __getitem__(self, item):
            raise AssertionError('held non-memmap arrays must not slice/sync GPU hashes')

    lm = SimpleNamespace(layers=[SimpleNamespace(engram=None), SimpleNamespace(engram=engram_mod)])
    deepseek_v41._engram_ahead(lm, _SyncingHashes(), 0)
    assert ahead_calls == []


@pytest.mark.skipif(not cuda_kernels.available(), reason="MLX's CUDA backend")
def test_held_engram_rows_read_by_the_gpu_are_mlx_dequantizes_bit_for_bit(engram, fresh):
    from mlx_vlm.models.deepseek_v41.engram import OffloadedEngramEmbedding

    path, source = engram
    plain = OffloadedEngramEmbedding(path, 'layers.1.engram.embed', source)
    assert cuda_engram.reads(plain) and cuda_engram._table(plain) is None
    held = _held(fresh[0](path, 'layers.1.engram.embed', source))
    table = cuda_engram._table(held)
    assert table is not None and table[:2] == ('engram', DIM)
    for count in (24, 48, 256):
        ids = _engram_ids(count)
        want = plain(ids)
        got = cuda_engram.rows(table, mx.array(ids))
        assert got.shape == want.shape == (1, 1, count, DIM)
        g, w = _bits(got), _bits(want)
        nan = np.isnan(np.array(want.astype(mx.float32)))
        assert np.array_equal(np.isnan(np.array(got.astype(mx.float32))), nan)
        assert np.array_equal(g[~nan], w[~nan])


@pytest.mark.skipif(not cuda_kernels.available(), reason="MLX's CUDA backend")
def test_engram_lookups_are_checked_then_served_with_their_ids_left_on_the_gpu(engram, fresh):
    from mlx_vlm.models.deepseek_v41.engram import OffloadedEngramEmbedding

    path, source = engram
    plain = OffloadedEngramEmbedding(path, 'layers.1.engram.embed', source)
    held = _held(fresh[0](path, 'layers.1.engram.embed', source))
    for seed in range(cuda_engram.CHECKS + 3):
        ids = np.random.default_rng(seed).integers(3, ROWS, (1, 1, 24))
        got = held(mx.array(ids) + 0)
        assert np.array_equal(_bits(got), _bits(plain(ids)))
    assert cuda_engram.STATS['checked'] == cuda_engram.CHECKS and cuda_engram.STATS['mismatched'] == 0
    assert cuda_engram.STATS['engram'] == 3


@pytest.mark.skipif(not cuda_kernels.available(), reason="MLX's CUDA backend")
def test_the_token_embedding_held_is_read_by_the_gpu(embedding, fresh):
    from quanteeze.arch.deepseek_v41 import SSDEmbedding

    path, weight = embedding
    plain = SSDEmbedding(path)
    assert cuda_engram.reads(plain) and cuda_engram._table(plain) is None
    held = _held(fresh[1](path))
    assert cuda_engram._table(held)[:2] == ('embedding', HIDDEN)
    for seed in range(cuda_engram.CHECKS + 2):
        ids = np.random.default_rng(seed).integers(0, VOCAB, (1, 1)).astype(np.uint32)
        got = held(mx.array(ids) + 0)
        assert np.array_equal(_bits(got), _bits(weight[mx.array(ids)]))
    assert cuda_engram.STATS['embedding'] == 2 and cuda_engram.STATS['mismatched'] == 0


@pytest.mark.skipif(not cuda_kernels.available(), reason="MLX's CUDA backend")
def test_a_prefill_numpy_ids_or_off_are_looked_up_on_the_host(engram, fresh, monkeypatch):
    path, source = engram
    held = _held(fresh[0](path, 'layers.1.engram.embed', source))
    own = type(held).__mro__[1].__call__
    many = np.random.default_rng(0).integers(0, ROWS, (1, 20, 24))
    assert cuda_engram.lookup(held, mx.array(many), own) is None
    assert cuda_engram.lookup(held, many[:, :1], own) is None
    monkeypatch.setattr(cuda_ple, 'ON', [False])
    assert cuda_engram.lookup(held, mx.array(many[:, :1]), own) is None


@pytest.mark.skipif(not cuda_kernels.available(), reason="MLX's CUDA backend")
def test_hold_takes_the_token_embedding_first_and_pins_what_the_gpu_reads(engram, embedding, monkeypatch):
    from mlx_vlm.models.deepseek_v41.engram import OffloadedEngramEmbedding

    from quanteeze.arch.deepseek_v41 import SSDEmbedding

    monkeypatch.setattr(cuda_ple, 'ON', [True])
    path, source = engram
    tables = nn.Module()
    tables.engram = OffloadedEngramEmbedding(path, 'layers.1.engram.embed', source)
    tables.embed_tokens = SSDEmbedding(embedding[0])
    assert ngram_prefetch.tables(tables) == [tables.embed_tokens, tables.engram]
    copy = ngram_prefetch.hold(tables, 1 << 40, 2)
    try:
        assert copy.wait() == ROWS * (DIM + DIM // 32) + VOCAB * HIDDEN * 2
        assert cuda_engram._table(tables.engram) is not None and cuda_engram._table(tables.embed_tokens) is not None
    finally:
        ngram_prefetch.let_go()


def test_ngram_host_copy_whole_within_tolerance_and_left_counts(tmp_path):
    """`HostCopy` bumps a budget shortfall within `min(WHOLE_WITHIN, wanted // WHOLE_SHARE)` to hold all arrays,
    and counts `left` / `left_bytes` when arrays genuinely exceed budget."""
    import mmap
    import mlx.nn as nn
    from quanteeze import ngram_prefetch

    n_rows, cols = 2000, 128
    arr_bytes = n_rows * cols
    f1 = tmp_path / 't1.bin'
    f2 = tmp_path / 't2.bin'
    f1.write_bytes(b'\x01' * arr_bytes)
    f2.write_bytes(b'\x02' * arr_bytes)

    class Holder:
        def __init__(self, a, b):
            self._arrays = {'m1': a, 'm2': b}

    m1 = np.memmap(f1, dtype=np.uint8, mode='r', shape=(n_rows, cols))
    m2 = np.memmap(f2, dtype=np.uint8, mode='r', shape=(n_rows, cols))
    h = Holder(m1, m2)
    wanted = 2 * arr_bytes
    # Short by 512 bytes (< wanted // 200 = 2560): tolerated -> both held, left == 0
    copy = ngram_prefetch.HostCopy([h], budget=wanted - 512, workers=1)
    try:
        assert copy.wait() == wanted
        assert copy.left == 0 and copy.left_bytes == 0
    finally:
        copy.close()

    # Short by more than tolerance (only room for 1 array): left == 1, left_bytes == arr_bytes
    m1b = np.memmap(f1, dtype=np.uint8, mode='r', shape=(n_rows, cols))
    m2b = np.memmap(f2, dtype=np.uint8, mode='r', shape=(n_rows, cols))
    h2 = Holder(m1b, m2b)
    copy2 = ngram_prefetch.HostCopy([h2], budget=arr_bytes, workers=1)
    try:
        assert copy2.wait() == arr_bytes
        assert copy2.left == 1 and copy2.left_bytes == arr_bytes
    finally:
        copy2.close()
