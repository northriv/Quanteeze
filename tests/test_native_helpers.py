# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
import shutil
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from quanteeze import dspark, engram_host, native_ext, ngram_prefetch, v41_attention


def test_touch_native_extension_and_fallback(tmp_path):
    arr = np.arange(16 * 512, dtype=np.int32).reshape(16, 512)
    out = np.zeros(2 * 3 * 4, dtype=np.uint8)
    rows = np.array([0, 7, 15], dtype=np.int64)
    got = ngram_prefetch._touch_native((arr, rows), out=out)
    assert got == 3
    pairs = out.view(np.int32).reshape(3, 2)
    assert np.array_equal(pairs[:, 0], arr[rows, 0])
    assert np.array_equal(pairs[:, 1], arr[rows, -1])

    # Normal _touch() path matches Python sum path on in-bounds rows and rejects OOB rows via Python
    assert ngram_prefetch._touch((arr, rows)) == 3
    with pytest.raises(IndexError):
        ngram_prefetch._touch((arr, np.array([0, 99], dtype=np.int64)))


def _ref_engram_hashes(token_map, mult, primes, offsets, pad, n, history_init, ids, mask):
    hist = list(history_init)
    start = len(hist)
    for j, tok in enumerate(ids):
        hist.append(engram_host.DEAD if (mask is not None and not mask[j]) else int(token_map[tok]))
    layers, cols = mult.shape[0], offsets.shape[1]
    heads = cols // (n - 1)
    out = np.empty((1, len(ids), layers, cols), dtype=np.int64)
    for j in range(len(ids)):
        pos = start + j
        toks = []
        blocked = False
        for shift in range(n):
            idx = pos - shift
            src = hist[idx] if idx >= 0 else engram_host.DEAD
            blocked = blocked or pos < shift or src == engram_host.DEAD
            toks.append(pad if blocked else src)
        for l in range(layers):
            rolling = (toks[0] * int(mult[l, 0])) & 0xFFFFFFFFFFFFFFFF
            for i in range(1, n):
                rolling ^= (toks[i] * int(mult[l, i])) & 0xFFFFFFFFFFFFFFFF
                signed = rolling if rolling < (1 << 63) else rolling - (1 << 64)
                for h in range(heads):
                    p = int(primes[l, i - 1, h])
                    out[0, j, l, (i - 1) * heads + h] = (signed % p) + int(offsets[l, (i - 1) * heads + h])
    return out


def test_engram_host_hashes_bit_exact(monkeypatch):
    engram_host.STATE.update(checked=0, mismatched=0, off=False, served=0, passes=0, tokens=0, resynced=0, said=False)
    monkeypatch.setenv('QUANTEEZE_ENGRAM_HASHES_ON_HOST', 'on')

    rng = np.random.default_rng(42)
    vocab, n, layers, heads = 64, 4, 2, 3
    cols = (n - 1) * heads
    token_map = rng.integers(1, 50, size=(vocab,), dtype=np.int64)
    mult = (rng.integers(1, 100000, size=(layers, n), dtype=np.int64) * 2) + 1
    primes = np.array([[[101, 103, 107], [109, 113, 127], [131, 137, 139]],
                       [[149, 151, 157], [163, 167, 173], [179, 181, 191]]], dtype=np.int64)
    offsets = rng.integers(0, 500, size=(layers, cols), dtype=np.int64)
    pad = 0
    ngram = SimpleNamespace(
        _token_map=mx.array(token_map),
        _multipliers=mx.array(mult),
        _primes=mx.array(primes),
        _offsets=mx.array(offsets),
        pad_id=pad,
        layout=SimpleNamespace(max_ngram_size=n),
    )

    row = SimpleNamespace(offset=0, engram=None)
    prompt = np.array([[5, 63, 12, 7, 3]], dtype=np.int64)
    img_id = 63
    want1 = _ref_engram_hashes(token_map, mult, primes, offsets, pad, n, [], prompt[0], prompt[0] != img_id)
    got1 = engram_host.pass_hashes(ngram, row, mx.array(prompt), None, lambda: mx.array(want1), img_id)
    assert got1 is not None
    assert np.array_equal(got1, want1)

    # Next decode step reusing host history without syncing GPU `row.engram`
    row.offset = 5
    before_tag = object()
    row.engram = before_tag
    hist_prefix = [engram_host.DEAD if t == img_id else int(token_map[t]) for t in prompt[0]]
    row._quanteeze_engram_host = (np.array(hist_prefix, dtype=np.int64), before_tag)
    step = np.array([[18]], dtype=np.int64)
    want2 = _ref_engram_hashes(token_map, mult, primes, offsets, pad, n, hist_prefix, step[0], step[0] != img_id)
    got2 = engram_host.pass_hashes(ngram, row, mx.array(step), before_tag, lambda: mx.array(want2), img_id)
    assert got2 is not None
    assert np.array_equal(got2, want2)


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal backend required')
def test_v41_attention_native_matches_python(monkeypatch):
    from mlx_vlm.models.deepseek_v4.language import DeepseekV4RoPE
    from mlx_vlm.models.mla import MultiLinear
    from quanteeze.arch.deepseek_v41 import _window_indices

    mx.random.seed(7)
    monkeypatch.setenv('QUANTEEZE_V41_ATTENTION_NATIVE', 'compare')
    v41_attention._STATE.update(ext=None, off=False, said=False, checked={}, served=0, compared=0, declined=0)

    ext = v41_attention._ext()
    assert ext is not None
    assert mx.array_equal(ext.probe(mx.array([1.0, 2.0], dtype=mx.float32)), mx.array([2.0, 3.0], dtype=mx.float32)).item()

    d_model, heads, head_dim, o_groups, win = 64, 2, 32, 2, 16
    q_rank = 64
    o_rank = 16
    rope = DeepseekV4RoPE(head_dim, base=10000.0, max_position_embeddings=64)
    wq_a = nn.Linear(d_model, q_rank, bias=False)
    wq_b = nn.Linear(q_rank, heads * head_dim, bias=False)
    wkv = nn.Linear(d_model, head_dim, bias=False)
    wo_a = MultiLinear((heads // o_groups) * head_dim, o_rank, o_groups)
    wo_b = nn.Linear(o_groups * o_rank, d_model, bias=False)
    for mod in (wq_a, wq_b, wkv, wo_a, wo_b):
        mod.weight = (mx.random.normal(mod.weight.shape) * 0.02).astype(mx.bfloat16)
    attn = SimpleNamespace(
        layer_idx=0,
        wq_a=wq_a,
        wq_b=wq_b,
        wkv=wkv,
        wo_a=wo_a,
        wo_b=wo_b,
        q_norm=nn.RMSNorm(q_rank, eps=1e-6),
        kv_norm=nn.RMSNorm(head_dim, eps=1e-6),
        rope=rope,
        attn_sink=mx.zeros((heads,), dtype=mx.float32),
        scale=head_dim ** -0.5,
        n_heads=heads,
        head_dim=head_dim,
        o_groups=o_groups,
        window_size=win,
        compress_ratio=0,
        config=SimpleNamespace(hidden_size=d_model, q_lora_rank=q_rank, o_lora_rank=o_rank),
        _compress_part=lambda x, qr, sp, cache: (None, None),
    )
    attn.q_norm.weight = attn.q_norm.weight.astype(mx.bfloat16)
    attn.kv_norm.weight = attn.kv_norm.weight.astype(mx.bfloat16)
    mx.eval(rope._get_freqs(head_dim, False), rope._get_freqs(head_dim, True))

    cache = SimpleNamespace(offset=0, window=[None])
    x0 = mx.random.normal((1, 4, d_model)).astype(mx.bfloat16)
    mx.eval(x0)
    y0 = v41_attention.attend(attn, cache, x0, _window_indices)
    assert y0 is not None
    assert v41_attention.stats()['compared'] == 1
    assert not v41_attention.stats()['off']

    cache.offset = 4
    x1 = mx.random.normal((1, 1, d_model)).astype(mx.bfloat16)
    mx.eval(x1)
    y1 = v41_attention.attend(attn, cache, x1, _window_indices)
    assert y1 is not None
    assert v41_attention.stats()['compared'] == 2
    assert not v41_attention.stats()['off']


@pytest.mark.skipif(not mx.metal.is_available(), reason='Metal backend required')
def test_dspark_native_markov_and_gate_match_python(monkeypatch):
    mx.random.seed(3)
    monkeypatch.setenv('QUANTEEZE_DSPARK_NATIVE', 'on')
    monkeypatch.setenv('QUANTEEZE_DSPARK_GATE', '0.5')
    dspark.NATIVE[:] = [None, 0, 0]
    dspark.GATE[0] = None

    ext = dspark._native()
    assert ext is not None
    assert mx.array_equal(ext.probe(mx.array([3.0], dtype=mx.float32)), mx.array([4.0], dtype=mx.float32)).item()

    conf = mx.array([[2.0, 2.0, -1.0, 2.0, 2.0]], dtype=mx.float32)
    assert dspark.gated(conf, 5) == dspark.python_gated(conf, 5, 0.5)

    vocab, rank = 64, 32
    emb = nn.Embedding(vocab, rank)
    head = nn.Linear(rank, vocab, bias=False)
    emb.weight = (mx.random.normal(emb.weight.shape) * 0.1).astype(mx.bfloat16)
    head.weight = (mx.random.normal(head.weight.shape) * 0.1).astype(mx.bfloat16)
    logits = mx.random.normal((1, dspark.BLOCK, vocab)).astype(mx.float32)
    mx.eval(emb.weight, head.weight, logits)

    drafter = dspark.Drafter.__new__(dspark.Drafter)
    drafter.markov_embed = emb
    drafter.markov_head = head

    got_g, gq, gemb = dspark.native_markov(drafter, ext, logits, 7, None, 0)
    ref_g, rqg, remb = drafter.python_markov(logits, 7, None, 0)
    mx.eval(got_g, gemb, ref_g, remb)
    assert mx.array_equal(got_g, ref_g).item()
    assert gq is None and rqg is None
    assert mx.array_equal(gemb, remb).item()

    sampling = dspark.Sampling(temperature=0.8, top_p=0.9, top_k=16, seed=123)
    got_s, got_sq, got_semb = dspark.native_markov(drafter, ext, logits, 7, sampling, 10)
    ref_s, ref_sq, ref_semb = drafter.python_markov(logits, 7, sampling, 10)
    mx.eval(got_s, got_sq, got_semb, ref_s, ref_sq, ref_semb)
    assert mx.array_equal(got_s, ref_s).item()
    assert mx.array_equal(got_sq, ref_sq).item()
    assert mx.array_equal(got_semb, ref_semb).item()


def test_nvrtc_pch_library_builds_and_exports_counts(monkeypatch):
    """`native_ext.library('nvrtc_pch')` compiles without importing MLX and exports `qz_nvrtc_pch_counts`."""
    import ctypes
    import quanteeze

    monkeypatch.setenv('QUANTEEZE_NATIVE', 'on')
    monkeypatch.setenv('QUANTEEZE_NVRTC_PCH', 'on')
    expected = native_ext.library_path('nvrtc_pch')
    lib_path = native_ext.library('nvrtc_pch', force=True, quiet=False)
    if lib_path is None:
        pytest.skip('no C++20 compiler available')
    assert lib_path == expected and lib_path.is_file()
    handle = ctypes.CDLL(str(lib_path))
    counts = (ctypes.c_uint64 * 3)(99, 99, 99)
    handle.qz_nvrtc_pch_counts(counts)
    assert (counts[0], counts[1], counts[2]) == (0, 0, 0)
    assert 'QUANTEEZE_NVRTC_PCH' in quanteeze.control.CHOICES


_STAND_IN_NVRTC = r"""
#include <stddef.h>
#include <string.h>
static int pch_options = 0, status_calls = 0;
static size_t heap = 268435456;
int nvrtcCreateProgram(void **p, const char *src, const char *name, int n, const char *const *c, const char *const *h) {
    *p = (void *)0x1234; return 0;
}
int nvrtcCompileProgram(void *p, int n, const char *const *options) {
    for (int i = 0; i < n; ++i) if (strcmp(options[i], "--pch") == 0) ++pch_options;
    return 0;
}
int nvrtcGetPCHCreateStatus(void *p) { return status_calls++ == 0 ? 14 : 0; }   /* first: heap exhausted */
int nvrtcGetPCHHeapSizeRequired(void *p, size_t *out) { *out = 1073741824; return 0; }
int nvrtcGetPCHHeapSize(size_t *out) { *out = heap; return 0; }
int nvrtcSetPCHHeapSize(size_t bytes) { heap = bytes; return 0; }
int stand_in_pch_options(void) { return pch_options; }
"""

_PCH_PROBE = r"""
import ctypes, json, sys
nvrtc = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
shim = ctypes.CDLL(sys.argv[2])
assert shim.qz_nvrtc_pch_dir(sys.argv[3].encode()) == 1
seen = []
for i in range(5):
    program = ctypes.c_void_p()
    assert shim.nvrtcCreateProgram(ctypes.byref(program), b'', b'custom_kernel_%d' % i, 0, None, None) == 0
    assert shim.nvrtcCompileProgram(program, 0, None) == 0
    seen.append(nvrtc.stand_in_pch_options())
counts = (ctypes.c_uint64 * 3)()
shim.qz_nvrtc_pch_counts(counts)
shim.qz_nvrtc_pch_bytes.restype = ctypes.c_uint64
shim.qz_nvrtc_pch_grown.restype = ctypes.c_uint64
print(json.dumps({'pch': seen, 'counts': list(counts), 'bytes': shim.qz_nvrtc_pch_bytes(),
                  'grown': shim.qz_nvrtc_pch_grown()}))
"""


@pytest.mark.skipif(not sys.platform.startswith('linux') or not (shutil.which('cc') or shutil.which('gcc')),
                    reason='Linux (dl_iterate_phdr) and a C compiler')
def test_nvrtc_pch_files_stay_within_their_bytes_once_the_heap_grows(tmp_path, monkeypatch):
    """Against a stand-in libnvrtc.so.13: the heap grown to 1 GiB for a PCH that did not fit, each PCH made after is
    counted at that heap, and --pch is no longer added once the next would pass MOST_BYTES (8 x 256 MiB)."""
    import json
    import subprocess

    monkeypatch.setenv('QUANTEEZE_NATIVE', 'on')
    shim = native_ext.library('nvrtc_pch', quiet=True)
    if shim is None:
        pytest.skip('no C++20 compiler available')
    source = tmp_path / 'nvrtc.c'
    source.write_text(_STAND_IN_NVRTC)
    library = tmp_path / 'libnvrtc.so.13'
    subprocess.run([shutil.which('cc') or shutil.which('gcc'), '-shared', '-fPIC', '-o', str(library), str(source)],
                   check=True)
    pch = tmp_path / 'pch'
    pch.mkdir()
    done = subprocess.run([sys.executable, '-c', _PCH_PROBE, str(library), str(shim), str(pch)], capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got['grown'] == 1                                   # the first PCH did not fit: grown to 1 GiB
    assert got['pch'] == [1, 2, 3, 3, 3]                       # made two at 1 GiB; a third would pass 2 GiB
    assert got['counts'] == [5, 3, 2]
    assert got['bytes'] == 2 * (1 << 30)
