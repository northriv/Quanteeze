# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for `quanteeze.metal_router`: single-kernel Metal decode routing and weighted expert sum."""
import mlx.core as mx
import numpy as np
import pytest

from quanteeze import cuda_router, kernel_checks, metal_router
from quanteeze.arch.qwen4_exp import _weighted_graph

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason='Metal backend required')


def _theirs(logits, k):
    return cuda_router.original(logits, k)


def _same(a, b):
    return kernel_checks.same_bits(a, b)


@pytest.mark.parametrize('e', [10, 17, 128, 129, 256, 380, 449, 512])
@pytest.mark.parametrize('k', [8, 10])
@pytest.mark.parametrize('dtype', [mx.bfloat16, mx.float16])
def test_metal_route_matches_mlx_ops_bit_for_bit(e, k, dtype):
    rng = np.random.default_rng(e * 31 + k)
    for trial in range(6):
        logits = mx.array(rng.normal(0, 2.0, (1, 1, e)).astype(np.float32)).astype(dtype)
        got = metal_router.route(logits, k)
        want = _theirs(logits, k)
        assert _same(got, want), (e, k, dtype, trial)


def test_metal_route_ties_and_peaked_gates_match_argpartition():
    rng = np.random.default_rng(5)
    tied = mx.array(np.repeat(rng.normal(0, 1, 32), 16)[None, None, :512].astype(np.float32)).astype(mx.bfloat16)
    assert _same(metal_router.route(tied, 10), _theirs(tied, 10))

    flat = mx.zeros((1, 1, 512), mx.bfloat16)
    assert _same(metal_router.route(flat, 10), _theirs(flat, 10))

    peaked = np.full((1, 1, 512), -60.0, np.float32)
    peaked[..., 40] = 30.0
    peaked = mx.array(peaked).astype(mx.bfloat16)
    assert _same(metal_router.route(peaked, 10), _theirs(peaked, 10))


def test_metal_routed_checks_declines_and_compile_trace(monkeypatch):
    choice, checked = kernel_checks.tables('test_metal_router')
    monkeypatch.setattr(metal_router, 'CHOICE', choice)
    monkeypatch.setattr(metal_router, 'CHECKED', checked)
    monkeypatch.setattr(metal_router, 'STATS', dict(served=0, checked=0, mismatched=0))
    monkeypatch.setattr(metal_router, 'SAID', [])
    monkeypatch.setattr(metal_router, 'ON', [True])

    rng = np.random.default_rng(9)
    # Tracing inside mx.compile before checks settle must fall back cleanly without poisoning tables.
    compiled_first = mx.compile(lambda l: metal_router.routed(l, 10, lambda: _theirs(l, 10)))
    trace_in = mx.array(rng.normal(0, 2, (1, 1, 512)).astype(np.float32)).astype(mx.bfloat16)
    traced_out = compiled_first(trace_in)
    assert _same(traced_out, _theirs(trace_in, 10))
    assert metal_router.STATS['mismatched'] == 0
    assert len(choice) == 0

    assert metal_router.pending(512, 10, mx.bfloat16, trace_in)
    for _ in range(metal_router.CHECKS + 2):
        logits = mx.array(rng.normal(0, 2, (1, 1, 512)).astype(np.float32)).astype(mx.bfloat16)
        got = metal_router.routed(logits, 10, lambda l=logits: _theirs(l, 10))
        assert _same(got, _theirs(logits, 10))

    assert metal_router.STATS == dict(served=2, checked=metal_router.CHECKS, mismatched=0)
    assert not metal_router.pending(512, 10, mx.bfloat16, trace_in)

    # Shape / batch / dtype / toggle guards
    two_tok = mx.array(rng.normal(0, 2, (1, 2, 512)).astype(np.float32)).astype(mx.bfloat16)
    batch_two = mx.array(rng.normal(0, 2, (2, 1, 512)).astype(np.float32)).astype(mx.bfloat16)
    too_many_experts = mx.zeros((1, 1, metal_router.MOST + 1), mx.bfloat16)
    assert not metal_router.takes(two_tok, 10)
    assert not metal_router.takes(batch_two, 10)
    assert not metal_router.takes(logits.astype(mx.float32), 10)
    assert not metal_router.takes(logits, 17)
    assert not metal_router.takes(logits, 0)
    assert not metal_router.takes(too_many_experts, 10)
    assert not metal_router.pending(512, 10, mx.bfloat16, two_tok)
    assert not metal_router.pending(512, 10, mx.bfloat16, batch_two)

    monkeypatch.setattr(metal_router, 'ON', [False])
    assert not metal_router.takes(logits, 10)
    assert not metal_router.pending(512, 10, mx.bfloat16, logits)


@pytest.mark.parametrize('k', [6, 8, 10, 16])
@pytest.mark.parametrize('dtype', [mx.bfloat16, mx.float16])
def test_metal_weighted_matches_compiled_and_eager_bit_for_bit(k, dtype):
    rng = np.random.default_rng(k)
    for d in (64, 100, 2048, 2560):
        for _ in range(4):
            output = mx.array(rng.normal(0, 1, (1, 1, k, d)).astype(np.float32)).astype(dtype)
            scores = mx.array(rng.dirichlet(np.ones(k))[None, None].astype(np.float32)).astype(dtype)
            shared = mx.array(rng.normal(0, 1, (1, 1, d)).astype(np.float32)).astype(dtype)
            got = metal_router.weigh(output, scores, shared)
            want_compiled = _weighted_graph(output, scores, shared)
            want_eager = (output * scores[..., None]).sum(axis=-2) + shared
            assert _same(got, want_compiled), (k, d, dtype)
            assert _same(got, want_eager), (k, d, dtype)


def test_metal_weighted_service_and_guards(monkeypatch):
    w_choice, w_checked = kernel_checks.tables('test_metal_weighted')
    monkeypatch.setattr(metal_router._Weighed, 'CHOICE', w_choice)
    monkeypatch.setattr(metal_router._Weighed, 'CHECKED', w_checked)
    monkeypatch.setattr(metal_router._Weighed, 'STATS', dict(served=0, checked=0, mismatched=0))
    monkeypatch.setattr(metal_router._Weighed, 'SAID', [])
    monkeypatch.setattr(metal_router, 'ON', [True])

    rng = np.random.default_rng(17)
    for _ in range(metal_router.CHECKS + 2):
        out = mx.array(rng.normal(0, 1, (1, 1, 10, 128)).astype(np.float32)).astype(mx.bfloat16)
        sc = mx.array(rng.dirichlet(np.ones(10))[None, None].astype(np.float32)).astype(mx.bfloat16)
        sh = mx.array(rng.normal(0, 1, (1, 1, 128)).astype(np.float32)).astype(mx.bfloat16)
        got = metal_router.weighted(out, sc, sh, lambda o=out, s=sc, r=sh: _weighted_graph(o, s, r))
        assert _same(got, _weighted_graph(out, sc, sh))
    assert metal_router._Weighed.STATS == dict(served=2, checked=metal_router.CHECKS, mismatched=0)

    assert not metal_router.weighs(mx.zeros((1, 2, 10, 128), mx.bfloat16), sc, sh)
    assert not metal_router.weighs(out, sc.astype(mx.float16), sh)
    assert not metal_router.weighs(mx.zeros((1, 1, 17, 128), mx.bfloat16), mx.zeros((1, 1, 17), mx.bfloat16), sh)
