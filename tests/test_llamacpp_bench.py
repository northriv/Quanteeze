# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for `bench/llamacpp` (`llama_bench.py`, `batched_bench.py`, `common.py`) and `quanteeze.forced`."""
import statistics as stats
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from quanteeze import forced

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench' / 'llamacpp'))
import batched_bench                                                              # noqa: E402
import common                                                                     # noqa: E402
import llama_bench                                                                # noqa: E402

LOG = """2026-11-22 19:45:05,406 - INFO - Generation queued: request=120fa5040 prompt_tokens=21 max_tokens=512 images=0
2026-11-22 19:45:05,406 - INFO - Prefill started: request=120fa5040 backend=continuous_batching prompt_tokens=21
2026-11-22 19:45:05,520 - INFO - Prefill completed: request=120fa5040 prompt_tokens=21 cached_tokens=0 elapsed=0.114s rate=184.2 tok/s
2026-11-22 19:45:07,327 - INFO - Decode completed: request=120fa5040 generated_tokens=154 elapsed=1.808s rate=84.6 tok/s finish_reason=stop
""".splitlines()

PROGRESS_LOG = """2026-11-22 19:45:00,000 - INFO - Generation queued: request=abcd1234 prompt_tokens=4608 max_tokens=16 images=0
2026-11-22 19:45:01,000 - INFO - Prefill progress: request=abcd1234 tokens=2048/4608 (44.4%)
2026-11-22 19:45:02,000 - INFO - Prefill progress: request=abcd1234 tokens=4096/4608 (88.9%)
2026-11-22 19:45:02,500 - INFO - Prefill completed: request=abcd1234 prompt_tokens=4608 cached_tokens=0 elapsed=2.500s rate=1843.2 tok/s
2026-11-22 19:45:03,500 - INFO - Decode completed: request=abcd1234 generated_tokens=11 elapsed=1.000s rate=10.0 tok/s finish_reason=length
""".splitlines()


@pytest.mark.parametrize('values', [[1.0], [3.0, 5.0], [10.0, 12.5, 11.0, 9.75, 13.0]])
def test_stdev_is_sample_standard_deviation(values):
    assert common.avg(values) == pytest.approx(sum(values) / len(values))
    assert common.stdev(values) == pytest.approx(0.0 if len(values) == 1 else stats.stdev(values))
    assert common.avg([]) == 0.0 and common.stdev([]) == 0.0


def test_llama_bench_names_order_and_timed_modes():
    assert llama_bench.name(512, 0, 0) == 'pp512'
    assert llama_bench.name(0, 128, 0) == 'tg128'
    assert llama_bench.name(512, 128, 4096) == 'pp512+tg128 @ d4096'
    assert llama_bench.tests([512], [128], [(512, 128)], [0, 4096]) == [
        (512, 0, 0),
        (0, 128, 0),
        (512, 128, 0),
        (512, 0, 4096),
        (0, 128, 4096),
        (512, 128, 4096),
    ]

    plain = {'cached': 0, 'prefill_s': 0.5, 'decode_s': 2.0, 'generated': 129, 'progress': {}}
    assert llama_bench.timed(plain, 512, 0, 0, lambda: 0.0) == (512, 0.5, '')
    assert llama_bench.timed(plain, 0, 128, 0, lambda: 0.0) == (128, 2.0, '')
    assert llama_bench.timed(plain, 512, 128, 0, lambda: 0.0) == (640, 2.5, '')

    cached = {'cached': 4096, 'prefill_s': 0.4, 'decode_s': 1.6, 'generated': 129, 'progress': {}}
    assert llama_bench.timed(cached, 512, 128, 4096, lambda: pytest.fail('should not run alone()')) == (
        640,
        2.0,
        '',
    )

    with_progress = common.only(PROGRESS_LOG)
    assert llama_bench.timed(with_progress, 512, 10, 4096, lambda: pytest.fail('should use progress')) == (
        pytest.approx((522, 1.5, ''))
    )

    no_progress = {'cached': 0, 'prefill_s': 2.5, 'decode_s': 1.0, 'generated': 11, 'progress': {}}
    assert llama_bench.timed(no_progress, 512, 10, 3000, lambda: 1.9) == pytest.approx((522, 1.6, '~'))


def test_log_parsing_single_and_multiple():
    got = common.only(LOG)
    assert (got['prompt'], got['cached'], got['generated']) == (21, 0, 154)
    assert got['prefill_s'] == pytest.approx(0.114) and got['decode_s'] == pytest.approx(1.808)
    assert (153 / got['decode_s']) == pytest.approx(84.6, abs=0.05)
    assert got['decode_end'] - got['queued'] == pytest.approx(1.921)

    with pytest.raises(RuntimeError):
        common.only(LOG + [line.replace('120fa5040', '99') for line in LOG])

    p_got = common.only(PROGRESS_LOG)
    assert common.after(p_got, 4096) == pytest.approx(0.5)
    assert common.after(p_got, 3000) is None


def test_batched_columns_union_and_timings():
    row = batched_bench.columns(128, 128, 2, 0.198, 5.227)
    assert row['n_kv'] == 512
    assert row['speed_pp'] == pytest.approx(1295.19, rel=2e-3)
    assert row['t_tg'] == pytest.approx(5.029)
    assert row['speed_tg'] == pytest.approx(50.90, rel=2e-3)
    assert row['speed'] == pytest.approx(97.95, rel=2e-3)

    together = [
        {'queued': 10.0, 'prefill_end': 10.5, 'prefill_s': 0.4, 'decode_end': 14.0},
        {'queued': 10.1, 'prefill_end': 10.9, 'prefill_s': 0.6, 'decode_end': 15.0},
    ]
    assert batched_bench.from_log(together) == pytest.approx((0.8, 5.0))

    one_by_one = [
        {'queued': 10.0, 'prefill_end': 10.5, 'prefill_s': 0.4, 'decode_end': 12.0},
        {'queued': 10.0, 'prefill_end': 12.5, 'prefill_s': 0.4, 'decode_end': 14.0},
    ]
    assert batched_bench.from_log(one_by_one) == pytest.approx((0.8, 4.0))

    assert batched_bench.union([(0, 1), (0.5, 2), (3, 4)]) == pytest.approx(3.0)
    assert batched_bench.from_client([(1.0, 1.2, 3.0), (1.1, 1.6, 2.5)]) == pytest.approx((0.6, 2.0))
    assert batched_bench.from_client([(1.0, None, 2.5), (1.2, 1.5, 2.0)]) == pytest.approx((1.5, 1.5))


def test_vocab_random_excludes_special_and_adds_bos():
    v = object.__new__(common.Vocab)
    v.bos, v.allowed, v.rng = 7, np.array([3, 4, 5], np.int64), np.random.default_rng(0)
    ids = v.random(200)
    assert ids[0] == 7 and set(ids[1:].tolist()) <= {3, 4, 5}
    assert set(v.random(50, first_bos=False).tolist()) <= {3, 4, 5}


def test_forced_processor_and_install_auth_and_hooks(monkeypatch):
    monkeypatch.setenv('QUANTEEZE_CONTROL_TOKEN', 'secret-tok')

    # 1. Forced logits processor enforces exact tokens across shapes and passes through once exhausted
    proc = forced.Forced([2, 0])
    logits_1d = mx.array([1.0, 2.0, -3.0, 4.0], dtype=mx.float32)
    out0 = proc(mx.array([1]), logits_1d)
    assert out0.shape == (4,) and int(mx.argmax(out0).item()) == 2
    logits_2d = mx.array([[10.0, 20.0, 30.0, 40.0]], dtype=mx.float32)
    out1 = proc(mx.array([1, 2]), logits_2d)
    assert out1.shape == (1, 4) and int(mx.argmax(out1, axis=-1).item()) == 0
    out2 = proc(mx.array([1, 2, 0]), logits_2d)
    assert mx.array_equal(out2, logits_2d).item()

    # 2. install() on fake request_normalization and generator_cls
    captured = {}

    class FakeNorm:
        @staticmethod
        def _build_gen_args(request, *args, **kwargs):
            return SimpleNamespace(max_tokens=99)

    class FakeGen:
        def generate(self, prompt, images=None, audio=None, args=None, videos=None):
            raw = self._preprocess_request(prompt, images=images, audio=audio, videos=videos)
            captured['raw'] = raw
            captured['processors'] = self._make_logits_processors(args, raw['input_ids'])
            return 'ctx', iter(())

        def _preprocess_request(self, prompt, images=None, audio=None, videos=None, **kwargs):
            if images:
                return {'input_ids': mx.array([[1]]), 'pixel_values': mx.zeros((1, 2))}
            return {'input_ids': mx.array([[99]]), 'attention_mask': mx.array([[1]], dtype=mx.int32)}

        def _make_logits_processors(self, args, input_ids=None):
            return ['existing']

    forced.install(request_normalization=FakeNorm, generator_cls=FakeGen)

    # Validation rejections
    for bad_spec in (
        {'token': 'wrong', 'prompt': [1], 'output': [2]},
        {'prompt': [1], 'output': [2]},
        {'token': 'secret-tok', 'prompt': [], 'output': [2]},
        {'token': 'secret-tok', 'prompt': [True], 'output': [2]},
        {'token': 'secret-tok', 'prompt': [1], 'output': [1.5]},
    ):
        req = SimpleNamespace(model_extra={forced.FIELD: bad_spec})
        with pytest.raises(ValueError):
            FakeNorm._build_gen_args(req)

    # Valid forced request replaces prompt IDs, sets max_tokens, and appends Forced processor
    ok_req = SimpleNamespace(
        model_extra={forced.FIELD: {'token': 'secret-tok', 'prompt': [10, 11, 12], 'output': [4, 5]}}
    )
    gen_args = FakeNorm._build_gen_args(ok_req)
    assert gen_args.max_tokens == 2
    assert getattr(gen_args, forced.FIELD)['prompt'] == [10, 11, 12]

    fg = FakeGen()
    fg.generate('.', args=gen_args)
    assert captured['raw']['input_ids'].tolist() == [[10, 11, 12]]
    assert captured['raw']['attention_mask'].tolist() == [[1, 1, 1]]
    assert captured['raw']['attention_mask'].dtype == mx.int32
    assert len(captured['processors']) == 2
    assert captured['processors'][0] == 'existing' and isinstance(captured['processors'][1], forced.Forced)
    assert getattr(forced._LOCAL, 'prompt', None) is None

    # Image payloads rejected when forced prompt IDs are supplied
    with pytest.raises(ValueError):
        fg.generate('.', images=['img.png'], args=gen_args)
    assert getattr(forced._LOCAL, 'prompt', None) is None
