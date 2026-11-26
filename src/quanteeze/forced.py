# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Control-token-gated forced token-ID prompts and deterministic token outputs (`quanteeze.forced`).

Used by `bench/llamacpp` (`--feed forced`) to supply exact prompt token IDs and force each generated
token ID step by step through a logits processor without altering normal requests.
"""
import hmac
import sys
import threading

import mlx.core as mx
import numpy as np

FIELD = 'quanteeze_forced'
_LOCAL = threading.local()
_INSTALLED = []


def _field(request):
    extra = getattr(request, 'model_extra', None) or {}
    spec = extra.get(FIELD) if isinstance(extra, dict) else None
    return spec if spec is not None else getattr(request, FIELD, None)


def _allowed(spec):
    from . import control
    given = str(spec.get('token') or '')
    return bool(given) and hmac.compare_digest(given, control.token())


def _int_ids(values, label):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f'{FIELD}: {label} must be a non-empty list of integer token IDs')
    for v in values:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f'{FIELD}: {label} entries must be integers')
    return np.asarray(values, np.int64)


class Forced:
    """Logits processor forcing `output` token by token and passing through once exhausted."""

    def __init__(self, output):
        self.output = _int_ids(output, 'output')
        self.n = 0

    def __call__(self, tokens, logits):
        if self.n >= self.output.size:
            return logits
        t = int(self.output[self.n])
        self.n += 1
        forced = mx.where(
            mx.arange(logits.shape[-1]) == t,
            mx.array(0.0, mx.float32),
            mx.array(-mx.inf, mx.float32),
        )
        return mx.broadcast_to(forced, logits.shape)


def install(request_normalization=None, generator_cls=None):
    """Install `quanteeze_forced` request parsing and generator hooks (idempotent per target)."""
    if request_normalization is None or generator_cls is None:
        if _INSTALLED:
            return
        from mlx_vlm.server import generation as _gen, request_normalization as _req_norm
        request_normalization = _req_norm if request_normalization is None else request_normalization
        generator_cls = _gen.ResponseGenerator if generator_cls is None else generator_cls
        _INSTALLED.append(True)

    own_build = request_normalization._build_gen_args
    if not getattr(own_build, '_quanteeze_forced', False):
        def build(request, *args, **kwargs):
            gen = own_build(request, *args, **kwargs)
            spec = _field(request)
            if spec is not None:
                if not isinstance(spec, dict) or not _allowed(spec):
                    raise ValueError(f'{FIELD}: valid control token required')
                prompt = _int_ids(spec.get('prompt'), 'prompt')
                output = _int_ids(spec.get('output'), 'output')
                spec = dict(spec, prompt=prompt.tolist(), output=output.tolist())
                setattr(gen, FIELD, spec)
                gen.max_tokens = len(spec['output'])
            return gen

        build._quanteeze_forced = True
        request_normalization._build_gen_args = build

    if not generator_cls.__dict__.get('_quanteeze_forced', False):
        own_generate = generator_cls.generate
        own_pre = generator_cls._preprocess_request
        own_make = generator_cls._make_logits_processors

        def generate(self, prompt, images=None, audio=None, args=None, videos=None, **kwargs):
            spec = getattr(args, FIELD, None)
            _LOCAL.prompt = None if spec is None else np.asarray(spec['prompt'], np.int64)
            try:
                return own_generate(self, prompt, images, audio, args, videos, **kwargs)
            finally:
                _LOCAL.prompt = None

        def preprocess(self, prompt, images=None, audio=None, videos=None, **kwargs):
            raw = own_pre(self, prompt, images, audio, videos, **kwargs)
            ids = getattr(_LOCAL, 'prompt', None)
            if ids is not None:
                if raw.get('pixel_values') is not None:
                    raise ValueError(f'{FIELD}: image inputs are not supported with forced token IDs')
                raw = dict(raw)
                raw['input_ids'] = mx.array(ids[None])
                if raw.get('attention_mask') is not None:
                    dtype = np.array(raw['attention_mask']).dtype
                    raw['attention_mask'] = mx.array(np.ones((1, ids.size), dtype=dtype))
            return raw

        def make(self, args, input_ids=None):
            processors = own_make(self, args, input_ids)
            spec = getattr(args, FIELD, None)
            if spec is not None:
                processors = list(processors) + [Forced(spec['output'])]
            return processors

        generator_cls.generate = generate
        generator_cls._preprocess_request = preprocess
        generator_cls._make_logits_processors = make
        generator_cls._quanteeze_forced = True
        print(f'quanteeze: requests may carry {FIELD} (with control token) for forced token-ID benchmarks',
              file=sys.stderr, flush=True)
