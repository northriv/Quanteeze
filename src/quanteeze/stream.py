# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""The stages each request's tokens pass through (`ResponseGenerator.generate`), in one place and
one order (`quanteeze.stream`).

ORDER, innermost first:
- `mtp` (`quanteeze.mtp`): before the request, a thinking budget taken off while a drafter serves;
- `live` (`quanteeze.shutdown`): the request's uid kept while it generates, to cancel at a signal;
- `toolcalls` (`quanteeze.toolcalls`): closes a tool call the model left open at its end;
- `loops` (`quanteeze.loops`): stops a generation stuck in a repetition loop;
- `reask` (`quanteeze.reask`): holds each DSML calls block and has a broken one written again.
"""
ORDER = ('mtp', 'live', 'toolcalls', 'loops', 'reask')


class Request:
    """State shared across the stages wrapping a single `ResponseGenerator.generate` call."""

    def __init__(self, prompt, images, args):
        self.prompt, self.images, self.args, self.ctx = prompt, images, args, None
        self.generator = None

    @property
    def uid(self):
        return getattr(self.ctx, 'uid', None)


def add(generator_cls, name, wrap=None, before=None):
    """Add (or replace) stage `name` on `generator_cls` (`ResponseGenerator`)."""
    if name not in ORDER:
        raise ValueError(f'stage {name!r}: one of {ORDER}')
    stages = generator_cls.__dict__.get('_quanteeze_stages')
    if stages is None:
        stages = {}
        generate = generator_cls.generate

        def _generate(self, prompt, images=None, audio=None, args=None, videos=None):
            request = Request(prompt, images, args)
            request.generator = self
            run = [stages[n] for n in ORDER if n in stages]
            for first, _ in run:
                if first is not None:
                    first(request)
            ctx, tokens = generate(self, prompt, images=images, audio=audio, args=args, videos=videos)
            request.ctx = ctx
            for _, then in run:
                if then is not None:
                    tokens = then(request, tokens)
            return ctx, tokens

        generator_cls.generate = _generate
        generator_cls._quanteeze_stages = stages
    stages[name] = (before, wrap)


def stages(generator_cls):
    """The stages added to `generator_cls`, innermost first."""
    have = generator_cls.__dict__.get('_quanteeze_stages') or {}
    return [n for n in ORDER if n in have]
