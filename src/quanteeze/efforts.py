# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Reasoning efforts a model's chat template rejects, read as ones it takes.

Clients send OpenAI's reasoning_effort levels: minimal, low, medium, high,
xhigh.  pydantic-ai passes its unified thinking setting through as one of
them.  Flash-Next's chat template takes xhigh (its default with thinking on),
medium and low, and raises on anything else: a request for high or minimal
failed with HTTP 500.  For that model (qwen4_exp) high is read as xhigh and
minimal as low.

install(args_cls) wraps mlx-vlm's GenerationArguments.__post_init__, so a
request's arguments carry the effort the template gets.  use_model(kind)
picks the readings for the model loaded:
its Arch's (quanteeze.arch), or READINGS' for a model type no Arch names; a
kind with none keeps every effort as sent.
"""
from . import arch

READINGS = {}                           # model type -> readings, for models no Arch names
_readings = {}


def use_model(kind):
    """Read efforts as the chat template of a `kind` model needs; returns the readings."""
    known = arch.get(kind)
    _readings.clear()
    _readings.update(known.efforts if known else READINGS.get(kind, {}))
    return dict(_readings)


def install(args_cls):
    """Read every request's reasoning_effort through the loaded model's readings."""
    post_init = args_cls.__post_init__

    def __post_init__(self):
        post_init(self)
        self.reasoning_effort = _readings.get(self.reasoning_effort, self.reasoning_effort)

    args_cls.__post_init__ = __post_init__
