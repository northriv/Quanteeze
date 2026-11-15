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
_THINKING_INSTALLED = False


def use_model(kind):
    """Read efforts as the chat template of a `kind` model needs; returns the readings."""
    known = arch.get(kind)
    _readings.clear()
    _readings.update(known.efforts if known else READINGS.get(kind, {}))
    return dict(_readings)


def _first_literal(raw):
    """Extract a single literal string (or first literal of a list) from a response_template anchor."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        return raw[0]
    return None


def _thinking_prefix(tokenizer, thinking_start_token=None):
    """Build the synthetic response prefix needed when a template pre-opens its thinking block."""
    spec = getattr(tokenizer, 'response_template', None)
    anchor_lit = ''
    open_lit = None
    if isinstance(spec, dict):
        anchor = _first_literal(spec.get('start_anchor'))
        if anchor is not None:
            anchor_lit = anchor
        fields = spec.get('fields')
        if isinstance(fields, dict):
            for key in ('reasoning_content', 'reasoning'):
                fld = fields.get(key)
                if isinstance(fld, dict):
                    lit = _first_literal(fld.get('open'))
                    if lit is not None:
                        open_lit = lit
                        break
    if open_lit is None:
        open_lit = (thinking_start_token or '<think>') + '\n'
    return anchor_lit + open_lit


def install_thinking():
    """Patch `mlx_vlm.server` so `chat_template_kwargs.enable_thinking` propagates to
    `GenerationArguments` and `tokenizer.response_template` parsers receive the pre-opened thinking
    prefix (`<think>\\n`) when generation starts inside a thinking block."""
    global _THINKING_INSTALLED
    if _THINKING_INSTALLED:
        return
    try:
        import importlib
        srv_pkg = importlib.import_module('mlx_vlm.server')
        anthropic_mod = importlib.import_module('mlx_vlm.server.anthropic')
        app_mod = importlib.import_module('mlx_vlm.server.app')
        openai_mod = importlib.import_module('mlx_vlm.server.openai')
        req_norm = importlib.import_module('mlx_vlm.server.request_normalization')
        resp_state = importlib.import_module('mlx_vlm.server.responses_state')
    except ImportError:
        return

    orig_build_gen_args = req_norm._build_gen_args

    def patched_build_gen_args(request, *args, **kwargs):
        if not req_norm._request_field_is_set(request, 'enable_thinking'):
            ctk = getattr(request, 'chat_template_kwargs', None)
            if ctk is not None and hasattr(ctk, 'model_dump'):
                ctk = ctk.model_dump(exclude_none=True)
            if isinstance(ctk, dict) and ctk.get('enable_thinking') is not None:
                try:
                    request.enable_thinking = bool(ctk['enable_thinking'])
                except Exception:
                    pass
                else:
                    fields_set = getattr(request, 'model_fields_set', None)
                    if isinstance(fields_set, set):
                        fields_set.add('enable_thinking')
        return orig_build_gen_args(request, *args, **kwargs)

    req_norm._build_gen_args = patched_build_gen_args

    orig_make_stream_state = resp_state.make_response_stream_state

    def patched_make_response_stream_state(
        processor,
        enable_thinking=False,
        thinking_start_token=None,
        thinking_end_token=None,
    ):
        tokenizer = resp_state._response_template_tokenizer(processor)
        if tokenizer is not None and hasattr(tokenizer, 'get_response_parser'):
            prefix = _thinking_prefix(tokenizer, thinking_start_token) if enable_thinking else ''
            try:
                return resp_state.ResponseTemplateStreamState(tokenizer.get_response_parser(prefix=prefix))
            except (AttributeError, TypeError, ValueError):
                pass
        return orig_make_stream_state(
            processor,
            enable_thinking,
            thinking_start_token,
            thinking_end_token,
        )

    for mod in (resp_state, srv_pkg, app_mod, openai_mod, anthropic_mod):
        if hasattr(mod, 'make_response_stream_state'):
            mod.make_response_stream_state = patched_make_response_stream_state

    orig_split_thinking = resp_state._split_thinking

    def patched_split_thinking(
        text,
        thinking_start_token=None,
        thinking_end_token=None,
        starts_in_thinking=False,
        processor=None,
    ):
        if not text:
            return None, text
        tokenizer = resp_state._response_template_tokenizer(processor)
        if tokenizer is not None and hasattr(tokenizer, 'parse_response'):
            markers = resp_state.ThinkingStreamState._build_open_close_markers(
                thinking_start_token, thinking_end_token
            )
            stripped = text.lstrip()
            has_open_in_text = any(stripped.startswith(sm) for sm, _ in markers)
            has_unopened_close = False
            for sm, em in markers:
                s_pos = text.find(sm)
                e_pos = text.find(em)
                if e_pos >= 0 and (s_pos < 0 or e_pos < s_pos):
                    has_unopened_close = True
                    break
            pre_opened = (bool(starts_in_thinking) or has_unopened_close) and not has_open_in_text
            prefix = _thinking_prefix(tokenizer, thinking_start_token) if pre_opened else ''
            try:
                parsed = tokenizer.parse_response(text, prefix=prefix)
                if isinstance(parsed, dict) and (
                    'content' in parsed or 'reasoning' in parsed or 'reasoning_content' in parsed
                ):
                    reasoning = parsed.get('reasoning_content') or parsed.get('reasoning')
                    content = parsed.get('content')
                    if (reasoning is None or isinstance(reasoning, str)) and (
                        content is None or isinstance(content, str)
                    ):
                        r_clean = reasoning.strip() if reasoning else None
                        c_clean = content.strip() if content else ''
                        if r_clean is not None or not has_unopened_close:
                            return r_clean, c_clean
            except (AttributeError, TypeError, ValueError):
                pass
            if has_unopened_close:
                return orig_split_thinking(
                    text,
                    thinking_start_token,
                    thinking_end_token,
                    starts_in_thinking,
                    processor=None,
                )
        return orig_split_thinking(
            text,
            thinking_start_token,
            thinking_end_token,
            starts_in_thinking,
            processor=processor,
        )

    resp_state._split_thinking = patched_split_thinking
    if hasattr(app_mod, '_split_thinking_text'):
        app_mod._split_thinking_text = patched_split_thinking
    _THINKING_INSTALLED = True


def install(args_cls):
    """Read every request's reasoning_effort through the loaded model's readings, and install
    thinking-block parsing fixes for `mlx_vlm.server`."""
    if not getattr(args_cls.__post_init__, '_quanteeze_installed', False):
        post_init = args_cls.__post_init__

        def __post_init__(self):
            post_init(self)
            self.reasoning_effort = _readings.get(self.reasoning_effort, self.reasoning_effort)

        __post_init__._quanteeze_installed = True
        args_cls.__post_init__ = __post_init__
    install_thinking()
