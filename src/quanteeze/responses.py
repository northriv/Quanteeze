# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Fixes for `/v1/responses` prompt construction in `mlx_vlm.server` (`QUANTEEZE_RESPONSES_IN_PLACE`):

1. Keep mid-conversation `developer` / `system` messages in place (as `role='user'`) instead of
   letting `_normalize_instruction_messages` hoist them into `messages[0]`: clients such as Codex
   CLI inject `<image_resize_notice>` (`role='developer'`) right after `view_image` tool results.
   Moving them into the leading system block shifts every earlier token (and image) by ~50 tokens
   and invalidates the entire APC prompt cache on every image turn. Only leading `system` /
   `developer` messages stay untouched for stock `_normalize_instruction_messages`.
2. Attach tool-call-only `assistant` turns (`function_call` / `shell_call` / `apply_patch_call`)
   onto the immediately preceding `assistant` message (`tool_calls`) instead of starting a new
   `assistant` turn: when commentary text precedes a tool call, splitting them emits
   `<|im_end|>\\n<|im_start|>assistant` between the status sentence and `<tool_call>`, teaching
   the model in-context to stop right after writing a status update without emitting tool calls.

Wraps `responses_state._response_items_to_chat` only, leaving `/v1/chat/completions` and Anthropic
Messages untouched. Off, stock `mlx_vlm` behavior per request.
"""
import importlib
import sys

from . import variables

_INSTRUCTION_ROLES = ('system', 'developer')
_SERVER_MODULES = (
    'mlx_vlm.server',
    'mlx_vlm.server.responses_state',
    'mlx_vlm.server.openai',
    'mlx_vlm.server.compaction',
    'mlx_vlm.server.app',
)
INSTALLED = [False]


def _call_alone(message):
    """Whether `message` is an `assistant` dict containing tool calls without text or reasoning."""
    if not isinstance(message, dict) or message.get('role') != 'assistant' or not message.get('tool_calls'):
        return False
    content = message.get('content')
    if isinstance(content, list):
        content = ''.join(str(p.get('text', '')) for p in content if isinstance(p, dict))
    return not content and not message.get('reasoning_content') and not message.get('reasoning')


def in_place(messages):
    """Return `/v1/responses` chat messages in client order without mutating `messages` or its dicts.

    Mid-conversation `system` / `developer` messages become `role='user'` in place (empty ones are
    dropped); leading instructions stay untouched so stock `_normalize_instruction_messages` joins
    them at the front. Consecutive or post-commentary tool-call-only `assistant` turns merge onto the
    immediately preceding `assistant` message's `tool_calls`.
    """
    out = []
    began = False
    for message in messages:
        role = message.get('role') if isinstance(message, dict) else None
        if role in _INSTRUCTION_ROLES:
            if began:
                if not message.get('content'):
                    continue
                message = {**message, 'role': 'user'}
        else:
            began = True
        if _call_alone(message) and out and isinstance(out[-1], dict) and out[-1].get('role') == 'assistant':
            before = out[-1]
            merged = {**before, 'tool_calls': list(before.get('tool_calls') or ()) + list(message['tool_calls'])}
            if merged.get('content') is None:
                merged['content'] = ''
            out[-1] = merged
            continue
        out.append(message)
    return out


def _wrap_items_to_chat(resp_state):
    """Return a `_response_items_to_chat` wrapper bound to `resp_state` (idempotent)."""
    original = resp_state._response_items_to_chat
    if getattr(original, '_quanteeze_responses', False):
        return original

    def _response_items_to_chat(items):
        messages, images = original(items)
        if variables.off('QUANTEEZE_RESPONSES_IN_PLACE'):
            return messages, images
        return in_place(messages), images

    _response_items_to_chat._quanteeze_responses = True
    _response_items_to_chat._quanteeze_original = original
    return _response_items_to_chat


def _patch_targets(resp_state, modules):
    targets = [resp_state]
    if modules is not None:
        targets.extend(modules)
    else:
        for name in _SERVER_MODULES:
            try:
                targets.append(importlib.import_module(name))
            except ImportError:
                continue
        for name, mod in list(sys.modules.items()):
            if (name == 'mlx_vlm.server' or name.startswith('mlx_vlm.server.')) and mod is not None:
                targets.append(mod)
    seen, unique = set(), []
    for mod in targets:
        if mod is not None and id(mod) not in seen:
            seen.add(id(mod))
            unique.append(mod)
    return unique


def install(resp_state=None, modules=None):
    """Wrap `_response_items_to_chat` in `/v1/responses` server modules (idempotent)."""
    default_mode = resp_state is None and modules is None
    if default_mode and INSTALLED[0]:
        return True
    if resp_state is None:
        resp_state = importlib.import_module('mlx_vlm.server.responses_state')

    wrapped = _wrap_items_to_chat(resp_state)
    original = getattr(wrapped, '_quanteeze_original', None)
    for mod in _patch_targets(resp_state, modules):
        cur = getattr(mod, '_response_items_to_chat', None)
        if cur is not None and (cur is original or getattr(cur, '_quanteeze_responses', False)):
            mod._response_items_to_chat = wrapped

    if default_mode:
        INSTALLED[0] = True
    return True
