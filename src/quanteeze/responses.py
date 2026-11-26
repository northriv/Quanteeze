# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Fixes for `/v1/responses` prompt construction in `mlx_vlm.server`:

1. Keep mid-conversation `developer` / `system` messages in place (as `role='user'`) instead of
   hoisting them into `messages[0]`: clients such as Codex CLI inject `<image_resize_notice>`
   (`role='developer'`) right after `view_image` tool results. Moving them into the leading system
   block shifts every earlier token (and image) by ~50 tokens and invalidates the entire APC
   prompt cache on every image turn. Only leading `system` / `developer` messages belong in
   `messages[0]`.
2. Attach `function_call` / `shell_call` / `apply_patch_call` items onto the immediately preceding
   `assistant` message (`tool_calls`) instead of starting a new `assistant` turn: when commentary
   text (`message(role='assistant')`) precedes a tool call, splitting them emits
   `<|im_end|>\\n<|im_start|>assistant` between the status sentence and `<tool_call>`, teaching
   the model in-context to stop right after writing a status update without emitting tool calls.
"""
import importlib
import sys

CALL_ITEM_TYPES = ('function_call', 'shell_call', 'apply_patch_call')
_INSTRUCTION_ROLES = ('system', 'developer')
_SERVER_MODULES = (
    'mlx_vlm.server',
    'mlx_vlm.server.request_normalization',
    'mlx_vlm.server.responses_state',
    'mlx_vlm.server.openai',
    'mlx_vlm.server.compaction',
    'mlx_vlm.server.app',
    'mlx_vlm.server.anthropic',
)
INSTALLED = [False]


def normalize_instruction_messages(chat_messages, instructions=None):
    """Fold `instructions` and leading `system` / `developer` messages into `chat_messages[0]`,
    leaving mid-conversation `system` / `developer` messages in place as `role='user'`."""
    instruction_parts = [instructions] if instructions else []
    conversation = []
    leading = True

    for message in chat_messages:
        role = message.get('role') if isinstance(message, dict) else None
        if role in _INSTRUCTION_ROLES:
            content = message.get('content')
            if leading:
                if content:
                    instruction_parts.append(str(content))
            elif content:
                conversation.append({**message, 'role': 'user'})
        else:
            leading = False
            conversation.append(message)

    normalized_instructions = '\n\n'.join(instruction_parts) or None
    if normalized_instructions:
        conversation.insert(0, {'role': 'system', 'content': normalized_instructions})
    chat_messages[:] = conversation
    return normalized_instructions


def merge_assistant_call(chat_messages, call_message):
    """Attach `call_message['tool_calls']` onto an immediately preceding assistant message."""
    if chat_messages and isinstance(chat_messages[-1], dict) and chat_messages[-1].get('role') == 'assistant':
        last = chat_messages[-1]
        calls = list(last.get('tool_calls') or ())
        calls.extend(call_message.get('tool_calls') or ())
        last['tool_calls'] = calls
        if last.get('content') is None:
            last['content'] = call_message.get('content', '')
        return
    chat_messages.append(call_message)


def _wrap_append(resp_state):
    """Return a `_append_response_item_to_prompt` wrapper bound to `resp_state` (idempotent)."""
    original = resp_state._append_response_item_to_prompt
    if getattr(original, '_quanteeze_responses', False):
        return original

    def _append_response_item_to_prompt(item, chat_messages, images):
        if isinstance(item, dict) and item.get('type') in CALL_ITEM_TYPES:
            call_message = resp_state._normalize_tool_message({
                'role': 'assistant',
                'content': None,
                'tool_calls': [resp_state._response_call_to_chat_tool_call(item)],
            })
            merge_assistant_call(chat_messages, call_message)
            return
        return original(item, chat_messages, images)

    _append_response_item_to_prompt._quanteeze_responses = True
    return _append_response_item_to_prompt


def _patch_targets(req_norm, resp_state, modules):
    targets = [req_norm, resp_state]
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


def install(req_norm=None, resp_state=None, modules=None):
    """Install `/v1/responses` prompt and instruction normalization fixes in `mlx_vlm.server`."""
    default_mode = req_norm is None and resp_state is None and modules is None
    if default_mode and INSTALLED[0]:
        return True
    if req_norm is None:
        req_norm = importlib.import_module('mlx_vlm.server.request_normalization')
    if resp_state is None:
        resp_state = importlib.import_module('mlx_vlm.server.responses_state')

    append_fn = _wrap_append(resp_state)
    for mod in _patch_targets(req_norm, resp_state, modules):
        if hasattr(mod, '_normalize_instruction_messages'):
            mod._normalize_instruction_messages = normalize_instruction_messages
        if hasattr(mod, '_append_response_item_to_prompt'):
            mod._append_response_item_to_prompt = append_fn

    if default_mode:
        INSTALLED[0] = True
    return True
