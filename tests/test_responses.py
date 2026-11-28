# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for `quanteeze.responses` (`/v1/responses` prefix stability and tool-call grouping)."""
import copy
import json

import jinja2

from quanteeze import control, responses, variables

QWEN_TEMPLATE = """\
{%- macro render_content(content) -%}
    {%- if content is string -%}{{- content -}}
    {%- elif content is iterable and content is not mapping -%}
        {%- for item in content -%}
            {%- if item.type == 'image' or 'image' in item -%}{{- '<|vision_start|><|image_pad|><|vision_end|>' -}}
            {%- elif 'text' in item -%}{{- item.text -}}{%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- if tools -%}
    {{- '<|im_start|>system\\n# Tools\\n' -}}
    {%- for tool in tools -%}{{- '\\n' ~ (tool | tojson) -}}{%- endfor -%}
    {%- if messages and messages[0].role == 'system' -%}{{- '\\n\\n' ~ render_content(messages[0].content)|trim -}}{%- endif -%}
    {{- '<|im_end|>\\n' -}}
{%- elif messages and messages[0].role == 'system' -%}
    {{- '<|im_start|>system\\n' ~ render_content(messages[0].content)|trim ~ '<|im_end|>\\n' -}}
{%- endif -%}
{%- for message in messages -%}
    {%- set content = render_content(message.content)|trim -%}
    {%- if message.role == 'system' -%}
        {%- if not loop.first -%}{{- raise_exception('System message must be at the beginning.') -}}{%- endif -%}
    {%- elif message.role == 'user' -%}
        {{- '<|im_start|>user\\n' ~ content ~ '<|im_end|>\\n' -}}
    {%- elif message.role == 'assistant' -%}
        {%- set rc = (message.reasoning_content or '')|trim if message.reasoning_content is string else '' -%}
        {{- '<|im_start|>assistant\\n<think>\\n' ~ rc ~ '\\n</think>\\n\\n' ~ content -}}
        {%- if message.tool_calls -%}
            {%- for tc in message.tool_calls -%}
                {%- set fn = tc.function if tc.function is defined else tc -%}
                {{- ('\\n\\n' if loop.first and content else '\\n' if not loop.first else '') -}}
                {{- '<tool_call>\\n<function=' ~ fn.name ~ '>\\n' -}}
                {%- if fn.arguments -%}
                    {%- for k, v in fn.arguments|items -%}
                        {{- '<parameter=' ~ k ~ '>\\n' ~ (v if v is string else (v | tojson)) ~ '\\n</parameter>\\n' -}}
                    {%- endfor -%}
                {%- endif -%}
                {{- '</function>\\n</tool_call>' -}}
            {%- endfor -%}
        {%- endif -%}
        {{- '<|im_end|>\\n' -}}
    {%- elif message.role == 'tool' -%}
        {%- if loop.previtem and loop.previtem.role != 'tool' -%}{{- '<|im_start|>user' -}}{%- endif -%}
        {{- '\\n<tool_response>\\n' ~ content ~ '\\n</tool_response>' -}}
        {%- if loop.last or loop.nextitem.role != 'tool' -%}{{- '<|im_end|>\\n' -}}{%- endif -%}
    {%- else -%}
        {{- raise_exception('Unexpected message role: ' ~ message.role) -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n<think>\\n' -}}{%- endif -%}"""


def _render_qwen(messages, tools=None, add_generation_prompt=True):
    env = jinja2.Environment(autoescape=False)
    env.filters['tojson'] = lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True)
    env.filters['items'] = lambda d: list(d.items()) if isinstance(d, dict) else []

    def raise_exception(msg):
        raise ValueError(msg)

    env.globals['raise_exception'] = raise_exception
    return env.from_string(QWEN_TEMPLATE).render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
    )


def _call(name, call_id):
    return {
        'role': 'assistant',
        'content': None,
        'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': {}}}],
    }


def test_in_place_keeps_mid_conversation_developer_without_mutating_inputs():
    messages = [
        {'role': 'developer', 'content': 'Base context.'},
        {'role': 'system', 'content': 'System rule.'},
        {'role': 'user', 'content': 'Inspect the repository.'},
        _call('view_image', 'c1'),
        {'role': 'tool', 'tool_call_id': 'c1', 'content': '[Image output attached in the next message]'},
        {'role': 'user', 'content': [{'type': 'image'}]},
        {'role': 'developer', 'content': '<image_resize_notice>Resized image 1.</image_resize_notice>'},
        {'role': 'developer', 'content': ''},
        {'role': 'system', 'content': 'Mid-turn note.'},
    ]
    snapshot = copy.deepcopy(messages)

    out = responses.in_place(messages)

    assert messages == snapshot
    assert [m['role'] for m in out] == ['developer', 'system', 'user', 'assistant', 'tool', 'user', 'user', 'user']
    assert out[0]['content'] == 'Base context.'
    assert out[1]['content'] == 'System rule.'
    assert out[-2] == {'role': 'user', 'content': '<image_resize_notice>Resized image 1.</image_resize_notice>'}
    assert out[-1] == {'role': 'user', 'content': 'Mid-turn note.'}


def test_in_place_merges_commentary_or_reasoned_assistant_calls_and_respects_boundaries():
    # Preceding commentary + reasoning merge with subsequent call-only turns cleanly.
    assistant_with_commentary = {'role': 'assistant', 'content': 'Reading files.', 'reasoning_content': 'need context'}
    messages = [
        {'role': 'user', 'content': 'Fix it.'},
        assistant_with_commentary,
        _call('read', 'c1'),
        _call('grep', 'c2'),
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'x'},
        _call('write', 'c3'),
    ]
    snapshot = copy.deepcopy(messages)
    out = responses.in_place(messages)

    assert messages == snapshot
    assert [m['role'] for m in out] == ['user', 'assistant', 'tool', 'assistant']
    assert out[1]['content'] == 'Reading files.'
    assert out[1]['reasoning_content'] == 'need context'
    assert [tc['id'] for tc in out[1]['tool_calls']] == ['c1', 'c2']
    assert [tc['id'] for tc in out[3]['tool_calls']] == ['c3']

    # Consecutive call-only turns without commentary merge and normalize None content to ''.
    no_commentary = [_call('a', 'c1'), _call('b', 'c2')]
    merged = responses.in_place(no_commentary)
    assert len(merged) == 1
    assert merged[0]['content'] == ''
    assert [tc['id'] for tc in merged[0]['tool_calls']] == ['c1', 'c2']

    # Call messages that carry their own text or reasoning are not merged into the previous assistant turn.
    own_text = {**_call('a', 'c1'), 'content': 'and now'}
    own_reasoning = {**_call('b', 'c2'), 'reasoning': 'because'}
    separate = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': 'x'},
        own_text,
        own_reasoning,
    ]
    assert responses.in_place(separate) == separate


def test_install_patches_all_server_modules_and_leaves_instruction_normalizer_stock(monkeypatch):
    from mlx_vlm.server import compaction, openai, request_normalization, responses_state

    assert 'QUANTEEZE_RESPONSES_IN_PLACE' in variables.SETTINGS
    assert variables.SETTINGS['QUANTEEZE_RESPONSES_IN_PLACE'].default == 'on'
    assert variables.SETTINGS['QUANTEEZE_RESPONSES_IN_PLACE'].when == 'request'
    assert control.CHOICES['QUANTEEZE_RESPONSES_IN_PLACE'] == ('on', 'off')

    assert responses.install() is True
    assert responses.install() is True

    wrapped = responses_state._response_items_to_chat
    assert getattr(wrapped, '_quanteeze_responses', False) is True
    assert openai._response_items_to_chat is wrapped
    assert compaction._response_items_to_chat is wrapped

    # Stock _normalize_instruction_messages is untouched so /v1/chat/completions behavior never changes.
    assert request_normalization._normalize_instruction_messages.__module__ == 'mlx_vlm.server.request_normalization'
    assert openai._normalize_instruction_messages is request_normalization._normalize_instruction_messages
    assert compaction._normalize_instruction_messages is request_normalization._normalize_instruction_messages

    # 1. Commentary + single call -> single assistant turn
    msgs, imgs = wrapped([
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Checking files next.'}]},
        {'type': 'function_call', 'call_id': 'c1', 'name': 'exec_command', 'arguments': '{"cmd":"ls"}'},
    ])
    assert imgs == []
    assert msgs == [{
        'role': 'assistant',
        'content': 'Checking files next.',
        'tool_calls': [{'type': 'function', 'id': 'c1', 'function': {'name': 'exec_command', 'arguments': {'cmd': 'ls'}}}],
    }]

    # 2. Commentary + parallel calls -> single assistant turn preserving call order
    msgs, _ = wrapped([
        {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'plan'}]},
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Running two tools.'}]},
        {'type': 'function_call', 'call_id': 'c1', 'name': 'exec_command', 'arguments': '{"cmd":"pwd"}'},
        {'type': 'shell_call', 'call_id': 'c2', 'action': {'commands': ['date']}},
    ])
    assert len(msgs) == 1
    assert msgs[0]['role'] == 'assistant'
    assert msgs[0]['content'] == 'Running two tools.'
    assert [tc['id'] for tc in msgs[0]['tool_calls']] == ['c1', 'c2']
    assert msgs[0]['tool_calls'][1]['function']['name'] == 'shell'

    # 3. User message in between prevents merging an unrelated subsequent call
    items = [
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Done.'}]},
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Next step.'}]},
        {'type': 'function_call', 'call_id': 'c3', 'name': 'exec_command', 'arguments': '{"cmd":"id"}'},
    ]
    msgs, _ = wrapped(items)
    assert [m['role'] for m in msgs] == ['assistant', 'user', 'assistant']
    assert 'tool_calls' not in msgs[0]
    assert [tc['id'] for tc in msgs[2]['tool_calls']] == ['c3']

    # 4. Per-request toggle QUANTEEZE_RESPONSES_IN_PLACE=off returns stock mlx-vlm output.
    dev_items = [
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'hi'}]},
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Looking.'}]},
        {'type': 'function_call', 'call_id': 'c1', 'name': 'ls', 'arguments': '{}'},
        {'type': 'message', 'role': 'developer', 'content': [{'type': 'input_text', 'text': 'a notice'}]},
    ]
    monkeypatch.setenv('QUANTEEZE_RESPONSES_IN_PLACE', 'off')
    assert wrapped(dev_items) == wrapped._quanteeze_original(dev_items)


def test_qwen_rendered_prompt_prefix_and_image_positions_stay_stable_across_turns():
    from mlx_vlm.server import openai, responses_state

    responses.install()
    tools = [{'type': 'function', 'function': {'name': 'view_image', 'parameters': {'type': 'object'}}}]
    instructions = 'System baseline prompt.'

    turn1_items = [
        {'type': 'message', 'role': 'developer', 'content': [{'type': 'input_text', 'text': 'Workspace context.'}]},
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Render icon and check.'}]},
    ]
    turn2_items = turn1_items + [
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Opening first image.'}]},
        {'type': 'function_call', 'call_id': 'call_1', 'name': 'view_image', 'arguments': '{"path":"img1.png"}'},
        {'type': 'function_call_output', 'call_id': 'call_1', 'output': [{'type': 'input_image', 'image_url': 'data:image/png;base64,AAA'}]},
        {'type': 'message', 'role': 'developer', 'content': [{'type': 'input_text', 'text': '<image_resize_notice>Image 1 resized.</image_resize_notice>'}]},
    ]
    turn3_items = turn2_items + [
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Checking second image.'}]},
        {'type': 'function_call', 'call_id': 'call_2', 'name': 'view_image', 'arguments': '{"path":"img2.png"}'},
        {'type': 'function_call_output', 'call_id': 'call_2', 'output': [{'type': 'input_image', 'image_url': 'data:image/png;base64,BBB'}]},
        {'type': 'message', 'role': 'developer', 'content': [{'type': 'input_text', 'text': '<image_resize_notice>Image 2 resized.</image_resize_notice>'}]},
    ]

    prompts, image_counts = [], []
    for items in (turn1_items, turn2_items, turn3_items):
        msgs, imgs = responses_state._response_items_to_chat(items)
        openai._normalize_instruction_messages(msgs, instructions)
        prompts.append(_render_qwen(msgs, tools=tools))
        image_counts.append(len(imgs))

    assert image_counts == [0, 1, 2]
    p1, p2, p3 = prompts

    # Commentary and tool_call share one assistant turn with no <|im_end|> separator between them.
    assert 'Opening first image.\n\n<tool_call>\n<function=view_image>' in p2
    assert 'Opening first image.<|im_end|>' not in p2

    # Turn 1 text prefix up to generation prompt is preserved verbatim in Turn 2.
    p1_prefix = p1.removesuffix('<|im_start|>assistant\n<think>\n')
    assert p2.startswith(p1_prefix)

    # Turn 2 prefix (including Image 1 and Image 1 resize notice) is preserved verbatim in Turn 3.
    p2_prefix = p2.removesuffix('<|im_start|>assistant\n<think>\n')
    assert p3.startswith(p2_prefix)
