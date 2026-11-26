# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for `quanteeze.responses` (`/v1/responses` prefix stability and tool-call grouping)."""
import json

import jinja2

from quanteeze import responses

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


def test_normalize_instruction_messages_keeps_mid_conversation_developer_in_place():
    chat = [
        {'role': 'developer', 'content': 'Base context.'},
        {'role': 'system', 'content': 'System rule.'},
        {'role': 'user', 'content': 'Inspect the repository.'},
        {'role': 'assistant', 'content': 'Running checks.', 'tool_calls': [{'function': {'name': 'run', 'arguments': {'cmd': 'ls'}}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': '[Image output attached in the next message]'},
        {'role': 'user', 'content': [{'type': 'image'}]},
        {'role': 'developer', 'content': '<image_resize_notice>Resized image 1.</image_resize_notice>'},
        {'role': 'developer', 'content': ''},
        {'role': 'system', 'content': 'Mid-turn note.'},
    ]

    merged = responses.normalize_instruction_messages(chat, instructions='Global instructions.')

    assert merged == 'Global instructions.\n\nBase context.\n\nSystem rule.'
    assert chat[0] == {'role': 'system', 'content': 'Global instructions.\n\nBase context.\n\nSystem rule.'}
    assert [m['role'] for m in chat] == ['system', 'user', 'assistant', 'tool', 'user', 'user', 'user']
    assert chat[1]['content'] == 'Inspect the repository.'
    assert chat[-2] == {'role': 'user', 'content': '<image_resize_notice>Resized image 1.</image_resize_notice>'}
    assert chat[-1] == {'role': 'user', 'content': 'Mid-turn note.'}


def test_install_patches_all_server_modules_and_merges_assistant_calls():
    from mlx_vlm.server import compaction, openai, request_normalization, responses_state

    assert responses.install() is True
    assert responses.install() is True
    assert request_normalization._normalize_instruction_messages is responses.normalize_instruction_messages
    assert openai._normalize_instruction_messages is responses.normalize_instruction_messages
    assert compaction._normalize_instruction_messages is responses.normalize_instruction_messages
    assert getattr(responses_state._append_response_item_to_prompt, '_quanteeze_responses', False) is True

    # 1. Commentary + single call -> single assistant turn
    msgs, imgs = responses_state._response_items_to_chat([
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
    msgs, _ = responses_state._response_items_to_chat([
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

    # 3. Parallel calls without commentary -> single assistant turn with empty content
    msgs, _ = responses_state._response_items_to_chat([
        {'type': 'function_call', 'call_id': 'c1', 'name': 'exec_command', 'arguments': '{"cmd":"a"}'},
        {'type': 'apply_patch_call', 'call_id': 'c2', 'patch': '*** Begin Patch'},
    ])
    assert len(msgs) == 1
    assert msgs[0]['role'] == 'assistant'
    assert msgs[0]['content'] == ''
    assert [tc['id'] for tc in msgs[0]['tool_calls']] == ['c1', 'c2']

    # 4. User message in between prevents merging an unrelated subsequent call
    msgs, _ = responses_state._response_items_to_chat([
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Done.'}]},
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Next step.'}]},
        {'type': 'function_call', 'call_id': 'c3', 'name': 'exec_command', 'arguments': '{"cmd":"id"}'},
    ])
    assert [m['role'] for m in msgs] == ['assistant', 'user', 'assistant']
    assert 'tool_calls' not in msgs[0]
    assert [tc['id'] for tc in msgs[2]['tool_calls']] == ['c3']


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
