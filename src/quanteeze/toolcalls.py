# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Close a tool call the model left open when it ended its turn.

Flash-Next writes tool calls in Qwen3-Coder's XML (mlx-vlm's qwen3_coder
parser):

    <tool_call>
    <function=run_code>
    <parameter=code>
    ...
    </parameter>
    </function>
    </tool_call>

Now and then it ends its turn right after a parameter's value, without the
closing tags.  mlx-vlm's parser then finds no call, and the server fails the
request after streaming part of it ("Generation ended with an incomplete tool
call").  Qwen's own parser (qwen3coder_tool_parser.py, which mlx-vlm's is
modified from) takes a call left open at the end of the output as ending
there.

install(generator_cls) does the same through mlx-vlm's ResponseGenerator.generate.
When the model itself ends a request (finish reason "stop") inside a call
whose function and parameter names are complete, the last token's text gets
the missing closing tags, and the server parses the call as written.  A
request that max_tokens cut off ("length") still fails: its last value may be
cut short.  Thinking text is left alone; the token ids stay the model's.
"""
import copy
import sys

from mlx_vlm.server.responses_state import prompt_has_open_thinking

CALL, CALL_END = '<tool_call>', '</tool_call>'
FUNCTION, FUNCTION_END = '<function=', '</function>'
PARAMETER, PARAMETER_END = '<parameter=', '</parameter>'


def closing_tags(content):
    """The tags that close a Qwen3-Coder call left open at the end of `content`.

    '' when there is none to close: no call open, or a function or parameter
    name cut short.
    """
    start = content.rfind(CALL)
    if start < 0 or CALL_END in content[start:]:
        return ''
    body = content[start + len(CALL):]
    function = body.find(FUNCTION)
    if function < 0 or '>' not in body[function:]:
        return ''
    body = body[function:]
    if FUNCTION_END in body:
        return '\n' + CALL_END
    tags = ''
    parameter = body.rfind(PARAMETER)
    if parameter >= 0 and PARAMETER_END not in body[parameter:]:
        if '>' not in body[parameter:]:
            return ''
        tags = ('' if body.endswith('\n') else '\n') + PARAMETER_END
    return tags + '\n' + FUNCTION_END + '\n' + CALL_END


class ClosingCalls:
    """A request's token iterator that closes a call the model left open at its own end."""

    def __init__(self, tokens, prompt, args, uid=None):
        self._tokens = tokens
        self._uid = uid
        self._text = []
        start = getattr(args, 'thinking_start_token', None)
        end = getattr(args, 'thinking_end_token', None)
        self._thinking_end = end or '</think>'
        self._starts_thinking = prompt_has_open_thinking(
            prompt, bool(getattr(args, 'enable_thinking', False)), start, end)

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._tokens)
        text = getattr(item, 'text', None) or ''
        self._text.append(text)
        if getattr(item, 'finish_reason', None) == 'stop':
            tags = closing_tags(self._content())
            if tags:
                item = copy.copy(item)
                item.text = text + tags
                print(f'quanteeze: request {self._uid}: the model ended its turn inside a '
                      f'tool call; closed it with {tags!r}', file=sys.stderr, flush=True)
        return item

    def __getattr__(self, name):            # close() and anything else the server asks for
        return getattr(self._tokens, name)

    def _content(self):
        """The output after thinking: what the server's tool parser reads."""
        text = ''.join(self._text)
        if self._thinking_end in text:
            return text.rsplit(self._thinking_end, 1)[1]
        return '' if self._starts_thinking else text


def install(generator_cls):
    """Close a call the model leaves open, on every request `generator_cls` serves."""
    from . import stream
    stream.add(generator_cls, 'toolcalls', lambda request, tokens: ClosingCalls(
        tokens, request.prompt, request.args, uid=request.uid))
