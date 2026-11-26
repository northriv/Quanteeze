# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Broken tool calls and premature end-of-thinking stops asked again before the client sees them.

install(generator_cls) adds the stage 'reask' (quanteeze.stream: outside the loop guard, so a
generation cut by it comes here as it ended):
- Where a generation starts inside an open thinking block (`<think>`, `<|channel>thought`,
  `<|START_THINKING|>`, or `args.thinking_start_token`) and hits `finish_reason == 'stop'` before
  closing it -- producing neither an answer nor a tool call, which halts clients such as Codex CLI
  on `/v1/responses` -- the premature stop is withheld and generation continues from the prompt plus
  the reasoning written so far, up to `QUANTEEZE_REASK` times, sampling at `RETRY_TEMPERATURE` and
  `RETRY_TOP_P` with another seed.
- Where the request's processor writes DSML (quanteeze.dsml), the tokens pass as they come until a
  calls block starts; the block is held until it closes or the generation ends, then checked
  (problems()): each call parsed as the server will parse it, its tool among the request's and its
  arguments among the tool's, no markup left over, no closing tag more than there were openings, no
  call written twice, and not the same call (tool and arguments) as each of the last REPEATS
  assistant turns.  A sound block goes on.  A broken one is dropped -- the client has not seen it --
  and the model asked to write it again from where it started.
"""
import contextvars
import copy
import json
import re
import sys

from . import variables

REPEATS = 5                     # the same call as each of this many assistant turns before it: going round
REPEAT_ATTEMPTS = 1             # asked again only once for that alone: polling a status is no fault
RETRY_TEMPERATURE, RETRY_TOP_P = 1.0, 0.95
SEED_STEP = 7919
THINKING_MARKERS = (
    ('<|channel>thought', '<channel|>'),
    ('<think>', '</think>'),
    ('<|START_THINKING|>', '<|END_THINKING|>'),
)
TOOL_CALL_ENDS = ('</tool_call>',)
_AGAIN = contextvars.ContextVar('quanteeze_reask_again', default=False)   # a retry's inner generate: not wrapped again


def _thinking_pairs(args):
    """`(open_marker, close_marker)` pairs for `args`."""
    start = getattr(args, 'thinking_start_token', None) if args is not None else None
    end = getattr(args, 'thinking_end_token', None) if args is not None else None
    pairs = [(start, end)] if start and end else []
    for pair in THINKING_MARKERS:
        if pair not in pairs:
            pairs.append(pair)
    return tuple(pairs)


def _prompt_open_thinking(prompt, pairs):
    """Closing marker if `prompt` ends with an open thinking marker from `pairs`, else None."""
    if not isinstance(prompt, str):
        return None
    stripped = prompt.rstrip()
    for start, end in pairs:
        if stripped.endswith(start):
            return end
    return None


def _format(generator):
    """(the calls block's opening, its closing, the parser module) where `generator`'s processor
    writes DSML, else None."""
    from . import dsml
    encoding = dsml._encoding_of(getattr(generator, 'processor', None))
    if encoding is None:
        return None
    fmt = dsml.format_of(encoding)
    module = dsml.PARSERS[fmt.model_type]
    return module.tool_call_start, module.tool_call_end, module


def _tools(body):
    """{tool name: (its parameters, whether others are refused)} of a chat request's tools."""
    out = {}
    for t in (body or {}).get('tools') or []:
        f = t.get('function') if isinstance(t, dict) else None
        if isinstance(f, dict) and f.get('name'):
            p = f.get('parameters') or {}
            out[f['name']] = (set((p.get('properties') or {}).keys()), p.get('additionalProperties') is False)
    return out


def _signature(name, arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    return name, json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def _recent_calls(body, turns=REPEATS):
    """The call signatures of each of the last `turns` assistant turns of a chat request."""
    out = []
    for m in reversed((body or {}).get('messages') or []):
        if isinstance(m, dict) and m.get('role') == 'assistant' and m.get('tool_calls'):
            out.append({_signature(c['function']['name'], c['function'].get('arguments'))
                        for c in m['tool_calls'] if isinstance(c, dict) and c.get('function')})
            if len(out) == turns:
                break
    return out


def problems(block, opening, closing, module, body=None):
    """What is wrong with a calls block as written (`block`: from its opening, its closing
    included where it was written), [] when nothing is."""
    from mlx_vlm.tools.extractor import process_tool_calls
    tools = (body or {}).get('tools') or []
    found = []
    if closing not in block:
        found.append('the block not closed')
    mark = '｜DSML｜'
    dsml = mark + (' ' if opening[1 + len(mark):2 + len(mark)] == ' ' else '')     # V4.1's has a space after
    opened = len(re.findall(re.escape('<' + dsml) + r'parameter\b', block))
    shut = block.count('</' + dsml + 'parameter>')
    if shut > opened:
        found.append(f'{shut} parameter tags closed, {opened} opened')
    parsed = process_tool_calls(block, module, tools)
    if parsed.remaining_text and parsed.remaining_text.strip():
        found.append('markup left over: ' + parsed.remaining_text.strip()[:60])
    if not parsed.calls:
        found.append('no call parsed')
    known = _tools(body)
    seen = []
    for c in parsed.calls:
        f = c.get('function') or {}
        name, arguments = f.get('name'), f.get('arguments')
        if known and name not in known:
            found.append(f'no tool {name!r}')
        elif known:
            params, closed = known[name]
            try:
                keys = set(json.loads(arguments or '{}'))
            except (ValueError, TypeError):
                keys = set()
                found.append(f'{name}: arguments not JSON')
            if closed and keys - params:
                found.append(f'{name}: no parameter {sorted(keys - params)[0]!r}')
        seen.append(_signature(name, arguments))
    if len(seen) != len(set(seen)):
        found.append('a call written twice')
    recent = _recent_calls(body)
    if seen and len(recent) == REPEATS and all(set(seen) & turn for turn in recent):
        found.append(f'the same call as the last {REPEATS} turns')
    return found


class Reasking:
    """A request's token iterator that continues premature stops inside `<think>` and re-asks broken
    DSML tool-call blocks (see the module's doc)."""

    def __init__(self, request, tokens, fmt, attempts, body=None):
        self._request, self._tokens, self._attempts = request, tokens, attempts
        self._opening, self._closing, self._module = fmt if fmt is not None else (None, None, None)
        self._body = body
        self._text = []                  # all the text gone on so far
        self._held = []                  # the block's items, held
        self._ready = []                 # items to give out
        self._used = 0                   # tokens the request has generated so far
        self._tried = 0
        self._done = False
        self._in_block = False
        self._think_pairs = _thinking_pairs(getattr(request, 'args', None))
        self._think_close = _prompt_open_thinking(getattr(request, 'prompt', None), self._think_pairs)
        self._thinking_done = False
        self._saw_content = False
        self._think_tail = ''

    def __iter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self._tokens, name)

    def __next__(self):
        while not self._ready:
            if self._done:
                raise StopIteration
            self._pull()
        return self._ready.pop(0)

    def _update_thinking(self, piece):
        if not piece or self._thinking_done:
            return
        if self._think_close is not None:
            window = self._think_tail + piece
            if self._think_close in window:
                self._think_close = None
                self._thinking_done = True
                self._think_tail = ''
            else:
                self._think_tail = window[-max(len(self._think_close), 1):]
            return
        if self._saw_content:
            return
        window = self._think_tail + piece
        stripped = window.lstrip()
        if not stripped:
            self._think_tail = ''
            return
        for start, end in self._think_pairs:
            if stripped.startswith(start):
                after = stripped[len(start):]
                if end in after:
                    self._thinking_done = True
                    self._think_tail = ''
                else:
                    self._think_close = end
                    self._think_tail = after[-max(len(end), 1):]
                return
        if any(start.startswith(stripped) for start, _ in self._think_pairs):
            self._think_tail = window
        else:
            self._saw_content = True
            self._think_tail = ''

    def _has_closed_tool_call(self, piece=''):
        ends = tuple(TOOL_CALL_ENDS) + ((self._closing,) if self._closing else ())
        text = ''.join(self._text) + ''.join(getattr(i, 'text', '') or '' for i in self._held) + (piece or '')
        return any(end in text for end in ends)

    def _continue_thinking(self, item):
        for prev in self._held:
            self._text.append(getattr(prev, 'text', '') or '')
            self._ready.append(prev)
        self._held = []
        piece = getattr(item, 'text', '') or ''
        if piece:
            self._text.append(piece)
            clean = copy.copy(item)
            clean.finish_reason = None
            self._ready.append(clean)
        force_close = self._tried >= 1
        self._tried += 1
        if force_close:
            close_marker = self._think_close
            accum = ''.join(self._text)
            need_nl = bool(accum and not accum.endswith('\n'))
            closing = ('\n' if need_nl else '') + close_marker + '\n\n'
            self._text.append(closing)
            bridge = copy.copy(item)
            bridge.text = closing
            bridge.finish_reason = None
            bridge.token_count = 0
            self._ready.append(bridge)
            self._think_close = None
            self._thinking_done = True
            self._think_tail = ''
            print(f'quanteeze: request {self._request.uid}: ended in thinking with no answer '
                  f'({self._tried} of {self._attempts}): closing {close_marker} and continuing '
                  f'from {self._used} tokens',
                  file=sys.stderr, flush=True)
            self._again(in_thinking=False)
        else:
            print(f'quanteeze: request {self._request.uid}: ended in thinking with no answer '
                  f'({self._tried} of {self._attempts}): continuing from {self._used} tokens',
                  file=sys.stderr, flush=True)
            self._again(in_thinking=True)

    def _pull(self):
        try:
            item = next(self._tokens)
        except StopIteration:
            if self._in_block:
                self._settle()
            elif self._held:
                self._release()
            self._done = True
            return
        self._used += getattr(item, 'token_count', 1)
        self._update_thinking(getattr(item, 'text', '') or '')
        finish = getattr(item, 'finish_reason', None)
        max_tok = getattr(getattr(self._request, 'args', None), 'max_tokens', None)
        if (not self._in_block and finish == 'stop' and self._think_close is not None
                and self._tried < self._attempts and (max_tok is None or self._used < max_tok)
                and not self._has_closed_tool_call(getattr(item, 'text', '') or '')):
            self._continue_thinking(item)
            return
        self._held.append(item)
        if self._opening is None:
            self._release()
            return
        text = ''.join(getattr(i, 'text', '') or '' for i in self._held)
        ended = bool(finish)
        if not self._in_block:
            if self._opening in text:
                self._in_block = True
            elif ended or not any(text.endswith(self._opening[:k]) for k in range(1, len(self._opening))):
                self._release()
                return
            else:
                return
        if self._closing in text[text.find(self._opening):] or ended:
            self._settle()

    def _block(self, items):
        text = ''.join(getattr(i, 'text', '') or '' for i in items)
        at = text.find(self._opening)
        return text[at:] if at >= 0 else text, text[:at] if at > 0 else ''

    def _settle(self):
        block, before = self._block(self._held)
        found = problems(block, self._opening, self._closing, self._module, self._body)
        limit = min(self._attempts, REPEAT_ATTEMPTS) if found and all('the same call' in p for p in found) \
            else self._attempts
        if not found or self._tried >= limit:
            self._release()
            return
        print(f'quanteeze: request {self._request.uid}: tool calls asked again ({self._tried + 1} of '
              f'{limit}): {"; ".join(found)}', file=sys.stderr, flush=True)
        if before:
            self._text.append(before)
        self._tried += 1
        self._again()

    def _release(self):
        for i in self._held:
            self._text.append(getattr(i, 'text', '') or '')
            self._ready.append(i)
        last = getattr(self._held[-1], 'finish_reason', None) if self._held else None
        self._held, self._in_block = [], False
        if last:
            self._done = True

    def _again(self, in_thinking=False):
        """Continue or rewrite from `prompt + ''.join(self._text)` with bumped seed and temperature."""
        try:
            self._tokens.close()
        except Exception:                         # noqa: BLE001
            pass
        r, args = self._request, copy.copy(self._request.args)
        if args is not None:
            if (getattr(args, 'temperature', 0) or 0) < RETRY_TEMPERATURE:
                args.temperature, args.top_p = RETRY_TEMPERATURE, RETRY_TOP_P
            args.seed = (getattr(args, 'seed', None) or 0) + SEED_STEP * self._tried
            if getattr(args, 'max_tokens', None):
                args.max_tokens = max(1, args.max_tokens - self._used)
            args.thinking_budget = None
            args._quanteeze_in_thinking = bool(in_thinking)
        prompt = r.prompt + ''.join(self._text)
        again = _AGAIN.set(True)
        try:
            _, self._tokens = r.generator.generate(prompt, images=r.images, args=args)
        finally:
            _AGAIN.reset(again)
        self._held, self._in_block = [], False


def install(generator_cls):
    """Continue premature stops inside `<think>` and re-ask broken DSML tool-call blocks on every
    request `generator_cls` serves (`QUANTEEZE_REASK` times; `off`: never)."""
    from . import stream
    if variables.off('QUANTEEZE_REASK', variables.at_start()):
        return

    def wrap(request, tokens):
        attempts = variables.integer('QUANTEEZE_REASK', variables.at_start())
        if attempts <= 0 or _AGAIN.get() or not isinstance(request.prompt, str):
            return tokens
        fmt = _format(getattr(request, 'generator', None))
        return Reasking(request, tokens, fmt, attempts, getattr(request, 'body', None))
    stream.add(generator_cls, 'reask', wrap)
