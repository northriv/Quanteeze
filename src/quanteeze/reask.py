# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""A broken tool call asked again before the client sees it.

install(generator_cls) adds the stage 'reask' (quanteeze.stream: outside the loop guard, so a
generation cut by it comes here as it ended).  Where the request's processor writes DSML
(quanteeze.dsml), the tokens pass as they come until a calls block starts; the block is held until
it closes or the generation ends, then checked (problems()): each call parsed as the server will
parse it, its tool among the request's and its arguments among the tool's, no markup left over, no
closing tag more than there were openings, no call written twice, and not the same call (tool and
arguments) as each of the last REPEATS assistant turns.  A sound block goes on.  A broken one is
dropped -- the client has not seen it -- and the model asked to write it again from where it
started (the prompt and the output before the block), up to QUANTEEZE_REASK times, sampling at
RETRY_TEMPERATURE and RETRY_TOP_P with another seed.
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
_AGAIN = contextvars.ContextVar('quanteeze_reask_again', default=False)   # a block written again: not wrapped again


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
    """A request's token iterator that holds each DSML calls block, and has a broken one written
    again (see the module's doc)."""

    def __init__(self, request, tokens, fmt, attempts, body=None):
        self._request, self._tokens, self._attempts = request, tokens, attempts
        self._opening, self._closing, self._module = fmt
        self._body = body
        self._text = []                  # all the text gone on so far
        self._held = []                  # the block's items, held
        self._ready = []                 # items to give out
        self._used = 0                   # tokens the request has generated so far
        self._tried = 0
        self._done = False
        self._in_block = False

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
        self._held.append(item)
        text = ''.join(getattr(i, 'text', '') or '' for i in self._held)
        ended = bool(getattr(item, 'finish_reason', None))
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

    def _again(self):
        """The block dropped, the model asked to write it again from where it started."""
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
        prompt = r.prompt + ''.join(self._text)
        again = _AGAIN.set(True)
        try:
            _, self._tokens = r.generator.generate(prompt, images=r.images, args=args)
        finally:
            _AGAIN.reset(again)
        self._held, self._in_block = [], False


def install(generator_cls):
    """Ask a broken DSML calls block again, on every request `generator_cls` serves
    (QUANTEEZE_REASK times; off: never)."""
    from . import stream
    if variables.off('QUANTEEZE_REASK'):
        return

    def wrap(request, tokens):
        fmt = _format(getattr(request, 'generator', None))
        attempts = variables.integer('QUANTEEZE_REASK')
        if fmt is None or attempts <= 0 or _AGAIN.get() or not isinstance(request.prompt, str):
            return tokens
        return Reasking(request, tokens, fmt, attempts, getattr(request, 'body', None))
    stream.add(generator_cls, 'reask', wrap)
