# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""DeepSeek V4's and V4.1's own chat formats in the server: their prompts and their DSML tool calls.

DeepSeek V4 and V4.1 ship no Jinja chat template.  Each repository gives its format as Python:
encode_messages() turns OpenAI-style messages into the prompt, and parse_tool_calls() reads the
calls a completion makes.  The two are not the same format (FORMATS):
- DeepSeek V4 (deepseek-ai/DeepSeek-V4-Flash-0731, encoding/encoding_dsv4.py): calls in a
  <｜DSML｜tool_calls> block; reasoning effort high or max;
- DeepSeek V4.1 (deepseek-ai/DeepSeek-V4.1-Flash-1017, encoding/encoding.py): calls in a
  <｜DSML｜ calls> block, its tags with a leading space; reasoning effort a budget from 1 to
  100 (EFFORTS_V41); a system message may come mid-conversation (<｜System｜>), and a tool's
  result goes into a user turn.
Each file comes with quanteeze, unmodified and under its own MIT License, in
quanteeze/vendor/deepseek_v4 and vendor/deepseek_v41 (see their NOTICE.md).  load() imports
one from there, or from the format's variable (QUANTEEZE_DSV4_ENCODING,
QUANTEEZE_DSV41_ENCODING: a directory or the file) for another version.

Without it mlx-vlm renders DeepSeek through a generic template with no tools and finds no
tool-call parser, so an agent's tools never reach the model and its calls never come back.
install() puts the formats into mlx-vlm's server.  use_model(model, processor) marks a
deepseek_v4 or deepseek_v41 model's processor when its encoding is there:
- the prompt (mlx_vlm.server.apply_chat_template):
  - the request's messages, as the server prepared them, go back into the encoder's shape
    (messages()).  The server turns tool-call arguments into mappings and a user's content
    into parts;
  - the tools go on the system message, on an empty one first when there is none;
  - tool-call arguments become JSON again, and text parts are joined.  An image becomes a
    mark: the model reads none here;
  - it is encoded in thinking mode when the request enables thinking, else in chat mode.
    The reasoning effort is the format's nearest the one asked (Format.efforts);
- the tool calls: the format's parser, 'dsml' or 'dsml_v41' (mlx_vlm.server.
  _infer_tool_parser_from_processor, load_tool_module).  It reads a DSML block with the
  encoder's own parser, and when that refuses one, by its tags alone, so a call written a
  little off is not lost.
Any other model goes through mlx-vlm as before.
"""
import importlib.util
import json
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

from . import variables

VENDOR = Path(__file__).resolve().parent / 'vendor'
DEFAULT = VENDOR / 'deepseek_v4'
DSML = '｜DSML｜'
CALLS, CALLS_END = f'<{DSML}tool_calls>', f'</{DSML}tool_calls>'
IMAGE = '[an image the user sent: this model cannot see images]'      # DeepSeek V4 Flash reads none
EFFORTS = {'none': None, 'minimal': 'low', 'low': 'low', 'medium': 'low', 'high': 'high', 'xhigh': 'max',
           'max': 'max'}
# V4.1's budget runs 1-100; its own levels are low 50, high 75 (its default) and max 100
EFFORTS_V41 = {'none': None, 'minimal': 25, 'low': 'low', 'medium': 62, 'high': 'high', 'xhigh': 88, 'max': 'max'}


@dataclass(frozen=True)
class Format:
    """One DeepSeek chat format: the model type it is for, its encoder's file, and the DSML
    block its calls come in."""
    model_type: str
    name: str
    file: str
    default: Path
    variable: str
    block: str                                          # the DSML block's tag: tool_calls, or V4.1's ' calls'
    parser: str                                         # the tool parser's name in mlx-vlm
    efforts: dict = field(default_factory=dict)
    numeric: bool = False                               # an effort may be a number (V4.1's 1-100)
    image: str = IMAGE

    @property
    def calls(self):
        return f'<{DSML}{self.block}>', f'</{DSML}{self.block}>'


V4 = Format('deepseek_v4', 'DeepSeek V4', 'encoding_dsv4.py', DEFAULT, 'QUANTEEZE_DSV4_ENCODING',
            'tool_calls', 'dsml', EFFORTS)
V41 = Format('deepseek_v41', 'DeepSeek V4.1', 'encoding.py', VENDOR / 'deepseek_v41', 'QUANTEEZE_DSV41_ENCODING',
             ' calls', 'dsml_v41', EFFORTS_V41, numeric=True,
             image='[an image the user sent: images are not served to this model here]')
FORMATS = {f.model_type: f for f in (V4, V41)}
SYSTEM_TOKEN = '<｜System｜>'                        # in V4.1's vocabulary, not V4's
USER_TOKEN = '<｜User｜>'                            # a user turn's first token, V4's and V4.1's (APC checkpoints)

_ENCODING = {}


def load(path=None, model_type='deepseek_v4'):
    """The encoder module of `model_type`'s format, its file at `path` (the file, or a directory
    holding it or its encoding/ folder), the format's variable, or its default (the copy that
    comes with quanteeze); None when it is not there."""
    fmt = FORMATS[model_type]
    path = Path(path or variables.get(fmt.variable) or fmt.default).expanduser()
    for candidate in (path, path / fmt.file, path / 'encoding' / fmt.file):
        if candidate.is_file():
            path = candidate.resolve()
            break
    else:
        return None
    if path not in _ENCODING:
        spec = importlib.util.spec_from_file_location(f'encoding_{fmt.parser}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._quanteeze_format = fmt
        _ENCODING[path] = module
    return _ENCODING[path]


def format_of(encoding):
    """The Format an encoder module load() gave is for (V4 for one from elsewhere)."""
    return getattr(encoding, '_quanteeze_format', V4)


def by_tokens(tok):
    """'deepseek_v41' or 'deepseek_v4' when `tok` has DeepSeek's DSML token (V4.1's also has
    <｜System｜>), else None."""
    def has(token):
        found = tok.convert_tokens_to_ids(token) if hasattr(tok, 'convert_tokens_to_ids') else None
        return found is not None and found != getattr(tok, 'unk_token_id', None)
    if not has(DSML):
        return None
    return 'deepseek_v41' if has(SYSTEM_TOKEN) else 'deepseek_v4'


def _text(content, image=IMAGE):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                kind = str(part.get('type'))
                if 'image' in kind:
                    parts.append(image)
                elif kind in ('text', 'input_text', 'None'):
                    parts.append(part.get('text') or part.get('content') or '')
        return ''.join(parts)
    return '' if content is None else str(content)


def _blocks(content):
    """A user turn's parts as V4.1's encoder takes them, its images as image blocks: the encoder
    places each as its image token, in order, and mlx-vlm's server hands the processor the images
    of the user turns in the same order.  None where the turn carries no image."""
    if not isinstance(content, list) or not any(isinstance(p, dict) and 'image' in str(p.get('type'))
                                                for p in content):
        return None
    out = []
    for part in content:
        if isinstance(part, str):
            out.append({'type': 'text', 'text': part})
        elif isinstance(part, dict) and 'image' in str(part.get('type')):
            url = part.get('image_url')
            url = (url or {}).get('url') if isinstance(url, dict) else url
            out.append({'type': 'image_url', 'image_url': {'url': url or 'image'}})
        elif isinstance(part, dict) and str(part.get('type')) in ('text', 'input_text', 'None'):
            out.append({'type': 'text', 'text': part.get('text') or part.get('content') or ''})
    return out


def messages(chat, tools=None, image=IMAGE, images=False):
    """OpenAI-style messages, as mlx-vlm's server prepared them, in the shape the encoder takes;
    with `images`, a user turn's images as image blocks (_blocks()), else each a mark (`image`)."""
    out = []
    for m in chat:
        role = m.get('role')
        blocks = _blocks(m.get('content')) if images and role == 'user' else None
        msg = {'role': role, 'content': blocks if blocks is not None else _text(m.get('content'), image)}
        if m.get('tool_calls'):
            calls = []
            for c in m['tool_calls']:
                function = dict(c.get('function') or {})
                args = function.get('arguments')
                function['arguments'] = args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False)
                calls.append({**c, 'function': function})
            msg['tool_calls'] = calls
        if m.get('tool_call_id') is not None:
            msg['tool_call_id'] = m['tool_call_id']
        reasoning = m.get('reasoning_content') or m.get('reasoning')
        if role == 'assistant' and reasoning:
            msg['reasoning_content'] = reasoning
        out.append(msg)
    if tools:
        if not out or out[0]['role'] != 'system':
            out.insert(0, {'role': 'system', 'content': ''})
        out[0]['tools'] = list(tools)
    return out


def effort(fmt, asked):
    """The reasoning effort `fmt`'s encoder takes for the one asked: its nearest level, or, for
    V4.1, a whole number from 1 to 100 as it is; None (the encoder's default) for any other."""
    if asked is None:
        return None
    if fmt.numeric:
        try:
            number = int(str(asked).strip())
            return number if 1 <= number <= 100 else None
        except ValueError:
            pass
    return fmt.efforts.get(str(asked).lower())


def prompt(encoding, chat, tools=None, enable_thinking=False, reasoning_effort=None, add_generation_prompt=True,
           images=False):
    """The prompt the model of `encoding`'s format takes for a request: the encoder's own, in
    thinking mode when `enable_thinking`, its user turns' images placed as image tokens with
    `images` (a model that reads them), else marked.  Without `add_generation_prompt`, without the
    assistant's opening."""
    fmt = format_of(encoding)
    mode = 'thinking' if enable_thinking else 'chat'
    text = encoding.encode_messages(messages(chat, tools, fmt.image, images), thinking_mode=mode,
                                    reasoning_effort=effort(fmt, reasoning_effort))
    if not add_generation_prompt:
        for opening in (encoding.ASSISTANT_SP_TOKEN + encoding.thinking_start_token,
                        encoding.ASSISTANT_SP_TOKEN + encoding.thinking_end_token):
            if text.endswith(opening):
                return text[:-len(opening)]
    return text


# ---------------------------------------------------------------- the tool-call parsers

# V4's tags and V4.1's (a space after ｜DSML｜) alike
_INVOKE = re.compile(rf'<{DSML}\s*invoke\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</{DSML}\s*invoke>', re.DOTALL)
_PARAMETER = re.compile(rf'<{DSML}\s*parameter\s+name="(?P<name>[^"]+)"(?:\s+string="(?P<string>true|false)")*\s*>'
                        rf'(?P<value>.*?)</{DSML}\s*parameter>', re.DOTALL)


def _lenient(block):
    """The calls in a DSML block by its tags alone: string="true" values as written, others as
    JSON when they read as JSON."""
    calls = []
    for invoke in _INVOKE.finditer(block):
        arguments = {}
        for p in _PARAMETER.finditer(invoke.group('body')):
            value = p.group('value')
            if p.group('string') != 'true':
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            arguments[p.group('name')] = value
        calls.append({'name': invoke.group('name'), 'arguments': arguments})
    return calls


def parse_tool_call(text, tools=None, encoding=None, model_type='deepseek_v4'):
    """mlx-vlm's parser interface: the calls in the inside of a DSML block of `model_type`'s
    format."""
    encoding = encoding or load(model_type=model_type)
    end = FORMATS[model_type].calls[1]
    calls = []
    if encoding is not None:
        try:
            _, _, found = encoding.parse_tool_calls(0, '>' + text + end)
            calls = [{'name': c['name'], 'arguments': json.loads(c['arguments'])} for c in found]
        except (ValueError, AssertionError):
            calls = []
        if any('"' in c['name'] or any('"' in k for k in c['arguments']) for c in calls):
            calls = []          # a tag written twice: name="topic" string="true" string="true" read as 'topic" string="true'
    if not calls:
        calls = _lenient(text)
    if not calls:
        raise ValueError('no DSML invocation found')
    return calls[0] if len(calls) == 1 else calls


def _parser_module(fmt):
    module = types.ModuleType(f'mlx_vlm.tools.parsers.{fmt.parser}')
    module.__doc__ = f"{fmt.name}'s DSML tool calls (quanteeze.dsml)."
    module.tool_call_start, module.tool_call_end = fmt.calls
    module.parse_tool_call = lambda text, tools=None: parse_tool_call(text, tools, model_type=fmt.model_type)
    return module


PARSER_NAME = V4.parser
PARSERS = {f.model_type: _parser_module(f) for f in FORMATS.values()}
parser, parser_v41 = PARSERS['deepseek_v4'], PARSERS['deepseek_v41']


# ---------------------------------------------------------------- the server

def _encoding_of(processor):
    return getattr(processor, '_quanteeze_dsml', None) or getattr(
        getattr(processor, 'tokenizer', None), '_quanteeze_dsml', None)


IMAGE_TOKEN = '<｜deepseek_image｜>'                 # what DeepSeek V4's vision builds put in a prompt for an image
PIXEL_OPTIONS = ('min_pixels', 'max_pixels', 'resized_height', 'resized_width')


def reads_images(processor):
    """Whether the model's tokenizer has DeepSeek V4's image token: only a vision build does."""
    tok = getattr(processor, 'tokenizer', processor)
    found = tok.convert_tokens_to_ids(IMAGE_TOKEN) if hasattr(tok, 'convert_tokens_to_ids') else None
    return found is not None and found != getattr(tok, 'unk_token_id', None)


def text_only(processor):
    """Hand `processor` no images from now on.  mlx-vlm's DeepSeek V4 processor puts an image
    token in the prompt for every image a request carries, and a build without vision (DeepSeek
    V4 Flash) has no such token: the request failed (500, 'Token not found in tokenizer'), as
    soon as an agent sent a plot.  The prompt keeps a mark where the image was (IMAGE)."""
    base = type(processor)

    def __call__(self, text, images=None, padding=True, padding_side='left', add_special_tokens=False,
                 return_tensors='mlx', **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in PIXEL_OPTIONS + ('videos', 'audio', 'audios')}
        return base.__call__(self, text, None, padding=padding, padding_side=padding_side,
                             add_special_tokens=add_special_tokens, return_tensors=return_tensors, **kwargs)

    processor.__class__ = type(f'TextOnly{base.__name__}', (base,), {'__call__': __call__})
    return processor


def sees_images(model):
    """Whether a DeepSeek V4.1 model holds its vision tower: arch.deepseek_v41.prepare let none go
    (QUANTEEZE_V41_READS_IMAGES)."""
    vision = model.get('vision') if isinstance(model, dict) else getattr(model, 'vision', None)
    try:
        return bool(vision is not None and vision.parameters())
    except AttributeError:
        return False


def use_model(model, processor):
    """Give a deepseek_v4 or deepseek_v41 model's processor DeepSeek's own format, when the
    encoding is there, and keep images from a build that reads none.  V4.1 holding its vision
    tower (sees_images()) reads a user turn's images, placed as its image tokens; without it, its
    prompt marks an image rather than placing its token, so its processor gets none either.
    Returns a line to report, or None for any other model."""
    fmt = FORMATS.get(getattr(getattr(model, 'config', None), 'model_type', None))
    if fmt is None:
        return None
    notes = []
    sees = fmt is V41 and sees_images(model)
    if (not sees) if fmt is V41 else not reads_images(processor):
        text_only(processor)
        why = 'its vision tower let go (QUANTEEZE_V41_READS_IMAGES=off)' if fmt is V41 \
            else f'this build has no {IMAGE_TOKEN}'
        notes.append(f'images are left out ({why}; the prompt marks each with {fmt.image})')
    elif sees:
        notes.append('a user turn\'s images read, each placed as its image token (QUANTEEZE_V41_READS_IMAGES)')
    tok = getattr(processor, 'tokenizer', processor)
    user = tok.convert_tokens_to_ids(USER_TOKEN) if hasattr(tok, 'convert_tokens_to_ids') else None
    if user is not None and user != getattr(tok, 'unk_token_id', None):
        for m in (model, getattr(model, 'language_model', None)):
            if m is not None:
                object.__setattr__(m, '_quanteeze_turn_starts', frozenset({int(user)}))
    encoding = load(model_type=fmt.model_type)
    if encoding is None:
        return '; '.join([f'no {fmt.name} encoding at {variables.get(fmt.variable) or fmt.default}: '
                          'its prompts go through the generic template, without tools'] + notes)
    for target in (processor, getattr(processor, 'tokenizer', None)):
        if target is not None:
            object.__setattr__(target, '_quanteeze_dsml', encoding)
            object.__setattr__(target, '_quanteeze_images', sees)
    return '; '.join([f'{fmt.name} prompts and DSML tool calls from {encoding.__file__}'] + notes)


def install(server):
    """Route mlx-vlm's server (the mlx_vlm.server package) through DeepSeek's format for a
    processor use_model() marked; any other goes as before."""
    template, infer = server.apply_chat_template, server._infer_tool_parser_from_processor
    for module in PARSERS.values():
        sys.modules[module.__name__] = module

    def apply_chat_template(processor, config, chat, *args, **kwargs):
        encoding = _encoding_of(processor)
        if encoding is None or kwargs.get('return_messages'):
            return template(processor, config, chat, *args, **kwargs)
        if isinstance(chat, str):
            chat = [{'role': 'user', 'content': chat}]
        add = args[0] if args else kwargs.get('add_generation_prompt', True)
        return prompt(encoding, chat, kwargs.get('tools'), bool(kwargs.get('enable_thinking')),
                      kwargs.get('reasoning_effort'), add_generation_prompt=add,
                      images=bool(getattr(processor, '_quanteeze_images', False)))

    def infer_tool_parser(processor, override=None):
        encoding = _encoding_of(processor)
        if override is None and encoding is not None:
            return format_of(encoding).parser
        return infer(processor, override=override)

    server.apply_chat_template = apply_chat_template
    server._infer_tool_parser_from_processor = infer_tool_parser
