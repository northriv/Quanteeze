# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""An MCP server of one tool, generate_image: a picture drawn on this machine by one of the image models found --
Z-Image-Turbo, FLUX.2-klein-4B or Qwen-Image-2.1 through mlx-vlm's pipelines (quanteeze.vlm_image), or
Nemotron-Labs-Image-30B-A3B (quanteeze.nemotron_image) -- for Codex or any MCP client (as a standalone stdio tool,
since Quanteeze serves text/VLM generation without returning images).  Their licenses differ, and the tool tells
each: Z-Image-Turbo and FLUX.2-klein-4B are Apache-2.0 (commercial use allowed), and are offered first;
Qwen-Image-2.1 (Qwen Research License) and Nemotron (the NVIDIA License: src/quanteeze/vendor/nemotron_image/
LICENSE and NOTICE.md) are for research or evaluation only.

    python -m quanteeze.image_mcp [--nemotron <converted checkpoint>] [--decoder <Nemotron-Labs-Image-Tokenizer>]
                                  [--checkpoint <name>=<path> ...] [--default <name>]
                                  [--out ~/.quanteeze/images] [--keep-gb 2] [--write-gb 2]

Codex (~/.codex/config.toml):

    [mcp_servers.images]
    command = "<repository>/.venv/bin/python"
    args = ["-m", "quanteeze.image_mcp"]
    tool_timeout_sec = 300

Stdio, newline-delimited JSON-RPC 2.0, one request at a time (no threads): initialize, ping, tools/list, tools/call;
notifications answered with nothing.  Each model is found in the model folders (LM Studio's, an external disk's
lmstudio, ~/.quanteeze/models) unless given; tools/list offers those found.  A model is loaded for each image and
let go after, where the memory has room.

generate_image(prompt, model, width=1024, height=1024, seed, path): writes a PNG and answers with its path, the
model, its license, the seed and the seconds (and, for Nemotron, whether every row came whole).  path: where to
write it (absolute; .png; not a file there already), else <out>/<time>-<model>-<words of the prompt>-<seed>.png.
Bounded writes: <out> kept at most --keep-gb of images, and this process writes at most --write-gb; past either, it
refuses and says so -- it removes nothing.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import convert

NAME, VERSION = 'quanteeze-images', '2'
PROTOCOL = '2025-06-18'


def roots():
    """The model folders looked in: LM Studio's, a Mac's external disks' lmstudio, and where the home page converts to
    (convert.fallback_root)."""
    out = [Path.home() / '.lmstudio' / 'models', convert.fallback_root()]
    try:
        out += sorted(Path('/Volumes').glob('*/lmstudio'))
    except OSError:
        pass
    return [r for r in out if r.is_dir()]


class Nemotron:
    name = 'nemotron-image'
    about = 'Nemotron-Labs-Image-30B-A3B: photographs and illustrations, about 25 s; sides 512-1536 in steps of 64'
    license = 'NVIDIA License: research or evaluation only'

    def __init__(self, model=None, decoder=None):
        self.given = (model, decoder)

    def find(self):
        from . import nemotron_image
        model, decoder = self.given
        if model is None or decoder is None:
            found = nemotron_image.find(roots())
            model, decoder = model or found[0], decoder or found[1]
        return model, decoder

    def check_size(self, width, height):
        from . import nemotron_image
        nemotron_image.check_size(width, height)

    def room(self, paths):
        from . import nemotron_image
        return nemotron_image.room(*paths)

    def draw(self, paths, prompt, width, height, seed, out):
        from . import nemotron_image
        r = nemotron_image.draw(*paths, prompt, width, height, seed, out=out, keep=('png',))
        whole = ('every row whole' if r['whole'] else
                 f'not whole: {r["rows"]} rows of lengths {r["lengths"]}, closed {r["closed"]} (gaps filled)')
        return r, f'{r["tokens"]} tokens in {r["seconds"]} s; {whole}'


class Pipeline:
    """A model mlx-vlm draws with its own pipeline (quanteeze.vlm_image.MODELS)."""

    def __init__(self, name, path=None):
        from . import vlm_image
        spec = vlm_image.MODELS[name]
        self.name, self.license, self.given = name, spec.license, path
        self.about = f'{spec.about}; sides 256-2048 in steps of 16'

    def find(self):
        from . import vlm_image
        return (self.given or vlm_image.find(self.name, roots()),)

    def check_size(self, width, height):
        from . import vlm_image
        vlm_image.check_size(width, height)

    def room(self, paths):
        from . import vlm_image
        return vlm_image.room(*paths)

    def draw(self, paths, prompt, width, height, seed, out):
        from . import vlm_image
        r = vlm_image.draw(*paths, prompt, width, height, seed, out=out)
        return r, f'{r["steps"]} steps in {r["seconds"]} s'


ORDER = ('z-image-turbo', 'flux2-klein', 'qwen-image', 'nemotron-image')  # default: first found, Apache-2.0 first


class Server:
    """The tool's state: its models, where it writes and what it wrote."""

    def __init__(self, models=None, default=None, out=None, keep_gb=2.0, write_gb=2.0):
        self.models = {m.name: m for m in (models if models is not None else
                                           [Pipeline(n) for n in ORDER[:3]] + [Nemotron()])}
        self.default = default
        self.out = Path(out).expanduser() if out else Path.home() / '.quanteeze' / 'images'
        self.keep, self.write = keep_gb * 1e9, write_gb * 1e9
        self.written = 0
        self.found = {}                                     # name: its paths, found once

    def kept(self):
        return sum(p.stat().st_size for p in self.out.glob('*.png')) if self.out.is_dir() else 0

    def paths(self, name):
        if name not in self.found:
            self.found[name] = self.models[name].find()
        return self.found[name]

    def here(self):
        """The models found on this machine, the default first."""
        out = []
        for name in self.models:
            try:
                self.paths(name)
                out.append(name)
            except (ValueError, OSError):
                pass
        if self.default in out:
            out.remove(self.default)
            out.insert(0, self.default)
        return out

    def tool(self):
        here = self.here()
        about = (' '.join(f'{n}: {self.models[n].about} ({self.models[n].license}).' for n in here)
                 or 'No image model is found on this machine.')
        return {
            'name': 'generate_image',
            'description': ('Draw a picture on this machine and save it as a PNG. Describe the image in detail '
                            '(subjects, colours, composition, style, any text in it). Returns the path of the PNG. '
                            'Mind each model\'s license: some are for research or evaluation only. ' + about),
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'prompt': {'type': 'string', 'description': 'What to draw, in detail.'},
                    'model': {'type': 'string', 'enum': here or list(self.models),
                              'description': f'Which model draws it (default {(here or list(self.models))[0]}).'},
                    'width': {'type': 'integer', 'description': 'Pixels (default 1024).'},
                    'height': {'type': 'integer', 'description': 'Pixels (default 1024).'},
                    'seed': {'type': 'integer', 'description': 'For the same picture again; random where left out.'},
                    'path': {'type': 'string', 'description': 'An absolute path ending in .png to write it to (not a '
                                                              'file there already); left out, the server\'s folder.'},
                },
                'required': ['prompt'],
            },
        }

    def generate(self, args):
        prompt = str(args.get('prompt') or '').strip()
        if not prompt:
            raise ValueError('prompt: what to draw')
        name = args.get('model') or (self.here() or [None])[0]
        if name not in self.models:
            raise ValueError(f'model: one of {", ".join(self.models)}, not {name!r}')
        model = self.models[name]
        width, height = int(args.get('width') or 1024), int(args.get('height') or 1024)
        model.check_size(width, height)
        if args.get('path'):
            target = Path(str(args['path'])).expanduser()
            if not target.is_absolute() or target.suffix.lower() != '.png':
                raise ValueError(f'path: an absolute path ending in .png, not {args["path"]!r}')
            if target.exists():
                raise ValueError(f'{target} is there already: give another path')
            if not target.parent.is_dir():
                raise ValueError(f'{target.parent}: no such folder')
        else:
            if self.kept() >= self.keep:
                raise ValueError(f'{self.out} holds {self.kept() / 1e9:.1f} GB of images, at most {self.keep / 1e9:g} '
                                 'GB kept (--keep-gb): remove some, or give a path')
            words = '-'.join(re.findall(r'[a-z0-9]+', prompt.lower())[:6])[:48] or 'image'
            self.out.mkdir(parents=True, exist_ok=True)
            target = None
        if self.written >= self.write:
            raise ValueError(f'this server has written {self.written / 1e9:.1f} GB of images, at most '
                             f'{self.write / 1e9:g} GB (--write-gb): start it again to write more')
        paths = self.paths(name)
        why = model.room(paths)
        if why:
            raise ValueError(why)
        seed = args.get('seed')
        seed = int(seed) if seed is not None else int(time.time() * 1000) % (2 ** 31)
        stem = (str(target.with_suffix('')) if target is not None
                else str(self.out / f'{time.strftime("%Y%m%d-%H%M%S")}-{name}-{words}-{seed}'))
        r, said = model.draw(paths, prompt, width, height, seed, stem)
        self.written += Path(r['png']).stat().st_size
        return f'Saved {r["png"]} ({name}, {width}x{height}, seed {r["seed"]}; {model.license}); {said}.'

    def answer(self, message):
        """The reply to one JSON-RPC message, or None (a notification, or a reply sent to us)."""
        method, ident = message.get('method'), message.get('id')
        if method is None or ident is None:
            return None
        params = message.get('params') or {}
        if method == 'initialize':
            asked = params.get('protocolVersion')
            result = {
                'protocolVersion': asked if isinstance(asked, str) and asked else PROTOCOL,
                'capabilities': {'tools': {'listChanged': False}},
                'serverInfo': {'name': NAME, 'version': VERSION},
                'instructions': ('generate_image draws a picture on this machine; mind the license of the '
                                 'model chosen (some are for research or evaluation only).'),
            }
        elif method == 'ping':
            result = {}
        elif method == 'tools/list':
            result = {'tools': [self.tool()]}
        elif method == 'tools/call':
            if params.get('name') != 'generate_image':
                return {'jsonrpc': '2.0', 'id': ident,
                        'error': {'code': -32602, 'message': f'no tool {params.get("name")!r}'}}
            try:
                result = {'content': [{'type': 'text', 'text': self.generate(params.get('arguments') or {})}],
                          'isError': False}
            except Exception as e:                      # noqa: BLE001 -- told to the agent, which may try again
                result = {'content': [{'type': 'text', 'text': f'{type(e).__name__}: {e}'}], 'isError': True}
        else:
            return {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32601, 'message': f'no method {method!r}'}}
        return {'jsonrpc': '2.0', 'id': ident, 'result': result}

    def serve(self, lines, write):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                write({'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'not JSON'}})
                continue
            for one in (message if isinstance(message, list) else [message]):
                reply = self.answer(one) if isinstance(one, dict) else None
                if reply is not None:
                    write(reply)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--nemotron')
    ap.add_argument('--decoder')
    ap.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH')
    ap.add_argument('--default', choices=ORDER)
    ap.add_argument('--out')
    ap.add_argument('--keep-gb', type=float, default=2.0)
    ap.add_argument('--write-gb', type=float, default=2.0)
    a = ap.parse_args(argv)
    given = {}
    for c in a.checkpoint:
        if '=' not in c:
            ap.error(f'--checkpoint: expected NAME=PATH, not {c!r}')
        k, v = c.split('=', 1)
        given[k] = v
    unknown = set(given) - set(ORDER[:3])
    if unknown:
        ap.error(f'--checkpoint: a name of {", ".join(ORDER[:3])}, not {", ".join(sorted(unknown))}')
    models = [Pipeline(n, given.get(n)) for n in ORDER[:3]] + [Nemotron(a.nemotron, a.decoder)]
    server = Server(models, a.default, a.out, a.keep_gb, a.write_gb)
    out = sys.stdout
    sys.stdout = sys.stderr                             # what mlx-lm, mlx-vlm or anything prints, kept off the protocol

    def write(message):
        out.write(json.dumps(message) + '\n')
        out.flush()

    print(f'{NAME}: ready; images into {server.out} (at most {a.keep_gb:g} GB kept, {a.write_gb:g} GB written; '
          'Apache-2.0 for Z-Image-Turbo / FLUX.2-klein-4B, research or evaluation only for Qwen-Image-2.1 / Nemotron)',
          file=sys.stderr, flush=True)
    server.serve(sys.stdin, write)


if __name__ == '__main__':
    main()
