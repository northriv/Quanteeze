# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""quanteeze.image_mcp, quanteeze.vlm_image, and quanteeze.nemotron_image without the model: the grid read from
tokens, sizes offered, checkpoints found in model folders, MCP protocol handling (initialize, tools/list, tools/call,
notifications, errors) with default-first Apache-2.0 model selection, per-model license strings, bounded writes,
checked paths, and NVIDIA License isolation & attribution checks."""
import json
import sys
from pathlib import Path

import pytest

from quanteeze import control, home, image_mcp, nemotron_image, serve, vlm_image

RANGES = {'framing': {'<|img_start|>': 131072, '<|img_end|>': 131073, '<|vis_row|>': 131074},
          'vision': {'start_id': 131075, 'end_id': 131075 + 196560}}
START, END, ROW, V = 131072, 131073, 131074, 131075


def test_the_grid_is_read_row_by_row_and_its_gaps_filled():
    rows = [[V + 10 * r + c for c in range(4)] for r in range(4)]
    tokens = [5, START, 7, 8] + [t for r in rows for t in [ROW, *r]] + [END]
    codes, said = nemotron_image.grid(tokens, RANGES, 64, 64)
    assert codes.tolist() == [[10 * r + c for c in range(4)] for r in range(4)]
    assert said == {'rows': 4, 'lengths': [4], 'closed': True, 'stray': 0, 'whole': True, 'head': [7, 8]}
    short = [START] + [ROW, V + 1, V + 2] + [ROW, V + 3, V + 4, V + 5, V + 6]
    codes, said = nemotron_image.grid(short, RANGES, 64, 64)
    assert codes[0].tolist() == [1, 2, 0, 0] and codes[2:].sum() == 0
    assert said['rows'] == 2 and said['lengths'] == [2, 4] and not said['closed'] and not said['whole']
    with pytest.raises(ValueError, match='began no image'):
        nemotron_image.grid([1, 2, 3], RANGES, 64, 64)


def test_the_tokens_asked_for_and_the_sizes_offered():
    assert nemotron_image.max_tokens(1024, 1024) == 64 * 65 + nemotron_image.HEADER_TOKENS
    nemotron_image.check_size(1024, 512)
    for bad in ((1000, 1024), (256, 1024), (1024, 2048)):
        with pytest.raises(ValueError, match='a side is one of'):
            nemotron_image.check_size(*bad)


def _converted(root, name='Nemotron-Labs-Image-30B-A3B-MLX-4bit', quantized=True):
    d = root / 'nvidia' / name
    d.mkdir(parents=True)
    (d / 'config.json').write_text(json.dumps({'model_type': 'nemotron_h', **({'quantization': {'bits': 4}}
                                                                               if quantized else {})}))
    (d / nemotron_image.MANIFEST).write_text('{}')
    return d


def test_the_model_and_its_decoder_are_found_in_the_model_folders(tmp_path):
    lmstudio, disk = tmp_path / 'lmstudio', tmp_path / 'disk'
    with pytest.raises(ValueError, match='no converted'):
        nemotron_image.find([lmstudio, disk])
    model = _converted(lmstudio)
    original = _converted(disk, 'Nemotron-Labs-Image-30B-A3B', quantized=False)
    with pytest.raises(ValueError, match='Nemotron-Labs-Image-Tokenizer'):
        nemotron_image.find([lmstudio, disk])
    (original / nemotron_image.DECODER).mkdir()
    (original / nemotron_image.DECODER / 'model.safetensors').write_bytes(b'x')
    assert nemotron_image.find([lmstudio, disk]) == (model, original / nemotron_image.DECODER)
    (model / nemotron_image.DECODER).mkdir()                         # copied beside: that one first
    (model / nemotron_image.DECODER / 'model.safetensors').write_bytes(b'x')
    assert nemotron_image.find([disk, lmstudio])[1] == model / nemotron_image.DECODER


class _Fake:
    """A model stood in: found or not, a side of 512 to 1536, the memory room it is told, a PNG of 1,000 bytes."""

    def __init__(self, name, here=True, room=None):
        self.name, self.about, self.here, self.why, self.drawn = name, f'{name} draws', here, room, []
        self.license = f'{name} license'

    def find(self):
        if not self.here:
            raise ValueError(f'no {self.name} here')
        return (f'/models/{self.name}',)

    def check_size(self, width, height):
        if not 512 <= width <= 1536 or not 512 <= height <= 1536:
            raise ValueError('a side is one of 512..1536')

    def room(self, paths):
        return self.why

    def draw(self, paths, prompt, width, height, seed, out):
        self.drawn.append((paths, prompt, width, height, seed, out))
        Path(out + '.png').write_bytes(b'p' * 1000)
        return {'png': out + '.png', 'seed': seed}, '23.3 s'


def _server(tmp_path, models=None, **kw):
    models = models or (_Fake('nemotron-image'), _Fake('qwen-image'))
    return image_mcp.Server(models, out=tmp_path / 'images', **kw), models


def _talk(server, *messages):
    out = []
    server.serve([json.dumps(m) for m in messages], out.append)
    return out


def _call(ident, **arguments):
    return {'jsonrpc': '2.0', 'id': ident, 'method': 'tools/call',
            'params': {'name': 'generate_image', 'arguments': arguments}}


def test_the_server_speaks_mcp_and_draws_with_the_model_asked(tmp_path):
    server, (nemotron, qwen) = _server(tmp_path)
    got = _talk(server,
                {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                 'params': {'protocolVersion': '2025-03-26', 'capabilities': {}, 'clientInfo': {'name': 'codex'}}},
                {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
                {'jsonrpc': '2.0', 'id': 3, 'method': 'ping'},
                _call(4, prompt='A red fox in the snow', seed=42),
                _call(5, prompt='A poster that says HELLO', model='qwen-image', width=768),
                {'jsonrpc': '2.0', 'id': 6, 'method': 'resources/list'})
    assert [m['id'] for m in got] == [1, 2, 3, 4, 5, 6]                   # the notification answered with nothing
    assert got[0]['result']['protocolVersion'] == '2025-03-26' and 'tools' in got[0]['result']['capabilities']
    tool = got[1]['result']['tools'][0]
    assert tool['name'] == 'generate_image' and tool['inputSchema']['required'] == ['prompt']
    assert tool['inputSchema']['properties']['model']['enum'] == ['nemotron-image', 'qwen-image']
    assert 'nemotron-image draws (nemotron-image license)' in tool['description']
    assert 'some are for research or evaluation only' in tool['description']
    assert got[2]['result'] == {}
    text = got[3]['result']['content'][0]['text']
    assert not got[3]['result']['isError'] and 'nemotron-image, 1024x1024, seed 42; nemotron-image license' in text
    assert '23.3 s' in text
    paths, prompt, w, h, seed, out = nemotron.drawn[0]
    assert (paths, prompt, w, h, seed) == (('/models/nemotron-image',), 'A red fox in the snow', 1024, 1024, 42)
    assert out.startswith(str(tmp_path / 'images')) and out.endswith('-nemotron-image-a-red-fox-in-the-snow-42')
    assert not got[4]['result']['isError'] and qwen.drawn[0][2:4] == (768, 1024)
    assert got[5]['error']['code'] == -32601
    assert _talk(server, {'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call',
                          'params': {'name': 'other'}})[0]['error']['code'] == -32602
    out = []
    server.serve(['not json'], out.append)
    assert out[0]['error']['code'] == -32700


def test_the_models_offered_are_those_found_the_default_first(tmp_path):
    server, (nemotron, qwen) = _server(tmp_path, models=(_Fake('nemotron-image', here=False), _Fake('qwen-image')))
    assert server.tool()['inputSchema']['properties']['model']['enum'] == ['qwen-image']
    assert not _talk(server, _call(1, prompt='x'))[0]['result']['isError'] and qwen.drawn
    result = _talk(server, _call(2, prompt='x', model='nemotron-image'))[0]['result']
    assert result['isError'] and 'no nemotron-image here' in result['content'][0]['text']
    assert 'model: one of' in _talk(server, _call(3, prompt='x', model='dall-e'))[0]['result']['content'][0]['text']
    server, _ = _server(tmp_path, default='qwen-image')
    assert server.tool()['inputSchema']['properties']['model']['enum'] == ['qwen-image', 'nemotron-image']
    busy, _ = _server(tmp_path, models=(_Fake('qwen-image', room='12 GB of memory free'),))
    result = _talk(busy, _call(4, prompt='x'))[0]['result']
    assert result['isError'] and '12 GB of memory free' in result['content'][0]['text']


def test_its_paths_are_checked_and_its_writes_bounded(tmp_path):
    server, (nemotron, _) = _server(tmp_path, keep_gb=2.5e-6, write_gb=3.5e-6)
    there = tmp_path / 'there.png'
    there.write_bytes(b'x')

    def error(**arguments):
        result = _talk(server, _call(1, **arguments))[0]['result']
        assert result['isError']
        return result['content'][0]['text']

    assert 'prompt' in error(prompt=' ')
    assert 'a side is one of' in error(prompt='x', width=2000)
    assert 'absolute path ending in .png' in error(prompt='x', path='relative.png')
    assert 'absolute path ending in .png' in error(prompt='x', path=str(tmp_path / 'a.jpg'))
    assert 'there already' in error(prompt='x', path=str(there))
    assert 'no such folder' in error(prompt='x', path=str(tmp_path / 'nowhere' / 'a.png'))
    assert not nemotron.drawn
    for n in range(3):                                                   # 1,000 bytes each: 3 kept then refused
        assert not _talk(server, _call(n, prompt=f'x {n}'))[0]['result']['isError']
    assert 'at most 2.5e-06 GB kept' in error(prompt='y')
    mine = tmp_path / 'mine.png'
    assert not _talk(server, _call(9, prompt='z', path=str(mine)))[0]['result']['isError']   # a path: kept not counted
    assert nemotron.drawn[-1][5] == str(tmp_path / 'mine') and mine.is_file()
    assert 'written 0.0 GB' in error(prompt='z', path=str(tmp_path / 'more.png'))             # 4,000 bytes written
    assert len(nemotron.drawn) == 4 and len(list((tmp_path / 'images').glob('*.png'))) == 3


def test_mlx_vlms_checkpoints_are_found_by_name_and_their_sizes_checked(tmp_path):
    with pytest.raises(ValueError, match='no qwen-image'):
        vlm_image.find('qwen-image', [tmp_path])

    def tree(org, name):
        d = tmp_path / org / name
        for part in vlm_image.PARTS:
            (d / part).mkdir(parents=True)
        (d / 'model_index.json').write_text('{}')
        (d / 'transformer' / 'model.safetensors').write_bytes(b'x' * 1000)
        return d

    qwen = tree('mlx-community', 'Qwen-Image-2.1-MLX-4bit')
    flat = tmp_path / 'mlx-community' / 'FLUX.2-Klein-4B-8bit'            # mflux's layout: not taken
    (flat / 'transformer').mkdir(parents=True)
    nine = tree('black-forest-labs', 'FLUX.2-klein-9B')                   # not Apache-2.0: not taken
    assert vlm_image.find('qwen-image', [tmp_path]) == qwen and vlm_image.gb(qwen) == 1e-6
    with pytest.raises(ValueError, match='no flux2-klein'):
        vlm_image.find('flux2-klein', [tmp_path])
    four = tree('black-forest-labs', 'FLUX.2-klein-4B')
    zimage = tree('mlx-community', 'Z-Image-Turbo-bf16')
    assert vlm_image.find('flux2-klein', [tmp_path]) == four and vlm_image.find('z-image-turbo', [tmp_path]) == zimage
    assert nine.is_dir() and flat.is_dir()
    assert [m.name for m in vlm_image.MODELS.values() if m.commercial] == ['z-image-turbo', 'flux2-klein']
    vlm_image.check_size(1328, 1328)
    with pytest.raises(ValueError, match='a side is one of'):
        vlm_image.check_size(1000, 1030)
    server = image_mcp.Server([image_mcp.Pipeline(n, f'/m/{n}') for n in image_mcp.ORDER[:3]], out=tmp_path / 'i')
    assert server.here() == ['z-image-turbo', 'flux2-klein', 'qwen-image']   # the Apache-2.0 ones first
    assert 'Apache-2.0' in server.tool()['description'] and 'Qwen Research License' in server.tool()['description']


def test_it_draws_only_where_the_memory_has_room(tmp_path, monkeypatch):
    model = tmp_path / 'm'
    model.mkdir()
    (model / 'model.safetensors').write_bytes(b'x' * 1000)
    monkeypatch.setattr(nemotron_image, 'available_gb', lambda: 3.0)
    assert 'unload what holds it' in nemotron_image.room(model, model)
    monkeypatch.setattr(nemotron_image, 'available_gb', lambda: 50.0)
    assert nemotron_image.room(model, model) is None
    monkeypatch.setattr(nemotron_image, 'available_gb', lambda: None)
    assert nemotron_image.room(model, model) is None


def test_nemotron_vendor_license_and_lazy_isolation():
    assert 'quanteeze.vendor.nemotron_image.l24sq' not in sys.modules
    assert all(m is not None for m in (serve, home, control, image_mcp, nemotron_image, vlm_image))

    vendor_dir = Path(nemotron_image.__file__).resolve().parent / 'vendor' / 'nemotron_image'
    if vendor_dir.is_dir():
        lic = (vendor_dir / 'LICENSE').read_text(encoding='utf-8')
        notice = (vendor_dir / 'NOTICE.md').read_text(encoding='utf-8')
        l24sq_src = (vendor_dir / 'l24sq.py').read_text(encoding='utf-8')
        assert 'NVIDIA License' in lic and 'research or evaluation purposes only' in lic
        assert 'GPL-2.0-only' in notice and 'NVIDIA License' in notice
        assert 'SPDX-License-Identifier: LicenseRef-NvidiaProprietary' in l24sq_src
    else:
        with pytest.raises(RuntimeError, match=r'quanteeze\.vendor\.nemotron_image\.l24sq is not installed'):
            nemotron_image.draw('.', '.', 'x')
    assert image_mcp.ORDER == ('z-image-turbo', 'flux2-klein', 'qwen-image', 'nemotron-image')
    assert 'NVIDIA License' in image_mcp.Nemotron.license
    assert 'research or evaluation only' in image_mcp.Nemotron.license
