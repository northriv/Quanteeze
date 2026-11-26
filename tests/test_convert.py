# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""quanteeze.convert: a checkpoint of float weights converted to MLX from the home page.

Checks float weight detection, tool/destination selection, refusal guards (model loaded, destination exists,
low disk space, unknown model_type, already quantized, active conversion), isolated staging without running
repository code, subprocess runner output caps and cleanup, end-to-end mlx-lm 4-bit conversion of a tiny
float32 Llama checkpoint, and home/control route integration.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from quanteeze import control, convert, home

LLAMA = {
    'model_type': 'llama',
    'architectures': ['LlamaForCausalLM'],
    'hidden_size': 64,
    'intermediate_size': 128,
    'num_hidden_layers': 2,
    'num_attention_heads': 4,
    'num_key_value_heads': 2,
    'rms_norm_eps': 1e-5,
    'vocab_size': 128,
    'rope_theta': 10000.0,
    'max_position_embeddings': 256,
    'tie_word_embeddings': False,
    'torch_dtype': 'float32',
}


def _weights(config, dtype=np.float32):
    rng = np.random.default_rng(0)
    h, i, v = config['hidden_size'], config['intermediate_size'], config['vocab_size']
    kv = h // config['num_attention_heads'] * config['num_key_value_heads']
    out = {
        'model.embed_tokens.weight': rng.standard_normal((v, h)),
        'model.norm.weight': np.ones(h),
        'lm_head.weight': rng.standard_normal((v, h)),
    }
    for n in range(config['num_hidden_layers']):
        p = f'model.layers.{n}.'
        out.update({
            p + 'self_attn.q_proj.weight': rng.standard_normal((h, h)),
            p + 'self_attn.k_proj.weight': rng.standard_normal((kv, h)),
            p + 'self_attn.v_proj.weight': rng.standard_normal((kv, h)),
            p + 'self_attn.o_proj.weight': rng.standard_normal((h, h)),
            p + 'mlp.gate_proj.weight': rng.standard_normal((i, h)),
            p + 'mlp.up_proj.weight': rng.standard_normal((i, h)),
            p + 'mlp.down_proj.weight': rng.standard_normal((h, i)),
            p + 'input_layernorm.weight': np.ones(h),
            p + 'post_attention_layernorm.weight': np.ones(h),
        })
    return {k: (0.02 * x).astype(dtype) for k, x in out.items()}


def _checkpoint(folder, config=None, extra=None, weights=None):
    """A Hugging Face checkpoint of float weights at `folder`: config, weights, word-level tokenizer."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    folder.mkdir(parents=True)
    config = dict(LLAMA, **(extra or {})) if config is None else config
    (folder / 'config.json').write_text(json.dumps(config))
    save_file(weights if weights is not None else _weights(LLAMA), str(folder / 'model.safetensors'))
    vocab = {f'w{n}': n for n in range(LLAMA['vocab_size'])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token='w0'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(folder / 'tokenizer.json'))
    (folder / 'tokenizer_config.json').write_text(json.dumps({
        'tokenizer_class': 'PreTrainedTokenizerFast',
        'unk_token': 'w0',
        'eos_token': 'w1',
    }))
    return folder


@pytest.fixture
def where(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, 'STATE_DIR', tmp_path / 'state')
    monkeypatch.setattr(convert, '_RUNNER', [None])
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path / 'home'))
    return tmp_path


def test_float_weights_are_told_from_quantized_ones(where):
    path = _checkpoint(where / 'models' / 'org' / 'tiny')
    found = convert.unquantized(path)
    assert found['dtype'] == 'f32' and found['params'] == sum(x.size for x in _weights(LLAMA).values())
    quantized = _checkpoint(where / 'models' / 'org' / 'q', extra={'quantization': {'bits': 4, 'group_size': 64}})
    assert convert.unquantized(quantized) is None
    fp8 = _checkpoint(where / 'models' / 'org' / 'fp8', extra={'quantization_config': {'quant_method': 'fp8'}})
    assert convert.unquantized(fp8) is None
    packed = _checkpoint(where / 'models' / 'org' / 'packed',
                         weights={'w': np.zeros((64, 64), np.uint32), 'b': np.zeros(8, np.float32)})
    assert convert.unquantized(packed) is None


def test_the_tool_the_destination_and_the_refusals(where, monkeypatch):
    root = where / 'models'
    path = _checkpoint(root / 'org' / 'tiny')
    assert convert.tool(path) == ('mlx_lm', None)
    vision = _checkpoint(root / 'org' / 'vl', config=dict(LLAMA, model_type='qwen3_5',
                                                          vision_config={'model_type': 'qwen3_5'}))
    assert convert.tool(vision) == ('mlx_vlm', None)
    unknown = _checkpoint(root / 'org' / 'odd', config=dict(LLAMA, model_type='nothing_known_xyz'))
    assert convert.tool(unknown)[0] is None and 'nothing_known_xyz' in convert.tool(unknown)[1]

    assert convert.destination(path, 4, [root]) == root.resolve() / 'org' / 'tiny-MLX-4bit'
    snapshot = _checkpoint(where / 'hub' / 'models--org--big' / 'snapshots' / 'abc')
    fake_home = where / 'home'
    assert (convert.destination(snapshot, 8, [root], 'org/big')
            == fake_home / '.quanteeze' / 'models' / 'org' / 'big-MLX-8bit')
    (fake_home / '.lmstudio' / 'models').mkdir(parents=True)
    assert (convert.destination(snapshot, 8, [root], 'org/big')
            == fake_home / '.lmstudio' / 'models' / 'org' / 'big-MLX-8bit')

    p = convert.plan(path, 4, [root])
    assert p['why'] is None and p['tool'] == 'mlx_lm' and p['gb'] == round(p['params'] * 4.5 / 8 / 1e9, 1)
    assert 'unload it first' in convert.plan(path, 4, [root], loaded='/m/other')['why']
    assert 'nothing_known_xyz' in convert.plan(unknown, 4, [root])['why']
    with pytest.raises(ValueError, match='bits'):
        convert.plan(path, 3, [root])
    (root / 'org' / 'tiny-MLX-4bit').mkdir()
    assert 'there already' in convert.plan(path, 4, [root])['why']
    monkeypatch.setattr(convert, '_free_gb', lambda folder: 15.0)
    assert 'would leave less than' in convert.plan(path, 8, [root])['why']
    assert 'quantized already' in convert.plan(_checkpoint(root / 'org' / 'q', extra={'quantization': {}}), 4,
                                              [root])['why']
    convert.STATE_DIR.mkdir(parents=True)
    (convert.STATE_DIR / 'state.json').write_text(json.dumps({'state': 'converting', 'pid': os.getpid(),
                                                              'dest': str(root)}))
    assert convert.running() and convert.plan(path, 8, [root])['why'] == 'a conversion is running'
    (convert.STATE_DIR / 'state.json').write_text(json.dumps({'state': 'converting', 'pid': 2 ** 22 + 12345}))
    assert not convert.running() and convert.state()['state'] == 'failed'


def test_the_staging_folder_runs_none_of_the_repositorys_code(where):
    src = _checkpoint(where / 'm', extra={'auto_map': {'AutoModel': 'modeling_x.X'}, 'model_file': 'modeling_x.py',
                                          'time_step_limit': [0.0, {'__float__': 'Infinity'}],
                                          'text_config': {'auto_map': {'a': 'b'}, 'x': 1}})
    (src / 'modeling_x.py').write_text('raise SystemExit("run")')
    (src / 'sub').mkdir()
    tok = json.loads((src / 'tokenizer_config.json').read_text())
    (src / 'tokenizer_config.json').write_text(json.dumps(dict(tok, auto_map={'AutoTokenizer': ['t.T', None]})))
    staged = convert.stage(src, where / 'staged')
    config = json.loads((staged / 'config.json').read_text())
    assert 'auto_map' not in config and 'model_file' not in config and 'auto_map' not in config['text_config']
    assert config['time_step_limit'] == [0.0, float('inf')] and config['text_config']['x'] == 1
    assert 'auto_map' not in json.loads((staged / 'tokenizer_config.json').read_text())
    assert not (staged / 'modeling_x.py').exists() and not (staged / 'sub').exists()
    assert (staged / 'model.safetensors').is_symlink()


def _python(code):
    return [sys.executable, '-c', code]


def test_the_runner_finishes_cleans_up_and_stops_what_writes_past_its_bound(where, monkeypatch):
    src = _checkpoint(where / 'm')
    (src / 'chat_template.jinja').write_text('{{ x }}')
    (src / 'm.py').write_text('')
    state = where / 'state'
    state.mkdir()
    dest = where / 'out' / 'm-MLX-4bit'
    write = (f'import pathlib; d = pathlib.Path({str(dest)!r}); d.mkdir(parents=True); '
             '(d / "w").write_bytes(b"x" * 1000)')
    assert convert.run(src, dest, 4, 'mlx_lm', 1000, state, command=_python(write), every=0.05) == 0
    s = json.loads((state / 'state.json').read_text())
    assert s['state'] == 'done' and s['bytes'] > 1000
    made = json.loads((dest / convert.MARKER).read_text())
    assert made['source'] == str(src) and made['bits'] == 4 and 'chat_template.jinja' in made['copied']
    assert (dest / 'chat_template.jinja').is_file() and not (dest / 'm.py').exists()
    assert not (dest.parent / f'.{dest.name}.staging').exists()

    other = where / 'out' / 'fails'
    assert convert.run(src, other, 4, 'mlx_lm', 1000, state, command=_python(
        f'import pathlib, sys; pathlib.Path({str(other)!r}).mkdir(); sys.exit(3)'), every=0.05) == 1
    s = json.loads((state / 'state.json').read_text())
    assert s['state'] == 'failed' and 'exited with 3' in s['error'] and not other.exists()

    monkeypatch.setattr(convert, 'SLACK', 0)
    big = where / 'out' / 'big'
    t = time.time()
    assert convert.run(src, big, 4, 'mlx_lm', 1000, state, command=_python(
        f'import pathlib, time; d = pathlib.Path({str(big)!r}); d.mkdir(); (d / "w").write_bytes(b"x" * 5000); '
        'time.sleep(60)'), every=0.05) == 1
    s = json.loads((state / 'state.json').read_text())
    assert 'past 1.5 times' in s['error'] and not big.exists() and time.time() - t < 30
    assert convert.run(src, dest, 4, 'mlx_lm', 1000, state, command=_python('pass')) == 1      # there already
    assert dest.is_dir() and 'there already' in json.loads((state / 'state.json').read_text())['error']


def test_a_tiny_float_llama_goes_through_mlx_lm_to_4_bits(where):
    root = where / 'models'
    src = _checkpoint(root / 'org' / 'tiny')
    s = convert.start(src, 4, [root])
    assert s['state'] in ('converting', 'done')
    for _ in range(600):
        s = convert.state()
        if s['state'] != 'converting':
            break
        time.sleep(0.2)
    assert s['state'] == 'done', (s, (convert.STATE_DIR / 'log.txt').read_text()[-2000:])
    dest = Path(s['dest'])
    assert dest == root.resolve() / 'org' / 'tiny-MLX-4bit'
    config = json.loads((dest / 'config.json').read_text())
    assert config['quantization']['bits'] == 4 and config['quantization']['group_size'] == 64
    assert convert.unquantized(dest) is None
    from mlx_lm import load
    import mlx.core as mx
    model, _ = load(str(dest))
    logits = model(mx.array([[1, 2, 3]]))
    assert logits.shape == (1, 3, LLAMA['vocab_size']) and mx.isfinite(logits).all().item()
    assert not any(p.name.endswith('.staging') for p in dest.parent.iterdir())


def test_float_weights_are_marked_and_their_conversion_told_and_refused_from_the_page(tmp_path, monkeypatch):
    """home + control routes mark float weights, plan conversions, refuse while a model is loaded, and block loads
    while a conversion runs."""
    assert {('POST', 'convert/info'), ('POST', 'convert'), ('GET', 'convert')} <= control.HOME_ROUTES
    monkeypatch.setattr(convert, 'STATE_DIR', tmp_path / 'convert')
    monkeypatch.setattr(convert, '_RUNNER', [None])
    monkeypatch.setattr(home, '_MODELS', {})
    monkeypatch.setattr(home, 'model_roots', lambda: [tmp_path / 'models'])
    monkeypatch.setattr(control, 'current', lambda: None)
    monkeypatch.setattr(control, 'chosen', lambda: None)
    monkeypatch.setitem(control._SWITCH, 'loader', lambda path: None)
    monkeypatch.setitem(control._SWITCH, 'busy', False)

    there = _checkpoint(tmp_path / 'models' / 'org' / 'tiny')
    info = home.model_info(there)
    assert info['unquantized']['dtype'] == 'f32' and info['converted_from'] is None
    told = home.handle('POST', 'convert/info', {'path': str(there), 'bits': 4})
    assert told['why'] is None and told['tool'] == 'mlx_lm'
    assert told['dest'] == str(there.resolve().parent / 'tiny-MLX-4bit')
    assert home.handle('GET', 'convert', {}) == {'state': 'idle'}

    monkeypatch.setattr(control, 'current', lambda: '/m/served')
    assert 'unload it first' in home.handle('POST', 'convert/info', {'path': str(there), 'bits': 8})['why']
    with pytest.raises(ValueError, match='unload it first'):
        home.handle('POST', 'convert', {'path': str(there), 'bits': 8})

    made = tmp_path / 'models' / 'org' / 'tiny-MLX-4bit'
    shutil.copytree(there, made)
    (made / convert.MARKER).write_text(json.dumps({'source': str(there)}))
    assert home.model_info(made)['converted_from'] == str(there)

    convert.STATE_DIR.mkdir()
    (convert.STATE_DIR / 'state.json').write_text(json.dumps({
        'state': 'converting',
        'pid': os.getpid(),
        'source': str(there),
        'dest': str(made),
    }))
    with pytest.raises(ValueError, match='converting tiny to MLX'):
        control.switch_model(made)
