# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""The Browser UI's Japanese: home.html's own swap() run in Node on the English the page shows."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from quanteeze import home

HTML = Path(home.__file__).resolve().parent / 'home.html'
NODE = shutil.which('node')
pytestmark = pytest.mark.skipif(NODE is None, reason='needs node to run the page script')

# labels the page keeps as they are in Japanese too
KEPT = {'SSD', 'bfloat16', 'DFlash2'}


def swap(strings):
    """swap() of each string, as the page's translator does it (None: left as it is)."""
    page = HTML.read_text(encoding='utf-8')
    tables = page[page.index('  const JA = {'):page.index('  const SKIP')]
    body = page[page.index('  function swap(s) {'):page.index('  function skipped(')]
    script = (tables + body + 'const given = JSON.parse(require("fs").readFileSync(0, "utf8"));'
              'process.stdout.write(JSON.stringify(given.map(swap)));')
    out = subprocess.run([NODE, '-e', script], input=json.dumps(strings), capture_output=True,
                         text=True, check=True, timeout=60)
    return json.loads(out.stdout)


def japanese(text):
    return any('぀' <= c <= 'ヿ' or '一' <= c <= '鿿' for c in text)


def test_every_preset_is_in_japanese(monkeypatch):
    import mlx.core as mx
    monkeypatch.setattr(sys, 'platform', 'linux')                 # the heat note
    monkeypatch.setattr(mx.metal, 'is_available', lambda: False)  # the wired note
    presets = home.presets()
    shown = []
    for group, out in presets.items():
        if not isinstance(out, dict) or 'choices' not in out:
            continue
        for c in out['choices']:
            shown += [c['label'], c['about']] + ([c['plain']] if 'plain' in c else [])
        if 'note' in out:
            shown.append(out['note'])
    assert len(shown) > 40
    untranslated = [s for s, j in zip(shown, swap(shown)) if s not in KEPT and not (j and japanese(j))]
    assert untranslated == []


def test_lines_the_page_builds():
    cases = {
        # a precision option: once caught by the hit rate's rule as "メモリ内ヒット 67.9 GB of experts · fits"
        '67.9 GB of experts · fits in memory': 'エキスパート 67.9 GB · メモリに収まる',
        '135.9 GB of experts · streams from SSD': 'エキスパート 135.9 GB · SSD からオンデマンド読込',
        'qwen3_5_moe · 8-bit experts · all weights in memory': 'qwen3_5_moe · 8ビット・エキスパート · 全重みをメモリ保持',
        '93.8% in memory': 'メモリ内ヒット 93.8%',
        '4-bit stock': '元 4 ビット',
        'slows from serious · rested 0 s': '「高い」以上で減速 · 休止計 0 秒',
        'org/Dense-27B is a dense model (16.1 GB): all weights load at their checkpoint precision.':
            'org/Dense-27B は密なモデル（16.1 GB）です。すべての重みをチェックポイントの元の精度で読み込みます。',
        # a load's progress, in Serving now and beside Serve this model
        '43.3% · reading the weights (29.7 GB of 37.7 GB) · 2 s': '43.3% · 重みを読み込んでいます（37.7 GB 中 29.7 GB） · 2 秒',
        'Loading model… 60% · quantizing the experts (24 of 40 MoE layers) · 9 s':
            'モデルを読み込み中… 60% · エキスパートを量子化しています（MoE 層 40 中 24） · 9 秒',
        'Loading model… 100% · starting · 12 s': 'モデルを読み込み中… 100% · 開始しています · 12 秒',
        # the sentences around <span class="here">this Mac</span>, each a text node of its own
        'Choose a model and its expert precision on ': ' ',
        '.': ' で使うモデルと、エキスパートの量子化精度を選びます。',
        'Pick a checkpoint found on ': ' ',
        " (in LM Studio or Hugging Face's cache), or fetch one from Hugging Face.":
            ' で見つかったモデル（LM Studio か Hugging Face のキャッシュにあるもの）を選ぶか、Hugging Face から取得します。',
    }
    assert dict(zip(cases, swap(list(cases)))) == cases


def test_requests_at_once_in_japanese():
    """The Requests at once card: its own text, the lines showRequestThreads builds, and why_alone's reasons."""
    import ast
    from quanteeze import request_threads
    tree = ast.parse(Path(request_threads.__file__).read_text(encoding='utf-8'))
    why = next(f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == 'why_alone')
    reasons = [r.value.value for r in ast.walk(why) if isinstance(r, ast.Return) and isinstance(r.value, ast.Constant)
               and isinstance(r.value.value, str)]
    reasons += [r.value.left.value + ' (a GGUF tier)' for r in ast.walk(why)
                if isinstance(r, ast.Return) and isinstance(r.value, ast.BinOp)]
    assert len(reasons) == 5
    shown = ['Requests at once', 'Requests served at once, at most', 'Save', 'Reload the model now', 'A whole number, 1 or more.',
             'One loop for every request (mlx-vlm’s).', 'Loading again…',
             'Saved. It takes effect when the model is next loaded.', 'Saved. It takes effect when a model is loaded.',
             'The model was loaded with 1; 3 takes effect when it is next loaded.', 'Up to 3 at once.',
             'Now 2 running; at most 3 at once so far.',
             'Up to 3 at once, but one at a time now: a drafter other than DFlash (MTP head) drafts with its request alone.']
    shown += [f'Up to 2 at once, but one at a time now: {r}.' for r in reasons]
    page = HTML.read_text(encoding='utf-8')
    note = page[page.index('<p class="note">Above 1, each request'):].split('</p>')[0].removeprefix('<p class="note">')
    shown.append(note)
    out = swap(shown)
    untranslated = [s for s, j in zip(shown, out) if not (j and japanese(j))]
    assert untranslated == []
    leftover = [j for j in out[-len(reasons) - 1:-1] if 'the ' in j]   # each reason in Japanese, not left in English
    assert leftover == []
