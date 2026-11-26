# SPDX-License-Identifier: MIT
"""Shared helpers for bench/llamacpp against a running Quanteeze server: server URL and control
token, checkpoint tokenizer, random token IDs, forced/plain requests, server log parsing, and
llama-bench mean/sample standard deviation.

Adapted from llama.cpp (https://github.com/ggml-org/llama.cpp at 8d60873, MIT License — see
bench/llamacpp/LICENSE): the statistics of tools/llama-bench/llama-bench.cpp (avg, stdev) and random
token draws matching its test_prompt and test_gen.
"""
import datetime
import json
import math
import os
import random
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

TOKEN_FILE = Path('~/.quanteeze/control-token').expanduser()
PREFILL = re.compile(r'^(\S+ \S+),(\d+) - INFO - Prefill completed: request=(\w+) prompt_tokens=(\d+) '
                     r'cached_tokens=(\d+) elapsed=([\d.]+)s')
DECODE = re.compile(r'^(\S+ \S+),(\d+) - INFO - Decode completed: request=(\w+) generated_tokens=(\d+) '
                    r'elapsed=([\d.]+)s')
QUEUED = re.compile(r'^(\S+ \S+),(\d+) - INFO - Generation queued: request=(\w+)')
PROGRESS = re.compile(r'^(\S+ \S+),(\d+) - INFO - Prefill progress: request=(\w+) tokens=(\d+)/')


def base_url(url):
    url = url.strip().rstrip('/')
    if '://' not in url:
        url = 'http://' + url
    return url if url.endswith('/v1') else url + '/v1'


def control_token():
    secret = os.environ.get('QUANTEEZE_CONTROL_TOKEN')
    if secret:
        return secret
    try:
        return TOKEN_FILE.read_text(encoding='utf-8').strip() or None
    except OSError:
        return None


def model_name(url):
    try:
        with urllib.request.urlopen(base_url(url) + '/models', timeout=30) as r:
            data = json.loads(r.read()).get('data') or []
        return Path(str(data[0].get('id'))).name if data else '?'
    except (OSError, ValueError):
        return '?'


# ---------------------------------------------------------------- llama-bench statistics

def avg(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    """llama-bench's stdev: sample standard deviation (0 for <= 1 sample)."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = avg(values)
    return math.sqrt(max(0.0, sum(v * v for v in values) / (n - 1) - mean * mean * n / (n - 1)))


# ---------------------------------------------------------------- tokens

class Vocab:
    """Checkpoint tokenizer excluding special and added tokens so random draws do not emit EOS."""

    def __init__(self, checkpoint, bos='auto', seed=0):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        self.n = len(self.tok)
        if bos not in ('auto', 'on', 'off'):
            raise ValueError(f'bos {bos!r}: auto, on or off')
        self.bos = None if bos == 'off' else self.tok.bos_token_id
        if bos == 'on' and self.bos is None:
            raise SystemExit(f'{checkpoint}: its tokenizer has no BOS')
        special = set(self.tok.all_special_ids) | set(getattr(self.tok, 'added_tokens_decoder', {}) or {})
        self.allowed = np.array(sorted(set(range(self.n)) - special), np.int64)
        self.rng = np.random.default_rng(seed)

    def random(self, n, first_bos=True):
        """Return `n` random token IDs, starting with BOS when enabled (llama-bench's test_prompt)."""
        ids = self.rng.choice(self.allowed, size=n)
        if first_bos and self.bos is not None and n:
            ids[0] = self.bos
        return ids.astype(np.int64)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------- requests

class Forced:
    """Send control-token-gated `quanteeze_forced` requests with exact prompt and output token IDs."""

    def __init__(self, url, token=None):
        self.url = base_url(url) + '/chat/completions'
        self.token = token or control_token()
        if not self.token:
            raise SystemExit(
                'forced requests need the control token (QUANTEEZE_CONTROL_TOKEN or ~/.quanteeze/control-token)'
            )

    def __call__(self, prompt, output, timeout=24 * 3600):
        """Send one forced request (thread-safe); return `(seconds, usage)`."""
        output_arr = np.asarray(output, np.int64)
        body = {
            'model': 'default',
            'messages': [{'role': 'user', 'content': '.'}],
            'max_tokens': int(output_arr.size),
            'temperature': 0.0,
            'stream': False,
            'quanteeze_forced': {
                'token': self.token,
                'prompt': np.asarray(prompt, np.int64).tolist(),
                'output': output_arr.tolist(),
            },
        }
        t = time.perf_counter()
        reply = post(self.url, body, timeout)
        return time.perf_counter() - t, reply.get('usage') or {}

    def close(self):
        pass


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_chat(url, messages, max_tokens, extra=None, timeout=3600):
    """Send a streamed chat request; return `(sent, first_content_time_or_None, end, usage)`."""
    body = {
        'model': 'default',
        'messages': messages,
        'max_tokens': int(max_tokens),
        'temperature': 0.0,
        'stream': True,
        'stream_options': {'include_usage': True},
        **(extra or {}),
    }
    req = urllib.request.Request(
        base_url(url) + '/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    sent, first, usage = time.perf_counter(), None, {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data:') or line == b'data: [DONE]':
                continue
            d = json.loads(line[5:])
            if d.get('usage'):
                usage = d['usage']
            for ch in d.get('choices') or []:
                delta = ch.get('delta') or {}
                if first is None and (delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning')):
                    first = time.perf_counter()
    return sent, first, time.perf_counter(), usage


def random_words(n, rng=None):
    """Build a prompt of `n` random words for `--feed text`."""
    rng = rng or random.Random(0)
    words = (
        'time', 'river', 'paper', 'stone', 'light', 'number', 'garden', 'signal', 'window', 'market', 'cloud',
        'engine', 'letter', 'bridge', 'forest', 'circle', 'winter', 'silver', 'pocket', 'mirror',
    )
    return ' '.join(rng.choice(words) for _ in range(n))


# ---------------------------------------------------------------- server logs

def _stamp(day, ms):
    return datetime.datetime.strptime(day, '%Y-%m-%d %H:%M:%S').timestamp() + int(ms) / 1000


class Log:
    """Tail new lines written to the server log since `mark()`."""

    def __init__(self, path):
        self.path = Path(path)
        self.at = 0

    def mark(self):
        self.at = self.path.stat().st_size

    def since(self, settle=0.3):
        time.sleep(settle)
        with self.path.open('rb') as f:
            f.seek(self.at)
            return f.read().decode('utf-8', 'replace').splitlines()


def parse(lines):
    """Parse server log lines into `{request_id: {queued, prefill_end, prompt, cached, prefill_s, decode_end, generated, decode_s, progress}}`."""
    got = {}
    for line in lines:
        m = QUEUED.match(line)
        if m:
            got.setdefault(m.group(3), {})['queued'] = _stamp(m.group(1), m.group(2))
            continue
        m = PROGRESS.match(line)
        if m:
            got.setdefault(m.group(3), {}).setdefault('progress', {})[int(m.group(4))] = _stamp(m.group(1), m.group(2))
            continue
        m = PREFILL.match(line)
        if m:
            got.setdefault(m.group(3), {}).update(
                prefill_end=_stamp(m.group(1), m.group(2)),
                prompt=int(m.group(4)),
                cached=int(m.group(5)),
                prefill_s=float(m.group(6)),
            )
            continue
        m = DECODE.match(line)
        if m:
            got.setdefault(m.group(3), {}).update(
                decode_end=_stamp(m.group(1), m.group(2)),
                generated=int(m.group(4)),
                decode_s=float(m.group(5)),
            )
    return got


def after(got, depth):
    """Return prefill seconds past `depth` tokens from `Prefill progress`, or None if no step ended at `depth`."""
    at = (got.get('progress') or {}).get(depth)
    return None if at is None or 'prefill_end' not in got else max(got['prefill_end'] - at, 1e-9)


def only(lines):
    """Return parsed metrics for the single prefilled request in `lines`."""
    got = [v for v in parse(lines).values() if 'prefill_s' in v]
    if len(got) != 1:
        raise RuntimeError(f'{len(got)} requests in the log where one was sent')
    return got[0]
