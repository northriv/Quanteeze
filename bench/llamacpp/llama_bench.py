# SPDX-License-Identifier: MIT
"""llama-bench compatible throughput measurements against a running server: pp<N>, tg<N>,
pp<N>+tg<M>, each at context depth d, repeated -r times after a warmup run, reporting mean and
sample standard deviation in tokens/s.

    python bench/llamacpp/llama_bench.py --url 127.0.0.1:8765 --checkpoint <dir> --log <the server's log>
        [-p 512] [-n 128] [-pg 512,128] [-d 0,4096] [-r 5] [--feed forced|text] [--engine label] [-o md|json|jsonl]

Adapted from llama.cpp (https://github.com/ggml-org/llama.cpp at 8d60873, MIT License — see
bench/llamacpp/LICENSE), tools/llama-bench/llama-bench.cpp: test names/order, defaults
(-p 512 -n 128 -r 5 plus warmup), and avg/stdev over per-repetition throughput.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                                     # noqa: E402


def name(n_prompt, n_gen, depth):
    """Return llama-bench test label."""
    if n_prompt and not n_gen:
        s = f'pp{n_prompt}'
    elif n_gen and not n_prompt:
        s = f'tg{n_gen}'
    else:
        s = f'pp{n_prompt}+tg{n_gen}'
    return s + (f' @ d{depth}' if depth else '')


def tests(prompts, gens, pgs, depths):
    """Return test schedule in llama-bench order: at each depth, pp, then tg, then pg."""
    out = []
    for d in depths:
        out += [(p, 0, d) for p in prompts if p]
        out += [(0, g, d) for g in gens if g]
        out += [(p, g, d) for p, g in pgs]
    return out


def timed(got, n_prompt, n_gen, depth, alone):
    """Compute `(tokens_counted, seconds, mark)` from parsed log metrics for one repetition."""
    mark = ''
    if not n_prompt:
        prefill = 0.0
    elif not depth or got['cached'] >= depth:
        prefill = got['prefill_s']
    else:
        prefill = common.after(got, depth)
        if prefill is None:
            prefill, mark = max(got['prefill_s'] - alone(), 1e-9), '~'
    steps = got['generated'] - 1
    if n_prompt and not n_gen:
        return n_prompt, max(prefill, 1e-9), mark
    if n_gen and not n_prompt:
        return steps, max(got['decode_s'], 1e-9), mark
    return n_prompt + steps, max(prefill + got['decode_s'], 1e-9), mark


class ForcedRunner:
    def __init__(self, args):
        if not args.log:
            raise SystemExit('--feed forced takes the times from the server\'s log: --log <file>')
        self.vocab = common.Vocab(args.checkpoint, args.bos, args.seed)
        self.send = common.Forced(args.url)
        self.log = common.Log(args.log)

    def one(self, prompt, output):
        self.log.mark()
        self.send(prompt, output)
        return common.only(self.log.since())

    def depth(self, d):
        """Return `d` random context token IDs for depth `d`."""
        return self.vocab.random(d).tolist() if d else []

    def run(self, n_prompt, n_gen, prefix):
        start = prefix if prefix else [] if n_prompt else self.vocab.random(1).tolist()
        prompt = start + (self.vocab.random(n_prompt, not prefix).tolist() if n_prompt else [])
        got = self.one(prompt, self.vocab.random(max(n_gen, 1), False))
        return timed(
            got,
            n_prompt,
            n_gen,
            len(prefix),
            lambda: self.one(prefix, self.vocab.random(1, False))['prefill_s'],
        )

    def close(self):
        self.send.close()


class TextRunner:
    STORY = 'Write a very long story about a lighthouse keeper and the ships that pass, with many chapters.'

    def __init__(self, args):
        import random
        self.url = args.url
        self.log = common.Log(args.log) if args.log else None
        self.rng = random.Random(args.seed)
        self.per_word = None

    def words_for(self, n):
        """Generate random words tuned to roughly `n` tokens based on an initial server probe."""
        if self.per_word is None:
            _, _, _, usage = common.stream_chat(
                self.url,
                [{'role': 'user', 'content': common.random_words(400, self.rng)}],
                1,
                {'enable_thinking': False},
            )
            self.per_word = max(0.1, (usage.get('prompt_tokens') or 400) / 400)
        return common.random_words(max(1, round(n / self.per_word)), self.rng)

    def ask(self, content, most):
        """Send a chat request and return server log timings or client clock fallback."""
        if self.log:
            self.log.mark()
        sent, first, end, usage = common.stream_chat(
            self.url,
            [{'role': 'user', 'content': content}],
            most,
            {'enable_thinking': False},
        )
        if self.log:
            return common.only(self.log.since())
        first_at = first if first is not None else end
        return {
            'prompt': usage.get('prompt_tokens') or 0,
            'cached': 0,
            'prefill_s': max(first_at - sent, 1e-9),
            'generated': usage.get('completion_tokens') or 0,
            'decode_s': max(end - first_at, 1e-9),
            'client': True,
        }

    def depth(self, d):
        return self.words_for(d) if d else ''

    def run(self, n_prompt, n_gen, prefix):
        content = ' '.join(x for x in (prefix, self.words_for(n_prompt) if n_prompt else '') if x)
        content += ('\n\n' + self.STORY) if n_gen else ''
        got = self.ask(content.strip(), max(n_gen, 1))
        tokens, seconds, mark = timed(
            dict(got, cached=0, progress={}),
            n_prompt,
            n_gen,
            1 if prefix else 0,
            lambda: self.ask(prefix, 1)['prefill_s'],
        )
        if n_prompt and not n_gen and not prefix:
            tokens = got['prompt'] - got['cached']
        return tokens, seconds, mark + ('c' if got.get('client') else '')

    def close(self):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--url', default='127.0.0.1:8765')
    ap.add_argument('--checkpoint', help='the served checkpoint (its tokenizer; --feed forced)')
    ap.add_argument('--log', help="the server's log (the times; --feed forced needs it)")
    ap.add_argument('-p', '--n-prompt', default='512')
    ap.add_argument('-n', '--n-gen', default='128')
    ap.add_argument('-pg', action='append', default=[], help='<pp>,<tg>')
    ap.add_argument('-d', '--n-depth', default='0')
    ap.add_argument('-r', '--repetitions', type=int, default=5)
    ap.add_argument('--no-warmup', action='store_true')
    ap.add_argument('--feed', choices=('forced', 'text'), default='forced')
    ap.add_argument('--bos', choices=('auto', 'on', 'off'), default='auto')
    ap.add_argument('--engine', default='quanteeze', help='label for the table')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('-o', '--output', choices=('md', 'json', 'jsonl'), default='md')
    args = ap.parse_args(argv)
    ints = lambda s: [int(x) for x in str(s).split(',') if x != '']           # noqa: E731
    pgs = [tuple(ints(x)) for x in args.pg]
    if args.feed == 'forced' and not args.checkpoint:
        raise SystemExit('--feed forced: --checkpoint <the served checkpoint> (its tokenizer)')
    runner = ForcedRunner(args) if args.feed == 'forced' else TextRunner(args)
    model = common.model_name(args.url)
    results, depth_at = [], {}
    if args.output == 'md':
        print('| model | engine | feed | test | t/s |\n| --- | --- | --- | ---: | ---: |', flush=True)
    try:
        for n_prompt, n_gen, d in tests(ints(args.n_prompt), ints(args.n_gen), pgs, ints(args.n_depth)):
            if d not in depth_at:
                depth_at.clear()
                depth_at[d] = runner.depth(d)
            prefix = depth_at[d]
            if not args.no_warmup:
                runner.run(n_prompt, min(n_gen, 1) if n_gen else 0, prefix)
            samples, marks = [], set()
            for _ in range(args.repetitions):
                tokens, seconds, how = runner.run(n_prompt, n_gen, prefix)
                samples.append(tokens / max(seconds, 1e-9))
                marks.update(how)
            row = {
                'model': model,
                'engine': args.engine,
                'feed': args.feed,
                'n_prompt': n_prompt,
                'n_gen': n_gen,
                'n_depth': d,
                'test': name(n_prompt, n_gen, d),
                'avg_ts': common.avg(samples),
                'stddev_ts': common.stdev(samples),
                'samples_ts': samples,
                'marks': ''.join(sorted(marks)),
                'test_time': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            results.append(row)
            if args.output == 'md':
                print(
                    f"| {model} | {args.engine} | {args.feed} | {row['test']}{row['marks']} | "
                    f"{row['avg_ts']:.2f} ± {row['stddev_ts']:.2f} |",
                    flush=True,
                )
            elif args.output == 'jsonl':
                print(json.dumps(row), flush=True)
    finally:
        runner.close()
    if args.output == 'json':
        print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
