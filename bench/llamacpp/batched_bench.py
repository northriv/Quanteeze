# SPDX-License-Identifier: MIT
"""llama-batched-bench against a running server: B concurrent requests, each a prompt of PP tokens
and TG generated, reporting llama-batched-bench columns.

    python bench/llamacpp/batched_bench.py --url 127.0.0.1:8765 --checkpoint <dir> --log <the server's log>
        [-npp 128,256,512] [-ntg 128,256] [-npl 1,2,4,8] [--feed forced|text] [--output-format md|jsonl]

Adapted from llama.cpp (https://github.com/ggml-org/llama.cpp at 8d60873, MIT License — see
bench/llamacpp/LICENSE), tools/batched-bench/batched-bench.cpp: sweep, columns, and arithmetic
(prompt not shared):

    N_KV = B*(PP + TG); S_PP = B*PP / T_PP; S_TG = B*TG / T_TG; T = T_PP + T_TG; S = B*(PP + TG) / T

Notes: overlapping requests require server concurrency (`QUANTEEZE_REQUEST_THREADS`, or mlx-vlm
continuous batching). Measure with speculative drafting off (`QUANTEEZE_MTP=off`).
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                                     # noqa: E402

HEAD = ('|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |\n'
        '|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|')


def columns(pp, tg, pl, t_pp, t):
    """Compute llama-batched-bench row metrics (prompt not shared)."""
    t_pp = max(t_pp, 1e-9)
    t = max(t, 1e-9)
    t_tg = max(t - t_pp, 1e-9)
    return {
        'pp': pp,
        'tg': tg,
        'pl': pl,
        'n_kv': pl * (pp + tg),
        't_pp': t_pp,
        'speed_pp': pl * pp / t_pp,
        't_tg': t_tg,
        'speed_tg': pl * tg / t_tg,
        't': t,
        'speed': pl * (pp + tg) / t,
    }


def union(intervals):
    """Return total length of the union of `(start, end)` intervals."""
    total, reach = 0.0, None
    for a, b in sorted(intervals):
        if reach is None or a > reach:
            total += b - a
            reach = b
        elif b > reach:
            total += b - reach
            reach = b
    return total


def from_log(entries):
    """Compute `(T_PP, T)` from server log entries (`T_PP` is the union of prefill intervals)."""
    start = min(e['queued'] for e in entries)
    t_pp = union([(e['prefill_end'] - e['prefill_s'], e['prefill_end']) for e in entries])
    return t_pp, max(e['decode_end'] for e in entries) - start


def from_client(timings):
    """Compute `(T_PP, T)` from client `(sent, first_token_or_None, end)` tuples."""
    start = min(s for s, _, _ in timings)
    return max((f if f is not None else e) for _, f, e in timings) - start, max(e for _, _, e in timings) - start


def at_once(jobs):
    """Run callables concurrently from barrier-synchronized threads; return results in order."""
    out, gate = [None] * len(jobs), threading.Barrier(len(jobs))

    def go(i, job):
        gate.wait()
        out[i] = job()

    threads = [threading.Thread(target=go, args=(i, j)) for i, j in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--url', default='127.0.0.1:8765')
    ap.add_argument('--checkpoint')
    ap.add_argument('--log')
    ap.add_argument('-npp', default='128,256,512')
    ap.add_argument('-ntg', default='128,256')
    ap.add_argument('-npl', default='1,2,4,8')
    ap.add_argument('--feed', choices=('forced', 'text'), default='forced')
    ap.add_argument('--bos', choices=('auto', 'on', 'off'), default='auto')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output-format', choices=('md', 'jsonl'), default='md')
    args = ap.parse_args(argv)
    ints = lambda s: [int(x) for x in str(s).split(',') if x != '']           # noqa: E731
    if args.feed == 'forced' and not (args.checkpoint and args.log):
        raise SystemExit('--feed forced: --checkpoint <the served checkpoint> and --log <the server\'s log>')
    log = common.Log(args.log) if args.log else None
    send = None
    if args.feed == 'forced':
        vocab = common.Vocab(args.checkpoint, args.bos, args.seed)
        send = common.Forced(args.url)
        job = lambda pp, tg: (lambda p=vocab.random(pp), o=vocab.random(tg, False): send(p, o))   # noqa: E731
    else:
        import random
        rng = random.Random(args.seed)
        story = '\n\nWrite a very long story about a lighthouse keeper, with many chapters.'
        job = lambda pp, tg: (lambda c=common.random_words(pp, rng) + story: common.stream_chat(   # noqa: E731
            args.url, [{'role': 'user', 'content': c}], tg, {'enable_thinking': False}))
    print(f'model {common.model_name(args.url)}, feed {args.feed}, prompt not shared', flush=True)
    if args.output_format == 'md':
        print(HEAD, flush=True)
    try:
        at_once([job(32, 4)])
        for pp in ints(args.npp):
            for tg in ints(args.ntg):
                for pl in ints(args.npl):
                    if log:
                        log.mark()
                    got = at_once([job(pp, tg) for _ in range(pl)])
                    if log:
                        entries = [e for e in common.parse(log.since()).values() if 'decode_end' in e]
                        if len(entries) != pl:
                            raise RuntimeError(f'{len(entries)} requests in the log where {pl} were sent')
                        t_pp, t = from_log(entries)
                    else:
                        t_pp, t = from_client([g[:3] for g in got])
                    row = columns(pp, tg, pl, t_pp, t)
                    if args.output_format == 'jsonl':
                        print(json.dumps({'feed': args.feed, 'is_pp_shared': 0, **row}), flush=True)
                    else:
                        print(
                            f"|{pp:6d} | {tg:6d} | {pl:4d} | {row['n_kv']:6d} | {row['t_pp']:8.3f} | "
                            f"{row['speed_pp']:8.2f} | {row['t_tg']:8.3f} | {row['speed_tg']:8.2f} | "
                            f"{row['t']:8.3f} | {row['speed']:8.2f} |",
                            flush=True,
                        )
                    time.sleep(0.1)
    finally:
        if send is not None:
            send.close()


if __name__ == '__main__':
    main()
