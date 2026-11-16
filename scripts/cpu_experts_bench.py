# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Benchmark single-expert CPU offload throughput (`quanteeze.cpu_experts`) against PCIe copy speed.

    python scripts/cpu_experts_bench.py <checkpoint> [layer] [experts] [--threads=1,2,4,8] [--isa=avx512,avx2,plain]
        [--spin=0,200] [--priority=idle,wake,nice,normal] [--activation=plain|deepseek_v41] [--limit=0]
    python scripts/cpu_experts_bench.py --shape=<D>x<I>:<affine bits group | mxfp4> [...]

From a checkpoint: reads `experts` (default 8) of MoE layer `layer` (default: first MoE layer) into host
RAM in native checkpoint format (MLX affine words or DeepSeek V4.1 MXFP4); or constructs synthetic experts
from `--shape`. Repeats each expert `REPEAT` times per configuration after running `native_reference`
verification, reporting median ms/expert and effective host bandwidth (GB/s). Nothing is modified on disk.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quanteeze import cpu_experts as ce                                       # noqa: E402

REPEAT = 20


def _option(name, default):
    got = [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith(f'--{name}=')]
    return got[-1] if got else default


def synthetic(spec, seed=0):
    """Build a synthetic expert for `<D>x<I>:affine:<bits>:<group>` or `<D>x<I>:mxfp4`."""
    shape, kind, *rest = spec.split(':')
    D, I = (int(v) for v in shape.lower().split('x'))
    rng = np.random.default_rng(seed)
    bits, group = (int(rest[0]), int(rest[1])) if kind == 'affine' else (4, 32)

    def proj(rows, cols):
        words = rng.integers(0, 2 ** 32, (rows, cols * bits // 32), dtype=np.uint64).astype(np.uint32)
        if kind == 'mxfp4':
            return ce.projection(
                words, rng.integers(120, 130, (rows, cols // 32)).astype(np.uint8),
                None, 4, 32, kind='mxfp4',
            )
        sc = (np.full((rows, cols // group), 0.01, np.float32).view(np.uint32) >> 16).astype(np.uint16)
        return ce.projection(words, sc, sc, bits, group)

    return [{'gate_proj': proj(I, D), 'up_proj': proj(I, D), 'down_proj': proj(D, I)}], D


def from_checkpoint(path, layer=None, count=8):
    """Load `count` experts from an MoE layer in checkpoint `path` into native host format."""
    from quanteeze import q2_experts
    ck = q2_experts.Checkpoint(path)
    moe = [i for i in range(1000) if ck.has(i, 'gate_proj', 'weight')]
    if not moe:
        raise ValueError(f'{path}: no routed MoE experts found')
    layer = moe[0] if layer is None else layer
    d_dim = ck.tensors[ck.name(layer, 'down_proj', 'weight')][2][1]
    i_dim = ck.tensors[ck.name(layer, 'gate_proj', 'weight')][2][1]
    in_dims = {'gate_proj': d_dim, 'up_proj': d_dim, 'down_proj': i_dim}
    out = []
    for e in range(count):
        expert = {}
        for p in q2_experts.PROJECTIONS:
            mode = ck.mode(layer, p) or 'affine'
            group, bits = ck.quantization(layer, p, in_dims[p])
            weight = np.asarray(ck._read(ck.name(layer, p, 'weight'), [e]))[0]
            scales = np.asarray(ck._read(ck.name(layer, p, 'scales'), [e]))[0]
            biases = np.asarray(ck._read(ck.name(layer, p, 'biases'), [e]))[0] if ck.has(layer, p, 'biases') else None
            if weight.dtype != np.uint32:
                weight = np.ascontiguousarray(weight).view(np.uint32)
            expert[p] = ce.projection(weight, scales, biases, bits, group, kind='mxfp4' if mode == 'mxfp4' else 'affine')
        out.append(expert)
    return out, out[0]['gate_proj']['cols']


def nbytes(expert):
    return sum(
        p['weight'].nbytes + p['scales'].nbytes + (0 if p['biases'] is None else p['biases'].nbytes)
        for p in expert.values()
    )


def describe(d, most):
    chunks = d['by_workers'] + d['by_caller']

    def per(ns, n):
        return f'{ns / n / 1e3:.1f} us' if n else '-'

    woke = f"{d['woke']}/{d['rung']} woke"
    late = f", {d['wake_ns'] / d['woke'] / 1e3:.0f} us after ring (max {most / 1e3:.0f} us)" if d['woke'] else ''
    return (
        f"{woke}{late}; chunks: workers {d['by_workers']}/{chunks} ({per(d['workers_ns'], d['by_workers'])} each), "
        f"caller {d['by_caller']} ({per(d['caller_ns'], d['by_caller'])} each), stolen {d['stolen']}"
    )


def where(d, held):
    core = ce.topology()
    workers = {core.get(c) for c in held} - {None}
    shared = sorted(c for c in d['caller_cpus'] if core.get(c) in workers)
    return (
        f"CPUs: workers {sorted(d['worker_cpus'])} (held {held or 'none'}; {d['moved']} moved), "
        f"caller {sorted(d['caller_cpus'])}" + (f', shared core {shared}' if shared else '')
    )


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    spec = _option('shape', None)
    experts, D = synthetic(spec) if spec else from_checkpoint(args[0], *(int(a) for a in args[1:3]))
    threads = [int(t) for t in _option('threads', '1,2,4,8').split(',')]
    isas = [ce.ISA_NAMES.index(n) for n in _option('isa', ','.join(ce.ISA_NAMES[ce.isa()::-1])).split(',')]
    spins = [int(v) for v in _option('spin', str(ce.SPIN_US)).split(',')]
    priorities = _option('priority', ce.PRIORITIES[0]).split(',')
    activation = _option('activation', ce.ACTIVATIONS[0])
    limit = float(_option('limit', '0'))
    size = nbytes(experts[0])
    print(f'expert size: {size / 1e6:.2f} MB ({experts[0]["gate_proj"]["kind"]}, D={D}); '
          f'host ISA: {ce.ISA_NAMES[ce.isa()]}', flush=True)
    x = np.random.default_rng(1).normal(size=D).astype(np.float32)
    for level, n, spin, priority in ((lv, t, sp, pr) for lv in isas for t in threads for sp in spins
                                     for pr in priorities):
        pool = ce.Pool(n, level, priority=priority, spin_us=spin)
        ce._STATE.update(checked=0, failed=None)
        pool.expert(experts[0], x, limit, 0.5, activation)
        if ce.verified()[1]:
            print(f'  {ce.ISA_NAMES[level]:7s} x{n}: failed verification ({ce.verified()[1]})', flush=True)
            pool.close()
            continue
        before = pool.stats()
        times = []
        for k in range(REPEAT):
            t = time.perf_counter()
            pool.expert(experts[k % len(experts)], x, limit, 0.5, activation)
            times.append(time.perf_counter() - t)
        after = pool.stats()
        pool.close()
        med = float(np.median(times))
        print(f'  {ce.ISA_NAMES[level]:7s} x{n} spin {spin:3d} us {priority:6s}: {med * 1e3:7.3f} ms/expert, '
              f'{size / med / 1e9:6.1f} GB/s', flush=True)
        d = {k: after[k] if isinstance(after[k], set) else after[k] - before[k] for k in after}
        print('      ' + describe(d, after['wake_ns_most']), flush=True)
        print('      ' + where(d, pool.cpus), flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main()
