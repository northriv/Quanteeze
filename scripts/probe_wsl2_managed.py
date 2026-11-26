# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Whether a CUDA machine holds more than its GPU's memory the way Quanteeze's CUDA path needs: arrays made on
MLX's CPU stream live in managed memory, past the GPU's own, and a GPU kernel reads them all (for example, 1 GB x 20
on a 16 GB RTX 5060 Ti).  Meant for WSL2 (and native Windows), where NVIDIA's Unified Memory may use Windows' feature
set -- no concurrent managed access, no demand paging, no oversubscription.
Needs MLX with CUDA alone (no Quanteeze install).

    python scripts/probe_wsl2_managed.py [--times 1.5] [--chunk-gb 1]

Prints: the platform (WSL2 or not), the driver's attributes (managed memory, concurrent managed access, pageable
memory access), the GPU's memory, then makes CPU-stream arrays of `times` x the GPU's memory in `chunk-gb` pieces
(the host must have that much free: WSL2 takes half the Windows host's by default, .wslconfig raises it) and has
the GPU sum each, then all of them again, with the seconds each took.  Writes nothing to disk.
"""
import argparse
import ctypes
import ctypes.util
import platform
import sys
import time

ATTRIBUTES = {'managed memory': 83, 'pageable memory access': 88, 'concurrent managed access': 89}


def wsl():
    try:
        with open('/proc/version', encoding='utf-8') as f:
            return 'microsoft' in f.read().lower()
    except OSError:
        return False


def driver():
    """(attribute: value, the GPU's memory in bytes, its name) from the CUDA driver, or an error."""
    cuda = None
    for name in ('libcuda.so.1', '/usr/lib/wsl/lib/libcuda.so.1', 'nvcuda.dll'):
        try:
            cuda = ctypes.CDLL(name)
            break
        except OSError:
            cuda = None
    if cuda is None:
        raise SystemExit('no CUDA driver library (libcuda.so.1, /usr/lib/wsl/lib/libcuda.so.1)')
    check = lambda r, what: r == 0 or sys.exit(f'{what}: CUDA error {r}')     # noqa: E731
    check(cuda.cuInit(0), 'cuInit')
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0), 'cuDeviceGet')
    got = {}
    for label, number in ATTRIBUTES.items():
        v = ctypes.c_int()
        r = cuda.cuDeviceGetAttribute(ctypes.byref(v), number, dev)
        got[label] = v.value if r == 0 else f'error {r}'
    total = ctypes.c_size_t()
    check(cuda.cuDeviceTotalMem_v2(ctypes.byref(total), dev), 'cuDeviceTotalMem')
    name = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name, 256, dev)
    return got, total.value, name.value.decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--times', type=float, default=1.5, help="the arrays' total over the GPU's memory")
    ap.add_argument('--chunk-gb', type=float, default=1.0)
    args = ap.parse_args()
    print(f'platform: {platform.platform()}; WSL2: {wsl()}')
    attrs, total, name = driver()
    print(f'GPU: {name}, {total / 2**30:.1f} GiB')
    for label, v in attrs.items():
        print(f'  {label}: {v}')
    import mlx.core as mx
    print(f'mlx {mx.__version__}; default device {mx.default_device()}')
    chunk = int(args.chunk_gb * 2**30) // 4
    n = max(1, int(round(args.times * total / (chunk * 4))))
    print(f'making {n} arrays of {chunk * 4 / 2**30:.2f} GiB on the CPU stream ({n * chunk * 4 / 2**30:.1f} GiB, '
          f'{n * chunk * 4 / total:.2f} x the GPU\'s memory)')
    arrays = []
    try:
        for i in range(n):
            a = mx.full((chunk,), float(i + 1), dtype=mx.float32, stream=mx.cpu)
            mx.eval(a)
            arrays.append(a)
        print('  made')
        for rnd in (1, 2):
            times = []
            for i, a in enumerate(arrays):
                t = time.perf_counter()
                s = mx.sum(a, stream=mx.gpu)
                mx.eval(s)
                times.append(time.perf_counter() - t)
                want = float(i + 1) * chunk
                if abs(s.item() - want) > 1e-3 * want:
                    print(f'  round {rnd}, array {i}: sum {s.item()} where {want} was wanted')
            print(f'  round {rnd}: the GPU read all {n} ({sum(times):.2f} s; '
                  f'{n * chunk * 4 / 2**30 / sum(times):.1f} GiB/s; slowest {max(times):.3f} s)')
        print('RESULT: more than the GPU\'s memory held and read -- the CUDA path\'s premise holds here')
    except Exception as e:                                         # noqa: BLE001
        print(f'RESULT: failed after {len(arrays)} arrays ({len(arrays) * chunk * 4 / 2**30:.1f} GiB): '
              f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
