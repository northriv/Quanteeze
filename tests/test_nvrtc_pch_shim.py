# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""CPU-only Linux shim test for `vendor/nvrtc_pch/nvrtc_pch.cpp` against a stand-in `libnvrtc.so.13`."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith('linux') or not (shutil.which('cc') or shutil.which('gcc')),
    reason='Linux and a C compiler',
)

STAND_IN = r"""
#include <stddef.h>
#include <string.h>
static int pch_options = 0, call_idx = 0;
static size_t heap = 268435456, set_to = 0;
int nvrtcCreateProgram(void **p, const char *src, const char *name, int n, const char *const *c, const char *const *h) {
    (void)src; (void)name; (void)n; (void)c; (void)h;
    *p = (void *)0x1234;
    return 0;
}
int nvrtcCompileProgram(void *p, int n, const char *const *options) {
    (void)p;
    for (int i = 0; i < n; ++i)
        if (options[i] && strcmp(options[i], "--pch") == 0) ++pch_options;
    return 0;
}
int nvrtcGetPCHCreateStatus(void *p) {
    (void)p;
    int idx = call_idx++;
    /* First custom compile triggers HEAP_EXHAUSTED (14) -> grow to 402653184;
     * second triggers HEAP_EXHAUSTED with a smaller required size -> shrink must be ignored;
     * subsequent custom compiles succeed (0) and consume 402653184 B each until MOST_BYTES cap stops --pch. */
    if (idx == 0 || idx == 1) return 14;
    return 0;
}
int nvrtcGetPCHHeapSizeRequired(void *p, size_t *out) {
    (void)p;
    *out = (call_idx == 1) ? 402653184 : 200000000;
    return 0;
}
int nvrtcGetPCHHeapSize(size_t *out) { *out = heap; return 0; }
int nvrtcSetPCHHeapSize(size_t bytes) { heap = bytes; set_to = bytes; return 0; }
int stand_in_pch_options(void) { return pch_options; }
size_t stand_in_set_to(void) { return set_to; }
"""

PROBE = r"""
import ctypes, json, sys
nvrtc = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
shim = ctypes.CDLL(sys.argv[2])
assert shim.qz_nvrtc_pch_dir(sys.argv[3].encode()) == 1
shim.qz_nvrtc_pch_grown.restype = ctypes.c_uint64

out = []
names = [
    b'custom_kernel_grow',
    b'mlx_own',
    b'custom_kernel_shrink_ignored',
    b'custom_kernel_1',
    b'custom_kernel_2',
    b'custom_kernel_3',
    b'custom_kernel_4',
    b'custom_kernel_5',
    b'custom_kernel_over_cap',
]
for name in names:
    program = ctypes.c_void_p()
    assert shim.nvrtcCreateProgram(ctypes.byref(program), b'', name, 0, None, None) == 0
    assert shim.nvrtcCompileProgram(program, 0, None) == 0
    out.append(nvrtc.stand_in_pch_options())

counts, heap = (ctypes.c_uint64 * 3)(), (ctypes.c_uint64 * 3)()
shim.qz_nvrtc_pch_counts(counts)
shim.qz_nvrtc_pch_heap(heap)
nvrtc.stand_in_set_to.restype = ctypes.c_size_t
print(json.dumps({
    'pch_after_each': out,
    'counts': list(counts),
    'grown': shim.qz_nvrtc_pch_grown(),
    'heap': list(heap),
    'set_to': nvrtc.stand_in_set_to(),
}))
"""


def test_nvrtc_pch_discovers_cuda13_grows_heap_and_caps_total_bytes(tmp_path, monkeypatch):
    from quanteeze import native_ext

    monkeypatch.setenv('QUANTEEZE_NATIVE', 'on')
    shim = native_ext.library('nvrtc_pch', force=True, quiet=False)
    assert shim is not None and shim.is_file()

    cc = shutil.which('cc') or shutil.which('gcc')
    source = tmp_path / 'nvrtc.c'
    source.write_text(STAND_IN)
    library = tmp_path / 'libnvrtc.so.13'
    subprocess.run([cc, '-shared', '-fPIC', '-o', str(library), str(source)], check=True)

    pch = tmp_path / 'pch'
    pch.mkdir()
    done = subprocess.run(
        [sys.executable, '-c', PROBE, str(library), str(shim), str(pch)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(Path(__file__).parent),
    )
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout.strip().splitlines()[-1])
    # 9 compiles total; mlx_own skipped; custom_kernel_over_cap skipped once 5 * 402,653,184 + 402,653,184 > 2 GiB
    assert got['pch_after_each'] == [1, 1, 2, 3, 4, 5, 6, 7, 7]
    assert got['counts'] == [9, 7, 5]
    assert got['grown'] == 1
    assert got['set_to'] == 402653184
    assert got['heap'] == [1, 402653184, 5 * 402653184]
