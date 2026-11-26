# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Shared pytest setup."""
import os
import sys

os.environ.setdefault('MLX_CUDA_GRAPH_CACHE_SIZE', '2000')
os.environ.setdefault('APC_DISK_MIN_FREE_RAM_GB', '0')
# Pin Linux test runs to a single visible GPU before MLX initializes so multi-GPU hosts do not leave
# another device active across `mx.clear_cache()` when copying host buffers from GPU 0's pool.
if sys.platform.startswith('linux'):
    os.environ['CUDA_VISIBLE_DEVICES'] = (os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0].strip()

import pytest  # noqa: E402

from quanteeze import apc_disk, serve, variables  # noqa: E402

# The server installs a `/proc/meminfo` free-RAM reader into `mlx_vlm.apc` at startup
# (mlx_vlm only knows psutil and macOS `vm_stat`). Mirror that here so APC tests on
# Linux without psutil see the same memory headroom as the running server.
serve._apc_reads_free_memory()
apc_disk.use_secret(bytes(range(32)))


@pytest.fixture(autouse=True)
def _isolate_env_and_snapshots():
    """Restore `os.environ` and clear start/load setting snapshots around every test."""
    saved = dict(os.environ)
    variables._AT_LOAD[0] = None
    variables._AT_START[0] = None
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        variables._AT_LOAD[0] = None
        variables._AT_START[0] = None
