# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Quanteeze: uniform least-squares MoE quantization & high-speed MLX inference server."""
import importlib
import importlib.abc
import importlib.util
import os
import sys

# A Metal command buffer holds this many operations and MB before MLX commits it (where the environment
# says nothing).  MLX reads them as it is imported, so here, before mlx.core is imported.
MLX_BUFFERS = {'MLX_MAX_OPS_PER_BUFFER': '300', 'MLX_MAX_MB_PER_BUFFER': '4000'}
if sys.platform == 'darwin':
    for _name, _value in MLX_BUFFERS.items():
        os.environ.setdefault(_name, _value)


def _gpu_model(pattern='/proc/driver/nvidia/gpus/*/information'):
    """The first NVIDIA GPU's model as the driver names it ('NVIDIA_A100-SXM4-40GB'), or None."""
    import glob
    for info in sorted(glob.glob(pattern)):
        with open(info) as f:
            for line in f:
                if line.startswith('Model:'):
                    return '_'.join(line.split(':', 1)[1].split())
    return None


if sys.platform != 'darwin' and _gpu_model():
    os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER', '200')
    os.environ.setdefault('MLX_MAX_MB_PER_BUFFER', '100000')
    os.environ.setdefault('MLX_CUDA_GRAPH_CACHE_SIZE', '100')
    os.environ.setdefault('CUDA_CACHE_MAXSIZE', str(4 << 30))
    try:
        from importlib.metadata import version as _version
        os.environ.setdefault('MLX_PTX_CACHE_DIR', os.path.join(
            os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
            'mlx', _version('mlx'), _gpu_model(), 'ptx',
        ))
    except Exception:
        pass

# mlx-vlm's modules quanteeze patches as they are imported, before any model is built or loaded.
PATCHES = {
    'mlx_vlm.utils': lambda m: importlib.import_module('quanteeze.checkpoints').install(m),
    'mlx_vlm.models.qwen4_exp.ple_storage':
        lambda m: importlib.import_module('quanteeze.ple_store').install(m.QuantizedMMapNGramEmbedding),
}


class _Patcher(importlib.abc.MetaPathFinder):
    """Runs PATCHES[name] on a module of PATCHES once it is executed (`quanteeze.checkpoints`: an
    MTP head under another name; `quanteeze.ple_store`: an n-gram table in any layout)."""

    def __init__(self):
        self.busy = set()

    def find_spec(self, fullname, path, target=None):
        if fullname not in PATCHES or fullname in self.busy:
            return None
        self.busy.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self.busy.discard(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return spec
        exec_module = spec.loader.exec_module

        def run(module):
            exec_module(module)
            PATCHES[fullname](module)

        spec.loader.exec_module = run
        return spec


for _name, _patch in PATCHES.items():
    if _name in sys.modules:
        _patch(sys.modules[_name])
if not any(isinstance(f, _Patcher) for f in sys.meta_path):
    sys.meta_path.insert(0, _Patcher())
