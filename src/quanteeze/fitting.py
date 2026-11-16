# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""How much memory a checkpoint's routed experts take at each uniform bit-width, and whether
they fit in the memory this machine leaves for them.

- Uniform least-squares quantization (`quanteeze.requant`) uses groups of 64 at 4 bits and above,
  and `requant.GROUP_SIZE` (128) at 3, 2, and 1 bit (`group_size`).
- What the routed experts may take (`room`): Metal's recommended working set (off macOS,
  `guards.WORKING_SET_SHARE` of the GPU's memory or of MemTotal) less what the rest of the model
  holds, and less `RESERVE` (`QUANTEEZE_MEMORY_RESERVE`, 0.3) of the working set for the prompts'
  caches, their passes, APC and MLX's freed buffers.
- A model whose experts (at the chosen bit-width) fit that room is served whole in memory; one
  whose experts do not is served on demand from the SSD within that room (`quanteeze.serve`).
"""
import mlx.core as mx

from . import guards, requant, variables

LADDER = (8, 4, 3, 2, 1)
GGML_MODES = ('q3_k', 'q2_k', 'iq2_xs')
FLOAT_BITS = 16                       # an unquantized checkpoint's experts
MODEL = [None]                        # active model_type, set by serve during model load
RESERVE_ON_CUDA = {'deepseek_v41': 0.13}


def lower_bits(stock_bits):
    """The uniform bit-widths below `stock_bits`, the most bits first."""
    return [b for b in LADDER if b < stock_bits]


def group_size(bits):
    """The group size of uniform quantization at `bits` (256 for ggml K/IQ formats, 32 for MXFP4)."""
    if bits == 'mxfp4':
        return 32
    if bits in GGML_MODES:
        return 256
    return 64 if int(bits) >= 4 else requant.GROUP_SIZE


def switch_bits(switch):
    """The bits of a SwitchGLU's experts (the least of its projections'), FLOAT_BITS when they are
    not quantized."""
    bits = [getattr(getattr(switch, p), 'bits', None) for p in ('gate_proj', 'up_proj', 'down_proj')]
    return FLOAT_BITS if None in bits else min(bits)


def weights(switch):
    """The number of scalar weights in a SwitchGLU's experts across all projections."""
    n = 0
    for p in ('gate_proj', 'up_proj', 'down_proj'):
        projection = getattr(switch, p)
        if projection.weight.ndim == 4 or getattr(projection, 'mode', 'affine') in GGML_MODES:
            experts, rows, nb, _ = projection.weight.shape
            n += experts * rows * nb * int(getattr(projection, 'group_size', 256))
            continue
        experts, rows, columns = projection.weight.shape
        bits = getattr(projection, 'bits', None)
        n += experts * rows * (columns * 32 // bits if bits else columns)
    return n


def quantized_bytes(weights, bits):
    """Bytes of `weights` scalar weights quantized uniformly at `bits` (integer affine bits, `'mxfp4'`,
    or ggml format `'q3_k'`, `'q2_k'`, `'iq2_xs'`)."""
    if bits == 'mxfp4':
        return weights * (4 + 8 / 32) / 8
    if bits in GGML_MODES:
        from . import ggml_quants
        return weights * ggml_quants.BITS[bits] / 8
    b = int(bits)
    return weights * (b + 32 / group_size(b)) / 8


def reserve():
    if (
        not variables.is_set('QUANTEEZE_MEMORY_RESERVE')
        and MODEL[0] in RESERVE_ON_CUDA
        and not mx.metal.is_available()
        and mx.default_device() == mx.gpu
    ):
        return RESERVE_ON_CUDA[MODEL[0]]
    return variables.number('QUANTEEZE_MEMORY_RESERVE')


HOST_RESERVE = 0.1               # share of the host's memory (or the job's cgroup limit) host_room() leaves


def host_room(readings=None):
    """Bytes of the host's memory free to hold experts in (`q2_experts.hold`): what is available
    (MemAvailable, or what the memory cgroup's limit leaves where that is less), less `HOST_RESERVE`
    of the whole (MemTotal, or the limit); None where nothing is known (off Linux)."""
    readings = guards.Readings() if readings is None else readings
    info = readings._meminfo() or {}
    total, free = info.get('MemTotal'), info.get('MemAvailable')
    limited = readings.cgroup_memory()
    if limited is not None:
        total = min(total, limited[0]) if total else limited[0]
        free = min(free, limited[0] - limited[1]) if free is not None else limited[0] - limited[1]
    if not total or free is None:
        return None
    return max(0, int(free - HOST_RESERVE * total))


def room(resident, working_set=None):
    """Bytes the routed experts may take beside `resident` (what MLX holds of the rest), or None
    where no working set is known (`guards.recommended_budget`)."""
    working_set = guards.recommended_budget() if working_set is None else working_set
    if not working_set:
        return None
    return working_set * (1 - reserve()) - resident
