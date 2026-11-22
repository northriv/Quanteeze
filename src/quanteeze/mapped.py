# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Single whole-file memory maps and typed views inside them.

Each open file needs at most one file descriptor regardless of how many
tensor slices live inside it, preventing FD exhaustion on wide models.
"""

from pathlib import Path
import numpy as np


def view(files, file, dtype, offset, shape):
    """Return a typed array view of ``shape`` starting at byte ``offset`` inside ``file``."""
    file = Path(file)
    whole = files.get(file)
    if whole is None:
        whole = files[file] = np.memmap(file, dtype=np.uint8, mode='r')
    dtype = np.dtype(dtype)
    n = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if offset < 0 or offset + n > whole.size:
        raise ValueError(f'{file}: {n} bytes at {offset} lie past its {whole.size}')
    out = whole[offset:offset + n].view(dtype).reshape(shape)
    out.offset = int(offset)
    return out
