# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""quanteeze's own folders kept out of Spotlight's index (macOS).

Spotlight indexes every folder of the data volume it is not told to leave, and its index grew
until it filled the disk.  ~/.quanteeze holds the n-gram views and quantization caches;
~/.cache/mlx-vlm the prompt cache on disk.  On macOS (26) a .metadata_never_index file in a
folder did not keep Spotlight out, and a folder named *.noindex did.  So each of folders() lives
at <its name>.noindex, and its own name is a link to that: every path the code and the settings
name goes on working, and Spotlight follows no link.  The server does it as it starts
(unindex_all): a folder of its own moved there (a rename on its disk, at once), or made there
where there is none.  A link already there, or both a folder and its .noindex, are left as they
are.  Off macOS nothing is done.
"""
import os
import sys
from pathlib import Path

SUFFIX = '.noindex'


def folders():
    """The folders kept out of the index: quanteeze's own and mlx-vlm's prompt cache."""
    home = Path.home()
    return (home / '.quanteeze', home / '.cache' / 'mlx-vlm')


def unindexed(path, platform=None):
    """`path` made a link to <path>.noindex, the folder moved there or made there.  Returns what was
    done, a line to say, or None where nothing was."""
    if (platform or sys.platform) != 'darwin':
        return None
    path = Path(path)
    target = path.with_name(path.name + SUFFIX)
    if path.is_symlink():
        return None
    if path.exists() and not path.is_dir():
        return None
    if path.exists() and target.exists():
        return f'{path} and {target} are both there: left as they are, {path} indexed by Spotlight'
    moved = path.exists()
    if moved:
        os.rename(path, target)
    else:
        target.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target.name, path)
    except FileExistsError:
        return f'{path} was made again while it moved to {target}: both left there'
    return f'{path} {"moved to" if moved else "made at"} {target}, out of Spotlight\'s index, {path} a link to it'


def unindex_all(platform=None):
    """unindexed() of every folder of folders(); the lines to say."""
    out = []
    for path in folders():
        try:
            line = unindexed(path, platform)
        except OSError as e:
            line = f'{path} not moved out of Spotlight\'s index ({type(e).__name__}: {e})'
        if line:
            out.append(line)
    return out
