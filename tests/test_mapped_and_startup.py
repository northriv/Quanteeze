# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from quanteeze import control, embed_ssd, home, mapped, ple_store, serve, vision_ssd


def _write_safetensors(path, tensors):
    """Write {name: np.ndarray} in safetensors format using raw numpy arrays (BF16 for uint16)."""
    offset = 0
    header = {}
    blobs = []
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        dtype = 'U32' if arr.dtype == np.uint32 else 'BF16' if arr.dtype == np.uint16 else 'F32'
        header[name] = {'dtype': dtype, 'shape': list(arr.shape), 'data_offsets': [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    meta = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack('<Q', len(meta)) + meta + b''.join(blobs))


def test_mapped_view_single_map_and_bounds(tmp_path):
    p = tmp_path / 'blob.bin'
    a = np.arange(12, dtype=np.uint16).reshape(3, 4)
    b = np.arange(100, 108, dtype=np.uint32).reshape(2, 4)
    p.write_bytes(a.tobytes() + b.tobytes())
    files = {}
    va = mapped.view(files, p, np.uint16, 0, (3, 4))
    vb = mapped.view(files, p, np.uint32, a.nbytes, (2, 4))
    assert len(files) == 1
    assert va.offset == 0
    assert vb.offset == a.nbytes
    assert np.array_equal(va, a)
    assert np.array_equal(vb, b)
    with pytest.raises(ValueError):
        mapped.view(files, p, np.uint32, a.nbytes + 4, (2, 4))


def test_ple_store_single_file_and_shared(tmp_path):
    w0 = np.arange(8, dtype=np.uint32).reshape(2, 4)
    s0 = np.arange(10, 14, dtype=np.uint16).reshape(2, 2)
    w1 = np.arange(20, 28, dtype=np.uint32).reshape(2, 4)
    s1 = np.arange(30, 34, dtype=np.uint16).reshape(2, 2)
    st = tmp_path / 'model.safetensors'
    _write_safetensors(st, {'s0.weight': w0, 's0.scales': s0, 's1.weight': w1, 's1.scales': s1})
    hdr_len = struct.unpack('<Q', st.read_bytes()[:8])[0]
    hdr = json.loads(st.read_bytes()[8:8 + hdr_len])
    base = 8 + hdr_len

    class FakePle:
        pass

    obj = FakePle()
    obj._shards = [
        (0, 2, {
            'weight': np.memmap(st, dtype=np.uint32, mode='r', offset=base + hdr['s0.weight']['data_offsets'][0], shape=(2, 4)),
            'scales': np.memmap(st, dtype=np.uint16, mode='r', offset=base + hdr['s0.scales']['data_offsets'][0], shape=(2, 2)),
        }),
        (2, 4, {
            'weight': np.memmap(st, dtype=np.uint32, mode='r', offset=base + hdr['s1.weight']['data_offsets'][0], shape=(2, 4)),
            'scales': np.memmap(st, dtype=np.uint16, mode='r', offset=base + hdr['s1.scales']['data_offsets'][0], shape=(2, 2)),
        }),
    ]
    ple_store._shared(obj)
    bases = {id(v.base) for _, _, d in obj._shards for v in d.values()}
    assert len(bases) == 1
    assert np.array_equal(obj._shards[0][2]['weight'], w0)
    assert np.array_equal(obj._shards[1][2]['scales'], s1)


def test_vision_ssd_and_embed_ssd_share_one_map(tmp_path):
    vw1 = np.arange(6, dtype=np.uint16).reshape(2, 3)
    vw2 = np.arange(10, 16, dtype=np.uint16).reshape(2, 3)
    ew = np.arange(8, dtype=np.uint32).reshape(4, 2)
    es = np.arange(20, 24, dtype=np.uint16).reshape(4, 1)
    st = tmp_path / 'model.safetensors'
    _write_safetensors(st, {
        'vision_tower.w1': vw1,
        'vision_tower.w2': vw2,
        'embed_tokens.weight': ew,
        'embed_tokens.scales': es,
    })
    vmaps = vision_ssd.bf16_maps(tmp_path, ['vision_tower.w1', 'vision_tower.w2'])
    assert vmaps['vision_tower.w1'].base is vmaps['vision_tower.w2'].base
    assert np.array_equal(vmaps['vision_tower.w1'], vw1)
    emaps = embed_ssd._maps(tmp_path, ['embed_tokens.weight', 'embed_tokens.scales'])
    assert emaps['embed_tokens.weight'][0].base is emaps['embed_tokens.scales'][0].base
    assert np.array_equal(emaps['embed_tokens.scales'][0], es)


def test_more_open_files_behavior():
    class FakeResource:
        RLIMIT_NOFILE = 7
        RLIM_INFINITY = -1

        def __init__(self, soft, hard, fail_above=None):
            self.limits = (soft, hard)
            self.fail_above = fail_above

        def getrlimit(self, kind):
            return self.limits

        def setrlimit(self, kind, new_limits):
            if self.fail_above is not None and new_limits[0] > self.fail_above:
                raise ValueError('too high')
            self.limits = new_limits

    r1 = FakeResource(256, 100000)
    assert serve._more_open_files(r1) == serve.OPEN_FILES
    assert r1.limits[0] == serve.OPEN_FILES

    r2 = FakeResource(256, -1, fail_above=10240)
    assert serve._more_open_files(r2) == 10240

    r3 = FakeResource(70000, 100000)
    assert serve._more_open_files(r3) == 70000


def test_home_model_roots_volumes(tmp_path, monkeypatch):
    ext = tmp_path / 'DiskA' / 'lmstudio'
    ext.mkdir(parents=True)
    monkeypatch.setattr(home, 'VOLUMES', tmp_path)
    roots = home.model_roots()
    assert ext in roots


def test_control_switch_model_undo_and_choices(tmp_path, monkeypatch):
    assert 'auto' in control.CHOICES['QUANTEEZE_DEMAND_STM']
    model_dir = tmp_path / 'm'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text('{"model_type": "qwen3_moe"}')

    monkeypatch.setitem(os.environ, 'QUANTEEZE_BITS', '3')
    saved_map = {}
    monkeypatch.setattr(control, 'saved', lambda: dict(saved_map))
    monkeypatch.setattr(control, '_save', lambda d: (saved_map.clear(), saved_map.update(d)))
    monkeypatch.setitem(control._SWITCH, 'served', 'prev-model')

    def fail_loader(_target):
        raise RuntimeError('oom')

    monkeypatch.setitem(control._SWITCH, 'loader', fail_loader)
    monkeypatch.setitem(control._SWITCH, 'check', None)
    monkeypatch.setitem(control._SWITCH, 'busy', False)
    monkeypatch.setitem(control._SWITCH, 'inflight', 0)

    control.switch_model(model_dir, wait=0.01, undo={'QUANTEEZE_BITS': '4'})
    deadline = time.time() + 2.0
    while control._SWITCH['busy'] and time.time() < deadline:
        time.sleep(0.01)
    assert control._SWITCH['state'] == 'failed'
    assert os.environ.get('QUANTEEZE_BITS') == '4'
    assert saved_map.get('QUANTEEZE_BITS') == '4'

    # If user edits setting mid-load, undo leaves their newer value intact.
    def fail_after_change(_target):
        os.environ['QUANTEEZE_BITS'] = '2'
        raise RuntimeError('oom2')

    monkeypatch.setitem(os.environ, 'QUANTEEZE_BITS', '3')
    monkeypatch.setitem(control._SWITCH, 'loader', fail_after_change)
    control.switch_model(model_dir, wait=0.01, undo={'QUANTEEZE_BITS': '4'})
    deadline = time.time() + 2.0
    while control._SWITCH['busy'] and time.time() < deadline:
        time.sleep(0.01)
    assert os.environ.get('QUANTEEZE_BITS') == '2'
