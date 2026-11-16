# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Tests for automatic hardware configuration across A100 multi-GPU supercomputers,
SLURM cpusets, dual-socket NUMA nodes, Lustre network filesystems, PCIe/CPU balance,
and integrated-memory CUDA SoCs (`parallel`, `cpu_experts`, `q2_experts`, `serve`, `cuda_ple`)."""
from contextlib import ExitStack
import json
import multiprocessing as mp
import struct
import threading

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from quanteeze import cpu_experts as ce
from quanteeze import cuda_ple, parallel, q2_experts, serve, variables


def _v41_test_config():
    from mlx_vlm.models.deepseek_v41 import ModelConfig as V41Config

    return V41Config(
        vocab_size=64,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=12,
        num_attention_heads=2,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        sliding_window=4,
        compress_ratios=[0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
        kv_source_layer_ids=[2, 6],
        index_source_layer_ids=[2, 6, 8, 10],
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        candidate_source_layer_id=6,
        candidate_topk_blocks=4,
        candidate_block_size=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        engram_layer_ids=[],
        vision_num_layers=1,
        vision_hidden_size=32,
        vision_num_heads=2,
        vision_intermediate_size=32,
    )


def _v41_build_model(seed):
    from mlx_vlm.models.deepseek_v41 import Model as V41Model
    from quanteeze.arch import deepseek_v41 as q_v41

    cfg = _v41_test_config()
    rng = np.random.default_rng(seed)
    proto = V41Model(cfg)
    shared_weights = [
        (k, mx.array(rng.normal(0, 0.05, size=v.shape).astype(np.float32)).astype(v.dtype))
        for k, v in tree_flatten(proto.parameters())
    ]
    mx.eval([v for _, v in shared_weights])
    m = V41Model(cfg)
    m.load_weights(shared_weights, strict=False)
    q_v41.speculative(m)
    extra = [v for _, mod in m.named_modules() for v in vars(mod).values() if isinstance(v, mx.array)]
    mx.eval(m.parameters(), *extra)
    return m, cfg


def _v41_run_single(m):
    prompt = mx.array([[1, 5, 9, 3, 7, 2]], dtype=mx.int32)
    dec1 = mx.array([[4]], dtype=mx.int32)
    dec2 = mx.array([[8]], dtype=mx.int32)
    c = m.language_model.make_cache()
    l0 = m.language_model(prompt, cache=c).logits
    l1 = m.language_model(dec1, cache=c).logits
    l2 = m.language_model(dec2, cache=c).logits
    mx.eval(l0, l1, l2)
    return [np.array(l0), np.array(l1), np.array(l2)]


def _v41_run_batched(m, num_hidden_layers):
    from mlx_vlm.models.deepseek_v41.language import BatchDeepseekV41Cache, DeepseekV41Cache

    b_p = mx.array([[0, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.int32)
    b_d = mx.array([[9], [10]], dtype=mx.int32)
    rows = [DeepseekV41Cache(num_hidden_layers), DeepseekV41Cache(num_hidden_layers)]
    bc = BatchDeepseekV41Cache(rows, left_padding=[1, 0])
    l0 = m.language_model(b_p, cache=[bc]).logits
    l1 = m.language_model(b_d, cache=[bc]).logits
    mx.eval(l0, l1)
    return [np.array(l0), np.array(l1)]


def _v41_rank_worker(rank_id, n_ranks, seed, conn):
    import traceback

    mx.set_default_device(mx.cpu)
    parallel.GROUP[:] = [object()]
    parallel.size = lambda: n_ranks
    parallel.rank = lambda: rank_id

    def _collective(x, op):
        dt = x.dtype
        mx.eval(x)
        arr = np.array(x.astype(mx.float32) if dt == mx.bfloat16 else x)
        conn.send(('coll', arr, str(dt), dt == mx.int32, x.shape))
        summed_np = conn.recv()
        return mx.array(summed_np).astype(dt)

    parallel._collective = _collective
    try:
        m_single, _ = _v41_build_model(seed)
        parallel.split(m_single)
        single_outs = _v41_run_single(m_single)

        m_batched, cfg = _v41_build_model(seed)
        parallel.split(m_batched)
        batched_outs = _v41_run_batched(m_batched, cfg.num_hidden_layers)
        conn.send(('done', single_outs, batched_outs))
    except BaseException:  # noqa: BLE001
        conn.send(('err', traceback.format_exc()))
    finally:
        conn.close()


def _write_minimal_gguf(path, tensors, alignment=32):
    def pack_str(s):
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b

    buf = bytearray()
    buf += b'GGUF'
    buf += struct.pack('<IQQ', 3, len(tensors), 1)
    buf += pack_str('general.alignment') + struct.pack('<II', 4, alignment)
    offset = 0
    blobs = []
    for name, dims, kind, data in tensors:
        buf += pack_str(name)
        buf += struct.pack('<I', len(dims))
        buf += struct.pack(f'<{len(dims)}Q', *dims)
        buf += struct.pack('<IQ', kind, offset)
        pad = (-len(data)) % alignment
        blobs.append(data + b'\x00' * pad)
        offset += len(data) + pad
    buf += b'\x00' * ((-len(buf)) % alignment)
    with open(path, 'wb') as f:
        f.write(buf)
        for blob in blobs:
            f.write(blob)


# ---------------------------------------------------------------- 1. parallel.sizes & parallel.plan


def test_parallel_plan_auto_working_set_share_and_deepseek_v41_sizing(tmp_path, monkeypatch):
    """`QUANTEEZE_WORKING_SET_SHARE=auto` must not crash `plan()`; `deepseek_v41` uses `0.13` reserve,
    skips off-GPU Engram/embed/MTP/vision tensors, and scales with `QUANTEEZE_BITS` / `--experts-gguf`."""
    ckpt = tmp_path / 'v41'
    ckpt.mkdir()
    config = {
        'model_type': 'deepseek_v41',
        'quantization': {'group_size': 32, 'bits': 4},
        'text_config': {'model_type': 'deepseek_v41', 'num_hidden_layers': 4},
    }
    (ckpt / 'config.json').write_text(json.dumps(config), encoding='utf-8')

    tensors = {
        'embed.weight': mx.zeros((1024, 512), dtype=mx.bfloat16),
        'vision.blocks.0.weight': mx.zeros((512, 512), dtype=mx.bfloat16),
        'aligner.proj.weight': mx.zeros((512, 512), dtype=mx.bfloat16),
        'mtp.layers.0.weight': mx.zeros((512, 512), dtype=mx.bfloat16),
        'head.weight': mx.zeros((1024, 512), dtype=mx.bfloat16),
    }
    for l_idx in range(4):
        tensors[f'layers.{l_idx}.attn.wq_a.weight'] = mx.zeros((512, 512), dtype=mx.bfloat16)
        tensors[f'layers.{l_idx}.engram.embed.0.weight'] = mx.zeros((4096, 512), dtype=mx.uint8)
        for e in range(4):
            for w in ('w1', 'w2', 'w3'):
                tensors[f'layers.{l_idx}.ffn.experts.{e}.{w}.weight'] = mx.zeros((512, 512), dtype=mx.int8)
                tensors[f'layers.{l_idx}.ffn.experts.{e}.{w}.scale'] = mx.zeros((512, 32), dtype=mx.uint8)
    mx.eval(*tensors.values())
    mx.save_safetensors(str(ckpt / 'model.safetensors'), tensors)

    monkeypatch.delenv('QUANTEEZE_WORKING_SET_SHARE', raising=False)
    monkeypatch.delenv('QUANTEEZE_MEMORY_RESERVE', raising=False)
    monkeypatch.delenv('QUANTEEZE_BITS', raising=False)
    monkeypatch.delenv('QUANTEEZE_EXPERTS_GGUF', raising=False)
    monkeypatch.setenv('QUANTEEZE_V41_READS_IMAGES', 'ssd')

    layer_bytes, rest = parallel.sizes(ckpt)
    assert set(layer_bytes) == {0, 1, 2, 3}
    expected_head = 1024 * 512 * 2
    assert rest == expected_head  # embed.weight, vision, aligner, mtp, and engram.embed all skipped

    # Total weight fits across 4 GPUs with 0.13 reserve, does NOT fit 1 GPU
    whole = sum(layer_bytes.values()) + rest
    one_gpu = int(whole / 0.87 * 0.55)
    gpus = [(f'GPU-{i}', one_gpu) for i in range(4)]
    chosen, msg = parallel.plan(['--model', str(ckpt)], gpus=gpus)
    assert chosen == gpus
    assert 'QUANTEEZE_WORKING_SET_SHARE auto' in msg
    assert 'QUANTEEZE_MEMORY_RESERVE 0.13' in msg

    # Lowering experts via QUANTEEZE_BITS=2 reduces per-layer bytes so 1 GPU holds it
    monkeypatch.setenv('QUANTEEZE_BITS', '2')
    layer_bytes_2b, _ = parallel.sizes(ckpt)
    assert sum(layer_bytes_2b.values()) < sum(layer_bytes.values())
    tiny_one = int((sum(layer_bytes_2b.values()) + rest) / 0.87) + 1024
    chosen_2b, msg_2b = parallel.plan(['--model', str(ckpt)], gpus=[(f'GPU-{i}', tiny_one) for i in range(4)])
    assert chosen_2b is None and 'one GPU holds it' in msg_2b
    monkeypatch.delenv('QUANTEEZE_BITS', raising=False)

    # `--experts-gguf` substitutes GGUF expert sizes into parallel.sizes/plan
    from quanteeze import iq_experts

    gguf_file = tmp_path / 'tiny.gguf'
    gguf_entries = []
    for l_idx in range(4):
        for name in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
            gguf_entries.append((
                f'blk.{l_idx}.{name}.weight',
                (256, 256, 4),
                iq_experts.Q4_K,
                np.zeros((4, 256, 1, iq_experts.BLOCK[iq_experts.Q4_K]), dtype=np.uint8).tobytes(),
            ))
    _write_minimal_gguf(gguf_file, gguf_entries)
    layer_bytes_gguf, rest_gguf = parallel.sizes(ckpt, experts_gguf=gguf_file)
    assert rest_gguf == rest
    assert sum(layer_bytes_gguf.values()) < sum(layer_bytes.values())
    fit_gguf = int((sum(layer_bytes_gguf.values()) + rest_gguf) / 0.87) + 1024
    chosen_g, msg_g = parallel.plan(
        ['--model', str(ckpt), f'--experts-gguf={gguf_file}'],
        gpus=[(f'GPU-{i}', fit_gguf) for i in range(4)],
    )
    assert chosen_g is None and 'one GPU holds it' in msg_g

    # Unsupported model_type degrades cleanly to single-GPU under auto and exits under layers
    (ckpt / 'config.json').write_text(json.dumps({'model_type': 'unsupported_moe'}), encoding='utf-8')
    chosen_u, msg_u = parallel.plan(['--model', str(ckpt)], gpus=gpus)
    assert chosen_u is None and 'cannot be split over GPUs yet' in msg_u
    monkeypatch.setenv('QUANTEEZE_PARALLEL', 'layers')
    with pytest.raises(SystemExit):
        parallel.plan(['--model', str(ckpt)], gpus=gpus)


def test_parallel_gpus_with_free_parses_nvidia_smi(monkeypatch):
    fake_csv = '0, GPU-aaaa-0000, 81920, 80000\n1, GPU-bbbb-1111, 81920, 40000\n'
    monkeypatch.setattr(
        parallel.subprocess,
        'run',
        lambda *a, **kw: type('P', (), {'stdout': fake_csv})(),
    )
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert parallel._gpus_with_free() == [
        ('GPU-aaaa-0000', 81920 << 20, 80000 << 20),
        ('GPU-bbbb-1111', 81920 << 20, 40000 << 20),
    ]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,GPU-aaaa')
    assert parallel._gpus_with_free() == [
        ('1', 81920 << 20, 40000 << 20),
        ('GPU-aaaa', 81920 << 20, 80000 << 20),
    ]


# ---------------------------------------------------------------- 2. Bit-exact multi-rank glm5_next & deepseek_v41


def _run_multi_rank(n_ranks, build_model, run_pass, monkeypatch):
    """Simulate `n_ranks` processes concurrently with barrier-synchronized `all_sum`."""
    tls = threading.local()
    barrier = threading.Barrier(n_ranks)
    lock = threading.Lock()
    mlx_lock = threading.Lock()
    slots = [None] * n_ranks

    def fake_size():
        return n_ranks

    def fake_rank():
        return tls.rank

    def fake_collective(x, op):
        mx.eval(x)
        r = tls.rank
        with lock:
            slots[r] = (np.array(x), x.dtype, x.shape)
        mlx_lock.release()
        try:
            barrier.wait(timeout=10.0)
            with lock:
                shapes = {s[2] for s in slots}
                dtypes = {s[1] for s in slots}
                assert len(shapes) == 1, f'rank shape mismatch across collective: {shapes}'
                assert len(dtypes) == 1, f'rank dtype mismatch across collective: {dtypes}'
                dtype = slots[0][1]
                acc = sum(s[0].astype(np.float64 if dtype != mx.int32 else np.int64) for s in slots)
            barrier.wait(timeout=10.0)
        finally:
            mlx_lock.acquire()
        return mx.array(acc).astype(dtype)

    monkeypatch.setattr(parallel, 'GROUP', [object()])
    monkeypatch.setattr(parallel, '_OWNED', [None])
    monkeypatch.setattr(parallel, 'size', fake_size)
    monkeypatch.setattr(parallel, 'rank', fake_rank)
    monkeypatch.setattr(parallel, '_collective', fake_collective)

    models = [build_model() for _ in range(n_ranks)]
    for r in range(n_ranks):
        tls.rank = r
        parallel.split(models[r])

    outputs = [None] * n_ranks
    errors = []

    def worker(r):
        tls.rank = r
        mlx_lock.acquire()
        try:
            outputs[r] = run_pass(models[r], r)
        except BaseException as exc:  # noqa: BLE001
            import traceback
            errors.append((r, repr(exc), traceback.format_exc()))
            barrier.abort()
        finally:
            mlx_lock.release()

    mx.disable_compile()
    try:
        threads = [threading.Thread(target=worker, args=(r,)) for r in range(n_ranks)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)
    finally:
        mx.enable_compile()
    assert not errors, '\n'.join(f'rank {r}: {err}\n{tb}' for r, err, tb in errors)
    return outputs


@pytest.mark.parametrize('n_ranks', [2, 4])
def test_parallel_split_glm5_next_matches_single_gpu(n_ranks, monkeypatch):
    from mlx_vlm.models.glm5_next import Model as GlmModel, ModelConfig as GlmConfig, TextConfig as GlmTextConfig

    text_cfg = GlmTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=8,
        num_nextn_predict_layers=0,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        kv_lora_rank=16,
        q_lora_rank=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        num_experts_per_tok=2,
        index_topk=4,
        index_head_dim=16,
        index_n_heads=2,
        index_kpool=2,
        linear_head_dim=16,
        linear_num_heads=2,
        linear_conv_kernel_dim=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        first_k_dense_replace=1,
    )
    cfg = GlmConfig(text_config=text_cfg)

    def make_weights():
        rng = np.random.default_rng(77)
        base = GlmModel(cfg)
        updates = [
            (k, mx.array(rng.normal(0, 0.05, size=v.shape).astype(np.float32)).astype(v.dtype))
            for k, v in tree_flatten(base.parameters())
        ]
        mx.eval([v for _, v in updates])
        return updates

    shared_weights = make_weights()

    def build():
        m = GlmModel(cfg)
        m.load_weights(shared_weights, strict=False)
        mx.eval(m.parameters())
        return m

    prompt = mx.array([[1, 5, 9, 3, 7, 2]], dtype=mx.int32)
    step1 = mx.array([[4]], dtype=mx.int32)
    step2 = mx.array([[8]], dtype=mx.int32)

    def run_all(m, _r=0):
        c = m.language_model.make_cache()
        out0 = m.language_model(prompt, cache=c).logits
        out1 = m.language_model(step1, cache=c).logits
        out2 = m.language_model(step2, cache=c).logits
        mx.eval(out0, out1, out2)
        return [np.array(out0), np.array(out1), np.array(out2)]

    ref = run_all(build())
    rank_outs = _run_multi_rank(n_ranks, build, run_all, monkeypatch)
    for r in range(n_ranks):
        for s_idx in range(3):
            np.testing.assert_allclose(rank_outs[r][s_idx], ref[s_idx], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('n_ranks', [2, 4])
def test_parallel_split_deepseek_v41_matches_single_gpu(n_ranks):
    seed = 123
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        m_single, cfg = _v41_build_model(seed)
        ref_single = _v41_run_single(m_single)
        m_batched, _ = _v41_build_model(seed)
        ref_batched = _v41_run_batched(m_batched, cfg.num_hidden_layers)
    finally:
        mx.set_default_device(prev)

    ctx = mp.get_context('spawn')
    conns = []
    procs = []
    try:
        for r in range(n_ranks):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_v41_rank_worker, args=(r, n_ranks, seed, child_conn))
            p.start()
            child_conn.close()
            conns.append(parent_conn)
            procs.append(p)

        rank_single = [None] * n_ranks
        rank_batched = [None] * n_ranks
        while True:
            msgs = []
            for r, c in enumerate(conns):
                assert c.poll(60.0), f'timed out waiting on rank {r}'
                msgs.append(c.recv())
            tags = {m[0] for m in msgs}
            if 'err' in tags:
                errs = '\n'.join(f'rank {r}:\n{m[1]}' for r, m in enumerate(msgs) if m[0] == 'err')
                raise AssertionError(f'worker error:\n{errs}')
            if tags == {'done'}:
                for r, m in enumerate(msgs):
                    rank_single[r] = m[1]
                    rank_batched[r] = m[2]
                break
            assert tags == {'coll'}, f'unexpected worker states: {[m[0] for m in msgs]}'
            dtypes = {m[2] for m in msgs}
            shapes = {m[4] for m in msgs}
            assert len(dtypes) == 1, f'rank dtype mismatch across collective: {dtypes}'
            assert len(shapes) == 1, f'rank shape mismatch across collective: {shapes}'
            is_int32 = msgs[0][3]
            acc = sum(m[1].astype(np.int64 if is_int32 else np.float64) for m in msgs)
            summed = acc.astype(np.int32 if is_int32 else np.float32)
            for c in conns:
                c.send(summed)

        for r, p in enumerate(procs):
            p.join(timeout=30.0)
            assert p.exitcode == 0, f'rank {r} exited with {p.exitcode}'

        for r in range(n_ranks):
            for s_idx in range(len(ref_single)):
                np.testing.assert_allclose(rank_single[r][s_idx], ref_single[s_idx], rtol=1e-5, atol=1e-5)
            for s_idx in range(len(ref_batched)):
                np.testing.assert_allclose(rank_batched[r][s_idx], ref_batched[s_idx], rtol=1e-5, atol=1e-5)
    finally:
        for c in conns:
            c.close()
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)


# ---------------------------------------------------------------- 3. SLURM cpuset Load, NUMA cores & batch priority


def test_load_scopes_proc_stat_to_cpuset_and_ignores_other_jobs(tmp_path):
    stat = tmp_path / 'stat'
    psi = tmp_path / 'pressure'
    own = tmp_path / 'self_stat'
    psi.write_text('some avg10=80.00 avg60=50.00 avg300=20.00 total=100\n', encoding='utf-8')
    own.write_text('1 (quanteeze) S 0 0 0 0 0 0 0 0 0 0 0 ' + '0 ' * 40 + '\n', encoding='utf-8')

    # CPUs 0-3 idle; CPUs 4-7 pinned at 100% by other users' jobs on shared node
    stat.write_text(
        'cpu  400 0 0 400 0 0 0 0 0 0\n'
        'cpu0 0 0 0 100 0 0 0 0 0 0\n'
        'cpu1 0 0 0 100 0 0 0 0 0 0\n'
        'cpu2 0 0 0 100 0 0 0 0 0 0\n'
        'cpu3 0 0 0 100 0 0 0 0 0 0\n'
        'cpu4 100 0 0 0 0 0 0 0 0 0\n'
        'cpu5 100 0 0 0 0 0 0 0 0 0\n'
        'cpu6 100 0 0 0 0 0 0 0 0 0\n'
        'cpu7 100 0 0 0 0 0 0 0 0 0\n',
        encoding='utf-8',
    )
    t0, mine0, _ = ce._read_proc(stat=str(stat), own=str(own), psi='/proc/pressure/cpu', cpus={0, 1, 2, 3})
    assert t0 == (0, 400) and mine0 == 0

    stat.write_text(
        'cpu  800 0 0 800 0 0 0 0 0 0\n'
        'cpu0 0 0 0 200 0 0 0 0 0 0\n'
        'cpu1 0 0 0 200 0 0 0 0 0 0\n'
        'cpu2 0 0 0 200 0 0 0 0 0 0\n'
        'cpu3 0 0 0 200 0 0 0 0 0 0\n'
        'cpu4 200 0 0 0 0 0 0 0 0 0\n'
        'cpu5 200 0 0 0 0 0 0 0 0 0\n'
        'cpu6 200 0 0 0 0 0 0 0 0 0\n'
        'cpu7 200 0 0 0 0 0 0 0 0 0\n',
        encoding='utf-8',
    )
    clock = [0.0]

    def read_scoped():
        return ce._read_proc(stat=str(stat), own=str(own), psi=str(psi), cpus={0, 1, 2, 3})

    # If node-wide /proc/stat had been read, `others` would be 50% and shut workers down.
    # Scoped to {0, 1, 2, 3}, `others` is 0% (still use non-`/proc/pressure/cpu` safe path with 0 pressure).
    psi.write_text('some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n', encoding='utf-8')
    load = ce.Load(threads=4, yield_at=25, clock=lambda: clock[0], read=read_scoped, least=2, cpus={0, 1, 2, 3})
    assert load.cores == 4
    load.sample()
    clock[0] += ce.CALM_S + 0.5
    workers = load.sample()
    assert load.others == pytest.approx(0.0)
    assert workers >= 2


def test_numa_cores_cap_and_batch_priority(tmp_path, monkeypatch):
    # Dual-socket 128-core EPYC sysfs (0..63 socket 0, 64..127 socket 1)
    cpu_root = tmp_path / 'cpu'
    for c in range(128):
        topo = cpu_root / f'cpu{c}' / 'topology'
        topo.mkdir(parents=True)
        (topo / 'core_id').write_text(str(c % 64), encoding='utf-8')
        (topo / 'physical_package_id').write_text('0' if c < 64 else '1', encoding='utf-8')

    pci_root = tmp_path / 'pci'
    dev = pci_root / '0000:c1:00.0'
    dev.mkdir(parents=True)
    (dev / 'local_cpulist').write_text('64-127\n', encoding='utf-8')
    (dev / 'current_link_speed').write_text('16.0 GT/s PCIe\n', encoding='utf-8')
    (dev / 'current_link_width').write_text('16\n', encoding='utf-8')

    monkeypatch.setattr(ce.os, 'sched_getaffinity', lambda _pid: set(range(128)), raising=False)
    local = ce.gpu_local_cpus('00000000:c1:00.0', pci_root=pci_root)
    assert local == set(range(64, 128))
    selected = ce.cores(64, root=cpu_root, siblings=False, local_cpus=local)
    assert len(selected) == 64 - ce.RESERVE_CORES
    assert all(64 <= c < 128 for c in selected)
    assert selected[:4] == [127, 126, 125, 124]

    # Without explicit local_cpus on dual-socket, single package still avoids cross-socket span
    pkg_only = ce.cores(ce.MAX_AUTO_THREADS, root=cpu_root, siblings=False, local_cpus=set())
    assert len(pkg_only) == ce.MAX_AUTO_THREADS
    assert all(c < 64 for c in pkg_only)

    # default_threads caps at MAX_AUTO_THREADS when QUANTEEZE_CPU_THREADS=auto
    monkeypatch.delenv('QUANTEEZE_CPU_THREADS', raising=False)
    monkeypatch.setattr(ce, 'cores', lambda n, **_kw: list(range(min(120, n))))
    assert ce.default_threads() == ce.MAX_AUTO_THREADS

    # Batch scheduler priority switches auto -> normal inside SLURM/PBS/PJM
    for k in ce.BATCH_JOB_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv('QUANTEEZE_CPU_PRIORITY', raising=False)
    assert ce.default_priority() == 'idle'
    monkeypatch.setenv('SLURM_JOB_ID', '987654')
    assert ce.default_priority() == 'normal'
    monkeypatch.setenv('QUANTEEZE_CPU_PRIORITY', 'wake')
    assert ce.default_priority() == 'wake'


# ---------------------------------------------------------------- 4. Lustre read_threads, HostCopy, PCIe & integrated CUDA


def test_lustre_read_threads_and_full_ram_by_rows(tmp_path, monkeypatch):
    mounts = tmp_path / 'mounts'
    lustre_dir = tmp_path / 'scratch' / 'ckpt'
    local_dir = tmp_path / 'nvme' / 'ckpt'
    lustre_dir.mkdir(parents=True)
    local_dir.mkdir(parents=True)
    mounts.write_text(
        f'/dev/nvme0n1 {local_dir} ext4 rw 0 0\n'
        f'10.0.0.1@o2ib:/fs {lustre_dir} lustre rw 0 0\n',
        encoding='utf-8',
    )
    monkeypatch.delenv('QUANTEEZE_READ_THREADS', raising=False)
    assert q2_experts.read_threads(lustre_dir, mounts=str(mounts)) == q2_experts.NETWORK_READ_THREADS
    assert q2_experts.read_threads(local_dir, mounts=str(mounts)) == q2_experts.LOCAL_READ_THREADS
    monkeypatch.setenv('QUANTEEZE_READ_THREADS', '8')
    assert q2_experts.read_threads(lustre_dir, mounts=str(mounts)) == 8

    monkeypatch.setattr(q2_experts, 'on_network', lambda p: str(p) == str(lustre_dir))
    assert serve._host_by_rows(lustre_dir, budget=100e9, total=200e9) is False
    assert serve._host_by_rows(lustre_dir, budget=300e9, total=200e9) is True
    assert serve._host_by_rows(local_dir, budget=10e9, total=200e9) is True


def test_pcie_and_integrated_cuda_auto_selection(tmp_path, monkeypatch):
    from quanteeze import cuda_kernels

    pci_root = tmp_path / 'pci'
    for bdf, speed, width in (
        ('0000:01:00.0', '16.0 GT/s PCIe', '8'),
        ('0000:02:00.0', '16.0 GT/s PCIe', '16'),
        ('0000:03:00.0', '32.0 GT/s PCIe', '16'),
    ):
        d = pci_root / bdf
        d.mkdir(parents=True)
        (d / 'current_link_speed').write_text(speed, encoding='utf-8')
        (d / 'current_link_width').write_text(width, encoding='utf-8')

    assert ce.pcie_bandwidth_gbs('0000:01:00.0', pci_root=pci_root) == pytest.approx(15.752)
    assert ce.pcie_bandwidth_gbs('0000:02:00.0', pci_root=pci_root) == pytest.approx(31.504)
    assert ce.pcie_bandwidth_gbs('0000:03:00.0', pci_root=pci_root) == pytest.approx(63.008)

    with ExitStack():
        monkeypatch.setattr(mx.metal, 'is_available', lambda: False)
        monkeypatch.setattr(cuda_kernels, 'available', lambda: True)
        monkeypatch.setattr(cuda_ple, 'integrated', lambda: False)
        monkeypatch.setattr(ce, 'isa', lambda: 1)  # AVX2
        monkeypatch.setattr(ce, 'default_threads', lambda: 8)
        assert ce.auto_cpu_experts('0000:01:00.0', pci_root=pci_root) is True
        assert ce.auto_cpu_experts('0000:03:00.0', pci_root=pci_root) is False  # Gen5 x16 + 8 AVX2 cores

        # With >=12 cores (or AVX-512), offload stays worthwhile even on Gen5 x16
        monkeypatch.setattr(ce, 'default_threads', lambda: 16)
        assert ce.auto_cpu_experts('0000:03:00.0', pci_root=pci_root) is True
        monkeypatch.setattr(ce, 'default_threads', lambda: 8)
        monkeypatch.setattr(ce, 'isa', lambda: 2)  # AVX-512
        assert ce.auto_cpu_experts('0000:03:00.0', pci_root=pci_root) is True

        # Integrated CUDA GPU skips HostCopy and CPU offload
        monkeypatch.setattr(cuda_ple, 'integrated', lambda: True)
        assert ce.auto_cpu_experts('0000:01:00.0', pci_root=pci_root) is False
        monkeypatch.delenv('QUANTEEZE_CPU_EXPERTS', raising=False)
        assert ce.cpu_setting() == 'off'
        monkeypatch.delenv('QUANTEEZE_HOST_EXPERTS_GB', raising=False)
        monkeypatch.delenv('QUANTEEZE_HOST_NGRAM_GB', raising=False)
        monkeypatch.setattr(mx, 'default_device', lambda: mx.gpu)
        assert serve._hold_experts(str(tmp_path)) is None
        assert serve._hold_tables(nn.Module(), str(tmp_path)) is None
