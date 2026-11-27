# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""Every QUANTEEZE_* environment variable, its default and what it does, in one table.

    python -m quanteeze.variables              the table, with what this environment sets
    python -m quanteeze.variables --markdown   the same as a Markdown table, for the docs

The package reads them through get() and the typed helpers (number, integer, off, path), so a
default is written here once; a name not in SETTINGS is refused, and the server warns at start
of any QUANTEEZE_* the environment sets that is not one.  An empty value is unset.

When a setting takes effect (WHEN), as the server reads it: 'start', once as it starts; 'load',
as it loads a model, so a change takes effect at the next load (a model loaded from the control
page, or after /unload the next request's); 'request', read anew each request.  quanteeze.control
manages the server's settings from a browser.
"""
import os
import sys
from pathlib import Path

WHEN = ('request', 'load', 'start')


class Setting:
    def __init__(self, name, default, group, when, doc):
        if when not in WHEN:
            raise ValueError(f'{name}: when is one of {WHEN}, not {when!r}')
        self.name, self._default, self.group, self.when, self.doc = name, default, group, when, doc

    @property
    def default(self):
        return self._default() if callable(self._default) else self._default


def _ple_threads():
    from . import ngram_prefetch
    return str(ngram_prefetch.THREADS)


_TABLE = [
    # -- memory and heat (quanteeze.serve, allocator, guards, ngram_prefetch)
    ('QUANTEEZE_ALLOCATOR', 'auto', 'memory', 'start',
     'preload bundled kamepoolalloc (rss-zero-loss; https://github.com/northriv/kamepoolalloc) for lock-free '
     'per-thread host allocation and warm SSD-slice buffer recycling with low RSS (auto: build and preload '
     'when a C++17 compiler is present, else system malloc; on: require kamepoolalloc; off: system malloc)'),
    ('QUANTEEZE_CACHE_GB', 'auto', 'memory', 'start',
     'GB of freed buffers MLX keeps (auto: 2, and on a GPU no more than 3% of its working set '
     '-- on CUDA 1 GB at least: an MTP verify pass turns over ~0.75 GB)'),
    ('QUANTEEZE_WIRED_GB', '0', 'memory', 'load',
     'GB of MLX\'s memory wired from each load on (0 or off: none beyond what mlx-vlm wires while generating; '
     'auto: Metal\'s recommended working set, held during load; no-op without Metal)'),
    ('QUANTEEZE_MEMORY_RESERVE', '0.3', 'memory', 'load',
     'share of Metal\'s working set kept from the experts for caches and passes (quanteeze.fitting)'),
    ('QUANTEEZE_WORKING_SET_SHARE', 'auto', 'memory', 'load',
     'share of the memory MLX computes in taken as the working set where Metal recommends none (the GPU\'s on '
     'CUDA, MemTotal on the CPU); auto: on CUDA all of the GPU\'s memory but what others held at the start '
     '(other programs, a desktop, the driver\'s context), on the CPU 0.75'),
    ('QUANTEEZE_ADMIT_SHARE', '0.9', 'memory', 'start',
     'share of the working set MLX may hold with the prompts a prefill takes in at once (off: mlx-vlm\'s 8)'),
    ('QUANTEEZE_ADMIT_AHEAD', '4096', 'memory', 'start',
     'tokens of cache growth kept free for each row decoding and each taken in (off: none; quanteeze.admission)'),
    ('QUANTEEZE_APC_ENCRYPT', 'on', 'memory', 'start',
     'seal the prompt cache\'s (APC) disk shards and checkpoints with AES-256-GCM under a convergent key derived '
     'from the prompt\'s own tokens and precision settings so neither plaintext token ids nor unencrypted KV '
     'states sit on disk (off: mlx-vlm\'s plaintext safetensors; quanteeze.apc_disk)'),
    ('QUANTEEZE_APC_DISK_WRITE_GB', None, 'memory', 'start',
     'maximum GB the server writes to the prompt cache\'s disk tier while running: past this cap it writes '
     'no more and serves from what is already stored without evicting to rewrite (unset: uncapped writes, '
     'retention bounded only by APC_DISK_MAX_GB)'),
    ('QUANTEEZE_APC_READ_BACK_PLAIN', 'on', 'memory', 'start',
     'serve a single-row unpadded warm prompt read back from the prefix cache (APC) without KV bits or a draft '
     'model on the model\'s plain caches (KVCache / ArraysCache), matching a fresh request (quanteeze.apc_plain): '
     'avoids Qwen3.5 / Qwen3.8-27B per-token batch extraction and copying during decode.  off: keep mlx-vlm\'s '
     'single-row batch caches'),
    ('QUANTEEZE_APC_DISK_SECRET', '~/.quanteeze/apc-secret', 'memory', 'start',
     'the file of this machine\'s secret mixed into the prompt cache\'s disk tier (its sealed checkpoints\' names '
     'and keys, HMAC-SHA-256; quanteeze.apc_disk): 32 random bytes made at first use, readable by the user alone '
     '(0600), and on macOS excluded from Time Machine -- so one with the disk or its backups but not the secret '
     'cannot confirm a guessed prompt from them.  Removed or found readable by others, a new one is generated and '
     'old cache files age out.  off: no secret, names and keys derived from the prompt alone'),
    ('QUANTEEZE_CONTEXT_COMPRESS', 'off', 'memory', 'start',
     'compression of full-attention QSA KV caches kept for later turns in APC (quanteeze.context_compress): '
     'off (keep whole cache), recent (keep sink and recent tail), similar (drop repeated neighboring blocks), '
     'h2o (keep heavy-hitter blocks attended most across indexer picks), snapkv (keep blocks attended by '
     'the trailing QUANTEEZE_CONTEXT_WINDOW queries, pooled locally)'),
    ('QUANTEEZE_CONTEXT_WINDOW', '64', 'memory', 'start',
     'snapkv: trailing observation window of query positions whose indexer picks score blocks for retention'),
    ('QUANTEEZE_CONTEXT_LAYERS', 'uniform', 'memory', 'start',
     'how the kept fraction is spread across full-attention layers: uniform (equal per layer) or pyramid '
     '(higher retention in lower layers, lower in upper layers)'),
    ('QUANTEEZE_CONTEXT_KEEP', '0.5', 'memory', 'start',
     'fraction of blocks between protected sink and recent tail retained when compressing QSA KV caches'),
    ('QUANTEEZE_CONTEXT_RECENT', '1024', 'memory', 'start',
     'tokens at the end of a cached prefix that are never dropped by context compression'),
    ('QUANTEEZE_CONTEXT_SINK', '64', 'memory', 'start',
     'initial attention-sink tokens at the start of a cached prefix that are never dropped'),
    ('QUANTEEZE_CONTEXT_MIN_TOKENS', '4096', 'memory', 'start',
     'minimum cached token length before context compression triggers (shorter caches remain intact)'),
    ('QUANTEEZE_KV_BITS', '', 'memory', 'load',
     'bit width of full-attention cached keys and values (quanteeze.kv_bits): unset or off keeps bfloat16; 8 uses MLX '
     'affine quantization with group size 64 on Qwen3.8 Flash-Next (1,088 vs 2,048 bytes per layer and token, 0.53x, '
     'leaving indexer keys and linear states unchanged) and on plain KVCache models such as Qwen3.8-27B '
     '(quanteeze.kv_bits.install_plain: mlx-vlm\'s 8-bit batch cache, 36.7 vs 65.5 kB per token, 0.56x), verified '
     'against dequantized matmul and supporting batching and APC snapshots (QUANTEEZE_CUDA_ATTN declines 8-bit KV '
     'caches)'),
    ('QUANTEEZE_KV_BITS_SELECTED', 'on', 'memory', 'load',
     'with QUANTEEZE_KV_BITS=8, compute 1-to-2-token decode/verify sparse attention past the indexer budget over '
     'only the indexer-selected blocks and tail keys (gathered, dequantized, and multiplied via MLX matmul, verified '
     'against masked full-key attention on first calls per shape; off: masked attention over every cached key)'),
    ('QUANTEEZE_KV_BITS_UNPACKED', 'auto', 'memory', 'load',
     'with QUANTEEZE_KV_BITS=8, unpack cached 8-bit keys and values to the query dtype per layer on prefill steps '
     'for MLX\'s fast scaled dot-product attention and mlx-vlm\'s Metal QSA sparse prefill kernel while running '
     '1-to-2-token decode/verify through mlx-vlm\'s quantized attention in one call (auto: enabled on Metal, off on '
     'CUDA; on or off overrides)'),
    ('QUANTEEZE_KEEP_GB', 'auto', 'memory', 'start',
     'GB the server\'s footprint keeps to (quanteeze.guards.Keeper): over it MLX\'s freed buffers are given back, '
     'and over it with the swap grown or the swap\'s disk short the server stops (auto: the working set Metal '
     'recommends; elsewhere the host\'s memory less a quarter up to 16 GB, or a tenth where that is more; off: no keeper)'),
    ('QUANTEEZE_NUMPY_HUGE_PAGES', 'off', 'memory', 'start',
     'on Linux, whether numpy asks for transparent huge pages for large arrays (off: disabled so pinned memory '
     'does not trigger kernel compaction stalls)'),
    ('QUANTEEZE_KEEP_FREE_GB', '8', 'memory', 'start',
     'GB of the machine\'s memory the server\'s footprint stops short of: there it stops itself'),
    ('QUANTEEZE_KEEP_SWAP_GB', '8', 'memory', 'start',
     'GB the swap may grow since the start while the server holds more than QUANTEEZE_KEEP_GB before it stops'),
    ('QUANTEEZE_KEEP_DISK_GB', '20', 'memory', 'start',
     'GB free on the disk the swap is written to (macOS) under which a server over QUANTEEZE_KEEP_GB stops'),
    ('QUANTEEZE_KEEP_PRESSURE', 'on', 'memory', 'start',
     'on: monitor host memory pressure alongside footprint -- reclaim freed MLX buffers and report top host '
     'consumers on warning, serialize intake at critical pressure; off: footprint limits only'),
    ('QUANTEEZE_KEEP_STRAIN', 'on', 'memory', 'start',
     'on (Linux): monitor host PSI memory/IO pressure, swap-out rate, and low MemAvailable below '
     'QUANTEEZE_KEEP_CUT_GB, triggering relief callbacks and summarizing strain episodes; off: disabled'),
    ('QUANTEEZE_KEEP_CUT_GB', '4', 'memory', 'start',
     'GB of MemAvailable below which Linux host strain is flagged and the target recovery floor for relief'),
    ('QUANTEEZE_SHUTDOWN_S', '60', 'memory', 'start',
     'seconds after SIGTERM or SIGINT before a server still generating exits'),
    ('QUANTEEZE_PLE_THREADS', _ple_threads, 'memory', 'start',
     'threads reading the n-gram embedding\'s pages ahead (0: off)'),
    ('QUANTEEZE_NGRAM_ADVISE', 'on', 'memory', 'start',
     'small n-gram and Engram row lookups advise the kernel via madvise(MADV_WILLNEED) on the calling thread '
     'rather than going through the thread pool (off: thread pool)'),
    ('QUANTEEZE_GPU_FREE_GB', '0.5', 'memory', 'load',
     'on a GPU with memory of its own (CUDA), GB kept free there by trimming MLX and the CUDA driver memory pool '
     'when unused reserved pool bytes reach 1 GB (off: disabled)'),
    ('QUANTEEZE_COMPACT', 'off', 'memory', 'load',
     'on: as much memory left to the experts as can be -- QUANTEEZE_MTP=off, QUANTEEZE_V41_READS_IMAGES=ssd, '
     'QUANTEEZE_VISION_SSD=on, QUANTEEZE_EMBED_SSD=on, and QUANTEEZE_NGRAM_SSD=on unless listed in '
     'QUANTEEZE_COMPACT_KEEP.  Images remain supported via on-demand SSD streaming while freeing ~1-1.5 GB '
     'of vision tower and embedding weights for experts'),
    ('QUANTEEZE_COMPACT_KEEP', None, 'memory', 'load',
     'comma-separated setting names or suffixes (for example MTP,EMBED_SSD) kept at their own values when '
     'QUANTEEZE_COMPACT=on'),
    ('QUANTEEZE_VISION_SSD', 'auto', 'memory', 'load',
     'stream vision tower weights from the checkpoint via mmap a block at a time when encoding images, '
     'freeing ~1 GB when idle: auto (stream when holding it keeps routed experts from fitting whole, else hold, '
     'and always hold on network file systems), on, or off (on under QUANTEEZE_COMPACT=on)'),
    ('QUANTEEZE_EMBED_SSD', 'off', 'memory', 'load',
     'on: read untied token embeddings a row at a time from the checkpoint via mmap, freeing ~0.4 GB of '
     'resident memory (off by default; on under QUANTEEZE_COMPACT=on)'),
    ('QUANTEEZE_NGRAM_SSD', 'auto', 'memory', 'load',
     'a checkpoint\'s n-gram (PLE) table read from the SSD through a view of it (quanteeze.ple_store; made at '
     'the load where none is: hard links, seconds) rather than held in memory: auto, where held it would leave the '
     'routed experts less room than they take (they would not fit whole beside it); on: always; off: never, a view '
     'named loaded as its checkpoint'),
    ('QUANTEEZE_HOST_NGRAM_GB', 'auto', 'memory', 'load',
     'GB of the n-gram tables (Flash-Next\'s PLE, DeepSeek V4.1\'s Engram) read once into the host\'s memory '
     'and looked up there, not in the file (auto: on a discrete CUDA GPU with memory of its own, as many as the '
     'host\'s memory or the job\'s cgroup leaves, before the experts\' copy; none on a Mac, an integrated CUDA GPU '
     'sharing host memory, or where the checkpoint is in memory already, /dev/shm; off: none)'),
    ('QUANTEEZE_THROTTLE_AT', '2', 'memory', 'load', 'macOS thermal pressure level the throttle starts at (off: none)'),
    ('QUANTEEZE_THROTTLE_DUTY', None, 'memory', 'load',
     'the GPU\'s share of the time while hot: one share, or <level>:<share>,... (guards.DUTIES)'),
    ('QUANTEEZE_THROTTLE_MAX_REST', '2', 'memory', 'load', 'seconds a rest at most'),
    ('QUANTEEZE_CPU_THREADS', 'auto', 'memory', 'start',
     'worker threads that compute experts on the CPU from the host\'s copy (quanteeze.cpu_experts; auto: physical '
     'cores on the GPU\'s NUMA node/socket less RESERVE_CORES, capped at MAX_AUTO_THREADS, each pinned to its core)'),
    ('QUANTEEZE_CPU_PRIORITY', 'auto', 'memory', 'start',
     'CPU expert worker scheduling priority: auto (normal inside SLURM/PBS/LSB/PJM/SGE batch jobs, idle elsewhere), '
     'idle (SCHED_IDLE on Linux, yielding to foreground tasks), nice (nice 19), normal (process default), or wake '
     '(sleep at normal priority for fast wakeup, switch to SCHED_IDLE while computing)'),
    ('QUANTEEZE_CPU_MIN_THREADS', '4', 'memory', 'start',
     'fewest workers CPU expert offload runs with: fewer lose to copying the expert over PCIe, so the GPU path '
     'is taken instead'),
    ('QUANTEEZE_CPU_YIELD_AT', '25', 'memory', 'start',
     'percent of CPU used by other processes within this process\'s cpuset at which CPU expert workers stop at once '
     'under QUANTEEZE_CPU_EXPERTS=auto (returning one worker at a time after 2.5 s under half of it)'),
    ('QUANTEEZE_CPU_Q8', 'on', 'memory', 'start',
     'quantize CPU expert inputs into 32-value int8 blocks and float scales for AVX2 8-bit integer dot products '
     'when weights are 4-bit affine or MXFP4 (quanteeze.cpu_experts; off: float32 inputs)'),
    # -- uniform expert quantization & on-demand SSD streaming (quanteeze.serve, requant, demand)
    ('QUANTEEZE_BITS', None, 'experts', 'load',
     'serve every routed expert uniformly at this precision (8, 4, 3, 2, 1, mxfp4, q3_k, q2_k, or iq2_xs), '
     'computed from the checkpoint in memory when the lowered model fits, or as each expert is read from the SSD '
     'when it does not.  Unset or off: keep the checkpoint\'s original precision'),
    ('QUANTEEZE_EXPERTS_GGUF', None, 'experts', 'load',
     'read routed MoE expert weights directly from this pre-distributed GGUF path (single file, split shards, '
     'or directory), keeping the model trunk, Vision, and MTP from the MLX checkpoint (--experts-gguf). '
     'Works whole in memory or streamed from SSD / HostCopy with QUANTEEZE_FROM_SSD_GB and '
     'QUANTEEZE_HOST_EXPERTS_GB'),
    ('QUANTEEZE_FROM_SSD_GB', None, 'experts', 'load',
     'stream routed experts from the SSD when routed to and waited for, with this many GB of them kept in '
     'memory (auto: what fits).  Unset: whole in memory where the experts fit, else on demand within the room'),
    ('QUANTEEZE_LOAD_WHOLE', None, 'experts', 'load',
     'on: every model loaded whole into memory, even where its experts do not fit'),
    ('QUANTEEZE_READ_THREADS', 'auto', 'experts', 'load',
     'threads reading experts from the checkpoint on demand (auto: 4 on local storage, 16 on Lustre/GPFS/NFS and '
     'other network file systems)'),
    ('QUANTEEZE_DEMAND_READ_NATIVE', 'auto', 'experts', 'load',
     'zero-GIL lock-free C++ pread reader pool for on-demand SSD streaming (quanteeze.native_reads): auto '
     '(compiled and bit-checked against Python readers; off: Python thread pool)'),
    ('QUANTEEZE_HOST_EXPERTS_BY', 'auto', 'experts', 'load',
     'how the host\'s copy of the experts (QUANTEEZE_HOST_EXPERTS_GB) is chosen: rows, an equal share of every '
     'layer\'s experts; layers, whole tensors from the first layer on; auto, rows (but layers on a network file '
     'system when host memory does not hold all routed experts)'),
    ('QUANTEEZE_WARM_UP', 'on', 'experts', 'load',
     'on CUDA, the model run once before the first request, so that quanteeze\'s own kernels are compiled then, '
     'not while it waits (not where experts are read on demand)'),
    ('QUANTEEZE_WARM_UP_BATCH', '0', 'experts', 'load',
     'with QUANTEEZE_WARM_UP on CUDA: the most requests of a batch the warm-up runs besides (2 to this many rows: '
     'a short prompt of each together, then two decode steps), so batch-shape kernels are compiled before the first '
     'request arrives (QUANTEEZE_REQUEST_THREADS=1 batch mode).  0 (default): none'),
    ('QUANTEEZE_HOST_EXPERTS_GB', 'auto', 'experts', 'load',
     'GB of the checkpoint\'s routed experts held in the host\'s memory and read from there, not the disk '
     '(auto: on a discrete CUDA GPU with memory of its own, as many as the host\'s memory or the job\'s cgroup '
     'leaves, less a tenth of it and 3 GB for what the server adds after; none on a Mac, an integrated CUDA GPU '
     'sharing host memory, or where the checkpoint is in memory already, /dev/shm; off: none)'),
    ('QUANTEEZE_CPU_EXPERTS', 'auto', 'experts', 'load',
     'when serving on demand with a host-RAM expert copy (QUANTEEZE_HOST_EXPERTS_GB), compute a decode step\'s '
     '(B=1..8) missing experts held in host memory on the CPU instead of copying them over PCIe '
     '(quanteeze.cpu_experts: workers pinned to NUMA-local physical cores, a C++ server thread answering the GPU '
     'in parallel with GPU-resident experts): auto (default; enabled on discrete CUDA GPUs with AVX2/AVX-512 when '
     'CPU threads and PCIe balance favor offload, yielding to other processes within the cpuset via '
     'QUANTEEZE_CPU_YIELD_AT / QUANTEEZE_CPU_MIN_THREADS; off on Metal, integrated CUDA, non-SIMD hosts, or wide '
     'PCIe Gen5 with small desktop CPUs); on, always attempt; off, stage over PCIe onto the GPU bit for bit'),
    ('QUANTEEZE_DEMAND_PROMPT_GB', '2', 'experts', 'load',
     'GB of experts a prefill pass on demand reads into temporary GPU memory at once, batching larger selections '
     'bit for bit (off: unbounded)'),
    ('QUANTEEZE_PREFILL_LAYERS', '16384', 'experts', 'load',
     'on demand, tokens of a one-row prompt taken a layer at a time (off: chunk by chunk through every layer)'),
    ('QUANTEEZE_PROMPT_MOE_TOKENS', '2048', 'experts', 'load',
     'on CUDA layer-by-layer prefill, join 512-token chunks up to this many tokens per MoE pass once verified '
     'bit for bit (off or 0: disabled)'),
    ('QUANTEEZE_QUANT_CACHE', None, 'experts', 'load',
     'a directory where lowered least-squares expert weights are cached once computed (quanteeze.requant.QuantCache; '
     'unset or off: computed at every start, nothing kept)'),
    ('QUANTEEZE_DEMAND_STM', 'auto', 'experts', 'load',
     'C++20 KAME STM placement, pass-lease, and lander engine for on-demand expert serving (auto: serve in C++ '
     'wherever supported on macOS and CUDA, falling back quietly to Python otherwise; on: same, warning if C++ '
     'cannot serve; shadow: mirror and verify placements after passes; off: Python only)'),
    # -- over several GPUs (quanteeze.parallel)
    ('QUANTEEZE_PARALLEL', 'auto', 'parallel', 'start',
     'one model over several GPUs, a process each: layers, each GPU a run of its decoder layers; off, one GPU '
     '(auto: layers where the job sees two GPUs or more and the --model checkpoint\'s weights do not fit one\'s room '
     'but fit theirs)'),
    ('QUANTEEZE_GPUS', 'auto', 'parallel', 'start', 'how many of the GPUs the job sees it is split over (auto: all)'),
    ('QUANTEEZE_PARALLEL_LOGS', '~/.quanteeze/parallel', 'parallel', 'start',
     'where the ranks but the first write their output (rank<r>.log)'),
    ('QUANTEEZE_PARALLEL_PORT', None, 'parallel', 'start', 'set by the launch: the port rank 0 hands on its requests at'),
    ('QUANTEEZE_PARALLEL_KEY', None, 'parallel', 'start', 'set by the launch: the key the ranks connect to rank 0 with'),
    ('QUANTEEZE_PARALLEL_CROSSING', 'refuse', 'parallel', 'start',
     'over several GPUs, how a rank handles requests touching another rank\'s layers (parallel.keep / '
     'parallel.require): refuse logs once, counts in crossings(), and skips optional work (required reads '
     'still raise Crossing); raise fails immediately with Crossing'),
    # -- kernels & speculative decoding (quanteeze.mtp, dspark, verify_qmv, expert_qmv, hyper, linear_attn)
    ('QUANTEEZE_MTP', 'auto', 'speed', 'load',
     'MTP / DSpark / DFlash2 speculative decoding.  auto: where the checkpoint bundles a head '
     'and its model type\'s verify pass is checked (GLM-5.3, Flash-Next, DeepSeek V4.1); dflash2: external '
     'DFlash2 block-diffusion drafter (e.g. incoai/Qwen3.8-Flash-Next-DFlash2); on: wherever a bundled head '
     'loads; off: never'),
    ('QUANTEEZE_DRAFTER', None, 'speed', 'load',
     'external speculative drafter repo/path or \'dflash2\' (overrides bundled MTP head when set; falls back to '
     'the bundled MTP head per QUANTEEZE_MTP where none is known for the model type, it is not local, or it fails '
     'to load; dense standalone drafters quantize to max(4, QUANTEEZE_MTP_BITS) or stay BF16 when QUANTEEZE_MTP_BITS=off)'),
    ('QUANTEEZE_MTP_BITS', 'auto', 'speed', 'load',
     'the MTP head\'s routed experts\' bits: a number; off, the checkpoint\'s own; auto (the default), by model '
     'family (quanteeze.mtp.AUTO_BITS: 4 for Qwen3.8 Flash-Next, 2 for the others)'),
    ('QUANTEEZE_MTP_EXACT', 'bits', 'speed', 'load',
     'how exactly an MTP verify pass computes what one-token decodes would (bits: bit for bit; distribution: '
     'MLX\'s multi-token pass)'),
    ('QUANTEEZE_MTP_BLOCK', 'auto', 'speed', 'load',
     'the tokens an MTP/DFlash2 round verifies, the bonus and its drafts (auto: 2 for GLM-5.3, adaptive up '
     'to 4 for Flash-Next MTP or 5 for DFlash2, 6 for DSpark; off: mlx-vlm\'s choice)'),
    ('QUANTEEZE_DSPARK_GATE', 'auto', 'speed', 'load',
     'DeepSeek V4.1\'s DSpark drafts kept while the product of its confidence head\'s chances stays at or above '
     'this, the rest not verified (auto: 0.6 with QUANTEEZE_MTP_EXACT=bits, 0.45 otherwise; off: every draft)'),
    ('QUANTEEZE_DSPARK_HEAD_BITS', '4', 'speed', 'load',
     'bits of the DSpark drafter\'s own copy of V4.1\'s head, which it reads whole each round (off: the target\'s '
     'BF16 head, 1.3 GB)'),
    ('QUANTEEZE_DSPARK_LOG', None, 'speed', 'load',
     'a file each DSpark round is added to as a JSON line: its drafts\' confidences and chances of being kept'),
    ('QUANTEEZE_MTP_TIMING', 'off', 'speed', 'load',
     'wall, sync or profile: each stage of an MTP round timed, ms a round said at exit'),
    ('QUANTEEZE_VERIFY_QMV', 'on', 'speed', 'start',
     'a verify pass\'s quantized projections read their weights once, as checked bit for bit against the decode '
     '(off: mlx-vlm\'s paths; a number: the tokens a threadgroup computes at most, 4 by default)'),
    ('QUANTEEZE_EXPERT_QMV', 'on', 'speed', 'start',
     'routed MoE expert decode and verify steps with each expert read once for the tokens on it, checked bit for '
     'bit against stock SwitchGLU (off: stock)'),
    ('QUANTEEZE_EXPERT_QMV_VERIFY_FUSED', 'on', 'speed', 'start',
     'with QUANTEEZE_EXPERT_QMV on: run verify passes of uniform MoE layers through the decode\'s fused two-kernel '
     'expert path (each verify token\'s slots as decode rows), checked bit for bit on first shapes (off: per-slot '
     'multi-token path)'),
    ('QUANTEEZE_KQUANT_QMV', 'on', 'speed', 'start',
     'run pre-distributed GGUF expert projections on Metal and CUDA (plus uniform Q2_K/Q3_K/IQ2_XS on Metal) '
     'through fused gather-qmv and tiled gather-mm kernels once verified against unpacking (off: unpack + gather_mm)'),
    ('QUANTEEZE_V41_HEAD_KERNEL', 'on', 'speed', 'load',
     'on MLX\'s CUDA backend, run DeepSeek V4.1\'s BF16 LM head in a single warp-per-row kernel without float32 '
     'intermediate buffers once verified (off: chunked float32 matmul)'),
    ('QUANTEEZE_HOST_NGRAM', 'on', 'speed', 'start',
     'compute 1-token decode PLE n-gram row ids on the host from the already-evaluated token id rather than '
     'syncing from the GPU (off: GPU)'),
    ('QUANTEEZE_ENGRAM_READ_AHEAD', 'on', 'speed', 'request',
     'prefetch later DeepSeek V4.1 Engram layers asynchronously while an earlier Engram layer runs (off: disabled)'),
    ('QUANTEEZE_ENGRAM_IDS_ONCE', 'on', 'speed', 'request',
     'read DeepSeek V4.1 1-token decode Engram hash ids to the host once per step across all Engram layers '
     '(off: once per Engram layer)'),
    ('QUANTEEZE_HC_KERNELS', 'on', 'speed', 'start',
     'Flash-Next\'s hyper-connections in a decode token in three kernels where mlx-vlm runs ten; GLM-5.3\'s in two '
     'where mlx-vlm runs nine; each shape checked bit for bit first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_LINEAR_ATTN_KERNELS', 'on', 'speed', 'start',
     'Flash-Next\'s and GLM-5.3\'s linear attention in a decode token with fused conv/silu/norm and gated RMSNorm '
     'kernels, each shape checked bit for bit first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_SIGMOID_ROUTER_KERNELS', 'on', 'speed', 'start',
     'GLM-5.3\'s routing of a decode token in two Metal kernels where mlx-vlm runs ten, checked bit for bit first '
     '(off: mlx-vlm\'s)'),
    ('QUANTEEZE_INDEXER_KERNEL', 'on', 'speed', 'start',
     'GLM-5.3\'s sparse-attention indexer at a decode token choosing its positions in one kernel where mlx-vlm '
     'runs some forty, checked bit for bit first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_CUDA_QMV', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a decode\'s one-row quantized projections through MLX\'s qmv with each row\'s loads '
     'issued ahead, checked bit for bit first (off: mx.quantized_matmul as it is)'),
    ('QUANTEEZE_CUDA_QMV_ROWS', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, 2-to-7-row affine 4-bit and 8-bit projections through multi-row qmv_rows kernels, '
     'checked bit for bit first (off: MLX\'s ops)'),
    ('QUANTEEZE_CUDA_DELTA', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a decode step\'s gated delta rule in one kernel where mlx-vlm\'s compiled ops run '
     'seven, checked bit for bit first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_CUDA_FUSE', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a decode token\'s layer in fewer kernels: each hyper-connection\'s norm and projections '
     'in one (the attention\'s output injected first), a MoE block in four with its output injected in the last, the '
     'linear attention\'s conv sliding the cache\'s window and its delta rule computing its gates, the PLE layer in '
     'two, each shape checked bit for bit first (off: the kernels of each part alone)'),
    ('QUANTEEZE_CUDA_PLE', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, Flash-Next\'s n-gram table held in the host\'s memory pinned for the GPU, which reads a '
     'decode step\'s rows there itself: the host builds the next step without waiting for the token, each shape '
     'checked bit for bit first (off: the rows read on the host); DeepSeek V4.1\'s Engram tables and token embedding '
     'the same way (quanteeze.cuda_engram)'),
    ('QUANTEEZE_CUDA_HEAD', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a greedy decode token\'s head (lm_head\'s rows and their argmax) in two kernels where '
     'MLX runs gather_qmv and argmax, the same token, each shape checked bit for bit first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_CUDA_ATTN', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a decode token\'s full attention while its prefix fits the indexer\'s budget in five '
     'kernels and MLX\'s cache update and matmul where mlx-vlm runs some forty, each softmax size checked bit for bit '
     'first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_CUDA_ATTN_ROWS', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, 2-to-7-token causal MTP verify full attention through fused multi-row kernels, '
     'checked bit for bit first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_CUDA_EXPERTS', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a decode token\'s 4-bit or 2-bit affine or MXFP4 routed experts in two kernels '
     '(gate+up+swiglu; down) where mlx-vlm runs four, checked bit for bit first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_METAL_ROUTER', 'on', 'speed', 'start',
     'on Metal, 1-token decode and 2-to-8-token verify-pass routing (softmax, top k, and renormalized weights) in '
     'one kernel where MLX runs seven ops, and routed outputs weighted, summed, and combined with the shared expert '
     'in one kernel where MLX runs three (quanteeze.metal_router), each shape checked bit for bit first '
     '(off: MLX\'s ops)'),
    ('QUANTEEZE_CUDA_ROUTER', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a decode token\'s routing in one kernel where MLX runs six, checked bit for bit '
     'first (off: MLX\'s ops)'),
    ('QUANTEEZE_COMPILE_DECODE', 'on', 'speed', 'start',
     'Flash-Next\'s decode token with its routers, shared experts, weighting and hyper-connections as compiled '
     'graphs (mx.compile) (off: op by op)'),
    ('QUANTEEZE_EVAL_LAYERS', 'auto', 'speed', 'start',
     'Flash-Next\'s decode token handed to the GPU this many layers at a time, not one by one (mlx-vlm\'s): '
     'auto uses 4 on Apple Silicon (Metal) and 12 on CUDA to preserve CUDA graph batching (positive int overrides; '
     '1: mlx-vlm\'s)'),
    ('QUANTEEZE_VERIFY_LEAD_LAYERS', 'auto', 'speed', 'start',
     'layers the host may build ahead of the GPU in a Flash-Next MTP verify pass: before building a layer, wait '
     'for the input to the layer this many steps behind so freed buffers do not pile up in the driver pool and '
     'evict managed-memory experts on memory-tight GPUs (auto: 4 on a single CUDA GPU, 0 on Apple Silicon or '
     'multi-GPU; off or 0: no bound)'),
    ('QUANTEEZE_NATIVE', 'on', 'speed', 'load',
     'compile and load C++20 nanobind extensions on demand (off: disabled, pure Python fallbacks)'),
    ('QUANTEEZE_NVRTC_PCH', 'on', 'speed', 'start',
     'on CUDA, enable NVRTC precompiled headers via libnvrtc_pch loaded before MLX imports, injecting --pch into '
     'runtime CUDA kernel compilations to reuse MLX headers across kernels (off: default NVRTC compilation)'),
    ('QUANTEEZE_NGRAM_TOUCH_NATIVE', 'on', 'speed', 'start',
     'read memmapped n-gram and Engram row pages in C++ with the GIL released before lookups (off: numpy indexing)'),
    ('QUANTEEZE_ENGRAM_HASHES_ON_HOST', 'on', 'speed', 'request',
     'compute DeepSeek V4.1 Engram hash ids on the host in C++20 alongside the GPU reference (checked bit for bit '
     'against GPU on first passes; off: wait on GPU hashes)'),
    ('QUANTEEZE_V41_ATTENTION_NATIVE', 'auto', 'speed', 'load',
     'build DeepSeek V4.1 short-pass full attention in C++20 on MLX (auto: Metal and CUDA; on/compare/off)'),
    ('QUANTEEZE_DSPARK_NATIVE', 'auto', 'speed', 'load',
     'run DeepSeek V4.1 DSpark Markov proposal loop and confidence gate in C++20 on MLX (auto: Metal; on/off)'),
    ('QUANTEEZE_CACHE_DIR', '~/.cache/quanteeze', 'speed', 'start',
     'directory where compiled C++20 nanobind extensions are cached (quanteeze.native_ext)'),
    # -- each request (quanteeze.loops, reask, dsml)
    ('QUANTEEZE_LOOPS', 'on', 'requests', 'start', 'stop a generation stuck in a repetition loop (off: disabled)'),
    ('QUANTEEZE_REASK', '2', 'requests', 'start',
     'times a broken DSML tool-call block is written again before the client sees it, at temperature 1.0 '
     '(quanteeze.reask: one that does not parse, names a tool or parameter not there, or repeats a call; off: never)'),
    ('QUANTEEZE_TEMPERATURE', None, 'requests', 'request',
     'the temperature a request that names none is sampled at (by default mlx-vlm\'s)'),
    ('QUANTEEZE_SEED', None, 'requests', 'request',
     'the seed a request that names none is sampled with (by default none)'),
    ('QUANTEEZE_DSV4_ENCODING', None, 'requests', 'load',
     'DeepSeek V4\'s encoding (a directory or the file; by default the copy that comes with quanteeze)'),
    ('QUANTEEZE_DSV41_ENCODING', None, 'requests', 'load',
     'DeepSeek V4.1\'s encoding (a directory or the file; by default the copy that comes with quanteeze)'),
    ('QUANTEEZE_V41_READS_IMAGES', 'on', 'requests', 'load',
     'on: DeepSeek V4.1 reads the images its user turns carry, following QUANTEEZE_VISION_SSD (auto holds the '
     '0.97 GB vision tower and aligner when routed experts fit beside them, else streams on demand except on '
     'network file systems); ssd: always stream vision tower weights on demand; off: let go'),
    ('QUANTEEZE_REQUEST_THREADS', '1', 'requests', 'load',
     'maximum concurrent request worker threads each running its own MLX stream (1: mlx-vlm single generation loop)'),
    ('QUANTEEZE_REQUEST_THREADS_ANY', 'off', 'requests', 'load',
     'allow concurrent request threads when all weights reside in memory without on-demand streaming (on or off)'),
    ('QUANTEEZE_REQUEST_THREADS_DRAFT_TOGETHER', 'off', 'requests', 'load',
     'allow DSpark speculative verify rounds while another request decodes concurrently (on or off)'),
    ('QUANTEEZE_PROMPT_DECODE_TURNS', 'on', 'requests', 'load',
     'when QUANTEEZE_REQUEST_THREADS is above 1 and C++ STM serves on-demand experts, arbitrate GPU turns '
     'between prompt units and decode steps by oldest birth first while letting decodes overlap freely '
     '(off: unarbitrated)'),
    ('QUANTEEZE_PROMPT_CONTENDED_TOKENS', '512', 'requests', 'load',
     'maximum prompt chunk size (always) and contended MoE window size when QUANTEEZE_PROMPT_DECODE_TURNS is active'),
    ('QUANTEEZE_REQUEST_LEASES', 'on', 'requests', 'load',
     'when QUANTEEZE_REQUEST_THREADS exceeds 1 under C++ STM, each concurrent request leases GPU work-room '
     'bytes (non-expert memory on the C++ buffer node) before starting and releases on completion; requests '
     'needing more room wait while the oldest running request grows into remaining room (quanteeze.leases)'),
    # -- control routes & browser settings (quanteeze.control)
    ('QUANTEEZE_CONTROL_TOKEN', None, 'control', 'start',
     'the token the control routes ask for (by default the one ~/.quanteeze/control-token keeps, made '
     'at a first start; delete the file for a new one)'),
    ('QUANTEEZE_SETTINGS', '~/.quanteeze/settings.json', 'control', 'start',
     'the settings saved from the browser, set at start where the environment sets none (off: none)'),
]
SETTINGS = {name: Setting(name, default, group, when, doc) for name, default, group, when, doc in _TABLE}
GROUPS = ('memory', 'experts', 'parallel', 'speed', 'requests', 'control')

RENAMED = {}


def _setting(name):
    try:
        return SETTINGS[name]
    except KeyError:
        raise KeyError(f'{name}: not a setting (quanteeze.variables.SETTINGS)') from None


def value(name, environ=None):
    """What the environment sets `name` to, from an old name of it where it sets none."""
    environ = os.environ if environ is None else environ
    if environ.get(name):
        return environ[name]
    for old, new in RENAMED.items():
        if name in new and environ.get(old):
            return new[name](environ[old])
    return None


def is_set(name, environ=None):
    _setting(name)
    return bool(value(name, environ))


COMPACT = {
    'QUANTEEZE_MTP': 'off',
    'QUANTEEZE_V41_READS_IMAGES': 'ssd',
    'QUANTEEZE_VISION_SSD': 'on',
    'QUANTEEZE_EMBED_SSD': 'on',
    'QUANTEEZE_NGRAM_SSD': 'on',
}


def compact(environ=None):
    """Whether QUANTEEZE_COMPACT is on: COMPACT's settings read as it gives them."""
    on = value('QUANTEEZE_COMPACT', environ)
    return on is not None and on.strip().lower() == 'on'


def compact_keeps(environ=None):
    """Full `QUANTEEZE_*` setting names listed in `QUANTEEZE_COMPACT_KEEP` to preserve when compact is on."""
    raw = value('QUANTEEZE_COMPACT_KEEP', environ)
    if not raw:
        return set()
    kept = set()
    for item in raw.split(','):
        token = item.strip().upper()
        if not token:
            continue
        kept.add(token if token.startswith('QUANTEEZE_') else f'QUANTEEZE_{token}')
    return kept


_AT_LOAD = [None]
_AT_START = [None]


def take_load_snapshot(environ=None):
    """Snapshot os.environ (or `environ`) when a model loads, for `'load'` settings."""
    _AT_LOAD[0] = dict(os.environ if environ is None else environ)
    return _AT_LOAD[0]


def at_load():
    """The snapshot taken at the most recent model load, or None before load."""
    return _AT_LOAD[0]


def take_start_snapshot(environ=None):
    """Snapshot os.environ (or `environ`) at server startup, for `'start'` settings."""
    _AT_START[0] = dict(os.environ if environ is None else environ)
    return _AT_START[0]


def at_start():
    """The snapshot taken at server startup, or None before start."""
    return _AT_START[0]


def get(name, environ=None):
    """The variable's value, or its default (None for none) where it is unset or empty; one of
    COMPACT's as it gives it while QUANTEEZE_COMPACT is on (unless listed in QUANTEEZE_COMPACT_KEEP)."""
    setting = _setting(name)
    compact_env = _AT_LOAD[0] if environ is None else environ
    if name in COMPACT and compact(compact_env) and name not in compact_keeps(compact_env):
        return COMPACT[name]
    set_to = value(name, environ)
    return set_to if set_to else setting.default


def with_new_names(values):
    """{name: value} (as the page saved them) with each old name replaced by what it reads as,
    where `values` sets none of those itself."""
    out = {k: v for k, v in values.items() if k not in RENAMED}
    for old, new in RENAMED.items():
        for name, f in new.items():
            if values.get(old) and not out.get(name) and (v := f(values[old])) is not None:
                out[name] = v
    return out


def renamed(environ=None):
    """The old names the environment sets: [(old, what it reads as: ['NEW=value', ...])]."""
    environ = os.environ if environ is None else environ
    return [(old, [f'{n}={v}' for n, f in sorted(new.items()) if (v := f(environ[old])) is not None])
            for old, new in RENAMED.items() if environ.get(old)]


def number(name, environ=None):
    return float(get(name, environ))


def integer(name, environ=None):
    return int(get(name, environ))


def integer_or(name, default=None, environ=None):
    """The setting as an int, or `default` when unset or not a valid integer."""
    val = get(name, environ)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def off(name, environ=None):
    """Whether the variable says off."""
    val = get(name, environ)
    return val is not None and val.lower() == 'off'


def path(name, environ=None):
    """The value as a path, ~ expanded, or None."""
    val = get(name, environ)
    return None if val is None else Path(val).expanduser()


def unknown(environ=None):
    """The QUANTEEZE_* names the environment sets that are not settings."""
    environ = os.environ if environ is None else environ
    return sorted(k for k in environ if k.startswith('QUANTEEZE_') and k not in SETTINGS and k not in RENAMED)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    markdown = '--markdown' in argv
    if markdown:
        print('| variable | default | read at | what it does |\n|---|---|---|---|')
    for group in GROUPS:
        if not markdown:
            print(f'\n[{group}]')
        for s in (s for s in SETTINGS.values() if s.group == group):
            default = '' if s.default is None else s.default
            if markdown:
                print(f'| `{s.name}` | {f"`{default}`" if default else "—"} | {s.when} | {s.doc} |')
                continue
            now = value(s.name)
            shown = f'= {now}' if now else f'  ({default})' if default else ''
            print(f'  {s.name:36s} {shown}\n      {s.doc} [read at {s.when}]')
    for name in [] if markdown else unknown():
        print(f'\nWARNING {name} is set but is not a setting: a misspelling?')
    for old, new in [] if markdown else renamed():
        print(f'\nWARNING {old} is an old name, read as {" and ".join(new)}')


if __name__ == '__main__':
    main()
