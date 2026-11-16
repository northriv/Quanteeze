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
     'GB of freed buffers MLX keeps (auto: 2, and on a GPU no more than 3% of its working set)'),
    ('QUANTEEZE_WIRED_GB', '0', 'memory', 'start', 'GB of MLX\'s memory wired from start (0: none)'),
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
    ('QUANTEEZE_VISION_SSD', 'off', 'memory', 'load',
     'on: stream vision tower weights from the checkpoint via mmap a block at a time when encoding images, '
     'freeing ~1 GB when idle (off by default; on under QUANTEEZE_COMPACT=on)'),
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
    ('QUANTEEZE_HOST_EXPERTS_BY', 'auto', 'experts', 'load',
     'how the host\'s copy of the experts (QUANTEEZE_HOST_EXPERTS_GB) is chosen: rows, an equal share of every '
     'layer\'s experts; layers, whole tensors from the first layer on; auto, rows (but layers on a network file '
     'system when host memory does not hold all routed experts)'),
    ('QUANTEEZE_WARM_UP', 'on', 'experts', 'load',
     'on CUDA, the model run once before the first request, so that quanteeze\'s own kernels are compiled then, '
     'not while it waits (not where experts are read on demand)'),
    ('QUANTEEZE_HOST_EXPERTS_GB', 'auto', 'experts', 'load',
     'GB of the checkpoint\'s routed experts held in the host\'s memory and read from there, not the disk '
     '(auto: on a discrete CUDA GPU with memory of its own, as many as the host\'s memory or the job\'s cgroup '
     'leaves, less a tenth of it and 3 GB for what the server adds after; none on a Mac, an integrated CUDA GPU '
     'sharing host memory, or where the checkpoint is in memory already, /dev/shm; off: none)'),
    ('QUANTEEZE_CPU_EXPERTS', 'auto', 'experts', 'load',
     'when serving on demand with a host-RAM expert copy (QUANTEEZE_HOST_EXPERTS_GB), compute a 1-token decode '
     'step\'s missing experts held in host memory on the CPU instead of copying them over PCIe '
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
    # -- kernels & speculative decoding (quanteeze.mtp, dspark, verify_qmv, expert_qmv, hyper, linear_attn)
    ('QUANTEEZE_MTP', 'auto', 'speed', 'load',
     'MTP / DSpark speculative decoding with the checkpoint\'s own head.  auto: where the checkpoint bundles a head '
     'and its model type\'s verify pass is checked (GLM-5.3, Flash-Next, DeepSeek V4.1); on: wherever a head loads; '
     'off: never'),
    ('QUANTEEZE_MTP_BITS', '2', 'speed', 'load', 'the MTP head\'s routed experts\' bits (off: the checkpoint\'s own)'),
    ('QUANTEEZE_MTP_EXACT', 'bits', 'speed', 'load',
     'how exactly an MTP verify pass computes what one-token decodes would (bits: bit for bit; distribution: '
     'MLX\'s multi-token pass)'),
    ('QUANTEEZE_MTP_BLOCK', 'auto', 'speed', 'load',
     'the tokens an MTP round verifies, the bonus and its drafts (auto: 2 for GLM-5.3 and Flash-Next, 6 for DSpark; '
     'off: mlx-vlm\'s choice)'),
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
    ('QUANTEEZE_KQUANT_QMV', 'on', 'speed', 'start',
     'run pre-distributed GGUF expert projections on Metal and CUDA (plus uniform Q2_K/Q3_K/IQ2_XS on Metal) '
     'through fused gather-qmv and tiled gather-mm kernels once verified against unpacking (off: unpack + gather_mm)'),
    ('QUANTEEZE_V41_HEAD_KERNEL', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, run DeepSeek V4.1\'s BF16 LM head in a single warp-per-row kernel without float32 '
     'intermediate buffers once verified (off: chunked float32 matmul)'),
    ('QUANTEEZE_HOST_NGRAM', 'on', 'speed', 'start',
     'compute 1-token decode PLE n-gram row ids on the host from the already-evaluated token id rather than '
     'syncing from the GPU (off: GPU)'),
    ('QUANTEEZE_ENGRAM_READ_AHEAD', 'on', 'speed', 'load',
     'prefetch later DeepSeek V4.1 Engram layers asynchronously while an earlier Engram layer runs (off: disabled)'),
    ('QUANTEEZE_ENGRAM_IDS_ONCE', 'on', 'speed', 'load',
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
     'checked bit for bit first (off: the rows read on the host)'),
    ('QUANTEEZE_CUDA_HEAD', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a greedy decode token\'s head (lm_head\'s rows and their argmax) in two kernels where '
     'MLX runs gather_qmv and argmax, the same token, each shape checked bit for bit first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_CUDA_ATTN', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a decode token\'s full attention while its prefix fits the indexer\'s budget in five '
     'kernels and MLX\'s cache update and matmul where mlx-vlm runs some forty, each softmax size checked bit for bit '
     'first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_CUDA_EXPERTS', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a decode token\'s 4-bit affine or MXFP4 routed experts in two kernels '
     '(gate+up+swiglu; down) where mlx-vlm runs four, checked bit for bit first (off: mlx-vlm\'s)'),
    ('QUANTEEZE_CUDA_ROUTER', 'on', 'speed', 'start',
     'on MLX\'s CUDA backend, a decode token\'s routing in one kernel where MLX runs six, checked bit for bit '
     'first (off: MLX\'s ops)'),
    ('QUANTEEZE_COMPILE_DECODE', 'on', 'speed', 'start',
     'Flash-Next\'s decode token with its routers, shared experts, weighting and hyper-connections as compiled '
     'graphs (mx.compile) (off: op by op)'),
    ('QUANTEEZE_EVAL_LAYERS', '12', 'speed', 'start',
     'Flash-Next\'s decode token handed to the GPU this many layers at a time, not one by one (mlx-vlm\'s): on '
     'MLX\'s CUDA backend each hand-over ends a CUDA graph, between which the GPU waits (1: mlx-vlm\'s)'),
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
     'on: DeepSeek V4.1 reads the images its user turns carry, its vision tower and aligner held (0.97 GB of the '
     'experts\' room); ssd: stream vision tower weights from the checkpoint on demand when encoding images; '
     'off: they are let go, and the prompt marks each image'),
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


def get(name, environ=None):
    """The variable's value, or its default (None for none) where it is unset or empty; one of
    COMPACT's as it gives it while QUANTEEZE_COMPACT is on (unless listed in QUANTEEZE_COMPACT_KEEP)."""
    setting = _setting(name)
    if name in COMPACT and compact(environ) and name not in compact_keeps(environ):
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


def number(name):
    return float(get(name))


def integer(name):
    return int(get(name))


def off(name):
    """Whether the variable says off."""
    val = get(name)
    return val is not None and val.lower() == 'off'


def path(name):
    """The value as a path, ~ expanded, or None."""
    val = get(name)
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
