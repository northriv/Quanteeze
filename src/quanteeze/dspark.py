# SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
"""DeepSeek V4.1 Flash's own draft head, DSpark, as a drafter of mlx-vlm's DFlash round loop.

The checkpoint keeps it under mtp.0-2 (7.9 GB, shards 44-46): three blocks of V4.1's own kind --
attention over a window of 128 alone, an MoE of 128 experts choosing 3, hyper-connections -- with
a projection of the target's hidden states before them (main_proj, main_norm, on the first) and a
norm, a Markov head and a confidence head after (the last).  DeepSeek's inference/model.py is the
reference (forward_spec: only its forward; nothing there drafts and verifies):

- The context: what enters the target's layers 37, 38 and 39, averaged over the hc copies,
  side by side (15,360 wide: quanteeze.arch.deepseek_v41.SpecLanguageModel captures them),
  projected to one hidden state a position (main_x).  Each block keeps its own window of the
  context's keys, a ring of 128 (DSparkAttention.seed): kv_norm(wkv(main_x)), rotated at the
  position, FP8-rounded.
- A round: the anchor -- the token the target chose last, not yet through it -- and four noise
  tokens (128799), at the positions after the context, go through the three blocks; each block
  attends from the five to its window and to the five themselves, no mask among them.  The
  last block's mix collapses the copies, and the target's head reads the five: position i
  proposes the token after position i, its logits plus the Markov head's bias for the token
  before it (an embedding of 256 and its own head), one token after another: five drafts.
  The confidence head (the hidden state and the Markov embedding: a logit a draft) is read in,
  not used yet.
- The target's embedding is the drafter's (bound at reset()); its head too, or the drafter's own
  copy of it at QUANTEEZE_DSPARK_HEAD_BITS (own_head(): with the BF16 head read whole each round
  the drafter took 9.2 ms greedy, 7.7 with its own at 8 bits, 7.1 at 4; acceptance no lower at 4,
  0.84-0.89 over some 600 rounds).  Compiled, the three stages took 0-0.6 ms
  less alone, and the rounds were no faster: they are not.

Every draft is the most likely by the drafter (draft_block ignores the sampler): mlx-vlm's
round loop verifies with the target's own sampling at each position and keeps the drafts the
target drew the same (dflash._sample_dflash_target_walk), which leaves the target's distribution
as it is whatever was proposed; the likeliest proposal is the one drawn most often.  Greedy, the
target's argmax against the drafts (dflash._speculative_walk).

The block was trained on five positions together, so the drafter always runs five and a round
of fewer verifies the first of them.  Its routed experts are FP4 in the checkpoint, read as
MXFP4 byte for byte, its FP8 projections as MXFP8 (mlx-vlm's _pack_source_weight); QUANTEEZE_MTP_BITS
lowers the experts (quanteeze.mtp.lower).
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import kernel_checks, per_load, per_thread

BLOCK = 5                       # positions a block drafts (dspark_block_size)
NOISE = 128799                  # the token the positions after the anchor are (dspark_noise_token_id)


def available(path):
    """Whether the checkpoint at `path` is DeepSeek V4.1 with a DSpark head (its model type, config
    and mtp.* keys)."""
    try:
        config = json.loads((Path(path) / 'config.json').read_text())
        text = config.get('text_config', config)
        if 'deepseek_v41' not in (config.get('model_type'), text.get('model_type')):
            return False                               # V4 (0731): mlx-vlm's deepseek_v4_dspark
        if not text.get('dspark_block_size') or not text.get('dspark_target_layer_ids'):
            return False
        names = json.loads((Path(path) / 'model.safetensors.index.json').read_text())['weight_map']
        return any(k.startswith('mtp.') for k in names)
    except (OSError, ValueError, KeyError):
        return False


def heads(path):
    """How many of the checkpoint's tensors are the head's (mtp.*)."""
    names = json.loads((Path(path) / 'model.safetensors.index.json').read_text())['weight_map']
    return sum(1 for k in names if k.startswith('mtp.'))


class Config:
    """What mlx-vlm's DFlash loop reads of a drafter's config, and DSpark's own."""

    model_type = 'deepseek_v41_dspark'

    def __init__(self, text):
        self.dspark_block_size = int(text.get('dspark_block_size') or BLOCK)
        self.block_size = self.dspark_block_size + 1             # tokens a round verifies: the anchor and drafts
        self.runtime_block_size = None
        self.target_layer_ids = list(text['dspark_target_layer_ids'])
        self.noise_token_id = int(text.get('dspark_noise_token_id') or NOISE)
        self.markov_rank = int(text.get('dspark_markov_rank') or 256)
        self.n_routed_experts = int(text.get('dspark_n_routed_experts') or 0)
        self.num_experts_per_tok = int(text.get('dspark_num_experts_per_tok')
                                       or text.get('dspark_n_activated_experts') or 0)
        self.n_stages = int(text.get('num_nextn_predict_layers') or 3)


def _attention_class():
    from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0 as fq
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Attention

    class DSparkAttention(DeepseekV41Attention):
        """A window-only attention (built as layer 0's: rope_theta without YaRN, as DeepSeek's
        pure sliding-window layers) whose keys are the context's window and the block's own."""

        def seed(self, state, main_x, start):
            """The context's keys at positions start.. into this block's ring (state.ring)."""
            n = main_x.shape[1]
            kv = self.kv_norm(self.wkv(fq(main_x)))
            kv = fq(self.rope(kv[:, None], start).reshape(main_x.shape[0], n, self.head_dim))
            win = self.window_size
            ring = state.ring.get(id(self))
            if ring is None:
                ring = mx.zeros((main_x.shape[0], win, self.head_dim), dtype=kv.dtype)
            keep = min(n, win)
            ring[:, (start + n - keep + mx.arange(keep)) % win] = kv[:, -keep:]
            state.ring[id(self)] = ring

        def __call__(self, x, state):
            batch, seqlen = x.shape[0], x.shape[1]
            at = state.anchor
            qr = self.q_norm(self.wq_a(fq(x)))
            q = self.wq_b(fq(qr)).reshape(batch, seqlen, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            q = self.rope(q, at)
            kv = self.kv_norm(self.wkv(fq(x)))
            kv = fq(self.rope(kv[:, None], at).reshape(batch, seqlen, self.head_dim))
            win = self.window_size
            valid = min(win, state.seen)
            context = mx.take(state.ring[id(self)], (at - valid + mx.arange(valid)) % win, axis=1)
            keys = mx.concatenate([context, kv], axis=1)[:, None].astype(q.dtype)
            out = mx.fast.scaled_dot_product_attention(q, keys, keys, scale=self.scale,
                                                       sinks=self.attn_sink.astype(q.dtype))
            out = self.rope(out, at, inverse=True)
            out = out.reshape(batch, self.o_groups, -1, seqlen, self.head_dim)
            out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
            out = self.wo_a(out)
            out = out.transpose(0, 2, 1, 3).flatten(-2)
            return self.wo_b(fq(out))

    return DSparkAttention


class State:
    """A request's drafting state: each block's ring of context keys, the positions seen."""

    def __init__(self):
        self.ring = {}
        self.seen = 0               # context positions written into the rings since the start
        self.anchor = 0             # the anchor's position: the target's offset when drafting


class Drafter(nn.Module):
    """DSpark: the three blocks (stages), main_proj and main_norm, the last block's norm, the
    Markov head (markov_embed, markov_head) and the confidence head (confidence)."""

    prefer_requested_block_size = True
    _quanteeze_counts_itself = True

    def __init__(self, config, cfg):
        super().__init__()
        from mlx_vlm.models.deepseek_v41.language import DeepseekV41Block, DeepseekV41MoE
        self.config, self.cfg = config, cfg
        attention = _attention_class()
        self.stages = []
        for _ in range(config.n_stages):
            stage = DeepseekV41Block(cfg, 0)
            stage.attn = attention(cfg, 0)
            stage.ffn = DeepseekV41MoE(cfg, n_routed_experts=config.n_routed_experts or None,
                                       num_experts_per_tok=config.num_experts_per_tok or None)
            self.stages.append(stage)
        d = cfg.hidden_size
        self.main_proj = nn.Linear(d * len(config.target_layer_ids), d, bias=False)
        self.main_norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        self.norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        self.markov_embed = nn.Embedding(cfg.vocab_size, config.markov_rank)
        self.markov_head = nn.Linear(config.markov_rank, cfg.vocab_size, bias=False)
        self.confidence = nn.Linear(d + config.markov_rank, 1, bias=False)
        self.head = None                # its own copy of the target's head, quantized (load(): HEAD_BITS)
        self.accept_lens, self.draft_lens = [], []
        self._target = None

    # mlx-vlm's DFlash drafter contract

    def reset(self, target_model):
        """Bind the target's embedding and head; a request's state afresh."""
        from .arch import deepseek_v41
        lm = deepseek_v41.speculative(target_model)
        object.__setattr__(self, '_target', lm)
        install_walk()
        self.accept_lens, self.draft_lens = [], []
        return State()

    def make_cache(self):
        """A row's state in mlx-vlm's batched rounds (_dflash_rounds_batch: each row drafted alone,
        the rows verified together): its drafts gated with the others' (_together)."""
        state = State()
        state.batched = True
        return state

    def draft_block(self, last_bonus, hidden, state, block_size, sampler=None, token_dtype=mx.int32, **kwargs):
        """block_size - 1 drafts after `last_bonus`, from `hidden`: the target's captured context
        for the positions just before it ([1, n, 15360], ending at the target's offset).  With
        the server's sampler at a temperature (Sampling.of), each drawn from the drafter's own
        distribution under the request's rule, kept (PENDING) for walk() to accept by p/q.  The
        gate (QUANTEEZE_DSPARK_GATE, gated()) may draft fewer.  A row of a batch (make_cache) gives
        every draft, the gate's count kept in ROWS."""
        offset = _offset(self._target)
        sampling = Sampling.of(sampler)
        base = int(getattr(sampler, 'positions', [0])[0]) if sampling is not None else 0
        anchor = int(last_bonus)
        drafts, q, conf = self.propose(anchor, hidden, state, offset, sampling, base)
        mx.async_eval(drafts, conf, *([] if q is None else [q]))
        if getattr(state, 'batched', False):                      # a row of several: as many drafts as the
            ROWS.append(gated(conf, max(0, block_size - 1)))      # fewest the gate keeps a row (_together)
            PENDING[0] = None                                     # mlx-vlm's walk, the drafts as the argmax
            return drafts[:, :max(0, block_size - 1)].astype(token_dtype)
        k = gated(conf, max(0, block_size - 1))
        drafts = drafts[:, :k].astype(token_dtype)
        PENDING[0] = Pending(drafts, None if q is None else q[:, :k], sampling, conf)
        return drafts

    def draft_block_greedy(self, last_bonus, hidden, state, block_size, sampler=None, token_dtype=mx.int32,
                           **kwargs):
        return self.draft_block(last_bonus, hidden, state, block_size, None, token_dtype)

    def propose(self, anchor, hidden, state, offset, sampling=None, base=0):
        """BLOCK drafts [1, BLOCK] after the token `anchor` at position `offset`, the context's
        `hidden` the positions up to it, None, and the confidence head's logits [1, BLOCK] (of
        each draft: the hidden state before the final norm and the Markov embedding of the token
        before it, as DeepSeek's DSparkConfidenceHead); with `sampling`, each draft drawn under
        its rule from the drafter's logits plus the Markov bias of the draft before (keys by
        position from `base`), and their log-probabilities [1, BLOCK, V] in place of None."""
        from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0 as fq
        from mlx_vlm.models.deepseek_v41.language import DeepseekV41Block, make_identity_pre_mix
        lm = self._target
        win = self.cfg.sliding_window
        hidden = hidden[:, -win:]
        n = hidden.shape[1]
        if n:
            main_x = self.main_norm(self.main_proj(fq(hidden)))
            for stage in self.stages:
                stage.attn.seed(state, main_x, offset - n)
            state.seen = min(state.seen + n, offset)
        state.anchor = offset
        batch = hidden.shape[0]
        ids = mx.array([[anchor] + [self.config.noise_token_id] * (BLOCK - 1)] * batch, dtype=mx.int32)
        h = lm.embed_tokens(ids)
        hc = self.cfg.hc_mult
        h = mx.broadcast_to(h[..., None, :], (batch, BLOCK, hc, h.shape[-1]))
        pre_mix = make_identity_pre_mix(batch, BLOCK, hc)
        for stage in self.stages:
            h, pre_mix = stage(h, pre_mix, None, state)
        x = DeepseekV41Block.hc_pre(h, pre_mix)
        head = self.head if self.head is not None else lm.head
        logits = head(self.norm(x)).astype(mx.float32)                       # [B, BLOCK, V]
        drafts, q, embeds = self.markov(logits, anchor, sampling, base)
        conf = self.confidence(mx.concatenate([x.astype(mx.float32), embeds.astype(mx.float32)], axis=-1))[..., 0]
        return drafts, q, conf

    def markov(self, logits, anchor, sampling=None, base=0):
        """Generate `BLOCK` speculative drafts, optional log-probs, and Markov embeddings via C++ when active."""
        native = _native()
        if native is not None:
            got = native_markov(self, native, logits, anchor, sampling, base)
            if NATIVE[1] >= NATIVE_CHECKS:
                return got
            ref = self.python_markov(logits, anchor, sampling, base)
            NATIVE[1] += 1
            same = all(
                (a is None and b is None)
                or (a is not None and b is not None and kernel_checks.same_bits(a, b))
                for a, b in zip(got, ref)
            )
            if not same:
                NATIVE[0] = False
                print(
                    'quanteeze: QUANTEEZE_DSPARK_NATIVE: C++ drafts parted from Python; using Python from now on',
                    file=sys.stderr,
                    flush=True,
                )
            return ref
        return self.python_markov(logits, anchor, sampling, base)

    def python_markov(self, logits, anchor, sampling=None, base=0):
        """Reference Python implementation of the sequential Markov head loop."""
        batch = logits.shape[0]
        token = mx.array([anchor] * batch, dtype=mx.int32)
        out, qs, embeds = [], [], []
        for i in range(BLOCK):
            embed = self.markov_embed(token)
            embeds.append(embed)
            bias = self.markov_head(embed).astype(mx.float32)
            if sampling is None:
                token = mx.argmax(logits[:, i] + bias, axis=-1).astype(mx.int32)
            else:
                q = sampling.logprobs(logits[:, i] + bias)
                token = mx.random.categorical(q, key=sampling.key(PROPOSE, base + i)).astype(mx.int32)
                qs.append(q)
            out.append(token)
        return mx.stack(out, axis=1), (mx.stack(qs, axis=1) if qs else None), mx.stack(embeds, axis=1)


NATIVE = [None, 0, 0]
NATIVE_CHECKS = 3


def _native():
    """Return the compiled `_dspark` C++20 extension when QUANTEEZE_DSPARK_NATIVE enables it."""
    if NATIVE[0] is None:
        from . import native_ext, variables
        m = (variables.get('QUANTEEZE_DSPARK_NATIVE', variables.at_load()) or 'auto').strip().lower()
        use = m == 'on' or (m == 'auto' and mx.metal.is_available())
        ext = native_ext.extension('dspark', with_mlx=True, with_stm=False, quiet=(m == 'auto')) if use else None
        NATIVE[0] = ext or False
    return NATIVE[0] or None


def _spec(native, module):
    """Describe a linear or embedding layer (plain or quantized) for `_dspark.proj`."""
    params = module.parameters() if hasattr(module, 'parameters') else {}
    quantized = 'scales' in params
    return native.proj(
        1 if quantized else 0,
        module['weight'],
        params.get('scales'),
        params.get('biases'),
        int(getattr(module, 'group_size', 0) or 0),
        int(getattr(module, 'bits', 0) or 0),
        str(getattr(module, 'mode', 'affine') or 'affine'),
        params.get('bias'),
    )


def native_markov(drafter, native, logits, anchor, sampling, base):
    """Run `Drafter.markov()` via the compiled `_dspark` extension."""
    specs = getattr(drafter, '_quanteeze_dspark_specs', None)
    if specs is None or specs[0] is not drafter.markov_embed or specs[1] is not drafter.markov_head:
        specs = (
            drafter.markov_embed,
            drafter.markov_head,
            _spec(native, drafter.markov_embed),
            _spec(native, drafter.markov_head),
        )
        try:
            object.__setattr__(drafter, '_quanteeze_dspark_specs', specs)
        except Exception:
            pass
    rule = None if sampling is None else (sampling.temperature, sampling.top_p, sampling.top_k)
    keys = [] if sampling is None else [sampling.key(PROPOSE, base + i) for i in range(logits.shape[1])]
    return native.markov(logits, int(anchor), specs[2], specs[3], rule, keys)


PENDING = per_thread.Field('round')     # the round being verified: Pending
GATE = [None]                   # QUANTEEZE_DSPARK_GATE as read (gated()), and the log (QUANTEEZE_DSPARK_LOG)
LOG = [None, None]              # its file, and when the last round was walked (perf_counter)
LOG_KEPT_BYTES = 64 << 20
LOG_WRITTEN_BYTES = 64 << 20
LOG_BYTES = {'file': 0, 'written': 0, 'said': False}


class Pending:
    """A round's drafts [1, k], their log-probabilities [1, k, V] (None: greedy, or a sampler of
    another rule), the request's Sampling, and the confidence head's logits of all BLOCK."""

    def __init__(self, drafts, q, sampling, conf):
        self.drafts, self.q, self.sampling, self.conf = drafts, q, sampling, conf


GATE_EXACT, GATE_FAST = 0.6, 0.45    # QUANTEEZE_DSPARK_GATE=auto's, the verify pass exact or not (gate_threshold)


def gate_threshold():
    """QUANTEEZE_DSPARK_GATE (0.6): drafts are kept while the product of their confidences
    (sigmoid of the head's logits) stays at or above it; off: every draft asked.

    The head is calibrated: over 3,000 rounds of five drafts (English, Japanese and code, at
    temperature 1 and 0) a draft of confidence 0.2-0.4 was kept 0.3 of the
    time, 0.6-0.8 0.7, above 0.9 0.95-0.99.  A round served on demand cost some 60 ms and 32 more
    a draft verified, so a draft pays where the chance of keeping it and those before it beats
    the 32 ms's share of the tokens a second (about 0.6): the logs' rounds gated at 0.6 gave
    some 20 tokens/s, against 18 drafting one always, 16.7 two, and 17.2-17.9 not drafting.

    auto (the default): GATE_EXACT where the verify pass is exact (QUANTEEZE_MTP_EXACT=bits), else
    GATE_FAST: with the pass as MLX computes its tokens together a draft verified costs some 8.5 ms a
    round, not 14-20, and the rounds at 0.45 ran 27-29 tokens/s against 26 at 0.6 and 22 at 0.75
    (0.3: 25-33, as spread as acceptance 0.58-0.71 makes it; offline, temperature 1, 300 tokens,
    three seeds).  serve._drafter resets it at each load."""
    if GATE[0] is None:
        from . import variables
        raw = (variables.get('QUANTEEZE_DSPARK_GATE', variables.at_load()) or 'off').strip().lower()
        if raw == 'auto':
            from .arch import deepseek_v41
            GATE[0] = (GATE_EXACT if deepseek_v41.EXACT[0] else GATE_FAST,)
        else:
            GATE[0] = (0.0 if raw in ('off', '0') else float(raw),)
    return GATE[0][0]


def gated(conf, k):
    """How many of the first k drafts the gate keeps (k with no gate): waits on confidence logits in C++ when active."""
    threshold = gate_threshold()
    if not threshold or not k:
        return k
    native = _native()
    if native is not None and conf.shape[0] == 1 and conf.dtype == mx.float32:
        got = native.gated(conf, int(k), float(threshold))
        if NATIVE[2] >= NATIVE_CHECKS:
            return got
        NATIVE[2] += 1
        ref = python_gated(conf, k, threshold)
        if got != ref:
            NATIVE[0] = False
            print(
                f'quanteeze: QUANTEEZE_DSPARK_NATIVE: C++ gate kept {got}, Python kept {ref}; using Python from now on',
                file=sys.stderr,
                flush=True,
            )
        return ref
    return python_gated(conf, k, threshold)


def python_gated(conf, k, threshold):
    """Reference Python implementation of `gated()`."""
    kept = mx.cumprod(mx.sigmoid(conf[0, :k])) >= threshold
    return int(mx.sum(mx.cumprod(kept.astype(mx.int32))).item())


def log_round(pending, alphas, accepted, budget):
    """QUANTEEZE_DSPARK_LOG: append one JSON line per round, capped at `LOG_KEPT_BYTES` and `LOG_WRITTEN_BYTES`."""
    import time
    if LOG[0] is None:
        from . import variables
        path = variables.get('QUANTEEZE_DSPARK_LOG', variables.at_load())
        LOG[0] = _open_log(Path(path).expanduser()) if path else False
    if not LOG[0]:
        return
    now = time.perf_counter()
    gap, LOG[1] = (None if LOG[1] is None else now - LOG[1]), now
    conf = mx.sigmoid(pending.conf[0]).tolist()
    line = json.dumps({'conf': [round(c, 4) for c in conf], 'alpha': [round(a, 4) for a in alphas],
                       'k': int(pending.drafts.shape[1]), 'a': int(accepted), 'budget': int(budget),
                       'temp': pending.sampling.temperature if pending.sampling else 0.0,
                       'gap_ms': None if gap is None else round(gap * 1e3, 2)}) + '\n'
    if _log_full(len(line)):
        return
    LOG[0].write(line)
    LOG_BYTES['file'] += len(line)
    LOG_BYTES['written'] += len(line)


def _open_log(path):
    """Open `path` for append unless it already exceeds `LOG_KEPT_BYTES`."""
    file = open(path, 'a', buffering=1)
    LOG_BYTES['file'] = path.stat().st_size
    return file if not _log_full(0, file) else False


def _log_full(size, file=None):
    """Close the log and stop appending if adding `size` bytes would exceed either 64 MB cap."""
    if LOG_BYTES['file'] + size <= LOG_KEPT_BYTES and LOG_BYTES['written'] + size <= LOG_WRITTEN_BYTES:
        return False
    (file or LOG[0]).close()
    LOG[0] = False
    if not LOG_BYTES['said']:
        LOG_BYTES['said'] = True
        print(
            f'quanteeze: QUANTEEZE_DSPARK_LOG: no more rounds logged -- the file holds {LOG_BYTES["file"] >> 20} MB '
            f'(at most {LOG_KEPT_BYTES >> 20}) and this process wrote {LOG_BYTES["written"] >> 20} MB '
            f'(at most {LOG_WRITTEN_BYTES >> 20}); the file is left as is',
            file=sys.stderr,
            flush=True,
        )
    return True
PROPOSE, UNIFORM, RESIDUAL = 0x0DFA5202, 0x5A3E1C07, 0x2B7F49D1     # salts of the keys by position


class Sampling:
    """A request's sampling rule as the server's positioned sampler draws the target's tokens
    (mlx-vlm's _PositionedTargetSampler: top-k on the log-probabilities, then the temperature,
    then top-p, each position's draw keyed by (seed, row, position)).  of(sampler): the rule of
    a sampler that has one, else None (greedy, or a sampler of other rules: drafts are then the
    argmax, accepted when the target draws them)."""

    def __init__(self, temperature, top_p, top_k, seed):
        self.temperature, self.top_p, self.top_k, self.seed = float(temperature), float(top_p), int(top_k), int(seed)

    @classmethod
    def of(cls, sampler):
        target = getattr(sampler, 'sampler', sampler)
        if not callable(getattr(target, 'sample_target', None)):
            return None
        try:
            t, p, k, s = target.temperature, target.top_p, target.top_k, target.seed
        except AttributeError:
            return None
        return cls(t, p, k, s) if t > 0 else None

    def logprobs(self, logits):
        """The log-probabilities [..., V] the rule draws from, normalized (-inf where it never draws)."""
        from mlx_vlm.sample_utils import apply_top_k, apply_top_p
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        if 0 < self.top_k < lp.shape[-1]:
            lp = apply_top_k(lp, self.top_k)
        lp = lp / self.temperature
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        if 0 < self.top_p < 1:
            lp = apply_top_p(lp, self.top_p)
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        return lp

    def key(self, salt, position, row=0):
        from mlx_vlm.server.generation import _position_seed
        return mx.random.key(_position_seed(self.seed ^ salt, row, position))


def accept(p, q, drafts, sampling, base):
    """Speculative sampling's walk over k drafts: draft i (drawn from q[i]) is kept with probability
    min(1, p[i]/q[i]) of the target's p at its position; at the first not kept, a token is drawn
    from max(0, p - q) renormalized (p where that is nothing) -- so each token comes from the
    target's distribution exactly, whatever q is; past the last draft p[k] gives the bonus.
    p [k + 1, V] and q [k, V] are log-probabilities, drafts [k].  Draws keyed by position from
    `base`.  Returns (accepted, the next token: the resampled one or the bonus) as arrays."""
    k = drafts.shape[0]
    at = drafts[:, None]
    ratio = mx.take_along_axis(p[:k], at, axis=-1)[:, 0] - mx.take_along_axis(q, at, axis=-1)[:, 0]
    u = mx.stack([mx.random.uniform(key=sampling.key(UNIFORM, base + i)) for i in range(k)]) if k else mx.zeros((0,))
    kept = mx.log(u) <= ratio
    accepted = mx.sum(mx.cumprod(kept.astype(mx.int32)))
    rest = mx.maximum(mx.exp(p[:k]) - mx.exp(q), 0.0)
    rest = mx.where(mx.sum(rest, axis=-1, keepdims=True) > 0, mx.log(rest), p[:k])
    drawn = [mx.random.categorical(rest[i], key=sampling.key(RESIDUAL, base + i)) for i in range(k)]
    drawn.append(mx.random.categorical(p[k], key=sampling.key(0, base + k)))
    drawn = mx.stack(drawn).astype(mx.int32)
    return accepted, mx.take(drawn, accepted)


def _walk(logits, draft_tokens, sampler, budgets, *, row_ids, base_positions):
    """mlx-vlm's _sample_dflash_target_walk, or for the round DSpark drafted under a Sampling
    (PENDING), accept(): one sync for the walk, as mlx-vlm's."""
    return _kept(_sampled_walk(logits, draft_tokens, sampler, budgets, row_ids=row_ids,
                               base_positions=base_positions))


def _sampled_walk(logits, draft_tokens, sampler, budgets, *, row_ids, base_positions):
    pending, PENDING[0] = PENDING[0], None
    DRAFTED[0] = int(draft_tokens.shape[1])
    if pending is None or pending.q is None or pending.drafts is not draft_tokens or logits.shape[0] != 1:
        return _WALK[0](logits, draft_tokens, sampler, budgets, row_ids=row_ids, base_positions=base_positions)
    drafts, q, sampling = pending.drafts, pending.q, pending.sampling
    k = drafts.shape[1]
    p = sampling.logprobs(logits[0, :k + 1])
    accepted, token = accept(p, q[0], drafts[0], sampling, int(base_positions[0]))
    if LOG[0] is not False:
        at = drafts[0][:, None]
        alpha = mx.minimum(mx.exp(mx.take_along_axis(p[:k], at, axis=-1)[:, 0]
                                  - mx.take_along_axis(q[0], at, axis=-1)[:, 0]), 1.0)
        log_round(pending, alpha.tolist(), accepted.item(), budgets[0])
    a, token, *tokens = mx.concatenate([mx.stack([accepted, token]), drafts[0].astype(mx.int32)]).tolist()
    return [a], [(tokens[:a] + [token])[:budgets[0]]]


def _greedy_walk(draft_tokens, target_tokens, budget):
    """mlx-vlm's _speculative_walk, the round logged where DSpark drafted it greedily."""
    pending = PENDING[0]
    DRAFTED[0] = int(draft_tokens.shape[1])
    out = _GREEDY_WALK[0](draft_tokens, target_tokens, budget)
    if pending is not None and pending.q is None and pending.drafts is draft_tokens and LOG[0] is not False:
        PENDING[0] = None
        drafts, target = draft_tokens[0].tolist(), target_tokens.reshape(-1).tolist()
        log_round(pending, [float(d == t) for d, t in zip(drafts, target)], out[0], budget)
    _kept(([out[0]], None))
    return out


def _kept(walked):
    """A round's walk, (accepted a row, ...)."""
    return walked


def _together(original):
    """mlx-vlm's _SpeculativeSamplerRNG.draft_tokens, the batched rounds' drafts of all rows at once
    cut to the fewest the gate kept a row (ROWS): mlx-vlm concatenates the rows' drafts and
    verifies them in one pass, which takes as many a row."""
    def draft_tokens(self, fn, *args, **kwargs):
        ROWS.clear()
        out = original(self, fn, *args, **kwargs)
        if ROWS:
            out = out[:, :min(ROWS)]
            DRAFTED[0] = min(ROWS)                # counted for each row (_record), greedy or not
            ROWS.clear()
        return out
    draft_tokens.quanteeze = True
    return draft_tokens


_WALK, _GREEDY_WALK, _RECORD = [None], [None], [None]
ROWS = per_thread.Rows('rows')          # the gate's count of each row a batched round drafted (_together)
DRAFTED = per_thread.Field('drafted')   # the drafts the round just walked had: the gate may have kept fewer


def _record(draft_model, accepted, draft_count):
    """mlx-vlm's _record_speculative_round, counting the drafts DSpark's gate kept, not the block's."""
    if isinstance(draft_model, Drafter) and DRAFTED[0] is not None:
        draft_count = DRAFTED[0]                  # every row of a batched round's (mlx-vlm records a row each)
    if isinstance(draft_model, Drafter):
        from . import mtp
        mtp.count_round(draft_model, accepted, draft_count)
    return _RECORD[0](draft_model, accepted, draft_count)


def install_walk():
    """mlx-vlm's DFlash rounds walk by accept() for DSpark's sampled drafts, log greedy rounds
    where asked, count the drafts kept, and verify a batch's rows with as many drafts each
    (_together) (once)."""
    from mlx_vlm.speculative import dflash
    if _WALK[0] is None:
        _WALK[0] = dflash._sample_dflash_target_walk
        dflash._sample_dflash_target_walk = _walk
        _GREEDY_WALK[0] = dflash._speculative_walk
        dflash._speculative_walk = _greedy_walk
        _RECORD[0] = dflash._record_speculative_round
        dflash._record_speculative_round = _record
        rng = dflash._SpeculativeSamplerRNG
        if not getattr(rng.draft_tokens, 'quanteeze', False):
            rng.draft_tokens = _together(rng.draft_tokens)


def _offset(lm):
    """The target's offset: the positions its cache holds (its last pass's, SpecLanguageModel.entry)."""
    entry = lm.entry()
    if entry is None:
        raise RuntimeError('DSpark: the target has run no pass yet')
    rows = getattr(entry, 'rows', None)
    return int(rows[0].offset) if rows is not None else int(entry.offset)


TOP = {'main_proj': 'main_proj', 'main_norm': 'main_norm', 'norm': 'norm', 'markov_head.embed': 'markov_embed',
       'markov_head.head': 'markov_head', 'confidence_head.proj': 'confidence'}     # the checkpoint's: the drafter's
SHARED = {'w1': 'gate_proj', 'w2': 'down_proj', 'w3': 'up_proj'}                    # an expert's: SwitchGLU's


def weights(path, config, cfg):
    """The drafter's weights from the checkpoint at `path` (its mtp.* keys), FP8 and FP4 packed
    as MLX's MXFP8 and MXFP4 layers, byte for byte: ({name: array}, {quantized layer: mode})."""
    from mlx_vlm.models.deepseek_v41.deepseek_v41 import _pack_source_weight
    index = json.loads((Path(path) / 'model.safetensors.index.json').read_text())['weight_map']
    source = {}
    for shard in sorted({f for k, f in index.items() if k.startswith('mtp.')}):
        source.update({k: v for k, v in mx.load(str(Path(path) / shard)).items() if k.startswith('mtp.')})
    out, modes, experts = {}, {}, {}
    for key, value in source.items():
        if key.endswith('.scale'):
            continue
        _, stage, rest = key.split('.', 2)
        module, _, leaf = rest.rpartition('.')
        scale = source.get(f'mtp.{stage}.{module}.scale') if leaf == 'weight' else None
        if module.startswith('ffn.experts.'):
            _, _, e, w = module.split('.')
            experts.setdefault((stage, w), {})[int(e)] = (value, scale)
            continue
        if module in TOP:
            name = TOP[module]
        else:
            for w, p in SHARED.items():
                module = module.replace(f'shared_experts.{w}', f'shared_experts.{p}')
            name = f'stages.{stage}.{module}' if module else f'stages.{stage}'
        if scale is not None:
            packed, scales, modes[name] = _pack_source_weight(value, scale)
            if module == 'attn.wo_a':
                packed = packed.reshape(cfg.o_groups, cfg.o_lora_rank, -1)
                scales = scales.reshape(cfg.o_groups, cfg.o_lora_rank, -1)
            out[f'{name}.weight'], out[f'{name}.scales'] = packed, scales
            continue
        if name.endswith('ffn.gate') and leaf == 'weight' or name == 'confidence':
            value = value.astype(mx.float32)                    # router scores and confidence in float32
        out[f'{name}.{leaf}'] = value
    for (stage, w), parts in experts.items():
        packed = [_pack_source_weight(*parts[e]) for e in range(len(parts))]
        name = f'stages.{stage}.ffn.switch_mlp.{SHARED[w]}'
        out[f'{name}.weight'] = mx.stack([p[0] for p in packed])
        out[f'{name}.scales'] = mx.stack([p[1] for p in packed])
        modes[name] = packed[0][2]
    return out, modes


def load(path, bits=2):
    """The DSpark drafter of the V4.1 checkpoint at `path`, its routed experts lowered to `bits`
    (None: the checkpoint's MXFP4).  Returns (drafter, 'dflash')."""
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    from . import mtp
    raw = json.loads((Path(path) / 'config.json').read_text())
    text = raw.get('text_config', raw)
    config, cfg = Config(text), ModelConfig.from_dict(raw)
    drafter = Drafter(config, cfg)
    tensors, modes = weights(path, config, cfg)
    for mode in ('mxfp8', 'mxfp4'):
        names = {n for n, m in modes.items() if m == mode}
        nn.quantize(drafter, group_size=32, bits=8 if mode == 'mxfp8' else 4, mode=mode,
                    class_predicate=lambda p, m, names=names: p in names and hasattr(m, 'to_quantized'))
    drafter.load_weights(list(tensors.items()), strict=True)
    del tensors, modes
    if bits is not None:
        mtp.lower(drafter, bits)
    drafter.head = own_head(path, head_bits())
    mx.eval(drafter.parameters())
    _native()
    return drafter, 'dflash'


HEAD_GROUP = 64


def head_bits():
    """QUANTEEZE_DSPARK_HEAD_BITS: the bits of the drafter's own copy of the target's head, or None
    (off: the target's own, read whole each round)."""
    from . import variables
    raw = (variables.get('QUANTEEZE_DSPARK_HEAD_BITS', variables.at_load()) or 'off').strip().lower()
    return None if raw in ('off', '0') else int(raw)


def own_head(path, bits):
    """The checkpoint's head (head.weight, V4.1's in BF16: 129,280 x 5,120, 1.3 GB, which the drafter
    read whole for its five positions each round) quantized at `bits` in groups of HEAD_GROUP, as
    the drafter's own: a draft's logits from it, the target's from its own.  None with no bits, no
    such tensor, or rows HEAD_GROUP does not divide."""
    if not bits:
        return None
    index = json.loads((Path(path) / 'model.safetensors.index.json').read_text())['weight_map']
    if 'head.weight' not in index:
        return None
    weight = mx.load(str(Path(path) / index['head.weight']))['head.weight']
    if weight.shape[-1] % HEAD_GROUP:
        return None
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight = weight.astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(linear, group_size=HEAD_GROUP, bits=bits)


@per_load.on_let_go
def let_go():
    """Close any DSpark log handle and reset per-load native extension/check/gate state on unload."""
    if LOG[0]:
        LOG[0].close()
    LOG[:] = [None, None]
    LOG_BYTES['file'] = 0
    NATIVE[:] = [None, 0, 0]
    GATE[0] = None
