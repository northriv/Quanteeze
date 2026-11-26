# llama.cpp-Compatible Benchmarks (Quanteeze)

**English** | **[日本語](README.ja.md)**

Python ports of llama.cpp's `llama-bench` and `llama-batched-bench` measurement definitions (test names and order, timing intervals, statistics, and table columns) targeting a live server over HTTP.

| llama.cpp Tool | Script Here | What It Measures |
|---|---|---|
| `llama-bench` | `llama_bench.py` | `pp512` (512-token prompt), `tg128` (128-token generation), `pp512+tg128`, `@ d4096` (after 4,096 context tokens); mean ± std `t/s` over 5 repetitions |
| `llama-batched-bench` | `batched_bench.py` | Concurrent `B`-request `PP`, `TG`, `B`, `N_KV`, `T_PP`, `S_PP`, `T_TG`, `S_TG`, `T`, `S` |

## Provenance & License

- Test definitions, statistics, and table layouts follow [llama.cpp](https://github.com/ggml-org/llama.cpp) (`master` `8d60873`) `tools/llama-bench/llama-bench.cpp` and `tools/batched-bench/batched-bench.cpp` (MIT License; see [LICENSE](LICENSE)).
- Adapted from llama.cpp's direct C API calls to measure a running HTTP server.

## Usage

Start a dedicated server instance on a separate port with `--log-level INFO` redirected to a log file (used by `--feed forced` to read exact server-side timings). Since llama.cpp's `llama-bench` / `llama-batched-bench` run without speculative decoding, pass `QUANTEEZE_MTP=off` for apples-to-apples comparisons:

```bash
env QUANTEEZE_SETTINGS=off QUANTEEZE_MTP=off APC_DISK_ENABLED=0 \
  uv run quanteeze --model <checkpoint> --host 127.0.0.1 --port 8766 \
  --enable-thinking --log-level INFO > serve.log 2>&1 &
```

### `llama_bench.py`

```bash
uv run python bench/llamacpp/llama_bench.py --url 127.0.0.1:8766 --checkpoint <checkpoint> --log serve.log \
    -p 512 -n 128 -pg 512,128 -d 0,4096,16384 -r 5
```

- **`--feed forced` (default, Quanteeze)**:
  - Sends exact `N`-token random token IDs (starting with BOS when present) and forces random output token IDs step by step via `quanteeze_forced` authenticated with the control token (`QUANTEEZE_CONTROL_TOKEN` or `~/.quanteeze/control-token`).
  - Reads exact server timings from `Prefill completed` / `Decode completed` (`tg` counts `generated - 1` steps, excluding the first token emitted by prefill).
  - **Depth `d`**: prepends `d` random tokens before the test prompt. Uses `Prefill progress` delta at `d` when aligned (multiples of 2,048 recommended), full prefill time when APC hits `d`, or subtracts a standalone `d`-token calibration (`~` marker).
- **`--feed text` (any OpenAI-compatible server)**:
  - Sends random word sequences and story prompts at `temperature=0`. Omitting `--log` falls back to client-side TTFT and completion wall times (`c` marker).
- **Note**:
  - `--feed forced` runs through a logits processor (bypassing fused greedy argmax shortcuts), and random token IDs exercise a more uniform expert distribution than natural text.

### `batched_bench.py`

```bash
uv run python bench/llamacpp/batched_bench.py --url 127.0.0.1:8766 --checkpoint <checkpoint> --log serve.log \
    -npp 128,512 -ntg 128 -npl 1,2,4,8
```

- Dispatches `B` unshared-prefix requests concurrently from `B` threads.
- Computes `T` from first `Generation queued` to last `Decode completed`, `T_PP` as the union of per-request prefill intervals, and `T_TG = T - T_PP`, matching `llama-batched-bench` columns (`S_PP = B*PP/T_PP`, `S_TG = B*TG/T_TG`, `S = B*(PP+TG)/T`).
- Concurrent execution requires either `QUANTEEZE_REQUEST_THREADS` > 1 or `mlx-vlm` continuous batching.

## Limitations

- Library-level llama.cpp flags such as `-pps` (shared prefix), `-b`/`-ub` (batch size), and `-ngl` do not apply.
