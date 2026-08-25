# Results: llama.cpp CPU inference on a 2019 MacBook (6C/12T i7)

## Environment
- Machine: 2019 MacBook, Intel i7-9750H, 6 physical cores / 12 logical (Hyper-Threading), DDR4.
- Runtime: llama.cpp `llama-server`, built **CPU-only** with `-DGGML_ACCELERATE=ON -DGGML_METAL=OFF`.
  - Metal was disabled after the first run crashed in `ggml-metal` (`GGML_ASSERT(buf_dst)`); the AMD dGPU path is broken on this Intel Mac, and leaving it on aborted mid-request.
- Model: `Qwen2.5-0.5B` (Instruct), GGUF `q4_k_m` (0.43 GB) and `q8_0` (0.68 GB).
- Methodology: 5 fixed prompts, `max_tokens=128`, `temperature=0` (deterministic).
  - Correctness: exact token-match vs a frozen reference output (no quality regression).
  - Latency: per-request tokens/sec (decode) and TTFT (time to first token = prompt eval).
  - Evaluator: `eval_client.py` (OpenAI-compatible `/v1/chat/completions`), results in `results/`.

## Headline finding
Two orthogonal, reproducible tweaks take this setup from **6.67 tok/s to 26.85 tok/s (~4.0×)** with identical outputs (exact-match = 1.0):

1. **Thread count = physical cores, not logical.** Using `--threads $(sysctl -n hw.ncpu)` (=12, all logical threads) was the *worst* config: Hyper-Thread siblings contend on the 6 physical cores and the decode path collapses.
2. **Quantize to `q4` for decode speed.** At the correct thread count, `q4` decodes ~1.5× faster than `q8` (smaller weight footprint = less memory traffic). Quality on these prompts was identical; `q8` is the safer default for quality-sensitive work.

## Thread-count sweep (q8_0, single stream)
| threads | tok/s (median) | TTFT (median) |
|---|---|---|
| 2  | 15.5 | 1.63 s |
| 3  | 16.0 | 1.40 s |
| 4  | 17.0 | 1.44 s |
| 6  | **17.4** | **1.28 s** |
| 8  | 14.0 | 1.66 s |
| 12 | **6.7** | 3.37 s |

Clear peak at physical-core count (6); sharp degradation at 8 and catastrophic collapse at 12.

## Quantization at the optimal thread count (threads = 6)
| config | tok/s | TTFT | exact-match |
|---|---|---|---|
| q8_0, t=6 | 17.4 | 1.28 s | 1.0 |
| q4_k_m, t=6 | **26.85** | 0.85 s | 1.0 |

## Batching (parallel) result (negative finding)
`--parallel 4` at threads=12 showed **no single-stream benefit** (5.57 tok/s vs 6.67) — expected, since the evaluator issues one request at a time. Batching helps only under *concurrent* load (aggregate throughput), which this single-stream study does not exercise.

## Scaling to larger models (1.5B, 3B) — findings generalize
Same protocol (reference frozen per model at q8/t6; exact-match holds for every
config including q4, so no quality regression):

| model | q8 t6 | q8 t12 | q4 t6 |
|---|---|---|---|
| 1.5B | 4.29 | 2.91 | **7.36** |
| 3B   | 3.64 | 1.81 | **7.76** |

- The **physical-core rule is even stronger at larger sizes**: at 3B, `t12` is
  **2× slower** than `t6` (1.81 vs 3.64 tok/s). Hyper-Thread contention hurts
  more when the working set is bigger.
- **`q4` at optimal threads is the fastest decode** at every size (it is
  memory-bandwidth bound; smaller weights = less traffic). 3B `q4` decodes at
  7.76 tok/s vs 3.64 for `q8` at the same threads.
- **End-to-end speedup vs naive defaults** (`q8`, `t=$(hw.ncpu)=12`):
  ~2.5× at 1.5B, **~4.3× at 3B**.

## Recommended defaults (baked into `scripts/serve_qwen25_*.sh`)
- `--threads $(sysctl -n hw.physicalcpu)` (6 on this machine) — never `hw.ncpu`.
- `--parallel 1` for single-stream interactive use.
- `q8_0` for quality; `q4_k_m` for max speed (set `MODEL=...` before launching).
- Build llama.cpp **without Metal** on this Intel Mac: `-DGGML_METAL=OFF -DGGML_ACCELERATE=ON`.

## Reproduce
```
bash run_study.sh          # baseline + 3 candidates (see results/*.json)
bash run_sweep.sh          # thread-count sweep (results/sweep_t*.json)
MODEL=qwen2.5-0.5b-q4_k_m.gguf bash scripts/serve_qwen25_instruct.sh
```

## Concurrent load (batching axis) — `--parallel` helps under load
Single-stream eval can't show batching value, so a threaded 8-client benchmark
(`concurrent_test.py`, 1.5B q8, threads=6) isolates it:

| config | aggregate tok/s | wall | TTFT p95 |
|---|---|---|---|
| `--parallel 1` | 9.39 | 51.8 s | 47.9 s |
| `--parallel 8` | **13.58** | 35.8 s | **11.9 s** |

Raising `--parallel` to match concurrency yields **~1.45× more aggregate
throughput and ~4× lower tail latency**. Takeaway: size `--parallel` to your
expected concurrent request count when serving, but keep `--threads` at physical
cores. (Authoritative single-request latency remains the single-stream numbers above.)

## Caveats
- Results are for Qwen2.5-0.5B; larger models will be more memory-bound and may favor `q4`/fewer-threads even more, but the "use physical cores" rule holds.
- Exact-match=1.0 is for these 5 short prompts; 4-bit may introduce subtle degradation on harder tasks — prefer `q8` when quality matters.
- `parallel` benefit under concurrent load is unverified here and worth a separate multi-client throughput test.
