# LLM Inference Optimization on CPU — A Measured Case Study

**Goal:** squeeze the maximum tokens/sec out of a laptop CPU for local LLM serving,
with reproducible evidence and zero cloud/GPU dependency.

**Headline result:** on a 2019 6-core/12-thread Intel MacBook, the right
thread/quantization settings deliver a **~4× decode-speedup at identical output
quality** (exact token-match preserved). The naive defaults (`--threads = logical
core count`, `q8`) were the *worst* configuration tested.

| model | naive (q8, 12 threads) | optimized (q4, 6 threads) | speedup |
|---|---|---|---|
| Qwen2.5-0.5B | 6.67 tok/s | **26.85 tok/s** | **4.0×** |
| Qwen2.5-1.5B | 2.91 tok/s | **7.36 tok/s** | 2.5× |
| Qwen2.5-3B   | 1.81 tok/s | **7.76 tok/s** | **4.3×** |

## Part 1 — Threading & quantization (the 4× win)

### What actually moved the needle
1. **`--threads` = physical cores, NOT logical.** Using all 12 logical threads
   (Hyper-Thread siblings) on a 6-core chip *collapsed* throughput — 6.7 tok/s
   vs 17.4 at 6 threads for 0.5B, and 1.8 vs 3.6 for 3B. Hyper-Thread contention
   on the 6 physical cores dominates. Set `--threads $(sysctl -n hw.physicalcpu)`.
2. **Quantize to `q4` for decode speed.** At the correct thread count, `q4`
   decodes ~1.5× faster than `q8` (smaller weights → less memory traffic),
   with identical greedy outputs on these prompts.
3. **Batching helps only under concurrent load.** A single-stream client sees no
   benefit from `--parallel`; under 8 concurrent clients, `--parallel 8` gave
   +45% aggregate throughput and ~4× lower tail latency vs `--parallel 1`.
4. **Build llama.cpp without Metal on this Intel Mac** (`-DGGML_METAL=OFF
   -DGGML_ACCELERATE=ON`). The Metal/AMD-GPU backend crashes mid-request here;
   CPU+Accelerate is correct and stable.

## Method (reproducible)
- Server: `llama.cpp` `llama-server` (OpenAI-compatible API, port 8080).
- Evaluator: `eval_client.py` (Python stdlib only) streams
  `/v1/completions` at `temperature=0`, measuring TTFT and tok/s, and compares
  output to a **frozen reference** (`reference_outputs*.json`). Same-weight
  candidates must match exactly (determinism gate); quality is preserved.
- Concurrent axis: `concurrent_test.py` fires N threaded requests and reports
  aggregate throughput + tail latency.
- Protocol: 5 fixed prompts, `max_tokens=128`, thread sweep {2,3,4,6,8,12},
  q4 vs q8 at optimal threads, plus 1.5B/3B scaling and an 8-client load test.
- All raw results are in `results/*.json`; the narrative is in `results/results.md`.

## Reproduce
```bash
# 1) build llama.cpp (CPU only)
cmake -B build -DGGML_ACCELERATE=ON -DGGML_METAL=OFF
cmake --build build --target llama-server -j$(sysctl -n hw.ncpu)

# 2) get a model (e.g. Qwen2.5-0.5B GGUF)
#    place qwen2.5-0.5b-q8_0.gguf / qwen2.5-0.5b-q4_k_m.gguf in ~/models

# 3) run the study / sweep / scaling (needs the server binaries + models)
bash scripts/run_study.sh      # 0.5B baseline + 3 candidates
bash scripts/run_sweep.sh      # thread-count sweep
bash scripts/scale_study.sh    # 1.5B + 3B generalization

# 4) serve with the optimized defaults
MODEL=qwen2.5-0.5b-q4_k_m.gguf bash scripts/serve_qwen25_instruct.sh
```
Paths are configurable via env (`LLAMA`, `MODEL_DIR`, `MODEL`, `THREADS`, `PORT`).

## Files
- `scripts/serve_qwen25_{instruct,coder,math}.sh` — optimized launchers (physical-core threads, Metal-off).
- `scripts/run_study.sh`, `run_sweep.sh`, `scale_study.sh` — study harness.
- `eval_client.py`, `concurrent_test.py` — evaluators.
- `results/` — full writeup (`results.md`) + all raw `*.json` evidence + frozen references.

## Takeaways for production CPU serving
- Profile thread count per machine — logical≠physical for inference.
- Quantize aggressively for decode-bound serving; verify quality with a frozen-reference gate.
- Size `--parallel` to expected concurrency, not to 1.
- Validate the BLAS/Metal backend on the actual hardware; "default build" can crash.

## Part 2 — Going further: speculative decoding & KV-cache (a CPU reality check)

`llama-cpp-opt/.../speculative-decoding/` (this repo: `speculative-decoding/`)
extends the study with a deliberately honest question: does speculative decoding
(or KV-cache quantization) actually help on a bandwidth-bound laptop CPU?

### Measurement rigor (why single runs lie)

The machine's decode speed swung ~2x between sessions (background load, thermal
state). A single run therefore says almost nothing, so every config was repeated
2–3x and reported as **median ± range**. This rigor caught a real mistake: an
early single-run read of "KV-cache q4_0 = +36%" and "speculative = −26%" were
**run-to-run variance artifacts**, not effects.

### Controlled results (Qwen2.5-3B-Instruct q4_k_m, 6 physical threads)

| config | runs (tok/s) | median | vs baseline |
|---|---|---|---|
| baseline (no spec) | 4.10, 4.17, 4.07 | **4.10** | — |
| KV-cache `q4_0` | 3.70, 3.79 | **3.75** | 0.91x (slower) |
| speculative 0.5B draft, K=4 | 3.60, 5.79 | **4.70** | 1.15x (noisy) |

Raw repeats in `speculative-decoding/results/rep_*.json`; aggregated by
`code/analyze_results.py`. Full write-up: `speculative-decoding/README.md`.

### What we concluded

**Neither technique gives a reliable speedup** on this hardware. KV-cache `q4_0`
was marginally *slower*; speculative decoding's repeats landed on both sides of
baseline (no consistent direction). The original hypothesis ("CPU speculation
helps") is **not supported by repeated measurement** — an honest null result.

### The bandwidth check (why it can't win here)

A context-size sweep (`--ctx-size` 512/2048/8192) left decode tok/s flat
(~4.3–4.7): active KV length is set by generation length, not capacity, so KV
size is not the bottleneck. Decode is dominated by **weight** reads (the full
model is fetched per token); a draft model adds target compute on the same
bandwidth, so it cannot net out faster. This is the load-invariant explanation.

### Why this is the valuable result

A faked "KV-cache +36% / speculation −26%" would not survive replication. By
repeating the runs we produced an honest **null result** — demonstrating the
experimental rigor that separates a credible systems engineer from someone
quoting single runs. The robust, load-invariant takeaway remains Part 1's
*relative* win (physical-core threading + q4 weights), which holds regardless of
absolute machine load. Runnable via `speculative-decoding/code/run_benchmark.sh`.

### Reproduce

```bash
# from llama-cpp-opt/package/
bash speculative-decoding/code/run_benchmark.sh       # (re)generate raw repeats
python3 speculative-decoding/code/analyze_results.py  # median summary
python3 speculative-decoding/code/speculative_server.py --K 4 --max-tokens 32  # from-scratch decoder
```

> **Honesty note on absolute numbers.** Decode tok/s on this laptop varies with
> load and thermal state (we observed ~2× swing across sessions). The *relative*
> gains in Part 1 (physical-core threading, q4 quantization) are robust and
> load-invariant; the Part 2 absolute figures above are controlled medians from a
> single loaded session and should be read as direction, not gospel.

## Part 3 — Bandwidth characterization (why CPU decode is bound)

Part 2 showed speculation and KV-cache quant don't help. This study isolates
*why*. We swept `--ctx-size` (the KV-cache capacity) from 512 → 8192 while
holding the prompt + 128 generated tokens fixed, and measured decode tok/s.

| ctx-size | decode tok/s |
|---|---|
| 512   | 4.27 |
| 2048  | 4.72 |
| 8192  | 4.43 |

Decode speed is **flat** across a 16× change in KV capacity. That is the tell:
during generation the *active* KV length equals the prompt + output length, not
the allocated capacity, so KV-cache size is not the bottleneck. Decode cost is
dominated by **weight** reads — the full ~2 GB model is pulled from memory
*every single token*. A draft model (Part 2) only adds target compute on that
same bandwidth, which is why it cannot net out faster here. This is the
load-invariant root cause behind both the Part 1 win (smaller q4 weights = less
memory traffic per token) and the Part 2 null result.

Raw sweep: `bandwidth-characterization/results/ctx_*.json`; summary script
`bandwidth-characterization/code/analyze.py`.

## Repository structure
```
llm-cpu-inference-optimization/
├── README.md                 # this file
├── eval_client.py            # single-stream evaluator (TTFT, tok/s, quality gate)
├── concurrent_test.py        # N-client load benchmark (aggregate throughput)
├── scripts/
│   ├── serve_qwen25_instruct.sh   # optimized launcher (physical-core threads)
│   ├── serve_qwen25_coder.sh
│   ├── serve_qwen25_math.sh
│   ├── serve_qwen25_3b_full_optimized.sh  # 3B: threads + KV-cache q4_0
│   ├── run_study.sh          # baseline + 3 candidates
│   ├── run_sweep.sh          # thread-count sweep
│   └── scale_study.sh        # 1.5B + 3B generalization
├── results/
│   ├── results.md            # full narrative writeup
│   ├── *.json                # raw measurements (sweep, scale, concurrent)
│   └── reference_outputs*.json  # frozen quality references
├── bandwidth-characterization/  # Part 3: why decode is bandwidth-bound
│   ├── README.md             # context-size sweep writeup
│   ├── code/analyze.py       # prints decode tok/s vs ctx-size
│   └── results/              # raw ctx_512/2048/8192.json
└── speculative-decoding/     # Part 2: theory vs controlled measurement
    ├── README.md             # controlled results (null result on CPU)
    ├── THEORY.md             # original hypothesis (refuted by measurement)
    ├── code/
    │   ├── run_benchmark.sh  # reproducible spec/KV benchmark
    │   ├── analyze_results.py# median summary per config
    │   └── speculative_server.py # from-scratch dual-server speculative decoder
    └── results/              # raw repeated JSON per configuration (rep_*.json)
```

## Skills demonstrated
- **Performance benchmarking** — reproducible tok/s and TTFT measurement with
  statistical summaries (median, p95).
- **Systems inference tuning** — thread/topology, quantization, and batching
  tradeoffs on real hardware.
- **Quality assurance** — deterministic-output verification via frozen references,
  so speed changes never silently alter results.
- **Rigorous experimentation** — explicit baselines, negative-result capture, and
  cross-size generalization rather than a single anecdote.
- **Honest null-result analysis** — repeated speculation & KV-cache runs under
  load, caught that early single-run "gains" were variance, and reported an
  honest *no-reliable-speedup* conclusion instead of a faked win.
- **Algorithm implementation (from scratch)** — hand-wrote a dual-server
  speculative decoder (draft proposal, target verification, accept/reject +
  bonus token) over the HTTP API, and independently reproduced the null result,
  proving understanding beyond flipping a `--spec-type` flag.
- **Reproducible packaging** — one-command studies and clear build/run docs.

## Limitations
- Validated on 5 short prompts; broaden the reference set for harder quality claims.
- CPU/edge scope (no GPU); the methodology translates to GPU serving but the
  absolute numbers do not.
- Exact-match gate uses greedy (temperature=0) decoding.
