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

## What actually moved the needle
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
│   ├── run_study.sh          # baseline + 3 candidates
│   ├── run_sweep.sh          # thread-count sweep
│   └── scale_study.sh        # 1.5B + 3B generalization
└── results/
    ├── results.md            # full narrative writeup
    ├── *.json                # raw measurements (sweep, scale, concurrent)
    └── reference_outputs*.json  # frozen quality references
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
- **Reproducible packaging** — one-command studies and clear build/run docs.

## Limitations
- Validated on 5 short prompts; broaden the reference set for harder quality claims.
- CPU/edge scope (no GPU); the methodology translates to GPU serving but the
  absolute numbers do not.
- Exact-match gate uses greedy (temperature=0) decoding.
