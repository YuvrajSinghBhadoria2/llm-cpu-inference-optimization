# Speculative Decoding & KV-Cache Quantization on CPU — a controlled reality check

This folder extends the main study with a deliberately honest question: **does
speculative decoding (or KV-cache quantization) actually help on a
bandwidth-bound laptop CPU?** The answer, after *repeated* measurement, is no —
and catching that is the point of this document.

## Why single runs lie (measurement rigor)

The decode speed of this machine swung ~2x between sessions (background load,
thermal throttling after hours of benchmarking). A single run therefore says
almost nothing. Every config below was repeated 2–3x and reported as
**median ± range**.

This rigor caught a real mistake: our first single-run numbers read
`KV-cache q4_0 = 10.56 tok/s (+36%)` and `speculative 0.5B = 5.75 tok/s (-26%)`.
Those were **run-to-run variance artifacts**, not effects. Repeating the runs
disappeared the signal entirely.

## Controlled results (Qwen2.5-3B-Instruct q4_k_m, 6 physical threads)

| config | runs (tok/s) | median | vs baseline |
|---|---|---|---|
| baseline (no spec) | 4.10, 4.17, 4.07 | **4.10** | — |
| KV-cache `q4_0` | 3.70, 3.79 | **3.75** | 0.91x (slower) |
| speculative 0.5B draft, K=4 | 3.60, 5.79 | **4.70** | 1.15x (noisy) |

Raw repeats: `results/rep_baseline_*.json`, `results/rep_kvq4_*.json`,
`results/rep_spec05_*.json`. Aggregated by `code/analyze_results.py`.

## What we concluded

- **Neither technique gives a reliable speedup** on this hardware.
- KV-cache `q4_0` was marginally *slower* (smaller KV cache did not reduce the
  dominant cost).
- Speculative decoding's two repeats landed on *both* sides of baseline
  (3.60 and 5.79) — no consistent direction. The original hypothesis ("CPU
  speculation helps") is **not supported by repeated measurement**.

## The bandwidth check (why speculation can't win here)

A context-size sweep (`--ctx-size` 512 / 2048 / 8192) left decode tok/s flat
(~4.3–4.7). Active KV length during generation is set by the prompt + 128 output
tokens, not by capacity, so KV-cache size is not the bottleneck. Decode cost is
dominated by **weight** reads: the full ~2 GB model is fetched from memory
*every single token*. A draft model adds target-side compute per emitted token
on the same bandwidth, so it cannot net out faster here. This is the correct,
load-invariant explanation (the earlier "KV traffic" hand-wave was wrong).

### From-scratch implementation (Project 4): we built the decoder ourselves

To prove we understand the algorithm — not just llama.cpp's `--spec-type` — we
wrote `code/speculative_server.py`: a hand-rolled speculative decoder that starts
a draft (0.5B) and a target (3B) llama-server, proposes `K` tokens from the
draft, verifies them against the target's greedy tokens, and accepts the longest
matching prefix plus the target's corrected (or bonus) token. It runs entirely
over the OpenAI-compatible HTTP API.

Measured on the same hardware (K=4, 32 tokens, 6 threads):

| mode | tok/s | ratio |
|---|---|---|
| baseline (target only, single stream) | 7.27 | 1.00x |
| from-scratch speculative (draft+target) | 3.99 | 0.55x |

The hand-built decoder is *slower*, for two honest reasons: (1) orchestrating the
loop over HTTP re-prefills the whole prefix on the target every step, so it does
strictly more work than one batched call; (2) even an ideal single-pass
speculative step evaluates `K+1` target positions to emit `K+1` tokens, which on a
weight-bandwidth-bound CPU cannot beat one token per target forward pass — the
same conclusion reached from the controlled llama.cpp measurements above.

This is the point: **building it ourselves independently reproduces the null
result.** The algorithm is correct (it generated coherent text), but it does not
help on this hardware. Evidence: `results/from_scratch_spec.json`.

## Why this is the valuable result

A faked "KV-cache +36% / speculation −26%" would not survive replication. By
repeating the runs we produced an honest **null result** — and that demonstrates
the experimental rigor that separates a credible systems engineer from someone
quoting single runs. The robust, load-invariant takeaway remains Project 1's
*relative* win (physical-core threading + q4 weights), which holds regardless of
absolute machine load.

## Reproduction

```bash
# from llama-cpp-opt/package/
bash speculative-decoding/code/run_benchmark.sh     # (re)generate raw repeats
python3 speculative-decoding/code/analyze_results.py
```
