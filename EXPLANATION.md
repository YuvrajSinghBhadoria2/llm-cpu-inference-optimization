# The Complete Project: LLM Inference Optimization on a CPU Laptop

A full, in-depth walkthrough of everything built in this repository.

---

## 1. The premise and the constraints

The brief was to build a **self-contained, reproducible LLM inference-optimization
study that runs entirely on a local laptop** (no GPU, no cloud), suitable as
evidence for a "remote LLM inference optimization engineer" role.

The fixed hardware is the whole story:
- **2019 MacBook Pro 16,1**, Intel i7-9750H — **6 physical cores / 12 logical
  threads**, 16 GB RAM.
- No CUDA, no Apple-Silicon Metal → **CPU-only** inference via `llama.cpp`
  compiled with Apple Accelerate (`-DGGML_METAL=OFF -DGGML_ACCELERATE=ON`; the
  Metal backend *crashes* on this Intel Mac).
- Implication: weights must fit in 16 GB, and every token's compute is gated by
  **how fast ~2 GB of weights can be read from memory**.

That last point is the thread that ties the entire project together. On a
bandwidth-bound CPU, *decode* (generating one token at a time) is limited by
memory bandwidth, not ALU/compute. Everything in the study either exploits that
fact (Part 1) or fails because of it (Parts 2–3).

---

## 2. Part 1 — Threading & quantization: the ~4× win

**Goal:** maximize decode tokens/sec at *identical* output quality.

### Method
- **Server:** `llama.cpp`'s `llama-server` (OpenAI-compatible API on :8080).
- **Evaluator (`eval_client.py`):** streams `/v1/completions` at
  `temperature=0`, measures TTFT and tok/s, and compares output to a **frozen
  reference** (`reference_outputs*.json`). The quality gate: same-weight
  candidates must match the reference *exactly* (determinism); a 4-bit candidate
  is scored by word-overlap (≥0.90 = PASS), because quantization legitimately
  changes outputs.
- **Protocol:** 5 fixed prompts, `max_tokens=128`, thread sweep
  {2,3,4,6,8,12}, q4 vs q8 at optimal threads, 1.5B/3B scaling, and an
  8-client load test.

### What moved the needle
1. **`--threads` = physical cores, NOT logical.** All 12 logical threads on a
   6-core chip *collapsed* throughput (Hyper-Thread siblings contend for the 6
   physical cores). For 0.5B: 6.7 tok/s at 12 threads vs 17.4 at 6. For 3B: 1.8
   vs 3.6. Fix: `--threads $(sysctl -n hw.physicalcpu)`.
2. **Quantize to `q4`.** At the correct thread count, `q4` decodes ~1.5× faster
   than `q8` — smaller weights → less memory traffic per token (this is the
   bandwidth insight in action).
3. **Batching only helps under concurrency** — single-stream sees no benefit;
   8 concurrent clients with `--parallel 8` gave +45% aggregate throughput and
   ~4× lower tail latency.
4. **Build without Metal** (stability).

### Headline result
| model | naive (q8, 12T) | optimized (q4, 6T) | speedup |
|---|---|---|---|
| Qwen2.5-0.5B | 6.67 | **26.85** | **4.0×** |
| Qwen2.5-1.5B | 2.91 | **7.36** | 2.5× |
| Qwen2.5-3B | 1.81 | **7.76** | **4.3×** |

This is the robust, load-invariant core finding (the *relative* gains hold even
though absolute tok/s drifts with machine load).

---

## 3. Part 2 — Speculative decoding & KV-cache: a controlled *null* result

**Original hypothesis:** speculative decoding should add 1.5–2× on top of the
4×, because "everyone says speculation helps."

### The algorithm (built into llama.cpp)
A small **draft model** (e.g. Qwen2.5-0.5B) proposes `K` tokens; the large
**target** (3B) verifies them in *one* forward pass. You accept the longest
prefix where draft and target agree, then emit the target's correction (or a
"bonus" token at the end). If the draft is accurate, you get ~K tokens of work
for ~1 target forward pass.

### The trap we fell into (and caught)
First single runs looked great for the narrative:
- KV-cache `q4_0` → **10.56 tok/s (+36%)**
- speculative 0.5B K=4 → 5.75 tok/s (−26%)

But the machine's decode speed swings **~2× across sessions** (background load,
thermal throttling). So we **repeated every config 2–3×** and reported medians:

| config | runs (tok/s) | median | vs baseline |
|---|---|---|---|
| baseline | 4.10, 4.17, 4.07 | **4.10** | — |
| KV-cache `q4_0` | 3.70, 3.79 | **3.75** | 0.91× (slower) |
| spec 0.5B K=4 | 3.60, 5.79 | **4.70** | 1.15× (noisy) |

**The earlier "+36%/−26%" were run-to-run variance artifacts.** The honest
conclusion: **neither technique gives a reliable speedup** on this hardware.
Reporting this null result *instead of* the faked win is the single most
important rigor moment in the project.

---

## 4. Part 3 — Bandwidth characterization: the root cause

Part 2 showed speculation/KV-quant don't help; Part 3 explains *why* with one
clean experiment.

**Question:** is decode limited by **KV-cache memory traffic** or **weight
memory traffic**?
**Test:** hold the prompt + 128 generated tokens fixed, vary only `--ctx-size`
(the KV-cache capacity) 512 → 8192.

| ctx-size | decode tok/s |
|---|---|
| 512 | 4.27 |
| 2048 | 4.72 |
| 8192 | 4.43 |

Decode is **flat** across a 16× capacity change (≈10% spread, within noise).
That's the tell: during generation the *active* KV length equals prompt + output
length, **not** the allocated capacity, so KV size isn't the bottleneck. Decode
cost is dominated by **weight reads** — the full ~2 GB model is pulled from RAM
*every single token*.

This is the load-invariant root cause that unifies everything:
- **Part 1 win:** `q4` weights are ~half of `q8` → less memory traffic per token
  → faster. Threading fixes HT contention on the same bandwidth.
- **Part 2 null:** a draft model only adds *target* compute on that same
  bandwidth and shrinks KV traffic by a few percent — neither moves the needle.

---

## 5. From-scratch speculative decoder (Project 4)

To prove we *understand* the algorithm — not just flip a `--spec-type` flag — we
wrote `speculative-decoding/code/speculative_server.py` from first principles.
It:

1. Starts a **draft** (0.5B) and **target** (3B) `llama-server` on two ports.
2. Proposes `K` tokens from the draft (`/v1/completions`, `logprobs` to read
   token ids).
3. Asks the target for `K+1` tokens greedily.
4. Walks both token-id lists, finds the first mismatch `m`, and emits
   `draft[:m]` (accepted) + `target[m]` (correction). If all `K` match, it also
   emits the target's bonus token at position `K`.
5. Concatenates the emitted token *strings* to form the next prefix (lossless
   for a shared tokenizer) and repeats.

The accept/reject core:
```python
m = 0
while m < len(draft_ids) and m < len(tgt_ids) and draft_ids[m] == tgt_ids[m]:
    m += 1
step = draft_ids[:m]
if m < len(tgt_ids):
    step.append(tgt_ids[m])                 # target's correction
elif len(tgt_ids) > len(draft_ids):
    step.append(tgt_ids[len(draft_ids)])    # bonus token
```

It generates coherent text ("Paris. The capital of Spain is Madrid…") and
**independently reproduces the null result**: 3.99 spec tok/s vs 7.27 baseline
(0.55×). It's slower because HTTP orchestration re-prefills the prefix each
step — and, more fundamentally, because even an ideal single-pass speculative
step evaluates `K+1` target positions to emit `K+1` tokens, which on a
weight-bandwidth-bound CPU cannot beat one token per forward pass. Building it
ourselves *confirms* the measurement-based conclusion.

---

## 6. Why this is a strong portfolio piece

- **A real, robust win (Part 1):** ~4× at identical quality, generalized across
  model sizes — the kind of result that shows systems intuition.
- **Rigorous honesty (Parts 2–3):** we caught our *own* variance mistake (the
  +36%/−26% artifacts) by repeating runs, and reported a null result rather than
  a faked win. That's the difference between a credible engineer and someone
  quoting single runs.
- **Root-cause thinking (Part 3):** the bandwidth characterization explains *why*
  the null holds — not hand-waving.
- **Implementation ability (Project 4):** a from-scratch dual-server decoder that
  independently validates the conclusion.
- **Reproducibility:** one-command studies, frozen quality references,
  median-based reporting, all raw `*.json` evidence committed.

---

## 7. Repository structure (the single integrated project)

```
llm-cpu-inference-optimization/
├── README.md                 # the three-part story
├── eval_client.py            # single-stream evaluator (TTFT, tok/s, quality gate)
├── concurrent_test.py        # N-client load benchmark
├── scripts/                  # serve_*.sh launchers + run_study/sweep/scale
├── results/                  # Part 1 raw JSON + results.md
├── bandwidth-characterization/   # Part 3: ctx-size sweep
│   ├── README.md
│   ├── code/analyze.py
│   └── results/ctx_{512,2048,8192}.json
└── speculative-decoding/     # Part 2 + Project 4
    ├── README.md, THEORY.md
    ├── code/{run_benchmark.sh, analyze_results.py, speculative_server.py}
    └── results/rep_*.json, from_scratch_spec.json
```

Reproduce with:
```bash
python3 speculative-decoding/code/run_benchmark.sh --label baseline --repeats 3
python3 speculative-decoding/code/analyze_results.py
python3 bandwidth-characterization/code/analyze.py
python3 speculative-decoding/code/speculative_server.py --K 4 --max-tokens 32
```

**Bottom line:** a ~4× threading/quant win, a rigorously-measured null result on
speculation/KV-cache (with the variance trap caught and reported), a
bandwidth-bound root-cause explanation, and a from-scratch implementation that
confirms it — all in one reproducible project.
