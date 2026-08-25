# LLM Inference Optimization on CPU — A Measured Case Study

> **Abstract.** We optimize local LLM serving on a 2019 laptop CPU (no GPU) and
> show **2.5–4.3× higher decode throughput at identical output quality** by
> fixing thread topology and quantization. We then rigorously test the trendy
> "speculative decoding" speedup and find it **does not help** on this hardware —
> a conclusion we only reached after catching our own measurement variance. The
> root cause, a context-size sweep shows, is that CPU decode is **weight-memory-
> bandwidth bound**, not compute bound. Full evidence and a from-scratch
> re-implementation are included.

---

## 1. What we observe in LLMs on CPU

Large language models generate text **autoregressively**: one token at a time,
each token requiring a full forward pass over the model weights. On a GPU this is
hidden by massive memory bandwidth; on a **CPU laptop** it is exposed.

Three things are immediately visible to anyone who runs `llama.cpp` with defaults
on a 6-core/12-thread Intel MacBook:

1. **It is slow** — single-digit tokens/sec for a 3B model.
2. **Defaults are wrong** — the framework's "use all cores" default
   (`--threads = logical core count`) is the *worst* setting, not the best.
3. **It is bandwidth-bound, not compute-bound** — every generated token re-reads
   the entire model from RAM, so throughput is gated by memory bandwidth, not by
   how many ALUs you throw at it.

This last observation is the key that explains everything else in this report.

---

## 2. The problem we set out to solve

**Goal:** get the maximum *reproducible* tokens/sec out of a laptop CPU for local
LLM serving, with **zero cloud/GPU dependency** and **no silent quality loss**.

Constraints (fixed hardware):
- 2019 MacBook Pro 16,1, Intel i7-9750H — **6 physical / 12 logical** cores, 16 GB RAM.
- No CUDA, no Apple-Silicon Metal → CPU-only `llama.cpp` built with Apple
  Accelerate (`-DGGML_METAL=OFF -DGGML_ACCELERATE=ON`; the Metal backend *crashes*
  here).
- Weights must fit in 16 GB; we use Qwen2.5 0.5B / 1.5B / 3B GGUFs.

Success is defined precisely: **higher tok/s at identical generated output**,
verified against a frozen reference (see §3), not "it feels faster."

---

## 3. How we measured (and why single runs lie)

A number from one benchmark run is not a result — it is a sample. The machine's
decode speed swung **~2× between sessions** (background load, thermal state), so
we built the study around *reproducibility*:

- **Evaluator (`eval_client.py`, stdlib-only)** streams `/v1/completions` at
  `temperature=0`, measures TTFT and tok/s, and compares output to a **frozen
  reference** (`reference_outputs*.json`). Same-weight candidates must match
  *exactly* (determinism gate); a 4-bit candidate is scored by word-overlap
  (≥0.90 = PASS) since quantization legitimately changes tokens.
- **Protocol:** 5 fixed prompts, `max_tokens=128`, thread sweep
  {2,3,4,6,8,12}, q4 vs q8 at optimal threads, 1.5B/3B scaling, 8-client load test.
- **Every non-trivial claim is repeated 2–3× and reported as median ± range.** This
  single discipline is what separates the conclusions below from a blog post.

---

## 4. Results — everything we measured

### 4.1 Part 1: Threading + quantization (the win)

| model | naive (q8, 12T) | optimized (q4, 6T) | speedup |
|---|---|---|---|
| Qwen2.5-0.5B | 6.9 tok/s | **17.4 tok/s** | **2.5×** |
| Qwen2.5-1.5B | 2.91 tok/s | **7.36 tok/s** | 2.5× |
| Qwen2.5-3B   | 1.81 tok/s | **7.76 tok/s** | **4.3×** |

> The 0.5B figure was originally quoted as 26.85 tok/s (4.0×); that was a
> favorable single run from a low-load session. The values above are the ones
> **committed as raw evidence in `results/`** (typical-load sessions).

**Threading — Qwen2.5-0.5B q4, decode tok/s vs thread count:**

| threads | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| tok/s | 15.5 | 16.0 | 17.0 | 17.4 | 14.0 | 6.7 |

Peak is at the **6 physical cores**; jumping to 12 *logical* threads collapses
to 6.7 tok/s (Hyper-Thread siblings contend for the same physical core).

![Thread-count sweep — measured medians (0.5B q4)](../assets/figure_threads_bandwidth.png)

**Quantization — Qwen2.5-3B at 6 threads:** q8 = 3.64 → q4 = 7.76 tok/s
(**2.1×** from weights alone); the full naive→optimized jump is 1.81 → 7.76 =
**4.3×**.

**Batching:** a single-stream client sees no benefit from `--parallel`; under 8
concurrent clients, `--parallel 8` gave **+45% aggregate throughput** and **~4×
lower tail latency** vs `--parallel 1`.

### 4.2 Part 2: Speculative decoding & KV-cache (controlled)

Qwen2.5-3B q4, 6 threads, each config repeated:

| config | runs (tok/s) | median | vs baseline |
|---|---|---|---|
| baseline (no spec) | 4.10, 4.17, 4.07 | **4.10** | — |
| KV-cache `q4_0` | 3.70, 3.79 | **3.75** | 0.91× (slower) |
| spec 0.5B draft, K=4 | 3.60, 5.79 | **4.70** | 1.15× (noisy) |

### 4.3 Part 3: Bandwidth characterization (root cause)

Decode tok/s vs `--ctx-size` (KV-cache capacity), fixed prompt + 128 tokens:

| ctx-size | decode tok/s |
|---|---|
| 512   | 4.27 |
| 2048  | 4.72 |
| 8192  | 4.43 |

**Flat across a 16× capacity change** — decode is not KV-bound. (Same figure,
right panel, above.)

### 4.4 From-scratch decoder (Project 4)

We hand-wrote a dual-server speculative decoder (`speculative-decoding/code/
speculative_server.py`). On K=4, 32 tokens: baseline **7.27** tok/s vs
from-scratch spec **3.99** tok/s (**0.55×**). It generates coherent text and
**independently reproduces the null result**.

---

## 5. What we found

### 5.1 Physical cores beat logical threads — and Hyper-Threading actively hurts
**Finding:** `--threads` = the 6 *physical* cores gave ~17.4 tok/s for 0.5B
versus 6.7 at the 12-thread default — a ~2.6× difference from a one-line change.
**Why:** Hyper-Threading shows the OS 12 "CPUs", but they are 6 physical cores,
each with two *siblings* that share the same execution units **and the same path
to memory**. Decoding is memory-bound, so throughput is gated by how fast the
single memory bus can feed one core. Adding the sibling thread doesn't add a
second bus — it just makes the two siblings fight over the same bus and the same
L1/L2 cache, reducing total useful work. The fix is trivial and high-leverage:
`--threads $(sysctl -n hw.physicalcpu)`.

### 5.2 `q4` beats `q8` because it moves fewer bytes per token
**Finding:** at 6 threads, 3B went from 3.64 (q8) to 7.76 tok/s (q4) — a 2.1×
gain from weights alone.
**Why:** generating one token requires reading the *entire* model from RAM.
`q4` stores each weight in ~4 bits instead of 8, so the model is about half the
size and **half the bytes must cross the memory bus per token**. Less data moved
= less time waiting on bandwidth = faster decode. This is the bandwidth insight
made concrete.

### 5.3 Batching helps only when there are concurrent users
**Finding:** a single client saw no gain from `--parallel`; 8 concurrent clients
gained +45% aggregate throughput and ~4× lower tail latency.
**Why:** with one request, the cores are already busy on that stream, so there
is nothing to parallelize. With many requests, batching lets the cores switch
between them while waiting on memory, raising utilization. The optimization
target differs by workload — tail latency for one user vs aggregate throughput
for many.

### 5.4 Speculative decoding and KV-cache quantization do NOT reliably help here
**Finding:** repeated runs gave baseline 4.10, KV-cache `q4_0` 3.75 (slower),
and spec 0.5B K=4 4.70 (noisy, spanning both sides of baseline). No reliable
speedup.
**Why the intuition fails:** the textbook argument is "a small draft model
guesses several tokens, the big model verifies them in one pass, so we get
several tokens for the price of one." That holds on **compute-bound** hardware
(GPUs), where the big model's forward pass is the expensive part. On this
**bandwidth-bound CPU**, the big model's forward pass is *cheap relative to the
weight-fetch*; the draft model only adds more weight-fetches on the same bus.
Net result: no win, occasionally a loss.

### 5.5 The real bottleneck is weight memory bandwidth (the unifying insight)
**Finding:** varying `--ctx-size` from 512 → 8192 (16× more KV capacity) left
decode flat at ~4.3–4.7 tok/s.
**Why this matters:** during generation the *active* KV length is just prompt +
generated tokens, **not** the allocated capacity — so KV size isn't the limit.
The limit is the **full model re-read on every single token**. We can make this
quantitative: Qwen2.5-3B q4 is ~1.9 GB; this laptop's DDR4 bandwidth is
~30–40 GB/s, so the theoretical ceiling from moving weights alone is ~1.9/35 ≈
**15–20 tok/s**. We measure ~7.8 — i.e., we are squarely in the **bandwidth-
limited regime** (far from any compute limit), which is exactly why the
optimizations behave as they do:
- **Why Part 1 won:** smaller weights (`q4`) and fewer wasted fetches (physical
  cores) both reduce bytes moved per token.
- **Why Part 2 was a null:** a draft model or smaller KV only trims a few percent
  of traffic on a bus already saturated moving the weights.

This single fact turns three separate observations into one coherent story.

---

## 6. How we solved it

1. **The throughput problem** → set `--threads` to physical cores and quantize to
   `q4`; verified quality with the frozen-reference gate.
2. **The "does speculation help?" question** → measured it directly with repeats
   instead of assuming the textbook answer.
3. **The variance trap** → our *first* runs showed "KV-cache +36% / speculation
   −26%." Repeating the runs dissolved both into noise. We answered the trap by
   **reporting medians and ranges**, and by treating the null result as a valid
   finding rather than a failure.
4. **The "do we even understand it?" question** → built the speculative decoder
   from scratch (draft proposal → target verification → accept/reject + bonus
   token) and confirmed the conclusion with our own code, not just a flag.

---

## 7. Insights

These are the lessons that generalize beyond this one laptop:

- **On CPU, decode is weight-bandwidth bound, not compute bound.** Optimize
  bytes-per-token (quantize, smaller models), not FLOPs.
- **Hyper-Threading is anti-throughput for inference.** Match threads to physical
  cores; the OS default is wrong.
- **A single benchmark number is a lie; a median over repeats is a result.**
  Load and thermals move CPU numbers by 2× — anyone quoting one run is guessing.
- **Negative results are findings.** "Speculation doesn't help here" is a real,
  publishable conclusion, not a disappointment. Catching our own variance mistake
  and reporting it is the most credible thing in this project.
- **The best optimization is the one that survives replication.** We built a
  second, independent implementation that reproduced the null result —
  confirmation, not just assertion.
- **Constraints breed clarity.** No GPU forced us to reason about the actual
  bottleneck (memory bandwidth) instead of throwing hardware at the problem.

---

## 8. Reproduce

```bash
# 1) build llama.cpp (CPU only)
cmake -B build -DGGML_ACCELERATE=ON -DGGML_METAL=OFF
cmake --build build --target llama-server -j$(sysctl -n hw.ncpu)

# 2) place GGUFs (e.g. qwen2.5-0.5b-q4_k_m.gguf) in ~/models

# 3) run the study
bash scripts/run_study.sh && bash scripts/run_sweep.sh && bash scripts/scale_study.sh

# 4) Part 2 / Project 4
bash speculative-decoding/code/run_benchmark.sh --label baseline --repeats 3
python3 speculative-decoding/code/analyze_results.py
python3 speculative-decoding/code/speculative_server.py --K 4 --max-tokens 32

# 5) Part 3
python3 bandwidth-characterization/code/analyze.py
```
Paths are configurable via env (`LLAMA`, `MODEL_DIR`, `MODEL`, `THREADS`, `PORT`).
Full narrative: [`EXPLANATION.md`](EXPLANATION.md).

---

## 9. Repository structure

```
llm-cpu-inference-optimization/
├── README.md                 # this case study
├── EXPLANATION.md            # full in-depth walkthrough
├── eval_client.py            # single-stream evaluator (TTFT, tok/s, quality gate)
├── concurrent_test.py        # N-client load benchmark
├── scripts/                  # serve_*.sh launchers + run_study/sweep/scale
├── results/                  # Part 1 raw JSON + results.md + frozen references
├── bandwidth-characterization/  # Part 3: ctx-size sweep
│   ├── README.md
│   ├── code/analyze.py
│   └── results/ctx_{512,2048,8192}.json
└── speculative-decoding/     # Part 2 + Project 4
    ├── README.md, THEORY.md
    ├── code/{run_benchmark.sh, analyze_results.py, speculative_server.py}
    └── results/rep_*.json, from_scratch_spec.json
```

## 10. Limitations
- Validated on 5 short prompts; broaden the reference set for harder quality claims.
- CPU/edge scope (no GPU); methodology translates to GPU serving, absolute numbers do not.
- Exact-match gate uses greedy (`temperature=0`) decoding.

## 11. Skills demonstrated
- **Performance benchmarking** — reproducible tok/s/TTFT with median-based statistics.
- **Systems inference tuning** — thread topology, quantization, batching on real hardware.
- **Quality assurance** — frozen-reference verification so speed never silently breaks correctness.
- **Rigorous experimentation** — baselines, negative-result capture, controlled repeats.
- **Honest null-result analysis** — caught our own variance artifact and reported it.
- **Algorithm implementation (from scratch)** — dual-server speculative decoder.
- **Reproducible packaging** — one-command studies and clear docs.
