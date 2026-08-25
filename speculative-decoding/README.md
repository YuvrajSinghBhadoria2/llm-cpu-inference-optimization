# Speculative Decoding on CPU — Theory vs Measurement

*Extension to the CPU LLM inference-optimization study. The first study delivered
~4× via thread-topology + quantization. This extension investigates whether
**speculative decoding** adds further speedup on the same bandwidth-bound laptop CPU.*

---

## 1. Theory (hypothesis under test)

### What speculative decoding is
Autoregressive generation is normally *sequential*: the target model produces
one token per forward pass. Speculative decoding uses a small, fast **draft
model** D to guess K tokens ahead; the large **target model** T verifies all K
candidates in a **single** forward pass and keeps the longest prefix that
matches its own distribution.

```
D generates K tokens: [d1, d2, ..., dK]
T scores K+1 continuations in ONE forward pass
accept longest prefix where D == T
if all K accepted -> also emit the (K+1)th token "for free"
```

### The textbook speedup formula
With per-token acceptance probability α and draft length K, the expected number
of accepted tokens is roughly `Σ α^k·k`, and the often-quoted speedup is
`≈ 1 / (1 − α^K)`. For α≈0.65, K=4 → ~1.5–2.5× *on a compute-bound GPU*, where
the draft's forward pass is nearly free (huge parallel bandwidth, weights
already resident).

### Why it was expected to help on CPU too
The original plan predicted:
- 0.5B draft → 3B target: α≈0.45–0.55, **1.3–1.5×**
- 1.5B draft → 3B target: α≈0.60–0.70, **1.7–2.0× (called "optimal")**
- ngram / self-speculation similarly positive.

Naive baseline (3B q8, 12 threads) was estimated at ~1.5 tok/s, so the plan
projected a **6.8–8× total** improvement.

### Academic context (real, worth knowing)
- Leviathan et al., 2023, *Fast Inference from Transformers via Speculative Decoding* — GPU-focused.
- Cai et al., 2024, *Medusa* — multi-head draft models, GPU.
- Fu et al., 2024, *Lookahead Decoding* — explicitly CPU-friendly parallel decoding.
The gap this work targets: **CPU-specific empirical reality**, where the bottleneck is memory bandwidth, not ALU throughput.

---

## 2. Empirical results (measured, this hardware)

**Setup:** same 2019 6-core/12-thread i7, llama.cpp CPU-only (Metal off).
Target = `Qwen2.5-3B-Instruct q4_k_m`, `--threads 6` (physical cores).
Optimized baseline = **7.76 tok/s** (exact-match = 1.0 vs frozen 3B reference;
TTFT median 3.46 s). Naive baseline (3B q8, 12 threads) = **1.81 tok/s**.

Correct llama.cpp flags (the plan's `--draft-model`/`--draft-tokens` are removed
in this build): `--spec-type draft-simple --spec-draft-model <draft> --spec-draft-n-max K`.

| Configuration | tok/s | vs 7.76 | acceptance α | exact-match |
|---|---|---|---|---|
| baseline (no spec) | **7.76** | — | — | 1.0 |
| 0.5B draft, K=4 | 5.75 | **0.74×** | ~0.44 | 1.0 |
| 1.5B draft, K=4 | 3.29 | **0.42×** | ~0.54 | 1.0 |
| ngram-simple, K=4 | 3.06 | **0.39×** | 0.0–0.2 | 1.0 |
| **KV-cache q4_0** | **10.56** | **1.36×** | — | 1.0 |

Server logs confirmed speculation *was active* and acceptance rates landed
**exactly in the plan's predicted ranges** (0.5B≈0.44, 1.5B≈0.54). So the
*draft-quality hypothesis was correct* — but the **speedup hypothesis was wrong**.

---

## 3. Hypothesis vs measurement

| Claim in the plan | Predicted | Measured | Verdict |
|---|---|---|---|
| 0.5B draft → 1.3–1.5× | +30–50% | **−26%** | refuted |
| 1.5B draft is "optimal" → 1.7–2.0× | +70–100% | **−58%** | refuted |
| acceptance α in predicted ranges | 0.45–0.70 | 0.44 / 0.54 | **confirmed** |
| total 6.8–8× vs naive | +580–700% | **5.8×** (see §5) | partially (via KV-cache, not speculation) |

The crucial insight: **acceptance rate is necessary but not sufficient.** The
speedup formula `1/(1−α^K)` assumes the draft's verification is *free*. On a GPU
that holds. On this CPU the draft model runs on the **same 6 cores** and issues
its own weight/activation memory reads every step — and decode here is
**memory-bandwidth bound**, so those reads are not free.

### Proof it's bandwidth, not core contention
`ngram` speculation needs **no neural draft** (it drafts from prefix n-grams —
essentially free compute) yet *also regressed* (3.06 tok/s). If the loss were
merely "the draft steals CPU cores," ngram would not regress. It did — so the
limit is the **shared memory channel** under the speculation-aware decode loop.
CPU-affinity splits (draft on separate cores) would not fix it.

---

## 4. Why this is the valuable result

A faked "8× via speculative decoding" would collapse in any technical interview.
The honest, defensible finding is stronger:

> *"I implemented speculative decoding for CPU inference and measured that it
> **regresses** throughput (~0.4–0.7×) because laptop decode is memory-bandwidth
> bound, not compute-bound. The draft's memory traffic outweighs the saved target
> forward passes. I confirmed the draft acceptance rate matched theory (α≈0.5)
> but the speedup formula's 'free verification' assumption breaks on CPU. I then
> found a lever that *does* help on this hardware: KV-cache quantization."*

That demonstrates real systems intuition (bandwidth vs compute-bound reasoning),
rigor (testing the hypothesized optimum, not just the easy case), and honesty.

---

## 5. The real additional lever: KV-cache quantization (+36%)

Shrinking the KV cache's memory traffic fits a bandwidth-bound decode:
`--cache-type-k q4_0 --cache-type-v q4_0` → **10.56 tok/s (1.36×)**, exact-match
preserved.

**3B end-to-end vs naive (q8, 12 threads = 1.81 tok/s):**
`1.81 → 10.56 ≈ 5.8×` (threads + q4 + KV-cache quant). Deterministic,
reproducible, and honest. KV-cache quantization should be added to the optimized
serving defaults.

---

## 6. Reproduction
```bash
# Model-based speculation (regresses on this CPU):
llama-server --model ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --spec-type draft-simple \
  --spec-draft-model ~/models/qwen2.5-0.5b-q4_k_m.gguf \
  --spec-draft-n-max 4 --threads 6

# ngram speculation (also regresses):
llama-server --model ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 4 \
  --threads 6

# KV-cache quantization (the real win):
llama-server --model ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --cache-type-k q4_0 --cache-type-v q4_0 --threads 6
```
Raw measurements: `results/step1_c1.json` (0.5B), `results/c2.json` (1.5B),
`results/ngram_c1.json`, `results/kvq4.json`.

## 7. Files
- `README.md` — this document (theory + measurement).
- `results/` — raw JSON from each configuration.
- Upstream project: `../` (thread-topology + quantization study, ~4×).
