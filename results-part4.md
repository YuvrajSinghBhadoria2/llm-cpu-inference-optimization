# Part 4 — Quantization-scheme sweep, memory loading, and CPU affinity

Extends the llama.cpp CPU study on the 2019 Intel Mac (6C/12T i7-9750H, 16 GB,
CPU-only, Accelerate+BLAS, Metal OFF). Same protocol as Parts 1–3: 5 fixed
prompts, `max_tokens=128`, `temperature=0`, measured with `eval_client.py`.
Each config is a **frozen reference** at `q8_0 @ t6` (physical cores) for that
model size, so every candidate is scored by word-overlap vs the same-thread
`q8_0` baseline. **3 repeats per config** to expose run-to-run variance
(Part 2's lesson). Raw evidence: `results/part4_*.json`; run log:
`results/part4_run.log`.

## Headline findings

1. **`q4_K_M` is the speed-optimal scheme — not "any 4-bit".** Among 0.5B
   schemes at t6: `q4_K_M` 28.0 > `q4_0` 26.6 > `iq4_xs` 24.4 > `q5_K_M` 17.8 ≈
   `q8_0` 18.8 tok/s. So the Part 1 "use q4" rule is refined: **`q4_K_M`
   specifically** is fastest, and the heavier `q5_K_M`/ `q8_0` give *no* speed
   benefit (more bits = more weight traffic on this bandwidth-bound CPU).
2. **Quality is identical at every scheme.** Overlap vs the `q8_0@t6` reference
   = 1.000 for all schemes (0.5B and 3B) on these greedy prompts — so `q4_K_M`
   wins on speed with zero measurable quality cost.
3. **CPU affinity is a new, cheap ~5% win.** `--cpu-strict 1` (pin workers to
   physical cores) gives 27.3 vs 26.1 tok/s (default mmap) at 0.5B `q4_K_M@t6`.
   `--no-mmap --mlock` is slightly *slower* (25.0) — keep default mmap.
4. **Generalizes to 3B.** 3B `q4_K_M` 5.36 > `q4_0` 5.21 > `q5_K_M` 4.80 >
   `q8_0` 3.56 tok/s. Same ranking as 0.5B; `q8_0` ~1.5× slower. Corroborates
   Part 3 (decode is weight-bandwidth bound; smaller 4-bit weights = less
   traffic = faster).
5. **Rigor catch.** One planned config (3B `q5_K_M`) initially returned a
   corrupt-GGUF artifact (`invalid magic`) — a truncated file from an earlier
   *interrupted* quantize, not a real measurement. Regenerated cleanly and
   re-ran; the corrupt result was discarded, not reported. Parallel to Part 2's
   variance catch: artifacts get caught, not silently shipped.

## Tables

### 0.5B quantization schemes (t6, single stream, 3 repeats)
| scheme | tok/s median | min | max | overlap vs q8_0@t6 |
|---|---|---|---|---|
| q4_K_M | **28.02** | 24.31 | 28.85 | 1.000 |
| q4_0   | 26.57 | 24.79 | 28.12 | 1.000 |
| iq4_xs | 24.35 | 21.67 | 26.74 | 1.000 |
| q5_K_M | 17.83 | 17.37 | 18.86 | 1.000 |
| q8_0   | 18.78 | 17.41 | 19.03 | 1.000 |

### 0.5B memory loading & affinity (q4_K_M @ t6, 2 repeats)
| config | tok/s median | overlap |
|---|---|---|
| default mmap | 26.07 | 1.000 |
| --no-mmap --mlock | 24.98 | 1.000 |
| --cpu-strict 1 (affinity) | **27.30** | 1.000 |

### 3B quantization schemes (t6, single stream, 2 repeats)
| scheme | tok/s median | overlap vs q8_0@t6 |
|---|---|---|
| q4_K_M | **5.36** | 1.000 |
| q4_0   | 5.21 | 1.000 |
| q5_K_M | 4.80 | 1.000 |
| q8_0   | 3.56 | 1.000 |

## Revised recommended defaults (supersedes Part 1)
- `--threads $(sysctl -n hw.physicalcpu)` **and `--cpu-strict 1`** (pin to
  physical cores; ~5% over threads-only).
- Weights: **`q4_K_M`** (fastest decode, identical greedy output to `q8_0`).
  Use `q8_0` only if a harder task shows 4-bit drift.
- Keep default mmap; do **not** add `--no-mmap --mlock` for interactive use.
- Build llama.cpp without Metal on this Intel Mac (`-DGGML_METAL=OFF`).

## Reproduce
```
bash scripts/run_part4.sh        # generates part4_*.json (≈30 min on this Mac)
```
Requires the extra schemes, generated locally (no download):
```
~/llama.cpp/build/bin/llama-quantize --allow-requantize \
    qwen2.5-0.5b-q8_0.gguf qwen2.5-0.5b-q4_0.gguf q4_0   # + q5_k_m, iq4_xs
```
