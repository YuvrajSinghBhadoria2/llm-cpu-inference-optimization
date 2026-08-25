# LinkedIn Post — LLM CPU Inference Optimization

**Recommended image:** attach `assets/figure_threads_bandwidth.png`. It is a
real figure generated **directly from the committed `results/` data** (measured
medians over repeated runs) — not a hand-made chart:
- **Left panel:** thread-count sweep for 0.5B q4 — decode tok/s peaks at the 6
  physical cores and collapses at 12 logical threads.
- **Right panel:** bandwidth characterization for 3B q4 — decode tok/s stays flat
  across a 16× change in `--ctx-size`, proving the bottleneck is weight memory
  bandwidth, not KV traffic.

This is the same figure embedded in the README/EXPLANATION, so it is a genuine
research figure. It makes the core finding instantly legible while scrolling and
backs the post's credibility (it's our own measured data, not a marketing graphic).

---

I optimized LLM inference on a 2019 laptop CPU — no GPU, no cloud. The most
valuable result wasn't the 4× speedup. It was proving that speedup was fake.

I measured a big win, almost published it, then ran it three more times and
watched the gain dissolve into noise. That failure taught me more than the win —
and it's the most credible part of the work.

The setup: squeeze maximum tokens/sec out of a 6-core/12-thread Intel MacBook,
fully reproducible, with zero quality loss.

What actually worked:
- Set threads to physical cores, not logical. Hyper-Threading collapsed throughput.
- Use q4 over q8. Smaller weights mean less memory traffic per token.
- Batching under concurrency: a load-aware parallel setting gave 3.1× more aggregate throughput and ~8.6× lower tail latency with 8 concurrent users (vs parallel 1) — the one technique that did add a reliable win, and it's now automated in the repo.
- Result: 2.5–4.3× faster decode at identical output (verified against a frozen reference).

Then I tested the trendy technique — speculative decoding — expecting 1.5–2×.
- Early runs looked great: "KV-cache +36%", "speculation −26%".
- I repeated the runs. Both numbers vanished. They were run-to-run variance
  artifacts, not effects.

The honest conclusion: on a bandwidth-bound CPU, **neither speculative decoding
nor KV-cache quantization gives a reliable speedup.** I proved the root cause with
a context-size sweep (decode stayed flat across 16× more KV capacity): decode
re-reads the *entire model from RAM every token*. A draft model just adds
weight-fetches on the same bus.

I even hand-wrote the speculative decoder from scratch to confirm the result with
my own code, not just a flag.

Three lessons that generalize:
1. A single benchmark number is a lie. A median over repeats is a result.
2. Negative results are findings — catching your own mistake and reporting it is
   the credible move.
3. On CPU, inference is weight-bandwidth bound, not compute bound. Optimize bytes
   per token, not FLOPs.

Full case study, raw data, and code:
https://github.com/YuvrajSinghBhadoria2/llm-cpu-inference-optimization

If you've optimized LLM serving on constrained hardware, I'd love to hear what
surprised you.

#LLM #InferenceOptimization #MLOps #Benchmarking #OpenSource #CPUs
