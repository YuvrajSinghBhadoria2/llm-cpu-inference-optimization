# LinkedIn Post — LLM CPU Inference Optimization

**Recommended image:** attach `assets/linkedin_chart.png`. It shows both panels of
the story — the real 2.5–4.3× win (left) and the "variance trap" where the early
+36% / −26% claims collapsed to a null on repeated measurement (right). A chart
beats a screenshot: it makes the honest-null hook instantly legible while
scrolling. (If you prefer a single static, the right panel alone also works.)

---

I optimized LLM inference on a 2019 laptop CPU — no GPU. The most valuable thing
I learned was that my biggest "win" was fake.

The setup: squeeze maximum tokens/sec out of a 6-core/12-thread Intel MacBook,
fully reproducible, with zero quality loss.

What actually worked:
- `--threads` = physical cores, NOT logical. Hyper-Threading *collapsed* throughput.
- `q4` over `q8`. Smaller weights = less memory traffic per token.
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
