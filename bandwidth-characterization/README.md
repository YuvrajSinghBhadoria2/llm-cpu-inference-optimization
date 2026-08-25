# Bandwidth characterization — why CPU decode is bound

This is **Part 3** of the study: the root-cause experiment behind both the
Part 1 win (smaller q4 weights) and the Part 2 null result (speculation / KV
quant don't help).

## Question
Is decode on this laptop limited by **KV-cache memory traffic** or by **weight
memory traffic**? If KV size mattered, a larger `--ctx-size` (bigger KV cache)
would slow decode. If weights matter, ctx-size should be irrelevant.

## Method
Same target (Qwen2.5-3B-Instruct q4_k_m, 6 physical threads), same 5 prompts,
`max_tokens=128`. Vary only `--ctx-size` across 512 / 2048 / 8192 and measure
decode tok/s with `eval_client.py`.

## Result

| ctx-size | decode tok/s |
|---|---|
| 512   | 4.27 |
| 2048  | 4.72 |
| 8192  | 4.43 |

Decode speed is **flat** across a 16× change in KV capacity (spread ≈ 10% of the
median — within run-to-run noise).

## Conclusion
During generation the *active* KV length equals the prompt + output length, not
the allocated capacity, so KV-cache size is not the bottleneck. Decode cost is
dominated by **weight** reads: the full ~2 GB model is pulled from memory
*every single token*. This is the load-invariant root cause:
- **Part 1 win:** `q4` weights are ~half the size of `q8`, so each token fetches
  less from memory → faster decode. Threading fixes HT contention on the same
  bandwidth.
- **Part 2 null:** a draft model only adds *target* compute on that same
  bandwidth and shrinks KV traffic by a few percent — neither moves the needle.

Raw data: `results/ctx_512.json`, `ctx_2048.json`, `ctx_8192.json`.
Summary: `code/analyze.py`.
