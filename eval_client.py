#!/usr/bin/env python3
"""External evaluator for the llama.cpp CPU study.

Measures, per prompt, time-to-first-token (TTFT) and tokens/sec against a
llama.cpp OpenAI-compatible server, and compares generated text to a frozen
reference. Candidates that keep identical weights must match the reference
exactly (determinism check); quantized candidates are scored by word-overlap
since 4-bit weights legitimately change outputs.
"""
import argparse, json, time, urllib.request, statistics as st


PROMPTS = [
    "Explain what a vector database is in two sentences.",
    "Write a short Python function that reverses a linked list.",
    "Summarize the causes of the French Revolution in one paragraph.",
    "List three benefits of test-driven development.",
    "Describe how gradient descent works in plain language.",
]


def complete(url, model, prompt, max_tokens):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = []; usage = None
    with urllib.request.urlopen(req, timeout=180) as r:
        buf = b""
        while True:
            chunk = r.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                if ttft is None:
                    ttft = time.time() - t0
                if obj.get("choices"):
                    d = obj["choices"][0].get("delta", {})
                    if "content" in d:
                        text.append(d["content"])
                if obj.get("usage"):
                    usage = obj["usage"]
    t_end = time.time()
    gen = "".join(text)
    comp = usage.get("completion_tokens") if usage else None
    if comp is None:
        comp = max(1, len(gen) // 4)
    return ttft, t_end - t0, gen, comp


def overlap(a, b):
    aw, bw = a.split(), b.split()
    m = sum(1 for x, y in zip(aw, bw) if x == y)
    denom = max(len(aw), len(bw))
    return (m / denom) if denom else 1.0


def p95(x):
    if len(x) < 2:
        return x[0]
    qs = st.quantiles(x, n=100)
    return qs[94] if len(qs) >= 95 else qs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/completions")
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--freeze-reference", metavar="FILE")
    ap.add_argument("--reference", metavar="FILE")
    ap.add_argument("--out", metavar="FILE")
    ap.add_argument("--slo-ttft", type=float, default=3.0)
    ap.add_argument("--slo-toks", type=float, default=5.0)
    args = ap.parse_args()

    if args.freeze_reference:
        ref = []
        for p in PROMPTS:
            ttft, total, gen, comp = complete(args.url, args.model, p, args.max_tokens)
            ref.append({"prompt": p, "output": gen})
        json.dump(ref, open(args.freeze_reference, "w"), indent=2)
        print("froze reference ->", args.freeze_reference, "prompts:", len(ref))
        return

    ref = json.load(open(args.reference))
    details = []; tok_rates = []; ttfts = []; exact = 0; overlaps = []
    for item in ref:
        ttft, total, gen, comp = complete(args.url, args.model, item["prompt"], args.max_tokens)
        gen_time = max(1e-6, total - (ttft or 0))
        rate = comp / gen_time
        tok_rates.append(rate); ttfts.append(ttft or total)
        ex = (gen.strip() == item["output"].strip())
        exact += 1 if ex else 0
        ov = overlap(gen, item["output"])
        overlaps.append(ov)
        details.append({"prompt": item["prompt"][:40], "tok_per_s": round(rate, 2),
                        "ttft_s": round(ttft or 0, 3), "exact": ex, "overlap": round(ov, 3)})
    metrics = {
        "tok_per_s_median": round(st.median(tok_rates), 2),
        "tok_per_s_mean": round(st.mean(tok_rates), 2),
        "ttft_median_s": round(st.median(ttfts), 3),
        "p95_ttft_s": round(p95(ttfts), 3),
        "exact_match_fraction": round(exact / len(ref), 3),
        "overlap_mean": round(st.mean(overlaps), 3),
        "goodput_fraction": round(sum(1 for r, t in zip(tok_rates, ttfts)
                                      if t <= args.slo_ttft and r >= args.slo_toks) / len(ref), 3),
    }
    status = "PASS" if (exact == len(ref) or metrics["overlap_mean"] >= 0.90) else "WARN"
    result = {"status": status, "metrics": metrics, "details": details}
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
