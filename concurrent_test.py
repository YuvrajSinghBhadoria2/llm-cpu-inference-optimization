#!/usr/bin/env python3
"""Concurrent-load benchmark for the llama.cpp server.

Fires N concurrent requests (threaded) and reports aggregate throughput
(total tokens / wall time across all clients) plus per-request latency.
This is what reveals whether --parallel (batching) helps under concurrent load,
which the single-stream evaluator cannot show.
"""
import argparse, json, threading, time, urllib.request


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
    with urllib.request.urlopen(req, timeout=300) as r:
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
    comp = usage.get("completion_tokens") if usage else max(1, len("".join(text)) // 4)
    return ttft, t_end - t0, comp


def worker(idx, args, results, barrier):
    prompt = PROMPTS[idx % len(PROMPTS)]
    barrier.wait()
    ttft, dur, comp = complete(args.url, args.model, prompt, args.max_tokens)
    results[idx] = {"ttft": ttft, "dur": dur, "comp": comp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/completions")
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    results = [None] * args.concurrency
    barrier = threading.Barrier(args.concurrency)
    threads = [threading.Thread(target=worker, args=(i, args, results, barrier))
               for i in range(args.concurrency)]
    wall0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - wall0

    comps = [r["comp"] for r in results]
    ttfts = [r["ttft"] for r in results]
    total_tok = sum(comps)
    agg = total_tok / wall
    per = [c / r["dur"] for c, r in zip(comps, results)]
    import statistics as st
    out = {
        "concurrency": args.concurrency,
        "wall_s": round(wall, 2),
        "total_tokens": total_tok,
        "aggregate_tok_per_s": round(agg, 2),
        "per_request_tok_per_s_median": round(st.median(per), 2),
        "per_request_tok_per_s_mean": round(st.mean(per), 2),
        "ttft_median_s": round(st.median(ttfts), 3),
        "ttft_p95_s": round(sorted(ttfts)[min(len(ttfts) - 1, int(len(ttfts) * 0.95))], 3),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
