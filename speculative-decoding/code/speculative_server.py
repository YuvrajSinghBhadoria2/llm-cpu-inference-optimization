#!/usr/bin/env python3
"""From-scratch speculative decoder over two llama.cpp servers (draft + target).

This is a *hand-written* implementation of speculative decoding (draft proposal +
target verification + accept/reject with a bonus token) that talks to stock
llama.cpp servers over the OpenAI-compatible API. It is deliberately built from
first principles to demonstrate the algorithm -- it is NOT llama.cpp's built-in
`--spec-type`. Because it orchestrates the loop over HTTP, each step re-prefills
the prefix on the target, so it is strictly slower than a single batched call;
that is fine and is exactly the point: on a bandwidth-bound CPU, speculative
decoding cannot beat one token per target forward pass.

Run:
    python3 speculative_server.py --prompt "Once upon a time" --K 4 --max-tokens 64
    python3 speculative_server.py --baseline-only        # reference single-stream
"""
import argparse
import atexit
import json
import subprocess
import sys
import time
import urllib.request

LLAMA = "/Users/apple/llama.cpp/build/bin/llama-server"
_PROCS = []


def _cleanup():
    for p in _PROCS:
        try:
            p.kill()
        except Exception:
            pass


atexit.register(_cleanup)


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_health(port, timeout=600):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return True
        except Exception:
            time.sleep(1)
    return False


def start_server(model, port, threads, ctx, alias):
    p = subprocess.Popen(
        [LLAMA, "--model", model, "--alias", alias, "--threads", str(threads),
         "--parallel", "1", "--ctx-size", str(ctx),
         "--port", str(port), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _PROCS.append(p)
    if not wait_health(port, 600):
        raise RuntimeError(f"server on port {port} failed to start")
    return p


def complete(base, prompt, n, temperature=0):
    out = post(
        base + "/v1/completions",
        {"model": "x", "prompt": prompt, "max_tokens": n,
         "temperature": temperature, "logprobs": 1, "echo": False},
        timeout=600,
    )
    ch = out["choices"][0]
    items = ch.get("logprobs", {}).get("content", [])
    ids = [it["id"] for it in items]
    toks = [it["token"] for it in items]
    return ids, toks, ch.get("finish_reason")


def speculative(prompt, K, max_tokens, draft_base, target_base):
    emitted_ids, emitted_toks = [], []
    prefix = prompt
    start = time.time()
    while len(emitted_ids) < max_tokens:
        draft_ids, draft_toks, _ = complete(draft_base, prefix, K, 0)
        tgt_ids, tgt_toks, tgt_finish = complete(target_base, prefix, K + 1, 0)
        if not tgt_ids:
            break
        m = 0
        while m < len(draft_ids) and m < len(tgt_ids) and draft_ids[m] == tgt_ids[m]:
            m += 1
        step_ids, step_toks = draft_ids[:m], draft_toks[:m]
        if m < len(tgt_ids):
            step_ids.append(tgt_ids[m]); step_toks.append(tgt_toks[m])   # target's correction
        elif len(tgt_ids) > len(draft_ids):
            step_ids.append(tgt_ids[len(draft_ids)]); step_toks.append(tgt_toks[len(draft_ids)])  # bonus
        emitted_ids.extend(step_ids)
        emitted_toks.extend(step_toks)
        prefix = prefix + "".join(step_toks)
        if len(tgt_ids) < K + 1 or tgt_finish == "stop":
            break
    return emitted_ids, emitted_toks, time.time() - start


def baseline(prompt, max_tokens, target_base):
    start = time.time()
    ids, toks, _ = complete(target_base, prompt, max_tokens, 0)
    return ids, toks, time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--target-model", default="/Users/apple/models/qwen2.5-3b-instruct-q4_k_m.gguf")
    ap.add_argument("--draft-model", default="/Users/apple/models/qwen2.5-0.5b-q4_k_m.gguf")
    ap.add_argument("--target-port", type=int, default=8080)
    ap.add_argument("--draft-port", type=int, default=8081)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--no-start", action="store_true", help="attach to running servers")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    if not args.no_start:
        print("[start] draft + target llama-server ...", file=sys.stderr)
        start_server(args.draft_model, args.draft_port, args.threads, args.ctx, "draft")
        start_server(args.target_model, args.target_port, args.threads, args.ctx, "target")

    tb = f"http://127.0.0.1:{args.target_port}"
    db = f"http://127.0.0.1:{args.draft_port}"

    if args.baseline_only:
        ids, toks, dt = baseline(args.prompt, args.max_tokens, tb)
        print("=== BASELINE (single-stream target) ===")
        print("text:", "".join(toks).strip())
        print("tokens: %d  time: %.2fs  tok/s: %.2f" % (len(ids), dt, len(ids) / dt))
        return

    eids, etoks, dt = speculative(args.prompt, args.K, args.max_tokens, db, tb)
    bids, btoks, bdt = baseline(args.prompt, args.max_tokens, tb)
    print("=== FROM-SCRATCH SPECULATIVE DECODER ===")
    print("text:", "".join(etoks).strip())
    print("spec tokens: %d  time: %.2fs  tok/s: %.2f" % (len(eids), dt, len(eids) / dt))
    print("--- reference baseline (target only) ---")
    print("base text:", "".join(btoks).strip())
    print("base tokens: %d  time: %.2fs  tok/s: %.2f" % (len(bids), bdt, len(bids) / bdt))
    print("spec/baseline tok/s ratio: %.2f" % ((len(eids) / dt) / (len(bids) / bdt)))


if __name__ == "__main__":
    main()
