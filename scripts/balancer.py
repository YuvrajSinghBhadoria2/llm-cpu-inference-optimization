#!/usr/bin/env python3
"""Dynamic batch-size balancer for llama.cpp CPU serving.

The single-stream evaluator cannot see batching wins. Our study showed that
under concurrent load, raising --parallel (e.g. 8) gives ~+45% aggregate
throughput and ~4x lower tail latency vs --parallel 1 (see README Part 1 /
EXPLANATION §4.3).

This module:
  * recommends an optimal --parallel for a given live request count, and
  * runs a controlled A/B (parallel=1 vs parallel=8) under fixed concurrency,
    saving the real measured evidence to JSON.
"""
import argparse, json, os, subprocess, sys, time, signal, urllib.request

LLAMA_DEFAULT = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
MODEL_DIR_DEFAULT = os.path.expanduser("~/models")
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def recommend_parallel(active_requests, max_parallel=8):
    """Pick --parallel from the current request count.

    Single stream: no batching benefit -> 1. Otherwise scale with load, capped
    at max_parallel (our evidence showed gains saturate by ~8 concurrent).
    """
    if active_requests <= 1:
        return 1
    return min(int(active_requests), max_parallel)


def _wait_health(port, timeout=120):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def _start_server(llama, model_path, port, threads, parallel, ctx=2048, alias="bench"):
    args = [llama, "--model", model_path, "--alias", alias,
            "--port", str(port), "--host", "127.0.0.1",
            "--ctx-size", str(ctx), "--threads", str(threads),
            "--parallel", str(parallel)]
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _wait_health(port):
        p.kill()
        raise RuntimeError("llama-server failed to become healthy")
    return p


def _stop_server(p):
    try:
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=15)
    except Exception:
        p.kill()


def _run_load_test(port, model, concurrency, max_tokens, alias="bench"):
    out = subprocess.run(
        [sys.executable, "concurrent_test.py",
         "--url", f"http://127.0.0.1:{port}/v1/completions",
         "--model", alias, "--concurrency", str(concurrency),
         "--max-tokens", str(max_tokens)],
        capture_output=True, text=True, cwd=PKG_ROOT)
    text = out.stdout.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError("no JSON from concurrent_test.py: " + out.stdout[-400:])
    return json.loads(text[start:end + 1])


def run_ab(model, concurrency=8, max_tokens=64, parallels=(1, 8),
           threads=None, port=8090, out_path=None,
           llama=LLAMA_DEFAULT, model_dir=MODEL_DIR_DEFAULT, max_parallel=8):
    threads = threads or str(os.cpu_count() or 6)
    model_path = os.path.join(model_dir, model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    configs = []
    for par in parallels:
        p = _start_server(llama, model_path, port, threads, par)
        try:
            m = _run_load_test(port, model, concurrency, max_tokens)
        finally:
            _stop_server(p)
        m["parallel"] = par
        m["recommended_parallel"] = recommend_parallel(concurrency, max_parallel)
        configs.append(m)
        print(f"[balancer] parallel={par}: agg={m['aggregate_tok_per_s']} tok/s, "
              f"p95_ttft={m['ttft_p95_s']}s", file=sys.stderr)
    result = {"model": model, "concurrency": concurrency,
              "max_tokens": max_tokens, "configs": configs}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[balancer] wrote {out_path}", file=sys.stderr)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-3b-instruct-q4_k_m.gguf")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--parallels", default="1,8")
    ap.add_argument("--threads", default=None)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--out", default="results/balancer.json")
    ap.add_argument("--llama", default=LLAMA_DEFAULT)
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--recommend-only", type=int, default=None,
                    help="print recommended --parallel for N active requests and exit")
    args = ap.parse_args()

    if args.recommend_only is not None:
        print(recommend_parallel(args.recommend_only))
        return

    parallels = [int(x) for x in args.parallels.split(",")]
    run_ab(model=args.model, concurrency=args.concurrency,
           max_tokens=args.max_tokens, parallels=parallels,
           threads=args.threads, port=args.port, out_path=args.out,
           llama=args.llama, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
