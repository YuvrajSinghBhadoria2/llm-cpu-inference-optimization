#!/usr/bin/env python3
"""Summarize speculative-decoding / KV-cache benchmark results.

Reads every JSON in ../results, treats baseline.json as the control, and
prints a comparison table with speedup vs baseline. No results are invented;
all numbers come from the recorded eval_client.py outputs.
"""
import json, os, glob

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
files = sorted(glob.glob(os.path.join(RES, "*.json")))

rows = []
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    m = d.get("metrics", {})
    name = os.path.basename(f)[:-5]
    if m.get("tok_per_s_median") is None:
        continue
    rows.append((name, m["tok_per_s_median"], m.get("ttft_median_s"),
                 m.get("exact_match_fraction"), d.get("status")))

base_tps = next((t for n, t, tt, em, st in rows if n == "baseline"), None)

print(f"{'config':22s} {'tok/s':>8s} {'TTFT(s)':>9s} {'exact':>7s} {'speedup':>9s}")
print("-" * 58)
for n, t, tt, em, st in rows:
    sp = (t / base_tps) if base_tps else None
    sps = f"{sp:.2f}x" if sp is not None else "  -  "
    print(f"{n:22s} {t:8.2f} {tt:9.3f} {str(em):>7s} {sps:>9s}")
print("-" * 58)
if base_tps:
    print(f"baseline (no speculation) tok/s = {base_tps:.2f}")
    best = max(rows, key=lambda r: r[1])
    print(f"best config: {best[0]} = {best[1]:.2f} tok/s ({best[1]/base_tps:.2f}x)")
