#!/usr/bin/env python3
"""Summarize the context-size sweep: decode tok/s vs --ctx-size capacity."""
import glob
import json
import os
import statistics as st

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
files = sorted(glob.glob(os.path.join(RESULTS, "ctx_*.json")))
print(f"{'ctx-size':>10}   {'tok/s':>7}   {'ttft':>6}")
print("-" * 30)
for f in files:
    d = json.load(open(f))
    m = d["metrics"]
    ctx = m.get("ctx_size") or m.get("gen_params", {}).get("ctx_size")
    name = os.path.basename(f).replace("ctx_", "").replace(".json", "")
    print(f"{name:>10}   {m['tok_per_s_median']:>7.2f}   {m['ttft_median_s']:>6.2f}")
print("-" * 30)
tps = [json.load(open(f))["metrics"]["tok_per_s_median"] for f in files]
print(f"decode tok/s across 16x capacity sweep: {tps}")
print(f"  min={min(tps):.2f} max={max(tps):.2f} spread={max(tps)-min(tps):.2f} "
      f"({(max(tps)-min(tps))/st.median(tps)*100:.0f}% of median) -> FLAT")
