#!/usr/bin/env python3
"""Plot the balancer A/B evidence: tail latency and aggregate throughput for
--parallel 1 vs 8 under fixed concurrency (from results/balancer.json)."""
import json, sys
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed; skipping plot (JSON evidence still saved)",
          file=sys.stderr)
    sys.exit(0)

path = sys.argv[1] if len(sys.argv) > 1 else "results/balancer.json"
out = sys.argv[2] if len(sys.argv) > 2 else "assets/balancer_plot.png"
with open(path) as f:
    data = json.load(f)

cfgs = {c["parallel"]: c for c in data["configs"]}
keys = sorted(cfgs)
labels = [f"parallel={k}" for k in keys]
p95 = [cfgs[k]["ttft_p95_s"] for k in keys]
agg = [cfgs[k]["aggregate_tok_per_s"] for k in keys]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
b = axL.bar(labels, p95, color="#7a1f1f")
axL.set_ylabel("p95 TTFT (s) — lower is better", fontsize=12)
axL.set_title(f"Tail latency drops under concurrency (N={data['concurrency']})",
              fontsize=12, weight="bold")
for r, v in zip(b, p95):
    axL.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=11)

b2 = axR.bar(labels, agg, color="#1f4e79")
axR.set_ylabel("aggregate tok/s — higher is better", fontsize=12)
axR.set_title("Aggregate throughput rises with --parallel", fontsize=12, weight="bold")
for r, v in zip(b2, agg):
    axR.text(r.get_x()+r.get_width()/2, v+0.1, f"{v:.1f}", ha="center", fontsize=11)

fig.suptitle("Batch-size balancer: real A/B (median over requests)",
             fontsize=14, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(out, dpi=130)
print(f"saved {out}")
