import json, glob, os, statistics as st, re

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
files = sorted(glob.glob(os.path.join(RESULTS, "*.json")))

groups = {}
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    m = re.match(r"rep_(.+)_(\d+)$", name)
    cfg = m.group(1) if m else name
    try:
        d = json.load(open(f))
        tps = d["metrics"]["tok_per_s_median"]
    except Exception:
        continue
    groups.setdefault(cfg, []).append(tps)

disp = {
    "baseline": "baseline (no spec)",
    "kvq4": "kvq4 (cache q4_0)",
    "spec05": "spec 0.5B K=4",
}

base_med = st.median(groups["baseline"])
rows = []
for cfg, vals in groups.items():
    med = st.median(vals)
    rows.append((disp.get(cfg, cfg), len(vals), med, med / base_med, min(vals), max(vals)))
rows.sort(key=lambda r: -r[2])

print(f"{'config':<22}{'n':>3}{'tok/s':>9}{'speedup':>9}   range")
print("-" * 52)
for name, n, med, sp, lo, hi in rows:
    print(f"{name:<22}{n:>3}{med:>9.2f}{sp:>8.2f}x   {lo:.2f}-{hi:.2f}")
print("-" * 52)
best = rows[0]
print(f"baseline median = {base_med:.2f} tok/s")
print(
    f"best by median  = {best[0]} {best[2]:.2f} tok/s ({best[3]:.2f}x) "
    f"-- within run-to-run noise, NOT a reliable speedup"
)
