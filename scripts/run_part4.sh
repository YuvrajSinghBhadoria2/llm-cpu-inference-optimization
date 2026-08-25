#!/usr/bin/env bash
# Part 4 — Quantization-scheme sweep + memory-loading + CPU affinity.
# Extends the existing llama.cpp CPU study on the 2019 Intel Mac (6C/12T i7).
# Reuses eval_client.py (identical protocol: 5 prompts, max_tokens=128, temp=0).
set -u
LLAMA=${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}
MODEL_DIR=${MODEL_DIR:-$HOME/models}
PROJ=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJ"
PY=python3
mkdir -p results
REPEATS=${REPEATS:-3}

# fixed optimal config from Part 1: physical cores, single stream
T=6
CTX=2048

freeze() {  # $1 model $2 outref
  pkill -9 -f llama-server 2>/dev/null; sleep 2
  "$LLAMA" --model "$MODEL_DIR/$1" --alias m --port 8080 --host 127.0.0.1 \
    --ctx-size $CTX --threads $T --parallel 1 > /tmp/llama.log 2>&1 &
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  "$PY" eval_client.py --freeze-reference "$2"
  pkill -9 -f llama-server 2>/dev/null; sleep 2
}

eval_cfg() {  # $1 model $2 extra-flags $3 outjson $4 ref
  pkill -9 -f llama-server 2>/dev/null; sleep 2
  "$LLAMA" --model "$MODEL_DIR/$1" --alias m --port 8080 --host 127.0.0.1 \
    --ctx-size $CTX --threads $T --parallel 1 $2 > /tmp/llama.log 2>&1 &
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  "$PY" eval_client.py --reference "$4" --out "$3"
  pkill -9 -f llama-server 2>/dev/null; sleep 2
}

echo "=== freezing references (q8_0 @ t6) ==="
freeze qwen2.5-0.5b-q8_0.gguf results/part4_ref_0.5b.json
freeze qwen2.5-3b-instruct-q8_0.gguf results/part4_ref_3b.json

echo "=== 0.5B quant-scheme sweep (t6, 3 repeats each) ==="
for q in q4_0 q4_k_m q5_k_m iq4_xs q8_0; do
  for r in $(seq 1 $REPEATS); do
    eval_cfg qwen2.5-0.5b-$q.gguf "" results/part4_0.5b_${q}_r${r}.json results/part4_ref_0.5b.json
  done
done

echo "=== memory loading: q4_k_m @ t6, mmap(default) vs no-mmap+mlock ==="
for r in $(seq 1 2); do
  eval_cfg qwen2.5-0.5b-q4_k_m.gguf "" results/part4_0.5b_q4km_mmap_r${r}.json results/part4_ref_0.5b.json
  eval_cfg qwen2.5-0.5b-q4_k_m.gguf "--no-mmap --mlock" results/part4_0.5b_q4km_nommap_r${r}.json results/part4_ref_0.5b.json
done

echo "=== CPU affinity: q4_k_m @ t6, --cpu-strict 1 ==="
for r in $(seq 1 2); do
  eval_cfg qwen2.5-0.5b-q4_k_m.gguf "--cpu-strict 1" results/part4_0.5b_q4km_aff_r${r}.json results/part4_ref_0.5b.json
done

echo "=== 3B quant-scheme sweep (t6, 2 repeats each) ==="
for q in q4_0 q4_k_m q5_k_m q8_0; do
  for r in $(seq 1 2); do
    eval_cfg qwen2.5-3b-instruct-$q.gguf "" results/part4_3b_${q}_r${r}.json results/part4_ref_3b.json
  done
done

echo "=== SUMMARY ==="
"$PY" - <<'PY'
import json, glob, statistics as st, re
def med(fs):
    rs=[]
    for f in fs:
        try: rs.append(json.load(open(f))["metrics"]["tok_per_s_median"])
        except Exception: pass
    return (st.median(rs), min(rs), max(rs)) if rs else (None,None,None)
def ov(fs):
    vs=[]
    for f in fs:
        try: vs.append(json.load(open(f))["metrics"]["overlap_mean"])
        except Exception: pass
    return round(st.mean(vs),3) if vs else None
groups=[
 ("0.5B q4_0",          "results/part4_0.5b_q4_0_r*.json"),
 ("0.5B q4_k_m",        "results/part4_0.5b_q4_k_m_r*.json"),
 ("0.5B q5_k_m",        "results/part4_0.5b_q5_k_m_r*.json"),
 ("0.5B iq4_xs",        "results/part4_0.5b_iq4_xs_r*.json"),
 ("0.5B q8_0",          "results/part4_0.5b_q8_0_r*.json"),
 ("0.5B q4_k_m mmap",   "results/part4_0.5b_q4km_mmap_r*.json"),
 ("0.5B q4_k_m no-mmap","results/part4_0.5b_q4km_nommap_r*.json"),
 ("0.5B q4_k_m aff",    "results/part4_0.5b_q4km_aff_r*.json"),
 ("3B q4_0",            "results/part4_3b_q4_0_r*.json"),
 ("3B q4_k_m",          "results/part4_3b_q4_k_m_r*.json"),
 ("3B q5_k_m",          "results/part4_3b_q5_k_m_r*.json"),
 ("3B q8_0",            "results/part4_3b_q8_0_r*.json"),
]
print(f"{'config':22} {'tok/s med':>10} {'min':>7} {'max':>7} {'overlap':>8}")
for name,pat in groups:
    fs=sorted(glob.glob(pat))
    m,lo,hi=med(fs); o=ov(fs)
    print(f"{name:22} {('%.2f'%m) if m else 'NA':>10} {(('%.2f'%lo) if lo else 'NA'):>7} {(('%.2f'%hi) if hi else 'NA'):>7} {('%.3f'%o) if o is not None else 'NA':>8}")
PY
