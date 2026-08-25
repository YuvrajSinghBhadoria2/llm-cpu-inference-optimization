#!/usr/bin/env bash
# Runs the full CPU inference-optimization study on the local Mac using llama.cpp.
# Usage: bash run_study.sh   (after building llama.cpp and downloading GGUFs)
set -u
LLAMA=${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}
MODEL_DIR=${MODEL_DIR:-$HOME/models}
PROJ=$(cd "$(dirname "$0")" && pwd)
cd "$PROJ"
PY=python3
mkdir -p results

run() {
  local model=$1; local extra=$2; local out=$3; local freeze=${4:-}
  pkill -9 -f llama-server 2>/dev/null; sleep 2
  echo ">>> starting $model $extra"
  "$LLAMA" --model "$MODEL_DIR/$model" --alias qwen0.5b --port 8080 --host 127.0.0.1 \
     --ctx-size 2048 $extra > /tmp/llama.log 2>&1 &
  for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
    sleep 1
  done
  if [ -n "$freeze" ]; then
    "$PY" eval_client.py --freeze-reference reference_outputs.json
  else
    "$PY" eval_client.py --reference reference_outputs.json --out "$out"
  fi
  pkill -9 -f llama-server 2>/dev/null; sleep 2
}

# baseline: q8_0, all 12 logical threads, single sequence
run qwen2.5-0.5b-q8_0.gguf "--threads 12 --parallel 1" results/baseline.json freeze
run qwen2.5-0.5b-q8_0.gguf "--threads 12 --parallel 1" results/baseline.json
# c1: 4-bit quantization (same architecture, smaller weights)
run qwen2.5-0.5b-q4_k_m.gguf "--threads 12 --parallel 1" results/c1.json
# c2: continuous-batching width (parallel sequences)
run qwen2.5-0.5b-q8_0.gguf "--threads 12 --parallel 4" results/c2.json
# c3: thread-count tuning (physical cores only vs all HT)
run qwen2.5-0.5b-q8_0.gguf "--threads 6 --parallel 1" results/c3.json

echo "=== SUMMARY ==="
"$PY" - <<'PY'
import json
for f in ["results/baseline.json","results/c1.json","results/c2.json","results/c3.json"]:
    try:
        d=json.load(open(f)); m=d["metrics"]
        print(f.split("/")[-1].ljust(14), d["status"],
              "tok/s=%.1f"%m["tok_per_s_median"],
              "ttft=%.2fs"%m["ttft_median_s"],
              "exact=%.2f"%m["exact_match_fraction"],
              "overlap=%.2f"%m["overlap_mean"],
              "goodput=%.2f"%m["goodput_fraction"])
    except Exception as e:
        print(f,"MISSING",e)
PY
