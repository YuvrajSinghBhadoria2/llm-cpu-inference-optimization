#!/usr/bin/env bash
# Scaling study: confirm the CPU-optimization findings generalize to 1.5B and 3B.
# For each family: freeze reference at q8 / threads=6, then measure:
#   q8 @ threads=6  (per-size baseline)
#   q8 @ threads=12 (must be slower -> physical-core rule holds)
#   q4 @ threads=6  (decode speed at optimal threading)
set -u
LLAMA=${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}
MODEL_DIR=${MODEL_DIR:-$HOME/models}
PROJ=$(cd "$(dirname "$0")" && pwd); cd "$PROJ"
PY=python3
PHYS=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc)

start_server() {  # model_file alias threads
  local model=$1 alias=$2 threads=$3
  pkill -9 -f llama-server 2>/dev/null; pkill -9 -f llama.cpp 2>/dev/null; sleep 4
  for i in $(seq 1 30); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 || break; sleep 1; done
  "$LLAMA" --model "$MODEL_DIR/$model" --alias "$alias" --port 8080 --host 127.0.0.1 \
     --ctx-size 2048 --threads "$threads" --parallel 1 >/tmp/llama.log 2>&1 &
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && return 0; sleep 1; done
  echo "SERVER FAILED: $model t=$threads"; tail -15 /tmp/llama.log; return 1
}
stop_server() { pkill -9 -f llama-server 2>/dev/null; pkill -9 -f llama.cpp 2>/dev/null; sleep 2; }

families=(
  "1.5b|qwen2.5-1.5b-instruct-q8_0.gguf|qwen2.5-1.5b-instruct-q4_k_m.gguf|qwen1.5b"
  "3b|qwen2.5-3b-instruct-q8_0.gguf|qwen2.5-3b-instruct-q4_k_m.gguf|qwen3b"
)

for fam in "${families[@]}"; do
  IFS='|' read -r tag q8 q4 alias <<< "$fam"
  ref="reference_outputs_${tag}.json"
  echo "########## FAMILY $tag ##########"
  # freeze reference at q8 / threads=physical
  start_server "$q8" "$alias" "$PHYS" || continue
  "$PY" eval_client.py --freeze-reference "$ref" --model "$alias" --max-tokens 128
  stop_server
  # q8 @ physical
  start_server "$q8" "$alias" "$PHYS" || continue
  "$PY" eval_client.py --reference "$ref" --model "$alias" --out "results/scale_${tag}_q8_t${PHYS}.json" --slo-ttft 10 --slo-toks 2
  stop_server
  # q8 @ logical (12)
  start_server "$q8" "$alias" 12 || continue
  "$PY" eval_client.py --reference "$ref" --model "$alias" --out "results/scale_${tag}_q8_t12.json" --slo-ttft 10 --slo-toks 2
  stop_server
  # q4 @ physical
  start_server "$q4" "$alias" "$PHYS" || continue
  "$PY" eval_client.py --reference "$ref" --model "$alias" --out "results/scale_${tag}_q4_t${PHYS}.json" --slo-ttft 10 --slo-toks 2
  stop_server
done

echo "=== SCALE SUMMARY ==="
for f in results/scale_*.json; do
  "$PY" -c "import json,os; d=json.load(open('$f')); m=d['metrics']; print('%-40s tok/s=%5.2f ttft=%5.2f em=%s ov=%s' % (os.path.basename('$f'), m['tok_per_s_median'], m['ttft_median_s'], m['exact_match_fraction'], m['overlap_mean']))"
done
