#!/usr/bin/env bash
set -u
LLAMA=${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}
MODEL_DIR=${MODEL_DIR:-$HOME/models}
PROJ=$(cd "$(dirname "$0")" && pwd); cd "$PROJ"
PY=python3
pkill -9 -f llama-server 2>/dev/null; pkill -9 -f llama.cpp 2>/dev/null; sleep 4

sweep() {
  local t=$1
  pkill -9 -f llama-server 2>/dev/null; pkill -9 -f llama.cpp 2>/dev/null; sleep 4
  for i in $(seq 1 30); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 || break; sleep 1; done
  echo ">>> threads=$t"
  "$LLAMA" --model "$MODEL_DIR/qwen2.5-0.5b-q8_0.gguf" --alias qwen0.5b --port 8080 --host 127.0.0.1 \
     --ctx-size 2048 --threads $t --parallel 1 >/tmp/llama.log 2>&1 &
  local up=0
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { up=1; break; }; sleep 1; done
  if [ "$up" -ne 1 ]; then echo "FAIL threads=$t"; tail -15 /tmp/llama.log; return 1; fi
  "$PY" eval_client.py --reference reference_outputs.json --out "results/sweep_t$t.json" --slo-ttft 5.0
  pkill -9 -f llama-server 2>/dev/null; pkill -9 -f llama.cpp 2>/dev/null; sleep 2
}

for t in 2 3 4 6 8 12; do sweep $t; done
echo "=== SWEEP DONE ==="
for t in 2 3 4 6 8 12; do
  "$PY" -c "import json; d=json.load(open('results/sweep_t$t.json')); m=d['metrics']; print('threads=%2d  tok/s=%5.2f  ttft=%5.2f  em=%s' % ($t, m['tok_per_s_median'], m['ttft_median_s'], m['exact_match_fraction']))"
done
