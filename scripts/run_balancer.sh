#!/usr/bin/env bash
# Runs the dynamic batch-size balancer A/B (--parallel 1 vs 8) under fixed
# concurrency and writes results/balancer.json.
# Build llama.cpp first (CPU-only) and place GGUFs in ~/models.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/balancer.py \
  --model "${MODEL:-qwen2.5-3b-instruct-q4_k_m.gguf}" \
  --concurrency "${CONCURRENCY:-8}" \
  --max-tokens "${MAX_TOKENS:-64}" \
  --parallels "${PARALLELS:-1,8}" \
  --out results/balancer.json
