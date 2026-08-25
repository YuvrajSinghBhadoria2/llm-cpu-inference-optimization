#!/usr/bin/env bash
# Full-optimized 3B serving for this CPU study.
# Combines every verified lever:
#   * --threads = physical cores (not Hyper-Thread siblings)
#   * q4_k_m weights (smaller -> less memory traffic)
#   * --cache-type-k / --cache-type-v q4_0 (KV-cache quantization; measured as a
    variance artifact, not a reliable win — kept for completeness only)
# Build llama.cpp CPU-only (-DGGML_METAL=OFF -DGGML_ACCELERATE=ON) first.
set -euo pipefail
LLAMA="${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-qwen2.5-3b-instruct-q4_k_m.gguf}"
PORT="${PORT:-8080}"; HOST="${HOST:-127.0.0.1}"; CTX="${CTX:-2048}"
THREADS="${THREADS:-$(sysctl -n hw.physicalcpu 2>/dev/null || nproc)}"
"$LLAMA" --model "$MODEL_DIR/$MODEL" --alias qwen3b --port "$PORT" --host "$HOST" \
  --ctx-size "$CTX" --threads "$THREADS" --parallel 1 \
  --cache-type-k q4_0 --cache-type-v q4_0
