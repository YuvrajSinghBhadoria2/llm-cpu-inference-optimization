#!/usr/bin/env bash
set -euo pipefail
LLAMA="${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-qwen2.5-0.5b-q8_0.gguf}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-2048}"
THREADS="${THREADS:-$(sysctl -n hw.physicalcpu 2>/dev/null || nproc)}"
"$LLAMA" --model "$MODEL_DIR/$MODEL" --alias qwen0.5b --port "$PORT" --host "$HOST" \
  --ctx-size "$CTX" --threads "$THREADS" --parallel 1
