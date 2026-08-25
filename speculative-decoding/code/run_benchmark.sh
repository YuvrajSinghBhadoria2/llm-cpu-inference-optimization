#!/usr/bin/env bash
# Reproducible benchmark for the speculative-decoding / KV-cache extension.
# Starts a llama.cpp server with a given configuration, runs eval_client.py
# against the frozen 3B reference, and writes the JSON result.
#
# All runs use the same 5 prompts and temperature=0 (see eval_client.py),
# so results are directly comparable to baseline.json.
#
# Examples:
#   bash run_benchmark.sh --label baseline --spec-type none
#   bash run_benchmark.sh --label spec05 --spec-type draft-simple \
#       --draft-model qwen2.5-0.5b-q4_k_m.gguf --draft-n-max 4
#   bash run_benchmark.sh --label kvq4 --cache-k q4_0 --cache-v q4_0
#   bash run_benchmark.sh --label baseline --repeats 3   # repeated for variance
#
# Each run is written to speculative-decoding/results/rep_<label>_<i>.json so
# code/analyze_results.py can group repeats and report medians (see README).
set -u
LLAMA="${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
CODE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$CODE/../.." && pwd)"
cd "$PROJ"

THREADS="${THREADS:-$(sysctl -n hw.physicalcpu 2>/dev/null || nproc)}"
TARGET="${TARGET:-qwen2.5-3b-instruct-q4_k_m.gguf}"
LABEL="run"
REPEATS=1
SPEC_TYPE="none"
DRAFT=""; DRAFT_N=""; NGRAM_N=""; NGRAM_M=""; CK=""; CV=""
PORT=8080; HOST=127.0.0.1; CTX=2048
REF="results/reference_outputs_3b.json"

while [ $# -gt 0 ]; do
  case "$1" in
    --label)        LABEL="$2"; shift 2;;
    --target)       TARGET="$2"; shift 2;;
    --threads)      THREADS="$2"; shift 2;;
    --repeats)      REPEATS="$2"; shift 2;;
    --spec-type)    SPEC_TYPE="$2"; shift 2;;
    --draft-model)  DRAFT="$2"; shift 2;;
    --draft-n-max)  DRAFT_N="$2"; shift 2;;
    --ngram-n)      NGRAM_N="$2"; shift 2;;
    --ngram-m)      NGRAM_M="$2"; shift 2;;
    --cache-k)      CK="$2"; shift 2;;
    --cache-v)      CV="$2"; shift 2;;
    --reference)    REF="$2"; shift 2;;
    --port)         PORT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

EXTRA="--threads $THREADS --parallel 1 --ctx-size $CTX"
[ -n "$CK" ] && EXTRA="$EXTRA --cache-type-k $CK"
[ -n "$CV" ] && EXTRA="$EXTRA --cache-type-v $CV"
if [ "$SPEC_TYPE" != "none" ]; then
  EXTRA="$EXTRA --spec-type $SPEC_TYPE"
  [ -n "$DRAFT" ]   && EXTRA="$EXTRA --spec-draft-model $MODEL_DIR/$DRAFT"
  [ -n "$DRAFT_N" ] && EXTRA="$EXTRA --spec-draft-n-max $DRAFT_N"
  [ -n "$NGRAM_N" ] && EXTRA="$EXTRA --spec-ngram-simple-size-n $NGRAM_N"
  [ -n "$NGRAM_M" ] && EXTRA="$EXTRA --spec-ngram-simple-size-m $NGRAM_M"
fi

for r in $(seq 1 "$REPEATS"); do
  OUT="speculative-decoding/results/rep_${LABEL}_${r}.json"
  pkill -9 -f llama-server 2>/dev/null; sleep 3
  for i in $(seq 1 30); do curl -sf http://$HOST:$PORT/health >/dev/null 2>&1 || break; sleep 1; done
  echo ">>> ($r/$REPEATS) starting $TARGET ($EXTRA)"
  "$LLAMA" --model "$MODEL_DIR/$TARGET" --alias qwen3b --port "$PORT" --host "$HOST" \
    $EXTRA > /tmp/sd_bench.log 2>&1 &
  for i in $(seq 1 300); do curl -sf http://$HOST:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  python3 eval_client.py --reference "$REF" --model qwen3b --out "$OUT"
  echo ">>> wrote $OUT"
  pkill -9 -f llama-server 2>/dev/null; sleep 2
done
