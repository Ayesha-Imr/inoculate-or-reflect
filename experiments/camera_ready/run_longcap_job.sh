#!/usr/bin/env bash
set -euo pipefail

MODEL="gemma4-12b"
SEED="42"
TRAINING_ROOT=""
OUTPUT_ROOT=""
BATCH_SIZE="16"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --training-root) TRAINING_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be supplied for the base checkpoint" >&2
  exit 1
fi

BASE="${TRAINING_ROOT:-outputs/camera_ready/training/$MODEL/seed-$SEED}"
OUT="${OUTPUT_ROOT:-outputs/camera_ready/longcap/$MODEL/seed-$SEED}"
for spec in \
  "untrained" \
  "contaminated|$BASE/contaminated/adapter" \
  "strong_ip|$BASE/strong_ip/adapter" \
  "crt_repair|$BASE/crt_repair/adapter"; do
  IFS='|' read -r ARM ADAPTER <<< "$spec"
  args=(
    --model "$MODEL" --seed "$SEED" --arm "$ARM"
    --output-dir "$OUT/$ARM" --batch-size "$BATCH_SIZE"
    --max-new-tokens 1024 --sycophancy-only
  )
  if [[ -n "${ADAPTER:-}" ]]; then
    args+=(--adapter "$ADAPTER")
  fi
  python3 experiments/camera_ready/generate_eval.py "${args[@]}"
done
