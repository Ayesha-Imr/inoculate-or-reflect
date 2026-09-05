#!/usr/bin/env bash
set -euo pipefail

MODEL=""
SEED=""
TRAINING_ROOT=""
OUTPUT_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --training-root) TRAINING_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$MODEL" || -z "$SEED" ]]; then
  echo "usage: $0 --model qwen3-8b|gemma4-12b --seed N [--training-root PATH] [--output-root PATH]" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be supplied for the base checkpoint" >&2
  exit 1
fi

BASE="${TRAINING_ROOT:-outputs/camera_ready/training/$MODEL/seed-$SEED}"
OUT="${OUTPUT_ROOT:-outputs/camera_ready/behavior/$MODEL/seed-$SEED}"
python3 experiments/camera_ready/generate_eval.py \
  --model "$MODEL" --seed "$SEED" --arm untrained \
  --output-dir "$OUT/untrained" --skip-generalization
python3 experiments/camera_ready/generate_eval.py \
  --model "$MODEL" --seed "$SEED" --arm contaminated \
  --adapter "$BASE/contaminated/adapter" \
  --output-dir "$OUT/contaminated" \
  --skip-generalization
python3 experiments/camera_ready/generate_eval.py \
  --model "$MODEL" --seed "$SEED" --arm strong_ip \
  --adapter "$BASE/strong_ip/adapter" \
  --output-dir "$OUT/strong_ip" \
  --skip-generalization
python3 experiments/camera_ready/generate_eval.py \
  --model "$MODEL" --seed "$SEED" --arm crt_repair \
  --adapter "$BASE/crt_repair/adapter" \
  --output-dir "$OUT/crt_repair" \
  --skip-generalization
