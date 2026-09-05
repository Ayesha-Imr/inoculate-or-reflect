#!/usr/bin/env bash
set -euo pipefail

MODEL=""
SEED=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$MODEL" || -z "$SEED" ]]; then
  echo "usage: $0 --model qwen3-8b|gemma4-12b --seed N" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be supplied for the base checkpoint" >&2
  exit 1
fi

python3 experiments/camera_ready/train_adapter.py \
  --model "$MODEL" --seed "$SEED" --arm contaminated
python3 experiments/camera_ready/train_adapter.py \
  --model "$MODEL" --seed "$SEED" --arm strong_ip
python3 experiments/camera_ready/train_adapter.py \
  --model "$MODEL" --seed "$SEED" --arm crt_repair \
  --init-adapter "outputs/camera_ready/training/$MODEL/seed-$SEED/contaminated/adapter"
