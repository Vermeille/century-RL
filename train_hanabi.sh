#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
set -a
source .env
set +a

exec uv run python trainers/coop.py \
  --game hanabi,mode=mini \
  --steps 900 \
  --perplexity-start 3 \
  --perplexity-end 1 \
  --entropy-strength 0.01 \
  --learning-rate 4e-4 \
  --kl-strength 0.1 \
  --kl-target 0.1 \
  --gae-lambda 0.9 \
  --value-lambda 0.9 \
  --tag YOLO \
  --seed 1 \
  --trackio \
  --no-progress \
  "$@"
