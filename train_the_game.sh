#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

exec uv run python trainers/coop.py \
  --game thegame,mode=omni \
  --steps 1800 \
  --perplexity-start 6 \
  --perplexity-end 1 \
  --learning-rate 4e-4 \
  --gae-lambda 0. \
  --value-lambda 1 \
  --kl-strength 0.1 \
  --kl-target 0.1 \
  --tag YOLO \
  --seed 1 \
  --trackio \
  --no-progress \
  "$@"
