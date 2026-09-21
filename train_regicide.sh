#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

exec uv run python trainers/coop.py \
  --game regicide \
  --steps 900 \
  --perplexity-start 4 \
  --perplexity-end 1 \
  --learning-rate 4e-4 \
  --kl-strength 0.1 \
  --kl-target 0.1 \
  --inference-batch-size 128 \
  --learner-batch-size 96 \
  --tag YOLO \
  --seed 1 \
  --trackio \
  --no-progress \
  --gradient-clip 5000 \
  "$@"
