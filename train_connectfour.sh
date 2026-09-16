#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
set -a
source .env
set +a

exec uv run python trainers/fightbot.py \
  --game connectfour \
  --steps 1800 \
  --perplexity-start 1.5 \
  --perplexity-end 1 \
  --learning-rate 4e-4 \
  --kl-strength 0.1 \
  --kl-target 0.01 \
  --gae-lambda 0. \
  --value-lambda 1 \
  --adam-beta1 0.5 \
  --inference-batch-size 512 \
  --learner-batch-size 1024 \
  --rollout-games 256 \
  --gradient-clip 10000 \
  --opponent-eval-strategy random \
  --opponent-bot random \
  --entropy-strength 0.01 \
  --evaluation-every 10 \
  --seed 1 \
  --trackio \
  --no-progress \
  --value-clip-epsilon none \
  "$@"
