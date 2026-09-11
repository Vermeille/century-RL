#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
set -a
source .env
set +a

exec uv run python trainers/fightbot.py \
  --game connectfour \
  --steps 1800 \
  --perplexity-start 3 \
  --perplexity-end 1 \
  --learning-rate 3e-4 \
  --kl-strength 0.1 \
  --kl-target 0.05 \
  --gae-lambda 0.5 \
  --value-lambda 1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --inference-batch-size 512 \
  --learner-batch-size 1024 \
  --rollout-games 512 \
  --gradient-clip 10000 \
  --opponent-eval-strategy tactical_random \
  --opponent-bot tactical_random \
  --entropy-strength 0.1 \
  --evaluation-every 10 \
  --seed 1 \
  --trackio \
  --no-progress \
  --value-clip-epsilon none \
  "$@"
