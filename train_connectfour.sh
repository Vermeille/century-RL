#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
set -a
source .env
set +a

exec uv run python trainers/adversarial.py \
  --game connectfour \
  --anticipatory 0.5 \
  --random-game-depth 8 \
  --steps 1800 \
  --perplexity-start 1.5 \
  --perplexity-end 1 \
  --learning-rate 1e-4 \
  --average-learning-rate 4e-4 \
  --kl-strength 0.1 \
  --kl-target 0.01 \
  --gae-lambda 0. \
  --value-lambda 1 \
  --adam-beta1 0.5 \
  --inference-batch-size 512 \
  --learner-batch-size 1024 \
  --rollout-games 256 \
  --gradient-clip 10000 \
  --opponent-eval-strategy tactical_random \
  --entropy-strength 0.1 \
  --evaluation-every 10 \
  --reservoir-capacity 1000000 \
  --seed 1 \
  --trackio \
  --no-progress \
  --value-clip-epsilon none \
  "$@"
