#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

ppo_overrides=(
  --perplexity-start "${AGENT_PPL_START:-5}"
  --perplexity-end "${AGENT_PPL_END:-1}"
)
advshape_overrides=(
  "${ppo_overrides[@]}"
  --environment-perplexity-start "${ENVIRONMENT_PPL_START:-5}"
  --agent-threshold "${AGENT_THRESHOLD:-0.98}"
)

train_adversarial "${TRAINER:-adversarial-advshape}" \
  --game connectfour \
  --steps 1800 \
  --gae-lambda 0 \
  --value-lambda 1 \
  --adam-beta1 0.5 \
  --inference-batch-size 1024 \
  --learner-batch-size 2048 \
  --rollout-games 512 \
  --evaluation-games 512 \
  --opponent-eval-strategy tactical_random \
  --save-every 25 \
  --value-clip-epsilon none \
  "$@"
