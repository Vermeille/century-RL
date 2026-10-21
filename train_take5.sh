#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

train_adversarial adversarial-advshape \
  --game take5,num_players=2 \
  --steps 1800 \
  --perplexity-start "${AGENT_PPL_START:-5}" \
  --environment-perplexity-start "${ENVIRONMENT_PPL_START:-5}" \
  --agent-threshold "${AGENT_THRESHOLD:-0.98}" \
  --gae-lambda 0 \
  --value-lambda 1 \
  --adam-beta1 0.5 \
  --inference-batch-size 1024 \
  --learner-batch-size 2048 \
  --rollout-games 512 \
  --evaluation-games 512 \
  --save-every 25 \
  --value-clip-epsilon none \
  "$@"
