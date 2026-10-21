#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

ppo_overrides=(
  --perplexity-start "${AGENT_PPL_START:-2}"
  --perplexity-end "${AGENT_PPL_END:-1.5}"
)
advshape_overrides=(
  "${ppo_overrides[@]}"
  --environment-perplexity-start "${ENVIRONMENT_PPL_START:-2.2}"
  --environment-perplexity-end "${ENVIRONMENT_PPL_END:-1.9}"
  --kl-target 0.005
  --environment-entropy-strength 0.1
)

train_openspiel "${TRAINER:-adversarial-advshape}" \
  --steps 2000 \
  --learning-rate 2e-5 \
  --gae-lambda 0.8 \
  --value-lambda 0.9 \
  --inference-batch-size 1024 \
  --learner-batch-size 256 \
  --rollout-games 1024 \
  --evaluation-games 512 \
  --evaluation-every 25 \
  --save-every 25 \
  --seed "${SEED:-1}" \
  --value-clip-epsilon none \
  "$@"
