#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

# Slide and placement targets stay within the slider's four-action space.
advshape_overrides=(
  --perplexity-start "${AGENT_PPL_START:-3}"
  --environment-perplexity-start "${ENVIRONMENT_PPL_START:-3}"
  --environment-perplexity-end "${ENVIRONMENT_PPL_END:-2}"
  --agent-threshold "${AGENT_THRESHOLD:-0.98}"
)

train_adversarial "${TRAINER:-adversarial-advshape}" \
  --game 2048versus,nolimits=True \
  --steps 1800 \
  --gae-lambda 0 \
  --value-lambda 1 \
  --evaluation-games 128 \
  --seed "${SEED:-1}" \
  --value-clip-epsilon none \
  "$@"
