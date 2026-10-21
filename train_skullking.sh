#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

train_adversarial adversarial-advshape \
  --game skullking,num_players=2,num_rounds=5 \
  --perplexity-start "${AGENT_PPL_START:-1.2}" \
  --environment-perplexity-start "${ENVIRONMENT_PPL_START:-1.6}" \
  --environment-perplexity-end "${ENVIRONMENT_PPL_END:-1.1}" \
  --gae-lambda 0.5 \
  --value-lambda 0.9 \
  --evaluation-games 128 \
  "$@"
