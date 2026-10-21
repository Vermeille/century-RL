#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

train_adversarial adversarial-advshape \
  --game rps \
  --perplexity-start "${AGENT_PPL_START:-1.5}" \
  --environment-perplexity-start "${ENVIRONMENT_PPL_START:-1.6}" \
  --environment-perplexity-end "${ENVIRONMENT_PPL_END:-1.2}" \
  "$@"
