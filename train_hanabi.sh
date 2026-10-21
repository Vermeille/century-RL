#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

train_coop \
  --game hanabi,mode=mini \
  --perplexity-start 3 \
  --entropy-strength 0.01 \
  --gae-lambda 0.9 \
  --value-lambda 0.9 \
  "$@"
