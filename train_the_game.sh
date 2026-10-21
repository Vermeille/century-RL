#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

train_coop \
  --game thegame,mode=omni \
  --steps 1800 \
  --perplexity-start 6 \
  --gae-lambda 0 \
  --value-lambda 1 \
  "$@"
