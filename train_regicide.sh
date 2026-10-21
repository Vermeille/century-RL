#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

train_coop \
  --game regicide \
  --perplexity-start 4 \
  --inference-batch-size 128 \
  --learner-batch-size 96 \
  --gradient-clip 5000 \
  "$@"
