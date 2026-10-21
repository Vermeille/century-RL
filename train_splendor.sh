#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

train_adversarial adversarial-advshape \
  --game splendor \
  --gae-lambda 0.5 \
  --value-lambda 0.8 \
  --rollout-games 64 \
  --evaluation-games 128 \
  --tag "${TAG:-splendor}" \
  "$@"
