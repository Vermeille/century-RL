#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
set -a
source .env
set +a

exec uv run python trainers/coop.py \
  --game skullking,num_rounds=5 \
  --steps 1800 \
  --perplexity-start 2 \
  --perplexity-end 1 \
  --learning-rate 4e-4 \
  --kl-strength 0.01 \
  --kl-target 0.02 \
  --gae-lambda 0.2 \
  --value-lambda 1.0 \
  --inference-batch-size 256 \
  --learner-batch-size 512 \
  --seed 1 \
  --tag full-decay-ppl1-99-lr50-100-n1800-s1 \
  --trackio \
  --no-progress \
  "$@"
