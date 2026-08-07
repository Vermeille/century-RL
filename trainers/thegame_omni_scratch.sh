#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

set -a
source .env
set +a

.venv/bin/python trainers/coop.py \
  --device cuda \
  --architecture patchformer-medium-p4-wide-canon \
  --game 'thegame,mode=omni' \
  --steps 1600 \
  --schedule-steps 500 \
  --lr-schedule-steps 1600 \
  --rollout-games 64 \
  --evaluation-games 512 \
  --evaluation-every 50 \
  --save-every 25 \
  --keep-checkpoints 10 \
  --save-best \
  --inference-batch-size 1024 \
  --learner-batch-size 384 \
  --learning-rate 0.0008 \
  --weight-decay 0.01 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --epochs 1 \
  --gradient-clip 2.0 \
  --discount 1.0 \
  --gae-lambda 0.2 \
  --value-lambda 0.2 \
  --perplexity-start 5.0 \
  --perplexity-end 2 \
  --perplexity-curve 1.0 \
  --exploration-regularizer entropy \
  --entropy-strength 0.01 \
  --entropy-baseline-ratio 0.0 \
  --support-floor-mass 0.05 \
  --support-strength 0.15 \
  --value-strength 1.0 \
  --kl-target 0.01 \
  --kl-strength 0.05 \
  --ppo-clip 0.2 \
  --eval-temperature 0.02 \
  --warmup 20 \
  --min-lr-scale 0.5 \
  --seed 0 \
  --checkpoint-root checkpoints/time80 \
  --tag omni-scratch-gae1-support15e4-lr8e4-s0 \
  --trackio \
  --trackio-url https://visdom.vermeille.fr \
  --no-progress \
  "$@"
