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
  --initialize-from 'checkpoints/time80/coop/thegame,mode=omni/patchformer-medium-p4-wide-canon/prenomvaluehead-rmnorms-14/best-1400.pth' \
  --steps 400 \
  --schedule-steps 150 \
  --lr-schedule-steps 400 \
  --rollout-games 64 \
  --evaluation-games 1024 \
  --evaluation-every 50 \
  --save-every 25 \
  --keep-checkpoints 10 \
  --save-best \
  --inference-batch-size 1024 \
  --learner-batch-size 384 \
  --learning-rate 0.00005 \
  --weight-decay 0.01 \
  --adam-beta1 0.5 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --epochs 1 \
  --gradient-clip 2.0 \
  --discount 1.0 \
  --gae-lambda 0.98 \
  --value-lambda 1.0 \
  --perplexity-start 1.10 \
  --perplexity-end 1.05 \
  --perplexity-curve 1.0 \
  --exploration-regularizer entropy \
  --entropy-strength 0.001 \
  --entropy-baseline-ratio 0.0 \
  --support-floor-mass 0.01 \
  --support-strength 0.001 \
  --value-strength 1.0 \
  --kl-target 0.01 \
  --kl-strength 0.05 \
  --ppo-clip 0.2 \
  --eval-temperature 0.02 \
  --warmup 10 \
  --min-lr-scale 0.2 \
  --seed 1 \
  --checkpoint-root checkpoints/time80 \
  --tag omni-rm14-supportfloor1pct-s1 \
  --trackio \
  --trackio-url https://visdom.vermeille.fr \
  --no-progress
