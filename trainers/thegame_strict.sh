#!/usr/bin/env bash
set -euo pipefail

# Best from-scratch The Game recipe. Command-line arguments may override the
# strict-mode defaults, allowing sweep launchers to reuse this exact recipe.
# step 325: 80.400 points on 1,000 games (seed 123)
#           80.212 points on 1,000 games (seed 456)
uv run python trainers/coop.py \
  --device cuda \
  --architecture patchformer-small-p4 \
  --game 'thegame,mode=strict' \
  --steps 325 \
  --schedule-steps 250 \
  --rollout-games 512 \
  --evaluation-games 256 \
  --evaluation-every 25 \
  --save-every 25 \
  --inference-batch-size 1024 \
  --learner-batch-size 512 \
  --learning-rate 0.0003 \
  --weight-decay 0.01 \
  --adam-beta1 0.5 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --epochs 1 \
  --discount 1.0 \
  --gae-lambda 0.99 \
  --value-lambda 1.0 \
  --perplexity-start 0.8 \
  --perplexity-end 0.10 \
  --entropy-strength 0.1 \
  --value-strength 1.0 \
  --kl-target 0.006 \
  --kl-strength 0.5 \
  --ppo-clip 0.2 \
  --eval-temperature 0.02 \
  --warmup 8 \
  --min-lr-scale 1.0 \
  --seed 0 \
  --tag strict-fast \
  --visdom-url https://visdom.vermeille.fr \
  --visdom-port 443 \
  --no-progress \
  "$@"
