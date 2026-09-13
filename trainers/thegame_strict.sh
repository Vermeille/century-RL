#!/usr/bin/env bash
set -euo pipefail

# Allow a follow-up experiment to be queued behind a running trainer without
# competing for the same GPU.
if [[ "${1:-}" == "--wait-for-pid" ]]; then
  wait_pid="${2:?--wait-for-pid requires a PID}"
  [[ "$wait_pid" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid PID: $wait_pid" >&2
    exit 2
  }
  shift 2
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 30
  done
fi

# Best from-scratch The Game recipe. Command-line arguments may override the
# strict-mode default, allowing the all-modes launcher to reuse this recipe.
# Exploration anneals over 350 steps. The independent 8e-4 LR schedule reaches
# approximately 3.7e-4 at step 1000, 2.9e-4 at step 1200, and its 2e-4 floor
# at step 1400.
# Omni, seed 0, step 1400: 80.096 points on 512 evaluation games.
source .venv/bin/activate
set -a
source .env
set +a

python trainers/coop.py \
  --device cuda \
  --architecture patchformer-medium-p4 \
  --game 'thegame,mode=strict' \
  --steps 1400 \
  --schedule-steps 500 \
  --schedule-start 0 \
  --lr-schedule-steps 1400 \
  --lr-schedule-start 0 \
  --lr-schedule-shape linear \
  --rollout-games 64 \
  --save-every 50 \
  --keep-checkpoints 1 \
  --learner-batch-size 512 \
  --adam-beta1 0.5 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --gae-lambda 0.98 \
  --value-lambda 1.0 \
  --perplexity-start 4 \
  --perplexity-end 1.3 \
  --entropy-strength 0.01 \
  --entropy-baseline-ratio 0.01 \
  --kl-target 0.01 \
  --kl-strength 0.05 \
  --checkpoint-root checkpoints/time80 \
  --tag thegame-medium-wide-k7-lr8e4-decay \
  --trackio \
  --no-progress \
  "$@"
