#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

set -a
source .env
set +a

python_bin="${PYTHON_BIN:-.venv/bin/python}"
steps="${STEPS:-2400}"
seed="${SEED:-0}"
rollout_games="${ROLLOUT_GAMES:-128}"
entropy_strength="${ENTROPY_STRENGTH:-0.1}"
entropy_baseline_ratio="${ENTROPY_BASELINE_RATIO:-0.05}"
target_perplexity="${TARGET_PERPLEXITY:-1.5}"
perplexity_adaptation_rate="${PERPLEXITY_ADAPTATION_RATE:-0.004}"
tag="${TAG:-value09-gae01-ppl15-altgaenorm2-lr8e4nonorm-altadam}"
checkpoint_root="${CHECKPOINT_ROOT:-checkpoints/three-phase}"
log_path="${LOG_PATH:-training-logs/${tag}.log}"

mkdir -p "$(dirname -- "$log_path")"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set +e
train_started_at="$(date +%s)"
"$python_bin" trainers/coop.py \
  --device cuda \
  --architecture patchformer-medium-p8 \
  --game 'thegame,mode=omni' \
  --steps "$steps" \
  --schedule-steps "$steps" \
  --schedule-start 0 \
  --lr-schedule-steps "$steps" \
  --lr-schedule-start 0 \
  --rollout-games "$rollout_games" \
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
  --adam-beta2 0.95 \
  --adam-eps 1e-5 \
  --epochs 1 \
  --gradient-clip 5.0 \
  --discount 1.0 \
  --gae-lambda 0.1 \
  --value-lambda 0.9 \
  --exploration-controller thermostat \
  --exploration-regularizer entropy \
  --perplexity-start "$target_perplexity" \
  --perplexity-end "$target_perplexity" \
  --perplexity-curve 1.0 \
  --perplexity-adaptation-rate "$perplexity_adaptation_rate" \
  --entropy-strength "$entropy_strength" \
  --entropy-baseline-ratio "$entropy_baseline_ratio" \
  --value-strength 1.0 \
  --kl-target 0.05 \
  --kl-strength 0.05 \
  --ppo-clip 0.2 \
  --eval-temperature 0.02 \
  --warmup 20 \
  --min-lr-scale 1.0 \
  --seed "$seed" \
  --checkpoint-root "$checkpoint_root" \
  --tag "$tag" \
  --trackio \
  --no-progress \
  "$@" 2>&1 | tee "$log_path"
train_status=${PIPESTATUS[0]}
train_finished_at="$(date +%s)"
echo "TRAIN_WALL_SECONDS=$((train_finished_at - train_started_at))" | tee -a "$log_path"
echo "TRAIN_EXIT=$train_status" | tee -a "$log_path"
exit "$train_status"
