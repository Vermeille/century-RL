#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

set -a
source .env
set +a

python_bin="${PYTHON_BIN:-.venv/bin/python}"
device="${DEVICE:-cuda}"
if [[ "$device" == cuda* ]]; then
  # Python safely offloads before stopping the whole job. Keep this wrapper
  # and tee alive until that offload is complete.
  trap '' TSTP
fi
architecture="${ARCHITECTURE:-patchformer-medium-p8}"
game="${GAME:-thegame,mode=omni}"
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
trackio="${TRACKIO:-1}"

if [[ "$trackio" != "0" && "$trackio" != "1" ]]; then
  echo "TRACKIO must be 0 or 1" >&2
  exit 2
fi

trackio_args=()
if [[ "$trackio" == "1" ]]; then
  trackio_args+=(--trackio)
  if [[ -n "${TRACKIO_URL:-}" ]]; then
    trackio_args+=(--trackio-url "$TRACKIO_URL")
  fi
fi

mkdir -p "$(dirname -- "$log_path")"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set +e
train_started_at="$(date +%s)"
"$python_bin" trainers/coop.py \
  --device "$device" \
  --architecture "$architecture" \
  --game "$game" \
  --steps "$steps" \
  --schedule-steps "$steps" \
  --schedule-start 0 \
  --lr-schedule-steps "$steps" \
  --lr-schedule-start 0 \
  --rollout-games "$rollout_games" \
  --evaluation-games "${EVALUATION_GAMES:-512}" \
  --evaluation-every "${EVALUATION_EVERY:-50}" \
  --save-every "${SAVE_EVERY:-25}" \
  --keep-checkpoints "${KEEP_CHECKPOINTS:-10}" \
  --save-best \
  --inference-batch-size "${INFERENCE_BATCH_SIZE:-1024}" \
  --learner-batch-size "${LEARNER_BATCH_SIZE:-384}" \
  --learning-rate "${LEARNING_RATE:-0.0008}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --adam-beta1 "${ADAM_BETA1:-0.9}" \
  --adam-beta2 "${ADAM_BETA2:-0.95}" \
  --adam-eps "${ADAM_EPS:-1e-5}" \
  --epochs "${EPOCHS:-1}" \
  --gradient-clip "${GRADIENT_CLIP:-5.0}" \
  --discount "${DISCOUNT:-1.0}" \
  --gae-lambda "${GAE_LAMBDA:-0.1}" \
  --value-lambda "${VALUE_LAMBDA:-0.9}" \
  --exploration-controller "${EXPLORATION_CONTROLLER:-thermostat}" \
  --perplexity-start "$target_perplexity" \
  --perplexity-end "$target_perplexity" \
  --perplexity-adaptation-rate "$perplexity_adaptation_rate" \
  --entropy-strength "$entropy_strength" \
  --entropy-baseline-ratio "$entropy_baseline_ratio" \
  --value-strength "${VALUE_STRENGTH:-1.0}" \
  --kl-target "${KL_TARGET:-0.05}" \
  --kl-strength "${KL_STRENGTH:-0.05}" \
  --ppo-clip "${PPO_CLIP:-0.2}" \
  --eval-temperature "${EVAL_TEMPERATURE:-0.02}" \
  --warmup "${WARMUP:-20}" \
  --min-lr-scale "${MIN_LR_SCALE:-1.0}" \
  --seed "$seed" \
  --checkpoint-root "$checkpoint_root" \
  --tag "$tag" \
  "${trackio_args[@]}" \
  --no-progress \
  "$@" 2>&1 | tee "$log_path"
train_status=${PIPESTATUS[0]}
train_finished_at="$(date +%s)"
echo "TRAIN_WALL_SECONDS=$((train_finished_at - train_started_at))" | tee -a "$log_path"
echo "TRAIN_EXIT=$train_status" | tee -a "$log_path"
exit "$train_status"
