#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

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
# Use the capacity tested in cooperative training; keep it fixed so
# an inherited ARCHITECTURE environment variable cannot launch another model.
architecture="patchformer-medium-p8"
game="${GAME:-connectfour}"
opponent_eval_strategy="${OPPONENT_EVAL_STRATEGY:-tactical_random}"
fixed_training_opponent_fraction="${FIXED_TRAINING_OPPONENT_FRACTION:-0}"
# Candidate NFSP recipe: a bounded policy update supplies stability while
# exploration decays to zero. Playing strength still requires a full run;
# these defaults are not a claim of a validated 98% result.
steps="${STEPS:-2000}"
seed="${SEED:-0}"
rollout_games="${ROLLOUT_GAMES:-128}"
anticipatory="${ANTICIPATORY:-0.7}"
entropy_strength="${ENTROPY_STRENGTH:-0.0}"
entropy_baseline_ratio="${ENTROPY_BASELINE_RATIO:-0}"
perplexity_start="${PERPLEXITY_START:-6}"
perplexity_end="${PERPLEXITY_END:-2}"
perplexity_schedule_shape="${PERPLEXITY_SCHEDULE_SHAPE:-cosine}"
perplexity_adaptation_rate="${PERPLEXITY_ADAPTATION_RATE:-0.004}"
gae_lambda="${GAE_LAMBDA:-0.5}"
value_lambda="${VALUE_LAMBDA:-0.9}"
average_epochs="${AVERAGE_EPOCHS:-2}"
schedule_start="${SCHEDULE_START:-18}"
schedule_steps="${SCHEDULE_STEPS:-1764}"
lr_schedule_start="${LR_SCHEDULE_START:-900}"
lr_schedule_steps="${LR_SCHEDULE_STEPS:-899}"
lr_schedule_shape="${LR_SCHEDULE_SHAPE:-cosine}"
min_lr_scale="${MIN_LR_SCALE:-0.1}"
tag="${TAG:-adversarial}"
checkpoint_root="${CHECKPOINT_ROOT:-checkpoints/adversarial}"
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
"$python_bin" trainers/adversarial.py \
  --device "$device" \
  --architecture "$architecture" \
  --game "$game" \
  --opponent-eval-strategy "$opponent_eval_strategy" \
  --steps "$steps" \
  --schedule-steps "$schedule_steps" \
  --schedule-start "$schedule_start" \
  --lr-schedule-steps "$lr_schedule_steps" \
  --lr-schedule-start "$lr_schedule_start" \
  --lr-schedule-shape "$lr_schedule_shape" \
  --rollout-games "$rollout_games" \
  --evaluation-games "${EVALUATION_GAMES:-512}" \
  --evaluation-every "${EVALUATION_EVERY:-25}" \
  --save-every "${SAVE_EVERY:-50}" \
  --keep-checkpoints "${KEEP_CHECKPOINTS:-10}" \
  --save-best \
  --inference-batch-size "${INFERENCE_BATCH_SIZE:-1024}" \
  --learner-batch-size "${LEARNER_BATCH_SIZE:-384}" \
  --learning-rate "${LEARNING_RATE:-0.0001}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --adam-beta1 "${ADAM_BETA1:-0.9}" \
  --adam-beta2 "${ADAM_BETA2:-0.95}" \
  --adam-eps "${ADAM_EPS:-1e-5}" \
  --epochs "${EPOCHS:-1}" \
  --gradient-clip "${GRADIENT_CLIP:-5.0}" \
  --discount "${DISCOUNT:-1.0}" \
  --gae-lambda "$gae_lambda" \
  --value-lambda "$value_lambda" \
  --exploration-controller "${EXPLORATION_CONTROLLER:-linear}" \
  --exploration-regularizer "${EXPLORATION_REGULARIZER:-entropy}" \
  --perplexity-start "$perplexity_start" \
  --perplexity-end "$perplexity_end" \
  --perplexity-schedule-shape "$perplexity_schedule_shape" \
  --perplexity-curve "${PERPLEXITY_CURVE:-1.0}" \
  --perplexity-adaptation-rate "$perplexity_adaptation_rate" \
  --entropy-strength "$entropy_strength" \
  --entropy-baseline-ratio "$entropy_baseline_ratio" \
  --value-strength "${VALUE_STRENGTH:-1.0}" \
  --kl-target "${KL_TARGET:-0.02}" \
  --kl-strength "${KL_STRENGTH:-0.01}" \
  --ppo-clip "${PPO_CLIP:-0.2}" \
  --eval-temperature "${EVAL_TEMPERATURE:-0.02}" \
  --warmup "${WARMUP:-20}" \
  --min-lr-scale "$min_lr_scale" \
  --seed "$seed" \
  --checkpoint-root "$checkpoint_root" \
  --tag "$tag" \
  --anticipatory "$anticipatory" \
  --fixed-training-opponent-fraction "$fixed_training_opponent_fraction" \
  --average-epochs "$average_epochs" \
  "${trackio_args[@]}" \
  --no-progress \
  --rollout-max-steps "${ROLLOUT_MAX_STEPS:-5000}" \
  "$@" 2>&1 | tee "$log_path"
train_status=${PIPESTATUS[0]}
train_finished_at="$(date +%s)"
echo "TRAIN_WALL_SECONDS=$((train_finished_at - train_started_at))" | tee -a "$log_path"
echo "TRAIN_EXIT=$train_status" | tee -a "$log_path"
exit "$train_status"
