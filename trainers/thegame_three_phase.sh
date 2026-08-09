#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

set -a
source .env
set +a

python_bin="${PYTHON_BIN:-.venv/bin/python}"
total_steps="${TOTAL_STEPS:-1600}"
phase1_steps=$((total_steps * 70 / 100))
phase2_steps=$((total_steps * 90 / 100))
phase1_duration="$phase1_steps"
phase2_duration=$((phase2_steps - phase1_steps))
phase3_duration=$((total_steps - phase2_steps))
lr_decay_steps=$((phase3_duration - 1))

if (( phase1_steps < 1 || phase2_duration < 1 || phase3_duration < 2 )); then
  echo "TOTAL_STEPS is too small for the three phases: $total_steps" >&2
  exit 2
fi

game="${GAME:-thegame,mode=omni}"
architecture="${ARCHITECTURE:-patchformer-medium-p4}"
learning_rate="${LEARNING_RATE:-0.0008}"
entropy_strength="${ENTROPY_STRENGTH:-0.01}"
support_floor_mass="${SUPPORT_FLOOR_MASS:-0.05}"
support_strength="${SUPPORT_STRENGTH:-0.015}"
use_support_floor="${USE_SUPPORT_FLOOR:-1}"
if [[ "$use_support_floor" != "0" && "$use_support_floor" != "1" ]]; then
  echo "USE_SUPPORT_FLOOR must be 0 or 1" >&2
  exit 2
fi
phase1_warmup="${WARMUP:-20}"
phase2_entropy_strength="$("$python_bin" -c \
  'import sys; print(float(sys.argv[1]) / 10.0)' "$entropy_strength")"
phase2_support_strength="$("$python_bin" -c \
  'import sys; print(float(sys.argv[1]) / 10.0)' "$support_strength")"
checkpoint_root="${CHECKPOINT_ROOT:-checkpoints/three-phase}"
base_tag="${TAG:-thegame-three-phase-omni-scratch}"
phase1_tag="${base_tag}-phase1"
phase2_tag="${base_tag}-phase2"
phase3_tag="${base_tag}-phase3"
seed="${SEED:-0}"
phase1_resume="${PHASE1_RESUME:-}"

phase1_checkpoint="$checkpoint_root/coop/$game/$architecture/$phase1_tag/step-$phase1_steps.pth"
phase2_checkpoint="$checkpoint_root/coop/$game/$architecture/$phase2_tag/step-$phase2_steps.pth"

common_args=(
  --device "${DEVICE:-cuda}"
  --architecture "$architecture"
  --game "$game"
  --rollout-games "${ROLLOUT_GAMES:-64}"
  --evaluation-games "${EVALUATION_GAMES:-512}"
  --evaluation-every "${EVALUATION_EVERY:-50}"
  --save-every "${SAVE_EVERY:-25}"
  --keep-checkpoints "${KEEP_CHECKPOINTS:-10}"
  --save-best
  --inference-batch-size "${INFERENCE_BATCH_SIZE:-1024}"
  --learner-batch-size "${LEARNER_BATCH_SIZE:-384}"
  --learning-rate "$learning_rate"
  --weight-decay "${WEIGHT_DECAY:-0.01}"
  --adam-beta1 "${ADAM_BETA1:-0.9}"
  --adam-beta2 "${ADAM_BETA2:-0.999}"
  --adam-eps "${ADAM_EPS:-1e-8}"
  --epochs "${EPOCHS:-1}"
  --gradient-clip "${GRADIENT_CLIP:-2.0}"
  --discount "${DISCOUNT:-1.0}"
  --gae-lambda "${GAE_LAMBDA:-0.2}"
  --value-lambda "${VALUE_LAMBDA:-0.2}"
  --exploration-controller linear
  --exploration-regularizer entropy
  --value-strength "${VALUE_STRENGTH:-1.0}"
  --kl-target "${KL_TARGET:-0.01}"
  --kl-strength "${KL_STRENGTH:-0.05}"
  --ppo-clip "${PPO_CLIP:-0.2}"
  --eval-temperature "${EVAL_TEMPERATURE:-0.02}"
  --seed "$seed"
  --checkpoint-root "$checkpoint_root"
  --no-progress
)

if [[ "$use_support_floor" == "1" ]]; then
  common_args+=(--support-floor-mass "$support_floor_mass")
fi

if [[ "${TRACKIO:-1}" == "1" ]]; then
  common_args+=(--trackio)
  if [[ -n "${TRACKIO_URL:-}" ]]; then
    common_args+=(--trackio-url "$TRACKIO_URL")
  fi
fi

run_phase() {
  local steps="$1"
  local schedule_start="$2"
  local schedule_steps="$3"
  local lr_schedule_start="$4"
  local lr_schedule_steps="$5"
  local min_lr_scale="$6"
  local phase_entropy_strength="$7"
  local entropy_baseline_ratio="$8"
  local phase_warmup="$9"
  local phase_support_strength="${10}"
  local phase_support_strength_end="${11}"
  local phase_tag="${12}"
  local resume_path="${13:-}"

  local phase_args=(
    "${common_args[@]}"
    --steps "$steps"
    --schedule-start "$schedule_start"
    --schedule-steps "$schedule_steps"
    --lr-schedule-start "$lr_schedule_start"
    --lr-schedule-steps "$lr_schedule_steps"
    --min-lr-scale "$min_lr_scale"
    --entropy-strength "$phase_entropy_strength"
    --entropy-baseline-ratio "$entropy_baseline_ratio"
    --warmup "$phase_warmup"
    --tag "$phase_tag"
  )
  if [[ "$use_support_floor" == "1" ]]; then
    phase_args+=(
      --support-strength "$phase_support_strength"
      --support-strength-end "$phase_support_strength_end"
    )
  fi
  if [[ -n "$resume_path" ]]; then
    if [[ ! -f "$resume_path" ]]; then
      echo "Missing phase resume checkpoint: $resume_path" >&2
      return 2
    fi
    # Coop --resume restores model, optimizer, and learner state.
    phase_args+=(--resume "$resume_path")
  fi

  "$python_bin" trainers/coop.py "${phase_args[@]}"
}

# Phase 1: 70% of the budget, with flat LR and regularizer strengths.
run_phase \
  "$phase1_steps" \
  0 "$phase1_duration" \
  0 "$phase1_duration" \
  1.0 \
  "$entropy_strength" 1.0 \
  "$phase1_warmup" \
  "$support_strength" "$support_strength" \
  "$phase1_tag" \
  "$phase1_resume"

# Phase 2: 20% more, decaying regularizers to one tenth of their base strengths.
run_phase \
  "$phase2_steps" \
  "$phase1_steps" "$phase2_duration" \
  "$phase1_steps" "$phase2_duration" \
  1.0 \
  "$entropy_strength" 0.1 \
  0 \
  "$support_strength" "$phase2_support_strength" \
  "$phase2_tag" \
  "$phase1_checkpoint"

# Phase 3: final 10%, hold the reduced entropy and decay the LR to zero.
run_phase \
  "$total_steps" \
  "$phase2_steps" "$phase3_duration" \
  "$phase2_steps" "$lr_decay_steps" \
  0.0 \
  "$phase2_entropy_strength" 1.0 \
  0 \
  "$phase2_support_strength" "$phase2_support_strength" \
  "$phase3_tag" \
  "$phase2_checkpoint"
