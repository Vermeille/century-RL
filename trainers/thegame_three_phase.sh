#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

set -a
source .env
set +a

python_bin="${PYTHON_BIN:-.venv/bin/python}"
if [[ "${DEVICE:-cuda}" == cuda* ]]; then
  # The active Python trainer stops the whole job after safely offloading CUDA.
  trap '' TSTP
fi
total_steps="${TOTAL_STEPS:-${STEPS:-2400}}"
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
architecture="${ARCHITECTURE:-patchformer-medium-p8}"
exploration_controller="${EXPLORATION_CONTROLLER:-thermostat}"
entropy_strength="${ENTROPY_STRENGTH:-0.1}"
entropy_baseline_ratio="${ENTROPY_BASELINE_RATIO:-0.05}"
support_floor_mass="${SUPPORT_FLOOR_MASS:-0.05}"
support_strength="${SUPPORT_STRENGTH:-0.001}"
use_support_floor="${USE_SUPPORT_FLOOR:-0}"
if [[ "$exploration_controller" != "thermostat" && "$exploration_controller" != "linear" ]]; then
  echo "EXPLORATION_CONTROLLER must be thermostat or linear" >&2
  exit 2
fi
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
phase2_resume="${PHASE2_RESUME:-$phase1_checkpoint}"
phase3_resume="${PHASE3_RESUME:-$phase2_checkpoint}"
start_phase="${START_PHASE:-1}"
end_phase="${END_PHASE:-3}"
phase2_reset_exploration_state="${PHASE2_RESET_EXPLORATION_STATE:-0}"

if [[ "$start_phase" != "1" && "$start_phase" != "2" && "$start_phase" != "3" ]]; then
  echo "START_PHASE must be 1, 2, or 3" >&2
  exit 2
fi
if [[ "$end_phase" != "1" && "$end_phase" != "2" && "$end_phase" != "3" ]]; then
  echo "END_PHASE must be 1, 2, or 3" >&2
  exit 2
fi
if (( start_phase > end_phase )); then
  echo "START_PHASE must not be greater than END_PHASE" >&2
  exit 2
fi
if [[ "$phase2_reset_exploration_state" != "0" && "$phase2_reset_exploration_state" != "1" ]]; then
  echo "PHASE2_RESET_EXPLORATION_STATE must be 0 or 1" >&2
  exit 2
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
  local reset_exploration_state="${14:-0}"

  local phase_args=(
    --steps "$steps"
    --schedule-start "$schedule_start"
    --schedule-steps "$schedule_steps"
    --lr-schedule-start "$lr_schedule_start"
    --lr-schedule-steps "$lr_schedule_steps"
    --min-lr-scale "$min_lr_scale"
    --warmup "$phase_warmup"
  )
  if [[ "$exploration_controller" == "linear" ]]; then
    phase_args+=(
      --entropy-strength "$phase_entropy_strength"
      --entropy-baseline-ratio "$entropy_baseline_ratio"
    )
  fi
  if [[ "$use_support_floor" == "1" ]]; then
    phase_args+=(--support-floor-mass "$support_floor_mass")
    if [[ "$exploration_controller" == "linear" ]]; then
      phase_args+=(
        --support-strength "$phase_support_strength"
        --support-strength-end "$phase_support_strength_end"
      )
    else
      phase_args+=(--support-strength "$support_strength")
    fi
  fi
  if [[ -n "$resume_path" ]]; then
    if [[ ! -f "$resume_path" ]]; then
      echo "Missing phase resume checkpoint: $resume_path" >&2
      return 2
    fi
    # Coop --resume restores model, optimizer, and learner state.
    phase_args+=(--resume "$resume_path")
  fi
  if [[ "$reset_exploration_state" == "1" ]]; then
    phase_args+=(--resume-reset-exploration-state)
  fi

  STEPS="$steps" \
  SEED="$seed" \
  TAG="$phase_tag" \
  LOG_PATH="training-logs/${phase_tag}.log" \
  CHECKPOINT_ROOT="$checkpoint_root" \
  GAME="$game" \
  ARCHITECTURE="$architecture" \
  "$script_dir/thegame_omni_scratch.sh" "${phase_args[@]}"
}

# Phase 1: 70% of the budget with the Omni defaults and a flat LR.
if (( start_phase <= 1 && end_phase >= 1 )); then
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
fi

# Phase 2: linearly decay entropy strength to 0.1x with the linear controller.
# The default thermostat instead keeps its perplexity target and continues.
if (( start_phase <= 2 && end_phase >= 2 )); then
  run_phase \
    "$phase2_steps" \
    "$phase1_steps" "$phase2_duration" \
    "$phase1_steps" "$phase2_duration" \
    1.0 \
    "$entropy_strength" 0.1 \
    0 \
    "$support_strength" "$phase2_support_strength" \
    "$phase2_tag" \
    "$phase2_resume" \
    "$phase2_reset_exploration_state"
fi

# Phase 3: linearly decay the learning rate to zero while holding exploration
# at the phase-2 endpoint (or continuing the default thermostat target).
if (( start_phase <= 3 && end_phase >= 3 )); then
  run_phase \
    "$total_steps" \
    "$phase2_steps" "$phase3_duration" \
    "$phase2_steps" "$lr_decay_steps" \
    0.0 \
    "$phase2_entropy_strength" 1.0 \
    0 \
    "$phase2_support_strength" "$phase2_support_strength" \
    "$phase3_tag" \
    "$phase3_resume"
fi
