#!/usr/bin/env bash
# Shared recipes and process setup for the root train_*.sh launchers.

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Multi-trainer recipes can supply arguments specific to one algorithm.
advshape_overrides=()
ppo_overrides=()

launch_training() {
  local trainer=$1 mode=$2
  shift 2
  local argument trackio_enabled=${TRACKIO:-1}
  local -a uv_command=(uv run) trainer_args=(--no-progress)

  # An explicit CLI request for Trackio also selects its optional dependency.
  for argument in "$@"; do
    if [[ $argument == --trackio ]]; then
      trackio_enabled=1
    fi
  done
  if [[ $trackio_enabled != 0 ]]; then
    uv_command+=(--extra trackio)
    trainer_args+=(--trackio)
  fi
  trainer_args+=("$@")
  local -a trainer_command=(
    "${uv_command[@]}" python "trainers/${trainer}.py" "${trainer_args[@]}"
  )

  for argument in "$@"; do
    case "$argument" in
    -h | --help) exec "${trainer_command[@]}" ;;
    esac
  done
  if [[ $mode == direct || ( $mode == arena && ${ARENA:-1} == 0 ) ]]; then
    exec "${trainer_command[@]}"
  fi

  local checkpoint_dir
  checkpoint_dir=$(
    uv run python trainers/arena.py checkpoint-dir \
      --trainer "$trainer" -- "${trainer_args[@]}"
  ) || return "$?"
  local -a run_command=("${trainer_command[@]}")
  if [[ ${ARENA:-1} != 0 ]]; then
    # Run the supervisor in the same uv environment as its two Python children.
    run_command=(
      "${uv_command[@]}" trainers/run_with_arena.sh \
      "$checkpoint_dir" "${trainer_command[@]}"
    )
  fi
  if [[ $mode == openspiel ]]; then
    "${run_command[@]}" || return "$?"
    if [[ ${FINAL_EXPLOITABILITY:-1} != 0 ]]; then
      exec uv run python evaluate_nashconv.py --training-run "$checkpoint_dir"
    fi
  else
    exec "${run_command[@]}"
  fi
}

train_coop() {
  launch_training coop direct \
    --steps 900 \
    --learning-rate 4e-4 \
    --perplexity-end 1 \
    --kl-strength 0.1 \
    --kl-target 0.1 \
    --tag YOLO \
    --seed 1 \
    "$@"
}

adversarial_recipe() {
  local trainer=$1 mode=$2
  shift 2
  local -a algorithm_args=()
  case "$trainer" in
  adversarial-advshape)
    algorithm_args=(
      --perplexity-start "${AGENT_PPL_START:-6}"
      --perplexity-end "${AGENT_PPL_END:-1}"
      --environment-perplexity-start "${ENVIRONMENT_PPL_START:-6}"
      --environment-perplexity-end "${ENVIRONMENT_PPL_END:-3}"
      --agent-threshold "${AGENT_THRESHOLD:-0.7}"
      --environment-threshold "${ENVIRONMENT_THRESHOLD:-0.7}"
      --kl-strength 0.01
      --kl-target 0.05
      --entropy-strength 0.1
      --environment-entropy-strength 0.5
      "${advshape_overrides[@]}"
    )
    ;;
  adversarial-ppo)
    algorithm_args=("${ppo_overrides[@]}")
    ;;
  adversarial-mmd)
    algorithm_args=(
      --mmd-coefficient "${MMD_COEFFICIENT:-0.05}"
      --mmd-reference-timesteps "${MMD_REFERENCE_TIMESTEPS:-10000000}"
    )
    ;;
  *)
    echo "unsupported TRAINER: $trainer" >&2
    return 2
    ;;
  esac

  # The existing Santorini recipe is the common adversarial baseline.
  # Algorithm defaults precede game overrides, which precede user CLI arguments.
  launch_training "$trainer" "$mode" \
    --random-move-prob 0 \
    --steps 600 \
    --learning-rate 6e-4 \
    --gae-lambda 0.2 \
    --value-lambda 0.5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --inference-batch-size 256 \
    --learner-batch-size 128 \
    --rollout-games 128 \
    --evaluation-games 256 \
    --gradient-clip 10000 \
    --opponent-eval-strategy random \
    --evaluation-every 10 \
    --save-every 10 \
    --seed 1 \
    --tag "${TAG:-$trainer}" \
    --value-clip-epsilon 1 \
    "${algorithm_args[@]}" "$@"
}

train_adversarial() {
  local trainer=$1
  shift
  adversarial_recipe "$trainer" arena "$@"
}

train_openspiel() {
  local trainer=$1
  shift
  adversarial_recipe "$trainer" openspiel "$@"
}
