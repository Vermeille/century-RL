#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

TRAINER="${TRAINER:-adversarial-advshape}"
case "$TRAINER" in
adversarial-advshape | adversarial-ppo | adversarial-mmd) ;;
*)
  echo "unsupported TRAINER: $TRAINER" >&2
  exit 2
  ;;
esac
TAG="${TAG:-$TRAINER}"

trainer_args=(
  --game 2048versus,nolimits=True
  --steps 1800
  --learning-rate 6e-4
  --gae-lambda 0.
  --value-lambda 1.
  --adam-beta1 0.9
  --adam-beta2 0.95
  --inference-batch-size 256
  --learner-batch-size 128
  --rollout-games 128
  --evaluation-games 128
  --gradient-clip 10000
  --opponent-eval-strategy random
  --evaluation-every 10
  --save-every 10
  --seed "${SEED:-1}"
  --tag "$TAG"
  --no-progress
  --value-clip-epsilon none
)

# Both policies visit slide and placement turns under the shared trainer's
# seat rotation; targets stay within the slider's four-action branching factor.
AGENT_PPL_START="${AGENT_PPL_START:-3}"
AGENT_PPL_END="${AGENT_PPL_END:-1}"
ENVIRONMENT_PPL_START="${ENVIRONMENT_PPL_START:-3}"
ENVIRONMENT_PPL_END="${ENVIRONMENT_PPL_END:-2}"

case "$TRAINER" in
adversarial-advshape)
  trainer_args+=(
    --perplexity-start "$AGENT_PPL_START"
    --perplexity-end "$AGENT_PPL_END"
    --environment-perplexity-start "$ENVIRONMENT_PPL_START"
    --environment-perplexity-end "$ENVIRONMENT_PPL_END"
    --agent-threshold "${AGENT_THRESHOLD:-0.98}"
    --environment-threshold "${ENVIRONMENT_THRESHOLD:-0.7}"
    --kl-strength 0.01
    --kl-target 0.05
    --entropy-strength 0.1
    --environment-entropy-strength 0.5
  )
  ;;
adversarial-mmd)
  trainer_args+=(
    --mmd-coefficient "${MMD_COEFFICIENT:-0.05}"
    --mmd-reference-timesteps "${MMD_REFERENCE_TIMESTEPS:-10000000}"
  )
  ;;
adversarial-ppo) ;;
esac

uv_command=(uv run)
if [[ ${TRACKIO:-1} != 0 ]]; then
  uv_command+=(--extra trackio)
  trainer_args+=(--trackio)
fi

# Explicit CLI arguments win over launcher defaults.
trainer_args+=("$@")
trainer_command=(
  "${uv_command[@]}" python "trainers/${TRAINER}.py" "${trainer_args[@]}"
)

for argument in "$@"; do
  case "$argument" in
  -h | --help) exec "${trainer_command[@]}" ;;
  esac
done

if [[ ${ARENA:-1} == 0 ]]; then
  exec "${trainer_command[@]}"
fi

checkpoint_dir=$(
  uv run python trainers/arena.py checkpoint-dir \
    --trainer "$TRAINER" -- "${trainer_args[@]}"
)
exec "${uv_command[@]}" trainers/run_with_arena.sh \
  "$checkpoint_dir" "${trainer_command[@]}"
