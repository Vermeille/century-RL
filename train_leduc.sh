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
SEED="${SEED:-1}"

trainer_args=(
  --game leduc
  --random-move-prob 0.
  --steps 2000
  --learning-rate 2e-5
  --gae-lambda 0.8
  --value-lambda 0.9
  --adam-beta1 0.9
  --adam-beta2 0.95
  --inference-batch-size 1024
  --learner-batch-size 256
  --rollout-games 1024
  --evaluation-games 512
  --gradient-clip 10000
  --opponent-eval-strategy random
  --evaluation-every 25
  --seed $SEED
  --tag "$TAG"
  --trackio
  --no-progress
  --value-clip-epsilon none
)

AGENT_PPL_START="${AGENT_PPL_START:-2.0}"
AGENT_PPL_END="${AGENT_PPL_END:-1.5}"
ENVIRONMENT_PPL_START="${ENVIRONMENT_PPL_START:-2.2}"
ENVIRONMENT_PPL_END="${ENVIRONMENT_PPL_END:-1.9}"
AGENT_THRESHOLD="${AGENT_THRESHOLD:-0.7}"
ENVIRONMENT_THRESHOLD="${ENVIRONMENT_THRESHOLD:-0.7}"

case "$TRAINER" in
adversarial-advshape)
  # The agent and environment use independent perplexity targets over the
  # same cosine schedule window.
  trainer_args+=(
    --perplexity-start "$AGENT_PPL_START"
    --perplexity-end "$AGENT_PPL_END"
    --environment-perplexity-start "$ENVIRONMENT_PPL_START"
    --environment-perplexity-end "$ENVIRONMENT_PPL_END"
    --agent-threshold "$AGENT_THRESHOLD"
    --environment-threshold "$ENVIRONMENT_THRESHOLD"
    --kl-strength 0.01
    --kl-target 0.005
    --entropy-strength 0.1
    --environment-entropy-strength 0.1
  )
  ;;
adversarial-mmd)
  trainer_args+=(
    --mmd-coefficient "${MMD_COEFFICIENT:-0.05}"
    --mmd-reference-timesteps "${MMD_REFERENCE_TIMESTEPS:-10000000}"
  )
  ;;
adversarial-ppo)
  trainer_args+=(
    --perplexity-start "$AGENT_PPL_START"
    --perplexity-end "$AGENT_PPL_END"
    --entropy-strength 0.1
  )
  ;;
esac

# Explicit CLI arguments win over launcher defaults.
trainer_args+=("$@")
trainer_command=(
  uv run --extra trackio python "trainers/${TRAINER}.py" "${trainer_args[@]}"
)

if [[ ${ARENA:-1} == 0 ]]; then
  exec "${trainer_command[@]}"
fi

checkpoint_dir=$(
  uv run python trainers/arena.py checkpoint-dir \
    --trainer "$TRAINER" -- "${trainer_args[@]}"
)
${trainer_command[@]}
