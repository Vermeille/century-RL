#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# The agent and environment use independent perplexity targets over the same
# cosine schedule window.
AGENT_PPL_START="${AGENT_PPL_START:-5}"
AGENT_PPL_END="${AGENT_PPL_END:-1}"
ENVIRONMENT_PPL_START="${ENVIRONMENT_PPL_START:-5.}"
ENVIRONMENT_PPL_END="${ENVIRONMENT_PPL_END:-3.}"
AGENT_THRESHOLD="${AGENT_THRESHOLD:-0.98}"
ENVIRONMENT_THRESHOLD="${ENVIRONMENT_THRESHOLD:-0.7}"
TAG="${TAG:-adversarial-advshape}"

trainer_args=(
  --game take5,num_players=2 \
  --random-move-prob 0. \
  --steps 1800 \
  --perplexity-start "$AGENT_PPL_START" \
  --perplexity-end "$AGENT_PPL_END" \
  --environment-perplexity-start "$ENVIRONMENT_PPL_START" \
  --environment-perplexity-end "$ENVIRONMENT_PPL_END" \
  --agent-threshold "$AGENT_THRESHOLD" \
  --environment-threshold "$ENVIRONMENT_THRESHOLD" \
  --learning-rate 6e-4 \
  --kl-strength 0.01 \
  --kl-target 0.05 \
  --gae-lambda 0. \
  --value-lambda 1 \
  --adam-beta1 0.5 \
  --adam-beta2 0.95 \
  --inference-batch-size 1024 \
  --learner-batch-size 2048 \
  --rollout-games 512 \
  --evaluation-games 512 \
  --gradient-clip 10000 \
  --opponent-eval-strategy random \
  --entropy-strength 0.1 \
  --environment-entropy-strength 0.5 \
  --evaluation-every 10 \
  --seed 1 \
  --tag "$TAG" \
  --trackio \
  --no-progress \
  --value-clip-epsilon none
  "$@"
)
trainer_command=(
  uv run --extra trackio python trainers/adversarial-advshape.py "${trainer_args[@]}"
)

if [[ ${ARENA:-1} == 0 ]]; then
  exec "${trainer_command[@]}"
fi

checkpoint_dir=$(uv run python trainers/arena.py checkpoint-dir -- "${trainer_args[@]}")
exec trainers/run_with_arena.sh "$checkpoint_dir" "${trainer_command[@]}"
