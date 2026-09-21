#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
set -a
source .env
set +a

# The agent and environment use independent perplexity targets over the same
# cosine schedule window.
AGENT_PPL_START="${AGENT_PPL_START:-6}"
AGENT_PPL_END="${AGENT_PPL_END:-1}"
ENVIRONMENT_PPL_START="${ENVIRONMENT_PPL_START:-6}"
ENVIRONMENT_PPL_END="${ENVIRONMENT_PPL_END:-3}"
AGENT_THRESHOLD="${AGENT_THRESHOLD:-0.7}"
ENVIRONMENT_THRESHOLD="${ENVIRONMENT_THRESHOLD:-0.7}"
TAG="${TAG:-adversarial-advshape}"

trainer_args=(
  --game santorini
  --random-move-prob 0.
  --steps 600
  --perplexity-start "$AGENT_PPL_START"
  --perplexity-end "$AGENT_PPL_END"
  --environment-perplexity-start "$ENVIRONMENT_PPL_START"
  --environment-perplexity-end "$ENVIRONMENT_PPL_END"
  --agent-threshold "$AGENT_THRESHOLD"
  --environment-threshold "$ENVIRONMENT_THRESHOLD"
  --learning-rate 6e-4
  --kl-strength 0.01
  --kl-target 0.05
  --gae-lambda 0.5
  --value-lambda 0.9
  --adam-beta1 0.9
  --adam-beta2 0.95
  --inference-batch-size 256
  --learner-batch-size 128
  --rollout-games 64
  --evaluation-games 128
  --gradient-clip 10000
  --opponent-eval-strategy random
  --entropy-strength 0.1
  --environment-entropy-strength 0.5
  --evaluation-every 10
  --save-every 10
  --seed 1
  --tag "$TAG"
  --trackio
  --no-progress
  --value-clip-epsilon 1
  "$@"
)
trainer_command=(
  python trainers/adversarial-advshape.py "${trainer_args[@]}"
)

if [[ ${ARENA:-1} == 0 ]]; then
  exec "${trainer_command[@]}"
fi

checkpoint_dir=$(python trainers/arena.py checkpoint-dir -- "${trainer_args[@]}")
exec trainers/run_with_arena.sh "$checkpoint_dir" "${trainer_command[@]}"
