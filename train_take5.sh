#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
set -a
source .env
set +a

# The agent and environment use independent perplexity targets over the same
# cosine schedule window.
AGENT_PPL_START="${AGENT_PPL_START:-5}"
AGENT_PPL_END="${AGENT_PPL_END:-1}"
ENVIRONMENT_PPL_START="${ENVIRONMENT_PPL_START:-5.}"
ENVIRONMENT_PPL_END="${ENVIRONMENT_PPL_END:-3.}"
AGENT_THRESHOLD="${AGENT_THRESHOLD:-0.98}"
ENVIRONMENT_THRESHOLD="${ENVIRONMENT_THRESHOLD:-0.7}"
TAG="${TAG:-adversarial-advshape}"

# NFSP-only arguments have no adversarial-advshape equivalent:
#   --anticipatory 0.5
#   --average-learning-rate 4e-4  # both PPO optimizers use --learning-rate
#   --reservoir-capacity 1000000

exec uv run python trainers/adversarial-advshape.py \
  --game century \
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
  --opponent-eval-strategy tempo_greedy \
  --entropy-strength 0.1 \
  --environment-entropy-strength 0.5 \
  --evaluation-every 10 \
  --seed 1 \
  --tag "$TAG" \
  --trackio \
  --no-progress \
  --value-clip-epsilon none \
  "$@"
