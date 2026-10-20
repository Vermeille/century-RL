#!/usr/bin/env bash

# WORK IN PROGRESS

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

AGENT_PPL_START=2 AGENT_PPL_END=1.5 ENVIRONMENT_PPL_START=2.2 ENVIRONMENT_PPL_END=1.9 ./train_openspiel.sh --game leduc "$@"
AGENT_PPL_START=4 ENVIRONMENT_PPL_START=5 ENVIRONMENT_PPL_END=4 ./train_openspiel.sh --game phantom_ttt --steps 800 --evaluation-every 20 --save-every 20 --learning-rate 5e-5 "$@"
