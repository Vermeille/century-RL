#!/usr/bin/env bash
set -euo pipefail

AGENT_PPL_START=4 ENVIRONMENT_PPL_START=5 ENVIRONMENT_PPL_END=3 ./train_leduc.sh --game phantom_ttt --steps 800 --evaluation-every 200
