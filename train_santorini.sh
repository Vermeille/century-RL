#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/trainers/launch.sh"

train_adversarial adversarial-advshape --game santorini "$@"
