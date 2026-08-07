#!/usr/bin/env bash
set -euo pipefail

if (( $# )); then
  modes=("$@")
else
  modes=(
    strict
    free
    omni
    strict_message_before_draw
    free_message_before_draw
    strict_message_after_draw
    free_message_after_draw
  )
fi

for mode in "${modes[@]}"; do
  trainers/thegame_strict.sh \
    --game "thegame,mode=${mode}" \
    --tag "thegame_${mode}"
done
