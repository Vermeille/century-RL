#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"
total_steps="${TOTAL_STEPS:-600}"
recipe="${RECIPE:-early-decay}"

case "$recipe" in
full-decay)
  default_ppl_start=1
  default_ppl_end=99
  ;;
early-decay)
  default_ppl_start=10
  default_ppl_end=45
  ;;
late-decay)
  default_ppl_start=45
  default_ppl_end=70
  ;;
balanced)
  default_ppl_start=20
  default_ppl_end=55
  ;;
*)
  echo "RECIPE must be early-decay, late-decay, or balanced" >&2
  exit 2
  ;;
esac

ppl_start_percent="${PPL_DECAY_START_PERCENT:-$default_ppl_start}"
ppl_end_percent="${PPL_DECAY_END_PERCENT:-$default_ppl_end}"
lr_start_percent="${LR_DECAY_START_PERCENT:-80}"
lr_end_percent="${LR_DECAY_END_PERCENT:-100}"

percentage_step() {
  local percent="$1"
  echo $((total_steps * percent / 100))
}

ppl_start_step="$(percentage_step "$ppl_start_percent")"
ppl_end_step="$(percentage_step "$ppl_end_percent")"
lr_start_step="$(percentage_step "$lr_start_percent")"
lr_end_step="$(percentage_step "$lr_end_percent")"
ppl_steps=$((ppl_end_step - ppl_start_step))
lr_steps=$((lr_end_step - lr_start_step - 1))

if ((ppl_steps < 1 || lr_steps < 1)); then
  echo "Invalid schedule boundaries: ppl=${ppl_start_percent}-${ppl_end_percent}%, lr=${lr_start_percent}-${lr_end_percent}%" >&2
  exit 2
fi

high_ppl="${HIGH_PERPLEXITY:-2.5}"
medium_ppl="${MEDIUM_PERPLEXITY:-1.5}"
tag="${TAG:-omni-${recipe}-ppl${ppl_start_percent}-${ppl_end_percent}-lr${lr_start_percent}-${lr_end_percent}-n${total_steps}-s${SEED:-0}}"

snapshot_training_source() {
  local run_tag="$1"
  local snapshot_index snapshot_tree safe_tag timestamp identity_name identity_email

  snapshot_index="$(mktemp)"
  rm -f -- "$snapshot_index"
  GIT_INDEX_FILE="$snapshot_index" git read-tree HEAD

  # Snapshot every tracked working-tree change plus untracked runtime source.
  # Deliberately exclude .env, checkpoints, and logs from the synthetic commit.
  GIT_INDEX_FILE="$snapshot_index" git add -u -- .
  GIT_INDEX_FILE="$snapshot_index" git add -- boardrl trainers examples
  snapshot_tree="$(GIT_INDEX_FILE="$snapshot_index" git write-tree)"
  rm -f -- "$snapshot_index"

  identity_name="$(git config user.name || true)"
  identity_email="$(git config user.email || true)"
  identity_name="${identity_name:-Training Source Snapshot}"
  identity_email="${identity_email:-training-snapshot@localhost}"
  source_commit="$({
    echo "Training source snapshot for $run_tag"
    echo
    echo "Base commit: $(git rev-parse HEAD)"
    echo "Launcher: trainers/thegame_omni_schedule.sh"
  } | GIT_AUTHOR_NAME="$identity_name" GIT_AUTHOR_EMAIL="$identity_email" \
      GIT_COMMITTER_NAME="$identity_name" GIT_COMMITTER_EMAIL="$identity_email" \
      git commit-tree "$snapshot_tree" -p HEAD)"

  safe_tag="$(printf '%s' "$run_tag" | sed -E 's/[^A-Za-z0-9_+-]+/-/g; s/^-+//; s/-+$//' | cut -c1-80)"
  safe_tag="${safe_tag:-run}"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  source_tag="run-source/${safe_tag}-${timestamp}-${source_commit:0:12}"
  git check-ref-format "refs/tags/$source_tag" >/dev/null
  git tag "$source_tag" "$source_commit"
}

snapshot_training_source "$tag"
echo "SOURCE_COMMIT=$source_commit"
echo "SOURCE_TAG=$source_tag"
echo "Inspect: git show $source_tag"
echo "Diff from base: git diff $source_tag^ $source_tag"

STEPS="$total_steps" \
  TAG="$tag" \
  SOURCE_COMMIT="$source_commit" \
  SOURCE_TAG="$source_tag" \
  LOG_PATH="${LOG_PATH:-training-logs/${tag}.log}" \
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/schedule-search}" \
  "$script_dir/thegame_omni_scratch.sh" \
  --schedule-start "$ppl_start_step" \
  --schedule-steps "$ppl_steps" \
  --perplexity-start "$high_ppl" \
  --perplexity-end "$medium_ppl" \
  --perplexity-schedule-shape cosine \
  --lr-schedule-start "$lr_start_step" \
  --lr-schedule-steps "$lr_steps" \
  --lr-schedule-shape cosine \
  --min-lr-scale "${MIN_LR_SCALE:-0.1}" \
  --learning-rate "${LEARNING_RATE:-0.0008}" \
  --kl-strength "${KL_STRENGTH:-0.05}" \
  --kl-target "${KL_TARGET:-0.05}" \
  "$@"
