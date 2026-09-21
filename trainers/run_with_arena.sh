#!/usr/bin/env bash
set -uo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
environment_file="$script_directory/../.env"
if [[ -f $environment_file ]]; then
  set -a
  source "$environment_file"
  set +a
fi

if (($# < 2)); then
  echo "usage: $0 CHECKPOINT_DIR TRAIN_COMMAND..." >&2
  exit 2
fi

checkpoint_dir=$1
shift

# Bash marks asynchronous commands as ignoring terminal stop/interrupt
# signals.  Reset those dispositions in a tiny child wrapper before exec so
# Python/bash trainers and the arena can actually receive the forwarded signal.
# Each child also gets its own process group.  That keeps a terminal-generated
# signal from reaching a child once from the terminal and a second time from
# this relay.
start_child() {
  trap - INT TERM HUP TSTP
  exec setsid -- "$@"
}

start_child "$@" &
trainer_pid=$!

arena_command=(
  python trainers/arena.py follow
  --checkpoint-dir "$checkpoint_dir"
  --training-pid "$trainer_pid"
)
if [[ -n ${ARENA_POOL_SIZE:-} ]]; then
  arena_command+=(--pool-size "$ARENA_POOL_SIZE")
fi
if [[ -n ${ARENA_GAMES_PER_BATCH:-} ]]; then
  arena_command+=(--games-per-batch "$ARENA_GAMES_PER_BATCH")
fi
if [[ -n ${ARENA_CALIBRATION_GAMES:-} ]]; then
  arena_command+=(--calibration-games "$ARENA_CALIBRATION_GAMES")
fi
if [[ -n ${ARENA_PLOT_EVERY_BATCHES:-} ]]; then
  arena_command+=(--plot-every-batches "$ARENA_PLOT_EVERY_BATCHES")
fi

start_child "${arena_command[@]}" &
arena_pid=$!

forward_signal() {
  local signal_name=$1
  # Signal the complete child process groups so subprocesses spawned by a
  # trainer or arena are suspended/stopped with their respective parent.
  kill -s "$signal_name" -- "-$trainer_pid" 2>/dev/null ||
    kill -s "$signal_name" "$trainer_pid" 2>/dev/null || true
  kill -s "$signal_name" -- "-$arena_pid" 2>/dev/null ||
    kill -s "$signal_name" "$arena_pid" 2>/dev/null || true
}

# The supervisor is itself the foreground job.  SIGINT/SIGTERM/HUP must be
# relayed before the shell continues waiting, otherwise an interrupted `wait`
# can make the supervisor exit while one child is still running.
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM
trap 'forward_signal HUP' HUP

# Bash normally stops on SIGTSTP before a shell trap can relay it.  Forward the
# stop first, then use SIGSTOP for the supervisor itself: SIGSTOP cannot be
# swallowed or deferred by a shell waiting for a child.  When the foreground
# job is continued, all three processes resume together.
forward_stop() {
  forward_signal TSTP
  kill -s STOP "$$"
}
trap forward_stop TSTP

wait_for_process() {
  local pid=$1
  local status
  while true; do
    wait "$pid"
    status=$?
    if ! kill -0 "$pid" 2>/dev/null; then
      return "$status"
    fi
    # A trapped terminal signal interrupts wait(2), but the child may still be
    # alive.  Keep reaping it after the signal has been forwarded.
  done
}

wait_for_process "$trainer_pid"
trainer_status=$?

# The arena monitors the trainer PID. Once this wait reaps the trainer, the
# arena initiates its own final capture, placement drain, and publication.
wait_for_process "$arena_pid"
arena_status=$?

if ((trainer_status != 0)); then
  exit "$trainer_status"
fi
exit "$arena_status"
