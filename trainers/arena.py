#!/usr/bin/env python3
"""Run the asynchronous CPU rating arena for adversarial-advshape."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path

from boardrl.arena import RatingArena, TrackioTelemetry, read_run_arguments


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def load_trainer_module():
    path = Path(__file__).with_name("adversarial-advshape.py")
    repository = str(path.parents[1])
    if repository not in sys.path:
        sys.path.insert(0, repository)
    spec = importlib.util.spec_from_file_location("trainers.adversarial_advshape", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def effective_checkpoint_directory(arguments: list[str]) -> Path:
    trainer = load_trainer_module()
    args = trainer.build_parser().parse_args(arguments)
    return trainer.checkpoint_directory(args)


def wait_for_run_arguments(
    checkpoint_directory: Path,
    *,
    started_at: float,
    training_pid: int,
) -> dict[str, object]:
    run_path = checkpoint_directory / "run.txt"
    while True:
        try:
            if run_path.stat().st_mtime >= started_at:
                return read_run_arguments(run_path)
        except (FileNotFoundError, OSError, ValueError):
            pass
        try:
            Path(f"/proc/{training_pid}").stat()
        except FileNotFoundError as exc:
            raise RuntimeError("training exited before writing run.txt") from exc
        time.sleep(0.2)


def process_started_at(training_pid: int) -> float:
    try:
        return Path(f"/proc/{training_pid}").stat().st_mtime
    except FileNotFoundError:
        return time.time()


def resume_trackio(arguments: dict[str, object]):
    if not arguments.get("trackio"):
        return None
    try:
        import trackio
    except ImportError as exc:
        raise RuntimeError(
            "the arena requires the trackio extra; run `uv sync --extra trackio`"
        ) from exc

    kwargs = {
        "project": str(arguments["game"]),
        "name": str(
            arguments.get("trackio_run_name") or arguments["tag"]
        ),
        "config": arguments,
        "resume": "must",
        "embed": False,
        "auto_log_gpu": False,
        "auto_log_cpu": False,
    }
    server_url = (
        arguments.get("trackio_url")
        or os.environ.get("TRACKIO_SERVER_URL")
        or os.environ.get("TRACKIO_URL")
    )
    if server_url:
        kwargs["server_url"] = str(server_url)
    return trackio.init(**kwargs)


def wait_for_trackio_run(
    arguments: dict[str, object],
    training_pid: int,
    *,
    poll_seconds: float = 0.2,
):
    """Attach after the trainer's newly-created run becomes visible.

    ``run.txt`` is intentionally written early in trainer startup, while a
    remote Trackio server may commit the corresponding run a little later.
    ``resume='must'`` raises during that visibility window, so only that
    specific error is retried.  Other Trackio/configuration failures remain
    actionable instead of being hidden by the startup poll.
    """

    if not arguments.get("trackio"):
        return None
    delay = max(0.05, poll_seconds)
    while True:
        try:
            return resume_trackio(arguments)
        except ValueError as exc:
            message = str(exc)
            if "does not exist in project" not in message:
                raise
            try:
                Path(f"/proc/{training_pid}").stat()
            except FileNotFoundError:
                # The trainer has already ended.  Arena scoring can continue
                # locally, but there is no point waiting forever for a run
                # that may never have been created.
                return None
            time.sleep(delay)


def local_run_id(arguments: dict[str, object], started_at: float) -> str:
    identity = (
        f"{arguments.get('game')}|{arguments.get('tag')}|"
        f"{arguments.get('seed')}|{started_at:.6f}"
    )
    return "local-" + hashlib.sha256(identity.encode()).hexdigest()[:16]


def stop_when_training_exits(arena: RatingArena, training_pid: int):
    def monitor():
        process = Path(f"/proc/{training_pid}")
        while not arena.stop_requested and process.exists():
            time.sleep(0.2)
        arena.request_stop()

    threading.Thread(target=monitor, name="arena-training-monitor", daemon=True).start()


def defer_stop_until_training_exits(arena: RatingArena, training_pid: int, *_):
    if not Path(f"/proc/{training_pid}").exists():
        arena.request_stop()


def follow(args) -> int:
    started_at = process_started_at(args.training_pid)
    arguments = wait_for_run_arguments(
        args.checkpoint_dir,
        started_at=started_at,
        training_pid=args.training_pid,
    )
    run = wait_for_trackio_run(
        arguments,
        args.training_pid,
        poll_seconds=args.poll_seconds,
    )
    run_id = run.id if run is not None else local_run_id(arguments, started_at)
    telemetry = None
    if run is not None:
        import trackio

        telemetry = TrackioTelemetry(
            run,
            html_factory=trackio.Html,
            alert_levels=trackio.AlertLevel,
        )
    arena = None
    try:
        arena = RatingArena(
            args.checkpoint_dir,
            arguments,
            run_id=run_id,
            telemetry=telemetry,
            started_at=started_at,
            pool_size=args.pool_size,
            games_per_batch=args.games_per_batch,
            calibration_games=args.calibration_games,
            plot_every_batches=args.plot_every_batches,
            poll_seconds=args.poll_seconds,
        )
        handler = partial(defer_stop_until_training_exits, arena, args.training_pid)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGHUP, handler)
        stop_when_training_exits(arena, args.training_pid)
        arena.run()
    finally:
        if arena is not None:
            arena.close()
        if run is not None:
            run.finish()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    directory = commands.add_parser(
        "checkpoint-dir",
        help="print the trainer's effective checkpoint directory",
    )
    directory.add_argument("trainer_arguments", nargs=argparse.REMAINDER)

    watcher = commands.add_parser("follow", help="watch and rate checkpoints")
    watcher.add_argument("--checkpoint-dir", type=Path, required=True)
    watcher.add_argument("--training-pid", type=int, required=True)
    watcher.add_argument("--pool-size", type=int, default=32)
    watcher.add_argument("--games-per-batch", type=int, default=32)
    watcher.add_argument("--calibration-games", type=int, default=64)
    watcher.add_argument("--plot-every-batches", type=positive_int, default=10)
    watcher.add_argument("--poll-seconds", type=float, default=1.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "checkpoint-dir":
        arguments = args.trainer_arguments
        if arguments[:1] == ["--"]:
            arguments = arguments[1:]
        print(effective_checkpoint_directory(arguments))
        return 0
    return follow(args)


if __name__ == "__main__":
    raise SystemExit(main())
