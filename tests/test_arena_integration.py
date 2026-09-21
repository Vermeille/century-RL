from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from boardrl.arena import RANDOM_ID, RatingArena, RatingFit, TrackioTelemetry
from boardrl.checkpoints import Checkpoints
from boardrl.models import toy
from trainers.arena import resume_trackio, wait_for_trackio_run


class RecordingRun:
    def __init__(self):
        self.logs = []
        self.alerts = []

    def log(self, metrics, step=None):
        self.logs.append((metrics, step))

    def alert(self, **alert):
        self.alerts.append(alert)


def arena_arguments(**overrides):
    return {
        "game": "tictactoe",
        "opponent_eval_strategy": "random",
        "inference_batch_size": 8,
        "eval_temperature": 1.0,
        "rollout_max_steps": 20,
        "tag": "test",
        "seed": 1,
        "trackio": False,
        **overrides,
    }


def test_watcher_ignores_stale_best_partial_and_wrong_trainer_files(tmp_path):
    stale = Checkpoints(tmp_path).save(
        1,
        {"agent": toy()},
        metadata={"trainer": "adversarial-advshape", "game": "tictactoe"},
    )
    os.utime(stale, (1, 1))
    Checkpoints(tmp_path, prefix="best").save(
        2,
        {"agent": toy()},
        metadata={"trainer": "adversarial-advshape", "game": "tictactoe"},
    )
    torch.save(
        {
            "step": 3,
            "models": {"agent": {}},
            "model_specs": {"agent": {}},
            "metadata": {"trainer": "other", "game": "tictactoe"},
        },
        tmp_path / "step-3.pth",
    )
    (tmp_path / ".step-4.pth.partial").write_bytes(b"partial")
    current = Checkpoints(tmp_path).save(
        5,
        {"agent": toy(), "environment": toy()},
        metadata={"trainer": "adversarial-advshape", "game": "tictactoe"},
    )
    arena = RatingArena(
        tmp_path,
        arena_arguments(),
        run_id="watcher",
        started_at=3,
    )

    assert arena.capture_checkpoints() == 1
    compact = arena.store.policy("checkpoint-5").path
    current.unlink()
    assert compact is not None and compact.exists()
    assert arena.capture_checkpoints() == 0
    assert [policy.id for policy in arena.store.policies(checkpoints_only=True)] == [
        "checkpoint-5"
    ]
    arena.close()


def test_cpu_smoke_rates_checkpoint_and_publishes_full_figures(tmp_path):
    Checkpoints(tmp_path).save(
        7,
        {"agent": toy(), "environment": toy()},
        metadata={"trainer": "adversarial-advshape", "game": "tictactoe"},
    )
    run = RecordingRun()
    telemetry = TrackioTelemetry(
        run,
        html_factory=lambda content: {"html": content},
        alert_levels=SimpleNamespace(INFO="info", WARN="warn", ERROR="error"),
    )
    arena = RatingArena(
        tmp_path,
        arena_arguments(),
        run_id="cpu-smoke",
        telemetry=telemetry,
        games_per_batch=2,
        calibration_games=0,
        placement_batches=3,
    )

    arena.request_stop()
    arena.run()

    assert arena.store.games_between("checkpoint-7", "random") == 6
    assert len(run.logs) == 1
    for metrics, step in run.logs:
        assert step == 7
        assert set(metrics) == {
            "arena/rating_curve",
            "arena/nontransitivity",
            "arena/dominance_graph",
            "arena/event_log",
        }
    rating_axis = run.logs[-1][0]["arena/rating_curve"].axes[0]
    assert any(
        collection.get_label() == "95% uncertainty"
        for collection in rating_axis.collections
    )
    opponent_lines = [
        line
        for line in rating_axis.lines
        if line.get_label().startswith("evaluation opponent")
    ]
    assert len(opponent_lines) == 1
    assert opponent_lines[0].get_linestyle() == ":"
    assert len(run.alerts) == 1
    assert run.alerts[0]["level"] == "info"
    assert "New arena best" in run.alerts[0]["title"]
    assert "New arena best" in run.logs[-1][0]["arena/event_log"]["html"]
    arena.close()


def test_nonrandom_opponent_is_calibrated_before_checkpoint_work(tmp_path):
    arena = RatingArena(
        tmp_path,
        arena_arguments(opponent_eval_strategy="longest_move"),
        run_id="calibration",
        games_per_batch=32,
        calibration_games=64,
    )
    calls = []

    def record_batch(first, second, purpose):
        calls.append((first, second, purpose))
        arena.store.record_match(
            batch_id=f"calibration-{len(calls)}",
            first=first,
            second=second,
            games=32,
            score=16.0,
            seed=len(calls),
            purpose=purpose,
        )

    arena.play_batch = record_batch
    arena.request_stop()
    arena.run()

    assert calls == [
        ("evaluation-opponent", "random", "calibration"),
        ("evaluation-opponent", "random", "calibration"),
    ]
    arena.close()


def test_calibration_publishes_reference_only_curve(tmp_path):
    run = RecordingRun()
    telemetry = TrackioTelemetry(
        run,
        html_factory=lambda content: {"html": content},
        alert_levels=SimpleNamespace(INFO="info", WARN="warn", ERROR="error"),
    )
    arena = RatingArena(
        tmp_path,
        arena_arguments(opponent_eval_strategy="longest_move"),
        run_id="calibration-plot",
        telemetry=telemetry,
        games_per_batch=32,
        calibration_games=32,
    )

    def record_batch(first, second, purpose):
        arena.store.record_match(
            batch_id="calibration-plot-batch",
            first=first,
            second=second,
            games=32,
            score=16.0,
            seed=1,
            purpose=purpose,
        )

    arena.play_batch = record_batch
    arena.request_stop()
    arena.run()

    assert len(run.logs) == 1
    metrics, step = run.logs[0]
    assert step == 0
    axis = metrics["arena/rating_curve"].axes[0]
    assert [line.get_label() for line in axis.lines] == [
        "random = 0",
        "evaluation opponent (longest_move) = 0.0",
    ]
    assert set(metrics) == {"arena/rating_curve"}
    arena.close()


def test_event_log_and_alerts_cover_bests_regressions_and_cycles(tmp_path):
    run = RecordingRun()
    telemetry = TrackioTelemetry(
        run,
        html_factory=lambda content: {"html": content},
        alert_levels=SimpleNamespace(INFO="info", WARN="warn", ERROR="error"),
    )
    arena = RatingArena(
        tmp_path,
        arena_arguments(),
        run_id="events",
        telemetry=telemetry,
        placement_batches=1,
    )

    def add_checkpoint(step):
        policy_id = f"checkpoint-{step}"
        path = tmp_path / f"agent-{step}.pth"
        path.touch()
        arena.store.add_policy(
            policy_id, kind="checkpoint", step=step, path=path
        )
        arena.store.record_match(
            batch_id=f"placement-{step}",
            first=policy_id,
            second=RANDOM_ID,
            games=32,
            score=16.0,
            seed=step,
            purpose=f"placement:{policy_id}:0",
        )
        return policy_id

    first = add_checkpoint(10)
    arena.publish(
        RatingFit(
            {RANDOM_ID: 0.0, first: 200.0},
            {RANDOM_ID: 0.0, first: 10.0},
            {(RANDOM_ID, RANDOM_ID): 0.0, (first, first): 100.0},
        )
    )
    second = add_checkpoint(20)
    arena.publish(
        RatingFit(
            {RANDOM_ID: 0.0, first: 200.0, second: 0.0},
            {RANDOM_ID: 0.0, first: 10.0, second: 10.0},
            {
                (RANDOM_ID, RANDOM_ID): 0.0,
                (first, first): 100.0,
                (second, second): 100.0,
            },
        )
    )
    third = add_checkpoint(30)
    for index, (winner, loser) in enumerate(
        ((first, second), (second, third), (third, first))
    ):
        arena.store.record_match(
            batch_id=f"cycle-{index}",
            first=winner,
            second=loser,
            games=64,
            score=48.0,
            seed=index,
            purpose="audit",
        )
    final_fit = RatingFit(
        {RANDOM_ID: 0.0, first: 200.0, second: 0.0, third: 50.0},
        {RANDOM_ID: 0.0, first: 10.0, second: 10.0, third: 10.0},
        {
            (RANDOM_ID, RANDOM_ID): 0.0,
            (first, first): 100.0,
            (second, second): 100.0,
            (third, third): 100.0,
        },
    )
    arena.publish(final_fit)
    alerts_after_events = len(run.alerts)
    arena.publish(final_fit)

    kinds = [event.kind for event in arena.store.events()]
    assert kinds.count("new_best") == 1
    assert kinds.count("regression") == 2
    assert kinds.count("cycle") == 1
    assert [alert["level"] for alert in run.alerts] == [
        "info",
        "warn",
        "warn",
        "warn",
    ]
    assert len(run.alerts) == alerts_after_events
    event_html = run.logs[-1][0]["arena/event_log"]["html"]
    assert "Confirmed policy cycle" in event_html
    assert "Strong regression" in event_html
    assert "over 64 games" in event_html
    assert len(run.logs[-1][0]["arena/dominance_graph"].axes[0].patches) >= 3
    arena.close()


def test_trackio_failure_does_not_stop_rating_and_is_retried(tmp_path):
    class FlakyTelemetry:
        def __init__(self):
            self.fail = True
            self.alerts = []
            self.logs = []

        def html(self, content):
            return content

        def alert(self, event):
            if self.fail:
                raise RuntimeError("temporary alert outage")
            self.alerts.append(event)

        def log(self, metrics, *, step):
            if self.fail:
                raise RuntimeError("temporary metric outage")
            self.logs.append((metrics, step))

    telemetry = FlakyTelemetry()
    arena = RatingArena(
        tmp_path,
        arena_arguments(),
        run_id="flaky-trackio",
        telemetry=telemetry,
        placement_batches=1,
    )
    path = tmp_path / "agent-10.pth"
    path.touch()
    arena.store.add_policy(
        "checkpoint-10", kind="checkpoint", step=10, path=path
    )
    arena.store.record_match(
        batch_id="placement-10",
        first="checkpoint-10",
        second=RANDOM_ID,
        games=32,
        score=24.0,
        seed=10,
        purpose="placement:checkpoint-10:0",
    )
    fit = RatingFit(
        {RANDOM_ID: 0.0, "checkpoint-10": 100.0},
        {RANDOM_ID: 0.0, "checkpoint-10": 10.0},
        {
            (RANDOM_ID, RANDOM_ID): 0.0,
            ("checkpoint-10", "checkpoint-10"): 100.0,
        },
    )

    arena.publish(fit)
    assert len(arena.store.pending_alerts()) == 1
    assert arena.store.get_int("last_published_step", -1) == -1

    telemetry.fail = False
    arena.publish(fit)

    assert len(arena.store.pending_alerts()) == 0
    assert arena.store.get_int("last_published_step", -1) == 10
    assert len(telemetry.alerts) == 1
    assert len(telemetry.logs) == 1
    arena.close()


def test_checkpoint_directory_cli_honors_last_argument_occurrence(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "trainers/arena.py",
            "checkpoint-dir",
            "--",
            "--game",
            "century",
            "--game",
            "connectfour",
            "--architecture",
            "cnn",
            "--checkpoint-root",
            str(tmp_path),
            "--tag",
            "first",
            "--tag",
            "last",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert Path(result.stdout.strip()) == (
        tmp_path / "adversarial-advshape" / "connectfour" / "cnn" / "last"
    )


def test_trackio_is_resumed_without_arena_hardware_telemetry(monkeypatch):
    calls = []
    resumed = SimpleNamespace(id="run-id")
    monkeypatch.setitem(
        sys.modules,
        "trackio",
        SimpleNamespace(init=lambda **kwargs: calls.append(kwargs) or resumed),
    )

    result = resume_trackio(
        arena_arguments(trackio=True, trackio_url="http://trackio.test")
    )

    assert result is resumed
    assert calls == [
        {
            "project": "tictactoe",
            "name": "test",
            "config": arena_arguments(
                trackio=True, trackio_url="http://trackio.test"
            ),
            "resume": "must",
            "embed": False,
            "auto_log_gpu": False,
            "auto_log_cpu": False,
            "server_url": "http://trackio.test",
        }
    ]


def test_trackio_resume_reads_server_url_from_environment(monkeypatch):
    calls = []
    resumed = SimpleNamespace(id="run-id")
    monkeypatch.setitem(
        sys.modules,
        "trackio",
        SimpleNamespace(init=lambda **kwargs: calls.append(kwargs) or resumed),
    )
    monkeypatch.setenv("TRACKIO_SERVER_URL", "https://env-trackio.example")
    monkeypatch.setenv("TRACKIO_WRITE_TOKEN", "test-token")

    result = resume_trackio(arena_arguments(trackio=True))

    assert result is resumed
    assert calls[0]["server_url"] == "https://env-trackio.example"
    assert "write_token" not in calls[0]


def test_trackio_resume_uses_current_unique_run_name(monkeypatch):
    calls = []
    resumed = SimpleNamespace(id="run-id")
    monkeypatch.setitem(
        sys.modules,
        "trackio",
        SimpleNamespace(init=lambda **kwargs: calls.append(kwargs) or resumed),
    )

    resume_trackio(
        arena_arguments(trackio=True, trackio_run_name="test-1234-abcd")
    )

    assert calls[0]["name"] == "test-1234-abcd"


def test_trackio_resume_polls_until_current_run_is_visible(monkeypatch):
    attempts = []
    resumed = SimpleNamespace(id="run-id")

    def delayed_resume(arguments):
        attempts.append(arguments)
        if len(attempts) < 3:
            raise ValueError("Run 'test' does not exist in project 'tictactoe'")
        return resumed

    monkeypatch.setattr("trainers.arena.resume_trackio", delayed_resume)

    result = wait_for_trackio_run(
        arena_arguments(trackio=True),
        os.getpid(),
        poll_seconds=0.001,
    )

    assert result is resumed
    assert len(attempts) == 3


def _fake_python(tmp_path: Path) -> Path:
    binary = tmp_path / "python"
    binary.write_text(
        """#!/usr/bin/env bash
if [[ ${FAKE_ARENA_EARLY:-0} == 1 ]]; then
  exit "${FAKE_ARENA_STATUS:-0}"
fi
training_pid=
while (( $# )); do
  if [[ $1 == --training-pid ]]; then
    training_pid=$2
    break
  fi
  shift
done
while kill -0 "$training_pid" 2>/dev/null; do sleep 0.05; done
exit "${FAKE_ARENA_STATUS:-0}"
"""
    )
    binary.chmod(0o755)
    return binary


def _run_supervisor(tmp_path: Path, trainer: str, **environment):
    _fake_python(tmp_path)
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}"} | environment
    return subprocess.run(
        ["trainers/run_with_arena.sh", str(tmp_path / "checkpoints"), "bash", "-c", trainer],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_supervisor_preserves_trainer_failure_over_arena_failure(tmp_path):
    result = _run_supervisor(
        tmp_path,
        "exit 3",
        FAKE_ARENA_STATUS="7",
    )

    assert result.returncode == 3


def test_supervisor_surfaces_arena_failure_after_successful_training(tmp_path):
    result = _run_supervisor(
        tmp_path,
        "exit 0",
        FAKE_ARENA_STATUS="7",
    )

    assert result.returncode == 7


def test_early_arena_failure_does_not_terminate_training(tmp_path):
    marker = tmp_path / "training-finished"
    result = _run_supervisor(
        tmp_path,
        f"sleep 0.1; touch {marker}",
        FAKE_ARENA_EARLY="1",
        FAKE_ARENA_STATUS="9",
    )

    assert marker.exists()
    assert result.returncode == 9


def test_root_launcher_can_disable_arena(tmp_path):
    invocation = tmp_path / "invocation"
    uv = tmp_path / "uv"
    uv.write_text(
        f"""#!/usr/bin/env bash
printf '%s\n' "$@" > {invocation}
"""
    )
    uv.chmod(0o755)
    env = os.environ | {
        "ARENA": "0",
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", "train_connectfour.sh", "--tag", "launcher-test"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    arguments = invocation.read_text().splitlines()
    assert arguments[:5] == [
        "run",
        "--extra",
        "trackio",
        "python",
        "trainers/adversarial-advshape.py",
    ]
    assert "trainers/arena.py" not in arguments
    assert arguments[-2:] == ["--tag", "launcher-test"]


def test_supervisor_forwards_terminal_signals_to_both_children(tmp_path):
    _fake_python(tmp_path)
    marker = tmp_path / "trainer-stopped"
    trainer = f"trap 'touch {marker}; exit 12' TERM; while true; do sleep 0.05; done"
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    process = subprocess.Popen(
        [
            "trainers/run_with_arena.sh",
            str(tmp_path / "checkpoints"),
            "bash",
            "-c",
            trainer,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.2)
    process.send_signal(signal.SIGTERM)

    process.communicate(timeout=10)

    assert marker.exists()
    assert process.returncode == 12


def test_supervisor_forwards_interrupt_and_preserves_trainer_status(tmp_path):
    _fake_python(tmp_path)
    marker = tmp_path / "trainer-interrupted"
    trainer = (
        f"trap 'touch {marker}; exit 12' INT; "
        "while true; do sleep 0.05; done"
    )
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    process = subprocess.Popen(
        [
            "trainers/run_with_arena.sh",
            str(tmp_path / "checkpoints"),
            "bash",
            "-c",
            trainer,
        ],
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.2)
    process.send_signal(signal.SIGINT)

    process.communicate(timeout=10)

    assert marker.exists()
    assert process.returncode == 12


def test_supervisor_forwards_stop_and_resumes_as_one_job(tmp_path):
    _fake_python(tmp_path)
    marker = tmp_path / "trainer-stopped"
    trainer = (
        f"trap 'touch {marker}' TSTP; "
        "while true; do sleep 0.05; done"
    )
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    process = subprocess.Popen(
        [
            "trainers/run_with_arena.sh",
            str(tmp_path / "checkpoints"),
            "bash",
            "-c",
            trainer,
        ],
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.2)
        process.send_signal(signal.SIGTSTP)
        deadline = time.monotonic() + 5
        state = ""
        while time.monotonic() < deadline:
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
            except FileNotFoundError:
                break
            state = next(
                line.split()[1]
                for line in status.splitlines()
                if line.startswith("State:")
            )
            if state.startswith("T"):
                break
            time.sleep(0.01)
        assert state.startswith("T")

        os.killpg(process.pid, signal.SIGCONT)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
    finally:
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
