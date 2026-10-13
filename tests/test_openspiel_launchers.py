import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from trainers.arena import load_trainer_module


@pytest.fixture
def launch(tmp_path):
    repository = Path(__file__).parents[1]
    for script in ("train_leduc.sh", "train_openspiel.sh"):
        shutil.copy2(repository / script, tmp_path / script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        + """
import json
import os
import subprocess
import sys

args = sys.argv[1:]
assert args.pop(0) == "run"
if args[:2] == ["--extra", "trackio"]:
    args = args[2:]

def record(event):
    with open(os.environ["LAUNCH_LOG"], "a") as stream:
        stream.write(json.dumps({"event": event, "args": args}) + "\\n")

if args[:2] == ["python", "trainers/arena.py"]:
    record("checkpoint-dir")
    print(os.environ["CHECKPOINT_DIRECTORY"])
elif args[0] == "trainers/run_with_arena.sh":
    record("arena-start")
    status = subprocess.call(args[2:])
    record("arena-finished")
    sys.exit(status or int(os.environ.get("ARENA_STATUS", "0")))
elif args[:2] == ["python", "evaluate_nashconv.py"]:
    record("exploitability")
    sys.exit(int(os.environ.get("EXACT_STATUS", "0")))
else:
    assert args[0] == "python" and args[1].startswith("trainers/adversarial-")
    record("training")
    sys.exit(int(os.environ.get("TRAINING_STATUS", "0")))
"""
    )
    uv.chmod(0o755)

    def run(script="train_leduc.sh", arguments=(), **environment):
        log = tmp_path / "launch.jsonl"
        env = (
            os.environ
            | {
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "LAUNCH_LOG": str(log),
                "CHECKPOINT_DIRECTORY": str(tmp_path / "checkpoints with spaces"),
                "TRAINER": "adversarial-advshape",
                "ARENA": "1",
                "FINAL_EXPLOITABILITY": "1",
                "TRAINING_STATUS": "0",
                "ARENA_STATUS": "0",
                "EXACT_STATUS": "0",
            }
            | environment
        )
        result = subprocess.run(
            ["bash", str(tmp_path / script), *arguments],
            cwd=repository,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        events = [json.loads(line) for line in log.read_text().splitlines()]
        return result, events

    return run


@pytest.mark.parametrize(
    "script,game", [("train_leduc.sh", "leduc"), ("train_openspiel.sh", "phantom_ttt")]
)
@pytest.mark.parametrize(
    "trainer", ["adversarial-advshape", "adversarial-ppo", "adversarial-mmd"]
)
@pytest.mark.parametrize("arena", ["0", "1"])
def test_launchers_rate_during_training_then_evaluate_exactly(
    launch, script, game, trainer, arena
):
    result, events = launch(
        script, ["--tag", "tag with spaces"], TRAINER=trainer, ARENA=arena
    )
    assert result.returncode == 0, result.stderr
    expected = ["checkpoint-dir", "training", "exploitability"]
    if arena == "1":
        expected = [
            "checkpoint-dir",
            "arena-start",
            "training",
            "arena-finished",
            "exploitability",
        ]
    assert [event["event"] for event in events] == expected
    training = next(event["args"] for event in events if event["event"] == "training")
    assert training[:2] == ["python", f"trainers/{trainer}.py"]
    args = load_trainer_module(trainer).build_parser().parse_args(training[2:])
    assert args.game == game
    assert args.tag == "tag with spaces"
    assert args.sampled_evaluation
    assert events[-1]["args"][:3] == [
        "python",
        "evaluate_nashconv.py",
        "--training-run",
    ]
    if arena == "1":
        assert events[1]["args"][1] == events[-1]["args"][3]


def test_launcher_preserves_game_and_evaluation_overrides(launch):
    result, events = launch(
        "train_openspiel.sh", ["--game", "abrupt_dark_hex", "--no-sampled-evaluation"]
    )
    assert result.returncode == 0, result.stderr
    training = next(event["args"] for event in events if event["event"] == "training")
    args = load_trainer_module().build_parser().parse_args(training[2:])
    assert args.game == "abrupt_dark_hex"
    assert not args.sampled_evaluation


@pytest.mark.parametrize(
    "status",
    [
        {"TRAINING_STATUS": "12"},
        {"ARENA_STATUS": "7"},
        {"TRAINING_STATUS": "130", "ARENA": "0"},
    ],
)
def test_failed_or_interrupted_training_does_not_start_exploitability(launch, status):
    result, events = launch(**status)
    assert result.returncode == int(
        status.get("TRAINING_STATUS", status.get("ARENA_STATUS"))
    )
    assert "exploitability" not in [event["event"] for event in events]


def test_final_exploitability_can_be_disabled_independently(launch):
    result, events = launch(FINAL_EXPLOITABILITY="0")
    assert result.returncode == 0
    assert [event["event"] for event in events] == [
        "checkpoint-dir",
        "arena-start",
        "training",
        "arena-finished",
    ]


def test_final_exploitability_failure_is_reported(launch):
    result, events = launch(EXACT_STATUS="9")
    assert result.returncode == 9
    assert events[-1]["event"] == "exploitability"
