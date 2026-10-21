import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def launch(tmp_path):
    """Exercise real shell recipes without starting trainers or an arena."""
    repository = Path(__file__).parents[1]
    workspace = tmp_path / "launch workspace"
    workspace.mkdir()
    for script in repository.glob("train_*.sh"):
        shutil.copy2(script, workspace / script.name)
    (workspace / "trainers").mkdir()
    shutil.copy2(repository / "trainers/launch.sh", workspace / "trainers/launch.sh")
    bin_dir = workspace / "bin"
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
extras = []
if args[:2] == ["--extra", "trackio"]:
    extras = args[:2]
    args = args[2:]

def record(event):
    with open(os.environ["LAUNCH_LOG"], "a") as stream:
        stream.write(json.dumps({
            "event": event, "args": args, "extras": extras,
            "cwd": os.getcwd(), "marker": os.environ.get("LAUNCH_TEST_MARKER"),
        }) + "\\n")

if args[:2] == ["python", "trainers/arena.py"]:
    record("checkpoint-dir")
    print(os.environ["CHECKPOINT_DIRECTORY"])
    sys.exit(int(os.environ["CHECKPOINT_STATUS"]))
elif args[0] == "trainers/run_with_arena.sh":
    record("arena-start")
    status = subprocess.call(args[2:])
    record("arena-finished")
    sys.exit(status or int(os.environ["ARENA_STATUS"]))
elif args[:2] == ["python", "evaluate_nashconv.py"]:
    record("exploitability")
    sys.exit(int(os.environ["EXACT_STATUS"]))
else:
    assert args[0] == "python" and args[1].startswith("trainers/")
    record("help" if "--help" in args or "-h" in args else "training")
    sys.exit(int(os.environ["TRAINING_STATUS"]))
"""
    )
    uv.chmod(0o755)

    def run(
        script="train_openspiel.sh",
        arguments=("--game", "leduc"),
        dotenv=None,
        **environment,
    ):
        log = workspace / "launch.jsonl"
        log.unlink(missing_ok=True)
        environment_file = workspace / ".env"
        environment_file.unlink(missing_ok=True)
        if dotenv is not None:
            environment_file.write_text(dotenv)
        launcher_variables = {
            "TRAINER", "TAG", "SEED", "ARENA", "TRACKIO", "FINAL_EXPLOITABILITY",
            "AGENT_PPL_START", "AGENT_PPL_END", "ENVIRONMENT_PPL_START",
            "ENVIRONMENT_PPL_END", "AGENT_THRESHOLD", "ENVIRONMENT_THRESHOLD",
            "MMD_COEFFICIENT", "MMD_REFERENCE_TIMESTEPS", "LAUNCH_TEST_MARKER",
        }
        env = {
            key: value for key, value in os.environ.items()
            if key not in launcher_variables
        }
        env.update({
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "LAUNCH_LOG": str(log),
            "CHECKPOINT_DIRECTORY": str(workspace / "checkpoints with spaces"),
            "TRAINER": "adversarial-advshape",
            "ARENA": "1",
            "TRACKIO": "1",
            "FINAL_EXPLOITABILITY": "1",
            "TRAINING_STATUS": "0",
            "ARENA_STATUS": "0",
            "EXACT_STATUS": "0",
            "CHECKPOINT_STATUS": "0",
        })
        env.update(environment)
        result = subprocess.run(
            ["bash", str(workspace / script), *arguments],
            cwd=tmp_path,
            check=False,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        events = []
        if log.exists():
            events = [json.loads(line) for line in log.read_text().splitlines()]
        return result, events

    return run
