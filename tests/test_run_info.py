import argparse
import subprocess
from pathlib import Path

from boardrl.run import RunInfo


class TextRecorder:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def text(self, name: str, value: str) -> None:
        self.values[name] = value


def test_run_info_publishes_and_saves_the_same_text(tmp_path: Path) -> None:
    executable = tmp_path / "trainer.py"
    executable.write_text("print('training')\n")
    info = RunInfo.capture(
        argparse.Namespace(game="thegame", checkpoint_root=tmp_path), executable
    )
    sink = TextRecorder()

    info.publish(sink)
    path = info.save(tmp_path / "checkpoints")

    assert sink.values["run"] == path.read_text()
    assert '"game": "thegame"' in path.read_text()
    assert str(executable.resolve()) in path.read_text()
    assert "(not a git checkout)" in path.read_text()


def test_run_info_captures_commit_status_and_dirty_diff(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "BoardRL tests"],
        cwd=tmp_path,
        check=True,
    )
    executable = tmp_path / "trainer.py"
    executable.write_text("print('before')\n")
    subprocess.run(["git", "add", "trainer.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    executable.write_text("print('after')\n")
    info = RunInfo.capture(argparse.Namespace(game="thegame"), executable)

    assert info.git_commit == commit
    assert " M trainer.py" in info.git_status
    assert "-print('before')" in info.git_diff
    assert "+print('after')" in info.git_diff
    assert commit in info.text
