import argparse
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
    assert "print('training')" in path.read_text()


def test_run_info_captures_additional_source_files(tmp_path: Path) -> None:
    executable = tmp_path / "trainer.py"
    dependency = tmp_path / "loss.py"
    executable.write_text("train()\n")
    dependency.write_text("class NovelLoss: pass\n")

    info = RunInfo.capture(
        argparse.Namespace(game="thegame"),
        executable,
        additional_sources=[dependency],
    )

    assert f"Source: {dependency.resolve()}" in info.text
    assert "class NovelLoss: pass" in info.text
