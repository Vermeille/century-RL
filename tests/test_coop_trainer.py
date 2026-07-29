from pathlib import Path

import pytest

from boardrl.checkpoints import Checkpoints
from trainers.coop import build_parser, run


class OfflineVisualizer:
    def __init__(self, *args) -> None:
        self.text: dict[str, str] = {}

    def html(self, name: str, value: str) -> None:
        self.text[name] = value


def test_coop_cli_selects_game_and_architecture():
    args = build_parser().parse_args(["--game", "tictactoe", "--architecture", "toy"])

    assert args.game == "tictactoe"
    assert args.architecture == "toy"


def test_coop_cli_selects_model_scale_and_patch_size_together():
    args = build_parser().parse_args(["--architecture", "patchformer-tiny-p8"])

    assert args.architecture == "patchformer-tiny-p8"


def test_coop_cli_accepts_separate_schedule_horizon():
    args = build_parser().parse_args(["--steps", "20", "--schedule-steps", "10"])

    assert args.steps == 20
    assert args.schedule_steps == 10


def test_coop_cli_separates_inference_and_learner_batch_sizes():
    args = build_parser().parse_args(
        ["--inference-batch-size", "256", "--learner-batch-size", "1024"]
    )

    assert args.inference_batch_size == 256
    assert args.learner_batch_size == 1024


def test_coop_cli_configures_bounded_best_checkpoints():
    args = build_parser().parse_args(
        ["--keep-checkpoints", "1", "--save-best"]
    )

    assert args.keep_checkpoints == 1
    assert args.save_best


def test_coop_cli_enables_wandb():
    args = build_parser().parse_args(["--wandb"])

    assert args.wandb


def test_coop_resume_and_initialize_are_mutually_exclusive():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--resume", "old.pth", "--initialize-from", "best.pth"])


def test_coop_zero_step_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    visualizer = OfflineVisualizer()
    monkeypatch.setattr("trainers.coop.OfflineVisualizer", lambda: visualizer)
    args = build_parser().parse_args(
        [
            "--game",
            "tictactoe",
            "--architecture",
            "toy",
            "--device",
            "cpu",
            "--steps",
            "0",
            "--checkpoint-root",
            str(tmp_path),
            "--no-progress",
        ]
    )

    path = run(args)
    state = Checkpoints(path.parent, prefix="step").load(path)

    assert path.parent == tmp_path / "coop" / "tictactoe" / "toy" / "coop"
    assert state["step"] == 0
    assert state["metadata"] == {
        "trainer": "coop",
        "game": "tictactoe",
        "architecture": "toy",
    }
    run_info = path.parent / "run.txt"
    assert run_info.exists()
    assert '"game": "tictactoe"' in run_info.read_text()
    assert "Cooperative PPO training from scratch" in run_info.read_text()
    assert "run" in visualizer.text
