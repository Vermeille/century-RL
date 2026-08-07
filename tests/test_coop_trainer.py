from pathlib import Path

import pytest
import torch

from boardrl.checkpoints import Checkpoints
import trainers.coop as coop
from trainers.coop import (
    apply_optimizer_hyperparameters,
    build_parser,
    resolve_schedule_steps,
    run,
)


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


def test_coop_can_decay_lr_more_slowly_than_exploration():
    args = build_parser().parse_args(
        [
            "--steps",
            "1400",
            "--schedule-steps",
            "350",
            "--lr-schedule-steps",
            "1400",
        ]
    )

    assert resolve_schedule_steps(args) == (1400, 350)


def test_coop_lr_schedule_defaults_to_exploration_schedule():
    args = build_parser().parse_args(
        ["--steps", "1400", "--schedule-steps", "350"]
    )

    assert resolve_schedule_steps(args) == (350, 350)


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


def test_coop_cli_enables_trackio():
    args = build_parser().parse_args(["--trackio"])

    assert args.trackio


def test_coop_trackio_run_name_includes_nucleus_threshold(monkeypatch):
    calls = []
    monkeypatch.setattr(
        coop,
        "make_trackio",
        lambda **kwargs: calls.append(kwargs) or None,
    )
    monkeypatch.setattr(coop, "_run", lambda args, trackio_sink: None)

    run(
        build_parser().parse_args(
            [
                "--trackio",
                "--tag",
                "late-default",
                "--trackio-url",
                "https://trackio.example",
            ]
        )
    )

    assert calls[0]["name"] == "late-default"
    assert calls[0]["server_url"] == "https://trackio.example"


def test_coop_cli_can_anneal_entropy_strength_to_zero():
    args = build_parser().parse_args(["--entropy-baseline-ratio", "0"])

    assert args.entropy_baseline_ratio == 0.0


def test_coop_cli_selects_reverse_kl_exploration_regularizer():
    args = build_parser().parse_args(
        ["--exploration-regularizer", "reverse-kl"]
    )

    assert args.exploration_regularizer == "reverse-kl"


def test_coop_cli_configures_selective_action_support():
    args = build_parser().parse_args(
        ["--support-floor-mass", "0.01", "--support-strength", "0.001"]
    )

    assert args.support_floor_mass == 0.01
    assert args.support_strength == 0.001


def test_coop_resume_and_initialize_are_mutually_exclusive():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--resume", "old.pth", "--initialize-from", "best.pth"])


def test_cli_adamw_hyperparameters_override_restored_optimizer_state():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    saved = torch.optim.AdamW(
        [parameter],
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-5,
        weight_decay=0.2,
    )
    optimizer = torch.optim.AdamW([parameter])
    optimizer.load_state_dict(saved.state_dict())
    args = build_parser().parse_args(
        [
            "--learning-rate",
            "0.0007",
            "--adam-beta1",
            "0.4",
            "--adam-beta2",
            "0.98",
            "--adam-eps",
            "1e-7",
            "--weight-decay",
            "0.03",
        ]
    )

    apply_optimizer_hyperparameters(optimizer, args)

    group = optimizer.param_groups[0]
    assert group["lr"] == 0.0007
    assert group["betas"] == (0.4, 0.98)
    assert group["eps"] == 1e-7
    assert group["weight_decay"] == 0.03


def test_coop_zero_step_smoke(tmp_path: Path) -> None:
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
