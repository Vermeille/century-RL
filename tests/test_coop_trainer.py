from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from boardrl.checkpoints import Checkpoints
from boardrl.metrics import Range
import trainers.coop as coop
from trainers.coop import (
    apply_optimizer_hyperparameters,
    build_parser,
    resolve_schedule_steps,
    run,
)


class _FakeEvaluationRollouts:
    def my_points(self, player, by="strategy"):
        return [1.0]


class _FakeEvaluation:
    rollouts = _FakeEvaluationRollouts()

    def win_rate(self):
        return 1.0

    def avg_points(self):
        return 1.0


class _FakeEvaluator:
    calls = []

    def __init__(self, make_game, *, progress, coop=False):
        del make_game, progress, coop

    def compare(self, players, *, names, games, max_steps, rotate=True):
        del players, names, games, max_steps, rotate
        self.calls.append(0)
        return _FakeEvaluation()


class _FakeRolloutRunner:
    def __init__(self, make_game, *, progress, coop=False):
        del make_game, progress, coop

    def play(self, players, *, games, max_steps, rotate=True):
        del players, games, max_steps, rotate
        return []


class _RecordingMetricLogger:
    instances = []

    def __init__(self, *sinks):
        del sinks
        self.logs = []
        self.games = []
        self.instances.append(self)

    def log(self, step, **values):
        self.logs.append((step, values))

    def game(self, step, metrics, *, histories=False):
        del metrics, histories
        self.games.append(step)


class _CompletedLearner:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback

    def safe_point(self):
        pass

    def state_dict(self):
        return {}

    def train(self, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(metrics={})


class _StatefulLearner(_CompletedLearner):
    def __init__(self, marker):
        self.state = {"marker": marker}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state


class _CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def spec(self):
        return {"test_model": True}


def test_coop_cli_selects_game_and_architecture():
    args = build_parser().parse_args(["--game", "tictactoe", "--architecture", "toy"])

    assert args.game == "tictactoe"
    assert args.architecture == "toy"


def test_coop_cli_defaults_to_the_shared_omni_recipe():
    args = build_parser().parse_args([])

    assert args.game == "thegame,mode=omni"
    assert args.architecture == "patchformer-medium-p8"
    assert args.steps == 2_400
    assert args.inference_batch_size == 1_024
    assert args.learner_batch_size == 384
    assert args.rollout_games == 128
    assert args.evaluation_games == 512
    assert args.learning_rate == 8e-4
    assert args.adam_beta1 == 0.9
    assert args.adam_beta2 == 0.95
    assert args.adam_eps == 1e-5
    assert args.gradient_clip == 5.0
    assert args.gae_lambda == 0.1
    assert args.value_lambda == 0.9
    assert args.perplexity_start == 2.5
    assert args.perplexity_end == 1.5
    assert args.lr_schedule_shape == "cosine"
    assert args.save_best


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


def test_coop_default_schedule_matches_full_decay_recipe():
    args = build_parser().parse_args(["--steps", "1800"])

    assert resolve_schedule_steps(args) == (899, 1764)


def test_coop_schedule_percentages_are_cli_configurable():
    args = build_parser().parse_args(
        [
            "--steps",
            "1000",
            "--schedule-start-percent",
            "10",
            "--schedule-end-percent",
            "90",
            "--lr-schedule-start-percent",
            "60",
        ]
    )

    assert args.schedule_start_percent == 10
    assert args.schedule_end_percent == 90
    assert args.lr_schedule_start_percent == 60
    assert resolve_schedule_steps(args) == (399, 800)


def test_coop_cli_separates_inference_and_learner_batch_sizes():
    args = build_parser().parse_args(
        ["--inference-batch-size", "256", "--learner-batch-size", "1024"]
    )

    assert args.inference_batch_size == 256
    assert args.learner_batch_size == 1024


def test_coop_cli_defaults_to_two_sigma_value_clipping():
    args = build_parser().parse_args([])

    assert args.value_clip_epsilon == 2.0


def test_coop_cli_can_disable_value_clipping():
    args = build_parser().parse_args(["--value-clip-epsilon", "none"])

    assert args.value_clip_epsilon is None


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


def test_coop_cli_reads_trackio_url_from_environment(monkeypatch):
    monkeypatch.setenv("TRACKIO_URL", "https://trackio.example")

    args = build_parser().parse_args([])

    assert args.trackio_url == "https://trackio.example"


def test_coop_cli_can_anneal_entropy_strength_to_zero():
    args = build_parser().parse_args(["--entropy-baseline-ratio", "0"])

    assert args.entropy_baseline_ratio == 0.0


def test_coop_cli_keeps_the_perplexity_thermostat_as_default():
    args = build_parser().parse_args([])

    assert args.exploration_controller == "thermostat"
    assert args.perplexity_adaptation_rate == 0.004
    assert args.perplexity_curve == 1.0
    assert args.perplexity_schedule_shape == "cosine"


def test_coop_cli_can_disable_default_best_checkpoints():
    args = build_parser().parse_args(["--no-save-best"])

    assert not args.save_best


def test_coop_cli_configures_perplexity_adaptation_rate():
    args = build_parser().parse_args(
        ["--perplexity-adaptation-rate", "0.01"]
    )

    assert args.perplexity_adaptation_rate == 0.01


def test_coop_cli_can_select_linear_exploration_controller():
    args = build_parser().parse_args(["--exploration-controller", "linear"])

    assert args.exploration_controller == "linear"


def test_coop_cli_supports_resumed_schedule_offsets():
    args = build_parser().parse_args(
        [
            "--schedule-start",
            "980",
            "--lr-schedule-start",
            "1260",
        ]
    )

    assert args.schedule_start == 980
    assert args.lr_schedule_start == 1260


def test_coop_cli_supports_cosine_lr_schedule():
    args = build_parser().parse_args(
        ["--lr-schedule-shape", "cosine"]
    )

    assert args.lr_schedule_shape == "cosine"


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
            "cnn",
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

    assert path.parent == tmp_path / "coop" / "tictactoe" / "cnn" / "coop"
    assert state["step"] == 0
    assert state["metadata"] == {
        "trainer": "coop",
        "game": "tictactoe",
        "architecture": "cnn",
    }
    run_info = path.parent / "run.txt"
    assert run_info.exists()
    assert '"game": "tictactoe"' in run_info.read_text()
    assert "Cooperative PPO training from scratch" in run_info.read_text()


def test_coop_evaluates_and_logs_before_zero_steps(
    tmp_path: Path, monkeypatch
) -> None:
    _FakeEvaluator.calls = []
    _RecordingMetricLogger.instances = []
    monkeypatch.setattr(coop, "Evaluator", _FakeEvaluator)
    monkeypatch.setattr(coop, "RolloutRunner", _FakeRolloutRunner)
    monkeypatch.setattr(coop, "MetricLogger", _RecordingMetricLogger)

    args = build_parser().parse_args(
        [
            "--game",
            "tictactoe",
            "--architecture",
            "cnn",
            "--device",
            "cpu",
            "--steps",
            "0",
            "--evaluation-games",
            "1",
            "--checkpoint-root",
            str(tmp_path),
            "--no-progress",
        ]
    )

    run(args)

    assert _FakeEvaluator.calls == [0]
    assert _RecordingMetricLogger.instances[0].logs == [
        (0, {"evaluation": {"win_rate": 1.0, "points": Range([1.0])}})
    ]


def test_coop_saves_final_checkpoint_when_steps_is_not_save_multiple(
    tmp_path: Path, monkeypatch
) -> None:
    _FakeEvaluator.calls = []
    _RecordingMetricLogger.instances = []
    monkeypatch.setattr(coop, "Evaluator", _FakeEvaluator)
    monkeypatch.setattr(coop, "RolloutRunner", _FakeRolloutRunner)
    monkeypatch.setattr(coop, "MetricLogger", _RecordingMetricLogger)
    monkeypatch.setattr(coop, "Learner", _CompletedLearner)

    args = build_parser().parse_args(
        [
            "--game",
            "tictactoe",
            "--architecture",
            "cnn",
            "--device",
            "cpu",
            "--steps",
            "3",
            "--save-every",
            "10",
            "--evaluation-games",
            "1",
            "--checkpoint-root",
            str(tmp_path),
            "--no-progress",
        ]
    )

    path = run(args)

    state = Checkpoints(path.parent, prefix="step").load(path)
    assert path.name == "step-3.pth"
    assert state["step"] == 3


def test_coop_resume_restores_model_optimizer_and_learner_state(
    tmp_path: Path, monkeypatch
) -> None:
    _FakeEvaluator.calls = []
    _RecordingMetricLogger.instances = []
    monkeypatch.setattr(coop, "Evaluator", _FakeEvaluator)
    monkeypatch.setattr(coop, "RolloutRunner", _FakeRolloutRunner)
    monkeypatch.setattr(coop, "MetricLogger", _RecordingMetricLogger)

    source_model = _CheckpointModel()
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.3)
    source_model.weight.sum().backward()
    source_optimizer.step()
    source_learner = _StatefulLearner("saved")
    checkpoint_dir = tmp_path / "coop" / "tictactoe" / "cnn" / "resume-test"
    checkpoint = Checkpoints(checkpoint_dir).save(
        4,
        {"current": source_model},
        optimizers={"current": source_optimizer},
        states={"learner": source_learner},
    )

    loaded = {}

    def make_model(architecture, game):
        del architecture, game
        model = _CheckpointModel()
        loaded["model"] = model
        return model

    class RecordingReferenceTargets:
        def __init__(self, model, **kwargs):
            del kwargs
            loaded["target_model"] = model

        def __call__(self, samples):
            loaded["target_model_training"] = loaded["target_model"].training
            return samples

    def make_learner(model, game, args):
        del game, args
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        learner = _StatefulLearner("fresh")
        loaded["optimizer"] = optimizer
        loaded["learner"] = learner
        return learner, optimizer

    monkeypatch.setattr(coop, "make_for_game", make_model)
    monkeypatch.setattr(coop, "make_learner", make_learner)
    monkeypatch.setattr(coop, "ReferenceTargets", RecordingReferenceTargets)

    args = build_parser().parse_args(
        [
            "--game",
            "tictactoe",
            "--architecture",
            "cnn",
            "--device",
            "cpu",
            "--steps",
            "5",
            "--evaluation-games",
            "1",
            "--checkpoint-root",
            str(tmp_path),
            "--tag",
            "resume-test",
            "--resume",
            str(checkpoint),
            "--no-progress",
        ]
    )

    run(args)

    assert torch.equal(loaded["model"].weight, source_model.weight)
    assert loaded["target_model"] is loaded["model"]
    assert not loaded["target_model_training"]
    assert loaded["learner"].state == {"marker": "saved"}
    source_state = next(iter(source_optimizer.state_dict()["state"].values()))
    loaded_state = next(iter(loaded["optimizer"].state_dict()["state"].values()))
    assert loaded_state["step"] == source_state["step"]
    assert torch.equal(loaded_state["exp_avg"], source_state["exp_avg"])
