import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from boardrl.metrics import Range, Wandb, make_wandb


class RunRecorder:
    def __init__(self) -> None:
        self.logs = []
        self.finished = False

    def log(self, values, *, step):
        self.logs.append((values, step))

    def finish(self):
        self.finished = True


def test_wandb_sink_normalizes_training_metrics() -> None:
    run = RunRecorder()
    sink = Wandb(run)

    sink.log(
        12,
        {
            "train": {"loss": torch.tensor(0.25)},
            "rollout": {"win_rate": [0.5, 0.75]},
            "game": {"points": Range([10.0, 14.0])},
        },
    )

    assert run.logs == [
        (
            {
                "train.loss": 0.25,
                "rollout.win_rate.0": 0.5,
                "rollout.win_rate.1": 0.75,
                "game.points": 12.0,
            },
            12,
        )
    ]


def test_wandb_sink_finishes_run() -> None:
    run = RunRecorder()

    Wandb(run).finish()

    assert run.finished


def test_make_wandb_is_opt_in() -> None:
    assert make_wandb(project=None) is None


def test_make_wandb_passes_run_configuration(monkeypatch, tmp_path: Path) -> None:
    run = RunRecorder()
    calls = []
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: (calls.append(kwargs) or run)
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    sink = make_wandb(
        project="century",
        entity="team",
        name="trial",
        config={"checkpoint_root": tmp_path},
    )

    assert sink is not None
    assert calls == [
        {
            "project": "century",
            "mode": "online",
            "config": {"checkpoint_root": str(tmp_path)},
            "entity": "team",
            "name": "trial",
        }
    ]
