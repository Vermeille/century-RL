import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from boardrl.metrics import MetricLogger, Range, Trackio, make_trackio


class RunRecorder:
    def __init__(self) -> None:
        self.logs = []
        self.finished = False

    def log(self, values, *, step):
        self.logs.append((values, step))

    def finish(self):
        self.finished = True


def test_trackio_sink_normalizes_training_metrics() -> None:
    run = RunRecorder()
    sink = Trackio(run)

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
                "train/loss": 0.25,
                "rollout/win_rate/0": 0.5,
                "rollout/win_rate/1": 0.75,
                "game/points": 12.0,
            },
            12,
        )
    ]


def test_metric_logger_can_exclude_a_metric_namespace() -> None:
    run = RunRecorder()

    MetricLogger(Trackio(run)).game(
        12,
        SimpleNamespace(
            metrics=lambda: {
                "strategy": {"0": {"win_rate": 0.5}},
                "seat": {"0": {"win_rate": 1.0}},
            }
        ),
        exclude=("seat",),
    )

    assert run.logs == [
        ({"game/strategy/0/win_rate": 0.5}, 12),
    ]


def test_trackio_sink_finishes_run() -> None:
    run = RunRecorder()

    Trackio(run).finish()

    assert run.finished


def test_make_trackio_is_opt_in() -> None:
    assert make_trackio(project=None) is None


def test_make_trackio_passes_run_configuration(monkeypatch, tmp_path: Path) -> None:
    run = RunRecorder()
    calls = []
    fake_trackio = SimpleNamespace(
        init=lambda **kwargs: (calls.append(kwargs) or run)
    )
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)

    sink = make_trackio(
        project="century",
        name="trial",
        config={"checkpoint_root": tmp_path},
    )

    assert sink is not None
    assert calls == [
        {
            "project": "century",
            "config": {"checkpoint_root": str(tmp_path)},
            "name": "trial",
        }
    ]


def test_make_trackio_passes_server_url_when_provided(monkeypatch) -> None:
    run = RunRecorder()
    calls = []
    fake_trackio = SimpleNamespace(
        init=lambda **kwargs: (calls.append(kwargs) or run)
    )
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)

    make_trackio(project="century", server_url="https://trackio.example")

    assert calls == [
        {
            "project": "century",
            "config": {},
            "server_url": "https://trackio.example",
        }
    ]
