import importlib.util
import sys
from pathlib import Path
from statistics import mean
from typing import ClassVar

import pytest

pytest.importorskip("pyspiel")

from boardrl.checkpoints import Checkpoints
from boardrl.evaluation import Evaluator
from boardrl.games.openspiel import evaluation as openspiel_evaluation
from trainers import adversarial_selfplay, coop, fightbot, imitate


def _trainer(filename):
    name = "trainers.evaluation_test_" + filename.replace("-", "_")
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "trainers" / (filename + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


advshape = _trainer("adversarial-advshape")
ppo = _trainer("adversarial-ppo")
mmd = _trainer("adversarial-mmd")


class RecordingLogger:
    logs: ClassVar[list] = []

    def __init__(self, *sinks):
        pass

    def log(self, step, **values):
        self.logs.append((step, values))


@pytest.mark.parametrize(
    "trainer,logger_owner,names",
    [
        (coop, coop, None),
        (fightbot, fightbot, None),
        (advshape, advshape, ("agent", "environment")),
        (ppo, adversarial_selfplay, ("agent",)),
        (mmd, adversarial_selfplay, ("agent",)),
        (imitate, imitate, None),
    ],
)
@pytest.mark.parametrize(
    "game,metric", [("phantom_ttt", "win_rate"), ("leduc", "points")]
)
def test_trainers_play_opponent_without_constructing_exact_evaluator(
    tmp_path,
    monkeypatch,
    trainer,
    logger_owner,
    names,
    game,
    metric,
):
    RecordingLogger.logs = []
    monkeypatch.setattr(logger_owner, "MetricLogger", RecordingLogger)

    def forbid_exact(*args, **kwargs):
        pytest.fail("sampled evaluation must not construct the exact evaluator")

    monkeypatch.setattr(openspiel_evaluation, "ExactOpenSpielEvaluator", forbid_exact)
    comparisons = []
    compare = Evaluator.compare

    def record_compare(self, players, **kwargs):
        comparisons.append(kwargs)
        return compare(self, players, **kwargs)

    monkeypatch.setattr(Evaluator, "compare", record_compare)
    argv = [
        "--game",
        game,
        "--architecture",
        "toy",
        "--device",
        "cpu",
        "--steps",
        "0",
        "--checkpoint-root",
        str(tmp_path),
        "--no-progress",
        "--opponent-eval-strategy",
        "random",
        "--evaluation-games",
        "6",
    ]
    if trainer is imitate:
        argv += ["--strategy", "random"]
    if trainer is fightbot:
        argv += ["--opponent-bot", "random"]
    path = trainer.run(trainer.build_parser().parse_args(argv))

    assert len(RecordingLogger.logs) == 1
    step, values = RecordingLogger.logs[0]
    assert step == 0
    groups = (
        [values["evaluation"]]
        if names is None
        else [values["evaluation"][name] for name in names]
    )
    assert len(comparisons) == len(groups)
    for options in comparisons:
        assert options["games"] == 6
        assert options.get("rotate", True) is True
    for metrics in groups:
        assert "exploitability" not in metrics
        assert "nash_conv" not in metrics
        assert 0 <= metrics["win_rate"] <= 1
        assert metric in metrics
    if trainer is not imitate:
        metadata = Checkpoints(path.parent, prefix="best").load()["metadata"]
        value = groups[0][metric]
        expected = mean(value.values) if metric == "points" else value
        assert metadata["evaluation_points"] == pytest.approx(expected)
