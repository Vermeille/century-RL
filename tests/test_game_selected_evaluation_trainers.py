import importlib.util
from pathlib import Path
import sys

import pytest
import torch

pytest.importorskip("pyspiel")

from boardrl.checkpoints import Checkpoints
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
    logs = []

    def __init__(self, *sinks):
        pass

    def log(self, step, **values):
        self.logs.append((step, values))


@pytest.mark.parametrize("trainer,logger_owner,names", [
    (coop, coop, None),
    (fightbot, fightbot, None),
    (advshape, advshape, ("agent", "environment")),
    (ppo, adversarial_selfplay, ("agent",)),
    (mmd, adversarial_selfplay, ("agent",)),
    (imitate, imitate, None),
])
def test_leduc_trainers_log_exact_metrics_and_select_best(
    tmp_path, monkeypatch, trainer, logger_owner, names,
):
    RecordingLogger.logs = []
    monkeypatch.setattr(logger_owner, "MetricLogger", RecordingLogger)
    argv = [
        "--game", "leduc", "--architecture", "toy", "--device", "cpu",
        "--steps", "0", "--checkpoint-root", str(tmp_path), "--no-progress",
    ]
    if trainer is imitate:
        argv += ["--strategy", "random"]
    else:
        argv += ["--opponent-eval-strategy", "does-not-exist"]
    if trainer is fightbot:
        argv += ["--opponent-bot", "random"]
    args = trainer.build_parser().parse_args(argv)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        path = trainer.run(args)
    finally:
        torch.set_num_threads(threads)
    assert len(RecordingLogger.logs) == 1
    step, values = RecordingLogger.logs[0]
    assert step == 0
    groups = [values["evaluation"]] if names is None else [values["evaluation"][name] for name in names]
    for metrics in groups:
        assert set(metrics) == {"exploitability", "nash_conv"}
        assert metrics["exploitability"] >= 0
        assert metrics["nash_conv"] == 2 * metrics["exploitability"]
    if trainer is not imitate:
        best = Checkpoints(path.parent, prefix="best").load()
        metadata = best["metadata"]
        assert metadata["evaluation_metric"] == "exploitability"
        assert metadata["evaluation_value"] == groups[0]["exploitability"]
        assert metadata["evaluation_score"] == -metadata["evaluation_value"]
        assert "evaluation_points" not in metadata
