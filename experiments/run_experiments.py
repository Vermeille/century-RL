"""Run toy training experiments and plot their learning curves.

Each experiment function returns a list of metrics recorded during training.
This script executes the chosen experiments for a set of model architectures
and saves a single image containing a subplot for each experiment.
"""

import argparse
import pathlib
import sys
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt

# Ensure the repo root is on the Python path so that `boardrl` is importable when
# this script is executed directly.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from boardrl.experiments.transformer import (
    ToyMLM,
    transformer_learns_alphabet_mlm,
    transformer_needs_context_mlm,
    transformer_positional,
)
from boardrl.experiments.model import (
    model_learns_policy_and_value,
    model_learns_from_context,
    model_dyck,
    model_arith_mod20,
)
from boardrl.rl.model.model import Model

Experiment = Callable[[Any], list[float]]
ModelFactory = Callable[[], Any]
ExperimentSpec = tuple[Experiment, list[tuple[str, ModelFactory]]]


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "transformer_alphabet": (
        transformer_learns_alphabet_mlm,
        [
            ("lpe", lambda: ToyMLM(dim=64)),
            ("rotary", lambda: ToyMLM(dim=64, rotary=True)),
        ],
    ),
    "transformer_shifted": (
        transformer_needs_context_mlm,
        [
            ("lpe", lambda: ToyMLM(dim=64)),
            ("rotary", lambda: ToyMLM(dim=64, rotary=True)),
        ],
    ),
    "transformer_positional": (
        transformer_positional,
        [
            ("lpe", lambda: ToyMLM()),
            ("rotary", lambda: ToyMLM(rotary=True)),
        ],
    ),
    "model_policy_value": (
        model_learns_policy_and_value,
        [
            (
                "default",
                lambda: Model(dim=32, num_layers=2, head_size=8, num_heads=4),
            )
        ],
    ),
    "model_context": (
        model_learns_from_context,
        [
            (
                "default",
                lambda: Model(dim=64, num_layers=3, head_size=8, num_heads=8),
            )
        ],
    ),
    "model_dyck": (
        model_dyck,
        [("default", lambda: Model(64, 4, 32, 2))],
    ),
    "model_arith_mod20": (
        model_arith_mod20,
        [("default", lambda: Model(64, 4, 32, 2))],
    ),
}


def run_experiment(name: str):
    """Run a single experiment and return its history."""
    func, model_fns = EXPERIMENTS[name]
    histories = []
    for label, fn in model_fns:
        print("running", name, label)
        history = func(fn())
        histories.append((label, history))
    return name, histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=list(EXPERIMENTS.keys()),
        nargs="*",
        help="Which experiments to run (default: all)",
    )
    args = parser.parse_args()
    names = args.experiment or list(EXPERIMENTS.keys())

    results = [run_experiment(name) for name in names]

    out_dir = pathlib.Path(__file__).resolve().parent
    fig, axes = plt.subplots(len(results), 1, figsize=(6, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, histories) in zip(axes, results):
        for label, hist in histories:
            ax.plot(hist, label=label)
        ax.set_title(name)
        ax.set_xlabel("iteration (×10)")
        ax.set_ylabel("metric")
        ax.legend()

    fig.tight_layout()
    path = out_dir / "experiments.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
