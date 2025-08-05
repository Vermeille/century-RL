"""Run toy training experiments and plot their learning curves.

Each experiment function returns a list of metrics recorded during training.
This script executes the chosen experiments for a set of model architectures
and saves a combined plot for each experiment.
"""

import argparse
import pathlib
import sys

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
)
from boardrl.rl.model.model import Model

EXPERIMENTS = {
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
        [("default", lambda: Model(dim=32, num_layers=2, head_size=8))],
    ),
    "model_context": (
        model_learns_from_context,
        [("default", lambda: Model(dim=64, num_layers=3, head_size=8))],
    ),
    "model_dyck": (
        model_dyck,
        [("default", lambda: Model(64, 4, 32))],
    ),
}


def run_experiment(name: str) -> None:
    func, model_fns = EXPERIMENTS[name]
    histories = []
    for label, fn in model_fns:
        history = func(fn())
        histories.append((label, history))

    out_dir = pathlib.Path(__file__).resolve().parent
    plt.figure()
    for label, hist in histories:
        plt.plot(hist, label=label)
    plt.legend()
    plt.title(name)
    plt.xlabel("iteration (×10)")
    plt.ylabel("metric")
    path = out_dir / f"{name}.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=EXPERIMENTS.keys(),
        nargs="*",
        default=list(EXPERIMENTS.keys()),
        help="Which experiments to run (default: all)",
    )
    args = parser.parse_args()
    for name in args.experiment:
        run_experiment(name)


if __name__ == "__main__":
    main()
