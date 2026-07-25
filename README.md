[![Tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

# Century RL

A composable reinforcement learning library for board games. Training
algorithms are ordinary Python programs: the library supplies rollout,
post-processing, optimization, metrics, checkpoint, and evaluation mechanics
without owning the experiment loop.

## Installation

This project requires **Python 3.12+**. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Tests

```bash
uv run pytest tests -k "not transformer and not model"
```

## Training

The complete PPO/TD(lambda) experiment for The Game is now executable Python:

```bash
uv run python trainers/coop.py --game thegame --architecture cnn
```

Start with [the training library guide](docs/training-library.md) for shorter
recipes including self-play, NFSP-like two-model training, champion promotion,
EMA opponents, imitation, tree-search imitation, and checkpoint populations.

Also see `AGENTS.md`.
