[![Tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

# Century RL

A composable reinforcement learning library for board games. Training
algorithms are ordinary Python programs: the library supplies rollout,
post-processing, optimization, metrics, checkpoint, and evaluation mechanics
without owning the experiment loop.

## Documentation

Start with the [documentation home](docs/index.md), then follow the
[quickstart](docs/quickstart.md) to train, evaluate, and play a tiny CPU model.

- [Training programs and controls](docs/training.md)
- [Game catalog](docs/games.md) and [adding a game](docs/adding-games.md)
- [Writing algorithms](docs/training-library.md) and [internal data flow](docs/internals/rollouts.md)
- [Evaluation and checkpoints](docs/evaluation.md)
- [Browser game workbench and HTTP API](docs/web-server.md)

The documentation has a searchable MkDocs site and a
[GitHub Pages deployment workflow](docs/documentation.md). Preview locally with
`uvx --from mkdocs==1.6.1 mkdocs serve`.

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
