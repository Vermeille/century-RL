# Contributor Guide

This repository contains a small reinforcement learning framework that supports several board games. Below are key concepts and commands to help you navigate the project.

## Repository Structure

- `boardrl/`: main Python package containing utilities, game logic, RL models and the web server.
- `boardrl/rl/model/`: transformer-based policy/value network implementation.
- `boardrl/training/`: sampling helpers and return calculations.
- `boardrl/experiments/`: toy tasks used by tests to sanity-check models and transformers.
- `boardrl/rollouts.py`: explicit player lineups and batched model inference.
- `boardrl/training/`: composable rollout processing and sample optimization.
- `boardrl/checkpoints.py`: named multi-model checkpoint management.
- `boardrl/evaluation.py`: evaluation results and optional scoreboards.
- `boardrl/metrics.py`: metric collection and display sinks.
- `trainers/`: executable training algorithms written as Python.
- `examples/`: smaller algorithm examples.
- `tests/`: unit tests for core functionality.
- `run_pit.py`: pits two strategies against each other for evaluation.
- `pyproject.toml`: Project metadata and dependencies managed by `uv`.
- `uv.lock`: Locked dependency versions.

### Utilities

`boardrl/utils.py` provides a batching helper used for async inference and a registry for creating objects by name. The batching helper collects requests and processes them in groups, while `RegisterByName` lets you instantiate classes using a string key. This mechanism is used throughout the project to register strategies, games and other components.

### Game Library

Games are registered under `boardrl/games/__init__.py` using a `GameDesc` object that describes how to create a game instance, the available strategies and metrics. Games such as `century`, `tictactoe` and `connectfour` are available. See `boardrl/games/AGENTS.md` for guidance on implementing new games.

### RL Models

`boardrl/rl/model/model.py` implements the transformer-based policy/value network. It defines building blocks like pooling layers and a `PolicyValue` container used throughout the training loop.

### Training Algorithms

There is no framework-owned training loop or training configuration language.
Experiments compose `RolloutRunner`, post-processing `Pipeline` steps,
`Learner`, `MetricLogger`, `Checkpoints`, and `Evaluator` in ordinary Python.
Opponent selection, reference updates, promotion, and data routing belong to
the experiment. See `docs/training-library.md`.

### Web Server

The FastAPI server in `boardrl/serve/serve.py` exposes a simple interface for playing a game with a chosen strategy. Strategies are discovered automatically from saved model files.

## Testing

Use `pytest` for the test suite:

```bash
uv run pytest tests -k "not transformer and not model"
```

The `tests/test_transformer.py` and `tests/test_model.py` suites run lengthy training loops and should only be executed when modifying the corresponding modules. `tests/test_fast_sample.py` builds the Cython extension `boardrl/cyutils.pyx`, so a C compiler is required if that file changes.

When adding new long-running tests, annotate them with `@pytest.mark.slow` so the CI workflow can skip them.

## Getting Started

1. **Install dependencies**:

   ```bash
   uv sync
   ```

2. **Train an agent**:

   ```bash
   uv run python trainers/coop.py --game thegame --architecture cnn
   ```

3. **Evaluate**:

   ```bash
   python run_pit.py --games 32
   ```

4. **Play via web**:

   ```bash
   uvicorn boardrl.serve.serve:app
   ```

## Style

We do very well designed OOP. Small classes, polymorphism. We try to avoid ifs that could be polymorphism, and under no circumstance do we use isinstance  or other ways to circumvent good OOP.

## Further Exploration

- Copy an example and edit its Python loop to try a different algorithm.
- Study individual game implementations in `boardrl/games/`.
- `boardrl/games/century/engine.pyx` contains a Cython-based environment for the Century game, which requires a C compiler if modified.
