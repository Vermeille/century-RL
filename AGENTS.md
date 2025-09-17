# Contributor Guide

This repository contains a small reinforcement learning framework that supports several board games. Below are key concepts and commands to help you navigate the project.

## Repository Structure

- `boardrl/`: main Python package containing utilities, game logic, RL models and the web server.
- `boardrl/rl/model/`: transformer-based policy/value network implementation.
- `boardrl/training/`: sampling helpers and return calculations.
- `boardrl/experiments/`: toy tasks used by tests to sanity-check models and transformers.
- `configs/`: YAML files specifying game choice, network size and training parameters.
- `tests/`: unit tests for core functionality.
- `main.py`: entry point for training agents.
- `run_pit.py`: pits two strategies against each other for evaluation.
- `pyproject.toml`: Project metadata and dependencies managed by `uv`.
- `uv.lock`: Locked dependency versions.

### Utilities

`boardrl/utils.py` provides a batching helper used for async inference and a registry for creating objects by name. The batching helper collects requests and processes them in groups, while `RegisterByName` lets you instantiate classes using a string key. This mechanism is used throughout the project to register strategies, games and other components.

### Game Library

Games are registered under `boardrl/games/__init__.py` using a `GameDesc` object that describes how to create a game instance, the available strategies and metrics. Games such as `century`, `tictactoe` and `connectfour` are available. See `boardrl/games/AGENTS.md` for guidance on implementing new games.

### RL Models

`boardrl/rl/model/model.py` implements the transformer-based policy/value network. It defines building blocks like pooling layers and a `PolicyValue` container used throughout the training loop.

### Training Loop

`main.py` defines the `Trainer` which loads a model, runs self-play and optimizes losses using the algorithms specified in the YAML configuration.

### Web Server

The FastAPI server in `boardrl/serve/serve.py` exposes a simple interface for playing a game with a chosen strategy. Strategies are discovered automatically from saved model files.

## Testing

Use `pytest` for the test suite:

```bash
uv run pytest tests -k "not transformer and not model and not config"
```

The `tests/test_transformer.py` and `tests/test_model.py` suites run lengthy training loops and should only be executed when modifying the corresponding modules. `tests/test_fast_sample.py` builds the Cython extension `boardrl/cyutils.pyx`, so a C compiler is required if that file changes. `tests/test_configs.py` runs the training entry point for every configuration and is slow; only run this file when modifying configs or `main.py`.

When adding new long-running tests, annotate them with `@pytest.mark.slow` so the CI workflow can skip them.

## Getting Started

1. **Install dependencies**:

   ```bash
   uv sync
   ```

2. **Train an agent**:

   ```bash
   python main.py configs/<config>.yaml
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

- Modify the YAML configs to try different algorithms or games.
- Study individual game implementations in `boardrl/games/`.
- `boardrl/games/century/engine.pyx` contains a Cython-based environment for the Century game, which requires a C compiler if modified.
