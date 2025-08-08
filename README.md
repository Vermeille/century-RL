[![Tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

# Century RL

A small reinforcement learning framework for board games.

## Installation

This project requires **Python 3.12+**. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Tests

```bash
uv run pytest tests -k "not transformer and not model and not config"
```

Also see AGENTS.md
