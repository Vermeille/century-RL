# Game Development Guide

This directory contains implementations of individual board games used by the
RL framework. To add a new game, follow the conventions below.

## 1. Create a Game Module
Each game lives in its own subfolder under `boardrl/games/<game_name>/`.
At a minimum, the folder should contain a `game.py` file defining a class with
these responsibilities:

- Maintain a list of legal moves in `self.moves` (as strings).
- Track the current player with a `current_player()` method and expose the
  current round via `round()`.
- Provide string representations with `display()` and
  `display_with_moves()`.
- Apply moves through `play_str(move)` and `play_idx(index)` using the same
  ordered legal-action list.
- Report terminal states with `ended()`. Cooperative outcomes also require
  `won()`; `winner()` is an optional game-specific convenience.
- Expose `points_for(player)`, `points()`, `diff_points_for(player)`, and
  `diff_points()` with explicit score semantics.
- Allow random playouts via `simulate_to_end()`.
- Support cloning through `copy()`.

See [`tictactoe/game.py`](tictactoe/game.py) or
[`connectfour/game.py`](connectfour/game.py) for reference implementations.

## 2. Metrics
Include a `metrics.py` that defines a `Metrics(GameMetrics)` class. It accepts
simulation traces and returns a nested dictionary from `metrics()`. Implement
`print_short_history()` when a compact trace display is useful. Experiments
configure console, Trackio, or custom sinks through `MetricLogger`; the same
metric values work with each sink. Examples can be found in
[`connectfour/metrics.py`](connectfour/metrics.py) and
[`sum/metrics.py`](sum/metrics.py).

## 3. Strategies (optional)
Game-specific strategies may be provided in `strategies.py`. Define a
`strategy_from_string` registry using `RegisterByName`, mirroring the pattern in
[`sum/strategies.py`](sum/strategies.py). Pass that registry to `register_game`;
it is copied and merged with the default strategies automatically.

## 4. Register the Game
Conventional Python games are registered declaratively in
`boardrl/games/__init__.py` with `register_game`. The helper loads the sibling
`metrics.Metrics` lazily, builds the `GameDesc`, and merges an optional custom
strategy registry with the defaults.

```python
from boardrl.games.mygame.game import MyGame
from boardrl.games.mygame.strategies import strategy_from_string as my_strategies

register_game(
    "mygame",
    MyGame,
    strategies=my_strategies,  # omit when there are no custom strategies
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
    args_from=MyGame,  # include only when constructor args belong in the game spec
)
```

Omit `args_from` for games whose constructor options are implementation details
rather than public registry arguments. For example, the rollout engine may pass
`num_players` to `make_game` without making `num_players` part of the textual
game specification.

Use a custom `GameDesc` factory only when ordinary registration cannot describe
the game. Century remains the main example because its Cython class cannot be
introspected normally and requires custom import/setup work.

## 5. Web UI

The shared workbench discovers registered games automatically and offers a raw
text fallback. Do not add a per-game HTML page. See
[`docs/web-server.md`](../../docs/web-server.md) for the canonical HTTP API,
checkpoint discovery, renderer contract, and tests.

- Read the canonical snapshot from `/state?game=<spec>`; use its ordered `moves`
  list for controls instead of scraping the observation.
- Submit one move to `/do-one` as JSON `{ "action": "<move>" }`.
- Add visual renderers in `boardrl/serve/static/app.js`, emitting
  `data-action="<move>"` controls and preserving the raw-state view.
- Run `tests/test_serve.py`, `tests/test_serve_tictactoe.py`, and
  `node --check boardrl/serve/static/app.js` for renderer changes.

The complete game tutorial and protocol are in
[`docs/adding-games.md`](../../docs/adding-games.md).
