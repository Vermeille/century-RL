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
- Apply moves through `play_str(move)` and optionally `play_idx(index)`.
- Report when the game ends using `ended()` and expose the winner with
  `winner()`.
- Score outcomes with `points_for(player)` (and `points()` or
  `diff_points()` for the current player).
- Allow random playouts via `simulate_to_end()`.
- Support cloning through `copy()`.

See [`tictactoe/game.py`](tictactoe/game.py) or
[`connectfour/game.py`](connectfour/game.py) for reference implementations.

## 2. Metrics
Include a `metrics.py` that defines a `Metrics(GameMetrics)` class. It accepts
simulation traces and returns a nested dictionary from `metrics()`. Implement
`print_short_history()` when a compact trace display is useful. Console,
Trackio, and future sinks are supplied automatically. Examples can be found in
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
Games may expose a small web interface for manual play. Place a
`ui.html` file in the game's directory and follow the pattern used by
existing games:

- Fetch the state from `/board` and render the board along with the
  list of legal moves (lines prefixed by `@`).
- Send actions to `/do` with a JSON body containing `action` and an
  optional `strategy` selected from `/strategies`.
- Provide an `Analyze` button that calls `/analyze` and visualises the
  returned move values. A `Play` button can call `/play-one` to let the
  server make a move.
- Reuse the dark style seen in `tictactoe/ui.html` or
  `connectfour/ui.html` for consistency.
- Present move choices with direct UI controls (e.g. buttons or
  drag-and-drop) rather than text prompts.

Refer to those HTML files for concrete examples when adding a new UI.
