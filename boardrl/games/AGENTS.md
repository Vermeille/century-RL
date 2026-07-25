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
Visdom, and future sinks are supplied automatically. Examples can be found in
[`connectfour/metrics.py`](connectfour/metrics.py) and
[`sum/metrics.py`](sum/metrics.py).

## 3. Strategies (optional)
Game-specific strategies may be provided in `strategies.py`. Define a
`strategy_from_string` registry using `RegisterByName`, mirroring the pattern in
[`sum/strategies.py`](sum/strategies.py). When registering the game, merge this
registry with the default one from `boardrl/games/strategies` so both sets of
strategies are available.

## 4. Register the Game
Expose the game to the rest of the library by registering it in
`boardrl/games/__init__.py` using `games_library.register` and `GameDesc`:

```python
from boardrl.games.strategies import strategy_from_string

@games_library.register("mygame")
class MyGame(GameDesc):
    def __init__(self):
        from boardrl.games.mygame.game import MyGame as Game
        from boardrl.games.mygame.metrics import Metrics
        from boardrl.games.mygame.strategies import (
            strategy_from_string as my_strats,
        )  # optional

        strats = my_strats.copy().update(strategy_from_string)
        super().__init__(Game, strats, Metrics)
```

For simpler games without custom strategies, pass `strategy_from_string` from
`boardrl/games/strategies` directly.

Use existing games such as `tictactoe`, `connectfour`, and `sum` as templates
when building new ones.

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
