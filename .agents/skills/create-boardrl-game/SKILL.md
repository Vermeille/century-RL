---
name: create-boardrl-game
description: Create and integrate a new game in the BoardRL Python framework. Use when adding a board or card game under boardrl/games, implementing its game state and rules, adding metrics or strategies, registering it in games_library, exposing an optional web UI, or writing and running focused tests for the new game.
---

# Create a BoardRL Game

Use this skill for end-to-end game additions in this repository. Preserve the
repository's existing game contracts and local AGENTS.md instructions, and keep
the implementation small enough to test rule-by-rule.

## Inspect Before Editing

1. Read the nearest `AGENTS.md`, especially `boardrl/games/AGENTS.md`.
2. Inspect two comparable implementations:
   - `boardrl/games/tictactoe/` or `connectfour/` for a simple alternating game.
   - `boardrl/games/nim/`, `sum/`, or `thegame/` when constructor arguments,
     custom strategies, or nontrivial move rules are needed.
3. Inspect `boardrl/games/__init__.py`, `boardrl/utils/registerbyname.py`, and
   the closest existing tests before choosing the registration shape.
4. Check `git status --short` and preserve unrelated user edits, generated
   checkpoints, logs, and documentation unless the request explicitly includes
   them.

## Define the Game Contract

Write down the rules and invariants before coding: player count, initial state,
move-string format, legal-move calculation, turn/round semantics, terminal
conditions, winner and scoring rules, and which state must be copied.

Create `boardrl/games/<game_name>/game.py` with a game class that provides the
framework-facing behavior below:

- Maintain `self.moves` as the current legal moves, represented as strings.
- Implement `current_player()` and `round()` consistently from the turn state.
- Implement `display()` and `display_with_moves()`. The latter must include
  legal actions on lines prefixed with `@`, because the model and web server
  consume that representation.
- Implement `play_str(move)` and, when index-based callers are useful,
  `play_idx(index)`. Reject moves that are not legal and reject play after the
  game ends.
- Implement `ended()`, `winner()`, and `points_for(player)`. Provide
  `points()` and `diff_points()`/`diff_points_for()` when the game model uses
  those convenience methods.
- Implement `simulate_to_end()` using random legal moves so rollouts can smoke
  test termination.
- Implement `copy()` with independent mutable state. Test that changing a copy
  does not change the original.

Prefer explicit domain objects and polymorphism in the style of this repository;
do not introduce `isinstance`-based dispatch to work around the design.

## Add Supporting Modules

Add `metrics.py` with a `Metrics(GameMetrics)` class when the game will be used
in training/evaluation. Store the supplied trace data, return a nested
dictionary from `metrics()`, and add `print_short_history()` when a compact
terminal trace is useful. Use existing metrics modules as the shape reference;
the framework supplies console and visualization sinks.

Add `strategies.py` only when the game needs game-specific strategies. Register
them with `RegisterByName`; pass the registry to `register_game`, which copies
and merges it with `boardrl.games.strategies.strategy_from_string` so the
default strategies remain available.

Add `augmentations.py` only for transformations that preserve game semantics
and are needed by training. Add `ui.html` only when manual web play is part of
the request; inspect the existing UI and `boardrl/serve/static/app.js` before
adding a new renderer or server-side game branch.

## Register the Game

Update `boardrl/games/__init__.py` with a declarative `register_game(...)` call.
The helper resolves the sibling `metrics.Metrics` class lazily and builds the
`GameDesc` for conventional Python games.

- Pass the game class directly.
- Pass `args_from=GameClass` only when constructor parameters should be exposed
  in textual specs such as `nim,num_stones=5,max_pick=2`.
- Omit `args_from` when constructor parameters are internal; the rollout engine
  can still pass values such as `num_players` directly to `make_game`.
- Pass a game-specific strategy registry with `strategies=` when one exists.
- Keep score semantics, outcome semantics, rewards, and augmentations explicit.
- Use a custom `GameDesc` factory only for a genuinely exceptional integration
  such as Century's Cython setup.

For a simple game, the registration should have the same essential shape as:

```python
from boardrl.games.mygame.game import MyGame

register_game(
    "mygame",
    MyGame,
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
)
```

For a configurable game with custom strategies:

```python
from boardrl.games.mygame.game import MyGame
from boardrl.games.mygame.strategies import strategy_from_string as my_strategies

register_game(
    "mygame",
    MyGame,
    strategies=my_strategies,
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
    args_from=MyGame,
)
```

Adjust the example to the game needs; do not add a descriptor subclass merely
to forward constructor arguments or merge registries.

## Test the Addition

Add focused tests, normally in `tests/test_<game_name>.py`, covering:

- registration through `games_library` and any constructor arguments;
- initial state, legal move strings, player/round behavior, and display format;
- representative legal moves and illegal moves;
- every terminal path, winner, draw handling, and score symmetry;
- copy independence and random playout termination;
- custom strategy registration when applicable.

If a web UI is added, test its renderer and the server state/action flow using
the existing `tests/test_serve.py` patterns. Avoid expanding shared static UI
code for a game that is not requested.

Run the smallest relevant checks first:

```bash
uv run pytest tests/test_<game_name>.py --capture=no
uv run pytest tests/test_games.py tests/test_<game_name>.py --capture=no
```

Then run the repository's non-long-running suite when the change crosses shared
registration or server code:

```bash
uv run pytest tests -k "not transformer and not model" --capture=no
```

Use `--capture=no` if pytest teardown reports a capture-path `FileNotFoundError`;
that is a known environment failure mode here. Run Cython/compiler-dependent
tests only when the relevant Cython game code changed.

Before handoff, inspect `git diff --check`, review the diff for scope, and
report the exact test commands and any remaining limitations.
