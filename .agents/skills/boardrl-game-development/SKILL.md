---
name: boardrl-game-development
description: Implement, modify, review, and integrate games in the BoardRL repository. Use for adding a new game, changing game rules or state, defining game metrics or strategies, wiring a game into the registry, adding action augmentations or web rendering, and tracing how game objects interact with rollouts, evaluation, training, serving, and tests.
---

# BoardRL Game Development

Use this skill when a task touches a game under `boardrl/games/` or any boundary
where a game is consumed. Treat the game as a small state machine behind a
stable protocol: legal actions are ordered strings, model inputs are serialized
state plus `@`-prefixed actions, and every transition must refresh the legal
action list.

## Begin with repository context

Read the applicable `AGENTS.md` files first: the repository guide,
`boardrl/games/AGENTS.md`, and the game-specific guide if one exists. Then read
[`references/source-map.md`](references/source-map.md) and inspect the closest
working examples before editing. Prefer `tictactoe` or `connectfour` for a
small alternating game, `sum` or `nim` for a compact rule/state machine,
`thegame` for phases, partial information, cooperative play, and
augmentations, and `century` only when Cython integration is required.

Keep the repository's OOP style: use small collaborating classes and
polymorphism; do not introduce `isinstance`-style type dispatch or unrelated
refactors. Preserve unrelated worktree changes and artifacts.

## Choose the change path

### Add a new game

1. Create `boardrl/games/<name>/game.py`. Add `metrics.py`; add
   `strategies.py`, `augmentations.py`, UI rendering, or a local `AGENTS.md`
   only when the game needs them.
2. Implement the game protocol below and write deterministic rule tests before
   wiring training or serving.
3. Register a conventional Python game with `register_game` in
   `boardrl/games/__init__.py`; use a custom `GameDesc` factory only for an
   exceptional integration such as Century's Cython setup.
4. Run registry, strategy, rollout, metrics, and server smoke tests. Add
   renderer tests only if `static/app.js` changes.

### Change an existing game

Trace the invariant through all consumers before editing: state construction
and `copy()`, legal move generation, `play_str`/`play_idx`, display text,
terminal and score semantics, metrics, augmentations, registry arguments,
rollout samples, and server rendering. Update focused tests for both the rule
and the affected boundary. Do not assume a game-only change is isolated if it
changes move strings, action ordering, visibility, player count, or scoring.

### Change a game boundary

For registration changes, inspect `register_game`, `GameDesc`, `RegisterByName`,
and all callers of `games_library`. For training changes, inspect `Record`,
`EndState`, `Rollouts`, and `RolloutRunner` in `boardrl/rollouts.py`. For
cooperative outcome changes, trace the `coop` flag through game descriptors,
rollout/evaluation metrics, and trainer callers; the flag alone does not
automatically redefine win rate. For web changes, follow
`boardrl/serve/README.md` and preserve the generic raw-text renderer and
canonical `moves` payload.

## Implement the game protocol

Make the following attributes and methods correct at every reachable state:

- `moves: list[str]`: the complete ordered legal-action list for the current
  player. Use stable, unambiguous strings. Keep it empty after terminal states.
- `num_players`: the supported player count or constructor-derived count.
- `current_player() -> int` and `round() -> int`: return the active seat and a
  monotonic round/turn measure used in traces.
- `display(force=-1) -> str`: serialize the state visible to the active player;
  honor `force` when the server or end-state reporting asks for a player view.
- `display_with_moves() -> str`: append one line `@<move>` for every entry in
  `moves`. The line order is the model's action index order. Keep the state
  text parseable and do not put unrelated lines beginning with `@` in it.
- `play_str(move)`: reject illegal or terminal moves, apply exactly one legal
  transition, update turn/phase/state, and recompute `moves` immediately.
- `play_idx(index)`: call the current ordered move list and then
  `play_str(self.moves[index])`; never maintain a second action mapping.
- `ended() -> bool`: report true for every terminal rule outcome, including
  draws, exhausted resources, or a stuck active player as appropriate.
- `points_for(player)`, `points()`, `diff_points_for(player)`, and
  `diff_points()`: provide the score values consumed by rollouts and
  evaluation. Keep the meaning explicit for wins, losses, ties, cooperative
  objectives, and truncation.
- `copy()`: return an independent state copy with no shared mutable board,
  hand, deck, history, or move list. If randomized copies are useful, accept
  `randomize=False` and preserve the same semantic state.
- `simulate_to_end()`: use only legal moves to reach a terminal state; keep it
  bounded by the game's own finite-state rules.

Maintain the transition invariant:

```text
valid_state -> moves = legal_moves(state, current_player)
play_idx(i) -> play_str(old_moves[i]) -> new state and fresh moves
ended() -> moves == [] (unless a deliberate, tested exception exists)
```

Do not make consumers parse game-specific move strings. Strategies receive
`game.moves` and `game.display_with_moves()`; the action index is the position
in that list, not a move value, board coordinate, or hand position.

## Register and expose the game

Register conventional games declaratively in `boardrl/games/__init__.py`:

```python
from boardrl.games.mygame.game import MyGame
from boardrl.games.mygame.strategies import strategy_from_string as my_strategies

register_game(
    "mygame",
    MyGame,
    strategies=my_strategies,  # omit when the game has no custom strategies
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
    args_from=MyGame,  # only when constructor args belong in the textual spec
)
```

`register_game` lazily resolves the sibling `metrics.Metrics` class so package
initialization does not eagerly import the metrics stack. When a custom
strategy registry is supplied, the helper copies it and merges the default
`random`, `argmax`, and `policy_sampling` strategies for each descriptor.

Omit `args_from` when constructor arguments are implementation details. The
rollout engine may still pass values such as `num_players` directly to
`GameDesc.make_game`; that does not require exposing them in strings like
`tictactoe,num_players=2`.

Use a custom `GameDesc` factory only when the declarative helper cannot describe
the integration. Century remains the main example because the Cython class
cannot be introspected normally and needs custom import/setup logic.

Registry specifications are comma-separated strings such as
`nim,num_stones=5,max_pick=2`; commas and equals signs in model paths need the
existing URL-encoding conventions.

## Integrate metrics, strategies, and augmentations

Implement `Metrics(GameMetrics)` with `__init__(data)`, `metrics()`, and an
optional `print_short_history()`. Return a small nested dictionary of numeric
values; sinks flatten nested names, and `Range` is available for compact
distribution summaries. Use game-specific metrics for rule health and
collapse, not as a replacement for the framework's generic rollout metrics.

Strategies must implement `async strategy(game)` and return
`(distribution, info)`, where the one-dimensional distribution length equals
`len(game.moves)`. Strategy factories belong in registry or lineup construction
code, not inside rollout execution. Use `RegisterByName` for game-specific
strategy constructors and test every registered strategy against the game's
initial state and a short legal rollout.

Register `shuffle_actions` when action-order augmentation is valid. It
permutes `@` lines and must apply the same permutation to `moves`,
`action_idx`, `action_distribution`, and `reference_policy`. Add a custom
augmentation only when its observation transformation preserves the action
semantics and all action-indexed metadata. Test hidden-information and
multi-phase states separately.

## Validate proportionally

Run the smallest useful checks first, then broaden:

```bash
uv run pytest tests/test_games.py
uv run pytest tests/test_strategies.py tests/test_rollouts.py tests/test_action_augmentation.py
uv run pytest tests/test_serve.py tests/test_serve_tictactoe.py
node --check boardrl/serve/static/app.js
```

Add the actual game-specific test file when one exists; not every game has a
dedicated test module. For Cython changes, include the focused Century/Cython
test and compiler-backed import. For model or training boundary changes,
instantiate the default descriptor and run direct caller tests rather than
relying only on unit tests for the game class.

Before declaring success, manually or with a test verify: initial display and
move count; every move in an initial short rollout; illegal moves; copy
independence; terminal state and score; truncation through `max_steps`; all
registered strategies; descriptor metadata; and `/state` plus `/do-one` for
the server. Report incomplete validation explicitly.

## Avoid common integration failures

- Never leave `moves` stale after a phase change, draw, pass, or terminal move.
- Never reorder `moves` without applying the same order to every action-indexed
  tensor and recorded action index.
- Never expose hidden state from `display()` unless the game mode explicitly
  provides full information; use `force` only for the requested viewer.
- Never use `points_for()` as a proxy for a cooperative objective without
  checking `coop`, `won`/terminal semantics, and evaluation behavior.
- Never add a per-game web page when the shared server renderer is sufficient;
  add a renderer in `boardrl/serve/static/app.js` only for real visual needs.
- Never report a change as complete from a passing constructor smoke test: the
  rollout path calls `display_with_moves`, samples by index, copies state in
  some strategies, and records end-state scores.
