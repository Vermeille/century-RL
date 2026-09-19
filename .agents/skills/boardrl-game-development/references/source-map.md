# BoardRL game system source map

Read this reference when the task crosses a game boundary or when the core
protocol is ambiguous. Paths are relative to the repository root.

## Topology

- `boardrl/games/<name>/game.py`: mutable game state, legal moves, transition
  rules, displays, scores, copying, and random playout.
- `boardrl/games/<name>/metrics.py`: `GameMetrics` adapter over
  `SelfPlayResults`.
- `boardrl/games/<name>/strategies.py`: optional `RegisterByName` entries;
  merge them with `boardrl/games/strategies.py` defaults.
- `boardrl/games/<name>/augmentations.py`: optional transformations of
  `TrainingSample` objects.
- `boardrl/games/__init__.py`: `GameDesc`, `games_library`, lazy imports,
  composed semantics, and augmentation selection.
- `boardrl/games/semantics.py`: point reporting, competitive/cooperative
  outcomes, reward construction, value-head selection, and evaluation ranking.
- `boardrl/utils/__init__.py`: the `Game` protocol consumed by rollouts and
  strategies.
- `boardrl/rl/eval/selfplay.py`: `Record`, `EndState`, `GameTrace`,
  `SelfPlayResults`, `play_game`, and sampling by action index.
- `boardrl/rollouts.py`: `RolloutRunner` and batched `Inference` around an
  explicit player lineup.
- `boardrl/evaluation.py`: `Evaluator`, `Evaluation`, and optional
  head-to-head `Scoreboard`.
- `boardrl/metrics.py`: `GameMetrics`, generic `rollout_metrics`, console
  flattening, `Range`, and Trackio flattening.
- `boardrl/training/postprocess.py` and `boardrl/training/sample.py`:
  post-processing and action-indexed training data.
- `boardrl/serve/serve.py`: game sessions, registry lookup, state snapshots,
  strategies, undo/redo, and HTTP endpoints.
- `boardrl/serve/static/app.js`: shared UI and optional game-specific renderers.
- `boardrl/serve/README.md`: current server payload, query-string game specs,
  renderer contract, and renderer test commands.

## Protocol data flow

```text
GameDesc.make_game
  -> RolloutRunner / self_play2
  -> strategy(game.display_with_moves(), game.moves)
  -> distribution[action_idx]
  -> Record(game, ...)
  -> game.play_idx(action_idx)
  -> EndState(game, player)
  -> SelfPlayResults -> Evaluation / GameMetrics / Learner
```

`Record` snapshots `state`, `moves`, `action_distribution`, `action_idx`,
`current_player`, round, and point values before the move. `EndState` records
the final forced-player display, terminal versus truncated cause, points, and
round. Consequently, changing display, ordering, scores, or terminal behavior
changes training/evaluation meaning even if the game unit tests still pass.

## Existing implementation patterns

- `tictactoe/game.py`: minimal fixed board, string cell moves, winner/draw
  tests, and `shuffle_actions`.
- `connectfour/game.py`: legal moves depend on board capacity; use it for
  repeated-action legality and richer terminal detection.
- `sum/game.py`: compact round-based scoring and optional custom strategies.
- `nim/game.py`: constructor arguments exposed through `args_from` and
  game-specific strategies.
- `take5/game.py`: explicit multi-phase state (`phase`), table resolution,
  and legal move regeneration after each subphase.
- `thegame/game.py`: multiple modes, hidden versus full-information displays,
  cooperative descriptor, and separate hand/action augmentations.
- `century/engine.pyx`: Cython implementation; its descriptor repeats some
  constructor information because signature inspection is limited.

## Registration details

`RegisterByName` stores `(class, constructor-argument metadata)` and parses
comma-separated specs. `args_from` lets a registration inspect a separate
constructor when the registered descriptor is a wrapper. `GameDesc` stores:

```text
make_game, strategy_from_string, make_metrics, augmentations,
scores, outcome, rewards
```

`scores` controls point metrics, `outcome` owns terminal success semantics,
and `rewards` owns return processing, value-head selection, and evaluation
ranking. Custom
strategies should be merged into a copy because mutating the shared default
registry leaks game-specific entries to other games.

## Display and action rules

The model receives text, not a structured board. `PolicySamplingStrategy`
passes `display_with_moves()` to the model and returns a distribution aligned
with `game.moves`. `shuffle_actions` identifies only lines beginning with
`@`; an `@` in ordinary prose is not an action. Keep action lines and their
metadata aligned. Game-specific renderers may parse the state, but raw text
must remain a valid fallback.

## Cooperative and terminal semantics

`GameDesc.coop` marks games such as `thegame` and `guessnumber`, but inspect
the current rollout/evaluation implementation before changing semantics:
generic `SelfPlayResults.win_rate` is based on per-strategy terminal
`current_diff_points`, while `EndState` distinguishes proper terminal games
from `max_steps` truncation. If a change introduces or repairs game-level
objective success, trace and test the descriptor flag, end-state representation,
rollout aggregation, `Evaluation.win_rate`, and trainer logging together. A
truncated game is not a successful terminal outcome.

## Server contract

The shared web workbench uses the game registry and canonical `/state` payload:
`name`, `spec`, `board`, `board_with_moves`, `moves`, `current_player`,
`round`, `ended`, `points`, `history`, `can_undo`, and `can_redo`. State-bearing
endpoints accept `?game=<spec>`; specs can include constructor arguments.
Prefer `moves` for controls over scraping `@` lines. If adding a renderer,
preserve raw-state support, put `data-action="<move>"` on controls, register
the renderer map entry, run `node --check`, and add a static regression using
the real `display_with_moves()` format.

## Focused validation map

- Core games and rule transitions: `tests/test_games.py`,
  `tests/test_guessnumber.py`, `tests/test_nim.py`, `tests/test_rps.py`,
  `tests/test_take5.py`, `tests/test_skullking.py`.
- Copy and phase invariants: `tests/test_thegame_copy.py` and game tests.
- Metrics: `tests/test_thegame_metrics.py`, `tests/test_metrics.py`.
- Registry, descriptors, and strategy smoke: `tests/test_strategies.py`.
- Rollouts, trace indexing, truncation, and evaluation:
  `tests/test_selfplay_oop.py`.
- Action ordering and augmentation metadata:
  `tests/test_action_augmentation.py`.
- Web state and renderers: `tests/test_serve.py` and
  `tests/test_serve_tictactoe.py`.
