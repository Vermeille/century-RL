# Add a game

A game is a mutable state machine that satisfies `boardrl.utils.Game`. It does
not need a Gym wrapper, tensor observation, inheritance hierarchy, or fixed
action space. The model learns from text with legal actions appended.

This tutorial implements Countdown: two players remove one or two counters;
taking the last counter wins. It then connects that game to rollouts, training,
metrics, and the browser. For larger games, use the
[source map](development.md) to find a close implementation.

## Implement state, moves, and observations

Create `boardrl/games/countdown/game.py`:

```python
# docs-test: countdown
import random


class Countdown:
    def __init__(self, num_players: int = 2, counters: int = 7):
        if num_players != 2 or counters < 1:
            raise ValueError("Countdown needs two players and positive counters")
        self.num_players = num_players
        self.counters = counters
        self.turn = 0
        self._winner = None
        self._refresh_moves()

    def _refresh_moves(self):
        self.moves = [str(n) for n in range(1, min(2, self.counters) + 1)]

    def current_player(self):
        return self.turn % self.num_players

    def round(self):
        return self.turn // self.num_players

    def ended(self):
        return self._winner is not None

    def display(self, force=-1):
        viewer = self.current_player() if force == -1 else force
        return (f"Viewer: {viewer}\nPlayer: {self.current_player()}\n"
                f"Counters: {self.counters}\nWinner: {self._winner}")

    def display_with_moves(self):
        return self.display() + "\n" + "\n".join(f"@{m}" for m in self.moves)

    def play_str(self, move):
        if self.ended() or move not in self.moves:
            raise ValueError(f"Illegal move: {move}")
        self.counters -= int(move)
        if self.counters == 0:
            self._winner = self.current_player()
        self.turn += 1
        self._refresh_moves()

    def play_idx(self, index):
        self.play_str(self.moves[index])

    def points_for(self, player):
        if not self.ended():
            return 0
        return 1 if player == self._winner else -1

    def points(self):
        return self.points_for(self.current_player())

    def diff_points_for(self, player):
        return self.points_for(player)

    def diff_points(self):
        return self.diff_points_for(self.current_player())

    def copy(self):
        other = Countdown.__new__(Countdown)
        other.num_players = self.num_players
        other.counters = self.counters
        other.turn = self.turn
        other._winner = self._winner
        other.moves = self.moves.copy()
        return other

    def simulate_to_end(self):
        while not self.ended():
            self.play_str(random.choice(self.moves))
```

The transition invariant is:

```text
moves = ordered legal actions for the current state and player
play_idx(i) = play_str(the old moves[i])
every transition -> refresh moves, including phase changes and passes
terminal state -> no playable moves
```

Some older games retain a move list at terminal states; rollout and server
callers check `ended()` first. For a new game, empty terminal moves are the
cleaner contract.

`display(force=player)` must return that player's observation, including at
truncated and terminal endpoints. Hidden information must stay hidden from
other seats. Avoid `@` anywhere except the beginning of legal-action lines:
the current model locates action positions by every `@` character. Use stable
ASCII serialization; character IDs index a 128-entry embedding.

Copy all mutable state independently. A randomized copy is optional: search
strategies can call `copy(randomize=True)` when that argument is supported.
Randomization must preserve the viewer's information and the legal state.

## Add metrics

Create `boardrl/games/countdown/metrics.py`:

```python
# docs-test: countdown-metrics
from boardrl.metrics import GameMetrics, GroupedTraceMetrics
from boardrl.games.semantics import OutcomeScores


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def metrics(self):
        return GroupedTraceMetrics(
            self.data.by_strategy, scores=OutcomeScores(),
        ).metrics()

    def print_short_history(self):
        for trace in self.data[0]:
            print([getattr(record, "action_idx", None) for record in trace[:-1]])
```

Return nested numeric mappings. `Range(values)` can represent a distribution
whose mean is displayed by the existing sinks. Add rule-specific diagnostics
such as passes, illegal-state counts, or resource use when they help interpret
learning. Group by strategy when seats rotate.

## Register semantics and constructor options

Add this registration to `boardrl/games/__init__.py` after its imports:

```python
# docs-test: countdown-registration
from boardrl.games.countdown.game import Countdown

register_game(
    "countdown", Countdown,
    scores=OutcomeScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
    augmentations=(shuffle_actions,),
    args_from=Countdown,
)
```

The helper resolves the sibling `metrics.Metrics` lazily, merges common
strategies, and exposes annotated constructor parameters through
`countdown,counters=11`. Omit `args_from` when constructor options should stay
internal. Use a custom `GameDesc` only for exceptional integrations (Century's
Cython constructor is the existing example).

Choose three independent policies deliberately:

| Component | Use when |
| --- | --- |
| `OutcomeScores()` / `PointScores()` | Report outcome-only / numeric point metrics |
| `CompetitiveOutcome()` | Win by scoring above every other seat; ties are neutral |
| `CooperativeOutcome()` | Everyone shares objective success; implement `game.won()` |
| `TerminalOutcomeRewards()` | Train on terminal -1/0/+1 utility; categorical critic |
| `PointDeltaRewards(scale=...)` | Train on changes in `diff_points_for`; Normal critic |

The score object does not select the critic; the reward object does. Make the
point-difference definition explicit for multiplayer and cooperative games.
See [returns](internals/returns.md) for how rewards span each player's own turns.

## Add a strategy or augmentation

A strategy is an async callable returning `(logits, info)`. It receives the
current game and must return one logit per current legal move. For example:

```python
import torch
from boardrl.utils import RegisterByName

strategy_from_string = RegisterByName()


@strategy_from_string.register("take_most")
class TakeMost:
    async def __call__(self, game):
        logits = torch.full((len(game.moves),), -torch.inf)
        logits[-1] = 0.0
        return logits, {}
```

Put this in `strategies.py`, import its registry in the central registration,
and pass `strategies=strategy_from_string` to `register_game`. Common strategies
are merged into a copy, so additions do not leak into other games.

`shuffle_actions` permutes action lines and matching targets. A board symmetry
must also rewrite action text, action index, behavior logits, and reference
logits consistently. Never transform observations without transforming their
action-indexed metadata. Test augmentation after `ReferenceTargets`, before
optimization, with variable action counts and hidden-information phases.

## Test the boundaries

Before training, add focused tests for deterministic rules, illegal/terminal
moves, forced views, copy independence, all phase transitions, and final scores.
For Countdown, both copies should retain their own move lists and changing one
should not alter the other's counters. Exercise every initial action and a full
terminal rollout, not just construction.

Then check the registered path:

```python
from boardrl import RolloutRunner
from boardrl.games import games_library

desc = games_library("countdown,counters=7")
bot = desc.strategy_from_string("random")
data = RolloutRunner(desc.make_game, outcome=desc.outcome, progress=False).play(
    [bot, bot], games=8, max_steps=10,
)
assert all(trace[-1].terminal for game in data for trace in game)
print(desc.make_metrics(data).metrics())
samples = desc.rewards.make_returns(1.0)(data)
```

Also run a deliberately short `max_steps=1` rollout to verify truncation and
bootstrap observation handling. Once registered, use the
[minimal trainer](training-library.md) with `--game countdown,counters=7`, then
evaluate and open it in the server.

```bash
uv run pytest tests/test_game_registration.py tests/test_games.py tests/test_strategies.py
uv run pytest tests/test_rollouts.py tests/test_action_augmentation.py
uv run pytest tests/test_serve.py tests/test_serve_tictactoe.py
```

Include your new game-specific test module in these checks.

## Expose it in the browser

The shared workbench discovers registered games automatically and has a raw
text fallback. No per-game HTML page is needed. First verify `/state` and
`/do-one` for the new game. If visuals are useful, add a renderer that reads
the state text and emits controls with canonical `data-action` values. The
[web server guide](web-server.md#add-a-renderer) describes the renderer contract
and tests.
