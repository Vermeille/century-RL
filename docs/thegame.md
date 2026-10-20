# The Game

`boardrl/games/thegame/game.py` implements a cooperative card-placement game.
The default deck contains cards 2 through 99, with piles initially
`[1, 1, 100, 100]`. The first two piles ascend and the last two descend. You
may reverse a pile's usual direction by exactly ten (for example, 40 on an
ascending 50). A legal card action is written `card->pile`, with pile indices
0 through 3.

Players draw to seven cards (six with four or five players). A turn requires
two card placements while the draw pile is nonempty, then one after it is
exhausted. Ending a turn refills the hand and advances the active player.
The implementation handles players with empty hands as the game runs out.
The team loses when the active turn cannot satisfy the required placements;
it wins when all cards have been played.

## Choose the information and turn rules

| Mode | End of turn after required placements | Observation |
| --- | --- | --- |
| `strict` | Automatic | Own hand and public state |
| `free` | More cards or `x` to finish | Own hand and public state |
| `omni` | Free-play rules | All hands and the next ten draw cards |
| `strict_message_before_draw` | Choose `A` through `J`, then refill and pass | Own hand, public state, and messages |
| `free_message_before_draw` | More cards or a message to finish | Own hand, public state, and messages |
| `strict_message_after_draw` | Refill, choose a message, then pass | Own hand including refill, public state, and messages |
| `free_message_after_draw` | More cards or `x`, refill, then message | Own hand including refill, public state, and messages |

The symbols are learned communication actions, not predefined meanings. They
and `x` consume rollout actions but do not place cards. Compare modes using
the full spec and allow enough environment actions for those extra phases.

The registration defaults to `thegame,mode=strict`; `coop.py` currently defaults
to `thegame,mode=omni`. For a controlled experiment, specify the mode explicitly:

```bash
uv run python trainers/coop.py --game thegame,mode=strict --architecture cnn
uv run python -m boardrl.serve.serve --game thegame,mode=free
```

## Scores, rewards, and baselines

All players share the point score `max_value - remaining cards`, which is
**2 + cards played** (initial score 2; perfect score 100 with the default deck). The
descriptor uses `PointDeltaRewards(scale=0.1)`, so at gamma 1 the point return
tracks the future card-placement count scaled by 0.1. The critic models this
with a Normal distribution. Objective win rate is the fraction of fully cleared
games, which is different from mean cards played.

`lowest_cost` is the built-in heuristic; it favors cheap pile changes and can
use the reverse-ten rule. It is available for imitation, evaluation in Python,
and the web workbench:

```python
from boardrl import Evaluator
from boardrl.games import games_library

desc = games_library("thegame,mode=strict")
bot = desc.strategy_from_string("lowest_cost")
result = Evaluator(desc.make_game, outcome=desc.outcome).compare(
    [bot, bot], games=32, max_steps=800,
)
print(result.avg_points(), result.win_rate())
```

`max_value` changes the deck range and pile endpoints, and `num_players` changes
the seat count. Choose a deck large enough to deal all initial hands. The supplied
trainers use two seats even when a spec supplies another count; use a custom
lineup for larger teams.

## Implementation notes

The state has separate game-mode objects for legal actions and phase changes.
`display()` hides other hands except in Omni. `copy(randomize=True)` can shuffle
unknown draw order; Omni preserves the visible next ten cards. Hand and action
augmentations preserve action-target alignment, which is essential because
policies select positions in the listed legal actions.

Tests under `tests/test_thegame_*.py` cover turn endings, messages, visibility,
copying, and metrics. [Strict-mode experiment notes](thegame-strict-rl.md) retain
historical measurements and longer recipes; they are not a promise that a
different revision, optimizer, or mode reproduces those scores.
