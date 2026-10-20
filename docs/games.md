# Game catalog

See [native rules and action formats](game-rules.md) for turn rules, visibility,
scoring quirks, and move-string conventions for each native game.

The source of truth is `boardrl/games/__init__.py`. A game spec is a registered
name followed by comma-separated constructor options, for example
`nim,num_stones=7,max_pick=2`. Boolean, numeric, and optional values are parsed
from the constructor's annotations; unknown arguments are rejected.

```python
from boardrl.games import games_library

games_library.display()                 # public options and defaults
desc = games_library("nim,num_stones=7,max_pick=2")
game = desc.make_game()
print(game.display_with_moves())
desc.strategy_from_string.display()     # available strategies and arguments
```

The following table describes the implemented variants, not every rule or
expansion of the physical games. Legal action strings in `game.moves` are
authoritative. Use [the workbench](web-server.md) to inspect the display, phases,
and legal moves interactively.

## Native competitive games

| Spec / public options | Implemented task and information | Objective and training reward | Extra strategies |
| --- | --- | --- | --- |
| `tictactoe` | Two players, 3×3, alternate marking empty cells | Three in a row; terminal outcome | — |
| `connectfour` | Two players, drop a piece into a non-full column | Four connected pieces; terminal outcome | `tactical_random` |
| `2048versus,num_players=2,seed=-1,nolimits=False` | Player 0 slides; player 1 places each randomly valued new tile on an empty cell | Player 0 reaches 2048; player 1 blocks all slides; signed merge-point differences ÷ 2048 | — |
| `hnefatafl,num_players=2,max_plies=500` | Fully visible 11×11 Fetlar (2018 panel rules); 24 attackers against 12 defenders and their king | Attackers capture the king; defenders escape to a corner; terminal outcome | — |
| `nim,num_players=2,num_stones=21,max_pick=3` | Remove 1 through `max_pick` stones from a single pile | Taking the last stone wins; terminal outcome | `optimal` |
| `rps,num_players=2,v_rock=1,v_paper=1,v_scissors=1` | One hidden choice per player, then reveal | Weighted rock/paper/scissors; terminal win/loss/draw sign | — |
| `sum` | Two players choose numbers in repeated rounds | Larger final points; terminal outcome | `sum_exact` |
| `century,goal_cards=-1,num_players=2` | Cython engine, resource conversion, acquiring action and point cards | Larger final points; terminal outcome | `tempo_greedy`, `random_buy`, `all_actions_then_random_buy`, `no_actions_random_buy`, `never_buy` |
| `take5,num_players=4,num_stacks=4,num_cards=104` | Hidden card selection followed by ordered resolution and occasional stack choice | Avoid penalty points; terminal outcome | — |
| `skullking,num_players=4,num_rounds=10` | Hidden hands, bidding, and trick-taking, including special cards | Accumulate bid/trick scores; terminal outcome | — |
| `santorini` | Two players place workers, then move/build on a 5×5 board; no god powers | Climb to level 3 or block the opponent; terminal outcome | — |
| `splendor,num_players=2` | Tokens, development cards, reservations, nobles; private reserved cards | Prestige with fewest purchased cards as tiebreak; terminal outcome | — |
| `traversal,num_players=2,terrain=random,seed=-1` | Six pieces per side race and fight on a fully visible 10×10 height map; 15 sampled website presets with rotations/reflections, plus `flat` | Score finished pieces, or finish a Sentinel first; terminal outcome | — |

`OutcomeScores` is used for Tic-Tac-Toe, Connect Four, Hnefatafl, Nim, RPS, and Santorini;
the others report `PointScores`. Reporting point totals does not imply point
rewards: 2048versus uses signed point-delta rewards and a continuous point
critic; the other competitive native registrations use terminal outcome
rewards and a categorical outcome critic. Splendor, Traversal, and 2048versus
have custom outcomes to respect victory conditions independently of raw points.

`tictactoe`, `connectfour`, `sum`, and `santorini` expose no textual constructor
options; their Python constructors still accept the player count supplied by
rollouts. Some games support more than two seats, but the standard trainers use
two-seat lineups. Check the constructor's validation before selecting another
count; custom algorithms must create the corresponding lineup.

## Single-player games

| Spec / public options | Task and visibility | Points / objective success | Training reward |
| --- | --- | --- | --- |
| `2048,num_players=1,seed=-1,nolimits=False` | Fully visible 4×4 grid; slide up, right, down, or left and merge equal tiles | Sum of merged tile values / create a 2048 tile | Difference in points ÷ 2048 |

2048 ends on the first 2048 tile or when no slide changes the board. Its
descriptor uses cooperative objective semantics for solo success. Use a
one-seat lineup such as `runner.play([bot], games=32, max_steps=10000)`;
the shipped two-seat trainers need a solo experiment loop for this game.
Nonnegative seeds reproduce initial tiles and subsequent spawns; `-1` uses
fresh randomness. The browser workbench supports solo play directly.
For the two-player placement variant, use `2048versus` with a two-seat lineup.
Both support `nolimits=True` (for example, `2048versus,nolimits=True`) to
continue beyond 2048 until no slide remains. Reaching 2048 stays the success
milestone; scores keep accumulating, with opposite signs for the two players.

## Cooperative games

| Spec / public options | Task and visibility | Points / objective success | Training reward |
| --- | --- | --- | --- |
| `thegame,num_players=2,max_value=100,mode=strict` | Place cards onto two ascending and two descending piles; own-hand information except Omni | 2 + cards played / play every card | Difference in points × 0.1 |
| `guessnumber,max_number=10,num_symbols=2,secret=None,num_players=2` | Player 0 sees a secret and sends symbols; player 1 guesses | Negative guess count / guess correctly | Terminal success/failure |
| `regicide,num_players=2` | Cooperative attacks and defense against a royal enemy deck; private hands | Progress score / defeat every enemy | Difference in points × 0.01 |
| `hanabi,num_players=2,mode=full` | Play fireworks using hints, discards, and others' visible hands; own cards hidden | Completed fireworks / achieve the mode's maximum score | Difference in points × 0.1 |

Hanabi supports `mode=full` and `mode=mini`; mini uses two colors, two-card hands,
three information tokens, and one life. The Game has seven modes described in
its [dedicated guide](thegame.md). Regicide supports one through four players;
The Game supports one through five. These counts apply to the game classes;
the shipped trainers still supply two players.

Cooperative descriptors define objective success through `won()`. A partial
score can improve without any complete wins. Points and outcome semantics
are separate from rewards and critic type; see [reward semantics](internals/returns.md).

## Strategies shared by games

Every conventional registration merges the common strategy registry with any
game-specific entries. `random` returns uniform logits. `longest_move` chooses
the longest action string and is primarily a diagnostic. `argmax` and
`policy_sampling` require a model callable; `gumbel` provides a model-backed
search strategy. See the constructors in
[`strategies.py`](https://github.com/Vermeille/century-RL/blob/master/boardrl/games/strategies.py)
for their parameters. The UI only lists strategies it can instantiate without
missing required arguments, plus discovered model files.

Strategy strings and game specs share comma-separated parsing. URL-encode
commas and equals signs within argument values, especially checkpoint paths;
the server does this for discovered files automatically. In your own Python
algorithm it is often simpler to construct strategy objects directly.

## Optional OpenSpiel games

Installing `open-spiel` adds `leduc`, `liars_dice`, `dark_hex`,
`abrupt_dark_hex`, `phantom_ttt`, `abrupt_phantom_ttt`, and `othello` when
`pyspiel` is importable. These use the same model, rollout, trainer, and server
interfaces. Chance events are resolved inside the adapter; hidden information
is presented through the acting player's information state.

The [benchmark guide](benchmarking.md) documents their fixed rules, optional
dependencies, native payoff units, exact evaluation, and limitations when
comparing against papers.

## Inspect or extend a game

Use `display(force=seat)` to inspect a particular viewer's observation, and
`display_with_moves()` for exactly the input used by model strategies. Public
state displays should reveal only what that viewer is allowed to know. Forced
views are also used for terminal and truncated endpoints.

Start from [Add a game](adding-games.md) for a complete small example, registration,
metrics, tests, augmentation, and web integration. The [development source map](development.md)
points to implementations for more involved card and phase mechanics.
