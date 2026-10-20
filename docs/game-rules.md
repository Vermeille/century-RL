# Native game rules and action formats

These notes describe the variants implemented here. Choose strings from the
current `game.moves`; the examples below explain formats and may be illegal in
a particular state. Constructor options and reward types are in the
[game catalog](games.md). The Game has a [mode guide](thegame.md), and
[OpenSpiel benchmarks](benchmarking.md) describes external presets.

## Tic-Tac-Toe and Connect Four

Tic-Tac-Toe uses cells `0` through `8` in row-major order on a 3×3 board:
`0` is top left and `4` is center. Players mark empty cells; three in a row
wins, and a full board otherwise draws. The observation prefixes the viewer's
symbol (`O` or `X`). Terminal points are +1/-1 for winner/loser and 0 for draws.

Connect Four uses columns `0` through `6`, with pieces falling to the lowest
empty location on a 7×6 board. Full columns disappear from legal moves.
Four in a row horizontally, vertically, or diagonally wins; a full board draws.
It uses the same +1/0/-1 outcome scores. Both games have complete information
and clickable browser boards.

## Nim

`nim,num_stones=21,max_pick=3` is single-pile normal-play Nim. Action strings
are counts to remove, bounded by `max_pick` and the remaining pile. The player
taking the last stone wins; terminal points are +1/-1. `optimal` uses the pile
modulo `max_pick + 1`. Small piles are useful for debugging training algorithms.

## Rock Paper Scissors

`rock`, `paper`, and `scissors` are collected sequentially but hidden from the
opponent until both choices have been made. Matching choices draw. The winning
choice's `v_rock`, `v_paper`, or `v_scissors` supplies the magnitude of raw points;
registered terminal rewards use the outcome sign. The second decision must not
observe the first choice.

## Sum

This toy task displays `a` and `b` from 0 through 9 and asks for
`floor((a+b)/2)`, chosen among strings `0` through `9`. A correct answer adds
one to the active player's counter and generates another question for that
player. An incorrect answer passes control. The game ends when a counter reaches
three. Despite the name, the target is the floored average, not `a+b`.

Both `points_for` and `diff_points_for` expose the counter difference from the
opponent. The `sum_exact` bot provides a behavior check. Repeated wrong answers
can require an action horizon even though the winning score is small.

## Century

The Cython engine models resource conversion and acquiring action and victory
cards. It enumerates legal resource exchanges; use the complete move strings.

| Format | Meaning |
| --- | --- |
| `R` | Recover discarded action cards |
| `V<i>` | Buy victory card at visible position `i` |
| `A<i> <payment>><gain>` | Acquire an action card, paying for skipped cards and collecting resources on it |
| `H<i> <instruction>` | Play hand action `i` with an enumerated conversion |

The game finishes at a round boundary after a player reaches the victory-card
goal. `goal_cards=-1` selects six for up to three players, five otherwise.
`points_for` currently reports victory-card points, excluding the resource-stock
bonus on the internal player object. Do not assume every physical-game scoring
rule is included. For two seats, `diff_points_for` is my victory points minus
the opponent's. At larger tables the leader is compared with second place,
and others with the leader. `tempo_greedy` is a stronger baseline than uniform
play. Observations include markets and viewer-relative player state.

## Take5

Players receive ten cards, with one starter per stack. During selection,
actions are cards in the active hand. After everyone chooses, cards resolve in
ascending order onto the stack with the greatest top below the card. Adding a
sixth card takes the existing five as penalties and starts a new stack. A card
smaller than every top requires a separate `S<i>` stack-choice action.

Points start at 0 and decrease as penalties are taken; higher is better. The
implemented penalty function adds 2 for multiples of 5, 3 for multiples of 10,
and 5 for multiples of 11; ordinary cards have zero penalty. This differs from
usual printed card scoring. The episode ends after all hands are empty.
`diff_points_for` is the player's own negative penalty score, not a subtraction
from an opponent. Hands are private; sequential selection does not make hidden
choices observable.

## Skull King

Rounds have bidding followed by trick play. Bids are integer strings from 0
through the number of cards dealt. Card actions use `suit:rank`, with suits
`r`, `y`, `b`, black trump `B`, and numbered ranks 1 through 14. Specials use
`_:`: escape `E`, pirate `P`, mermaid `M`, Skull King `K`, and Tigress `T`,
whose play chooses `_:T=P` or `_:T=E`.

Numbered cards follow the lead suit when possible; specials remain playable.
Black trumps ordinary suits. Mermaids beat the Skull King, the Skull King beats
pirates, and pirates beat mermaids. `_winning_play` resolves combinations
explicitly; consult it when changing special rules.

A correct nonzero bid earns 20 per trick. A correct zero bid earns 10 times
cards dealt; a failed zero bid loses that amount. Other misses lose 10 times
the bid/trick difference. Bonuses for captured 14s and special captures apply
on successful bids. Cumulative scores determine the winner. Hands are private;
bids and trick history provide public context.

## Santorini

The first four actions place two workers per player on a 5×5 board. Coordinates
run `a1` through `e5`, and placement is `P:a1`. An ordinary action combines
movement and building, for example `a1>b2+c3`.

Move to an adjacent unoccupied cell, climb at most one level, and avoid domes
(level 4). Then build on an adjacent free non-dome cell, possibly the cell just
vacated. Climbing from level 2 to 3 wins immediately and omits the build suffix.
A player with no legal move loses. There are no god powers. The board is fully
observable, terminal points are +1/-1, and symmetry augmentations transform
board coordinates and actions together.

## Splendor

Two through four players take tokens, reserve cards, or buy developments.
Reservations are capped at three; gold substitutes for colored payment.
Exceeding ten tokens causes a discard phase. Eligible nobles are awarded after
the main action and discard; multiple eligible nobles cause a choice phase.

| Format | Meaning |
| --- | --- |
| `T:<colors>` | Take distinct colors (up to three when available) or two of one color with at least four in the bank |
| `R:<tier>.<slot>` / `R:<tier>.D` | Reserve a market card / hidden top card |
| `B:<tier>.<slot>` / `B:H<slot>` | Buy a market / own reserved card |
| `~<colors>` buy suffix | Colored costs paid using gold |
| `D:<colors>` | Return excess tokens |
| `N:<id>` | Choose an eligible noble |

Reaching 15 prestige triggers completion of the round. Most prestige wins,
with fewest purchased developments breaking ties. A custom outcome preserves
that rule in training; `diff_points_for` encodes a unique prestige-tied winner
as +0.01 and loser as -0.01 at terminal. No legal main action causes a stalemate
draw. Own reservations are visible; opponents' reserved identities stay hidden.

## Traversal

[Official rules](https://www.traversalgame.com/rules). Each side starts with
two Explorers (movement 5, strength 1), two Vanguards (4, 2), and two Sentinels
(3, 3). They occupy columns c through h in S-V-E-E-V-S order. Player 0 starts
on row 10 and exits above row 1; player 1 starts on row 1 and exits below row
10. Actions such as `e10>e5` move or capture; `e2>off` finishes a piece.
Equivalent exit columns are represented by a single `>off` action.

Normal movement follows orthogonal paths, may turn, and costs
`1 + abs(height difference)` per step. Pieces block paths. An Explorer can
instead step diagonally or leap two orthogonal squares, ignoring terrain;
these specials cannot be chained with normal movement. Leaps cannot cross
pieces. Specials may attack or cross the goal edge, matching the site's local
engine. A normal finish needs one movement beyond reaching the final row.

An attack includes the cost of entering the enemy square and ends the turn.
Attack strength is base strength, plus or minus one for starting above or
below the target, plus the base strengths of other allies orthogonally
adjacent to the target. Defense is base strength plus one per adjacent ally.
The attacker captures and occupies the target on a tie or better; weaker
attacks are illegal. A Vanguard adds one attack strength only when the
cheapest path to its target costs exactly four movement.

A finished piece scores its strength. The second finish ends the game with
the higher score winning, or a draw on equal scores. A Sentinel that is the
first piece to finish wins immediately. At 200 actions, compare scores, then
each side's most advanced piece, then strength among equally advanced pieces.
The site's additional local-engine rules are also implemented: elimination
ends play and credits each side's strongest surviving piece to its score;
three identical consecutive position cycles cause a draw. The published
rules leave immobilization unspecified; this implementation treats a player
with no legal action as a stalemate draw.

Terrain is sampled offline from 20 boards per named preset collected from the
[official board API](https://api.traversalgame.com/api/boards/generate).
Choose `terrain=mountains`, `valleys`, `highlands`, `woodlands`, `desert`,
`dunes`, `jungle`, `meadows`, `canyons`, `rivers`, `ridges`, `petrified`,
`everest`, `mariana`, or `hell`. `terrain=random` chooses uniformly among
these 15 presets; the website's preset probabilities are unknown. Each game
samples a board and one of eight orientations (four rotations, optionally
reflected). These transformations affect terrain before standard pieces are
placed; the players still travel north/south. Boards are not forced to be
symmetric. Adjacent height differences stay at most two.

`seed=-1` uses fresh randomness; a fixed seed reproduces the preset, sample,
and orientation. `terrain=flat` remains a zero-height board for rule
experiments. Flat/symmetric presets have fewer distinct boards/orientations,
so 20 samples does not imply 160 unique maps for every preset. Runtime play
requires no network. Collect or resume the checked-in pool with:

```bash
uv run python -m boardrl.games.traversal.collect_terrains --per-type 20
```

The collector records its source and collection time, validates boards, and
resumes existing files. Use `--output <new-path>` to collect a fresh pool.
The observation keeps the move counter, viewer, terrain, both armies, and
finished pieces. The counter is necessary for the 200-action limit; its parity
identifies the side to move. The viewer identifies the value perspective in
forced endpoint observations. During play, scores are the sum of finished-piece
strengths, and captures follow from the initial material minus the board and
finished pieces, so neither needs a separate field. Terminal observations add
actual scores (including elimination bonuses), the end reason, and the winner.

`Repetition:` contains reversible moves since the last capture or finish.
Earlier positions have different material and cannot recur. The current board
and this ordered suffix reconstruct all repetition-relevant positions; a
position-count summary alone would lose the rule's ordered-cycle information.
Before the first material change, the starting position is not counted by the
site's repetition rule; afterward, the position immediately after the change
is counted. The move counter and suffix length distinguish these cases.
Terminal states need no repetition context. Full move history remains in the
game object and browser session for auditing and undo/redo.

`points_for` preserves actual scores. `diff_points_for` is the score difference
during play; terminal draws return zero, and a score-tied unique winner gets
a difference of +0.01 (the loser -0.01), following Splendor's convention.
A custom outcome policy supplies the actual win/loss/draw training reward.
The shared browser workbench renders the height map, piece shapes, scores,
captured and finished pieces, and the result. Select a piece to reveal legal
destinations, captures, and its finishing edge; Esc clears the selection.
The raw text board and canonical move list remain available.

Traversal metrics include global `end_reason` shares, terrain preset shares,
the fraction of distinct maps in the batch (`terrain.unique_map_share`),
mean map height range, and mean absolute height difference over orthogonal
edges. The observation records the selected preset even for `terrain=random`.

Each `strategy.<id>` group keeps win rate, action count, sensitivity, and
point-difference summaries, and adds mean raw endpoint points (`avg_points`)
and victory causes among that strategy's wins (`win_cause`). Strategy identity
stays fixed when seats rotate. Truncated endpoints contribute to `avg_points`
and terrain summaries. `draw_rate` counts terminal draws, and `win_cause`
considers terminal wins; the inherited generic win-rate convention is unchanged.

The `behavior` metrics describe recorded decisions:

| Metric | Meaning / denominator |
| --- | --- |
| `piece_share` | Explorer, Vanguard, and Sentinel shares of decisions |
| `capture_share`, `finish_share` | Captures or finishes per decision |
| `direction_share` | Forward, backward, and lateral shares relative to the moving player's goal |
| `avg_forward_rows` | Mean signed row displacement toward the goal; finishing includes the extra step off the board |
| `opportunity.*_available_rate` | Fraction of decisions with a legal capture, finish, or immediate first-Sentinel finish |
| `opportunity.*_conversion_rate` | Fraction of those opportunities where the corresponding action was chosen |
| `combat.*_capture_share` | Among captures, how often the attacker started higher/lower or had allies beside the target |
| `combat.reinforced_target_share` | Among captures, how often the defender had an adjacent ally |
| `finish_type_share` | Piece type shares among recorded finishes |

Behavior counts decisions uniformly and uses the actual ordered action list.
Net row displacement is not path length or movement spent. Outcomes use the
game's win/loss/draw utility, including advancement tiebreaks and repetition
draws. Zero denominators report zero. Unparseable observations are excluded
from behavior/terrain summaries, with `observation_coverage` reporting the
fraction successfully parsed; older observations without a preset label use
`terrain.preset_share.unknown`.

## Hanabi

Players see others' cards and only knowledge constraints for their own cards.
Moves use hand positions: `p <i>` plays, `d <i>` discards, and
`h p<offset> c<color>` or `h p<offset> r<rank>` hints at a relative seat offset.
Hints must match at least one target card and also exclude that color/rank from
other cards. Discarding is unavailable at full information tokens.

Build each firework in rank order. Successful maximum-rank plays restore one
information token; failed plays consume a life. Discards restore information
and draw replacements. The game ends on exhausted lives, maximum score, or the
final turns after the deck empties. Points sum completed ranks; objective
success requires the mode's maximum. Full mode has five colors; mini has two
colors, two-card hands, three information tokens, and one life.

## Regicide

The team faces Jacks, Queens, and Kings with 20/30/40 health and 10/15/20 attack.
Attack moves list codes such as `5H` or combinations joined by `+`. Aces can
pair with another card; matching ranks can combine within the legal total-value
cap. Suit powers heal discarded cards into the deck (hearts), draw (diamonds),
build defense (spades), or double damage (clubs), subject to enemy immunity.

If the enemy survives, the active player discards enough card value to cover
incoming damage in a defense phase. Defeating every enemy wins; inability to
make required progress loses. `pass`, multiplayer `joker:P<seat>`, and solo
`solo-joker` appear only in their legal phases. Normal observations keep private
hands and deck order hidden.

Points are defeated-enemy health plus damage to the current enemy capped at
its health. They measure progress even in losing games, scaled by 0.01 for
training rewards. The objective remains clearing the entire enemy deck.
