# External board-game benchmarks

BoardRL keeps benchmark dependencies outside the core training stack. The
training algorithms stay in `trainers/`; OpenSpiel is used as a game backend
and as an evaluator.

OpenSpiel code lives in `boardrl/games/openspiel/`. Each preset has its own
subdirectory: `leduc/`, `liars_dice/`, `dark_hex/`, `abrupt_dark_hex/`,
`phantom_ttt/`, `abrupt_phantom_ttt/`, and `othello/`. Each contains a concrete
`game.py` class derived from the abstract `OpenSpiel` adapter. Native game
specifications are inline in those constructors. The standard `register_game`
calls in `boardrl/games/__init__.py` select scores, rewards, augmentations, and
evaluation explicitly, with `metrics_from=OpenSpiel` sharing the payoff metrics.
Game-specific solvers and backend settings live in each game's `evaluation.py`.
The adapter's display methods handle native text, player identity and terminal
outcomes. Leduc and Dark Hex override the relevant methods in `game.py`.
Each concrete evaluator constructs its own game adapter and uses those same
display methods. Registrations select evaluator classes directly.

Shared adapter, policy, result, and evaluator code stays directly under
`boardrl/games/openspiel/`. `exp_a_spiel.py` provides the shared cached traversal
for the four hidden board games. The `register_game(exact_evaluation=...)` calls
select evaluator factories separately from game classes. `evaluation.py` exposes the external
`nash_conv` and reusable `ExactOpenSpielEvaluator` APIs.

## Install OpenSpiel

OpenSpiel is deliberately optional so benchmark tooling does not enlarge the
runtime dependency set for every BoardRL experiment:

```bash
uv pip install open-spiel
```

`open-spiel` is the distribution name; it installs the native `pyspiel` module
and the `open_spiel` Python package. CI installs that same optional distribution.

## Train benchmark games with the normal BoardRL trainers

The following regular game descriptors are backed by OpenSpiel. Chance
nodes are sampled inside the game adapter, so rollout code still sees only
player decisions. In imperfect-information games the model receives the acting
player's information state; in perfect-information games it receives the
observation/board state.

Leduc preserves native observation fields: `Observer` identifies the requested
viewer and `Player` identifies the actor. Neither field is rewritten at game end.
Private cards come from the native observation requested for that viewer.
Games without native player tags receive a `[Player: <seat>]` prefix on both
live and final views. Terminal observations append `Game: ended`,
`Result: win`, `loss`, or `draw` for that player, and the native `Payoff`.
The observation retains that player's private information and public history;
ending the game does not expose additional hidden cards or cells.
Truncated rollouts keep the normal observation and do not claim the game ended.

Exact evaluators construct the same observation headers as live games, including
the synthetic board and dice inputs. Existing checkpoints still load, but receive
different text than before this format change; new runs train on the new format.

For example, any adversarial trainer that accepts a game spec can use:

```text
leduc
othello
phantom_ttt
```

No training algorithm needs to be ported to OpenSpiel.

Training uses named registrations such as `leduc` or `phantom_ttt`; the generic
`openspiel,game=...` registration has been removed. Add another concrete game
and register its training semantics and optional exact backend explicitly. The
evaluation CLI uses these same registrations to preserve the training settings.

| Game spec | Configuration |
| --- | --- |
| `leduc` | Two-player Leduc poker |
| `liars_dice` | 2D5F: two players, two five-sided dice each |
| `dark_hex` | Classical Dark Hex, 3x3 |
| `abrupt_dark_hex` | Abrupt Dark Hex, 3x3 |
| `phantom_ttt` | Classical Phantom Tic-Tac-Toe, 3x3 |
| `abrupt_phantom_ttt` | Abrupt Phantom Tic-Tac-Toe, 3x3 |
| `othello` | Othello |

Liar's Dice uses OpenSpiel's `reset-face` bidding rule: increase the quantity,
or increase the face at the same quantity. The highest face (5) is wild.
Calling `Liar` settles the single round, with native utilities +1 and -1.

The Dark Hex and Phantom Tic-Tac-Toe presets use `obstype=reveal-nothing`
and OpenSpiel's information-state strings, retaining each player's attempted
move history and discovered cells. An unknown cell stays a legal attempt even
when the opponent secretly occupies it. A collision reveals that cell; the
classical variants let the player retry, while the abrupt variants consume
the turn. Hex connects opposite board edges; Tic-Tac-Toe wins with three in
a row and can draw.

These five presets use terminal win/loss/draw rewards and support external exact
exploitability evaluation. They share the standard strategies, action-shuffling augmentation,
payoff metrics, and raw text web workbench. For example:

```bash
uv run python trainers/adversarial-advshape.py --game abrupt_dark_hex
```

### Matching arXiv:2502.08938

The presets implement the games in
[Reevaluating Policy Gradient Methods for Imperfect-Information Games, v4](https://arxiv.org/html/2502.08938v4),
Section 4. The board-game parameters match the authors'
[experiment registry](https://github.com/nathanlct/IIG-RL-Benchmark/blob/591d35dadf243df846534c9df9f77759f7441814/main.py#L19).
Liar's Dice uses the paper's two dice per player and five faces, with OpenSpiel's
default `reset-face` bidding and highest-face wild rule. Validation uses
OpenSpiel 2.0.2.

The model still receives text. For Dark Hex, the adapter removes the global
attempt counter from OpenSpiel's native information-state string. That counter
reveals the opponent's failed attempts, despite `obstype=reveal-nothing`, and
is absent from the information-state tensor used in the paper. An explicit
player field retains the tensor's seat identity even before the first own
attempt. The private board and ordered own attempts remain intact. The evaluation
backend constructs exactly the same text and legal-action order as training. Earlier Dark Hex
checkpoints trained with the counter need a new run to match this setting.

All trainers use the existing `Evaluator` and ordinary opponent rollouts during
training. Exact exploitability is evaluated externally. For example:

```bash
./train_openspiel.sh --game leduc --opponent-eval-strategy random --evaluation-games 256
```

This uses ordinary rollouts with seat rotation and never constructs the exact
evaluator or enumerates information states. `--opponent-eval-strategy` accepts
the game's registered strategies. The resulting win rate (or points for Leduc)
also selects best checkpoints; it measures performance against that opponent,
not exploitability. Cooperative games retain sampled self-play; `imitate.py`
also defaults to self-play unless an opponent is supplied.

`train_openspiel.sh` accepts the game through `--game` and runs the checkpoint
arena for Elo ratings alongside training.
After training succeeds and the arena finishes its final ratings, it computes
exact exploitability once for the final `agent` checkpoint in both seats and
save `exploitability.json` in the checkpoint directory. This uses the game,
evaluation temperature, device, and inference batch size recorded in `run.txt`.
Games without a registered exact evaluator are skipped. Failed or interrupted
runs do not start final exploitability.

`ARENA=0` disables only the arena; `FINAL_EXPLOITABILITY=0` disables only the
final exact calculation.

Run the benchmark games sequentially with:

```bash
./train_openspiel_benchmarks.sh
```

This script calls the shared launcher once per game: Leduc first, then Phantom
Tic-Tac-Toe with its existing perplexity and evaluation schedule overrides.
Extra CLI arguments are forwarded to every run. A failed run stops the sequence.

To repeat the automatic final evaluation manually with the saved run settings:

```bash
uv run python evaluate_nashconv.py --training-run /path/to/checkpoint-directory
```

Exact exploitability remains available independently of the training evaluator:

```bash
uv run python evaluate_nashconv.py /path/to/checkpoint.pth \
  --game phantom_ttt --model agent --temperature 1
```

Run this on the final checkpoint after training, or on any saved checkpoint
manually. The same script accepts `leduc`, `liars_dice`, `dark_hex`,
`abrupt_dark_hex`, `phantom_ttt`, and `abrupt_phantom_ttt`, and selects their
exact backends automatically. For an adversarial checkpoint, choose
`--model agent` or `--model environment` to evaluate that policy in both seats. A separate
process builds its own information-set cache, so this still incurs the full
exact evaluation cost once per invocation.

Exact evaluation reports:

- `nash_conv`: the sum of the two players' unilateral best-response gains.
- `exploitability`: `nash_conv / 2`, matching Section 2.1 of the paper.
- `player_improvements`: those gains in responding-player order (0, 1).

The board backend uses the authors'
[exp-a-spiel](https://github.com/gabrfarina/exp-a-spiel/tree/9412882c24e521fad7cbd96fdd68311d9bf6f203)
sequence-form evaluator. Install its pinned optional dependency after making
a C++20 compiler and Boost headers (1.81 or newer) available; its build also
needs CMake 3.30 or newer:

```bash
uv pip install open-spiel==2.0.2
uv pip install 'eas @ git+https://github.com/gabrfarina/exp-a-spiel.git@9412882c24e521fad7cbd96fdd68311d9bf6f203'
```

The pinned library's `expl[p]` is the *other* player's improvement, so the
adapter reverses its entries before reporting `player_improvements`. Its sum
is NashConv; it must be divided by two exactly once. Neural policy probabilities
are inferred in bounded batches and illegal actions receive zero probability.

The authors' published revision contains the four board games but no dice
evaluator. The dice backend therefore performs exact dynamic programming over
the public bidding tree, integrating independent private-hand probabilities
before maximizing each responding player's actions. It retains all bid
histories and weights unordered dice combinations by their true multiplicity.
Tests compare it against exhaustive OpenSpiel best responses on smaller dice
games and an analytic policy on the full 2D5F game.

The full 2D5F analytic test was measured on 2026-10-13 with
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /usr/bin/time -v uv run pytest --capture=no
tests/test_paper_exploitability.py::test_full_2d5f_exact_result_against_analytic_policy -q`.
It passed in 18.8 seconds wall time, with 1,150,712 KiB peak RSS (1.10 GiB) and no
swapping on a 16 GiB host. The two solver arrays total 480 MiB. This check uses an
analytic policy; a neural model adds its own memory and inference time.

```bash
uv run python evaluate_nashconv.py checkpoints/step-100.pth \
  --model agent --game liars_dice --temperature 1 --batch-size 1024
```

The CLI accepts registered BoardRL games. For smaller native games and exhaustive
cross-checks, the Python API accepts a native `spiel_game` and an explicit backend
instance. Native games require that explicit backend; registered Dark Hex always
uses its paper-compatible exp-a-spiel backend.

Exact evaluation is more expensive than sampled matches: these games contain
millions of information states, all of which must be covered. The pinned board
backend also uses substantial RAM (approximately 7 GiB for classical games and
27–32 GiB for abrupt variants, including working arrays). Insufficient memory
or a missing backend produces an explicit error, never a sampled substitute.

Python callers can retain one `ExactOpenSpielEvaluator` across checkpoints:

```python
from boardrl.games import games_library
from boardrl.games.openspiel.evaluation import ExactOpenSpielEvaluator

descriptor = games_library("phantom_ttt")
native = descriptor.make_game(resolve_chance=False)._game
evaluator = ExactOpenSpielEvaluator(native, backend=descriptor.exact_evaluation(native))
result = evaluator.evaluate(model, temperature=1, batch_size=1024)
```

The first evaluation enumerates information sets; later calls reuse the traverser
and its working buffers. Each call recomputes probabilities and best responses
for the current model and temperature. Cache messages include the PID.
The cache belongs to that evaluator in that process. Running the CLI again builds
a new cache; the pinned exp-a-spiel library cannot save or restore its traverser.

The full reference checks are marked `slow`:

```bash
OMP_NUM_THREADS=4 uv run pytest --capture=no tests/test_paper_exploitability.py -m slow
```

This matches game rules, information available to policies, and the evaluation
metric. Trainer architectures, schedules, and optimization hyperparameters
remain experiment choices; the presets do not reproduce the paper's training
protocol automatically.

Leduc uses point-delta rewards and a points value head. The framework records
player-minus-opponent points, which are twice the native payoff in two-player
zero-sum poker. Its current `PointDeltaRewards(scale=0.1)` therefore produces
training returns equal to 0.2 times the chip payoff at discount 1. Exact
exploitability still uses unscaled native chip payoffs. `leduc` explicitly
requests information states with complete betting histories.

OpenSpiel game metrics report `strategy/<id>/total_payoff` and `avg_payoff`
in native utility units (chips for Leduc), grouped by strategy across seat
rotations. `payoff_games` counts completed hands; truncated games are excluded
from these totals and averages. Generic `points` metrics retain the framework's
player-minus-opponent convention.

Existing Leduc outcome-based checkpoints have a different value-head shape;
start a new run with the points value head.

## Leduc: exact NashConv / exploitability

Training evaluates against the configured opponent using `--evaluation-games`,
`--evaluation-every`, and `--eval-temperature`. Leduc best checkpoints use points.
Resuming an older checkpoint ranked by exploitability resets the best sampled
score, so incompatible metrics are never compared.

External exploitability evaluates the chosen model in both seats, in native chips
per hand. Lower values are better. Evaluation temperature changes the policy being
measured; temperature 1 is the default for the manual command.

Evaluate a checkpoint with OpenSpiel's best-response implementation:

```bash
uv run python evaluate_nashconv.py checkpoints/step-100.pth \
  --model agent \
  --game leduc
```

The JSON output reports NashConv, two-player exploitability (`NashConv / 2`),
per-player unilateral improvement, and the number of distinct information
states evaluated by the BoardRL model. `--temperature 1` evaluates the policy
as trained; `--temperature 0` evaluates deterministic argmax play.

## Othello evaluation

Othello uses the standard `Evaluator` with registered random or model opponents,
and the checkpoint arena supplies relative Elo ratings. It has no registered
exact exploitability backend.

[KLENT's Othello benchmark](https://arxiv.org/html/2602.10894v1#S5.SS1) uses a
pretrained Pgx policy as its fixed opponent. The
[author's code](https://github.com/KazukiOhta/klent/blob/main/main.py#L235-L241)
loads it through `pgx.make_baseline_model`. That opponent is not integrated here,
so current Othello evaluation does not reproduce KLENT's fixed-opponent results.
