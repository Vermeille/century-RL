# External board-game benchmarks

Century keeps benchmark dependencies outside the core training stack. The
training algorithms stay in `trainers/`; OpenSpiel is used as a game backend
and as an evaluator, while Othello engines are subprocess opponents.

OpenSpiel code lives in `boardrl/games/openspiel/`. Each preset has its own
subdirectory: `leduc/`, `liars_dice/`, `dark_hex/`, `abrupt_dark_hex/`,
`phantom_ttt/`, `abrupt_phantom_ttt/`, and `othello/`. Each contains a concrete
`game.py` class derived from the abstract `OpenSpiel` adapter. Native game
specifications are inline in those constructors. The standard `register_game`
calls in `boardrl/games/__init__.py` select scores, rewards, augmentations, and
evaluation explicitly, with `metrics_from=OpenSpiel` sharing the payoff metrics.
Game-specific solvers and backend settings live in each game's `evaluation.py`,
and Dark Hex's private-state text lives in its
`representation.py`. Othello's match runner and engine client live together.

Shared adapter, policy, result, and evaluator code stays directly under
`boardrl/games/openspiel/`. `exp_a_spiel.py` provides the shared cached traversal
for the four hidden board games. `evaluation.py` selects a backend and exposes
`nash_conv`, `ExactOpenSpielEvaluator`, and the trainer's `OpenSpielModelEvaluator`.

## Install OpenSpiel

OpenSpiel is deliberately optional so benchmark tooling does not enlarge the
runtime dependency set for every Century experiment:

```bash
uv pip install open-spiel
```

## Train benchmark games with the normal Century trainers

The following regular game descriptors are backed by OpenSpiel. Chance
nodes are sampled inside the game adapter, so rollout code still sees only
player decisions. In imperfect-information games the model receives the acting
player's information state; in perfect-information games it receives the
observation/board state.

Leduc keeps its native observation format, including `[Observer: <seat>]` and
`[Player: <seat>]`. At termination its native `Player` field still names the last
actor, so the adapter corrects it to the player whose final view is requested.
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
and register its training semantics explicitly. The evaluation CLI still accepts
native OpenSpiel specifications, such as `--game kuhn_poker`.

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

These five presets use terminal win/loss/draw rewards and exact exploitability
evaluation. They share the standard strategies, action-shuffling augmentation,
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

By default, all five presets select exact evaluation and best checkpoints by
`-exploitability`. Both seats use the model being evaluated, with stochastic
play at temperature 1 by default. Changing evaluation temperature changes the
policy being measured. Rewards are not rescaled in the evaluator.

For faster evaluation against a fixed opponent, pass `--sampled-evaluation`
to any trainer. For example, the Phantom Tic-Tac-Toe launcher accepts:

```bash
./train_openspiel.sh --sampled-evaluation --opponent-eval-strategy random --evaluation-games 256
```

This uses ordinary rollouts with seat rotation and never constructs the exact
evaluator or enumerates information states. `--opponent-eval-strategy` accepts
the game's registered strategies. The resulting win rate (or points for Leduc)
also selects best checkpoints; it measures performance against that opponent,
not exploitability. Cooperative games retain sampled self-play; `imitate.py`
also defaults to self-play unless an opponent is supplied.

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

```bash
uv run python evaluate_nashconv.py checkpoints/step-100.pth \
  --model agent --game liars_dice --temperature 1 --batch-size 1024
```

The CLI accepts these BoardRL presets or native OpenSpiel specifications.
`--game liars_dice` means the 2D5F preset; an explicit native specification such
as `liars_dice(numdice=1,dice_sides=3)` selects a smaller game.
`--backend openspiel` is available for exhaustive cross-checks on small games. It rejects
reveal-nothing Dark Hex because native OpenSpiel best responses use the extra
clock in its information-state string.

Exact evaluation is more expensive than sampled matches: these games contain
millions of information states, all of which must be covered. The pinned board
backend also uses substantial RAM (approximately 7 GiB for classical games and
27–32 GiB for abrupt variants, including working arrays). Insufficient memory
or a missing backend produces an explicit error, never a sampled substitute.

Trainers retain one exact evaluator for the run. For the four board games,
the first evaluation discovers the information sets and builds the exp-a-spiel
traverser; later evaluations reuse it, including agent, environment, and best
checkpoint checks. The backend and its working buffers remain in RAM until the
evaluator is released. Each evaluation recomputes policy probabilities and best
responses from the current model and temperature; no policy results are cached.
Neural inference and the library's payoff traversal still run each time.
Exact evaluation uses the trainer's `--inference-batch-size`, including best
checkpoint reevaluation. Cache and evaluation-phase messages include the PID
so concurrent trainers can be distinguished. `building infosets (cache miss)`
occurs on first use; `reusing infosets (cache hit)` occurs on subsequent calls.
The library's `thread NN found` lines are progress within one discovery, while
`begin gradient computation` is policy-dependent work repeated every evaluation.

This cache is local to the evaluator and process. Restarting training or running
the CLI again enumerates once more: the pinned exp-a-spiel library has no API to
save or restore its traverser. Python callers evaluating multiple checkpoints
can import `ExactOpenSpielEvaluator` from `boardrl.games.openspiel.evaluation`,
retain `ExactOpenSpielEvaluator(spiel_game)`, and call its `evaluate(model,
temperature=1, batch_size=1024)` method for each model. `nash_conv(...)` remains a
one-shot convenience function. Existing training processes need a restart or
checkpoint resume to use the cache.

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

By default, all trainers use the evaluation method selected by the game
descriptor. For two-player Leduc, evaluation logs `exploitability` and
`nash_conv` instead of sampled win rate or points against random. Adversarial
trainers retain their `evaluation/agent/` and `evaluation/environment/` metric groups where applicable;
each model is independently evaluated in both seats. Exploitability is measured
in chips per hand and lower is better. Best checkpoints minimize the agent's
exploitability using the selection score `-exploitability`.

The existing evaluation cadence and `--eval-temperature` still apply. Exact
evaluation covers every information state using `--inference-batch-size`;
`--evaluation-games` and `--opponent-eval-strategy` only apply when
`--sampled-evaluation` is supplied.
The five paper presets also use exact evaluation through the backends above.
Other games retain their sampled opponent or self-play evaluation.

Best-checkpoint metadata records `evaluation_metric`, `evaluation_value`,
`evaluation_score`, and `evaluation_temperature`. On resume, a retained best
Leduc checkpoint with legacy point-based metadata or a different
metric/temperature is reevaluated before comparison. This leaves the live model
and training RNG unchanged. Other games retain their legacy sampled scores.
Running processes must be restarted or resumed to use the new code.

Evaluate a checkpoint with OpenSpiel's best-response implementation:

```bash
uv run python evaluate_nashconv.py checkpoints/step-100.pth \
  --model agent \
  --game leduc_poker
```

The JSON output reports NashConv, two-player exploitability (`NashConv / 2`),
per-player unilateral improvement, and the number of distinct information
states evaluated by the Century model. `--temperature 1` evaluates the policy
as trained; `--temperature 0` evaluates deterministic argmax play.

## Othello: fixed external engine strength

The match runner lives in `boardrl/games/openspiel/othello/benchmark.py`, alongside its
GTP client in `boardrl/games/openspiel/othello/gtp.py`. `evaluate_othello_engine.py` is the
command-line entry point.

The evaluator speaks the standard subset of GTP used by Egaroucid and Edax. It
runs paired games from the same randomized opening, with the Century agent once
as black and once as white, to reduce first-player and opening bias.

Egaroucid example (flags may differ between releases):

```bash
uv run python evaluate_othello_engine.py checkpoints/step-100.pth \
  --model agent \
  --engine "./Egaroucid_for_Console -gtp -quiet -nobook -l 10 -t 4" \
  --pairs 64 \
  --opening-plies 8 \
  --seed 0
```

Edax can be used the same way by passing its GTP command. Sweep fixed engine
levels with the same paired openings to obtain an externally anchored strength
curve without moving the training code into an engine repository.
