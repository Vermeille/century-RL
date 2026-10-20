# Train with the supplied programs

The scripts in `trainers/` are complete experiments written in Python. They
compose the same public APIs you can use in your own loop. Run a script with
`--help` for its full current options; many adversarial trainers inherit a parser
from `coop.py`, so some inherited exploration options have no effect on them.

## Choose a trainer

| Program | Training behavior | Saved model names |
| --- | --- | --- |
| `trainers/coop.py` | One shared policy controls both seats; PPO, exploration controller, adaptive KL, TD(lambda) critic | `current` |
| `trainers/adversarial-ppo.py` | Shared-policy self-play; PPO and critic, without exploration or KL penalty | `agent` |
| `trainers/adversarial-mmd.py` | Shared-policy PPO with timestep-dependent entropy and reverse KL regularization | `agent` |
| `trainers/adversarial-advshape.py` | Two policies; each policy's advantage is multiplied by its threshold minus its batch win rate | `agent`, `environment` |
| `trainers/fightbot.py` | Train one policy against a fixed registered bot | `current` |
| `trainers/imitate.py` | Match a registered teacher's action logits with cross entropy | `current` |

The shipped loops construct **two-entry lineups**. Games supporting three to five
players need a custom Python loop with one lineup entry per seat. The lineup
length is passed to the game constructor as `num_players`; a `num_players=4`
game spec alone does not turn these scripts into four-player trainers.

## Cooperative PPO

This is a starting command for a larger run, not a claim of convergence:

```bash
uv run python trainers/coop.py \
  --game thegame,mode=strict --architecture cnn --device cpu \
  --steps 100 --rollout-games 32 --evaluation-games 32 \
  --inference-batch-size 64 --learner-batch-size 64 \
  --checkpoint-root checkpoints --tag strict-cnn
```

Use `--device cuda` on a suitable PyTorch/CUDA installation. The default game
for `coop.py` is currently `thegame,mode=omni`, whereas the game registry's
`thegame` default is `strict`. Always record the full game spec.

`coop.py` evaluates before training, periodically, and after the final update
when best-checkpoint saving is enabled. Cooperative evaluation uses the model
in both seats. For competitive games, evaluation uses
`--opponent-eval-strategy` (default `random`). Its evaluation horizon is fixed
at 800 environment actions; `--rollout-max-steps` controls training rollouts.

## Competitive self-play and fixed opponents

```bash
uv run python trainers/adversarial-ppo.py \
  --game connectfour --architecture cnn \
  --opponent-eval-strategy tactical_random --steps 1000 \
  --checkpoint-root checkpoints --tag connectfour-ppo
```

Self-play trains from both seats of one model. The evaluation opponent measures
progress but is not used as the training opponent. To train against a fixed bot:

```bash
uv run python trainers/fightbot.py \
  --game nim,num_stones=21,max_pick=3 --architecture cnn \
  --opponent-bot optimal --steps 1000 \
  --checkpoint-root checkpoints --tag nim-vs-optimal
```

`fightbot.py` uses `--opponent-bot` for both training and evaluation. If you
change its game from Connect Four, set a bot valid for that game. Similarly,
the adversarial trainers default to `tactical_random` for evaluation, so supply
`--opponent-eval-strategy random` for games without that strategy.

MMD and advantage shaping use the same common arguments:

```bash
uv run python trainers/adversarial-mmd.py --help
uv run python trainers/adversarial-advshape.py --help
```

For the exact policy update and regularizer schedules, see
[Losses and optimization](internals/optimization.md).

## Imitation

`imitate.py` collects states by playing the teacher and trains the model to match
the teacher distribution. Supply a strategy registered for the selected game:

```bash
uv run python trainers/imitate.py \
  --game thegame,mode=strict --architecture cnn --strategy lowest_cost \
  --steps 100 --rollout-games 32 --evaluation-games 32 \
  --checkpoint-root checkpoints --tag imitate-lowest-cost
```

This trainer uses `--batch-size`, unlike PPO's separate inference and learner
batch-size options. With `--pretrain`, it also writes a `pretrain-<step>.pth`
checkpoint with reinitialized heads for subsequent RL. To start PPO from the learned policy,
choose the normal `step-*.pth` file; use the `pretrain-*` artifact intentionally
when you want the encoder but fresh heads.

## Controls that change the experiment

| Option | Meaning |
| --- | --- |
| `--steps` | Total outer updates, including already completed updates on resume |
| `--rollout-games` | Concurrent games collected per update, not transitions |
| `--rollout-max-steps` | Maximum environment actions per training game; reaching it truncates the trace |
| `--inference-batch-size` | Maximum batched live model evaluations; also used for target annotation in current PPO scripts |
| `--learner-batch-size` | Optimization minibatch size |
| `--epochs` | Passes over the same batch when all losses support the minibatch update path |
| `--discount` | Gamma applied per transition in a player's own trace |
| `--gae-lambda` | Trace decay for the policy advantage estimator |
| `--value-lambda` | Trace decay for critic targets; distinct from GAE lambda |
| `--ppo-clip` | Radius of the importance-ratio clipping interval |
| `--value-clip-epsilon` | Clip critic targets within this many reference standard deviations; `none` disables it |
| `--random-move-prob` | Generate a random opening prefix for training states |
| `--eval-temperature` | Temperature of the evaluated policy; 0 means argmax |
| `--seed` | Seed Python and Torch random generators; not a bitwise reproducibility guarantee |

Exploration (`coop.py` and advantage shaping): `--exploration-controller thermostat`
adapts a regularizer to a scheduled perplexity target; `linear` schedules its
strength directly. `--exploration-regularizer` selects entropy, reverse KL to
uniform, or symmetric KL to uniform. `--perplexity-start/end` are **absolute**
effective action counts: 1 is deterministic and `n` is uniform over `n` actions.
Targets below 1 are unattainable and drive the controller toward its baseline.
See the
[controller implementation guide](internals/optimization.md#exploration).

Exploration and learning rate have separate schedule windows. Use
`--schedule-start`, `--schedule-steps`, `--lr-schedule-start`, and
`--lr-schedule-steps` for explicit update counts. Percentage defaults are based
on `--steps`; changing the total on resume can therefore change the schedule.
The learning rate also has warmup and a final multiplier `--min-lr-scale`.

## Artifacts, logging, and continuation

For PPO trainers, the directory is:

```text
<checkpoint-root>/<trainer>/<full-game-spec>/<architecture>/<tag>/
```

Periodic `step-*.pth` files contain models, specs, optimizers, learner/loss state,
step, and metadata. `--keep-checkpoints` bounds periodic retention (default 10).
`best-*.pth` retains one sampled evaluation winner unless `--no-save-best` is
set. A best sampled score can be noisy; evaluate final and best policies
independently. Reusing a directory can overwrite files or prune older files, so
use a unique tag for each experiment.

`run.txt` records the command, arguments, source revision, and local diff.
See [checkpoints and resuming](evaluation.md#resume-or-initialize) for a concrete
continuation command and the distinction between resume and initialization.

The console works without extra services. For Trackio logging:

```bash
uv sync --locked --extra trackio
uv run python trainers/coop.py --game thegame,mode=strict \
  --architecture cnn --tag tracked --trackio
```

`--trackio-url` overrides `TRACKIO_SERVER_URL` (or `TRACKIO_URL`); omit it to
use Trackio's configured local behavior. Console names use dots; Trackio paths
use slashes, for example `evaluation/win_rate`. `--no-progress` hides progress
bars, not metrics. Measure score distributions, objective success, evaluation
temperature, and truncation alongside losses; see [troubleshooting](troubleshooting.md).
