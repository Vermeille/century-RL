# Evaluation and checkpoints

Evaluation executes concrete player lineups using the same game interface as
training. Seat rotation is enabled by default to reduce first-player bias.
Strategy identity remains the position in the input lineup even when seats move.

## Compare checkpoint files

```bash
uv run python run_pit.py --game connectfour --games 256 --max-steps 100 \
  --player1 checkpoints/run-a/step-100.pth \
  --player2 checkpoints/run-b/step-100.pth --temperature 1
```

Replace the paths with real files. An omitted player is random; both omitted
means random versus random. `--temperature 0` selects argmax; positive values
sample from scaled logits. `--no-rotate` fixes seating. This CLI constructs
two players, loads the first named model from each file, and applies one
temperature to both. Use Python to choose model names, bots, or player counts.

## Compare in Python

```python
from boardrl import Evaluator, Inference
from boardrl.games import games_library
from boardrl.rl.model import load_model

game = games_library("connectfour")
model = load_model("checkpoints/run-a/step-100.pth", name="agent", device="cpu")
inference = Inference(model, batch_size=64)
with inference.evaluating():
    result = Evaluator(game.make_game, outcome=game.outcome).compare(
        [inference.policy(temperature=0), game.strategy_from_string("tactical_random")],
        names=["agent", "tactical_random"], games=256, max_steps=100,
    )
print(result.win_rate(), result.avg_points(), result.games)
```

For cooperative evaluation, repeat the player once per seat and pass
`outcome=game.outcome`. `result.win_rate()` then measures the game's objective
success via `won()`. It does not measure whether one teammate outscored another.

## What the numbers mean

| Quantity | Meaning |
| --- | --- |
| Competitive `win_rate()` | Mean outcome of strategy 0: win=1, draw=0.5, loss=0, with descriptor-specific tie rules |
| Cooperative `win_rate()` | Fraction of games satisfying the shared objective; truncations are failures |
| `avg_points()` / `my_points()` | Mean / list of final **score differences** (`diff_points_for`), not necessarily raw `points_for` |
| `end.my_points` | Raw final points for that seat |
| `terminal` | Game rules ended the episode |
| `truncated` | The action budget ended before the game did |

With the default competitive outcome, truncation has neutral utility 0 and is
therefore counted as half a win in the competitive summary. Inspect endpoints
before interpreting an apparent 50% result. Generic point summaries include
truncated endpoints; some game-specific metrics (such as OpenSpiel payoff
metrics) explicitly exclude them.

For a custom evaluation report, use `result.rollouts.by_strategy.group(0)` and
inspect `trace[-1]` in each trace. Use `.by_seat` when analyzing board positions.
`Scoreboard.record(result)` accumulates pairwise point comparisons; it is an
optional experiment tool, not an automatic promotion rule or cooperative score.

Fix game modes, opponent, temperature, action horizon, and sample count when
comparing runs. Use multiple seeds and independent evaluation after selecting
checkpoints. A win rate against one bot or an Elo rating does not establish low
exploitability; [OpenSpiel benchmarks](benchmarking.md) describe exact evaluation.

## Resume or initialize

To continue the quickstart from step 2 through step 4:

```bash
OMP_NUM_THREADS=2 uv run python trainers/adversarial-ppo.py \
  --game tictactoe --opponent-eval-strategy random \
  --architecture toy --device cpu --steps 4 \
  --rollout-games 8 --rollout-max-steps 20 \
  --evaluation-games 8 --evaluation-every 1 \
  --inference-batch-size 32 --learner-batch-size 32 \
  --warmup 0 --save-every 1 \
  --checkpoint-root checkpoints/quickstart --tag first --no-progress \
  --resume checkpoints/quickstart/adversarial-ppo/tictactoe/toy/first/step-2.pth
```

`--resume` restores weights, optimizer state, completed step, and adaptive learner
state. Supply the same game, architecture, trainer, and ordered losses. The PPO
scripts reapply command-line optimizer hyperparameters after loading Adam state;
CLI schedules remain authoritative. This is training continuation, not exact
process replay: RNG state and the full command configuration are not restored
by `Checkpoints`.

`--initialize-from path` starts at step 0 with loaded model weights and a new
optimizer/learner. Use a new tag. It is mutually exclusive with `--resume`.
Single-policy and adversarial trainers accept different saved model names;
inspect each trainer's initialization function before crossing trainer families.

## Use checkpoints in your own algorithm

```python
from boardrl import Checkpoints

checkpoints = Checkpoints("checkpoints/my-experiment", keep=5)
path = checkpoints.save(
    10, {"current": model},
    optimizers={"current": optimizer}, states={"learner": learner},
    metadata={"game": "connectfour", "seed": 0},
)
saved = checkpoints.load(
    path, models={"current": model}, optimizers={"current": optimizer},
    states={"learner": learner}, map_location="cpu",
)
start = saved["step"]
```

This fragment assumes your model, optimizer, and learner already exist. Save all
models an algorithm needs (opponent, target, average), and any stateful selector
or schedule that implements `state_dict()` and `load_state_dict()`. Ordinary
objects such as `ReplayBuffer` are not persisted automatically.

Files are written atomically as `<prefix>-<step>.pth`. `.latest` selects the
highest step, and `.sample(exclude_latest=True)` selects an older file; it raises
`LookupError` if no eligible file exists. A model's `spec()` records constructor
arguments so `load_model(path, name=...)` can reconstruct it for inference.
`load_model` defaults to CUDA when available; pass `device="cpu"` to override.

These files use PyTorch object deserialization (`weights_only=False`); load only
trusted checkpoints. Older arbitrary `.pth` formats are not necessarily
compatible with the named `models` / `model_specs` schema.

## Checkpoint arena

The asynchronous CPU arena watches adversarial checkpoints, plays matches, and
maintains relative ratings and plots outside the training loop. The shell
wrapper accepts the effective checkpoint directory followed by a full training
command. Run it inside the project's environment so both child processes can
import the library:

```bash
uv run bash trainers/run_with_arena.sh \
  checkpoints/adversarial-ppo/connectfour/cnn/arena-ppo \
  python trainers/adversarial-ppo.py --game connectfour --architecture cnn \
  --checkpoint-root checkpoints --tag arena-ppo
```

See `uv run python trainers/arena.py --help` for its `checkpoint-dir` and `follow`
subcommands, and the wrapper source for environment controls. Keep the arena
pool/opponents and temperature fixed when interpreting ratings. The
[OpenSpiel launchers](benchmarking.md) also use it and optionally perform final
exact exploitability evaluation.
