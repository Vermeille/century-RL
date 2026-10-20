# Quickstart

This walkthrough installs BoardRL, trains a tiny Tic-Tac-Toe model on CPU, evaluates
it against random play, and opens the browser workbench.

## Install

Use Python 3.12 or newer, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a C compiler. Rollouts import a Cython sampling extension on first use;
Century additionally compiles its environment. On Debian/Ubuntu, the compiler
and Python headers are provided by `build-essential` and the matching
`python3-dev` package.

```bash
git clone https://github.com/Vermeille/century-RL.git
cd century-RL
uv sync --locked
```

`uv run` selects the project's environment. A GPU is optional. The commands below
force CPU so they work independently of CUDA availability. See
[troubleshooting](troubleshooting.md) for compiler, device, and import failures.

## Train a tiny agent

```bash
OMP_NUM_THREADS=2 uv run python trainers/adversarial-ppo.py \
  --game tictactoe --opponent-eval-strategy random \
  --architecture toy --device cpu \
  --steps 2 --rollout-games 8 --rollout-max-steps 20 \
  --evaluation-games 8 --evaluation-every 1 \
  --inference-batch-size 32 --learner-batch-size 32 \
  --warmup 0 --save-every 1 \
  --checkpoint-root checkpoints/quickstart --tag first --no-progress
```

This trains one shared policy from both seats, rotating the seats across games.
Expect baseline evaluation, training metrics, and `.pth` files under:

```text
checkpoints/quickstart/adversarial-ppo/tictactoe/toy/first/
  run.txt
  step-1.pth
  step-2.pth
  best-<step>.pth
```

Only the best observed evaluation is retained under `best-*.pth`. The final
weights are `step-2.pth`. Two updates do not make a strong player; increase
training after checking this pipeline. See [Training](training.md) for the
cooperative trainer, imitation, fixed opponents, and controls.

## Evaluate

```bash
uv run python run_pit.py --game tictactoe --games 32 --max-steps 20 \
  --player1 checkpoints/quickstart/adversarial-ppo/tictactoe/toy/first/step-2.pth \
  --temperature 0
```

The omitted second player is random. `--temperature 0` means deterministic
argmax play. The JSON reports the number of games, player names, win rate, and
average score differences. Competitive win rate counts draws as half a win.
Thirty-two games are a sanity check; use larger independent evaluations to
compare training recipes. [Evaluation](evaluation.md) explains score units and
multi-model checkpoints.

## Play in the browser

```bash
uv run python -m boardrl.serve.serve --game tictactoe
```

Open <http://127.0.0.1:8000>. Select the checkpoint in the strategy selector,
analyze the state, click a legal move, or ask the agent to play one move. The
server discovers checkpoints recursively from its working directory when a
game session is created; restart it after training to refresh the list.

The [web workbench guide](web-server.md) covers comparing strategies, undo/redo,
custom game specs, rendering a new game, and the HTTP API.

## Continue

- Use [the game catalog](games.md) to choose a larger game.
- Follow [The Game guide](thegame.md) for cooperative card-game modes.
- Copy the [minimal training program](training-library.md) to change the algorithm.
- Follow [Add a game](adding-games.md) to train on your own environment.
