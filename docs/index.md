# BoardRL

BoardRL (the `century-rl` Python package) trains policy/value networks to play
board and card games. Games expose text observations and an ordered list of legal
actions. The same interface supports model training, heuristic opponents,
evaluation, and a browser workbench.

The library supplies the mechanics; your Python experiment chooses the algorithm.
There is no training configuration language or framework-owned experiment loop.

| I want to… | Start here |
| --- | --- |
| Install and run my first training job | [Quickstart](quickstart.md) |
| Train an existing game with a supplied algorithm | [Training](training.md) |
| Compare agents, resume, or load weights | [Evaluation and checkpoints](evaluation.md) |
| Play or inspect a policy in a browser | [Web workbench](web-server.md) |
| Pick a game or understand its scoring | [Game catalog](games.md) and [native rules](game-rules.md) |
| Implement another game | [Add a game](adding-games.md) |
| Change opponents, targets, or the learning algorithm | [Write a training loop](training-library.md) |
| Understand the implementation | [Rollouts](internals/rollouts.md), [returns](internals/returns.md), [optimization](internals/optimization.md), [models](internals/models.md) |

## From game to trained agent

```text
GameDesc.make_game -> RolloutRunner + player strategies -> Rollouts
  -> reward calculation -> optional strategy selection -> TrainingSample
  -> reference predictions / targets -> Learner -> updated model
  -> Evaluator / Checkpoints / browser workbench
```

For a first CPU run, use Tic-Tac-Toe and the `toy` architecture. For larger
experiments, select a game, an architecture, and a trainer explicitly; the CLI
defaults are experimental recipes and can change. A successful smoke run checks
the pipeline, not playing strength.

All shell commands in these guides run from the repository root unless stated
otherwise. Code links point to the
[repository source](https://github.com/Vermeille/century-RL); the documentation
site is built from the Markdown files in `docs/`.
