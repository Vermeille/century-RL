# Troubleshooting

## The first import fails while compiling Cython

Rollout imports compile `boardrl/cyutils.pyx` using `pyximport`; Century also
compiles `engine.pyx`. Install a C/C++ toolchain and development headers for the
Python interpreter used by `uv`. A source checkout with only Python packages is
not enough. The initial build can take longer than later imports. Check that the
user can write to the compiler cache directory (`~/.pyxbld` by default).

## A trainer rejects a strategy or game argument

Use `--help` on the actual trainer and list the registry:

```bash
uv run python -c 'from boardrl.games import games_library; games_library.display()'
```

The adversarial scripts inherit `tactical_random` as their evaluation opponent;
set `--opponent-eval-strategy random` when switching to another game. A game spec
accepts only the constructor arguments exposed by its registration. In Python,
`desc.strategy_from_string.display()` lists that game's strategy specs.

## A benchmark game is missing

OpenSpiel registrations exist only when `pyspiel` is importable. Install it
with `uv pip install open-spiel`; use `uv run --no-sync` afterward so environment
synchronization does not discard an ad hoc optional installation. Exact board
evaluators need further optional dependencies; see [benchmarks](benchmarking.md).

## Imports fail when running a copied script

Run commands from the repository root using `uv run`. For a new script, use a
module invocation such as `uv run python -m examples.train_minimal`, or install
the project into the environment. A script's containing directory affects
Python's import path.

## CUDA is unavailable or memory runs out

Check `uv run python -c 'import torch; print(torch.cuda.is_available())'`.
Use `--device cpu` to separate installation issues from training issues.
Reduce inference batch size, learner batch size, architecture size, or concurrent
rollout games. Observation length and the number of actions also affect memory.
See [models](internals/models.md) for the character input contract. Large exact
OpenSpiel traversals can require much more RAM than sampled training.

## Evaluation stays around 50%, or cooperative success is zero

Inspect `terminal`, `truncated`, score differences, and the game-specific metrics.
Competitive truncations have neutral utility and count as half a win. Raise
the action budget when games need it; `coop.py` evaluates with a fixed 800-action
horizon. Cooperative objective success can remain zero while partial points
improve. Confirm that runner and evaluator received `outcome=game.outcome`.

## A custom policy chooses the wrong action

Strategies return **logits**, not probabilities. Rollouts apply softmax and
sample an **index into the current `moves` list**. Match tensor length and
ordering to that list, and return `probabilities.log()` if starting from
probabilities. Keep the `@` action lines and all action-indexed targets aligned
through augmentation. Avoid `@` elsewhere in model observations.

## Losses become NaN or a model stops improving

Truncated Monte Carlo `returns` and `score` are deliberately NaN. Use
`ReferenceTargets` and a bootstrap loss for truncated data, and avoid metrics or
losses that expect complete returns unless you filter complete traces. Even
`ValueMetrics` can report NaN on truncated batches. Keep at least two samples
for normalized GAE, whose standard deviation uses a sample estimate.

Compare policy importance ratios, clipping fraction, entropy/perplexity, critic
error, and independent game results. A lower loss alone does not establish
better play. Small lambda values can remove delayed credit; gamma and the two
lambda parameters have separate roles. See [returns](internals/returns.md).

## Checkpoint resume has missing keys or wrong tensor shapes

Use the original trainer, game reward semantics, architecture, and loss ordering.
Point returns use a two-parameter Normal head; outcome returns use three
categorical logits. Old checkpoints or different heads can be incompatible.
Missing learner state cannot be reconstructed from weights alone; use
initialization intentionally if you want fresh optimizer/controller state.

## A model does not appear in the browser

Start the server from the directory containing the checkpoints. Discovery is
recursive, uses the base game name in the path, and happens at session creation.
Restart to refresh it. The server loads the first named model in each file;
multi-policy selection requires exporting the desired model into its own named
checkpoint. See [web model discovery](web-server.md#checkpoint-discovery).

## Pause a long run

On supported POSIX terminals, the learner context handles Ctrl+Z at safe
boundaries, moves models and optimizer tensors off CUDA, and suspends the
process. Use shell job control (`fg`) to resume. This is not a saved checkpoint.
Custom loops must enter the learner context and call `learner.safe_point()`
between rollout/target/update work. Two-learner experiments use one aggregate
pause handler, as in `adversarial-advshape.py`.
