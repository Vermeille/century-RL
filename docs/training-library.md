# Write a training loop

An experiment is an ordinary Python program. BoardRL supplies `RolloutRunner`,
`Pipeline`, `Learner`, `MetricLogger`, `Checkpoints`, and `Evaluator`; your
program decides who plays, where samples go, what predictions stay fixed, and
when opponents or targets change.

## Run a complete small experiment

[`examples/train_minimal.py`](https://github.com/Vermeille/century-RL/blob/master/examples/train_minimal.py)
is a complete CPU PPO program with a CLI, seed, metrics, checkpoint, and
descriptor-selected reward semantics. Run it before editing:

```bash
OMP_NUM_THREADS=2 uv run python -m examples.train_minimal \
  --game tictactoe --steps 2 --games 8 --directory checkpoints/minimal/tictactoe
```

It saves `step-2.pth` with model name `current`. Change `--game` to another
two-player registration to keep the mechanics and change the task. For larger
experiments, copy `trainers/coop.py` or an adversarial script instead. The older
[`adversarial_self_play.py`](https://github.com/Vermeille/century-RL/blob/master/examples/adversarial_self_play.py)
and
[`current_vs_reference.py`](https://github.com/Vermeille/century-RL/blob/master/examples/current_vs_reference.py)
are longer-running illustrative experiments with their own reward choices.

## Assemble the components

The core of the minimal example is:

```python
model = make_for_game("toy", game)
inference = Inference(model, batch_size=32)
runner = RolloutRunner(game.make_game, outcome=game.outcome)
prepare = Pipeline(
    game.rewards.make_returns(1.0),
    ToSamples(),
    ReferenceTargets(model, batch_size=32, discount=1.0,
                     gae_lambda=1.0, value_lambda=1.0),
)
```

This is an excerpt; imports and the optimizer/learner are in the executable
file. `make_for_game` chooses a value head matching the descriptor rewards.
Use the descriptor's return processor to avoid accidentally changing terminal
outcome learning into point learning.

For each update, collect data and annotate targets in evaluation mode, then
call `learner.train(samples, progress=...)`. Finish target computation before
changing the reference weights. `progress` is a normalized schedule position
chosen by the experiment. Evaluation mode is separate from disabling gradients.

The [rollout guide](internals/rollouts.md) describes the data shapes,
[returns](internals/returns.md) derives the targets, and
[optimization](internals/optimization.md) explains the loss interface.

## Train only the current player

In current-versus-opponent experiments, keep strategy identities explicit:

```python
prepare_current = Pipeline(
    game.rewards.make_returns(1.0),
    Select(strategies=[0]),
    ToSamples(),
    ReferenceTargets(current, batch_size=64, discount=1.0,
                     gae_lambda=1.0, value_lambda=1.0),
)
with current_inference.evaluating(), opponent_inference.evaluating():
    data = runner.play(
        [current_inference.policy(), opponent_inference.policy()],
        games=64, max_steps=800,
    )
    samples = prepare_current(data)
learner.train(samples)
```

This fragment assumes you created both models and inference wrappers.
`Select(strategies=[0])` follows the current player across seat rotation;
`Select(seats=[0])` would select the first physical seat instead.
Replace the opponent with `game.strategy_from_string("random")` for a fixed
bot, omitting its inference context.

## Champion, reference, and EMA opponents

An opponent and a target reference serve different roles. Opponent weights
determine the environment's behavior; reference predictions determine the
learning objective. They may use separate models and update schedules.

For a champion, use `copy.deepcopy(current).eval()` and an `Inference` wrapper.
Promote it only when your evaluation rule succeeds:

```python
with current_inference.evaluating(), champion_inference.evaluating():
    result = evaluator.compare(
        [current_inference.policy(), champion_inference.policy()],
        names=["current", "champion"], games=200, max_steps=800,
    )
if result.win_rate(0) > 0.55:
    champion.load_state_dict(current.state_dict())
```

The threshold is an example policy, not a statistical test. Choose evaluation
size and promotion rules for your experiment.

`ExponentialMovingAverage` in `boardrl.models` owns `.model` and `update(current)`.
Construct the opponent's `Inference` from `ema.model`, train the current player,
then update the EMA. Save both model weights and any experiment bookkeeping you
need for continuation. Hard reference copies can use `copy_weights(target, source)`.

## Opponent populations and curricula

`RolloutRunner.play` accepts a function selecting a lineup per game:

```python
def lineup(game_index):
    old_path = checkpoints.sample(exclude_latest=True)
    old = cached_inference_for(old_path)
    return [current_inference.policy(), old.policy()]

data = runner.play(lineup, games=128, max_steps=800)
```

`cached_inference_for` is your cache, not a library function. Seed the checkpoint
population before excluding the latest, bound loaded-model memory, and decide
what to do with an empty population. For a curriculum, use the same function
to select bots, difficulty, or a policy mixture based on experiment state.

## Imitation and search

Strategies return behavior logits, which the rollout records retain. To imitate
a teacher or search distribution, collect its trajectories and use
`ImitationCELoss`:

```python
teacher = game.strategy_from_string("random")
data = runner.play([teacher, teacher], games=32, max_steps=800)
samples = Pipeline(game.rewards.make_returns(1.0), ToSamples())(data)
imitation_learner.train(samples)
```

Create `imitation_learner` with `losses=[ImitationCELoss()]`. This fits the
teacher's full softmax distribution; `CELoss` instead fits sampled action
indices. A custom tree-search strategy follows the same contract. For an
offline dataset, populate `TrainingSample(state=..., action_distribution=...)`
for distribution imitation, or `action_idx=...` for hard action labels. No
rollout runner is needed for already prepared data.

Search improvement and self-play scheduling are your algorithm. BoardRL does
not supply a turnkey AlphaZero or NFSP implementation. A two-model algorithm
can create independent learners, route best-response traces to RL, maintain
its own reservoir for average-policy imitation, and choose behavior mixtures
in its lineup function. `ReplayBuffer` is uniform recent replay, not a
reservoir or an automatic off-policy correction.

## Q learning

For a Double-DQN-style experiment, compose return/reward calculation,
`ToSamples`, a replay buffer, and `DoubleQTargets(online, target, batch_size=...)`.
That processor picks the maximizing next action using the online model and
evaluates it with the target model. `QMSELoss(discount_factor=...)` fits
`reward + gamma * next_reference_max_q` using the model's centered
policy-plus-value Q representation. The experiment owns target copies,
exploration, terminal handling, and replay persistence.

This is a building block rather than a complete trained DQN recipe. See
`tests/test_training_library.py` for compact executable API compositions.

## Metrics and checkpoints

```python
logger = MetricLogger(Console())
logger.log(step, train=result.metrics)
logger.game(step, game.make_metrics(data))
```

A custom sink implements `log(step, values)`. Game metric adapters return
nested values; sinks handle formatting. `GroupedTraceMetrics` and `TraceMetrics`
require `scores=game.scores` and keep strategy or seat identity explicit.

Save named models, optimizers, and learner states together; see
[checkpoint examples](evaluation.md#use-checkpoints-in-your-own-algorithm).
For multiple learners, save each in the `states` mapping. Ordered loss state
includes adaptive KL and exploration controllers, and must match on resume.

## Verify a new algorithm

First run on a tiny deterministic game with short complete episodes. Inspect
the action distributions, reward units, terminal and truncated endpoints,
reference predictions, loss target fields, and which strategy's samples reach
each learner. Confirm one update changes the intended model, then save/load
and evaluate it. Only scale game count and architecture after those boundaries
work. Slow performance comparisons belong in experiments, not a unit test that
assumes one seed must learn.
