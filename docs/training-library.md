# Writing training algorithms

There is intentionally no framework-owned training loop. An experiment is a
Python program which composes five independent mechanisms:

- `RolloutRunner.play` automatically accepts either an explicit lineup or a
  function that chooses a different lineup for each game.
- `Pipeline` composes return calculation, filtering, sample conversion, model
  targets, and arbitrary user functions.
- `Learner` performs optimization on prepared samples and returns metrics.
- `MetricLogger` sends the same nested values to the console, Trackio, or custom
  sinks. Games implement `GameMetrics.metrics()` once; sink support is automatic.
- `Checkpoints` saves and restores named models, optimizers, and stateful
  training objects. `Learner.state_dict()` includes both its batch
  normalization baseline and the ordered state of every loss.
- `Evaluator` compares explicit players and returns an object on which custom
  promotion rules can operate.

See `trainers/coop.py` for the complete former TDMSE configuration and
`examples/adversarial_self_play.py` for the smallest adversarial loop.

## Common algorithms

The important policy decisions remain visible in the experiment.

### Current versus a champion

```python
champion = copy.deepcopy(current)
result = evaluator.compare(
    [current_inference.policy(), champion_inference.policy()],
    names=["current", "champion"], games=200, max_steps=100,
)
if result.win_rate(0) > 0.55:  # any custom rule is possible
    champion.load_state_dict(current.state_dict())
```

### Current versus EMA

```python
ema = ExponentialMovingAverage(current, decay=0.999)
games = rollouts.play(
    [current_inference.policy(), ema_inference.policy()],
    games=64, max_steps=100,
)
learner.train(prepare(games.only_strategy([0])))
ema.update(current)
```

### Current versus a hard-coded player

Strategies are normal objects, not strings:

```python
from boardrl.games.strategies import RandomStrategy
games = rollouts.play(
    [current_inference.policy(), RandomStrategy()], games=64, max_steps=100
)
```

Game-specific strategies work the same way, for example
`LowestCostStrategy()` from `boardrl.games.thegame.strategies`.

### Metrics by strategy or seat

Seat identity is board position; strategy identity is the lineup entry and
survives seat rotation. Keep that choice explicit when computing metrics:

```python
agent = games.by_strategy.group(0)
first_seat = games.by_seat.group(0)

agent.win_rate()
agent.points()
agent.avg_actions()
agent.sensitivity()
```

`TraceMetrics(agent).metrics()` produces the common outcome summary for one
identity. `GroupedTraceMetrics(games.by_strategy).metrics()` produces a nested
mapping keyed by strategy identity, suitable for `MetricLogger` and Trackio.
Game-specific metric classes can subclass `TraceMetrics` to add policy-specific
measurements without combining different strategies.

### NFSP-like two-model training

Create two independent `Learner` objects. Pick the behavior lineup in Python,
then train each learner from the samples it owns:

```python
def lineup(_game_index):
    opponent = average.policy() if random.random() < anticipatory else best_response.policy()
    return [best_response.policy(), opponent]

games = rollouts.play(lineup, games=128, max_steps=100)
best_response_learner.train(reinforcement_targets(games.only_strategy([0])))
average_learner.train(imitation_targets(reservoir.sample(4096)))
```

### Imitation and AlphaZero-like training

A strategy's returned action logits are stored in every rollout record.
Therefore a tree-search strategy can be imitated with no special rollout loop:

```python
games = rollouts.play([TreeSearch(current), TreeSearch(current)], games=64, max_steps=100)
samples = Pipeline(ComputeReturns(1.0), ToSamples())(games)
imitation_learner.train(samples)  # use ImitationCELoss
```

Pure offline imitation is simply `imitation_learner.train(dataset)`.

### Distribution of past versions

Checkpoint selection is also experiment policy:

```python
def lineup(_game_index):
    old_path = checkpoints.sample(exclude_latest=True)
    old = cached_inference_for(old_path)
    return [current.policy(), old.policy()]

games = rollouts.play(lineup, games=128, max_steps=100)
```

This deliberately replaces the old `MatchMaker`: a distribution, curriculum,
league, or adaptive opponent selector is a short function with access to the
full experiment state.
