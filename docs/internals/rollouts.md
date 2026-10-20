# Rollouts and training samples

`boardrl/rollouts.py` owns game execution and inference batching. Algorithms
provide concrete strategies, choose opponents, and decide which traces to train
on. Rollout code does not choose a league, reference model, or promotion policy.

## Strategy and lineup contract

```python
async def strategy(game):
    # Tensor with one logit per entry of game.moves, in the same order.
    return logits, info
```

Rollouts apply `softmax(logits)` and sample an action index. `info` is optional
metadata in a dictionary; `info["state"]` overrides the recorded observation,
and model strategies attach reference predictions. This override must keep
the action list aligned with the sampled logits.

`RolloutRunner(make_game, outcome=desc.outcome).play(lineup, games=...,
max_steps=..., rotate=True)` accepts either a list of players or a callable
`lineup(game_index)` returning a list. Every game is constructed with
`num_players=len(lineup)`. Concurrent games run as async tasks; model requests
from those tasks meet in an inference batch.

`play_games` is the underlying primitive for already materialized lineups.
`max_steps` counts all environment actions across seats, including phase moves,
messages, and passes. It is not a count of rounds or one player's decisions.

## Seat and strategy identity

For game index `i`, physical seat `s` receives lineup player
`(s + i) % num_players` when rotation is enabled. Thus strategy 0 may occupy
different seats across games.

| Access | Use |
| --- | --- |
| `game_trace.by_seat[seat]` | Follow one physical board position |
| `game_trace.by_strategy[index]` | Follow one input lineup entry |
| `rollouts.by_strategy.group(index)` | Aggregate that player across games |
| `rollouts.by_seat.group(seat)` | Analyze first-player or seat bias |
| `Select(strategies=[0])` | Train the current policy while excluding opponent decisions |
| `Select(seats=[0])` | Intentionally train only a fixed board position |

If a lineup factory changes the opponent object per game, strategy identity is
still the lineup position; it is not the Python object identity or model name.

## Trace representation

`Rollouts` contains `GameTrace` objects. Each contains one `PlayerTrace` per seat.
Each player trace consists of that player's decision records followed by exactly
one endpoint. A player that never acted may have only its endpoint.

| Object | Stored information |
| --- | --- |
| `Record` | Observation before the move, ordered moves, behavior logits, sampled action index, player, round, raw points, score difference, and metadata |
| `EndState` | Final forced-player observation, points, score difference, outcome utility, terminal/truncated flags, and cause |
| `PlayerTrace` | `seat_id`, `strategy_id`, ordered records and endpoint |
| `GameTrace` | Explicit views by seat and strategy |

The endpoint's `cause` is `proper` for rule termination or `toolong` for
truncation. Cooperative endpoints record shared `won`; competitive utility is
computed through the descriptor's outcome. Pass the outcome object to the runner
and evaluator; their default is competitive.

`Rollouts.num_samples()` includes endpoints, whereas `ToSamples()` emits only
action-bearing records. Compare optimizer throughput using the latter when
you mean training transitions.

## Processing order

```text
Rollouts
 -> ComputeReturns / desc.rewards.make_returns: mutate rewards and returns
 -> Select: optionally retain chosen players
 -> ToSamples: flatten records into linked TrainingSample objects
 -> ReferenceTargets: populate frozen policy/value predictions and targets
 -> Learner: augment, collate, move tensors, and optimize
```

`ToSamples` links samples to the next decision **of the same player**, not
the next environment action. The last action links to that player's `EndState`.
Rewards therefore include what happened between those decisions. This matters
in multi-action turns and games with long gaps between a player's turns.

`TrainingSample` has a fixed dataclass schema in `boardrl/training/sample.py`.
It carries behavior/action fields, scalar rewards/returns, reference fields,
`advantage`, `gae`, `normalized_gae`, `td_lambda`, and a `next` link.
`ToSamples` copies only known reference metadata fields from rollout records;
arbitrary `info` keys do not become sample attributes. The `moves` field is
optional and is not copied by the current conversion; legal actions remain in
the observation text.

## Batching and mutation

`TrainingSample.collate(samples)` stacks same-shaped tensors and scalars and
keeps variable-length policies as lists. Fields must be populated consistently:
a mixture of `None` and values raises an error. The linked `next` field is
excluded from collation. `.to(device)` transfers direct tensor fields and lists
of tensors.

Return annotation mutates records; reference annotation mutates samples and
memoizes recursive GAE/TD(lambda) results. Treat prepared samples as belonging
to one fixed target computation. Rebuild them (or clear all cached target fields
deliberately) before using a different reference model or target parameters.

`ReplayBuffer(capacity, samples_per_update)` retains sample objects and samples
uniformly without replacement. It does not implement a reservoir, prioritized
replay, deep copying, checkpoint persistence, or target invalidation. Linked
samples may keep later transitions alive beyond deque capacity. An off-policy
algorithm must own those choices and refresh the fields its losses require.
