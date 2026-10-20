# Models and inference

`boardrl.rl.model.Model` maps a batch of game strings to `PolicyValue`: a list
of one-dimensional legal-action logit tensors and a batched value distribution.
No fixed action vocabulary or game-specific output size is required.

## Text and action encoding

The encoder uses `ord(character)` IDs with a **128-entry** embedding: keep
observations and action strings ASCII. ID 0 is padding and ID 1 is the appended
end marker. The current `maxlen` is 2048, but the encoder is not a safe automatic
truncation mechanism; mixed overlong inputs can fail batching, and backbones
may impose position limits. Keep serialized states within the supported budget
or change/test the model deliberately.

Policy positions are the locations of every `@` character in the original
string. The policy head gathers features at those positions, combines local
action features with pooled state context, and produces one scalar per action.
Use one line `@<move>` per legal move, in exactly `game.moves` order, and no
other `@` characters. Final value-only observations can omit actions.

The value head uses a learned query with cross-attention over state features,
then projects to distribution parameters. Shared policy/value backbones are
the default; `shared_backbone=False` creates separate encoders.

## Choose a preset

```python
from boardrl.games import games_library
from boardrl.models import architectures, make_for_game

print(sorted(architectures))
game = games_library("tictactoe")
model = make_for_game("toy", game)
output = model([game.make_game().display_with_moves()])
print(output.policy[0].shape, output.value.mean)
```

| Preset family | Purpose / backbone |
| --- | --- |
| `toy` | Small transformer for API checks and tiny experiments |
| `cnn`, `cnn-large` | Character convolutional encoders |
| `patchformer-{tiny,small,medium,large}-p{4,8,16}` | Local convolution/canonicalization with patch-level transformer context |
| `extra-small`, `small`, `medium`, `large`, `extra-large` | Transformer configurations; inspect `models.py` for exact widths/depths |
| `minimal-lstm` | Compact recurrent encoder |

Use `architectures` to see every current preset rather than assuming the table
is exhaustive. `make_for_game(name, desc, **overrides)` selects
`points_based=desc.rewards.points_value_head`. `make(name, **overrides)` and
direct preset factories do not infer game semantics; the raw `Model` default
is a points-based critic.

## Value distributions and Q values

Point rewards select a Normal distribution with raw mean and softplus standard
deviation, clamped away from zero. Outcome rewards select a categorical
distribution over `[-1, 0, +1]` (loss/draw/win). Fractional targets within that
interval interpolate cross entropy between neighboring atoms. Predictions
expose `.mean`, `.stddev`, `.log_prob(target)`, `.prediction_loss(target)`, and
`.unbatched()`.

`PolicyValue` unpacks as `policy, value = model(states)`. Its `q_value()` uses
a centered action decomposition:

```text
Q(s,a) = value.mean(s) + policy_logit(s,a) - mean_a(policy_logit(s,a))
```

These are Q estimates only when trained with that interpretation, not calibrated
action returns from an arbitrary PPO policy. See `QMSELoss` and `DoubleQTargets`.

`model.spec()` records constructor arguments for named checkpoints. If you add
a backbone, implement its module in the `BACKBONES` mapping, preserve per-character
output shape and masks, and include constructor settings in the spec. Test
padding, variable action counts, action-free endpoints, device transfer, and
save/load. Architecture changes generally require new training or an explicit
checkpoint conversion.

## Batched async inference

`Inference(model, batch_size=..., timeout=...)` owns a `BatchProcessor` and
creates `PolicySamplingStrategy` objects through `.policy()`. Requests from
concurrent games are collected until the batch cap or timeout, then evaluated
together. This is inference batching, not parallel environment processes.

`policy(temperature=1)` samples model logits normally; temperature 0 returns
deterministic argmax logits. `epsilon` mixes in Dirichlet exploration. The
strategy records behavior logits after exploration and raw reference predictions
for optional target reuse. Do not confuse those two distributions in PPO/KL.

```python
inference = Inference(model, batch_size=64)
with inference.evaluating():
    player = inference.policy(temperature=0)
    data = runner.play([player, player], games=32, max_steps=100)
```

The context switches the model to evaluation mode and restores its previous
mode afterward. Keep rollout and reference annotation within that context,
then let the learner enter training mode. For inference-only checkpoint loads,
`load_model(path, name=..., device=...)` reconstructs and returns an eval model;
see [evaluation](../evaluation.md).
