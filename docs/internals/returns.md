# Rewards, returns, and bootstrapping

`boardrl/games/semantics.py` separates reporting, winning, and learning:
`scores` controls numeric metrics, `outcome` defines terminal utility and success,
and `rewards` constructs training rewards and selects the value-head family.
Choose all three when registering a game.

## From game scores to rewards

`Record` stores the active player's raw `points()` and `diff_points()` before
an action. `EndState` stores those quantities at the final forced-player view.
Each player trace advances between that player's decisions. Let `d_t` be its
recorded score difference:

```text
PointDeltaRewards(scale=c): r_t = c * (d_(t+1) - d_t)
TerminalOutcomeRewards:     r_t = 0, except final action gets terminal utility
```

Competitive terminal utility is +1 for scoring above every opponent, -1 for
scoring below the best opponent, and 0 for tying the best opponent. A custom
outcome can supply other rules (Splendor handles its card-count tiebreak).
Cooperative utility is +1 for `game.won()` and -1 for a proper terminal failure.
Truncation is not a terminal failure: it has neutral utility and no invented
terminal reward.

`desc.rewards.make_returns(gamma)` chooses the correct reward function and
returns a `ComputeReturns` processor. Direct `ComputeReturns(gamma)` defaults
to **unscaled point deltas**; it does not infer descriptor semantics. Prefer the
descriptor factory unless intentionally changing the training objective.

An optional sampled entropy reward adds `strength * -log pi(a_t | s_t)` to
each step reward. This changes the return objective and is separate from an
entropy loss applied directly during optimization.

## Monte Carlo returns and final scores

For proper terminal traces, `ComputeReturns` assigns:

```text
returns_t = r_t + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
score_t   = final score difference
```

For truncated traces, `returns` and `score` are NaN deliberately: the missing
tail is unknown. `ToSamples` preserves these values, and bootstrapped targets
are populated separately by `ReferenceTargets`. Filter complete traces before
using Monte Carlo-only losses or metrics. A truncation does not justify setting
the value tail to zero.

Gamma is applied per transition in a player's own trace, even if several
environment actions happened in between. If your algorithm discounts per
environment step or wall-clock interval, implement an appropriate processor.

## Reference predictions

`ReferenceTargets(model, batch_size=..., discount=gamma, gae_lambda=...,
value_lambda=...)` evaluates observations without gradients and records:

- `reference_policy`: action logits for KL regularization and related targets.
- `reference_value` and `reference_value_stddev`: scalar distribution summaries.
- `reference_max_q`: largest centered policy-plus-value estimate.
- `next_reference_value` and `next_reference_max_q`: successor estimates.

For a terminal successor the next values are zero. A truncated successor is
evaluated from its endpoint observation. An ordinary successor uses the next
same-player sample's reference prediction.

`reuse_rollout_predictions=True` reuses cached model predictions only when all
required reference fields are present. This saves inference for compatible
on-policy data. It is invalid when you intentionally want a different target
network. Run annotation while the model is in evaluation mode; `no_grad()`
alone does not switch module mode. Keep that reference fixed throughout an
update.

## GAE and TD(lambda) are different targets

With frozen value estimates `V_t`, the implementation computes:

```text
delta_t = r_t + gamma * V_(t+1) - V_t
gae_t   = delta_t + gamma * gae_lambda * gae_(t+1)
td_t    = r_t + gamma * ((1 - value_lambda)*V_(t+1) + value_lambda*td_(t+1))
```

At a terminal boundary, the value and recursive tails are zero. At truncation,
GAE ends its recursive tail while retaining the endpoint value in `delta_t`;
TD(lambda)'s tail uses that endpoint value. With lambda 1, complete trajectories
use Monte Carlo targets, and truncated trajectories still bootstrap the missing
tail.

`gae_lambda` determines how far policy credit propagates through TD residuals.
`value_lambda` mixes bootstrap estimates with longer critic targets. Gamma
changes the objective's time preference. Lowering a lambda trades longer-range
credit for greater reliance on the critic; it does not change gamma.

`normalized_gae` subtracts the mean and divides by `std + 1e-4` over the entire
selected on-policy batch, using the sample standard deviation. It is computed
before learner shuffling or minibatching. Use at least two transitions. If you
select only strategy 0 before reference annotation, its normalization statistics
come from strategy 0's samples.

## Select a target for the loss

| Loss / weight | Sample target |
| --- | --- |
| `ValueMSELoss`, `ValueLogProb` | Complete Monte Carlo `returns` |
| `BootstrapValueMSELoss`, `BootstrapValueLogProbLoss` | `td_lambda` |
| `PolicyGradientLoss(weight="advantage")` | One-step `advantage` (`delta_t`) |
| `PolicyGradientLoss(weight="gae")` | Unnormalized GAE |
| `PolicyGradientLoss(weight="normalized_gae")` | Batch-normalized GAE |
| `PolicyGradientLoss(weight="returns")` | Complete Monte Carlo `returns` |
| `QMSELoss` | Reward plus discounted `next_reference_max_q` |

Value targets must match the critic's units. Outcome critics accept [-1, 1];
arbitrary point returns require a Normal head. Reward scaling, entropy rewards,
and custom processors can move targets outside the categorical range, so choose
the head and loss together. See [models](models.md) and [optimization](optimization.md).
