# Losses and optimization

`Learner` optimizes prepared samples. It does not collect games, select
opponents, compute missing targets, or refresh references. Implement those
operations in your Python experiment before invoking `train`.

## Loss interface

A loss receives variable-length policy logits, a value-distribution object,
a collated `TrainingSample`, and a mutable per-batch `training_state` mapping.
It returns a differentiable scalar `LossResult.objective` and optional metrics.
The learner sums all objectives before backpropagation.

```python
# docs-test: custom-loss
import torch.nn.functional as F
from boardrl.rl.model.loss import Loss, LossResult


class ReturnMAE(Loss):
    supports_off_policy = True
    supports_partial_trajectories = False
    needs_reference_policy_value = False

    def __call__(self, policy, value, sample, training_state):
        return LossResult(F.l1_loss(value.mean, sample.returns))
```

This example requires complete Monte Carlo returns. Set flags accurately,
but do not expect the learner to filter unsupported traces or populate targets
based on them. Use explicit processors. Stateful losses override `state_dict`
and `load_state_dict`; their ordered states are saved by the learner.

## Policy gradient and PPO

`PolicyGradientLoss` selects a scalar sample weight (`returns`, `score`,
`baseline_value`, `advantage`, `gae`, or `normalized_gae`) and minimizes weighted
negative log probability of the sampled action.

With `drift="ppo"`, the ratio uses the **recorded behavior distribution** in
`action_distribution`, not the model supplying `reference_policy`:

```text
ratio = exp(log pi_current(a | s) - log pi_behavior(a | s))
surrogate = min(ratio*A, clip(ratio, 1-epsilon, 1+epsilon)*A)
```

The implementation detaches the ratio and converts the clipped surrogate into
weights for cross entropy; its policy gradient realizes the clipped objective.
It reports mean `importance_ratio` and `clip_fraction`. A high clipping fraction
means many positive-advantage actions exceeded the upper ratio or negative ones
fell below the lower ratio. Other drift choices include `importance_sampling`,
`ppo-rb`, and `spo`; inspect `loss.py` before changing their assumptions.

With no drift correction, the learner accumulates one rollout-wide update.
PPO enables repeated minibatch updates; this does not make arbitrary replayed
data a sound PPO dataset. Keep behavior logits, action indices, target values,
and reference policies fixed for that batch.

## Value prediction

`BootstrapValueLogProbLoss` fits `td_lambda`. By default it clips targets to
`reference_value +/- epsilon * reference_value_stddev`, then applies the value
distribution's prediction loss. `epsilon=None` disables clipping. The current
PPO scripts expose this as `--value-clip-epsilon` and default to no clipping.

For a Normal critic with `normalize_variance=True` (the default), the loss is:

```text
loss = -detach(variance) * Normal(mean, stddev).log_prob(target)
```

Likelihood and detached weighting are computed in double precision and cast
back. The direct mean gradient is `mean - target`, avoiding inverse-variance
amplification of mean errors at tiny predicted scales. This does not bound
the full network's parameter gradients. `normalize_variance=False` uses ordinary
negative likelihood. Categorical outcome critics use cross entropy in either
case, interpolating targets between loss/draw/win atoms.

Changing this objective changes scalar loss units. Compare critic error,
uncertainty calibration, and evaluated play rather than comparing likelihood
and variance-preconditioned loss values. `BootstrapValueMSELoss` fits the same
TD target's mean with squared error; `ValueMSELoss` and `ValueLogProb` require
complete Monte Carlo returns instead.

## KL regularization

`KLPenalty` and `AdaptiveKLPenalty` penalize KL from the current policy to the
stored reference policy. This reference is distinct from PPO's behavior logits.
The adaptive version changes its strength according to a target KL and retains
controller state across updates. Its initial strength is also a lower bound;
raising the target alone may not relax a penalty already at that bound.

The shared-policy MMD trainer uses `alpha = eta = c * sqrt(T/t)` and reverse-KL
strength `1/eta`, where `t` counts accumulated training samples. It saves that
counter in the learner state. This is the supplied PPO-style MMD experiment,
not a general implementation of every mirror-descent variant.

## Exploration

`EntropyBonus` rewards policy entropy. `ReverseEntropyBonus` penalizes divergence
from uniform to the policy; `SymmetricUniformKLPenalty` combines both directions.
Linear variants schedule regularizer strength using `Scheduler`.

`ScheduledPerplexity` adapts a regularizer's strength to a scheduled target.
Raw perplexity is `exp(entropy)`: 1 for a deterministic policy and `n` for
uniform choice among `n` actions. Normalized perplexity is
`(exp(entropy)-1)/(n-1)` and excludes forced single-action states.
The current controller always measures raw effective action count; the
normalized helper is available separately. Targets below 1 are unattainable,
so the controller relaxes toward its baseline rather than attaining them.
Older experiment notes used sub-unit settings under a different convention.

When measured perplexity is below target, it increases strength; otherwise it
relaxes toward a baseline. An EMA smooths measurements, and strength is clamped
within bounds derived from initial strength and baseline ratio. Save the
learner to retain strength, EMA, and controller diagnostics on resume.

## Learner update behavior

If every configured loss has truthy `supports_off_policy`, `BatchUpdates`
shuffles samples, applies augmentations each epoch, and steps the optimizer
per minibatch for the requested epoch count. `normalize_lr=True` scales learning
rate by the first batch-count baseline divided by the current batch count.

Otherwise `RolloutUpdate` uses one epoch, accumulates gradients over minibatches
weighted by their fraction of all samples, and performs one optimizer step.
Thus one loss's flag changes the update granularity of the whole learner.

`gradient_clip` controls optional norm clipping; `gradient_norm` reports the
pre-clipping norm. Empty sample lists return empty metrics. `TrainResult.metrics`
averages values across minibatches, not necessarily weighted by sample count.
Repeated loss class names share metric names, so use distinct reporting if your
experiment needs to distinguish two instances of one class.

`lr_schedule` scales initial optimizer learning rates from experiment-supplied
normalized progress. `learner.state_dict()` retains the batch-count baseline
and loss states; model/optimizer state must be saved separately. Schedules and
loss hyperparameters are reconstructed by the experiment, not restored from
that learner state automatically.
