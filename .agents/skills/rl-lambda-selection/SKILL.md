---
name: rl-lambda-selection
description: Choose, justify, and tune separate TD(lambda) value targets and GAE lambda for BoardRL/self-play experiments. Use this skill when setting `value_lambda` / `--value-lambda`, `gae_lambda` / `--gae-lambda`, diagnosing noisy returns or poor temporal credit assignment, or deriving sensible defaults from a game's reward and turn structure. Do not confuse either lambda with gamma.
---

# RL Lambda Selection

Use this skill when selecting TD(lambda) and GAE lambda for BoardRL experiments, especially self-play board games. Treat them as two different hyperparameters. The repository already exposes them independently as `value_lambda` and `gae_lambda`.

## Core rule

Keep `gamma` conceptually separate. Gamma defines the objective's temporal discounting. The lambdas below define estimators/traces.

Start from the conceptual endpoints:

```text
value_lambda = 1.0
gae_lambda   = 0.0
```

Interpretation:

> Reality teaches the critic; the critic teaches the actor.

Move away from those endpoints only to compensate for an identifiable failure mode.

## Value lambda: how much should the critic learn from reality vs itself?

`value_lambda = 1` is Monte Carlo / realized-return regression. It has no bootstrap bias and propagates terminal outcomes directly to every visited state in the trajectory, but its targets can be noisy because later sampled actions affect the realized outcome.

Lowering `value_lambda` replaces part of that sampled future with the critic's prediction. A competent critic can therefore denoise the target by averaging over possible continuations, but this introduces bootstrap error and slows backward propagation of sparse/terminal rewards.

Use this tuning rule:

1. Start at `value_lambda = 1.0`.
2. Lower it only if realized-return noise is making value learning sample-inefficient.
3. Stop lowering it once bootstrapping causes distant real outcomes to propagate too weakly or too slowly.

### Environment-derived prior for value lambda

Estimate:

```text
D_v = typical distance to the next non-bootstrap learning signal
```

A non-bootstrap signal is a terminal outcome, real reward, end-of-round score, or other actual scoring event.

A signal `D_v` steps away reaches an earlier state through the lambda-return with roughly `value_lambda ** D_v` relative weight. If a fraction `c` of the signal should survive directly, use:

```text
value_lambda >= c ** (1 / D_v)
```

Treat this as a rough lower bound, not an optimum. `c = 0.1..0.3` is a reasonable sanity range.

Examples with `c = 0.2`:

| D_v | approximate lower bound |
|---:|---:|
| 2 | .45 |
| 5 | .72 |
| 10 | .85 |
| 20 | .92 |
| 50 | .97 |
| 100 | .98 |

Push value lambda upward for sparse/terminal rewards, long anchor spacing, weak critics, and low continuation noise. Push it downward when future play is highly stochastic/exploratory and the critic can accurately average over those continuations.

Do not call a lower value lambda better merely because its training loss is smaller: the target itself increasingly contains the critic's own predictions. Validate against independent realized returns, repeated continuations from identical states, successor-state ranking, or downstream policy strength.

## GAE lambda: how far may later critic realizations repair current action credit?

`gae_lambda = 0` uses only the immediate TD residual. It asks the critic to rank the immediate successor states correctly. Increasing GAE lambda allows TD errors that appear later in the trajectory to flow backward and repair earlier action credit.

In a fully Markov environment with a perfect critic, `gae_lambda = 0` is sufficient: every consequence that can affect the future is already encoded in the immediate successor state, and a perfect value function captures it.

A nonzero GAE lambda is therefore mainly compensation for the actual critic's **recognition delay**: consequences may be present in the state but too subtle for the current function approximator to understand immediately.

Estimate:

```text
D_a = typical number of later transitions before the current critic can recognize
      that an earlier action was better or worse than it first appeared
```

A correcting TD residual `D_a` steps later reaches the original action with roughly `gae_lambda ** D_a` relative weight. If a fraction `c` of that correction should survive, use:

```text
gae_lambda ~= c ** (1 / D_a)
```

For a first environment-based prior, use `c = 0.3..0.5`. With `c = 0.5`:

| D_a | approximate GAE lambda |
|---:|---:|
| 0 | 0 |
| 1 | .50 |
| 2 | .71 |
| 3 | .79 |
| 5 | .87 |
| 10 | .93 |

These values are starting priors, not universal optima.

### Correct for intervening decision noise

Recognition delay alone is insufficient. Count how much independent decision-making happens before the consequence becomes legible.

```text
action -> forced sequence -> consequence
```

can tolerate a much longer GAE trace than:

```text
action -> opponent choice -> own choice -> opponent choice -> consequence
```

for the same raw delay.

Therefore:

- longer recognition delay -> increase `gae_lambda`;
- more independent/high-impact intervening decisions -> decrease `gae_lambda`.

This is especially important in adversarial self-play: high GAE lambda lets later exploratory moves, opponent mistakes, and unrelated decisions contaminate earlier action credit, increasing gradient variance and making updates bounce with trajectory-level luck.

## Turn structure and semantic time

A lambda decay occurs per environment transition, so the same numeric lambda does not represent the same strategic horizon across games.

For example, with lambda `.8`, one full cycle in a 2-player alternating game retains `.8^2 = .64`, while one cycle in a 4-player game retains `.8^4 ~= .41`.

Reason in natural game units when possible: own turns, opponent-response cycles, tricks, rounds, scoring phases, or other semantic phases.

A useful alternative is a trace half-life `H`:

```text
lambda = 2 ** (-1 / H)
```

where `H` is the number of relevant transitions after which a delayed TD correction has half its original lambda weight.

## Relationship between the two lambdas

Do not force:

```text
value_lambda == gae_lambda
```

They solve different problems.

A useful prior for stochastic self-play is often:

```text
gae_lambda < value_lambda
```

because the critic benefits from long-range outcome information while the actor benefits from localized credit and protection from later sampled decisions.

The knobs are separate but coupled: `value_lambda` affects critic quality, which affects how low `gae_lambda` can safely be.

## Training-time intuition

Both lambdas may decrease as training matures, but for different reasons.

For value lambda, a weak critic makes bootstrapping unreliable, so `1.0` is a natural initial choice. Later, a good critic may justify lowering it to denoise Monte Carlo returns.

For GAE lambda, a randomly initialized critic may be too weak for zero-lambda advantages to carry useful long-range signal. Start higher if needed, then lower it as the critic becomes able to rank successor states and recognize strategic consequences earlier.

Do not schedule either lambda automatically merely because this direction is plausible. Prefer a fixed simple default unless measurements show a benefit.

## Diagnostics

For `value_lambda`, ask:

> At equal environment-sample budget, which target produces the best prediction of independent reality?

Useful checks include held-out Monte Carlo returns, repeated rollouts from the same state, value calibration, and final policy strength.

For `gae_lambda`, ask:

> How well does the critic rank the immediate successor states of alternative actions, and how many steps are needed before that ranking becomes correct?

When affordable, fork a state, estimate high-quality `Q(s,a)` values with repeated continuations, and compare them with one-step estimates such as `r + V(s')`. If immediate rankings are already good, prefer low GAE lambda. If several later TD corrections are consistently needed, raise it just enough to transmit those corrections.

## Compact procedure

When entering a new game:

1. Keep `discount/gamma` separate from this decision.
2. Set the initial value prior to `value_lambda = 1.0`.
3. Estimate reward-anchor spacing `D_v`; do not lower value lambda so far that real outcomes effectively disappear before reaching earlier states.
4. Estimate critic-recognition delay `D_a` from the game semantics and expected function-approximation difficulty.
5. Reduce the GAE estimate when many independent decisions intervene during `D_a`.
6. Prefer `gae_lambda < value_lambda` as a self-play prior, not a law.
7. Tune the two lambdas independently rather than performing a shared one-dimensional sweep.
8. Judge value lambda against independent reality, and GAE lambda by action-credit quality / downstream policy strength.

The shortest mental model is:

```text
value_lambda = how much should reality teach the critic instead of the critic teaching itself?
gae_lambda   = how much future reality is still needed, after consulting the critic, to judge this action?
```
