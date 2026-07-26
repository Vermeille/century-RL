# The Game strict-mode RL recipe

This note records the reproducible search for a from-scratch reinforcement
learning recipe that matches the `lowest_cost` policy in strict two-player
mode. It is intentionally an experiment report rather than a claim based on a
single favorable checkpoint.

## Acceptance criterion

A successful recipe must satisfy all of the following:

- initialize a new model rather than fine-tune an existing checkpoint;
- train only from environment interaction and rewards;
- use the raw `display_with_moves()` observation (no engineered card costs);
- never query or imitate `lowest_cost` during training;
- score at least as well as `lowest_cost` under the same 1,000 seeded strict
  games, with the evaluation command below.

```bash
uv run python evaluate_thegame.py \
  --games 1000 \
  checkpoints/coop/thegame,mode=strict/shared-patch-small/<tag>/<checkpoint>.pth
```

The evaluator resets every RNG before each policy, so the model and baseline
see the same deck sequence. It reports the mean, standard deviation and a 95%
confidence interval in addition to diagnostic action metrics.

The measured baseline is 81.236 points over 1,000 games (standard deviation
11.700). A second seeded measurement produced 81.76, illustrating why the
acceptance run uses 1,000 games rather than the trainer's smaller monitoring
evaluations.

## Raw-observation architecture diagnosis

The first models were not merely suffering from noisy RL targets. A supervised
architecture gate trained them to predict `lowest_cost` from the same raw
character observations. These weights are diagnostic only and are never used
to initialize an RL run.

Strict-mode observations have median length 175, p99 length 271, and a median
of 15 legal moves. The original five-layer CNN has a theoretical radius of
about 72 characters. More importantly, its fixed-offset convolutions have to
bind variable-position pile values to each action. The original policy head
then compresses the state into one action-independent 64-dimensional summary.

The controlled gates separated local parsing, global communication, and the
action readout:

| Architecture | Fresh-state accuracy | Interpretation |
| --- | ---: | --- |
| five-layer CNN + pooled head | about 50% | local and pooled bottlenecks |
| five-layer CNN + raw cross-attention | about 54% | readout alone is insufficient |
| multiscale dilated CNN + pooled head | 65.7% at step 50 | global convolution helps |
| multiscale dilated CNN + cross-attention | 67.6% at step 50 | pooling is not the main remaining gap |
| patch transformer with no local CNN | about 13% | every `@` query is the same raw embedding |
| two-layer CNN + patch-transformer head | 81.9% at step 50 | two local layers are enough |
| shared patch-transformer backbone + small cross head | **87.0% at step 50** | strong, but the head still performs global retrieval |

That final cross-attention result did not fully isolate the backbone: both the
policy head and the value head still had learned global pooling. The cleaned
`shared-patch` architecture now has three explicit roles:

1. two local CNN layers bind each `@` marker to its card and pile text;
2. learned stride-four patches and the global transformer layers provide
   content-addressable global communication;
3. a token-local linear policy readout scores each encoded `@` position.

The critic reduces the encoded state with one fixed learned cross-attention
query, followed by normalization and a linear projection. This gives the
scalar value prediction a length-agnostic readout without allowing the critic
to alter policy representations during the forward pass. The historical
cross-attention and patch-transformer policy heads have been removed rather
than retained as production complexity.

A 15-step medium-scale control validated the attention-free construction. With
the inherited convolution blocks, its training-batch action accuracy reached
71.2%, compared with 69.6% for the old cross-attention readout under the same
protocol. Replacing the dilated, depthwise, expanded convolution block with a
single dense kernel-seven residual convolution raised accuracy to **77.5%**
and reduced cross-entropy from 0.740 to **0.577**. Runtime also fell slightly,
from 7:01 to 6:55 despite the dense convolution. The final fixed-deck
closed-loop score improved from 67.15 to **74.43**. Thus neither a powerful
policy head nor a complicated local encoder is needed.

The cleaned architecture family keeps the two-layer local parser and scales
only global reasoning capacity:

| Preset | Width | Global layers | Parameters |
| --- | ---: | ---: | ---: |
| `shared-patch-tiny` | 32 | 2 | 64k |
| `shared-patch-small` / `shared-patch` | 64 | 4 | 387k |
| `shared-patch-medium` | 128 | 4 | 1.52M |
| `shared-patch-large` | 256 | 6 | 8.26M |

All presets default to four-character patches. `--patch-size 8` or
`--patch-size 16` further reduces transformer sequence length for long
observations. The local stream and exact action positions remain
full-resolution; only the shared global stream is compressed. Checkpoints save
the selected patch size in the model specification.

The strongest historical shared-backbone PPO control used the now-legacy
cross-attention readout:

```bash
uv run python trainers/coop.py \
  --device cuda \
  --architecture shared-patch-small \
  --game thegame,mode=strict \
  --steps 800 \
  --rollout-games 256 \
  --evaluation-games 256 \
  --evaluation-every 25 \
  --save-every 25 \
  --inference-batch-size 512 \
  --learner-batch-size 512 \
  --learning-rate 0.0003 \
  --adam-beta1 0.5 \
  --epochs 1 \
  --discount 1.0 \
  --trace-decay 1.0 \
  --perplexity-start 0.8 \
  --perplexity-end 0.05 \
  --entropy-strength 0.1 \
  --kl-target 0.003 \
  --kl-strength 1.0 \
  --eval-temperature 0.02 \
  --warmup 15 \
  --min-lr-scale 0.3 \
  --seed 1 \
  --tag ppo-mc-shared-small-fast-s1 \
  --no-progress
```

## Controlled PPO results

All results in this section used a new initialization unless explicitly marked
diagnostic.

| Run | Relevant change | Best 256-game monitoring mean |
| --- | --- | ---: |
| conservative PPO | GAE `lambda=0.95`, four epochs, low LR | about 52 |
| Monte-Carlo PPO | `lambda=1`, one epoch, scheduled exploration | 76.37 |
| exploit diagnostic | fresh optimizer from the 76-point checkpoint | 78.25 |
| gentle exploit diagnostic | initialized from the prior diagnostic | 78.54 |
| shared patch, slow 2,000-step schedule | stopped at step 450 after plateau | 58.75 |
| shared patch, 800-step schedule | same PPO, faster exploration/LR annealing | 72.18 |

The two fine-tunes are diagnostics, not candidate recipes. Both regressed with
continued updates. They show that the network can represent a near-baseline
policy, while also showing that selecting a lucky fine-tuning checkpoint is not
a robust from-scratch recipe.

The 800-step schedule established that the original shared-patch result was
partly an optimization artifact. On the same 256 seeded decks, step 400 of the
slow schedule scored 58.75, while the faster schedule reached 72.81 at step
625. This was still below `lowest_cost` at 81.12. Continuing to force
normalized perplexity downward also hurt greedy evaluation: it peaked around
steps 575--625 and fell to about 67 late in training even as stochastic rollout
scores rose. Exploration annealing is therefore a real control knob, but
annealing all the way to 0.05 is not itself the missing recipe.

The independent 1,000-game means at seed 123 were 81.236 for `lowest_cost`,
76.618 for Monte-Carlo PPO, 77.219 for the first fine-tune and 77.171 for the
gentle fine-tune. Thus the smaller monitoring evaluations overstated the
fine-tuning gain. A 500-game temperature sweep from 0.001 through 0.1 changed
the best checkpoint by less than half a point and did not close the gap.

The dominant PPO issue is long-range credit assignment. Playing any legal card
immediately yields essentially the same reward; the quality of the choice is
only revealed when it preserves or destroys legal moves much later. Lowering
GAE lambda therefore removed useful signal. PPO also discards each rollout
after its update, which is expensive for this delayed effect.

An old repository commit labeled "gets 80" is not comparable evidence: it used
a message-enabled mode and an earlier game implementation whose termination
condition differed from current strict mode.

## Final from-scratch recipe

The cleaned local encoder, shared patch-transformer backbone, token-local
policy head, and learned-query critic reached the target with ordinary PPO:

```bash
uv run python trainers/coop.py \
  --device cuda \
  --architecture shared-patch-small \
  --game thegame,mode=strict \
  --steps 1000 \
  --schedule-steps 500 \
  --rollout-games 256 \
  --evaluation-games 256 \
  --evaluation-every 25 \
  --save-every 25 \
  --inference-batch-size 512 \
  --learner-batch-size 512 \
  --learning-rate 0.0003 \
  --adam-beta1 0.5 \
  --epochs 1 \
  --discount 1.0 \
  --trace-decay 1.0 \
  --perplexity-start 0.8 \
  --perplexity-end 0.10 \
  --entropy-strength 0.1 \
  --value-strength 1.0 \
  --kl-target 0.003 \
  --kl-strength 1.0 \
  --eval-temperature 0.02 \
  --warmup 15 \
  --min-lr-scale 1.0 \
  --seed 0 \
  --tag strict-small-query-entropy-sched500-s0 \
  --visdom-url https://visdom.vermeille.fr \
  --visdom-port 443 \
  --no-progress
```

This run began from a fresh random initialization and used neither imitation
nor heuristic features. On 1,000 strict games with seed 123, its final
checkpoint averaged **81.409** points (95% CI 80.778--82.040), while
`lowest_cost` averaged **81.236** (95% CI 80.511--81.961) on the same protocol.
The intervals overlap, so this establishes a match rather than statistically
significant superiority.

The learning-rate and exploration schedules finish at step 500 and then hold.
Monitoring first crossed 80 at step 700; the final checkpoint was stronger and
avoids checkpoint selection as part of the recipe. The next optimization goal
is therefore time-to-80: reduce model size, rollout count, and/or schedule
horizon while keeping the same fixed evaluation protocol.

## Efficiency controls

The 64k `shared-patch-tiny` preset was trained with the final recipe as a
one-variable scaling control. It ran more updates per hour but plateaued around
67--70 monitoring points from steps 375 through 750. It was stopped at step
765. Width 32 is therefore below the useful capacity threshold for this task
under the current optimizer.

A second control doubled rollout games from 256 to 512 and the live inference
batch cap from 512 to 1024, while halving total steps and the schedule horizon.
Learner minibatches remained 512. This preserves the total number of
environment games and approximately preserves the number of Adam minibatches.
It increased early interaction throughput by about 40%, though the advantage
shrank as trajectories grew longer. The best 256-game monitoring result
through step 475 was 78.38, below the original recipe at equal interaction
count.

The cause was the unit of PPO's trust region. The reference policy is frozen
once per outer rollout, and the adaptive KL controller holds the entire outer
update near its target. Doubling rollout size gives more minibatches inside
that same KL budget. Halving outer updates therefore reduces the total number
of allowed policy displacements even though samples and optimizer minibatches
are preserved.

Two controls isolated the adaptive controller's behavior. Raising the target
from 0.003 to 0.006 while retaining an initial strength of 1.0 did nothing:
the initial strength is also the controller's lower bound, so it could not
relax the penalty. Lowering the initial strength to 0.5 while retaining the
0.003 target only changed the transient; the controller raised the coefficient
toward the same equilibrium. Raising the target and lowering the floor together
was the effective intervention.

The resulting faster from-scratch recipe is:

```bash
uv run python trainers/coop.py \
  --device cuda \
  --architecture shared-patch-small \
  --game thegame,mode=strict \
  --steps 500 \
  --schedule-steps 250 \
  --rollout-games 512 \
  --evaluation-games 256 \
  --evaluation-every 25 \
  --save-every 25 \
  --inference-batch-size 1024 \
  --learner-batch-size 512 \
  --learning-rate 0.0003 \
  --adam-beta1 0.5 \
  --epochs 1 \
  --discount 1.0 \
  --trace-decay 1.0 \
  --perplexity-start 0.8 \
  --perplexity-end 0.10 \
  --entropy-strength 0.1 \
  --value-strength 1.0 \
  --kl-target 0.006 \
  --kl-strength 0.5 \
  --eval-temperature 0.02 \
  --warmup 8 \
  --min-lr-scale 1.0 \
  --seed 0 \
  --tag strict-small-rollout512-kl006-strength05-s0 \
  --visdom-url https://visdom.vermeille.fr \
  --visdom-port 443 \
  --no-progress
```

On the fixed 1,000-game seed-123 evaluation, the final step-500 checkpoint
scored **81.770** (95% CI 81.172--82.368), compared with **81.236** (95% CI
80.511--81.961) for `lowest_cost`. Step 475 scored 81.939, but the final
checkpoint is the recipe result so no favorable checkpoint selection is
required. Monitoring first crossed 80 at step 275 and the larger evaluation
gave step 325 a mean of 80.400. Wall-clock time to the first monitored
80-point checkpoint fell from about 3 hours 20 minutes to about 3 hours,
although equal total interaction and learner work meant the complete runs
still took roughly the same five and a half hours.

`PolicyGradientLoss` now reports the mean importance ratio and the fraction of
samples whose PPO surrogate is clipped. `coop.py` exposes the clipping radius
as `--ppo-clip`, making the next rollout-scaling control observable rather than
guesswork.

The doubled-rollout run was interrupted after step 490, then resumed from its
step-475 checkpoint. This exposed a checkpoint bug: model and optimizer state
were restored, but adaptive loss state was not. Entropy strength reset from
about 0.020 to 0.1, and resumed performance regressed. Consequently step 475 is
the valid endpoint of that control; the resumed step-500 checkpoint is not
comparable. `Learner.state_dict()` now owns its learning-rate normalization
baseline and the ordered state of every loss. Saving the learner therefore
restores scheduled-perplexity strength and EMA, adaptive-KL strength and
diagnostics, and any future stateful loss without trainer-specific checkpoint
wiring.
