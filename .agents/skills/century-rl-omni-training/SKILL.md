---
name: century-rl-omni-training
description: Train, monitor, resume, and diagnose The Game Omni PPO models using the repository's trainers/thegame_omni_scratch.sh and trainers/thegame_three_phase.sh launchers. Use this skill for Omni or three-stage runs, Trackio inspection, checkpoint comparisons, metric interpretation, and variance-aware conclusions about score or learning progress.
---

# Century RL Omni Training

Use this skill when working on PPO training for `thegame,mode=omni`. The two
shell launchers are the source of truth: read them before changing a recipe.
`trainers/thegame_three_phase.sh` delegates shared training defaults to
`trainers/thegame_omni_scratch.sh`; it adds checkpoint handoffs and phase
schedules.

## Score anchors

Keep these project-level reference points in view:

- About **80 points** is reachable by a naive greedy strategy in Omni. Treat it
  as a useful baseline, not a hard game-theoretic ceiling.
- A trained model reached **94 points**, but only after roughly **2,600
  training steps**. This is evidence of attainable progress and training
  timescale, not proof that every 94-point result is reproducible or that 94 is
  the ceiling.
- A score difference is meaningful only after matching the checkpoint, code,
  evaluation protocol, and enough fresh games to separate signal from noise.

Do not turn either number into a conclusion from one noisy evaluation. A
single lucky batch can make a model look above 80; a single unlucky batch can
make a good model look below it.

## Launchers and defaults

Before launching, check CUDA visibility with `nvidia-smi` and PyTorch, confirm
the intended checkout, and inspect the current scripts for drift. The launchers
source `.env`, tee output to `training-logs/`, preserve the trainer exit code,
and enable unbuffered output plus expandable CUDA allocation by default.

### One-stage Omni

The normal scratch launch is:

```bash
TRACKIO=1 ./trainers/thegame_omni_scratch.sh
```

The current launcher defaults are approximately:

| Area | Default |
|---|---|
| game / model | `thegame,mode=omni` / `patchformer-medium-p8` |
| steps / seed | `2400` / `0` |
| rollout / evaluation | `128` / `512` games |
| evaluation / checkpoint cadence | every `50` / every `25` steps |
| inference / learner batch | `1024` / `384` |
| optimizer | AdamW, LR `0.0008`, betas `(.9, .95)`, eps `1e-5`, weight decay `.01` |
| PPO / returns | clip `.2`, discount `1.0`, GAE lambda `.1`, value lambda `.9` |
| exploration | thermostat, target perplexity `1.5`, entropy strength `.1`, baseline ratio `.05` |
| KL | target `.05`, strength `.05` |
| evaluation | temperature `.02` |
| warmup / LR floor | `20` / scale `1.0` |

Environment variables and trailing arguments override these values. Use a new
`TAG` and isolated checkpoint/log locations for each candidate. Keep the full
command, environment overrides, seed, architecture, and starting checkpoint
with the experiment record.

### Three stages

```bash
TOTAL_STEPS=2400 TRACKIO=1 ./trainers/thegame_three_phase.sh
```

The default budget is split by cumulative boundaries at 70% and 90%:

1. **Phase 1 (70%)**: Omni defaults, flat learning rate and exploration.
2. **Phase 2 (20%)**: resumes phase 1. With `EXPLORATION_CONTROLLER=linear`,
   the exploration strength moves to `0.1x` its initial value. With the default
   `thermostat`, the perplexity target continues instead; it is not silently
   replaced by linear entropy decay.
3. **Phase 3 (10%)**: resumes phase 2, holds exploration at its phase-2
   endpoint (or continues the thermostat), and decays the learning rate to
   zero.

The wrapper passes global schedule offsets so the phase boundaries are real
global steps. Resuming restores model, optimizer, and learner/controller state.
Use `PHASE2_RESET_EXPLORATION_STATE=1` only when intentionally starting a new
exploration-control state.

For a selected stage, set both bounds and provide or verify the handoff:

```bash
START_PHASE=2 END_PHASE=2 \
PHASE2_RESUME=checkpoints/three-phase/coop/thegame,mode=omni/patchformer-medium-p8/<phase1-tag>/step-<phase1-step>.pth \
TAG=<experiment>-phase2 \
./trainers/thegame_three_phase.sh
```

The wrapper creates `*-phase1`, `*-phase2`, and `*-phase3` identities. Keep
their Trackio runs and checkpoint directories separate; a phase is not
complete merely because the process started.

## What the metrics mean

Coop logs an evaluation at step zero, at `evaluation-every`, and at the end.
Training and game metrics are logged every five completed steps. The common
Trackio paths use slashes, for example `evaluation/points` and
`train/PolicyMetrics/perplexity`.

Prioritize metrics in this order:

| Metric | Interpretation | Decision use |
|---|---|---|
| `evaluation/points` | Mean final points across fresh evaluation games, using the configured low temperature (`.02` by default) | Primary learning metric; compare late, repeated evaluations |
| `evaluation/win_rate` | Self-vs-self win rate; usually not informative because both seats use the same policy | Sanity check only, not a quality ranking |
| `rollout/points` | Mean points from the sampled training rollouts | Behavior/data diagnostic; noisy and coupled to training sampling |
| `rollout/win_rate`, `rollout/games`, `rollout/samples` | Rollout composition and volume | Check that the run is producing the intended data |
| `train/PolicyMetrics/perplexity` | Effective action count, `exp(entropy)`; 1 is nearly deterministic and larger values are more diffuse | Check exploration against the thermostat or phase schedule |
| `train/PolicyMetrics/normalized_nucleus_size_threshold_0_95` | Normalized number of legal actions needed to cover 95% probability | Detect policy concentration/support changes; do not equate it with score |
| `train/ValueMetrics/pearson`, `explained_variance`, `mae` | Critic agreement and error against training targets | Diagnose value learning; not a direct game-strength score |
| `train/PolicyGradientLoss`, `importance_ratio`, `clip_fraction` | PPO objective and how often updates are clipped | Diagnose update regime; a lower loss is not automatically a stronger policy |
| `train/AdaptiveKLPenalty/kl`, `target`, `strength` | Policy drift from the rollout/reference policy and the adaptive penalty response | Detect unexpectedly large updates or a controller fighting the learner |
| `train/lr`, `train/gradient_norm`, `train/lr_scale` | Optimizer state and update scale | Confirm warmup, decay, clipping, and comparable update budgets |
| `game/avg_cost`, `ratio_lowest_cost`, `ten_rule_moves`, `plays_before_x`, `message_information` | The Game behavior: move efficiency, lowest-cost choice, 10-rule use, skipped plays, and message-state dependence | Explain *how* points changed and catch strategy collapse |

`evaluation/points` and `game/points` are averages. The current sink does not
preserve a per-game score distribution in the line metric, so Trackio alone is
not enough to calculate a reliable confidence interval. For a decisive claim,
rerun the checkpoint on a larger fresh evaluation and record mean, standard
deviation, game count, and evaluation temperature externally or in a dedicated
analysis.

## Trackio: read live, then verify

With `TRACKIO=1`, Coop initializes a run under project
`thegame,mode=omni` and uses `TAG` as the run name. Inspect the live dashboard
or query it without stopping training:

```bash
.venv/bin/trackio show --project 'thegame,mode=omni'
.venv/bin/trackio list runs --project 'thegame,mode=omni' --json
.venv/bin/trackio list metrics --project 'thegame,mode=omni' --run '<tag>' --json
.venv/bin/trackio get metric --project 'thegame,mode=omni' --run '<tag>' --metric evaluation/points --json
.venv/bin/trackio get snapshot --project 'thegame,mode=omni' --run '<tag>' --step 200 --json
```

If `TRACKIO_URL` is set, the launcher passes it as `--trackio-url`; otherwise
the trainer uses its normal local/default Trackio behavior. For a remote Space,
add the configured `--space` (and authentication when required) to read-only
`list`, `get`, or `query` commands. Use JSON output for comparisons and scripts.

Live Trackio is observability, not validation. First establish that the
process is healthy from the tee log, checkpoints, exit status, and GPU usage;
then judge learning with evaluation. A moving dashboard proves that metrics
are being logged, not that the model is improving.

### Do not mistake the 50-step cadence for a learning timescale

`evaluation-every=50` means “log another evaluation here.” It does **not** mean
that the score should improve within 50 optimizer iterations, and it does not
make 50 steps a valid plateau test. PPO learning is noisy and non-monotonic:
an improving run can be flat for several evaluations, dip, and then recover or
improve later. Expecting a monotonic point increase at every 50-step check is
the wrong acceptance criterion.

Treat one 50-step interval with no visible gain as **inconclusive**, not as
evidence that the run is stalled or that the recipe failed. Use a review horizon
long enough to contain multiple evaluation intervals and enough updates for
the controller and policy to move relative to the observed evaluation noise.
Choose that horizon from the run's variance and update scale; do not impose a
universal “must improve every N steps” rule. Smoothing is useful for seeing a
trend, but it must not replace fresh repeated evaluations or be used to hide
regressions.

## Variance: sources and guardrails

Assume every observed score contains noise. Separate these sources before
interpreting a curve:

1. **Finite evaluation games.** A mean over 512 games is still a sample. Decks,
   shuffles, initial states, seat rotation, and game trajectories vary. The
   configured `.02` evaluation temperature is very low but is still a sampling
   protocol, not necessarily literal argmax.
2. **Training-rollout sampling.** The default 128 rollout games produce noisy
   gradients and noisy `rollout/*` metrics. A rollout spike can be luck or a
   changed batch, not a durable policy improvement.
3. **Rollout count changes two things here.** `Learner(normalize_lr=False)` does
   not rescale the optimizer for a different number of batches. Changing
   `ROLLOUT_GAMES` changes both gradient noise *and* the number of samples/
   updates per step. It is not a clean variance-only experiment.
4. **Randomness in training.** The seed controls Python, PyTorch, CUDA, and
   the Cython game RNG, but GPU kernels and asynchronous batching can still
   matter. Data order, action sampling, augmentation, and self-play rotation
   all affect a trajectory.
5. **Checkpoint and time selection.** Evaluating many checkpoints and keeping
   the highest one creates winner's-curse bias: the selected maximum is partly
   the luckiest evaluation. Re-evaluate both the reported best checkpoint and
   the final checkpoint on fresh games.
6. **Recipe and state mismatch.** Architecture, current checkout, seed,
   starting checkpoint, optimizer state, learner/controller state, phase
   offsets, exploration controller, evaluation temperature, rollout count,
   and evaluation count can all change the result. A historical score from a
   different code version is not a matched control.
7. **Controller transients.** Perplexity, entropy strength, KL strength, and
   learning rate can be in warmup, decay, or adaptive recovery. A temporary
   score change during a controller transition is not automatically a new
   equilibrium.
8. **Metric mismatch.** `rollout/points`, `game/points`, and
   `evaluation/points` come from different data and policies. Train loss,
   perplexity, or value explained variance can improve while greedy game score
   does not.

Never draw a conclusion from one of these alone:

- one evaluation point;
- no improvement across one 50-step evaluation interval;
- a non-monotonic dip or rebound between adjacent evaluations;
- a rollout reward spike;
- the current best checkpoint selected from many evaluations;
- a live process, GPU utilization, or Trackio heartbeat;
- lower PPO/value loss without evaluation improvement;
- a change made together with rollout count, seed, architecture, or code.

## Decision protocol for experiments

For any claim that a recipe is better or that learning has plateaued:

1. **Freeze the comparison.** Record commit/checkout, full CLI and relevant
   environment overrides, architecture, checkpoint step, seed, rollout and
   evaluation games, `normalize_lr`, controller state, and evaluation
   temperature. Change one causal factor at a time.
2. **Check health separately.** Confirm logs advance, checkpoints are written,
   CUDA is active, and there are no NaN/OOM/traceback signs. Do not confuse
   “running” with “learning,” or “not improving” with “hung.”
3. **Compare matched evaluations.** Use the same current code, checkpoint
   architecture, protocol, and game count for control and candidate. Prefer
   repeated fresh evaluations or a larger evaluation budget. If a paired
   evaluator is available, reuse the same game randomness for both policies.
4. **Look for agreement, not monotonicity.** Over multiple evaluation
   intervals, check whether `evaluation/points` has a durable trend beyond its
   observed noise and whether behavior metrics, perplexity, KL, LR, and critic
   metrics provide a coherent mechanism. Do not require every adjacent point to
   rise, and do not let `rollout/points` overrule repeated greedy evaluation.
5. **Check late behavior.** Judge a mature checkpoint or a smoothed late
   window, not the first transient peak. For a plateau, repeat the same-code
   late evaluation before changing hyperparameters.
6. **Report uncertainty.** State game count, mean, spread or repeated-run
   range, checkpoint selection rule, and exactly what changed. Say “suggestive”
   when the evidence is noisy; reserve “better” for a matched, replicated
   result.

The practical target is durable improvement over the ~80-point naive-greedy
reference, with 94 at roughly step 2600 treated as an encouraging historical
milestone to reproduce—not a reason to chase every transient high score.

## Source files

When this skill is invoked, inspect these files in the active checkout before
making claims about defaults or metric names:

- `trainers/thegame_omni_scratch.sh`
- `trainers/thegame_three_phase.sh`
- `trainers/coop.py`
- `boardrl/metrics.py`
- `boardrl/games/thegame/metrics.py`
- `boardrl/training/learner.py`
- `boardrl/rl/model/loss.py`

Use the project Trackio skill for detailed CLI/API behavior and read-only
retrieval patterns. Do not delete Trackio runs or checkpoints from this
workflow without the exact target identity and explicit confirmation.
