# The Game RL Experiments

Objective: improve The Game score using generic RL improvements only. No hardcoded strategy logic, hints, or game-specific action preferences.

Evaluation principle: The Game is an endurance game. Longer rollouts and higher sample counts per epoch usually mean the policy is surviving longer, playing more cards, and scoring better. Treat them as a positive outcome signal unless checkpoint evaluation contradicts it. Throughput only measures experiment cost and update cadence; it is not, by itself, evidence of worse learning.

## Current Baseline

- User-reported current policy score: approximately 60 points, with evaluation taking hours.
- Best confirmed score from this investigation: `thegame-ckpt/rl-801.pth`, policy sampling temperature `0.1`, 512 self-play games, average `58.544921875`, min `38`, max `89`.
- Quick checkpoint sanity check, 8 games, `thegame,messages=True`, self-play with `policy_sampling`:
  - `thegame-ckpt/rl-0.pth`: average 15.375
  - `thegame-ckpt/rl-1000.pth`: average 55.875
  - Later checkpoints may need compatibility handling before evaluation because some saved CNN checkpoints have stale RMSNorm bias keys.
- Quick clean-environment evaluator, 8 games, `thegame`, self-play with `policy_sampling`:
  - `thegame-ckpt/rl-8000.pth`: average 77.5
- Architecture caveat discovered later: both strong 300-epoch clean scratch checkpoints (`/tmp/thegame-ppo-300` and `/tmp/thegame-ppo-300-fixed`) were trained with separate `policy_backbone` and `value_backbone` state-dict prefixes. The later GE4-400 and TD-MSE candidates were trained after reverting to a shared `backbone` and were weaker at epoch 100. This does not prove the separate backbone caused the score, but it means the shared-backbone speed revert is a real experimental confound, not a pure wall-clock-only change.

## Generic RL Fixes Applied

- `chunk()` no longer drops exact full batches.
- `main.py` now uses all rollout samples, including the final partial minibatch.
- `gradient_epochs` is now active only when every configured loss supports rollout reuse/off-policy correction.
- Strict on-policy losses ignore `gradient_epochs > 1` and perform one correctly weighted accumulated update over the rollout.
- PPO/SPO-style corrected losses can reuse rollout data for multiple shuffled minibatch epochs.
- Gradients are clipped before optimizer steps.
- Empty trainsets and traces with only terminal records are skipped safely.
- KL regularization is normalized by batch size, so changing minibatch size does not change KL strength.
- Old checkpoints with harmless stale keys can be loaded for evaluation/resume.
- If an old checkpoint's optimizer state no longer matches the current model, resume keeps the model and resets optimizer state.
- The LR schedule now actually decays to zero over the post-warmup horizon. Previously it divided decay progress by total iterations instead of remaining iterations, so LR stayed above zero at the configured final epoch.
- Resuming from a checkpoint with a different `train.iterations` now prints a warning because that changes the LR schedule and can create an LR jump.
- Added `train.print_histories` to keep Visdom metric logging while suppressing expensive/noisy terminal history dumps in long runs.

## Clean Config

Added `configs/thegame-ppo-clean.yaml`.

Design:

- Environment: `game: thegame`, intentionally no message-action branch.
- Algorithm: PPO-style policy gradient with normalized GAE.
- Auxiliary losses: entropy bonus, KL to reference, bootstrap value loss.
- Self-play: current policy vs current policy, rotated seats.
- Evaluation: current policy with low temperature vs random.

## Smoke Run

Command shape:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-clean.yaml \
  -x model=toy \
  -x train.iterations=1 \
  -x train.batch_size=8 \
  -x train.gradient_epochs=2 \
  -x self_play.num_games=2 \
  -x self_play.max_len=80 \
  -x pit.every=999 \
  -x pit.num_games=1 \
  -x pit.max_len=80 \
  -x train.show_every=1 \
  -x train.save_every=999 \
  -x visdom_url=offline
```

Result:

- Completed successfully from `/tmp` to avoid overwriting existing checkpoints.
- Collected 25 training samples.
- Ran two PPO gradient epochs over four minibatches each.

## Valid Scratch PPO Run

This run started from random initialization. It did not load any existing checkpoint.

Command shape:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-clean.yaml \
  -x train.iterations=50 \
  -x train.lr=0.0005 \
  -x train.batch_size=64 \
  -x train.gradient_epochs=4 \
  -x self_play.num_games=16 \
  -x self_play.max_len=500 \
  -x pit.every=99999 \
  -x pit.num_games=1 \
  -x pit.max_len=80 \
  -x train.show_every=10 \
  -x train.save_every=10 \
  -x visdom_url=offline
```

Notes:

- Ran from `/tmp/thegame-scratch`, so it did not overwrite repository checkpoints.
- `visdom_url=offline` was used only for this smoke scratch run to avoid depending on a local Visdom server. Future real runs should leave Visdom enabled unless it blocks execution.
- Scratch checkpoints were saved at epochs 0, 10, 20, 30, 40, and 50.
- Training sample counts rose from about 270 samples per epoch around epoch 9 to 440 samples at epoch 49, suggesting longer games from scratch.

Quick 8-game self-play evaluation:

- `rl-0.pth`: 16.875 average
- `rl-10.pth`: 18.125 average
- `rl-20.pth`: 24.5 average
- `rl-30.pth`: 28.375 average
- `rl-40.pth`: 28.875 average
- `rl-50.pth`: 27.375 average

Larger validation:

- `rl-40.pth`, 64 self-play games, temperature 1.0: 27.09375 average, min 17, max 38.

Interpretation:

- This is valid from-scratch learning, but not close to the user-reported approximately 60-point policy.
- The current clean PPO setup is therefore a correctness-improving baseline, not yet a competitive The Game training recipe.

## Longer Valid Scratch PPO Run

This run also started from random initialization. It did not load any existing checkpoint.

Command shape:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-clean.yaml \
  -x train.iterations=300 \
  -x train.lr=0.0005 \
  -x train.batch_size=64 \
  -x train.gradient_epochs=4 \
  -x self_play.num_games=24 \
  -x self_play.max_len=500 \
  -x pit.every=99999 \
  -x pit.num_games=1 \
  -x pit.max_len=80 \
  -x train.show_every=25 \
  -x train.save_every=50
```

Notes:

- Running from `/tmp/thegame-ppo-300`, so it does not overwrite repository checkpoints.
- Visdom was left at the normal CLI default for this run.
- Logging was reduced before this run: strategy outcome debug prints were removed and full move histories now print only on `show_every`.
- Training sample counts rose to about 600+ samples per 24-game epoch by epoch 50.

Epoch 50 evaluation:

- `rl-50.pth`, policy sampling, temperature 0.2, 64 self-play games: 31.5 average, min 20, max 46.
- `rl-50.pth`, argmax, 64 self-play games: 32.640625 average, min 17, max 50.
- Invalidated: `rl-50.pth`, gumbel one-step lookahead, `num_evals=8,q_scale=1.0`, 32 self-play games: 29.53125 average, min 21, max 42. This evaluation used `g.copy()` before the lookahead fix, so for The Game it could see the exact future deck order after candidate moves.

Interpretation:

- This is a valid improvement over the 50-epoch scratch smoke run's best 64-game result of 29.375 at temperature 0.2.
- The valid best score from this checkpoint is therefore argmax at 32.640625 over 64 games.
- Generic value-guided gumbel must be retested only after the randomized-copy fix. Do not use the invalidated number for score claims.
- Continue the run to at least checkpoint 100 before deciding whether the PPO recipe has plateaued.

Epoch 100 evaluation:

- `rl-100.pth`, policy sampling, temperature 0.2, 64 self-play games: 40.109375 average, min 23, max 61.
- `rl-100.pth`, argmax, 64 self-play games: 38.609375 average, min 27, max 55.

Interpretation:

- Checkpoint 100 is a substantial valid improvement over checkpoint 50.
- Low-temperature sampling currently beats argmax, so stochasticity is still useful.
- Continue to checkpoint 150 before changing the PPO recipe.

Epoch 150 evaluation:

- `rl-150.pth`, policy sampling, temperature 0.2, 64 self-play games: 43.984375 average, min 25, max 60.
- `rl-150.pth`, argmax, 64 self-play games: 44.578125 average, min 30, max 63.

Interpretation:

- Checkpoint 150 is a further valid improvement over checkpoint 100.
- Argmax has overtaken low-temperature sampling at this checkpoint.
- Continue to checkpoint 200 before changing the PPO recipe.

Epoch 200 evaluation:

- `rl-200.pth`, policy sampling, temperature 0.2, 64 self-play games: 48.640625 average, min 33, max 71.
- `rl-200.pth`, argmax, 64 self-play games: 50.34375 average, min 33, max 69.

Epoch 250 evaluation:

- `rl-250.pth`, policy sampling, temperature 0.2, 64 self-play games: 54.171875 average, min 33, max 69.
- `rl-250.pth`, argmax, 64 self-play games: 54.109375 average, min 39, max 67.

Epoch 300 evaluation:

- `rl-300.pth`, policy sampling, temperature 0.2, 64 self-play games: 54.484375 average, min 35, max 72.
- `rl-300.pth`, argmax, 64 self-play games: 54.625 average, min 41, max 68.

128-game decoder sweep for `rl-300.pth`:

- Policy sampling, temperature 0.05: 55.15625 average, min 37, max 81.
- Policy sampling, temperature 0.1: 56.046875 average, min 37, max 89.
- Policy sampling, temperature 0.2: 55.3046875 average, min 37, max 80.
- Argmax: 54.7109375 average, min 37, max 75.

Additional decoder checks for `rl-300.pth` while later training was running:

- Policy sampling, temperature 0.075, 256 games: 55.11328125 average, min 35, max 97.
- Policy sampling, temperature 0.125, 256 games: 55.84765625 average, min 35, max 77.

Interpretation:

- The current valid best from this scratch run is `rl-300.pth` with policy sampling at temperature 0.1: 56.046875 average over 128 games.
- The run improved strongly through epoch 250, then only marginally from 250 to 300.
- This is close to, but still below, the user-reported approximately 60-point policy.

## Continuation Past Epoch 300

The same from-scratch run was resumed from its own `rl-300.pth` checkpoint with optimizer state intact. This is a continuation of the valid scratch experiment, not an external warm start.

Command shape:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-clean.yaml \
  --ckpt /tmp/thegame-ppo-300/thegame-ckpt/rl-300.pth \
  -x train.iterations=600 \
  -x train.lr=0.0005 \
  -x train.batch_size=64 \
  -x train.gradient_epochs=4 \
  -x self_play.num_games=24 \
  -x self_play.max_len=500 \
  -x pit.every=99999 \
  -x pit.num_games=1 \
  -x pit.max_len=80 \
  -x train.show_every=25 \
  -x train.save_every=50
```

Notes:

- Visdom was left at the normal CLI default.
- The run was stopped manually after evaluating epoch 350 because the continuation had degraded.
- `rl-359.pth` exists because the trainer saves on interrupt, but it is not a candidate checkpoint.
- Caveat: this is not equivalent to training a 600-epoch schedule from scratch. The original 300-epoch run's linear LR schedule had already decayed to zero by epoch 300; resuming with `train.iterations=600` made the scheduler jump LR back up at epoch 300.

Evaluation:

- `rl-250.pth`, policy sampling, temperature 0.1, 128 games: 53.84375 average, min 37, max 76.
- `rl-350.pth`, policy sampling, temperature 0.1, 128 games: 54.484375 average, min 32, max 85.
- `rl-350.pth`, argmax, 128 games: 52.921875 average, min 34, max 84.
- `rl-359.pth`, policy sampling, temperature 0.1, 64 games: 53.4375 average, min 37, max 83.

Interpretation:

- Continuing this recipe past epoch 300 made the direct policy worse.
- The next improvement should not be "resume the same checkpoint with a longer LR horizon" unless the LR schedule is handled deliberately.
- A real longer-run test must start from scratch with the longer horizon configured from epoch 0, or use a resume rule that does not change the LR schedule discontinuously.
- After this finding, the LR scheduler was fixed. Future scratch runs are not directly apples-to-apples with the earlier run because the end-of-run LR behavior is now different.

## Fixed-Scheduler 600-Horizon Scratch Run

This was the first clean longer-horizon run after fixing the LR scheduler. It started from random initialization, not from a checkpoint.

Command shape:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-clean.yaml \
  -x train.iterations=600 \
  -x train.lr=0.0005 \
  -x train.batch_size=64 \
  -x train.gradient_epochs=4 \
  -x self_play.num_games=24 \
  -x self_play.max_len=500 \
  -x pit.every=99999 \
  -x pit.num_games=1 \
  -x pit.max_len=80 \
  -x train.show_every=25 \
  -x train.save_every=100
```

Notes:

- Visdom was left at the normal CLI default.
- The run reached `rl-119.pth` and was stopped manually after the epoch-100 evaluation.
- This run is valid as a scratch run, but not a candidate.

Evaluation:

- `rl-100.pth`, policy sampling, temperature 0.1, 128 games: 31.6953125 average, min 20, max 47.
- `rl-100.pth`, argmax, 64 games: 32.46875 average, min 21, max 47.

Interpretation:

- This is far behind the earlier 300-epoch scratch run at epoch 100.
- The result does not isolate the cause. The scheduler fix, the longer horizon, and training variance are still entangled.
- The next ablation is to run the same 300-epoch recipe again with the fixed scheduler and compare epoch 100.

## Fixed-Scheduler 300-Horizon Scratch Ablation

This run repeats the earlier 300-epoch recipe from scratch after the LR scheduler fix.

Command shape:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-clean.yaml \
  -x train.iterations=300 \
  -x train.lr=0.0005 \
  -x train.batch_size=64 \
  -x train.gradient_epochs=4 \
  -x self_play.num_games=24 \
  -x self_play.max_len=500 \
  -x pit.every=99999 \
  -x pit.num_games=1 \
  -x pit.max_len=80 \
  -x train.show_every=25 \
  -x train.save_every=100
```

Partial evaluation:

- `rl-100.pth`, policy sampling, temperature 0.2, 128 games: 37.140625 average, min 23, max 55.
- `rl-100.pth`, policy sampling, temperature 0.1, 128 games: 37.1171875 average, min 21, max 55.
- `rl-100.pth`, argmax, 64 games: 37.078125 average, min 21, max 60.
- `rl-200.pth`, policy sampling, temperature 0.2, 128 games: 49.953125 average, min 30, max 81.
- `rl-200.pth`, policy sampling, temperature 0.1, 128 games: 49.4921875 average, min 31, max 77.
- `rl-200.pth`, argmax, 64 games: 50.03125 average, min 36, max 67.
- `rl-300.pth`, policy sampling, temperature 0.05, 128 games: 55.0546875 average, min 35, max 91.
- `rl-300.pth`, policy sampling, temperature 0.1, 128 games: 55.359375 average, min 41, max 78.
- `rl-300.pth`, policy sampling, temperature 0.2, 128 games: 56.0 average, min 37, max 79.
- `rl-300.pth`, argmax, 128 games: 54.7890625 average, min 37, max 77.
- Larger head-to-head: `rl-300.pth`, policy sampling, temperature 0.2, 512 games: 54.494140625 average, min 37, max 83.
- Larger head-to-head current-best baseline: `/tmp/thegame-ppo-300/thegame-ckpt/rl-300.pth`, policy sampling, temperature 0.1, 512 games: 55.759765625 average, min 35, max 83.

Interpretation:

- This is below the earlier pre-fix 300-epoch scratch run at epoch 100, which scored 40.109375 with temperature 0.2.
- It is much better than the failed 600-horizon scratch run at epoch 100, which scored 31.6953125 with temperature 0.1.
- By epoch 200, this run has caught up to the earlier scratch run's epoch-200 range.
- The fixed scheduler may have shifted the early learning curve downward, but the catastrophic 600-horizon result is not explained by the scheduler fix alone.
- The 128-game final sweep was close to the current best, but a 512-game head-to-head favors the earlier checkpoint.
- This run is not a new best. The current best remains `/tmp/thegame-ppo-300/thegame-ckpt/rl-300.pth` with policy sampling at temperature 0.1.

## LR-Floor 300-Horizon Scratch Run

This run keeps the fixed scheduler semantics but adds an explicit LR floor instead of relying on the old denominator bug.

Config:

- `configs/thegame-ppo-lrfloor.yaml`
- `lr: 0.0005`
- `lr_min_scale: 0.05`
- `iterations: 300`
- `save_every: 100`
- self-play and PPO losses match the clean 300-epoch recipe.

Partial evaluation:

- `rl-100.pth`, policy sampling, temperature 0.2, 128 games: 40.1640625 average, min 25, max 79.
- `rl-200.pth`, policy sampling, temperature 0.2, 128 games: 43.9140625 average, min 21, max 62.
- `rl-200.pth`, policy sampling, temperature 0.1, 128 games: 44.0 average, min 29, max 73.
- `rl-200.pth`, argmax, 128 games: 45.546875 average, min 29, max 69.

Interpretation:

- This matches or slightly beats the earlier pre-fix 300-epoch run at epoch 100, which scored 40.109375 with temperature 0.2.
- It clearly beats the fixed zero-floor 300-epoch run at epoch 100, which scored 37.140625 with temperature 0.2.
- The epoch-200 score is far behind both the earlier pre-fix 300-epoch run and the fixed zero-floor 300-epoch run.
- Decoder choice does not rescue the checkpoint.
- The run was stopped manually after the epoch-200 evaluation.
- `lr_min_scale: 0.05` helped early learning but destabilized or derailed mid-training for this recipe.
- The current best remains `/tmp/thegame-ppo-300/thegame-ckpt/rl-300.pth` with policy sampling at temperature 0.1.

## Gated-CNN Scratch Probe

This run tested whether a larger generic gated CNN would improve the policy/value representation without changing game logic.

Config:

- `configs/thegame-ppo-gated.yaml`
- `model: minimal-gated-cnn`
- `seed: 0`
- PPO/self-play settings match the clean 300-epoch recipe.

Result:

- The run was stopped manually around epoch 24 before a meaningful evaluation checkpoint.
- Throughput was roughly 93-98 samples/s, compared with roughly 320+ samples/s for the small CNN recipe.
- The model had about 1.0M parameters, versus about 0.28M for the CNN baseline.

Interpretation:

- This is not a failed learning result; it is a cost result.
- The gated model is too expensive for quick iteration under the current CPU-only setup.
- Current optimization effort should stay on the cheaper CNN recipe unless a later architecture test is strongly motivated.

## Linear Entropy-Decay Scratch Run

This run tests whether fixed entropy regularization is too high late in training.

Code/config:

- Added generic `linear_entropy_bonus,start=...,end=...`.
- Added `configs/thegame-ppo-entropydecay.yaml`.
- The only intended recipe change versus the clean 300-epoch PPO setup is:
  - replace `entropy_bonus,strength=0.01`
  - with `linear_entropy_bonus,start=0.01,end=0.001`

Run:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-entropydecay.yaml
```

Notes:

- Running from `/tmp/thegame-ppo-entropydecay`.
- Visdom was left at the normal CLI default.
- Startup succeeded and used the intended losses.
- Initial throughput was about 328 samples/s, matching the cheap CNN baseline class.
- First real gate should be `rl-100.pth`, evaluated against:
  - earlier pre-fix clean epoch 100, temp 0.2: 40.109375 over 64 games
  - fixed-scheduler clean epoch 100, temp 0.2: 37.140625 over 128 games
  - LR-floor epoch 100, temp 0.2: 40.1640625 over 128 games

Epoch 100 evaluation:

- `rl-100.pth`, policy sampling, temperature 0.2, 128 games: 37.390625 average, min 21, max 53.
- `rl-100.pth`, policy sampling, temperature 0.1, 128 games: 37.4140625 average, min 26, max 51.
- `rl-100.pth`, argmax, 128 games: 38.7265625 average, min 23, max 57.

Interpretation:

- This is below the LR-floor epoch-100 result and below the earlier pre-fix clean epoch-100 result.
- It is close to the fixed-scheduler clean epoch-100 result, which later recovered by epoch 300.
- Continue to epoch 200 before stopping; the epoch-100 result alone is not enough to reject this schedule.

Epoch 200 evaluation:

- `rl-200.pth`, policy sampling, temperature 0.2, 128 games: 48.859375 average, min 32, max 74.
- `rl-200.pth`, policy sampling, temperature 0.1, 128 games: 49.0703125 average, min 36, max 73.
- `rl-200.pth`, argmax, 128 games: 48.1875 average, min 34, max 77.

Interpretation:

- This is behind the fixed-scheduler clean epoch-200 result: temp 0.2 scored 49.953125 and argmax scored 50.03125.
- It is also behind the earlier pre-fix clean epoch-200 argmax result of 50.34375.
- It is not far enough behind to reject before the final checkpoint, because both clean runs improved substantially from epoch 200 to epoch 300.
- Continue to epoch 300 and run a final decoder sweep before deciding.

Status:

- The run completed to `rl-300.pth`.

Epoch 300 evaluation:

- `rl-300.pth`, policy sampling, temperature 0.05, 128 games: 54.6171875 average, min 37, max 75.
- `rl-300.pth`, policy sampling, temperature 0.1, 128 games: 54.0390625 average, min 39, max 74.
- `rl-300.pth`, policy sampling, temperature 0.2, 128 games: 55.4453125 average, min 39, max 81.
- `rl-300.pth`, argmax, 128 games: 54.90625 average, min 41, max 83.
- `rl-300.pth`, policy sampling, temperature 0.2, 512 games: 54.802734375 average, min 35, max 81.

Interpretation:

- The best 128-game decoder is temperature 0.2, but it does not beat the current valid best checkpoint's 128-game result of 56.046875.
- The 512-game confirmation is also below the current valid best checkpoint's 512-game result of 55.759765625.
- Linear entropy decay is therefore not a new best. The current best remains `/tmp/thegame-ppo-300/thegame-ckpt/rl-300.pth` with policy sampling at temperature 0.1.

## Low-Temperature Self-Play Candidate

Motivation:

- The clean recipe trains self-play with default `policy_sampling` temperature `1.0`.
- Evaluations consistently prefer lower-temperature decoding, usually `0.1` or `0.2`.
- A generic low-temperature self-play run tests whether the behavior policy should explore less during rollout collection, without adding any game-specific logic.

Config:

- Added `configs/thegame-ppo-lowtemp.yaml`.
- Same small CNN and PPO setup as the clean 300-epoch recipe.
- Self-play strategies changed from:
  - `policy_sampling,model=this`
  - to `policy_sampling,model=this,temperature=0.2`

Planned gate:

- Launch only after the active entropy-decay run finishes or is stopped.
- Compare `rl-100.pth` against the same epoch-100 gates used above.

Run:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-lowtemp.yaml
```

Notes:

- Running from `/tmp/thegame-ppo-lowtemp`.
- Visdom was left at the normal CLI default.
- Startup succeeded with the intended PPO losses.
- Self-play uses `policy_sampling,model=this,temperature=0.2` for both seats.
- Initial throughput was about 320 samples/s.
- Active session id at launch: `22559`.
- Stopped manually during epoch 243 after evaluation showed the run was not competitive.
- Throughput collapsed to about 60-70 samples/s late in the run because the bad policy produced longer, slower episodes.

Gate results:

- `rl-100.pth`, 128 games:
  - policy sampling temperature `0.2`: average `22.65625`, min `13`, max `39`.
  - policy sampling temperature `0.1`: average `22.65625`, min `11`, max `35`.
  - argmax: average `22.6953125`, min `12`, max `40`.
- `rl-200.pth`, 128 games:
  - policy sampling temperature `0.2`: average `22.0`, min `14`, max `33`.
  - policy sampling temperature `0.1`: average `23.5234375`, min `13`, max `36`.
  - argmax: average `23.2109375`, min `13`, max `43`.

Conclusion:

- Rejected.
- This was worse than the clean PPO baselines by epoch 100 and epoch 200.
- Low-temperature decoding at evaluation time is useful, but using the same low temperature as the rollout behavior policy appears to collapse exploration too early. The policy then trains on narrow, low-quality trajectories and does not recover.

## Lookahead Validity

- Do not count any lookahead result unless the copied game state cannot reveal hidden future randomness.
- For The Game, this means candidate successor states must randomize the unrevealed deck before evaluating the post-move state. Otherwise the evaluator sees exact future draw order, which is cheating.
- The `gumbel` strategy now calls `copy(randomize=True)` when the game exposes that argument. The helper checks the method signature instead of catching `TypeError`, so real copy bugs are not silently hidden.
- Even with randomized hidden future state, `gumbel` is not currently part of the valid best-score path. It uses a successor-state value estimate from whatever player is current after the candidate move, so its value semantics are not clean enough to use as a headline score without further design work.

## Invalidated Resume Smoke From Existing Strong Checkpoint

This section is retained only as an evaluator/resume compatibility check. It is invalid for the objective because it starts from a learned checkpoint instead of training from scratch.

Command shape:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-clean.yaml \
  --ckpt /home/vermeille/century-RL/thegame-ckpt/rl-8000.pth \
  -x train.iterations=8001 \
  -x train.batch_size=8 \
  -x train.gradient_epochs=1 \
  -x self_play.num_games=2 \
  -x self_play.max_len=80 \
  -x pit.every=999 \
  -x pit.num_games=1 \
  -x pit.max_len=80 \
  -x train.show_every=1 \
  -x train.save_every=999 \
  -x visdom_url=offline
```

Result:

- Completed successfully from `/tmp`.
- Loaded `rl-8000.pth` after dropping 10 stale checkpoint keys.
- Optimizer state was reset because old optimizer parameter groups no longer match.
- Collected 160 samples.
- Ran one PPO gradient epoch over 20 minibatches.
- Saved `/tmp/thegame-ckpt/rl-8001.pth`.
- Quick 8-game clean-environment evaluator: average 79.75. This is only a smoke signal; the sample is too small and stochastic to claim a real improvement.
- Invalid for score chasing under the no-cheating constraint.

## Evaluation Tool

Added:

```bash
uv run python scripts/evaluate_thegame.py CHECKPOINT --game thegame --games 8 --max-len 500 --opponent self
```

The evaluator uses the learned policy through the normal strategy/model path. It does not add game-specific action logic.

## Temperature 0.5 Self-Play Candidate

Motivation:

- Temperature `0.2` self-play failed badly, probably by collapsing rollout exploration too early.
- Clean self-play uses default sampling temperature `1.0`.
- Temperature `0.5` is a generic middle point: less noisy than default behavior sampling, but not as deterministic as the rejected `0.2` run.

Config:

- Added `configs/thegame-ppo-temp05.yaml`.
- Same small CNN and PPO setup as the clean 300-epoch gated recipe.
- Self-play strategies use `policy_sampling,model=this,temperature=0.5` for both seats.
- Training is from scratch in `/tmp/thegame-ppo-temp05`.
- Visdom was left at the normal CLI default.

Run:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-temp05.yaml
```

Status:

- Session id: `11362`.
- Startup succeeded.
- Early throughput is roughly `175-190` samples/s.
- By epoch 39, throughput had declined to roughly `120-150` samples/s.
- At `2026-07-04 14:56 CEST`, only `rl-0.pth` existed; `rl-100.pth` was not available yet.
- By epoch 60, throughput had collapsed to about `60` samples/s.
- Stopped manually during epoch 61 before the first gate.
- No score is claimed for this run because no useful checkpoint was produced.
- Marked inconclusive, not rejected by score. In The Game, longer trajectories can be a positive signal because surviving longer means playing more cards and usually scoring better.
- The stop was a compute-management decision made before the first checkpoint, so it should not be used as evidence that rollout temperature `0.5` learns badly.

Conclusion:

- Keep low temperature for evaluation/decoding only based on evaluated scores.
- Do not judge rollout/self-play temperature by throughput alone. Judge it by checkpoint evaluation; throughput is only a compute-cost signal.

## Gradient Epochs 2 Candidate

Motivation:

- The clean PPO recipe reuses each rollout for `4` gradient epochs.
- PPO can tolerate limited rollout reuse, but less reuse should reduce stale-batch pressure and directly addresses the concern that old batches make gradients less faithful.
- This is a generic PPO/data-freshness knob, not game-specific logic.

Config:

- Added `configs/thegame-ppo-ge2.yaml`.
- Same small CNN, default rollout sampling, learning rate, loss mix, and 300-epoch gated recipe as the clean runs.
- Changed only `train.gradient_epochs` from `4` to `2`.
- Set `train.print_histories: false` for future resumes/restarts; this does not affect the already-running session because it loaded the previous config.
- Training is from scratch in `/tmp/thegame-ppo-ge2`.
- Visdom was left at the normal CLI default.

Run:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-ge2.yaml
```

Status:

- Active session id: `82113`.
- Startup succeeded.
- Early throughput is roughly `120-165` samples/s, so it is not obviously faster in wall-clock time than the clean run under current CPU conditions.
- Do not stop this run merely because episodes get longer or throughput falls; in this endurance game that may mean better play. Stop only on a checkpoint score gate, instability, or impractical runtime.
- First gate `rl-100.pth`, 128 games:
  - policy sampling temperature `0.2`: average `33.421875`, min `18`, max `47`.
  - policy sampling temperature `0.1`: average `33.765625`, min `21`, max `49`.
  - argmax: average `33.4765625`, min `19`, max `61`.
- This is behind the clean epoch-100 gates, so GE2 is not an early winner.
- Continuing to `rl-200.pth` because the intended tradeoff is less stale reuse but potentially slower learning; epoch 100 alone does not fully test that.
- At `2026-07-04 15:29 CEST`, the run was still active around epoch `148`; `rl-200.pth` was not available yet.
- At `2026-07-04 15:33 CEST`, the run was still active around epoch `163`; `rl-200.pth` was not available yet.
- At `2026-07-04 15:34 CEST`, the run was still active around epoch `169`; `rl-200.pth` was not available yet.
- At `2026-07-04 15:35 CEST`, the run was still active around epoch `174`; `rl-200.pth` was not available yet.
- At `2026-07-04 15:36 CEST`, the run was still active around epoch `180`; `rl-200.pth` was not available yet.
- At `2026-07-04 15:38 CEST`, the run was still active around epoch `190`; `rl-200.pth` was not available yet.
- At `2026-07-04 15:39 CEST`, the run was still active around epoch `195`; `rl-200.pth` was not available yet.
- At `2026-07-04 15:40 CEST`, `rl-200.pth` became available and was evaluated over 128 games:
  - policy sampling temperature `0.2`: average `39.09375`, min `25`, max `57`.
  - policy sampling temperature `0.1`: average `38.140625`, min `23`, max `57`.
  - argmax: average `38.5390625`, min `23`, max `56`.
- Rejected by score. This is far below the clean epoch-200 baseline around `49-50`, so `gradient_epochs=2` is not a good tradeoff for this recipe.
- The run was stopped during epoch `203` after the rejection decision.

## Gradient Epochs 3 Candidate

Motivation:

- `gradient_epochs=4` is the current best recipe but reuses each PPO rollout more.
- `gradient_epochs=2` is fresher but appears slower at the first gate.
- `gradient_epochs=3` is the direct middle point: it tests whether the best tradeoff is less stale than the clean recipe without cutting update work as aggressively as GE2.

Config:

- Added `configs/thegame-ppo-ge3.yaml`.
- Same small CNN, default rollout sampling, learning rate, loss mix, and 300-epoch recipe as the clean/GE2 runs.
- Changed `train.gradient_epochs` to `3`.
- Set `train.save_every` to `50` to expose earlier gates for experiment management.
- Set `train.print_histories: false`; Visdom metrics still run, but full terminal game-history dumps are suppressed.

Validation:

- Zero-iteration startup check passed:

```bash
uv run python main.py configs/thegame-ppo-ge3.yaml \
  -x train.iterations=0 \
  -x pit.every=99999 \
  -x self_play.num_games=1 \
  -x train.save_every=99999
```

Status:

- Started from scratch in `/tmp/thegame-ppo-ge3`.
- Active session id: `2566`.
- Initial startup succeeded; `rl-0.pth` should be written at epoch 0.
- At `2026-07-04 15:43 CEST`, the run was active around epoch `11`; only `rl-0.pth` was available.
- First useful gate is `rl-50.pth` because this config uses `train.save_every: 50`.
- Do not claim any score until checkpoints are evaluated.

Run:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-ge3.yaml
```

Result:

- The run reached `rl-300.pth`.
- Final checkpoint `rl-300.pth`, 128-game evaluation:
  - policy sampling temperature `0.1`: average `44.421875`, min `29`, max `63`.
  - policy sampling temperature `0.2`: average `46.1328125`, min `29`, max `70`.
  - argmax: average `44.6171875`, min `28`, max `65`.
- Rejected by score. It is below the best valid scratch result from `/tmp/thegame-ppo-300/thegame-ckpt/rl-300.pth`, which reached about `56` average.
- Longer rollout sample counts during training were not treated as a negative; for The Game they usually mean better survival. The rejection is based on evaluated point score.

## RL-8000 Checkpoint Recipe Candidate

Motivation:

- Existing checkpoint `thegame-ckpt/rl-8000.pth` has a strong score and stores its training config.
- The checkpoint weights themselves are not a valid starting point for this objective, but the embedded config is useful experimental evidence.

Embedded config differences:

- `model: cnn`, with current `model-configs/cnn.yaml` already matching `num_layers: 5`, `dim: 64`.
- `train.gradient_epochs: 1`, avoiding repeated optimization of stale PPO batches.
- `train.lr: 0.0005`.
- `train.discount_factor: 1.0`.
- `train.gae_lambda: 0.98`.
- `train.reward_rescale: 0.1`.
- `self_play.num_games: 32`.
- Losses:
  - PPO policy gradient with normalized GAE and clip `0.2`.
  - `scheduled_perplexity,start=0.4,end=-0.05,adaptation_rate=2`.
  - bootstrap MSE strength `0.1`.
- No KL penalty in that checkpoint recipe.
- The checkpoint used `game: thegame,messages=True`; the scratch candidate keeps `game: thegame` for comparability with current evaluations because `messages=True` changes the legal end-turn actions and observation text.

Config:

- Added `configs/thegame-ppo-rl8000-recipe.yaml`.
- This is a scratch config based on the embedded hyperparameters, not a resume from the checkpoint.
- `pit` is mostly disabled for training speed, Visdom remains enabled by default, and `print_histories` is disabled to avoid terminal spam.

Validation:

- Initial zero-iteration smoke test exposed a small bug in scheduled-loss progress handling: `training_state["progress"]` divided by `train.iterations`, so `train.iterations=0` crashed.
- Fixed progress calculation in `main.py` by clamping through `Trainer._training_progress()`.
- `python3 -m py_compile main.py` passed.
- `uv run pytest tests/test_optimizer_schedule.py tests/test_policy_gradient_loss.py -q` passed: `9 passed`.
- Zero-iteration startup check passed:

```bash
uv run python main.py configs/thegame-ppo-rl8000-recipe.yaml \
  -x train.iterations=0 \
  -x pit.every=99999 \
  -x self_play.num_games=1 \
  -x train.save_every=99999
```

Run:

```bash
uv run --project /home/vermeille/century-RL python /home/vermeille/century-RL/main.py \
  /home/vermeille/century-RL/configs/thegame-ppo-rl8000-recipe.yaml
```

Status:

- Started from scratch in `/tmp/thegame-ppo-rl8000-recipe`.
- Active session id: `48545`.
- Initial startup succeeded.
- Early epochs run around `300+` samples/s because the recipe uses `gradient_epochs: 1`.
- First real evaluation gate is `rl-100.pth`.
- `rl-100.pth` became available and was evaluated over 128 games:
  - policy sampling temperature `0.2`: average `33.4296875`, min `22`, max `52`.
  - policy sampling temperature `0.1`: average `34.015625`, min `23`, max `47`.
  - argmax: average `33.2890625`, min `21`, max `49`.
- This is below the clean epoch-100 baseline around `40`, so it is not an early winner.
- Continuing to `rl-200.pth` because this recipe came from an 8000-epoch checkpoint and uses only one gradient epoch, so slower score development is plausible. Do not infer rejection from throughput or rollout length; use evaluated score gates.
- The Visdom/loss behavior looked weaker than the clean PPO family, and the score evidence supported that read.
- The run was interrupted around epoch `165`; `rl-165.pth` was saved and evaluated over 128 games:
  - policy sampling temperature `0.2`: average `43.96875`, min `28`, max `65`.
  - policy sampling temperature `0.1`: average `44.7421875`, min `27`, max `61`.
  - argmax: average `44.328125`, min `27`, max `68`.
- Rejected by score and training behavior. It improved over `rl-100`, but it remains below the clean PPO epoch-200 baseline around `49-50` and far below the best valid scratch result around `56`.
- Interpretation: copying the old checkpoint recipe as metadata was useful, but the scheduled-perplexity/no-KL/one-gradient-epoch combination appears weaker for the current `thegame` environment than fixed entropy plus KL PPO.

## GE4 400-Epoch Candidate

Motivation:

- The strongest valid scratch result so far came from the fixed entropy + KL PPO family with `gradient_epochs=4`, `lr=0.0005`, 24 self-play games, and 300 epochs.
- GE2, GE3, entropy decay, low-temperature self-play, LR-floor, and the `rl-8000` scheduled-perplexity recipe were weaker.
- This candidate returns to the known-good family and gives it a 400-epoch horizon so it can reach the old strong 300-epoch region with learning rate still available afterward.
- Before launching this candidate, the model was reverted from separate policy/value backbones back to one shared backbone feeding both heads. The separate-backbone design roughly doubled encoder compute and wall-clock cost without evidence that it improved The Game results.

Config:

- Added `configs/thegame-ppo-ge4-400.yaml`.
- Uses fixed entropy bonus `0.01`, KL `0.1`, bootstrap MSE `0.1`, PPO normalized GAE, `gradient_epochs=4`.
- Uses `self_play.num_games=24`, `batch_size=64`, `lr=0.0005`, `warmup_epochs=15`, `iterations=400`, `save_every=50`.
- Pit is mostly disabled for speed; Visdom remains enabled by default; terminal history printing is disabled.

Shared-backbone validation:

- `python3 -m py_compile boardrl/rl/model/model.py` passed.
- `uv run pytest tests/test_config.py tests/test_modelpool.py -q` passed: `7 passed`.
- Checkpoint compatibility smoke loaded both a recent dual-backbone checkpoint and old `thegame-ckpt/rl-8000.pth`.
- Current shared CNN model has `142530` parameters, down from about `276610` for the separate-backbone variant.

Run:

- Started from scratch in `/tmp/thegame-ppo-ge4-400`.
- Startup reported `#parameters 0.14253 M`, confirming the shared-backbone revert is active.
- Early CPU throughput was roughly `585-610` samples/s before rollout lengths grew.
- `rl-50.pth`, 128 games:
  - policy sampling temperature `0.2`: average `29.25`, min `17`, max `52`.
  - policy sampling temperature `0.1`: average `29.21875`, min `17`, max `48`.
  - argmax: average `30.3046875`, min `17`, max `49`.
- `rl-100.pth`, 128 games:
  - policy sampling temperature `0.2`: average `37.5078125`, min `23`, max `53`.
  - policy sampling temperature `0.1`: average `37.234375`, min `21`, max `53`.
  - argmax: average `38.4296875`, min `26`, max `58`.
- Rejected as an optimization axis for now. Epoch 100 is roughly in the fixed-clean epoch-100 range, behind the stronger clean epoch-100 gate around `40`, and not enough to justify spending more CPU while more structural critic experiments are available.
- The run was interrupted after epoch `116`; no score beyond `rl-100.pth` was used for decisions.

## TD(lambda) Critic MSE Candidate

Motivation:

- `bootstrap_mse_loss` is historically named as MSE but currently trains the value distribution by log-prob of a clipped TD(lambda) target.
- That behavior may be intentional for older configs, so it was not changed in place.
- Added a new explicit loss, `bootstrap_value_mse_loss`, that fits `pred_value.mean` directly to `sample.td_lambda`.
- This is a generic critic/GAE improvement attempt, not The Game-specific logic.

Config:

- Added `configs/thegame-ppo-tdmse.yaml`.
- Same practical fixed-entropy + KL PPO family as the strongest scratch runs:
  - `gradient_epochs=4`
  - `lr=0.0005`
  - `gae_lambda=0.98`
  - `reward_rescale=0.1`
  - `self_play.num_games=24`
  - shared CNN backbone
- Only intended algorithmic change versus the GE4-400 candidate is replacing `bootstrap_mse_loss,strength=0.1` with `bootstrap_value_mse_loss,strength=0.1`.

Validation:

- `uv run pytest tests/test_policy_gradient_loss.py -q` passed: `7 passed`.
- Zero-iteration startup smoke passed:

```bash
uv run python main.py configs/thegame-ppo-tdmse.yaml \
  -x train.iterations=0 \
  -x pit.every=99999 \
  -x self_play.num_games=1 \
  -x train.save_every=99999
```

Next gate:

- Start from scratch in `/tmp/thegame-ppo-tdmse`.
- First useful checkpoint is `rl-50.pth`, then `rl-100.pth`.
- Compare against GE4-400 and clean epoch-100 gates before continuing.

Run:

- Started from scratch in `/tmp/thegame-ppo-tdmse`.
- Active session id: `49219`.
- Startup reported `BootstrapValueMSELoss` in the loss list and `#parameters 0.14253 M`.
- Early throughput is broadly comparable to GE4-400.
- `rl-50.pth`, 128 games:
  - policy sampling temperature `0.2`: average `32.015625`, min `15`, max `51`.
  - policy sampling temperature `0.1`: average `32.8828125`, min `19`, max `51`.
  - argmax: average `32.7734375`, min `18`, max `52`.
- This is better than GE4-400 at epoch 50 and close to the earlier clean 50-epoch gate, so it deserves the epoch-100 gate.
- The original process was no longer running when checked later, with only `rl-0.pth` and `rl-50.pth` present.
- Resumed from this run's own scratch checkpoint, `/tmp/thegame-ppo-tdmse/thegame-ckpt/rl-50.pth`, with the original `train.iterations=300` schedule intact.
- `rl-100.pth`, 128 games:
  - policy sampling temperature `0.2`: average `35.78125`, min `22`, max `49`.
  - policy sampling temperature `0.1`: average `35.75`, min `22`, max `51`.
  - argmax: average `36.2109375`, min `25`, max `51`.
- Rejected as an optimization axis for now. Epoch 100 is below GE4-400 (`37.5`-`38.4`) and clearly below the stronger clean epoch-100 gate around `40`.
- The resumed process was interrupted after `rl-100.pth` existed. Interrupt handling also produced `rl-109.pth`; it is ignored for decisions because the planned gate was epoch 100.
- A second unintended host-level TD-MSE process was later found and stopped. It produced `rl-150.pth` and `rl-179.pth`; those are ignored for decisions unless explicitly evaluated later.

## Dual-Backbone 400-Epoch Candidate

Motivation:

- Both strong 300-epoch clean scratch checkpoints used separate `policy_backbone` and `value_backbone`.
- Later shared-backbone candidates were cheaper but weaker at epoch 100.
- To avoid mixing architecture capacity with algorithm/config conclusions, the model now has an explicit `net.shared_backbone` flag.
- Shared remains the default for speed; this candidate opts into dual backbones to match the strong-checkpoint capacity.

Code/config:

- Added `NetConfig.shared_backbone: bool = True`.
- `Model(..., shared_backbone=False)` creates separate policy/value backbones.
- Checkpoint normalization can map shared checkpoints into dual models and dual checkpoints into shared models.
- `main.py` now merges YAML `net:` overrides after loading `model-configs/<model>.yaml`, so `net.shared_backbone: false` is not silently discarded.
- Added `configs/thegame-ppo-dual400.yaml`.

Validation:

- `python3 -m py_compile main.py boardrl/config.py boardrl/rl/model/model.py` passed.
- `uv run pytest tests/test_config.py tests/test_modelpool.py -q` passed: `8 passed`.
- Checkpoint-load smoke passed for a dual-backbone old checkpoint and a shared-backbone new checkpoint.
- Zero-iteration startup smoke for `configs/thegame-ppo-dual400.yaml` passed and reported `#parameters 0.27661 M`.

Next gate:

- Start from scratch in `/tmp/thegame-ppo-dual400`.
- Evaluate `rl-50.pth` and `rl-100.pth`.
- If epoch 100 is not at least near the clean/strong epoch-100 gate around `40`, stop rather than spending CPU to epoch 400.

Run:

- Started from scratch in `/tmp/thegame-ppo-dual400`.
- Startup reported `#parameters 0.27661 M`, matching the old dual-backbone capacity.
- Early throughput was roughly `300-330` samples/s.
- The process was stopped after it had passed epoch 50 to avoid leaving it running during evaluation. Interrupt handling produced `rl-59.pth`; the planned first gate remains `rl-50.pth`.
- `rl-50.pth`, 128 games:
  - policy sampling temperature `0.2`: average `31.125`, min `19`, max `49`.
  - policy sampling temperature `0.1`: average `31.0625`, min `19`, max `52`.
  - argmax: average `32.5390625`, min `19`, max `45`.
- Interpretation at epoch 50: not a breakout, but not dead. Argmax is close to the original strong run's epoch-50 argmax (`32.640625`) and above the shared-backbone GE4 epoch-50 argmax (`30.3046875`), so continue to the epoch-100 gate.
- Resumed from this run's own scratch checkpoint `rl-62.pth` with the original `iterations=400` schedule intact.
- `rl-100.pth`, 128 games:
  - policy sampling temperature `0.2`: average `40.71875`, min `27`, max `65`.
  - policy sampling temperature `0.1`: average `40.125`, min `24`, max `57`.
  - argmax: average `39.7734375`, min `25`, max `55`.
- Interpretation at epoch 100: pass. This is above the original strong run's temp-0.2 epoch-100 reference (`40.109375`) and clearly above the shared-backbone GE4/TD-MSE epoch-100 gates. Dual-backbone capacity was a real confound. Continue to `rl-200.pth`.
- Interrupt handling after the epoch-100 gate produced `rl-104.pth`; ignore it for decisions unless explicitly evaluated later.
- The run later completed through `rl-400.pth`. Interrupt/final-save artifacts such as `rl-401.pth` are ignored unless explicitly evaluated.
- `rl-200.pth`, 128 games:
  - policy sampling temperature `0.2`: average `44.484375`, min `29`, max `62`.
  - policy sampling temperature `0.1`: average `45.78125`, min `30`, max `75`.
  - argmax: average `44.1875`, min `27`, max `61`.
- Interpretation at epoch 200: below the original strong run's epoch-200 gate around `49-50`; continue evaluating later checkpoints only because they already exist.
- `rl-300.pth`, 128 games:
  - policy sampling temperature `0.2`: average `48.4140625`, min `33`, max `77`.
  - policy sampling temperature `0.1`: average `49.7421875`, min `35`, max `71`.
  - argmax: average `50.078125`, min `33`, max `70`.
- Interpretation at epoch 300: materially below the old best scratch checkpoint's 128-game sweep (`56.046875` at temperature `0.1`), so no 512-game confirmation.
- `rl-400.pth`, 128 games:
  - policy sampling temperature `0.2`: average `54.390625`, min `36`, max `81`.
  - policy sampling temperature `0.1`: average `54.9609375`, min `33`, max `83`.
  - argmax: average `54.65625`, min `40`, max `75`.
- `rl-400.pth`, 512-game confirmation:
  - policy sampling temperature `0.1`: average `54.720703125`, min `37`, max `88`.
- Interpretation at epoch 400: late recovery is real and dual-backbone capacity remains valuable, but this 400-epoch schedule does not beat the prior valid best (`/tmp/thegame-ppo-300/thegame-ckpt/rl-300.pth`, `55.759765625` over 512 games at temperature `0.1`). The next useful experiment should keep dual backbones but change one algorithmic factor, most likely direct critic MSE strength, rather than continuing this exact recipe.

## Dual-Backbone TD(lambda) Critic MSE Sweep

Motivation:

- Shared-backbone TD-MSE at strength `0.1` was weaker by epoch 100, but that result is confounded by the shared-backbone architecture.
- Dual400 showed that dual policy/value backbones still matter, but the historical value-distribution loss did not produce a new best under the 400-epoch schedule.
- Direct TD(lambda) MSE is still a plausible generic critic improvement, but its scale is not equivalent to the old clipped Normal log-prob loss. Start with smaller strengths instead of assuming `0.1` or `1.0` is correct.

Configs:

- Added `configs/thegame-ppo-dual-tdmse-001.yaml`.
- Added `configs/thegame-ppo-dual-tdmse-003.yaml`.
- Both are based on `configs/thegame-ppo-dual400.yaml`:
  - `net.shared_backbone: false`
  - fixed entropy bonus `0.01`
  - KL `0.1`
  - PPO normalized GAE with `gradient_epochs=4`
  - `self_play.num_games=24`, `batch_size=64`, `lr=0.0005`, `warmup_epochs=15`, `iterations=400`
- Only intended algorithmic change versus dual400 is replacing `bootstrap_mse_loss,strength=0.1` with:
  - `bootstrap_value_mse_loss,strength=0.01`
  - `bootstrap_value_mse_loss,strength=0.03`

Validation:

- Zero-iteration startup check for both configs passed from `/tmp`.
- Both checks reported `BootstrapValueMSELoss` in the loss list.
- Both checks reported `#parameters 0.27661 M`, confirming dual backbones are active.
- Trainer caveat: `train.iterations=0` still executes epoch `0`, so the startup check is a tiny one-epoch smoke, not a pure construction-only check.

Next gate:

- Do not launch these while another long The Game training process is running unless CPU contention is acceptable.
- Start with `configs/thegame-ppo-dual-tdmse-001.yaml` because `0.01` is the least likely to destabilize policy learning.
- Evaluate `rl-50.pth` and `rl-100.pth` over 128 games with:
  - policy sampling temperature `0.2`
  - policy sampling temperature `0.1`
  - argmax
- Continue only if epoch 100 is competitive with the dual400 epoch-100 gate around `40.7` and not clearly behind the original clean epoch-100 gate around `40.1`.

## Large-Batch Shared TD-MSE Run

Provenance:

- This was a user-started run using the current `configs/thegame-ppo-tdmse.yaml`.
- It wrote fresh checkpoints in the repository `thegame-ckpt/` directory through `rl-801.pth`.
- Checkpoint metadata for `rl-800.pth` confirms:
  - `net.shared_backbone: true`
  - `self_play.num_games: 96`
  - `train.batch_size: 256`
  - `train.gradient_epochs: 1`
  - `bootstrap_value_mse_loss,strength=1`
  - `train.iterations: 800`
  - `device: cuda`

Interpretation caveat:

- This is not a clean one-factor comparison against dual400. It changes backbone sharing, TD-MSE strength, rollout batch size, minibatch size, gradient epochs, horizon, and device.
- It is still a valid generic-RL scratch result if the run started from random initialization, and it is currently the strongest evaluated checkpoint in this investigation.

`rl-800.pth`, 128-game decoder sweep:

- Policy sampling, temperature `0.2`: average `58.1484375`, min `41`, max `85`.
- Policy sampling, temperature `0.1`: average `58.7890625`, min `41`, max `81`.
- Argmax: average `57.203125`, min `42`, max `85`.

`rl-800.pth`, 512-game confirmation:

- Policy sampling, temperature `0.1`: average `57.76953125`, min `35`, max `96`.

`rl-801.pth`, metadata:

- Checkpoint epoch: `801`.
- Config train iterations: `800`.
- Treat as a final-save artifact from the same run, but evaluate because it may include one extra update/save.

`rl-801.pth`, 128-game decoder sweep:

- Policy sampling, temperature `0.05`: average `57.8046875`, min `43`, max `93`.
- Policy sampling, temperature `0.075`: average `58.3515625`, min `42`, max `81`.
- Policy sampling, temperature `0.1`: average `58.9453125`, min `45`, max `78`.
- Policy sampling, temperature `0.125`: average `57.90625`, min `39`, max `85`.
- Policy sampling, temperature `0.15`: average `59.3203125`, min `41`, max `81`.
- Policy sampling, temperature `0.2`: average `59.0`, min `39`, max `77`.

`rl-801.pth`, 512-game confirmations:

- Policy sampling, temperature `0.1`: average `58.544921875`, min `38`, max `89`.
- Policy sampling, temperature `0.15`: average `58.16015625`, min `34`, max `92`.
- Policy sampling, temperature `0.2`: average `58.21484375`, min `36`, max `94`.

Current decision:

- New confirmed best is `thegame-ckpt/rl-801.pth` with policy sampling temperature `0.1`: `58.544921875` over 512 games.
- The 128-game temperature wins at `0.15` and `0.2` did not survive 512-game confirmation.
- The large-batch/GE1/TD-MSE-strength-1 recipe is now the most important recipe to isolate. Next clean experiments should distinguish whether the gain came from direct TD-MSE strength, bigger rollout batches, fewer gradient epochs/staler-gradient reduction, longer 800-epoch schedule, CUDA throughput, or some interaction.
