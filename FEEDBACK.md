# RL/System Review Notes

Review focus: things that can plausibly make agents fail to learn, learn the wrong objective, or make evaluation/debugging misleading. I prioritized the active training path (`main.py`, self-play, returns, losses, model, game contracts) over style-only issues.

I did one targeted runtime check and one targeted test:

- `chunk(..., skip_last=True)` confirmed to drop exact full batches: `32 -> []`, `64 -> [32]` for batch size 32.
- Mixed short/long model inputs confirmed to crash in `Model.text_encode`.
- `Config.from_dict` confirmed to raise `KeyError` when `self_play` is present without `num_games`.
- `uv run pytest tests/test_config.py -q` currently fails because tests still expect `model.backbone`, while the implementation now has `policy_backbone` and `value_backbone`.

## Highest-Risk Learning Issues

### 1. Minibatch chunking drops valid training data

Evidence:

- `boardrl/utils/__init__.py:23` implements `chunk`.
- `main.py:223` trains with `chunk(data, batch_size, skip_last=True)`.
- Runtime check: with `batch_size=32`, lengths produce:
  - `32 -> []`
  - `64 -> [32]`
  - `65 -> [32, 32]`

The loop condition is `while i + size < len(data)`, so an exact final full batch is dropped. If the trainset has exactly one batch, the epoch performs no training at all. If it has `N` exact batches, it trains on `N - 1`.

Why this can prevent learning:

- Small games/configs can silently do zero updates.
- Larger runs lose a systematic fraction of samples.
- Metrics/throughput still print, so this is easy to miss.

Suggested fix:

- Replace `chunk` with a straightforward range-based implementation.
- Add tests for `skip_last=True` at `0`, `<batch`, `==batch`, `batch+1`, and exact multiples.

### 2. `gradient_epochs` is configured but not actually used

Evidence:

- `TrainConfig.gradient_epochs` exists in `boardrl/config.py:25`.
- `configs/thegame.yaml:29` sets `gradient_epochs: 4`.
- `main.py:300` multiplies throughput by `gradient_epochs`.
- There is no loop over `gradient_epochs` in `main.py:_train_epoch_on_policy`.

Impact:

- Configs that expect multiple passes over self-play data only get one pass.
- The throughput log is inflated, which can hide the issue.
- PPO/off-policy-style configs get far fewer optimizer opportunities than intended.

Suggested fix:

- Add an explicit `for grad_epoch in range(config.train.gradient_epochs)` loop.
- Shuffle between gradient epochs.
- Log actual samples processed, not intended samples.

### 3. Optimizer stepping and gradient scaling are coupled to the wrong concept

Evidence:

- `main.py:147` sets `multiple_steps=all(loss.supports_off_policy for loss in self.losses)`.
- `main.py:223` iterates batches.
- `main.py:282` divides accumulated gradients by `len(data)`.

Current behavior:

- If every loss says `supports_off_policy=True`, optimizer steps once per batch.
- If any loss says `supports_off_policy=False`, gradients are accumulated over the whole epoch and stepped once.
- For the one-step path, each batch loss is already a mean, then gradients are divided by total sample count. This shrinks updates by roughly an extra factor of `batch_size`.

Why this is suspicious:

- Whether a loss is off-policy-safe should not decide the optimizer accumulation schedule.
- Vanilla policy gradient configs may be doing one tiny optimizer step per self-play episode.
- Gradient clipping happens after per-batch optimizer steps in the multi-step path, so it does not protect those updates.

Suggested fix:

- Separate these concepts:
  - `gradient_accumulation_steps`
  - `gradient_epochs`
  - `off_policy_reuse_allowed`
- Clip before each optimizer step.
- If accumulating, divide by number of accumulated batches or weight losses by batch size consistently.

### 4. Truncated games are treated as terminal games

Evidence:

- `boardrl/rl/eval/selfplay.py:25` defines `EndState`.
- `boardrl/rl/eval/selfplay.py:33` always sets `self.final = True`.
- `EndState.cause` distinguishes `"proper"` vs `"toolong"`, but `compute_returns` only checks `.final`.
- `boardrl/training/returns.py:54` computes full returns whenever `history[-1].final` is true.

Impact:

- If `max_len` cuts off a game, value targets treat the cutoff score as a true terminal outcome.
- No bootstrap value is used at truncation.
- Long games such as Century/The Game can learn the value of "score at arbitrary cutoff" instead of game outcome.

Suggested fix:

- Split `terminal` from `truncated`.
- For truncation, either bootstrap from the value model or explicitly train a finite-horizon objective.
- Keep `cause` in the training sample and log the fraction of truncated games prominently.

### 5. `to_trainset` corrupts `next` links for traces with no decisions

Evidence:

- `main.py:102` defines `to_trainset`.
- `main.py:134` unconditionally does `out[-1].next = end` after stripping the end state.

If a player trace contains only `EndState` and no action records, `hist = hist[:-1]` is empty. The code then assigns that end state as the `.next` of the previous sample from some other trace.

Why this matters:

- This can silently connect one player's/action's transition to another player's terminal state.
- GAE/TD-lambda and bootstrap targets will be wrong.
- It is likely in very short games, `max_len` cutoffs, or games where one player ends the game before another has a decision.

Suggested fix:

- After `hist = hist[:-1]`, skip if `not hist`.
- Add a regression test with a trace containing only `EndState`.

### 6. Reference-model semantics are fragile and several configs use stale references

Evidence:

- Default `reference_model_update` is `"False"` in `boardrl/config.py:40`.
- `ReferenceModelHandler.version` is initialized in `boardrl/rl/utils.py:73`.
- `version` is exposed to update expressions in `boardrl/rl/utils.py:91`.
- On update, `boardrl/rl/utils.py:102` sets `self.model.version = epoch` but does not update `self.version`.
- Losses such as `baseline_value`, `advantage`, `gae`, `normalized_gae`, `bootstrap_mse_loss`, and `q_mse_loss` depend on reference annotations.

Impact:

- Configs using advantage/baseline losses without `reference_model_update` use the initial random model as the baseline forever.
- Any update rule using `version` will see a constant `0`.
- Stale baselines do not necessarily bias REINFORCE, but they can dramatically increase variance and make actor-critic/GAE/bootstrap targets poor.

Suggested fix:

- Increment/update `self.version` in `ReferenceModelHandler.update`.
- If any loss needs a reference, require an explicit reference update policy or log a loud warning.
- Consider naming this `target_model_update` for bootstrap/Q losses and `baseline_model_update` for actor-critic losses, because the semantics differ.

## Model/Loss Issues

### 7. Text `maxlen` is not enforced, and mixed long/short batches crash

Evidence:

- `Model.maxlen = 2048` in `boardrl/rl/model/model.py:197`.
- `text_encode` is at `boardrl/rl/model/model.py:249`.
- Runtime check with one short input and one input longer than 2048 raised: `RuntimeError stack expects each tensor to be equal size`.

Root cause:

- `maxlen = min(maxlen, max(len(g) for g in txts))`.
- For a sequence longer than `maxlen`, `do_pad` does not truncate; negative padding becomes an empty list.
- Shorter examples are padded to `maxlen + 1`, while longer examples keep their longer length.

Impact:

- Long Century prompts can crash training depending on batch composition.
- Even if all long examples happen to share length rarely, the model is not respecting its configured max length.

Suggested fix:

- Truncate all encoded strings to `maxlen` before appending the sentinel.
- Add tests for all-short, all-long, and mixed short/long batches.
- Log truncation rate if truncation is used.

### 8. Character vocabulary is hard-coded to ASCII-size embeddings

Evidence:

- Embeddings use `nn.Embedding(128, ...)` at `boardrl/rl/model/model.py:115`, `:137`, `:156`, and `:170`.
- `text_encode` uses `ord(c)` directly in `boardrl/rl/model/model.py:255`.

Impact:

- Any non-ASCII character in a game state or move string crashes with an embedding index error.
- There is a `tokenizer.json` and `train-tokenizer.py`, but the active model path ignores them.

Suggested fix:

- Either enforce ASCII-only game displays with validation, or use a real tokenizer/vocabulary with unknown-token handling.

### 9. KL penalty scale depends on batch size

Evidence:

- `KLPenalty` is defined at `boardrl/rl/model/loss.py:315`.
- It sums over samples and returns `self.strength * loss` at `boardrl/rl/model/loss.py:333`.
- Most other losses average over the batch.

Impact:

- Changing batch size changes the effective KL strength.
- In configs like `thegame.yaml`, `kl,strength=0.5` can dominate more than intended at larger batch sizes.

Also suspicious:

- The implemented direction is `KL(current || reference)` because `input=log_ref`, `target=log_current`.
- The current test does not catch direction mistakes because its current and reference logits have the same softmax distribution.

Suggested fix:

- Divide by batch size.
- Make direction explicit in the class name or constructor.
- Add a non-degenerate KL test where current and reference distributions differ.

### 10. `bootstrap_mse_loss` is not MSE and has an unused `clip` argument

Evidence:

- `BootstrapMSELoss` is defined at `boardrl/rl/model/loss.py:449`.
- It returns negative Normal log-probability, not MSE.
- `clip` is accepted but not used to control behavior.

Impact:

- Config names and hyperparameters are misleading.
- This makes tuning harder because the scale is distributional NLL, not squared error.

Suggested fix:

- Rename to something like `bootstrap_value_nll`.
- Remove or implement `clip`.

### 11. Policy logits are overloaded as policy, advantage, and Q-values

Evidence:

- `PolicyValue.q_value()` in `boardrl/rl/model/model.py:32` interprets policy logits as dueling-style action advantages.
- `q_mse_loss` in `boardrl/rl/model/loss.py:473` trains `pred_policy` as Q advantages.
- `policy_gradient_loss` in `boardrl/rl/model/loss.py:219` trains the same tensor as action logits.
- `Gumbel` strategy uses model value lookahead, while DQN-style code uses `q_value()`.

Impact:

- It is conceptually valid only if a run is clearly in "policy-logit mode" or "Q-advantage mode".
- Combining losses or strategies across those modes can produce incoherent training signals.

Suggested fix:

- Split heads/types: `policy_logits`, `q_advantages`, and `value`.
- Make losses declare which head they consume.

## Game/Environment Contract Issues

### 12. `Game.copy()` does not have reliable semantics across games

Evidence:

- Generic lookahead calls `g.copy()` in `boardrl/games/strategies.py:116`.
- Draft MCTS also calls `game.copy()` in `boardrl/draft/mcts.py:45` and `boardrl/draft/mcts.py:212`.
- `Century.copy(randomize=True)` defaults to randomization in `boardrl/games/century/engine.pyx:803`.
- `Century.copy` randomizes copied victory hidden cards via `VictoryPile.copy(randomize)` at `boardrl/games/century/engine.pyx:810`.
- `Sum.copy()` in `boardrl/games/sum/game.py:21` does not copy `current_player_`, `turn`, or `round_`.
- `GuessNumber.copy()` in `boardrl/games/guessnumber/game.py:44` reconstructs with positional arguments that do not preserve `num_symbols` correctly when it is not the default.

Impact:

- Generic Gumbel/MCTS/search may evaluate a different state than the real state.
- Some game copies reset turn/player metadata.
- This breaks the core assumption needed for planning and model-based evaluation.

Suggested fix:

- Define explicit methods instead of one overloaded `copy()`:
  - `clone_exact()`
  - `determinize_for_player(player)`
  - `clone_for_rollout(player)`
- Add a shared game-contract test suite that every game must pass.

### 13. Century reward is not full Century scoring

Evidence:

- `Century.points_for` is at `boardrl/games/century/engine.pyx:883`.
- The comment says `FIXME:: this should be .points()...`.
- `VictoryPile` also has `# FIXME add coins` at `boardrl/games/century/engine.pyx:543`.

Impact:

- The agent is optimizing victory-card points only, not full game score.
- If cube values, coins, or tie-breakers matter, the learned policy can be systematically wrong.

Suggested fix:

- Decide whether this is a deliberate shaped objective or a bug.
- If deliberate, rename/log it as the training objective.
- If not deliberate, implement true scoring and add tests against known Century end states.

### 14. Perspective normalization is inconsistent

Evidence:

- `ConnectFour.display` builds a perspective-relative `rep` at `boardrl/games/connectfour/game.py:38`, then immediately overwrites it with absolute player symbols at `boardrl/games/connectfour/game.py:39`.
- Other games vary between absolute symbols, current-player headers, and force-rendered player views.

Impact:

- A shared policy may need to learn duplicated seat-specific policies.
- Evaluation with rotation can be harder to interpret.
- This is not necessarily wrong, but it should be intentional and tested.

Suggested fix:

- Make each game's display contract explicit: absolute board or current-player-normalized board.
- Add a game-contract test for `display(force=p)` and `display_with_moves()` invariants.

### 15. Legal action serialization depends on fragile text conventions

Evidence:

- The model locates legal move logits by scanning for `"@"` in `boardrl/rl/model/model.py:269`.
- Every game must ensure `"@"` appears only as legal-move prefixes.
- There is no central validation.

Impact:

- A stray `"@"` in game text creates extra policy logits and shifts action alignment.
- This is a silent policy-target corruption risk.

Suggested fix:

- Return structured observations with explicit legal-action spans, or validate that `len(moves_pos) == len(game.moves)` in self-play/model inference.

## Config/Test/Evaluation Drift

### 16. Test suite is not currently a reliable safety net

Evidence:

- `uv run pytest tests/test_config.py -q` fails.
- Failure: `tests/test_config.py:79` expects `model.backbone`, but `Model` now has `policy_backbone` and `value_backbone`.

Impact:

- Tests can be stale relative to the current architecture.
- Slow config tests are likely not being run often enough to catch config drift.

Suggested fix:

- Fix `test_config.py` to check both backbones.
- Add small, fast training smoke tests for one policy-gradient config and one bootstrap/GAE config.

### 17. Some configs/scripts use stale APIs

Evidence:

- `configs/run-flat-alphazero.yaml:15` uses `train.loss`, but `TrainConfig` expects `train.losses`.
- `main.py` requires top-level `model`, while `run-flat-alphazero.yaml` only has `net` unless overridden.
- `run_pit.py:21` and `run_pit.py:22` build `argmax:<path>` and `policy_sampling:<path>`, but `RegisterByName` parses comma-separated `name,arg=value` strings at `boardrl/utils/registerbyname.py:61`.
- `run_pit.py:53` uses the generic `strategy_from_string`, not `game_desc.strategy_from_string`, so Century-specific strategies like `random_buy` are not resolved there.

Impact:

- Evaluation scripts may fail or evaluate different strategies than intended.
- Config smoke tests may fail after unrelated changes.

Suggested fix:

- Update stale configs or move them to an `archive/` folder excluded from config tests.
- Update `run_pit.py` to use `game_desc.strategy_from_string` and current strategy syntax.

### 18. Empty trainsets are not guarded

Evidence:

- `configs/benchmark.yaml` sets `only_players: []`.
- `to_trainset` will produce no samples for an empty player filter.
- `_train_epoch_on_policy` does not have an early return for `len(data) == 0`.

Impact:

- Empty trainsets can cause division-by-zero logging or useless epochs.
- If an empty filter is accidental, the run appears to proceed while learning nothing.

Suggested fix:

- Treat empty `only_players` / `only_strategies` as invalid unless explicitly marked benchmark/eval-only.
- Add `if not trainset: raise ValueError(...)` for training configs.

## Recommended Fix Order

1. Fix `chunk(skip_last=True)` and add tests.
2. Implement real `gradient_epochs` and clean optimizer stepping/scaling.
3. Split terminal vs truncated trajectories and fix `to_trainset` empty-trace handling.
4. Repair reference-model versioning and require explicit update semantics for reference-dependent losses.
5. Add a shared game contract test suite covering exact clone, determinized clone, display/action alignment, reward sign, and terminal/truncated behavior.
6. Fix `Model.text_encode` truncation/vocabulary validation.
7. Normalize KL loss scaling and clarify loss names/directions.
8. Bring configs, `run_pit.py`, and stale tests back in sync.

## Fast Tests I Would Add

- `test_chunk_skip_last_exact_batch`.
- `test_to_trainset_skips_empty_player_trace_without_cross_linking`.
- `test_truncated_endstate_does_not_mark_terminal_return`.
- `test_reference_model_version_changes_on_update`.
- `test_text_encode_truncates_mixed_length_batch`.
- `test_game_copy_contract_for_all_registered_games`.
- `test_model_legal_move_count_matches_game_moves`.
- `test_kl_penalty_is_batch_size_invariant`.

