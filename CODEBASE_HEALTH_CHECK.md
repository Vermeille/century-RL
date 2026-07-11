# Codebase Health Check

Date: 2026-07-10

## Scope

This is a read-only audit of the current working tree, with `configs/thegame-ppo-tdmse.yaml` treated as the intended product boundary. The working tree is not clean: it contains modified strategy, CNN, serving, and test files, plus many untracked configs, checkpoints, serving assets, and notes. Those changes were not reverted or edited.

The target training path is:

`main.py` -> `Config` -> CNN `Model` -> The Game -> `self_play`/pit -> PPO, KL, entropy, and bootstrap-value losses -> checkpointing.

Everything outside that path should have to justify its existence. The current repository is a framework and experiment archive, not a small product: seven registered games, four model backbones, a broad loss registry, two evaluation layers, a web server, draft MCTS/pretraining code, toy learning experiments, and dozens of recipe configs.

## Executive Recommendation

Reduce the supported product to one game, one model family, one training recipe, one evaluation path, and one optional serving command:

- The Game, with one explicitly supported mode.
- CNN policy/value model, retaining only the dimensions needed by the reference recipe.
- One typed configuration schema; no arbitrary Python expressions in YAML.
- One policy-sampling strategy, plus random and deterministic argmax for evaluation.
- One self-play implementation and a simple fixed pit. Remove adaptive opponent selection.
- Only the losses used by the reference recipe: PPO policy gradient, entropy bonus, KL penalty, and bootstrap value MSE.
- Checkpoint save/load and a small set of game, model, loss, and one-epoch smoke tests.

Do not start by massaging every old feature into a cleaner abstraction. Archive or delete the unsupported surface first. That will make the remaining behavioral bugs visible and make future changes reviewable.

## Highest-Priority Findings

### P0: truncated rollouts are represented as terminal outcomes

`boardrl/rl/eval/selfplay.py:30-38` always emits `EndState.final = True`, even when `cause` is `"toolong"`. `boardrl/training/returns.py:54-67` therefore computes ordinary terminal returns for a game cut off by `max_len` and never bootstraps the value at the cutoff.

This directly affects the target recipe, whose self-play and pit limits are 500 and 800. The critic can learn the score at an arbitrary time limit as if it were the game result. Keep `cause`, but model `terminal` and `truncated` separately; either bootstrap truncated states or make the finite-horizon objective explicit.

### P0: the old pit entry point is not a reliable evaluator

`run_pit.py:8-22` hard-codes Century-only strategies, including names that are not registered by the generic strategy registry, and formats checkpoint strategies as `argmax:<path>` / `policy_sampling:<path>`. `RegisterByName._read` in `boardrl/utils/registerbyname.py:59-69` only supports `name,arg=value` syntax. `run_pit.py:5-6` also calls the generic `strategy_from_string` rather than the selected `GameDesc` registry, so Century-specific names such as `random_buy` cannot resolve there.

The script also chooses random strategy pairs and persists results to an ad hoc `games.json`; it is not a reproducible fixed evaluation. Retire it or replace it with a small command that takes an explicit game, two explicit strategy specs, a seed, and a game count.

### P0: adaptive matchmaking is both unnecessary and structurally wrong

`boardrl/rl/eval/matchmaker.py:10-46` contains the meta-strategy system. `MatchMaker.run_self_play` resolves all meta strategies before running the batch (`:108-127`), so selection is not updated game by game. `meta_best_opponent` assumes that `state["matrix"][to]` exists and falls back to the first partially empty row (`:35-46`), which is arbitrary and can select the wrong opponent.

More importantly, `_record_outcomes` collapses results into a dict keyed by strategy text (`:165-188`). Two players with the same spec, which is the normal target configuration, become one entry and produce no head-to-head data. Dict strategy descriptions are also accepted by the public type but are unhashable when used as keys. This is a strong reason to delete the meta-matchmaker rather than repair it for the reduced product.

### P1: game cloning contracts are inconsistent

Generic lookahead exists in `boardrl/games/strategies.py:63-69` and the draft MCTS. Several games do not preserve exact state:

- `boardrl/games/sum/game.py:21-27` copies board and scores but not `current_player_`, `turn`, or `round_`.
- `boardrl/games/guessnumber/game.py:44-53` calls `GuessNumber(self.num_symbols, ...)`, passing `num_symbols` as `max_number`; non-default symbol counts are corrupted.
- Century's Cython `copy` defaults to randomization (`boardrl/games/century/engine.pyx:803`) and has separate hidden-state semantics.

The overloaded `copy()` contract is not safe for generic planning. The reduced product should either remove lookahead entirely or define and test an exact clone operation. Do not retain Gumbel/MCTS until exact cloning and player-perspective semantics are designed.

### P1: model input length handling is unsafe

`boardrl/rl/model/model.py:280-288` computes a batch width but does not truncate strings longer than `maxlen`; `do_pad` can produce different lengths for a mixed batch. A short/long batch can fail at `torch.stack`, and the declared `Model.maxlen = 2048` is not actually enforced. The same code maps raw `ord(c)` values into embeddings of size 128, so non-ASCII display text fails with an embedding index error.

For the reduced product, enforce ASCII display strings and truncate or reject overlong states before batching. Add a mixed-length regression test. The existing `tokenizer.json` and `train-tokenizer.py` are not part of the active model path and should be removed unless a real tokenizer migration is undertaken.

### P1: reference-model control is an embedded programming language

`TrainConfig.reference_model_update` is a Python source string (`boardrl/config.py:43`), executed through `PythonExec` and `exec`/`eval` in `boardrl/utils/pythonexec.py:9-36`. The reference version is initialized at `boardrl/rl/utils.py:73` but the update path sets `self.model.version` rather than incrementing `self.version` (`:84-103`). Any expression relying on `version` therefore sees stale state.

The target config only needs an always-copy or fixed-period update. Replace the expression with a typed enum/interval field, then delete `PythonExec`. This reduces configuration risk and removes a surprising execution surface.

### P1: current tests and model metadata are already out of sync

The prescribed non-model test subset currently reports 154 passed and 2 failed:

- `tests/test_nim.py::test_nim_metrics_reports_per_match_choice_probability`: the new test fixtures have no `state` on action records, but `boardrl/games/nim/metrics.py:15-27` requires it.
- `tests/test_rps.py::test_metrics_probabilities`: current metric aggregation returns `[0.2, 0.5, 0.3]`, while the test expects `[0.25, 0.4, 0.35]`.

`tests/test_config.py` and `tests/test_model_configs.py` were run separately. Config tests pass. Model-config tests fail because `model-configs/cnn-large.yaml` says `0.07M`, while the current uncommitted CNN implementation instantiates `1.77M` parameters. This is working-tree drift, but it means the test suite is not currently a clean gate.

## Active-Path Design and Maintainability Issues

### Training script is an oversized composition root

`main.py` is 539 lines and owns optimizer construction, learning-rate scheduling, trainset conversion, checkpoint format, reference-model lifecycle, batching, metrics, self-play, pit evaluation, CLI YAML mutation, and the training loop. This is the main source of organic complexity.

Keep the behavior, but split into four small units:

1. `train/config.py`: typed config loading and model-config selection.
2. `train/rollout.py`: game execution and return annotation.
3. `train/update.py`: optimizer and the four supported losses.
4. A short CLI that wires them together.

Do not preserve the generic `fix_dict` mutation interface (`main.py:464-482`) after the config set is reduced. It bypasses schema intent and makes arbitrary nested values part of the CLI API.

### String registries are doing too much

`RegisterByName` (`boardrl/utils/registerbyname.py`) combines registration, reflection, string parsing, type conversion, and dependency injection. It has no escaping rules, uses assertions for user input, and has weak error messages for malformed specs. The registry is useful for a small plugin boundary, but not as the internal API for every game, model, loss, and strategy.

For the reduced product, use typed constructors or a small explicit mapping. If the registry remains for strategies, standardize one syntax and test every supported argument. The current `get_model` helper asserts that the provided object is a `ModelPool` and contradicts its own `default` fallback description (`boardrl/games/strategies.py:7-18`). `ModelPool` documentation also advertises `recent-N`, but `boardrl/utils/modelpool.py:82-99` only handles `this`, `reference`, and filesystem paths.

### Strategy surface contains known dead or suspect behavior

Keep `random`, `argmax`, and `policy_sampling` for the reduced product. Archive:

- `gumbel` in `boardrl/games/strategies.py:128-164`: experimental value-guided behavior with ambiguous player/value semantics; the experiment notes explicitly say it is not a valid headline path.
- `longest_move` (`:71-80`): its diagnostic distribution uses `len(g.moves) / total` for every move rather than `len(move) / total`.
- `mean` (`:61`) is unused.
- Century-specific strategies, Sum's exact strategy, and Nim's optimal strategy if those games are removed.

The target `include_moves: false` addition is reasonable for payload size, but strategy output currently mixes logits and probabilities in the `info["moves"]` field depending on the epsilon path. Rename fields or make the representation consistent before relying on metrics.

### Loss registry is much larger than the target algorithm

`boardrl/rl/model/loss.py` is 558 lines. The target config uses only:

- `policy_gradient_loss` with `normalized_gae` and PPO drift;
- `entropy_bonus`;
- `kl`;
- `bootstrap_value_mse_loss`.

The imitation losses, `z_loss`, scheduled/linear/reverse entropy variants, value log-probability, plain value MSE, `bootstrap_mse_loss`, and DQN `q_mse_loss` are separate algorithm families. `BootstrapMSELoss` is also named MSE while implementing a clipped Normal negative log probability (`:496-518`). Delete or archive those families instead of keeping a registry of unvalidated alternatives. Rename the retained loss classes to match their actual targets where practical.

The model also overloads policy logits as Q-value advantages through `PolicyValue.q_value()` (`boardrl/rl/model/model.py:32-36`), while `q_mse_loss` trains them as Q terms. This coupling is another reason to remove DQN code from the product boundary rather than preserve an ambiguous shared head.

### The CNN rewrite needs to settle before cleanup

The target uses CNN, but the current uncommitted `boardrl/rl/model/cnn.py` rewrite changes parameter count and architecture while `model-configs/cnn*.yaml` comments still describe old counts. `CNNBackbone` also creates `self.norm` (`boardrl/rl/model/model.py:170-172`) but never applies it in `forward` (`:174-176`). Decide whether the rewrite is the intended model, update metadata/tests, and then delete the unused transformer/gated/LSTM branches.

### Self-play data structures have compatibility baggage

`boardrl/rl/eval/selfplay.py` contains `PlayerTrace`, `GameTrace`, `SelfPlayResults`, legacy `games` access, both seat and strategy indexing, collapse metrics, pit helpers, a compatibility `call_strategy`, `self_play2`, `self_play`, and `pit`. For one game and two identical policy players, strategy identity is unnecessary. Use seat-indexed traces and one explicit `play_batch` API; retain only the metrics used by The Game.

The current `call_strategy` catches any synchronous `TypeError` and retries with `strategy()(game)` (`:195-200`). This can hide a real programming error and makes callable strategy contracts unclear. Pick one async strategy interface.

## What Can Be Trimmed

### Delete or archive immediately

- `boardrl/draft/` and its MCTS/pretraining/model-extras code. No production imports reach it; MCTS also contains disabled Graphviz code under `if False`.
- `boardrl/experiments/`, `experiments/run_experiments.py`, and toy training tests if model research is no longer a goal.
- `profiler.py`, `train-tokenizer.py`, `tokenizer.json`, `mcts.dot`, `mcts.png`, `game.txt`, and ad hoc profiling scripts after extracting any useful one-epoch smoke command.
- `run_pit.py` in its current form. Replace it rather than maintain it.
- `FEEDBACK.md`, `OPTIMIZATION_LOG.md`, and `THEGAME_EXPERIMENTS.md` from the product tree unless these are intentionally retained as research history. They are large operational notebooks, not maintained documentation.

### Remove from the supported package

- Games: Century, Connect Four, GuessNumber, Nim, RPS, Sum, and their UIs/metrics/tests. Keep The Game only.
- Model backbones: transformer, LSTM, and gated CNN. Keep CNN only.
- Model configs: retain `cnn-large.yaml`; optionally retain `cnn.yaml` as a deliberately supported small smoke model. Remove the other configs.
- Strategies: remove Gumbel, longest-move, meta strategies, and game-specific strategies outside The Game.
- Losses: retain only the four target losses and their direct helpers.
- Utilities: remove `CachedBatchProcessor`, `PythonExec`, recent-model discovery, and the generic registry if replaced by typed factories.

### Keep, but simplify

- `boardrl/games/thegame/game.py`: keep the game, but reduce the six game modes to the one used by the target config or make the mode a separately tested product feature.
- `boardrl/games/thegame/metrics.py`: retain only metrics used in training and pit dashboards.
- `boardrl/rl/eval/selfplay.py`: one direct rollout implementation, explicit truncation handling, and no strategy-index compatibility layer.
- `boardrl/utils/batchprocessor.py` and `ModelPool`: keep only if asynchronous model inference remains a measured requirement. Otherwise direct batched model calls will be easier to reason about.
- `boardrl/serve/`: make it an optional extra or remove it from the training repository. It recursively scans the entire current directory for checkpoints (`serve.py:154-175`), which makes startup and model discovery depend on local artifact clutter.

## Configuration and Repository Hygiene

There are 28 config files, many of them named as incremental experiments: `ge2`, `ge3`, `ge4`, `dual400`, `lowtemp`, `lrfloor`, `entropydecay`, and similar. Keep one canonical config and perhaps one tiny CPU smoke config. Move old recipes to an external experiment archive.

`configs/run-flat-alphazero.yaml` is stale: it uses `train.loss` rather than `train.losses` and has no top-level `model`, while `main.py:504-516` requires one. It should not live in the active config directory. The config test parametrizes every YAML file (`tests/test_configs.py:7-31`), so archival files currently expand the test surface and make historical recipes a release gate.

The repository contains large untracked checkpoint trees (`thegame-ckpt`, `connectfour-ckpt`, `nim-ckpt`, and others), a `.venv`, and generated files. Checkpoints should be outside Git or in a dedicated artifact store. Add ignores for `.venv/`, `*-ckpt/`, `*.pth`, generated profiler/Graphviz files, and local result JSON. Do not delete the user's current artifacts without an explicit cleanup operation.

The dependency list in `pyproject.toml` is also an archive of optional concerns. For the reduced training product, likely runtime dependencies are `torch`, `tqdm`, `cython`, `pyyaml`, `pydantic`, and optionally `visdom`. `fastapi` should be an optional serving extra. `heavyball` is only needed for the Muon branch and draft code; `matplotlib` is for experiments; `psutil` is only for unused recent-model discovery; `natsort` and `crayons` are serving/legacy presentation helpers; pytest packages belong in a dev group. `tokenizers` is imported by `train-tokenizer.py` but is not declared, so that script is already unsupported.

## Suggested End State

The cleaned tree should be approximately:

```text
configs/thegame-ppo-tdmse.yaml
model-configs/cnn-large.yaml
boardrl/config.py
boardrl/games/thegame/{game.py,metrics.py}
boardrl/rl/model/{model.py,cnn.py,loss.py,utils.py}
boardrl/rl/eval/selfplay.py
boardrl/training/{returns.py,sample.py}
boardrl/utils/{batchprocessor.py,modelpool.py,visualizer.py}
main.py
evaluate.py
tests/
```

Before making algorithmic changes, establish these gates:

1. One-epoch CPU smoke run from the canonical config with no checkpoint artifacts required.
2. Exact clone and legal-action alignment tests for The Game.
3. Terminal versus truncated return tests.
4. Reference update/version test without Python execution.
5. Mixed-length model input test.
6. Fixed-seed pit test with explicit strategy names and no adaptive matchmaking.
7. Clean `pytest` run and accurate model parameter metadata.

## Verification Performed

- `uv run pytest tests -k "not transformer and not model and not config"`: 154 passed, 2 failed, 55 deselected.
- `uv run pytest tests/test_config.py tests/test_model_configs.py -q`: 6 passed, 1 failed; CNN-large parameter metadata mismatch described above.
- `uv run python -m compileall -q boardrl main.py run_pit.py experiments scripts profiler.py train-tokenizer.py`: passed.
- `ruff` could not be run through `uv` because the environment attempted to access a read-only uv cache; no lint result is claimed.

