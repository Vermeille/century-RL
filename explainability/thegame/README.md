# The Game policy explainability

These are the standalone analysis programs used to compare the final
`YOLO-yay2` Omni policy with the final `YOLO` Strict policy and to distill the
Strict policy into an interpretable move-ranking rule.

The latest practical target is **calculator-free advice for human play**, not
exact neural-policy recovery. See [`results/HUMAN_RULES.md`](results/HUMAN_RULES.md)
for the tested two-threshold awkward-card tactic and the ideas that failed screening.
Further threshold/priority refinement is documented in
[`results/AWKWARD_REFINEMENT.md`](results/AWKWARD_REFINEMENT.md), including a
4,096-deal untouched confirmation and a one-threshold simplification.
Omni's extra-card versus end-turn decision is isolated in
[`results/OMNI_STOPPING.md`](results/OMNI_STOPPING.md), with both neural-card
hybrids and a fully non-neural lowest-cost strategy.
Configuration screening and held-out strategy-gap comparisons are in
[`results/CONFIG_GAP.md`](results/CONFIG_GAP.md). The pile/minimum variants are
experiment-only; the production game is unchanged.

## Scripts

| Script | Purpose |
|---|---|
| `checkpoint_analysis.py` | Large fresh checkpoint evaluation, score distributions, behavior summaries, and input-scrambling sensitivity tests. |
| `memory_analysis.py` | Offline and online `Mem:` interventions, including zeroing, scrambling, and removing played-card memory. |
| `identity_memory_analysis.py` | Replaces `Mem:` with a different reachable memory vector matched on visible state identity and measures action/score changes. |
| `nonlowest_analysis.py` | Samples decisions where the policy rejects the cheapest card and replays the chosen branch against every minimum-cost alternative. |
| `strategy_ablation.py` | Factorial evaluation of lowest-cost tie-breaking, unrestricted card selection, and Omni stop/continue behavior. |
| `distill_strict.py` | Fits an interpretable linear move ranker to Strict, performs one DAgger iteration, and evaluates the distilled strategies in full games. |
| `semantic_tree.py` | Collects a visible-state corpus and fits small trees over named action generators. |
| `pile_tree.py` | Fits shared per-pile trees and one-parameter rules; evaluates frozen rules on paired initial deals. |
| `sparse_rule.py` | Forward-selects short adjusted-cost formulas using joint-choice imitation loss. |
| `compact_strategy.py` | Standalone four-term strategy: no neural network, tree, or fitting dependency at play time. |
| `summarize_semantic.py` | Archives fitted rules, paired scores, provenance, and the readable semantic-strategy report. |
| `discrete_rules.py` | Fits Boolean replacement trees and screens/evaluates discrete tactics on matched deals. |
| `human_rules.py` | Human-readable tactics using only integer thresholds: awkward cards, reserve piles, runs, and reverse setups. |
| `summarize_human.py` | Archives every screened candidate plus untouched confirmation scores and writes the practical guide. |
| `awkward_refinement.py` | Refines awkward-card targeting, integer thresholds, starter eligibility, and reverse-jump ties using deterministic full games. |
| `summarize_refinement.py` | Archives all refinement screens, validation, and fresh confirmation, with paired intervals and a practical report. |
| `omni_stopping.py` | Collects optional-stop labels, fits small Boolean rules, and isolates stop/continue behavior while holding neural card ranking fixed. |
| `summarize_omni_stopping.py` | Archives Omni stopping rules, held-out classification, matched hybrid scores, and fully non-neural checks. |
| `config_gap.py` | Screens rule configurations with paired greedy/awkward/formula play; independently calibrates optional stopping. |
| `summarize_config_gap.py` | Reports normalized gaps, paired gap-widening intervals, wins, and remaining-card reduction. |

All programs write detailed JSON suitable for follow-up analysis. They do not
modify checkpoints or training state.

## Checkpoints analyzed

```text
checkpoints/schedule-search/coop/thegame,mode=omni/patchformer-medium-p8/YOLO-yay2/step-900.pth
checkpoints/schedule-search/coop/thegame,mode=strict/patchformer-medium-p8/YOLO/step-900.pth
```

For exact historical reproduction, run a script with the BoardRL source that
created its checkpoint. The original investigation used source overlays for
the two runs because the active worktree had subsequently changed. When the
active checkout remains checkpoint-compatible, the `PYTHONPATH` prefix below
can be omitted.

```bash
STRICT_SOURCE=/tmp/thegame-policy-sources/strict
OMNI_SOURCE=/tmp/thegame-policy-sources/omni-yay2
PYTHON=.venv/bin/python
STRICT_CHECKPOINT='checkpoints/schedule-search/coop/thegame,mode=strict/patchformer-medium-p8/YOLO/step-900.pth'
OMNI_CHECKPOINT='checkpoints/schedule-search/coop/thegame,mode=omni/patchformer-medium-p8/YOLO-yay2/step-900.pth'
```

The temporary overlay paths above record the paths used during the original
analysis; they are not repository dependencies. Replace them with durable
source snapshots when reproducing elsewhere.

## Primary evaluation

Run both modes with the same game count and batching protocol:

```bash
PYTHONPATH="$OMNI_SOURCE" "$PYTHON" explainability/thegame/checkpoint_analysis.py \
  --mode omni --checkpoint "$OMNI_CHECKPOINT" --games 4096 \
  --ablation-games 1024 --output /tmp/thegame-omni-analysis.json

PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/checkpoint_analysis.py \
  --mode strict --checkpoint "$STRICT_CHECKPOINT" --games 4096 \
  --ablation-games 1024 --output /tmp/thegame-strict-analysis.json
```

## Memory interventions

`memory_analysis.py` produces its own matched baseline before running the
online interventions:

```bash
PYTHONPATH="$OMNI_SOURCE" "$PYTHON" explainability/thegame/memory_analysis.py \
  --mode omni --checkpoint "$OMNI_CHECKPOINT" \
  --output /tmp/thegame-omni-memory.json

PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/memory_analysis.py \
  --mode strict --checkpoint "$STRICT_CHECKPOINT" \
  --output /tmp/thegame-strict-memory.json
```

The identity-matched intervention is run similarly:

```bash
PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/identity_memory_analysis.py \
  --mode strict --checkpoint "$STRICT_CHECKPOINT" \
  --baseline-results /tmp/thegame-strict-analysis.json \
  --output /tmp/thegame-strict-identity-memory.json
```

Use `--mode omni`, the Omni source/checkpoint, and its primary JSON for the
corresponding Omni run.

## Non-lowest counterfactuals

```bash
PYTHONPATH="$OMNI_SOURCE" "$PYTHON" explainability/thegame/nonlowest_analysis.py \
  --game 'thegame,mode=omni' --checkpoint "$OMNI_CHECKPOINT" \
  --label omni-yay2 --games 256 --samples 2048 \
  --output /tmp/thegame-omni-nonlowest.json

PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/nonlowest_analysis.py \
  --game 'thegame,mode=strict' --checkpoint "$STRICT_CHECKPOINT" \
  --label strict-yolo --games 256 --samples 2048 \
  --output /tmp/thegame-strict-nonlowest.json
```

Each sampled state is replayed after the policy choice and after every tied
minimum-cost card choice. End-turn `x` is deliberately excluded from card-cost
comparisons.

## Strategy-factor ablation

```bash
PYTHONPATH="$OMNI_SOURCE" "$PYTHON" explainability/thegame/strategy_ablation.py \
  --mode omni --checkpoint "$OMNI_CHECKPOINT" --games 1024 \
  --output /tmp/thegame-omni-strategy-ablation.json

PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/strategy_ablation.py \
  --mode strict --checkpoint "$STRICT_CHECKPOINT" --games 1024 \
  --output /tmp/thegame-strict-strategy-ablation.json
```

The hybrids separate model tie-breaking, non-lowest card sequencing, and—in
Omni—learned stop/continue decisions. All variants reuse the same seed scheme
so their per-game scores are paired.

## Strict policy distillation

```bash
PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/distill_strict.py \
  --checkpoint "$STRICT_CHECKPOINT" \
  --teacher-games 512 --dagger-games 512 --eval-games 1024 \
  --output /tmp/thegame-strict-distillation.json
```

The JSON contains held-out action agreement, gameplay evaluation, the largest
coefficients, and complete weight maps for the initial and DAgger rankers with
and without memory features. The high-scoring interpretable result from the
original investigation was the initial no-memory ranker; the DAgger refit is a
diagnostic and should not automatically replace it merely because it was fit
on more states.

## Runtime notes

- These are CUDA-oriented analyses and intentionally do not silently fall back
  to CPU for checkpoint evaluation.
- Large evaluations can take several minutes.
- Keep game counts, seeds, checkpoint paths, and source revision matched when
  comparing outputs.
- JSON outputs can be large because some scripts retain per-game or
  per-decision records for paired statistics.

## Compact semantic strategy

The completed first experiment is documented in
[`results/SEMANTIC_REPORT.md`](results/SEMANTIC_REPORT.md).
Its four-term rule scores 86.47 versus Strict's 87.50 and lowest-cost's 80.96
on 1,024 identical initial deals. Its action agreement is only 79.1%: this is
a strong compact surrogate, **not** near-exact algorithm recovery.

For each pile, keep only its cheapest legal card, then minimize:

```python
cost + hardest_remaining / 4 - pile_gap_gain / 5 + 8 * cost / space_left
```

Definitions and edge cases are explicit in `compact_strategy.py`. The strategy
reads only the displayed hand and pile tops, not hidden hands/deck order or
played-card memory. Full fitted JSON and per-deal scores are under `results/`.

Run these commands from the repository root. The historical Strict source tag
is `run-source/YOLO-20260901T125727Z-c6bb16ffa32e`; use its source overlay as
`STRICT_SOURCE`. Use `-P -m` and put the repository after the source overlay
in `PYTHONPATH`, so Python does not prepend the current directory ahead of
the historical BoardRL package. The evaluated checkout's game/model/rollout
sources were also verified equal to that source tag (the only BoardRL git
difference was an unused reservoir utility).

```bash
ANALYSIS_PATH="$STRICT_SOURCE:$PWD"
```

Fit-only dependencies can be isolated without changing this project's venv
or dependency lock:

```bash
uv pip install --target /tmp/thegame-tree-deps --no-deps \
  scikit-learn==1.7.2 scipy==1.16.2 joblib==1.6.0 \
  threadpoolctl==3.6.0 cloudpickle==3.1.2
```

Collect and fit (512 training games, 128 validation, 256 final test):

```bash
PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.semantic_tree collect \
  --checkpoint "$STRICT_CHECKPOINT" --games 896 --seed 1840000 \
  --data /tmp/strict-semantic-corpus.npz

PYTHONPATH="/tmp/thegame-tree-deps:$ANALYSIS_PATH" "$PYTHON" \
  -P -m explainability.thegame.semantic_tree fit \
  --data /tmp/strict-semantic-corpus.npz --output /tmp/strict-semantic-trees.json

PYTHONPATH="/tmp/thegame-tree-deps:$ANALYSIS_PATH" "$PYTHON" \
  -P -m explainability.thegame.pile_tree fit \
  --data /tmp/strict-semantic-corpus.npz --output /tmp/strict-pile-rules.json

PYTHONPATH="/tmp/thegame-tree-deps:$ANALYSIS_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON" -P -m explainability.thegame.sparse_rule \
  --data /tmp/strict-semantic-corpus.npz --output /tmp/strict-sparse-rules.json
```

Freeze candidates before opening test results; evaluate full trajectories:

```bash
PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.pile_tree evaluate \
  --data /tmp/strict-semantic-corpus.npz \
  --models /tmp/strict-pile-rules.json /tmp/strict-sparse-rules.json \
  --checkpoint "$STRICT_CHECKPOINT" --games 1024 --seed 1940000 \
  --selected tree_8 tree_64 fraction sparse_2 sparse_4 sparse_5 \
  --output /tmp/strict-semantic-evaluation.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.pile_tree evaluate \
  --data /tmp/strict-semantic-corpus.npz --models /tmp/strict-sparse-rules.json \
  --checkpoint "$STRICT_CHECKPOINT" --games 1024 --seed 1940000 \
  --selected rounded_4 rounded_5 \
  --baseline-results /tmp/strict-semantic-evaluation.json \
  --output /tmp/strict-rounded-evaluation.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.summarize_semantic \
  --data /tmp/strict-semantic-corpus.npz \
  --evaluations /tmp/strict-semantic-evaluation.json /tmp/strict-rounded-evaluation.json \
  --models /tmp/strict-semantic-trees.json /tmp/strict-pile-rules.json /tmp/strict-sparse-rules.json \
  --output-dir explainability/thegame/results
```

Unlike merely resetting a global seed before each policy, these new evaluations
preconstruct each initial game with `random.seed(seed + game_index)` and preserve
game order. Strategy-dependent randomness cannot shift later deals. The baseline
reuse option checks seed, game count, and checkpoint path before accepting scores.

Tests (including standalone/exported-rule parity on reachable states):

```bash
uv run pytest --capture=no -q tests/test_thegame_semantic_tree.py
```

## Calculator-free tactics

These experiments use the previous corpus only to suggest Boolean trees;
human-readable tactics are then judged primarily by complete-game scores.
The previous score formula is a reference, not part of any discrete policy.
Keep the development and confirmation seeds separate.

```bash
PYTHONPATH="/tmp/thegame-tree-deps:$ANALYSIS_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON" -P -m explainability.thegame.discrete_rules fit \
  --data /tmp/strict-semantic-corpus.npz --output /tmp/strict-discrete-models.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.discrete_rules evaluate \
  --models /tmp/strict-discrete-models.json --games 256 --seed 2040000 --development \
  --selected compact_3_w1 compact_4_w1 compact_6_w1 reserve_1 reserve_2 reserve_3 \
    endpoint_2 endpoint_reserve_2 constrained_2 \
  --output /tmp/strict-discrete-development.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.discrete_rules evaluate \
  --models /tmp/strict-discrete-models.json --games 256 --seed 2040000 --development \
  --selected run_3 setup_5 setup_10 hardest_3 run_reserve_3 setup_reserve_3 hardest_reserve_3 \
  --output /tmp/strict-human-development.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.discrete_rules evaluate \
  --models /tmp/strict-discrete-models.json --checkpoint "$STRICT_CHECKPOINT" \
  --games 1024 --seed 2140000 --selected hardest_3 reserve_3 \
  --output /tmp/strict-human-confirmation.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.summarize_human \
  --development /tmp/strict-discrete-development.json /tmp/strict-human-development.json \
  --confirmation /tmp/strict-human-confirmation.json --models /tmp/strict-discrete-models.json \
  --output-dir explainability/thegame/results

uv run pytest --capture=no -q tests/test_thegame_human_rules.py tests/test_thegame_semantic_tree.py
```

Development JSON from early invocations used the historical key
`delta_vs_teacher` for differences against the compact formula, because no neural
teacher was evaluated in screening. The summary archive renames that field to
`delta_vs_reference` and explicitly records which reference was used. Differences
against lowest-cost are separately recomputed from the paired per-game scores.

## Refining awkward-card play

The original human rule is retained as `previous` in every comparison. `grid`
screens 50 target/threshold/budget combinations; `assigned` and `reverse`
screen 12 and six additional variants respectively. No neural inference is
needed. The deterministic evaluator is parity-tested against full rollouts.

```bash
PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.awkward_refinement \
  --stage screen --games 256 --seed 2240000 --selected grid \
  --output /tmp/awkward-refinement-screen.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.awkward_refinement \
  --stage screen --games 256 --seed 2240000 --selected assigned \
  --output /tmp/awkward-refinement-assigned.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.awkward_refinement \
  --stage screen --games 256 --seed 2240000 --selected reverse \
  --output /tmp/awkward-refinement-reverse.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.awkward_refinement \
  --stage validation --games 1024 --seed 2440000 \
  --selected worst_t0_b3 affordable_t20_b5 affordable_t15_b3 affordable_t20_b3 \
    worst_t0_b3_reverse_endpoint worst_t0_b3_reverse_chain affordable_t20_b5_reverse_chain \
  --output /tmp/awkward-refinement-validation.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.awkward_refinement \
  --stage confirmation --games 4096 --seed 2640000 \
  --selected worst_t0_b3 affordable_t20_b3 \
  --output /tmp/awkward-refinement-confirmation.json

PYTHONPATH="$ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.summarize_refinement \
  --screen /tmp/awkward-refinement-screen.json /tmp/awkward-refinement-assigned.json \
    /tmp/awkward-refinement-reverse.json \
  --validation /tmp/awkward-refinement-validation.json \
  --confirmation /tmp/awkward-refinement-confirmation.json \
  --output-dir explainability/thegame/results

uv run pytest --capture=no -q tests/test_thegame_awkward_refinement.py \
  tests/test_thegame_human_rules.py tests/test_thegame_semantic_tree.py
```

## Omni extra-card decisions

Use the **Omni** historical source overlay, not the Strict one. This study uses
`YOLO-yay2/step-900.pth` and source tag
`run-source/YOLO-yay2-20260901T062612Z-604c1b8083e4`. The active checkout has model
source differences, so source priority matters.

```bash
OMNI_ANALYSIS_PATH="$OMNI_SOURCE:$PWD"

PYTHONPATH="$OMNI_ANALYSIS_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON" -P -m explainability.thegame.omni_stopping collect \
  --checkpoint "$OMNI_CHECKPOINT" --data /tmp/omni-stop-corpus.npz \
  --games 1024 --seed 2840000

PYTHONPATH="/tmp/thegame-tree-deps:$OMNI_ANALYSIS_PATH" \
  "$PYTHON" -P -m explainability.thegame.omni_stopping fit \
  --data /tmp/omni-stop-corpus.npz --output /tmp/omni-stop-rules.json

PYTHONPATH="$OMNI_ANALYSIS_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON" -P -m explainability.thegame.omni_stopping evaluate \
  --checkpoint "$OMNI_CHECKPOINT" --data /tmp/omni-stop-corpus.npz \
  --models /tmp/omni-stop-rules.json --games 1024 --seed 2940000 \
  --selected always_stop cost_3 cost_4 phase_exact phase_rounded \
  --output /tmp/omni-stop-evaluation.json

PYTHONPATH="$OMNI_ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.omni_stopping greedy \
  --data /tmp/omni-stop-corpus.npz --models /tmp/omni-stop-rules.json \
  --games 2048 --seed 3040000 --selected cost_3 cost_4 phase_exact phase_rounded \
  --output /tmp/omni-stop-greedy.json

PYTHONPATH="$OMNI_ANALYSIS_PATH" "$PYTHON" -P -m explainability.thegame.summarize_omni_stopping \
  --data /tmp/omni-stop-corpus.npz --models /tmp/omni-stop-rules.json \
  --evaluation /tmp/omni-stop-evaluation.json --greedy /tmp/omni-stop-greedy.json \
  --output-dir explainability/thegame/results

uv run pytest --capture=no -q tests/test_thegame_omni_stopping.py
```

The corpus split is 512 training games, 256 validation games, and 256 test games.
Only decisions with both a card move and `x` are labeled. Hybrids never skip
compulsory plays. Own-hand-only stopping rules can also run without Omni's
other-hand and upcoming-draw information; invariance tests cover that contract.

## Configuration gap search

This experiment does not load neural checkpoints or change production rules.
The experimental subclass retains reverse-ten jumps, the usual player-dependent
hand size, and the one-card minimum after the draw pile empties. Half the piles
ascend and half descend. The generalized formula is a probe, not a calibrated
or optimal policy for each configuration. All scores and source hashes are saved.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m explainability.thegame.config_gap --games 128 --seed 3240000 --workers 4 \
  --output explainability/thegame/results/config_gap_screen.json

# Configurations were frozen after screening, before fresh confirmation.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m explainability.thegame.config_gap --games 2048 --seed 3340000 --workers 4 \
  --configurations explainability/thegame/results/config_gap_selected.json \
  --output explainability/thegame/results/config_gap_confirmation.json

# Remaining high-ranked six-pile/max-150 settings from the completed screen.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m explainability.thegame.config_gap --games 2048 --seed 3340000 --workers 4 \
  --configurations explainability/thegame/results/config_gap_supplement_selected.json \
  --output explainability/thegame/results/config_gap_supplement_confirmation.json

# Finalist and default: untouched deals after all configuration selection.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m explainability.thegame.config_gap --games 4096 --seed 3640000 --workers 2 \
  --configurations explainability/thegame/results/config_gap_final_selected.json \
  --output explainability/thegame/results/config_gap_final_confirmation.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m explainability.thegame.config_gap --games 256 --seed 3440000 --workers 2 \
  --configurations explainability/thegame/results/config_gap_free_selected.json \
  --stopping 0 1 2 3 5 8 \
  --output explainability/thegame/results/config_gap_free_calibration.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m explainability.thegame.config_gap --games 2048 --seed 3540000 --workers 2 \
  --configurations explainability/thegame/results/config_gap_free_selected.json \
  --calibrated-from explainability/thegame/results/config_gap_free_calibration.json \
  --output explainability/thegame/results/config_gap_free_confirmation.json

.venv/bin/python -m explainability.thegame.summarize_config_gap
uv run pytest --capture=no tests/test_thegame_config_gap.py
```

The final Free comparison selects one stopping threshold per policy using only
calibration games. Its meaningful baseline is `greedy_calibrated`, not forced
stopping. Source and action parity, legal transitions, termination, hidden-state
independence, and score normalization are covered by the focused tests.
