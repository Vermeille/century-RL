# Epoch Optimization Log

Reference config: `configs/thegame-ppo-tdmse.yaml`

Measurement command:

```bash
uv run python scripts/profile_epoch.py configs/thegame-ppo-tdmse.yaml --epoch 1 -x train.iterations=2
```

Notes:

- Measurements taken before the user stopped the unrelated GPU process are invalid and intentionally excluded.
- Game length varies as the agent samples different trajectories, so comparisons include both whole-epoch wall time and samples/second.
- The reference config contains `include_moves=False`; `PolicySamplingStrategy` now accepts that argument so the config runs.

## Valid Baseline

Captured after the unrelated GPU process was stopped.

- Wall time: `5.406488s`
- Training samples: `1742`
- Trace samples: `1934`
- Samples/second: `322.205`
- Stages:
  - self-play + returns + metrics: `1.309436s`
  - reference update: `0.002267s`
  - trainset + shuffle: `0.003743s`
  - reference annotation: `1.077788s`
  - train: `3.013254s`

## Rejected: Byte-Buffer Text Encoding

Change tried: replace Python `ord` list construction in `Model.text_encode` with `torch.frombuffer`.

- Wall time: `6.076105s`
- Training samples: `1875`
- Samples/second: `308.586`
- Multiplier vs baseline by samples/second: `0.958x`

Decision: reverted. It was slower for the full epoch and emitted a non-writable-buffer warning.

## Accepted: Reuse Rollout Model Outputs For Fresh Reference Annotation

Observation: for the reference config, `reference_model_update: "True"` copies the current model into the reference model immediately after self-play and before annotation. `policy_sampling` already evaluated the same model for most rollout states, so the raw policy logits and value can be carried through the rollout record and reused for reference annotation. The code falls back to model evaluation when cached rollout outputs are absent or when the reference model was not refreshed.

Measurement 1:

- Wall time: `4.615480s`
- Training samples: `1759`
- Samples/second: `381.109`
- Multiplier vs baseline by samples/second: `1.183x`
- Reference annotation stage: `0.039908s`

Measurement 2:

- Wall time: `4.935785s`
- Training samples: `1824`
- Samples/second: `369.546`
- Multiplier vs baseline by samples/second: `1.147x`
- Reference annotation stage: `0.044944s`

Decision: kept. The whole-epoch speedup is material in both repeated measurements.

Verification:

```bash
uv run pytest tests/test_returns.py tests/test_training_sample.py tests/test_strategies.py -q
```

Result: `52 passed`.

## Current-State Profile After Reference Cache

- Wall time: `5.086646s`
- Training samples: `1831`
- Trace samples: `2023`
- Samples/second: `359.962`
- Stages:
  - self-play + returns + metrics: `1.527114s`
  - reference update: `0.002572s`
  - trainset + shuffle: `0.009146s`
  - reference annotation: `0.049229s`
  - train: `3.498584s`

## Rejected: One-Pass Tokenization And Move Marker Scan

Change tried: combine `Model.text_encode` token ID construction with `@` move-position discovery to avoid scanning each game string twice.

- Wall time: `5.017035s`
- Training samples: `1772`
- Samples/second: `353.197`
- Multiplier vs current state by samples/second: `0.981x`

Decision: reverted. The full epoch was slightly faster in raw wall time only because it produced fewer samples; normalized by samples/second it was slower.

## Accepted: Vectorized KL Penalty

Observation: after reference annotation reuse, the training step dominates the epoch. The `kl` loss still looped over each variable-length policy tensor and called `F.kl_div` once per sample. Packing logits and masking padded entries computes the same KL in a single tensor expression.

Measurement 1:

- Wall time: `4.934268s`
- Training samples: `1807`
- Samples/second: `366.214`
- Multiplier vs current state by samples/second: `1.017x`

Measurement 2:

- Wall time: `4.752044s`
- Training samples: `1810`
- Samples/second: `380.889`
- Multiplier vs current state by samples/second: `1.058x`

Decision: kept. The average samples/second across two measurements is `373.552`, a `1.038x` multiplier over the current-state profile.

## Accepted: Vectorized PPO Importance Denominator

Observation: `PolicyGradientLoss` packed current logits for the PPO numerator, but still computed the denominator by looping over `sample.action_distribution` and applying `log_softmax` once per sample. Packing the sampled action distributions lets the denominator use the same vectorized gather path.

Measurement 1:

- Wall time: `4.639127s`
- Training samples: `1726`
- Samples/second: `372.053`
- Multiplier vs vectorized-KL average by samples/second: `0.996x`

Measurement 2:

- Wall time: `4.708812s`
- Training samples: `1898`
- Samples/second: `403.074`
- Multiplier vs vectorized-KL average by samples/second: `1.079x`

Decision: kept. The average samples/second across two measurements is `387.564`, a `1.038x` multiplier over the vectorized-KL average.

## Accepted: Per-Batch Packed Policy Cache

Observation: after vectorizing KL and PPO ratio computation, `pack()` became a visible cost and `pred_policy` was packed more than once per batch. A shared `training_state` per batch now caches packed tensors by input list identity, so policy-gradient and KL reuse the same packed current-policy tensor.

Measurement 1:

- Wall time: `4.287299s`
- Training samples: `1771`
- Samples/second: `413.081`
- Multiplier vs vectorized-PPO average by samples/second: `1.066x`
- Train stage: `2.777925s`

Measurement 2:

- Wall time: `4.519257s`
- Training samples: `1837`
- Samples/second: `406.483`
- Multiplier vs vectorized-PPO average by samples/second: `1.049x`
- Train stage: `3.032563s`

Decision: kept. The average samples/second across two measurements is `409.782`, a `1.057x` multiplier over the vectorized-PPO average.

## Accepted: Vectorized Entropy Bonus

Observation: after vectorizing KL/PPO and caching packed policies, `EntropyBonus` still looped over one policy tensor per sample. It now computes entropy from the packed current-policy tensor and masks padding to avoid `0 * -inf` NaNs.

Measurement 1:

- Wall time: `4.188276s`
- Training samples: `1836`
- Samples/second: `438.367`
- Multiplier vs packed-policy-cache average by samples/second: `1.070x`
- Train stage: `2.709542s`

Measurement 2:

- Wall time: `4.383519s`
- Training samples: `1825`
- Samples/second: `416.332`
- Multiplier vs packed-policy-cache average by samples/second: `1.016x`
- Train stage: `2.848159s`

Decision: kept. The average samples/second across two measurements is `427.350`, a `1.043x` multiplier over the packed-policy-cache average.

## Accepted: Reuse Policy State String In Rollout Record

Observation: `policy_sampling` serialized `display_with_moves()` for inference, then `Record` immediately serialized the same game state again for storage. The strategy now passes the exact state string through `info`, and `Record` falls back to serializing only when a strategy does not provide it.

Measurement 1:

- Wall time: `4.067620s`
- Training samples: `1850`
- Samples/second: `454.811`
- Multiplier vs vectorized-entropy average by samples/second: `1.064x`

Measurement 2:

- Wall time: `4.189397s`
- Training samples: `1797`
- Samples/second: `428.940`
- Multiplier vs vectorized-entropy average by samples/second: `1.004x`

Decision: kept. The average samples/second across two measurements is `441.876`, a `1.034x` multiplier over the vectorized-entropy average.

## Current-State Profile After State String Reuse

- Wall time: `4.259866s`
- Training samples: `1855`
- Trace samples: `2047`
- Samples/second: `435.460`
- Stages:
  - self-play + returns + metrics: `1.487056s`
  - reference update: `0.002575s`
  - trainset + shuffle: `0.008740s`
  - reference annotation: `0.042343s`
  - train: `2.719152s`

## Rejected: `pad_sequence` For Variable-Length Policy Packing

Change tried: replace the manual `pack()` loop with `torch.nn.utils.rnn.pad_sequence(..., padding_value=-inf)`.

Measurement 1:

- Wall time: `3.947303s`
- Training samples: `1776`
- Samples/second: `449.927`
- Multiplier vs current state by samples/second: `1.033x`

Measurement 2:

- Wall time: `4.198839s`
- Training samples: `1810`
- Samples/second: `431.071`
- Multiplier vs current state by samples/second: `0.990x`

Decision: reverted. The average samples/second was `440.499`, effectively noise versus the current-state profile and below the prior kept average for state string reuse.

## Rejected: Scalarized Reference Max-Q In Policy Sampling

Change tried: compute cached `reference_max_q` as scalar `raw_value + raw_policy.max().item() - raw_policy.mean().item()` instead of the existing tensor expression.

- Wall time: `4.348913s`
- Training samples: `1844`
- Samples/second: `424.014`
- Multiplier vs current state by samples/second: `0.974x`

Decision: reverted. The full epoch was slower, and the profile showed substantially more `.item()` calls.

## Rejected: List Indexing For Move Logit Selection

Change tried: replace per-sample `torch.tensor(moves_pos[i])` index tensors in `Model.forward` with direct Python list indexing.

Measurement 1:

- Wall time: `4.113440s`
- Training samples: `1842`
- Samples/second: `447.800`
- Multiplier vs latest kept average by samples/second: `1.013x`

Measurement 2:

- Wall time: `4.299273s`
- Training samples: `1872`
- Samples/second: `435.422`
- Multiplier vs latest kept average by samples/second: `0.985x`

Decision: reverted. The average samples/second was `441.611`, effectively identical to and slightly below the latest kept average.

## Final Verification

```bash
uv run pytest tests/test_policy_gradient_loss.py tests/test_returns.py tests/test_training_sample.py tests/test_strategies.py tests/test_selfplay_oop.py -q
```

Result: `64 passed`.

`git diff --check` passed.

The broader recommended subset still has unrelated existing Nim/RPS metric failures observed during verification.

## Stability Fix: Safe Padded KL/Entropy Arithmetic

Issue: the vectorized KL penalty and entropy bonus could create NaN intermediates on padded `-inf` logits before masking them out. That corrupted model parameters after the first optimizer step and crashed later self-play for `configs/thegame-ppo-tdmse.yaml` and `configs/adversarial-epsilon.yaml`.

Fix: keep `-inf` padding for correct `log_softmax`, but replace padded log-prob terms with finite zeros before subtraction and multiplication. Added a regression test that backprops through variable-length KL and entropy together.

Verification:

```bash
uv run pytest tests/test_policy_gradient_loss.py tests/test_returns.py tests/test_training_sample.py tests/test_strategies.py tests/test_selfplay_oop.py -q
```

Result: `64 passed`.

Crash repros completed through epoch 1:

```bash
uv run python main.py configs/thegame-ppo-tdmse.yaml -x train.iterations=1 -x train.save_every=1000000 --visdom-url offline
uv run python main.py configs/adversarial-epsilon.yaml -x train.iterations=1 -x train.save_every=1000000 --visdom-url offline
```

Fresh corrected-tree measurements:

- Measurement 1: `1872` samples, `4.180600s`, `447.783` samples/second.
- Measurement 2: `1833` samples, `4.071595s`, `450.192` samples/second.
- Average: `448.988` samples/second.

## Summary

- Valid baseline: `322.205` samples/second.
- Latest kept average: `448.988` samples/second.
- Cumulative speed multiplier: `1.393x`.
