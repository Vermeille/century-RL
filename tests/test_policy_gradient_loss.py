import math

import torch
import torch.nn.functional as F
from types import SimpleNamespace
from boardrl.rl.model.loss import (
    PolicyGradientLoss,
    EntropyBonus,
    ReverseEntropyBonus,
    SymmetricUniformKLPenalty,
    LinearEntropyBonus,
    ScheduledPerplexity,
    KLPenalty,
    AdaptiveKLPenalty,
    BootstrapValueMSELoss,
    BootstrapValueLogProbLoss,
)
from boardrl.training import Scheduler


def test_entropy_regularizer_zero_grad_at_uniform():
    logits = torch.zeros(4, requires_grad=True)
    sample = SimpleNamespace(action_idx=torch.tensor([0]), returns=torch.tensor([0.0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    policy = PolicyGradientLoss(weight="returns")
    entropy_bonus = EntropyBonus(strength=1.0)
    loss = policy([logits], pred_value, sample, training_state={}).objective + entropy_bonus(
        [logits], pred_value, sample, training_state={}
    ).objective
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-6)


def test_entropy_regularizer_drives_uniform_policy():
    torch.manual_seed(0)
    logits = torch.randn(5, requires_grad=True)
    sample = SimpleNamespace(action_idx=torch.tensor([0]), returns=torch.tensor([0.0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    policy = PolicyGradientLoss(weight="returns")
    entropy_bonus = EntropyBonus(strength=1.0)
    opt = torch.optim.SGD([logits], lr=0.2)
    for _ in range(400):
        opt.zero_grad()
        loss = policy([logits], pred_value, sample, training_state={}).objective + entropy_bonus(
            [logits], pred_value, sample, training_state={}
        ).objective
        loss.backward()
        opt.step()
    probs = logits.softmax(dim=0)
    assert torch.allclose(probs, torch.full_like(probs, 1 / probs.numel()), atol=1e-3)


def test_reverse_entropy_is_mean_reverse_kl_from_uniform():
    pred_policy = [
        torch.tensor([0.0, 2.0]),
        torch.tensor([-1.0, 0.0, 1.0]),
    ]
    sample = SimpleNamespace(action_idx=torch.tensor([0, 0]))
    pred_value = SimpleNamespace(mean=torch.zeros(2))

    objective = ReverseEntropyBonus(strength=0.3)(
        pred_policy, pred_value, sample, training_state={}
    ).objective
    expected = torch.stack(
        [
            -logits.log_softmax(0).mean() - math.log(logits.numel())
            for logits in pred_policy
        ]
    ).mean()

    assert torch.allclose(objective, 0.3 * expected)


def test_reverse_entropy_keeps_recovery_gradient_for_forgotten_action():
    logits = torch.tensor([0.0, -30.0], requires_grad=True)
    sample = SimpleNamespace(action_idx=torch.tensor([0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))

    objective = ReverseEntropyBonus(strength=1.0)(
        [logits], pred_value, sample, training_state={}
    ).objective
    objective.backward()

    assert torch.allclose(logits.grad[1], torch.tensor(-0.5))


def test_symmetric_uniform_kl_averages_both_kl_directions():
    pred_policy = [
        torch.tensor([0.0, 2.0]),
        torch.tensor([-1.0, 0.0, 1.0]),
    ]
    sample = SimpleNamespace(action_idx=torch.tensor([0, 0]))
    pred_value = SimpleNamespace(mean=torch.zeros(2))

    objective = SymmetricUniformKLPenalty(strength=0.3)(
        pred_policy, pred_value, sample, training_state={}
    ).objective
    per_state = []
    for logits in pred_policy:
        log_probs = logits.log_softmax(0)
        probs = log_probs.exp()
        log_count = math.log(logits.numel())
        forward_kl = (probs * log_probs).sum() + log_count
        reverse_kl = -log_probs.mean() - log_count
        per_state.append(0.5 * (forward_kl + reverse_kl))

    assert torch.allclose(objective, 0.3 * torch.stack(per_state).mean())


def test_symmetric_uniform_kl_recovers_a_forgotten_action():
    logits = torch.tensor([0.0, -30.0], requires_grad=True)
    sample = SimpleNamespace(action_idx=torch.tensor([0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))

    objective = SymmetricUniformKLPenalty(strength=1.0)(
        [logits], pred_value, sample, training_state={}
    ).objective
    objective.backward()

    assert torch.allclose(logits.grad[1], torch.tensor(-0.25), atol=1e-5)


def test_scheduled_perplexity_can_use_reverse_entropy():
    loss = ScheduledPerplexity(
        start=0.5,
        init_strength=0.1,
        regularizer_factory=ReverseEntropyBonus,
    )

    assert type(loss.regularizer) is ReverseEntropyBonus


def test_scheduled_perplexity_uses_a_progress_scheduler():
    loss = ScheduledPerplexity(
        start=1.0,
        end=0.5,
        schedule=Scheduler(start=0.25, end=0.75),
    )

    assert loss.target_ppl(0.0) == 1.0
    assert loss.target_ppl(0.5) == 0.75
    assert loss.target_ppl(1.0) == 0.5


def test_linear_entropy_bonus_interpolates_strength():
    loss = LinearEntropyBonus(start=0.01, end=0.001)

    assert loss.strength(-1.0) == 0.01
    assert loss.strength(0.0) == 0.01
    assert loss.strength(0.5) == 0.0055
    assert loss.strength(1.0) == 0.001
    assert loss.strength(2.0) == 0.001


def test_scheduled_perplexity_normalizes_uniform_and_single_action_policies():
    assert torch.allclose(
        ScheduledPerplexity.normalized_perplexity(
            [torch.zeros(4), torch.tensor([0.0])]
        ),
        torch.tensor(1.0),
    )
    assert torch.allclose(
        ScheduledPerplexity.normalized_perplexity([torch.tensor([0.0])]),
        torch.tensor(0.0),
    )


def test_scheduled_perplexity_uses_baseline_and_asymmetric_control():
    loss = ScheduledPerplexity(
        start=0.5,
        ppl_beta=0.0,
        init_strength=0.1,
        baseline_ratio=0.2,
        adaptation_rate=0.05,
    )

    loss.update_strength(measured_ppl=0.0, target_ppl=0.5)
    assert loss.entropy.strength > loss.init_strength
    assert loss.entropy.strength <= loss.max_strength

    increased_strength = loss.entropy.strength
    loss.update_strength(measured_ppl=1.0, target_ppl=0.5)
    assert loss.entropy.strength < increased_strength
    assert loss.entropy.strength > loss.baseline_strength


def test_scheduled_perplexity_deadband_relaxes_toward_baseline():
    loss = ScheduledPerplexity(
        start=0.5,
        init_strength=0.1,
        baseline_ratio=0.2,
        ppl_beta=0.0,
        deadband=0.02,
    )

    loss.update_strength(measured_ppl=0.49, target_ppl=0.5)

    assert loss.entropy.strength < loss.init_strength
    assert loss.entropy.strength > loss.baseline_strength


def test_scheduled_perplexity_can_relax_entropy_strength_to_zero():
    loss = ScheduledPerplexity(
        start=0.5,
        init_strength=0.1,
        baseline_ratio=0.0,
        ppl_beta=0.0,
    )

    for _ in range(10_000):
        loss.update_strength(measured_ppl=1.0, target_ppl=0.0)

    assert loss.entropy.strength < 1e-3


def test_scheduled_perplexity_can_freeze_controller_during_evaluation():
    loss = ScheduledPerplexity(start=0.5, init_strength=0.1)
    sample = SimpleNamespace(action_idx=torch.tensor([0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    logits = torch.tensor([1.0, -1.0], requires_grad=True)

    result = loss(
        [logits],
        pred_value,
        sample,
        {"progress": 0.5, "update_entropy_controller": False},
    )

    assert loss.ppl_ema is None
    assert loss.last_target_ppl == 0.275
    assert loss.last_ppl is not None
    assert loss.entropy.strength == loss.init_strength
    assert torch.isfinite(result.objective)
    assert result.metrics["perplexity"] == loss.last_ppl
    assert result.metrics["target"] == loss.last_target_ppl
    assert result.metrics["strength"] == loss.entropy.strength


def test_scheduled_perplexity_preserves_entropy_gradient():
    loss = ScheduledPerplexity(start=1.0, init_strength=0.1)
    logits = torch.tensor([1.0, -1.0], requires_grad=True)
    sample = SimpleNamespace(action_idx=torch.tensor([0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))

    result = loss([logits], pred_value, sample, {"progress": 0.0})
    result.objective.backward()

    assert torch.isfinite(result.objective)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_scheduled_perplexity_state_round_trip():
    saved = ScheduledPerplexity(
        start=0.5,
        init_strength=0.1,
        ppl_beta=0.0,
    )
    saved.update_strength(measured_ppl=0.0, target_ppl=0.5)
    saved.last_target_ppl = 0.5
    saved.last_ppl = 0.0

    restored = ScheduledPerplexity(start=0.5, init_strength=0.1)
    restored.load_state_dict(saved.state_dict())

    assert restored.entropy.strength == saved.entropy.strength
    assert restored.ppl_ema == saved.ppl_ema
    assert restored.last_target_ppl == saved.last_target_ppl
    assert restored.last_ppl == saved.last_ppl
    assert restored.last_ppl_ema == saved.last_ppl_ema
    assert restored.last_strength == saved.last_strength


def test_reverse_kl_regularizer_matches_manual():
    logits = torch.tensor([0.5, -0.5], requires_grad=True)
    reference_logits = torch.tensor([-0.25, 0.25])
    sample = SimpleNamespace(
        action_idx=torch.tensor([0]),
        returns=torch.tensor([1.0]),
        state=[None],
        reference_policy=[reference_logits],
    )

    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    policy = PolicyGradientLoss(weight="returns")
    kl_penalty = KLPenalty(strength=0.5)
    loss = policy([logits], pred_value, sample, training_state={}).objective + kl_penalty(
        [logits], pred_value, sample, training_state={}
    ).objective
    ce = F.cross_entropy(logits, sample.action_idx[0])
    reverse_kl = F.kl_div(
        F.log_softmax(reference_logits, dim=0),
        F.log_softmax(logits, dim=0),
        # KLPenalty treats each state as one batch item, even though this
        # standalone distribution is represented by a 1-D tensor.
        reduction="sum",
        log_target=True,
    )
    expected = ce + 0.5 * reverse_kl
    assert torch.allclose(loss, expected)


def test_kl_regularizer_batch_size_invariant():
    logits = torch.tensor([0.5, -0.5])
    reference_logits = torch.tensor([-0.5, 0.5])
    pred_value = SimpleNamespace(mean=torch.tensor([0.0, 0.0]))
    kl_penalty = KLPenalty(strength=0.5)

    one_sample = SimpleNamespace(reference_policy=[reference_logits])
    two_samples = SimpleNamespace(reference_policy=[reference_logits, reference_logits])

    one_loss = kl_penalty([logits], pred_value, one_sample, training_state={}).objective
    two_loss = kl_penalty([logits, logits], pred_value, two_samples, training_state={}).objective

    assert torch.allclose(one_loss, two_loss)


def test_reverse_kl_regularizer_matches_manual_for_variable_move_counts():
    pred_policy = [
        torch.tensor([0.5, -0.5], requires_grad=True),
        torch.tensor([1.0, 0.0, -1.0], requires_grad=True),
    ]
    reference_policy = [
        torch.tensor([-0.25, 0.25]),
        torch.tensor([0.1, 0.2, 0.3]),
    ]
    pred_value = SimpleNamespace(mean=torch.tensor([0.0, 0.0]))
    sample = SimpleNamespace(reference_policy=reference_policy)

    loss = KLPenalty(strength=0.5)(
        pred_policy, pred_value, sample, training_state={}
    ).objective
    manual = sum(
        F.kl_div(
            F.log_softmax(ref_logit, dim=0),
            F.log_softmax(logit, dim=0),
            reduction="sum",
            log_target=True,
        )
        for logit, ref_logit in zip(pred_policy, reference_policy)
    )

    assert torch.allclose(loss, 0.5 * manual / len(reference_policy))


def test_adaptive_kl_increases_strength_above_fixed_target():
    loss = AdaptiveKLPenalty(
        target=0.01,
        init_strength=0.1,
        adaptation_rate=0.05,
        deadband=0.0,
    )
    pred_policy = [torch.tensor([1.0, -1.0])]
    reference_policy = [torch.tensor([-1.0, 1.0])]
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    sample = SimpleNamespace(reference_policy=reference_policy)

    loss(pred_policy, pred_value, sample, training_state={})

    assert loss.last_kl > loss.target
    assert loss.kl.strength > loss.init_strength


def test_adaptive_kl_relaxes_but_not_below_base_strength():
    loss = AdaptiveKLPenalty(
        target=0.01,
        init_strength=0.1,
        deadband=0.0,
    )
    pred_policy = [torch.tensor([1.0, -1.0])]
    reference_policy = [torch.tensor([-1.0, 1.0])]
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    sample = SimpleNamespace(reference_policy=reference_policy)

    loss(pred_policy, pred_value, sample, training_state={})
    increased_strength = loss.kl.strength

    loss(
        [torch.tensor([0.0, 0.0])],
        pred_value,
        SimpleNamespace(reference_policy=[torch.tensor([0.0, 0.0])]),
        training_state={},
    )

    assert loss.last_kl < loss.target
    assert increased_strength > loss.init_strength
    assert loss.kl.strength < increased_strength
    assert loss.kl.strength >= loss.init_strength


def test_adaptive_kl_has_fixed_target_and_can_freeze_controller():
    loss = AdaptiveKLPenalty(target=0.1, init_strength=0.1)
    logits = torch.tensor([1.0, -1.0], requires_grad=True)
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    sample = SimpleNamespace(reference_policy=[torch.tensor([-1.0, 1.0])])

    result = loss(
        [logits],
        pred_value,
        sample,
        {"progress": 0.9, "update_kl_controller": False},
    )
    result.objective.backward()

    assert loss.last_kl is not None
    assert loss.kl.strength == loss.init_strength
    assert result.metrics["kl"] == loss.last_kl
    current_prob = logits.softmax(dim=0)
    reference_prob = sample.reference_policy[0].softmax(dim=0)
    expected_total_variation = 0.5 * (current_prob - reference_prob).abs().sum()
    assert math.isclose(
        result.metrics["total_variation"], expected_total_variation.item()
    )
    assert result.metrics["target"] == loss.target
    assert result.metrics["strength"] == loss.kl.strength
    assert torch.isfinite(logits.grad).all()


def test_adaptive_kl_state_round_trip():
    saved = AdaptiveKLPenalty(target=0.01, init_strength=0.1)
    saved.update_strength(measured_kl=0.5)
    saved.last_kl = 0.5

    restored = AdaptiveKLPenalty(target=0.01, init_strength=0.1)
    restored.load_state_dict(saved.state_dict())

    assert restored.kl.strength == saved.kl.strength
    assert restored.last_target_kl == saved.last_target_kl
    assert restored.last_kl == saved.last_kl
    assert restored.last_strength == saved.last_strength


def test_vectorized_regularizers_have_finite_gradients_with_padding():
    pred_policy = [
        torch.tensor([0.5, -0.5], requires_grad=True),
        torch.tensor([1.0, 0.0, -1.0], requires_grad=True),
    ]
    reference_policy = [
        torch.tensor([-0.25, 0.25]),
        torch.tensor([0.1, 0.2, 0.3]),
    ]
    pred_value = SimpleNamespace(mean=torch.tensor([0.0, 0.0]))
    sample = SimpleNamespace(
        action_idx=torch.tensor([0, 1]),
        reference_policy=reference_policy,
    )

    training_state = {}
    loss = EntropyBonus(strength=0.5)(
        pred_policy, pred_value, sample, training_state=training_state
    ).objective
    loss = loss + KLPenalty(strength=0.5)(
        pred_policy, pred_value, sample, training_state=training_state
    ).objective

    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(logit.grad).all() for logit in pred_policy)


def test_policy_gradient_loss_with_normalized_gae():
    logits = torch.zeros(2, requires_grad=True)
    sample = SimpleNamespace(
        action_idx=torch.tensor([0]),
        normalized_gae=torch.tensor([1.0]),
    )
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    policy = PolicyGradientLoss(weight="normalized_gae")
    loss = policy([logits], pred_value, sample, training_state={}).objective
    expected = F.cross_entropy(logits, sample.action_idx[0], label_smoothing=0.002)
    assert torch.allclose(loss, expected)


def test_ppo_reports_importance_ratio_and_clip_fraction():
    pred_policy = [
        torch.tensor([math.log(4.0), 0.0]),
        torch.tensor([0.0, math.log(4.0)]),
    ]
    sample = SimpleNamespace(
        action_idx=torch.tensor([0, 0]),
        normalized_gae=torch.tensor([1.0, -1.0]),
        action_distribution=[torch.zeros(2), torch.zeros(2)],
    )
    pred_value = SimpleNamespace(mean=torch.zeros(2))

    result = PolicyGradientLoss(
        weight="normalized_gae",
        drift="ppo",
        imp_ratio_clip=0.2,
    )(pred_policy, pred_value, sample, training_state={})

    assert torch.allclose(result.metrics["importance_ratio"], torch.tensor(1.0))
    assert torch.allclose(result.metrics["clip_fraction"], torch.tensor(1.0))


def test_policy_gradient_normalizer_state_round_trip():
    saved = PolicyGradientLoss(weight="returns", normalizer_alpha=0.9)
    saved.normalizer.update(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    restored = PolicyGradientLoss(weight="returns", normalizer_alpha=0.9)
    restored.load_state_dict(saved.state_dict())

    assert (
        restored.normalizer.running_mean.iter
        == saved.normalizer.running_mean.iter
    )
    assert restored.normalizer.running_var.iter == saved.normalizer.running_var.iter
    assert torch.equal(
        restored.normalizer.running_mean.running,
        saved.normalizer.running_mean.running,
    )
    assert torch.equal(
        restored.normalizer.running_var.running,
        saved.normalizer.running_var.running,
    )


def test_bootstrap_value_mse_loss_targets_td_lambda_mean():
    pred_value = SimpleNamespace(
        mean=torch.tensor([1.0, 3.0], requires_grad=True),
    )
    sample = SimpleNamespace(td_lambda=torch.tensor([2.0, 1.0]))
    loss = BootstrapValueMSELoss(strength=0.5)(
        [], pred_value, sample, training_state={}
    ).objective

    assert torch.allclose(loss, torch.tensor(1.25))


def test_bootstrap_value_log_prob_clips_to_rollout_distribution():
    pred_value = torch.distributions.Normal(
        torch.tensor([0.0, 0.0], requires_grad=True),
        torch.tensor([1.0, 1.0], requires_grad=True),
    )
    sample = SimpleNamespace(
        td_lambda=torch.tensor([20.0, 3.0]),
        reference_value=torch.tensor([10.0, 2.0]),
        reference_value_stddev=torch.tensor([2.0, 1.0]),
    )

    result = BootstrapValueLogProbLoss(strength=0.5)(
        [], pred_value, sample, training_state={}
    )
    expected = -0.5 * pred_value.log_prob(torch.tensor([14.0, 3.0])).mean()

    assert torch.allclose(result.objective, expected)
    assert torch.allclose(result.metrics["clip_ratio"], torch.tensor(0.5))


def test_bootstrap_value_log_prob_can_disable_clipping():
    pred_value = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    sample = SimpleNamespace(
        td_lambda=torch.tensor([20.0]),
        reference_value=torch.tensor([10.0]),
        reference_value_stddev=torch.tensor([2.0]),
    )

    result = BootstrapValueLogProbLoss(epsilon=None)(
        [], pred_value, sample, training_state={}
    )

    assert torch.allclose(result.objective, -pred_value.log_prob(sample.td_lambda).mean())
    assert result.metrics["clip_ratio"].item() == 0.0