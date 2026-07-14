import torch
import torch.nn.functional as F
from types import SimpleNamespace
from boardrl.rl.model.loss import (
    PolicyGradientLoss,
    EntropyBonus,
    LinearEntropyBonus,
    ScheduledPerplexity,
    KLPenalty,
    BootstrapValueMSELoss,
)


def test_entropy_regularizer_zero_grad_at_uniform():
    logits = torch.zeros(4, requires_grad=True)
    sample = SimpleNamespace(action_idx=torch.tensor([0]), returns=torch.tensor([0.0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    policy = PolicyGradientLoss(weight="returns")
    entropy_bonus = EntropyBonus(strength=1.0)
    loss = policy([logits], pred_value, sample, training_state={}) + entropy_bonus(
        [logits], pred_value, sample, training_state={}
    )
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
        loss = policy([logits], pred_value, sample, training_state={}) + entropy_bonus(
            [logits], pred_value, sample, training_state={}
        )
        loss.backward()
        opt.step()
    probs = logits.softmax(dim=0)
    assert torch.allclose(probs, torch.full_like(probs, 1 / probs.numel()), atol=1e-3)


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
        torch.tensor(0.5),
    )


def test_scheduled_perplexity_adapts_strength_in_log_space():
    loss = ScheduledPerplexity(
        start=0.8,
        ppl_beta=0.0,
        adaptation_rate=10.0,
        init_strength=0.1,
        min_strength=0.05,
        max_strength=0.2,
    )
    sample = SimpleNamespace(action_idx=torch.tensor([0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))

    low_ppl_logits = torch.tensor([10.0, -10.0], requires_grad=True)
    loss([low_ppl_logits], pred_value, sample, {"progress": 0.0})
    assert abs(loss.entropy.strength - 0.2) < 1e-12

    high_ppl_logits = torch.zeros(2, requires_grad=True)
    loss([high_ppl_logits], pred_value, sample, {"progress": 0.0})
    assert abs(loss.entropy.strength - 0.05) < 1e-12


def test_scheduled_perplexity_preserves_entropy_gradient():
    loss = ScheduledPerplexity(start=1.0, init_strength=0.1)
    logits = torch.tensor([1.0, -1.0], requires_grad=True)
    sample = SimpleNamespace(action_idx=torch.tensor([0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))

    result = loss([logits], pred_value, sample, {"progress": 0.0})
    result.backward()

    assert torch.isfinite(result)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_kl_regularizer_matches_manual():
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
    loss = policy([logits], pred_value, sample, training_state={}) + kl_penalty(
        [logits], pred_value, sample, training_state={}
    )
    ce = F.cross_entropy(logits, sample.action_idx[0])
    kl = F.kl_div(
        F.log_softmax(logits, dim=0),
        F.log_softmax(reference_logits, dim=0),
        reduction="batchmean",
        log_target=True,
    )
    expected = ce + 0.5 * kl
    assert torch.allclose(loss, expected)


def test_kl_regularizer_batch_size_invariant():
    logits = torch.tensor([0.5, -0.5])
    reference_logits = torch.tensor([-0.5, 0.5])
    pred_value = SimpleNamespace(mean=torch.tensor([0.0, 0.0]))
    kl_penalty = KLPenalty(strength=0.5)

    one_sample = SimpleNamespace(reference_policy=[reference_logits])
    two_samples = SimpleNamespace(reference_policy=[reference_logits, reference_logits])

    one_loss = kl_penalty([logits], pred_value, one_sample, training_state={})
    two_loss = kl_penalty([logits, logits], pred_value, two_samples, training_state={})

    assert torch.allclose(one_loss, two_loss)


def test_kl_regularizer_matches_manual_for_variable_move_counts():
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

    loss = KLPenalty(strength=0.5)(pred_policy, pred_value, sample, training_state={})
    manual = sum(
        F.kl_div(
            F.log_softmax(logit, dim=0),
            F.log_softmax(ref_logit, dim=0),
            reduction="sum",
            log_target=True,
        )
        for logit, ref_logit in zip(pred_policy, reference_policy)
    )

    assert torch.allclose(loss, 0.5 * manual / len(reference_policy))


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
    )
    loss = loss + KLPenalty(strength=0.5)(
        pred_policy, pred_value, sample, training_state=training_state
    )

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
    loss = policy([logits], pred_value, sample, training_state={})
    expected = F.cross_entropy(logits, sample.action_idx[0], label_smoothing=0.002)
    assert torch.allclose(loss, expected)


def test_bootstrap_value_mse_loss_targets_td_lambda_mean():
    pred_value = SimpleNamespace(
        mean=torch.tensor([1.0, 3.0], requires_grad=True),
    )
    sample = SimpleNamespace(td_lambda=torch.tensor([2.0, 1.0]))
    loss = BootstrapValueMSELoss(strength=0.5)(
        [], pred_value, sample, training_state={}
    )

    assert torch.allclose(loss, torch.tensor(1.25))
