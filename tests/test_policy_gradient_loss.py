import torch
import torch.nn.functional as F
from types import SimpleNamespace
from boardrl.rl.model.loss import (
    PolicyGradientLoss,
    EntropyBonus,
    KLPenalty,
)


def test_entropy_regularizer_zero_grad_at_uniform():
    logits = torch.zeros(4, requires_grad=True)
    sample = SimpleNamespace(action_idx=[torch.tensor(0)], returns=torch.tensor([0.0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    policy = PolicyGradientLoss(weight="returns")
    entropy_bonus = EntropyBonus(strength=1.0)
    loss = policy([logits], pred_value, sample) + entropy_bonus([logits], pred_value, sample)
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-6)


def test_entropy_regularizer_drives_uniform_policy():
    torch.manual_seed(0)
    logits = torch.randn(5, requires_grad=True)
    sample = SimpleNamespace(action_idx=[torch.tensor(0)], returns=torch.tensor([0.0]))
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    policy = PolicyGradientLoss(weight="returns")
    entropy_bonus = EntropyBonus(strength=1.0)
    opt = torch.optim.SGD([logits], lr=0.2)
    for _ in range(400):
        opt.zero_grad()
        loss = policy([logits], pred_value, sample) + entropy_bonus([logits], pred_value, sample)
        loss.backward()
        opt.step()
    probs = logits.softmax(dim=0)
    assert torch.allclose(probs, torch.full_like(probs, 1 / probs.numel()), atol=1e-3)


def test_kl_regularizer_matches_manual():
    logits = torch.tensor([0.5, -0.5], requires_grad=True)
    reference_logits = torch.tensor([1.0, 0.0])
    sample = SimpleNamespace(
        action_idx=[torch.tensor(0)],
        returns=torch.tensor([1.0]),
        state=[None],
        reference_policy=[reference_logits],
    )

    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    policy = PolicyGradientLoss(weight="returns")
    kl_penalty = KLPenalty(strength=0.5)
    loss = policy([logits], pred_value, sample) + kl_penalty([logits], pred_value, sample)
    ce = F.cross_entropy(logits, sample.action_idx[0], label_smoothing=0.002)
    kl = F.kl_div(
        F.log_softmax(logits, dim=0),
        F.log_softmax(reference_logits, dim=0),
        reduction="batchmean",
        log_target=True,
    )
    expected = ce + 0.5 * kl
    assert torch.allclose(loss, expected)


def test_policy_gradient_loss_with_normalized_gae():
    logits = torch.zeros(2, requires_grad=True)
    sample = SimpleNamespace(
        action_idx=[torch.tensor(0)],
        normalized_gae=torch.tensor([1.0]),
    )
    pred_value = SimpleNamespace(mean=torch.tensor([0.0]))
    policy = PolicyGradientLoss(weight="normalized_gae")
    loss = policy([logits], pred_value, sample)
    expected = F.cross_entropy(logits, sample.action_idx[0], label_smoothing=0.002)
    assert torch.allclose(loss, expected)
