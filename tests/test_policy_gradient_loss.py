import torch
import torch.nn.functional as F
from types import SimpleNamespace
from boardrl.rl.model.loss import PolicyGradientLoss


def test_entropy_regularizer_zero_grad_at_uniform():
    logits = torch.zeros(4, requires_grad=True)
    sample = SimpleNamespace(action_idx=[torch.tensor(0)], returns=torch.tensor([0.0]))
    loss_fn = PolicyGradientLoss(
        weight="returns", label_smoothing=1.0, aux_logits_coef=0.0
    )
    loss = loss_fn([logits], None, sample)
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-6)


def test_entropy_regularizer_drives_uniform_policy():
    torch.manual_seed(0)
    logits = torch.randn(5, requires_grad=True)
    sample = SimpleNamespace(action_idx=[torch.tensor(0)], returns=torch.tensor([0.0]))
    loss_fn = PolicyGradientLoss(
        weight="returns", label_smoothing=1.0, aux_logits_coef=0.0
    )
    opt = torch.optim.SGD([logits], lr=0.2)
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn([logits], None, sample)
        loss.backward()
        opt.step()
    probs = logits.softmax(dim=0)
    assert torch.allclose(probs, torch.full_like(probs, 1 / probs.numel()), atol=1e-3)


def test_kl_regularizer_matches_manual():
    logits = torch.tensor([0.5, -0.5], requires_grad=True)
    prev_logits = torch.tensor([1.0, 0.0])
    sample = SimpleNamespace(
        action_idx=[torch.tensor(0)],
        returns=torch.tensor([1.0]),
        state=[None],
        reference_policy=[prev_logits],
    )

    loss_fn = PolicyGradientLoss(
        weight="returns",
        kl_strength=0.5,
        aux_logits_coef=0.0,
    )
    loss = loss_fn([logits], None, sample)
    ce = F.cross_entropy(logits, sample.action_idx[0], label_smoothing=0.002)
    kl = F.kl_div(
        F.log_softmax(logits, dim=0),
        F.log_softmax(prev_logits, dim=0),
        reduction="batchmean",
        log_target=True,
    )
    expected = ce + 0.5 * kl
    assert torch.allclose(loss, expected)
