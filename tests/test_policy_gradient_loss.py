import torch
from types import SimpleNamespace
from boardrl.rl.model.loss import PolicyGradientLoss


def test_entropy_regularizer_zero_grad_at_uniform():
    logits = torch.zeros(4, requires_grad=True)
    sample = SimpleNamespace(action_idx=[torch.tensor(0)], returns=torch.tensor([0.0]))
    loss_fn = PolicyGradientLoss(weight="returns", label_smoothing=1.0)
    loss = loss_fn([logits], None, sample)
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-6)


def test_entropy_regularizer_drives_uniform_policy():
    torch.manual_seed(0)
    logits = torch.randn(5, requires_grad=True)
    sample = SimpleNamespace(action_idx=[torch.tensor(0)], returns=torch.tensor([0.0]))
    loss_fn = PolicyGradientLoss(weight="returns", label_smoothing=1.0)
    opt = torch.optim.SGD([logits], lr=0.2)
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn([logits], None, sample)
        loss.backward()
        opt.step()
    probs = logits.softmax(dim=0)
    assert torch.allclose(probs, torch.full_like(probs, 1 / probs.numel()), atol=1e-3)
