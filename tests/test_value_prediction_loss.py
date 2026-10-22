"""Conditional-moment and gradient regressions for the PPO value loss."""

from types import SimpleNamespace

import pytest
import torch

from boardrl.rl.model.loss import BootstrapValueLogProbLoss
from boardrl.rl.model.model import NormalValueDistribution, OutcomeValueDistribution


@pytest.mark.parametrize("normalize_variance", [None, False, True])
def test_bootstrap_value_log_prob_matches_selected_gaussian_objective(normalize_variance):
    mean = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([0.01, 1.0, 4.0], dtype=torch.float64, requires_grad=True)
    value = NormalValueDistribution(mean, scale)
    target = torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float64)
    expected = -value.log_prob(target)
    if normalize_variance is not False:
        expected = value.variance.detach() * expected
    expected = 0.5 * expected.mean()
    options = {} if normalize_variance is None else {"normalize_variance": normalize_variance}
    actual = BootstrapValueLogProbLoss(strength=0.5, epsilon=None, **options)(
        [], value, SimpleNamespace(td_lambda=target), {}
    ).objective
    expected_gradients = torch.autograd.grad(expected, (mean, scale), retain_graph=True)
    actual_gradients = torch.autograd.grad(actual, (mean, scale))

    assert torch.allclose(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.allclose(actual_gradient, expected_gradient)


@pytest.mark.parametrize("stddev", [1e-12, 1e-6, 0.01, 1.0, 100.0])
def test_gaussian_mean_gradient_is_error_independently_of_uncertainty(stddev):
    mean = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=torch.float64,
                        requires_grad=True)
    scale = torch.full_like(mean, stddev, requires_grad=True)
    value = NormalValueDistribution(mean, scale)
    target = torch.zeros_like(mean)
    loss = value.prediction_loss(target)
    dmean, dscale = torch.autograd.grad(loss.sum(), (mean, scale))

    assert torch.isfinite(loss).all()
    assert torch.allclose(dmean, mean.detach() - target, atol=1e-12, rtol=1e-12)
    assert torch.allclose(dscale, scale.detach() - mean.detach().square() / scale.detach())


def test_gaussian_raw_uncertainty_gradient_does_not_explode_as_sigma_shrinks():
    raw = torch.tensor([[0.0, -3.0], [0.0, -10.0], [0.0, -30.0]],
                       dtype=torch.float64, requires_grad=True)
    value = NormalValueDistribution.from_raw(raw)
    value.prediction_loss(torch.ones(3, dtype=torch.float64)).sum().backward()

    assert torch.allclose(raw.grad[:, 0], -torch.ones(3, dtype=torch.float64))
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[:, 1].abs().max() <= 1
    assert raw.grad[-1, 1].item() == pytest.approx(-1.0)


def test_gaussian_loss_keeps_extreme_raw_uncertainty_finite():
    raw = torch.tensor([[0.0, -1000.0], [2.0, -100.0], [-3.0, 1000.0]],
                       requires_grad=True)
    target = torch.tensor([1.0, -1.0, 2.0])
    value = NormalValueDistribution.from_raw(raw)
    loss = value.prediction_loss(target)
    loss.sum().backward()

    assert (value.stddev > 0).all()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(raw.grad).all()
    assert torch.allclose(raw.grad[:, 0], raw.detach()[:, 0] - target)


def test_preconditioned_gaussian_loss_avoids_standardized_error_overflow():
    mean = torch.tensor([1e10], requires_grad=True)
    value = NormalValueDistribution(mean, torch.tensor([1e-20]))
    loss = value.prediction_loss(torch.zeros(1))
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert loss.item() == pytest.approx(0.5e20)
    assert torch.allclose(mean.grad, mean.detach())


def test_gaussian_loss_preserves_mean_and_variance_for_skewed_returns():
    # A Gaussian is only an approximation here, but the critic must still
    # predict the expected return, not a median-like location.
    targets = torch.tensor([-1.0, -1.0, -1.0, 5.0], dtype=torch.float64)
    empirical_mean = targets.mean().item()
    empirical_scale = targets.std(correction=0).item()
    for mu, sigma, mean_direction, scale_direction in [
        (empirical_mean, empirical_scale, 0, 0),
        (empirical_mean - 1, empirical_scale, -1, 0),
        (empirical_mean + 1, empirical_scale, 1, 0),
        (empirical_mean, empirical_scale / 2, 0, -1),
        (empirical_mean, empirical_scale * 2, 0, 1),
    ]:
        mean = torch.tensor(mu, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(sigma, dtype=torch.float64, requires_grad=True)
        loss = NormalValueDistribution(mean, scale).prediction_loss(targets).mean()
        dmean, dscale = torch.autograd.grad(loss, (mean, scale))
        if mean_direction:
            assert dmean.item() * mean_direction > 0
        else:
            assert abs(dmean.item()) < 1e-12
        if scale_direction:
            assert dscale.item() * scale_direction > 0
        elif mu == empirical_mean:
            assert abs(dscale.item()) < 1e-12


@pytest.mark.parametrize("epsilon", [None, 2.0])
@pytest.mark.parametrize("normalize_variance", [False, True])
def test_bootstrap_value_loss_preserves_rollout_target_clipping(epsilon, normalize_variance):
    value = NormalValueDistribution(torch.tensor([0.0, 0.0]), torch.tensor([2.0, 0.5]))
    sample = SimpleNamespace(td_lambda=torch.tensor([20.0, 3.0]),
                             reference_value=torch.tensor([10.0, 2.0]),
                             reference_value_stddev=torch.tensor([2.0, 1.0]))
    result = BootstrapValueLogProbLoss(
        strength=0.5, epsilon=epsilon, normalize_variance=normalize_variance
    )([], value, sample, {})
    target = sample.td_lambda if epsilon is None else torch.tensor([14.0, 3.0])
    expected = value.prediction_loss(target) if normalize_variance else -value.log_prob(target)
    assert torch.allclose(result.objective, 0.5 * expected.mean())
    assert result.metrics['clip_ratio'].item() == (0.0 if epsilon is None else 0.5)


@pytest.mark.parametrize("normalize_variance", [False, True])
def test_categorical_value_loss_keeps_fractional_target_cross_entropy(normalize_variance):
    logits = torch.tensor([[2.0, -1.0, 0.0], [-2.0, 1.0, 3.0]], requires_grad=True)
    value = OutcomeValueDistribution.from_raw(logits)
    target = torch.tensor([-0.3, 0.8])
    sample = SimpleNamespace(td_lambda=target)
    result = BootstrapValueLogProbLoss(
        epsilon=None, normalize_variance=normalize_variance
    )([], value, sample, {})
    assert torch.allclose(result.objective, -value.log_prob(target).mean())
    result.objective.backward()
    assert torch.isfinite(logits.grad).all()
