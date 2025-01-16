import torch
import torch.nn.functional as F
from boardrl.rl.model.utils import js_div, jeffreys_div
from boardrl.utils import RegisterByName

loss_from_string = RegisterByName()


@loss_from_string.register("imitation_ce_loss")
class ImitationCELoss:
    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_distribution)
        loss = 0
        for logit, act in zip(pred_policy, sample.action_distribution):
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)

            loss += F.cross_entropy(logit, F.softmax(act, dim=1))
        return loss / len(sample.action_distribution)


@loss_from_string.register("ce_loss")
class CELoss:
    def __init__(self, label_smoothing: float = 0.0):
        self.label_smoothing = label_smoothing

    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_idx)
        loss = 0
        for logit, act in zip(pred_policy, sample.action_idx):
            loss += F.cross_entropy(logit, act, label_smoothing=self.label_smoothing)
        return loss / len(sample.action_distribution)


@loss_from_string.register("imitation_jeffreys_loss")
class ImitationJeffreysLoss:
    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_distribution)
        loss = 0
        for logit, act in zip(pred_policy, sample.action_distribution):
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)

            loss += jeffreys_div(logit, act)
        return loss / len(sample.action_distribution)


@loss_from_string.register("imitation_js_loss")
class ImitationJSLoss:
    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_distribution)
        loss = 0
        for logit, act in zip(pred_policy, sample.action_distribution):
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)

            loss += js_div(logit, act)
        return loss / len(sample.action_distribution)


@loss_from_string.register("imitation_mse_loss")
class ImitationMSELoss:
    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_distribution)
        loss = 0
        for logit, act in zip(pred_policy, sample.action_distribution):
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)

            loss += F.mse_loss(logit, act)
        return loss / len(sample.action_distribution)


@loss_from_string.register("imitation_kl_loss")
class ImitationKLLoss:
    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_distribution)
        loss = 0
        for logit, act in zip(pred_policy, sample.action_distribution):
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)

            loss += F.kl_div(
                F.log_softmax(logit, dim=1),
                F.log_softmax(act, dim=1),
                reduction="batchmean",
                log_target=True,
            )
        return loss / len(sample.action_distribution)


@loss_from_string.register("imitation_reverse_kl_loss")
class ImitationReverseKLLoss:
    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_distribution)
        loss = 0
        for logit, act in zip(pred_policy, sample.action_distribution):
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)

            loss += F.kl_div(
                F.log_softmax(act, dim=1),
                F.log_softmax(logit, dim=1),
                reduction="batchmean",
                log_target=True,
            )
        return loss / len(sample.action_distribution)


def weight_score(samples, prev_model, discount_factor):
    return samples.score


def weight_returns(samples, prev_model, discount_factor):
    return samples.returns


def weight_baseline_value(samples, prev_model, discount_factor):
    return samples.returns - prev_model(samples.state).value.mean


def weight_advantage(samples, prev_model, discount_factor):
    next_value = prev_model([n.state for n in samples.next]).value.mean
    next_value = torch.where(
        torch.tensor([n.final for n in samples.next], device=next_value.device),
        torch.tensor(0.0, device=next_value.device),
        next_value,
    )
    current_value = prev_model(samples.state).value.mean
    return (samples.reward + next_value * discount_factor) - current_value


class RunningStat:
    def __init__(self, beta):
        self.running = None
        self.beta = beta

    def update(self, x):
        if self.running is None:
            self.running = x
        else:
            self.running = self.beta * self.running + (1 - self.beta) * x

    def __call__(self):
        return self.running


class RunningNormalizer:
    def __init__(self, beta):
        self.running_mean = RunningStat(beta)
        self.running_std = RunningStat(beta)

    def update(self, x):
        if x.numel() > 3:
            self.running_mean.update(x.mean().item())
            self.running_std.update(x.std().item())

    def __call__(self, x):
        return (x - self.running_mean()) / (self.running_std() + 1e-4)


@loss_from_string.register("policy_gradient_loss")
class PolicyGradientLoss:
    def __init__(
        self,
        weight: str = "returns",
        label_smoothing: float = 0.0,
        renormalize: bool = False,
        discount_factor: float = None,
        prev_model=None,
    ):
        assert weight in ["returns", "score", "advantage", "baseline_value"]
        self.label_smoothing = label_smoothing
        self.weight_fn = {
            "returns": weight_returns,
            "score": weight_score,
            "baseline_value": weight_baseline_value,
            "advantage": weight_advantage,
        }[weight]
        self.renormalize = renormalize
        self.discount_factor = discount_factor
        self.prev_model = prev_model
        self.normalizer = RunningNormalizer(0.99)

    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_idx)
        loss = 0

        with torch.no_grad():
            weight = self.weight_fn(sample, self.prev_model, self.discount_factor)

        if self.renormalize:
            self.normalizer.update(weight)
            weight = self.normalizer(weight)

        for logit, act, w in zip(pred_policy, sample.action_idx, weight):
            # WARNING: There is an exp that makes all the returns positive.
            # This is not standard but negative returns seems to make training unstable.
            loss += (1 - self.label_smoothing) * w.exp() * F.cross_entropy(
                logit, act
            ) + self.label_smoothing * F.cross_entropy(logit, act, label_smoothing=1)
        return loss / len(sample.action_idx)


@loss_from_string.register("value_mse_loss")
class ValueMSELoss:
    def __init__(self, strength: float = 1):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        # print("\npred", pred_value.mean, "\ntarget", sample.returns)
        assert pred_value.mean.shape == sample.returns.shape
        return self.strength * F.mse_loss(pred_value.mean, sample.returns)


@loss_from_string.register("value_log_prob")
class ValueLogProb:
    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        lp = pred_value.log_prob(sample.returns)
        return -self.strength * lp.mean()


@loss_from_string.register("bootstrap_mse_loss")
class BootstrapMSELoss:
    def __init__(self, discount: float, strength: float = 1, prev_model=None):
        self.prev_model = prev_model
        self.discount = discount
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        with torch.no_grad():
            bootstrap_value = self.prev_model([n.state for n in sample.next]).value.mean
            bootstrap_value = torch.where(
                torch.tensor([n.final for n in sample.next]),
                torch.tensor(0.0),
                bootstrap_value,
            )
        target = sample.reward + self.discount * bootstrap_value
        return self.strength * F.mse_loss(pred_value.mean, target)


@loss_from_string.register("q_mse_loss")
class QMSELoss:
    def __init__(self, renormalize: bool = False):
        self.renormalize = renormalize
        self.normalizer = RunningNormalizer(0.99)

    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_idx)
        loss = 0

        if self.renormalize:
            self.normalizer.update(sample.returns)
            target = self.normalizer(sample.returns)
        else:
            target = sample.returns

        for logit, act, q in zip(pred_policy, sample.action_idx, sample.returns):
            loss += F.mse_loss(logit[act], q)
        return loss / len(sample.action_idx)
