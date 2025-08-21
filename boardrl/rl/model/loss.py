import math
import torch
import torch.nn.functional as F
from boardrl.rl.model.utils import js_div, jeffreys_div
from boardrl.utils import RegisterByName, entropy

loss_from_string = RegisterByName()


@loss_from_string.register("imitation_ce_loss")
class ImitationCELoss:
    supports_off_policy = True
    supports_partial_trajectories = True

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
    supports_off_policy = True
    supports_partial_trajectories = True

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
    supports_off_policy = True
    supports_partial_trajectories = True

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
    supports_off_policy = True
    supports_partial_trajectories = True

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
    supports_off_policy = True
    supports_partial_trajectories = True

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
    supports_off_policy = True
    supports_partial_trajectories = True

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
    supports_off_policy = True
    supports_partial_trajectories = True

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
    after = samples.reward + next_value * discount_factor
    for b, i, af in zip(current_value, range(len(samples.state)), after):
        print(
            f"{b.item():.2f} {samples.moves[i][samples.action_idx[i]]} {af.item():.2f} (r={samples.reward[i]:.2f})"
        )
    return after - current_value


class RunningStat:
    def __init__(self, beta):
        self.running = None
        self.beta = beta
        self.iter = 1

    def update(self, x):
        if self.running is None:
            self.running = 0.0
        else:
            self.running = self.beta * self.running + (1 - self.beta) * x
            self.iter += 1

    def __call__(self):
        return self.running / (1 - self.beta**self.iter)


class RunningNormalizer:
    def __init__(self, beta):
        self.running_mean = RunningStat(beta)
        self.running_var = RunningStat(beta)

    def update(self, x):
        if x.numel() > 3:
            self.running_mean.update(x.mean().item())
            self.running_var.update(x.var().item())

    def __call__(self, x):
        return (x - self.running_mean()) / (math.sqrt(self.running_var()) + 0.1)


@loss_from_string.register("policy_gradient_loss")
class PolicyGradientLoss:
    @property
    def supports_off_policy(self):
        return self.weight in ["advantage"]

    @property
    def supports_partial_trajectories(self):
        return self.weight in ["advantage"]

    def __init__(
        self,
        *,
        weight: str = "returns",
        label_smoothing: float = 0.0,
        renormalize: bool = False,
        discount_factor: float = None,
        prev_model=None,
        kl_strength: float = 0.0,
    ):
        assert weight in ["returns", "score", "advantage", "baseline_value"]
        self.label_smoothing = label_smoothing
        self.weight_fn = {
            "returns": weight_returns,
            "score": weight_score,
            "baseline_value": weight_baseline_value,
            "advantage": weight_advantage,
        }[weight]
        self.weight = weight
        self.renormalize = renormalize
        self.discount_factor = discount_factor
        self.prev_model = prev_model
        self.kl_strength = kl_strength
        self.normalizer = RunningNormalizer(0.999)

    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_idx)

        with torch.no_grad():
            weight = self.weight_fn(sample, self.prev_model, self.discount_factor)
            prev_policy = (
                self.prev_model(sample.state).policy
                if self.kl_strength is not None and self.kl_strength != 0
                else None
            )

        if self.renormalize:
            self.normalizer.update(weight)
            weight = self.normalizer(weight)
        prev_policy_iter = (
            prev_policy if prev_policy is not None else [None] * len(pred_policy)
        )

        loss = 0.0
        for logit, act, w, prev_logit in zip(
            pred_policy, sample.action_idx, weight, prev_policy_iter
        ):
            print(logit, act.item(), w)
            loss_step = (
                w * F.cross_entropy(logit, act, label_smoothing=0.002)
                + 1e-6 * logit.pow(2).sum()
            )

            if self.label_smoothing != 0:
                loss_step -= self.label_smoothing * entropy(logit, dim=0)

            if self.kl_strength is not None and self.kl_strength != 0:
                loss_step += self.kl_strength * F.kl_div(
                    F.log_softmax(prev_logit, dim=0),
                    F.log_softmax(logit, dim=0),
                    reduction="sum",
                    log_target=True,
                )
            loss += loss_step
        return loss / len(sample.action_idx)


@loss_from_string.register("value_mse_loss")
class ValueMSELoss:
    supports_off_policy = True
    supports_partial_trajectories = False

    def __init__(self, strength: float = 1):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        assert pred_value.mean.shape == sample.returns.shape
        return self.strength * F.mse_loss(pred_value.mean, sample.returns)


@loss_from_string.register("value_log_prob")
class ValueLogProb:
    supports_off_policy = True
    supports_partial_trajectories = False

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        lp = pred_value.log_prob(sample.returns)
        return -self.strength * lp.mean()


@loss_from_string.register("bootstrap_mse_loss")
class BootstrapMSELoss:
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, discount_factor: float, strength: float = 1, prev_model=None):
        self.prev_model = prev_model
        self.discount = discount_factor
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        with torch.no_grad():
            bootstrap_value = self.prev_model([n.state for n in sample.next]).value.mean
            bootstrap_value = torch.where(
                torch.tensor(
                    [n.final for n in sample.next], device=bootstrap_value.device
                ),
                torch.tensor(0.0, device=bootstrap_value.device),
                bootstrap_value,
            )
        target = sample.reward + self.discount * bootstrap_value
        return self.strength * F.mse_loss(pred_value.mean, target)


@loss_from_string.register("q_mse_loss")
class QMSELoss:
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(
        self, discount_factor: float, renormalize: bool = False, prev_model=None
    ):
        self.discount_factor = discount_factor
        self.prev_model = prev_model

    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_idx)

        with torch.no_grad():
            q = self.prev_model([n.state for n in sample.next]).q_value()
            q_next = torch.where(
                torch.tensor(
                    [n.final for n in sample.next], device=sample.reward.device
                ),
                torch.tensor(0.0, device=sample.action_idx.device),
                torch.stack([q.max() for q in q]),
            )

        loss = 0
        for adv, act, v, r, nxt in zip(
            pred_policy,
            sample.action_idx,
            pred_value.mean,
            sample.reward,
            q_next,
        ):
            assert adv.ndim == 1
            loss += F.mse_loss(
                v + adv[act] - adv.mean(), r + self.discount_factor * nxt
            )

        return loss / len(sample.action_idx)
