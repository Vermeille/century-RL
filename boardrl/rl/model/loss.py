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
    needs_reference_policy_value = False

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
    needs_reference_policy_value = False

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
    needs_reference_policy_value = False

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
    needs_reference_policy_value = False

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
    needs_reference_policy_value = False

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
    needs_reference_policy_value = False

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
    needs_reference_policy_value = False

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


def weight_score(samples, discount_factor):
    return samples.score


def weight_returns(samples, discount_factor):
    return samples.returns


def weight_baseline_value(samples, discount_factor):
    return samples.returns - samples.reference_value


def weight_advantage(samples, discount_factor):
    return samples.advantage


def weight_gae(samples, discount_factor):
    return samples.gae


def weight_normalized_gae(samples, discount_factor):
    return samples.normalized_gae


class RunningStat:
    def __init__(self, beta):
        self.running = 0
        self.beta = beta
        self.iter = 0

    def update(self, x):
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
            self.running_mean.update(x.mean())
            self.running_var.update(x.var())

    def __call__(self, x):
        return (x - self.running_mean()) / (math.sqrt(self.running_var()) + 0.0001)


@loss_from_string.register("policy_gradient_loss")
class PolicyGradientLoss:
    needs_reference_policy_value = False

    @property
    def supports_off_policy(self):
        return self.importance_sampling

    @property
    def supports_partial_trajectories(self):
        return self.weight in ["advantage"]

    def __init__(
        self,
        *,
        weight: str = "returns",
        discount_factor: float = None,
        normalizer_alpha: float = None,
        importance_sampling: bool = False,
    ):
        assert weight in [
            "returns",
            "score",
            "advantage",
            "baseline_value",
            "gae",
            "normalized_gae",
        ]
        self.weight_fn = {
            "returns": weight_returns,
            "score": weight_score,
            "baseline_value": weight_baseline_value,
            "advantage": weight_advantage,
            "gae": weight_gae,
            "normalized_gae": weight_normalized_gae,
        }[weight]
        self.weight = weight
        self.discount_factor = discount_factor
        self.normalizer = None
        if normalizer_alpha is not None:
            self.normalizer = RunningNormalizer(normalizer_alpha)
        self.needs_reference_policy_value = weight in [
            "advantage",
            "baseline_value",
            "gae",
            "normalized_gae",
        ]
        self.importance_sampling = importance_sampling

    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_idx)

        with torch.no_grad():
            weight = self.weight_fn(sample, self.discount_factor)

        if self.normalizer:
            self.normalizer.update(weight)
            weight = self.normalizer(weight)

        maxlen = max((p.numel() for p in pred_policy), default=0)
        padded = pred_policy[0].new_full((len(pred_policy), maxlen), float("-inf"))

        for i, logits in enumerate(pred_policy):
            padded[i, : logits.numel()] = logits

        if self.importance_sampling:
            with torch.no_grad():
                top = (
                    F.log_softmax(padded, dim=1)
                    .gather(1, sample.action_idx[..., None])
                    .squeeze(1)
                )
                bottom = [
                    F.log_softmax(sample.action_distribution[i], dim=0)[
                        sample.action_idx[i]
                    ]
                    for i in range(len(sample.action_idx))
                ]
                bottom = torch.stack(bottom, dim=0)
                imp_ratio = torch.exp(top - bottom)
                weight *= imp_ratio

        return torch.mean(
            weight * F.cross_entropy(padded, sample.action_idx, reduction="none")
        )


@loss_from_string.register("kl")
class KLPenalty:
    needs_reference_policy_value = True
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float = 0.0):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        loss = torch.zeros((1,), device=pred_value.mean.device)
        for logit, ref_logit in zip(pred_policy, sample.reference_policy):
            if self.strength is not None and self.strength != 0:
                loss += F.kl_div(
                    F.log_softmax(ref_logit, dim=0),
                    F.log_softmax(logit, dim=0),
                    reduction="sum",
                    log_target=True,
                )
        return self.strength * loss


@loss_from_string.register("z_loss")
class ZLoss:
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float = 1e-6):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        s = self.strength / len(sample.action_idx)
        return s * sum(logit.pow(2).sum() for logit in pred_policy)


@loss_from_string.register("entropy_bonus")
class EntropyBonus:
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        s = self.strength / len(sample.action_idx)
        return -s * sum(entropy(logit, dim=0) for logit in pred_policy)


@loss_from_string.register("reverse_entropy_bonus")
class ReverseEntropyBonus:
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        s = self.strength / len(sample.action_idx)
        return -s * sum(F.log_softmax(logit, dim=0).sum() for logit in pred_policy)


@loss_from_string.register("value_mse_loss")
class ValueMSELoss:
    supports_off_policy = True
    supports_partial_trajectories = False
    needs_reference_policy_value = False

    def __init__(self, strength: float = 1):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        assert pred_value.mean.shape == sample.returns.shape
        return self.strength * F.mse_loss(pred_value.mean, sample.returns)


@loss_from_string.register("value_log_prob")
class ValueLogProb:
    supports_off_policy = True
    supports_partial_trajectories = False
    needs_reference_policy_value = False

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample):
        lp = pred_value.log_prob(sample.returns)
        return -self.strength * lp.mean()


@loss_from_string.register("bootstrap_mse_loss")
class BootstrapMSELoss:
    supports_off_policy = True
    supports_partial_trajectories = True
    needs_reference_policy_value = True

    def __init__(self, discount_factor: float, strength: float = 1, clip: float = -1):
        self.discount = discount_factor
        self.strength = strength
        self.clip = clip

    def __call__(self, pred_policy, pred_value, sample):
        target = sample.td_lambda
        if self.clip > 0:
            with torch.no_grad():
                target = torch.clamp(
                    target,
                    min=pred_value * (1 - self.clip),
                    max=pred_value * (1 + self.clip),
                )
        return self.strength * F.mse_loss(pred_value.mean, target)


@loss_from_string.register("q_mse_loss")
class QMSELoss:
    supports_off_policy = True
    supports_partial_trajectories = True
    needs_reference_policy_value = True

    def __init__(self, discount_factor: float, renormalize: bool = False):
        self.discount_factor = discount_factor

    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_idx)

        loss = 0
        for adv, act, v, r, nxt in zip(
            pred_policy,
            sample.action_idx,
            pred_value.mean,
            sample.reward,
            sample.next_reference_max_q,
        ):
            assert adv.ndim == 1
            loss += F.mse_loss(
                v + adv[act] - adv.mean(), r + self.discount_factor * nxt
            )

        return loss / len(sample.action_idx)
