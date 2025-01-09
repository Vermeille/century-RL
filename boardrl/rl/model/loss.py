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


@loss_from_string.register("imitation_js_loss")
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
class ImiationJSLoss:
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


@loss_from_string.register("policy_gradient_loss")
class PolicyGradientLoss:
    def __init__(self, label_smoothing: float = 0.0):
        self.label_smoothing = label_smoothing

    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_idx)
        loss = 0
        for logit, act, r in zip(pred_policy, sample.action_idx, sample.returns):
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)

            loss += (1 - self.label_smoothing) * r * F.cross_entropy(
                logit, act
            ) + self.label_smoothing * F.cross_entropy(
                logit, act, reduction="none", label_smoothing=1
            )
        return loss / len(sample.action_idx)


@loss_from_string.register("policy_gradient_with_baseline_loss")
class PolicyGradientWithBaselineLoss:
    def __init__(self, label_smoothing: float = 0.0, prev_model=None):
        assert prev_model is not None
        self.label_smoothing = label_smoothing
        self.prev_model = prev_model

    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.returns)
        # WARNING: NOT TODAY SATAN: Don't forget to detach the value function
        with torch.no_grad():
            advantage = sample.returns - self.prev_model(sample.state).value.mean

        loss = 0
        for logit, act, adv in zip(pred_policy, sample.action_idx, advantage):
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)

            loss += (1 - self.label_smoothing) * adv * F.cross_entropy(
                logit, act
            ) + self.label_smoothing * F.cross_entropy(logit, act, label_smoothing=1)
        return loss / len(sample.returns) * 1


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
    def __call__(self, pred_policy, pred_value, sample):
        print(
            "\nmean",
            pred_value.mean,
            "\nvar",
            pred_value.scale,
            "\ntarget",
            sample.returns,
        )
        return -pred_value.log_prob(sample.returns).mean()


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
