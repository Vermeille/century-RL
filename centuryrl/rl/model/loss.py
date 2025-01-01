import torch
import torch.nn.functional as F
from centuryrl.rl.model.utils import js_div, jeffreys_div
from centuryrl.century.utils import RegisterByName

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
    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_idx)
        loss = 0
        for logit, act in zip(pred_policy, sample.action_idx):
            loss += F.cross_entropy(logit, act)
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
    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_policy) == len(sample.action_idx)
        loss = 0
        for logit, act, r in zip(pred_policy, sample.action_idx, sample.returns):
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)

            loss += r * F.cross_entropy(logit, act, reduction="none")
        return loss / len(sample.action_idx)


@loss_from_string.register("policy_gradient_with_baseline_loss")
class PolicyGradientWithBaselineLoss:
    def __call__(self, pred_policy, pred_value, sample):
        assert len(pred_value) == len(sample.returns)
        assert len(pred_policy) == len(sample.returns)
        advantage = sample.returns - pred_value

        loss = 0
        for logit, act, adv in zip(pred_policy, sample.action_idx, advantage):
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)

            loss += adv * F.cross_entropy(logit, act, reduction="none")
        return loss / len(sample.returns)


@loss_from_string.register("value_mse_loss")
class ValueMSELoss:
    def __call__(self, pred_policy, pred_value, sample):
        print("\npred", pred_value.mean, "\ntarget", sample.returns)
        return F.mse_loss(pred_value.mean, sample.returns)


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
    def __init__(self, model, discount: float = 0.98):
        self.model = model
        self.discount = discount

    def __call__(self, pred_policy, pred_value, sample):
        with torch.no_grad():
            bootstrap_value = self.model([n.state for n in sample.next]).value.mean
            bootstrap_value = torch.where(
                torch.tensor([n.final for n in sample.next]),
                torch.tensor(0.0),
                bootstrap_value,
            )
        target = sample.reward + self.discount * bootstrap_value
        print("\npred", pred_value.mean, "\ntarget", target)
        return F.mse_loss(pred_value.mean, target)
