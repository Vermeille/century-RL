import torch.nn.functional as F
from centuryrl.rl.model.utils import js_div, jeffreys_div

loss_registry = {}


def register_loss(fn):
    loss_registry[fn.__name__] = fn
    return fn


@register_loss
def imitation_ce_loss(pred_policy, pred_value, sample):
    assert len(pred_policy) == len(sample.action_distribution)
    loss = 0
    for logit, act in zip(pred_policy, sample.action_distribution):
        logit = logit.unsqueeze(0)
        act = act.unsqueeze(0)

        loss += F.cross_entropy(logit, F.softmax(act, dim=1))
    return loss / len(sample.action_distribution)


@register_loss
def ce_loss(pred_policy, pred_value, sample):
    assert len(pred_policy) == len(sample.action_idx)
    loss = 0
    for logit, act in zip(pred_policy, sample.action_idx):
        loss += F.cross_entropy(logit, act)
    return loss / len(sample.action_distribution)


@register_loss
def imitation_jeffreys_loss(pred_policy, pred_value, sample):
    assert len(pred_policy) == len(sample.action_distribution)
    loss = 0
    for logit, act in zip(pred_policy, sample.action_distribution):
        logit = logit.unsqueeze(0)
        act = act.unsqueeze(0)

        loss += jeffreys_div(logit, act)
    return loss / len(sample.action_distribution)


@register_loss
def imitation_js_loss(pred_policy, pred_value, sample):
    assert len(pred_policy) == len(sample.action_distribution)
    loss = 0
    for logit, act in zip(pred_policy, sample.action_distribution):
        logit = logit.unsqueeze(0)
        act = act.unsqueeze(0)

        loss += js_div(logit, act)
    return loss / len(sample.action_distribution)


@register_loss
def imitation_mse_loss(pred_policy, pred_value, sample):
    assert len(pred_policy) == len(sample.action_distribution)
    loss = 0
    for logit, act in zip(pred_policy, sample.action_distribution):
        logit = logit.unsqueeze(0)
        act = act.unsqueeze(0)

        loss += F.mse_loss(logit, act)
    return loss / len(sample.action_distribution)


@register_loss
def imitation_kl_loss(pred_policy, pred_value, sample):
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


@register_loss
def imitation_reverse_kl_loss(pred_policy, pred_value, sample):
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


@register_loss
def policy_gradient_loss(pred_policy, pred_value, sample):
    assert len(pred_policy) == len(sample.action_idx)
    loss = 0
    for logit, act, r in zip(pred_policy, sample.action_idx, sample.returns):
        logit = logit.unsqueeze(0)
        act = act.unsqueeze(0)

        loss += r * F.cross_entropy(logit, act, reduction="none")
    return loss / len(sample.action_idx)


@register_loss
def policy_gradient_with_baseline_loss(pred_policy, pred_value, sample):
    assert len(pred_value) == len(sample.returns)
    assert len(pred_policy) == len(sample.returns)
    advantage = sample.returns - pred_value

    loss = 0
    for logit, act, adv in zip(pred_policy, sample.action_idx, advantage):
        logit = logit.unsqueeze(0)
        act = act.unsqueeze(0)

        loss += adv * F.cross_entropy(logit, act, reduction="none")
    return loss / len(sample.returns)


@register_loss
def value_mse_loss(pred_policy, pred_value, sample):
    return F.mse_loss(pred_value, sample.returns)


def loss_from_string(name):
    return loss_registry[name]
