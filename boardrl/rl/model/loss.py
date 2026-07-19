import math
import torch
import torch.nn.functional as F
from boardrl.rl.model.utils import js_div, jeffreys_div
from boardrl.utils import RegisterByName

loss_from_string = RegisterByName()


@loss_from_string.register("imitation_ce_loss")
class ImitationCELoss:
    supports_off_policy = True
    supports_partial_trajectories = True
    needs_reference_policy_value = False

    def __call__(self, pred_policy, pred_value, sample, training_state):
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

    def __call__(self, pred_policy, pred_value, sample, training_state):
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

    def __call__(self, pred_policy, pred_value, sample, training_state):
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

    def __call__(self, pred_policy, pred_value, sample, training_state):
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

    def __call__(self, pred_policy, pred_value, sample, training_state):
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

    def __call__(self, pred_policy, pred_value, sample, training_state):
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

    def __call__(self, pred_policy, pred_value, sample, training_state):
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


@torch.jit.script
def ppo(imp_ratio, advantage, clip_val: float, rectification: float):
    mask = ((advantage > 0) & (imp_ratio > 1 + clip_val)) | (
        (advantage < 0) & (imp_ratio < 1 - clip_val)
    )
    imp_ratio[mask] *= -rectification
    return imp_ratio * advantage


def pack(logits):
    maxlen = max((p.numel() for p in logits), default=0)
    padded = logits[0].new_full((len(logits), maxlen), float("-inf"))

    for i, logit in enumerate(logits):
        padded[i, : logit.numel()] = logit
    return padded


def pack_cached(logits, training_state, name):
    cache = training_state.setdefault("_pack_cache", {})
    key = (name, id(logits))
    if key not in cache:
        cache[key] = pack(logits)
    return cache[key]


def importance_sampling(r, A):
    return r * A


@torch.jit.script
def spo(r, A, clip: float):
    return r * (A - A.abs() / clip * (r - 1))


@loss_from_string.register("policy_gradient_loss")
class PolicyGradientLoss:
    needs_reference_policy_value = False

    @property
    def supports_off_policy(self):
        return self.drift

    @property
    def supports_partial_trajectories(self):
        return self.weight in ["advantage"]

    def __init__(
        self,
        *,
        weight: str = "returns",
        discount_factor: float = None,
        normalizer_alpha: float = None,
        drift: str = None,
        imp_ratio_clip: float = 1.0,
        rectification: float = 0.0,
        strength: float = 1.0,
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
        self.drift = {
            "importance_sampling": importance_sampling,
            "ppo": (lambda r, A: ppo(r, A, clip_val=imp_ratio_clip, rectification=0)),
            "ppo-rb": (
                lambda r, A: ppo(
                    r, A, clip_val=imp_ratio_clip, rectification=rectification
                )
            ),
            "spo": (lambda r, A: spo(r, A, imp_ratio_clip)),
        }.get(drift)
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
        self.imp_ratio_clip = imp_ratio_clip
        self.rectification = rectification
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        assert len(pred_policy) == len(sample.action_idx)

        with torch.no_grad():
            weight = self.weight_fn(sample, self.discount_factor)

        if self.normalizer:
            self.normalizer.update(weight)
            weight = self.normalizer(weight)

        padded = pack_cached(pred_policy, training_state, "pred_policy")

        if self.drift:
            with torch.no_grad():
                top = (
                    F.log_softmax(padded, dim=1)
                    .gather(1, sample.action_idx[..., None])
                    .squeeze(1)
                )
                bottom = (
                    F.log_softmax(
                        pack_cached(
                            sample.action_distribution,
                            training_state,
                            "action_distribution",
                        ),
                        dim=1,
                    )
                    .gather(1, sample.action_idx[..., None])
                    .squeeze(1)
                )
                imp_ratio = torch.exp(top - bottom)
                weight = self.drift(imp_ratio, weight)

        return self.strength * torch.mean(
            weight * F.cross_entropy(padded, sample.action_idx, reduction="none")
        )


@loss_from_string.register("kl")
class KLPenalty:
    needs_reference_policy_value = True
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float = 0.0):
        self.strength = strength

    def divergence(self, pred_policy, sample, training_state):
        padded_policy = pack_cached(pred_policy, training_state, "pred_policy")
        padded_reference = pack_cached(
            sample.reference_policy, training_state, "reference_policy"
        )
        mask = torch.isfinite(padded_reference)
        log_policy = F.log_softmax(padded_policy, dim=1)
        log_reference = F.log_softmax(padded_reference, dim=1)
        safe_log_policy = torch.where(mask, log_policy, torch.zeros_like(log_policy))
        safe_log_reference = torch.where(
            mask, log_reference, torch.zeros_like(log_reference)
        )
        reference_prob = torch.where(
            mask, safe_log_reference.exp(), torch.zeros_like(safe_log_reference)
        )
        return (reference_prob * (safe_log_reference - safe_log_policy)).sum() / len(
            sample.reference_policy
        )

    def __call__(self, pred_policy, pred_value, sample, training_state):
        if self.strength is None or self.strength == 0:
            return torch.zeros((1,), device=pred_value.mean.device)

        return self.strength * self.divergence(pred_policy, sample, training_state)


@loss_from_string.register("adaptive_kl")
class AdaptiveKLPenalty:
    """Adapt KL penalty strength to keep KL below a fixed target.

    The controller increases the penalty when the measured KL is above the
    target and relaxes it toward the initial base strength otherwise. The
    target is intentionally constant; this loss does not depend on training
    progress.
    """

    needs_reference_policy_value = True
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(
        self,
        target: float,
        init_strength: float = 0.01,
        adaptation_rate: float = 0.05,
        deadband: float = 0.0,
    ):
        assert target >= 0.0
        assert init_strength > 0.0
        assert adaptation_rate > 0.0
        assert deadband >= 0.0

        self.target = target
        self.init_strength = init_strength
        self.max_strength = self.init_strength * 10.0
        self.adaptation_rate = adaptation_rate
        self.deadband = deadband

        self.kl = KLPenalty(init_strength)

        # Useful for logging and inspection.
        self.last_target_kl = target
        self.last_kl = None
        self.last_strength = init_strength

    def update_strength(self, measured_kl: float):
        error = measured_kl - self.target
        strength = self.kl.strength

        if error > self.deadband:
            # The policy is outside its KL budget.
            strength += self.adaptation_rate * self.init_strength * error
        else:
            # Recover from an earlier increase, but never below the base
            # strength used to initialize the controller.
            relax_rate = 0.1 * self.adaptation_rate
            strength += relax_rate * (self.init_strength - strength)

        strength = max(self.init_strength, min(self.max_strength, strength))
        self.kl.strength = strength
        self.last_strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        divergence = self.kl.divergence(pred_policy, sample, training_state)
        measured_kl = divergence.detach().item()

        if training_state.get("update_kl_controller", True):
            self.update_strength(measured_kl)

        self.last_kl = measured_kl
        penalty = self.kl.strength * divergence

        # Preserve the KL gradient while reporting the current controller
        # strength as the scalar value, matching ScheduledPerplexity.
        return penalty - penalty.detach() + penalty.new_tensor(measured_kl)


@loss_from_string.register("z_loss")
class ZLoss:
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float = 1e-6):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        s = self.strength / len(sample.action_idx)
        return s * sum(logit.pow(2).sum() for logit in pred_policy)


@loss_from_string.register("entropy_bonus")
class EntropyBonus:
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        padded = pack_cached(pred_policy, training_state, "pred_policy")
        mask = torch.isfinite(padded)
        log_probs = F.log_softmax(padded, dim=1)
        safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
        probs = torch.where(mask, safe_log_probs.exp(), torch.zeros_like(safe_log_probs))
        terms = probs * safe_log_probs
        return self.strength * terms.sum() / len(sample.action_idx)


@loss_from_string.register("linear_entropy_bonus")
class LinearEntropyBonus:
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, start: float, end: float = 0.0):
        self.start = start
        self.end = end

    def strength(self, progress: float) -> float:
        progress = max(0.0, min(1.0, progress))
        return self.start * (1 - progress) + self.end * progress

    def __call__(self, pred_policy, pred_value, sample, training_state):
        progress = training_state["progress"]
        return EntropyBonus(self.strength(progress))(
            pred_policy, pred_value, sample, training_state
        )


@loss_from_string.register("scheduled_perplexity")
class ScheduledPerplexity:
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(
        self,
        start: float,
        end: float = 0.05,
        init_strength: float = 0.01,
        baseline_ratio: float = 0.2,
        adaptation_rate: float = 0.05,
        ppl_beta: float = 0.99,
        deadband: float = 0.02,
    ):
        assert start >= 0.0
        assert end >= 0.0
        assert init_strength > 0.0
        assert 0.0 < baseline_ratio <= 1.0
        assert adaptation_rate > 0.0
        assert 0.0 <= ppl_beta < 1.0
        assert deadband >= 0.0

        self.start = start
        self.end = end

        self.init_strength = init_strength
        self.baseline_strength = init_strength * baseline_ratio

        # Hard clamps derived from init/baseline, not exposed as knobs.
        self.min_strength = self.baseline_strength * 0.1
        self.max_strength = self.init_strength * 10.0

        self.adaptation_rate = adaptation_rate
        self.ppl_beta = ppl_beta
        self.deadband = deadband

        self.entropy = EntropyBonus(init_strength)

        self.ppl_ema = None

        # Useful for logging.
        self.last_target_ppl = None
        self.last_ppl = None
        self.last_ppl_ema = None
        self.last_strength = init_strength

    @staticmethod
    def normalized_perplexity(policy):
        """
        policy: iterable of 1D logits tensors, one per state.
                Each tensor should already contain only legal-action logits.

        Returns normalized perplexity in [0, 1]:

            0 = deterministic
            1 = uniform over legal actions
        """

        vals = []

        for logits in policy:
            n = logits.numel()
            if n <= 1:
                continue

            # Use float32 for stable entropy math, while preserving device.
            logits = logits.float()

            logp = torch.log_softmax(logits, dim=0)
            p = logp.exp()
            entropy = -(p * logp).sum()

            ppl = entropy.exp()
            norm_ppl = (ppl - 1.0) / (n - 1.0)
            vals.append(norm_ppl)

        if not vals:
            # Degenerate batch: no state with >1 legal action.
            # Return a tensor on a reasonable device.
            first = next(iter(policy))
            return first.new_tensor(0.0)

        return torch.stack(vals).mean()

    def target_ppl(self, progress: float) -> float:
        assert 0.0 <= progress <= 1.0
        return self.start * (1.0 - progress) + self.end * progress

    def update_strength(self, measured_ppl: float, target_ppl: float):
        """
        Thermostat logic.

        If PPL is below target:
            increase entropy strength.

        If PPL is above target:
            slowly relax toward baseline_strength, not toward zero.
        """

        if self.ppl_ema is None:
            self.ppl_ema = measured_ppl
        else:
            self.ppl_ema = (
                self.ppl_beta * self.ppl_ema
                + (1.0 - self.ppl_beta) * measured_ppl
            )

        strength = self.entropy.strength

        error = target_ppl - self.ppl_ema

        if error > self.deadband:
            # Policy is too deterministic.
            #
            # Additive increase, scaled by init_strength so adaptation_rate
            # stays dimensionless.
            strength += self.adaptation_rate * self.init_strength * error

        else:
            # Policy is exploratory enough.
            #
            # Relax gently toward the nonzero baseline.
            # Downward motion is intentionally slower than upward correction.
            relax_rate = 0.1 * self.adaptation_rate
            strength += relax_rate * (self.baseline_strength - strength)

        strength = max(self.min_strength, min(self.max_strength, strength))

        self.entropy.strength = strength

        self.last_strength = strength
        self.last_ppl_ema = self.ppl_ema

    def __call__(self, pred_policy, pred_value, sample, training_state):
        progress = training_state["progress"]
        target = self.target_ppl(progress)

        ppl_tensor = self.normalized_perplexity(pred_policy)
        ppl = ppl_tensor.detach().item()

        # Optional escape hatch: useful if this loss is called during eval/logging.
        update_controller = training_state.get("update_entropy_controller", True)

        if update_controller:
            self.update_strength(
                measured_ppl=ppl,
                target_ppl=target,
            )

        self.last_target_ppl = target
        self.last_ppl = ppl

        e = self.entropy(pred_policy, pred_value, sample, training_state)

        # Preserve the gradient of the entropy bonus,
        # but report the current entropy strength as the scalar value.
        return e - e.detach() + e.new_tensor(self.entropy.strength)


@loss_from_string.register("reverse_entropy_bonus")
class ReverseEntropyBonus:
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        s = self.strength / len(sample.action_idx)
        return -s * sum(F.log_softmax(logit, dim=0).sum() for logit in pred_policy)


@loss_from_string.register("value_mse_loss")
class ValueMSELoss:
    supports_off_policy = True
    supports_partial_trajectories = False
    needs_reference_policy_value = False

    def __init__(self, strength: float = 1):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        assert pred_value.mean.shape == sample.returns.shape
        return self.strength * F.mse_loss(pred_value.mean, sample.returns)


@loss_from_string.register("value_log_prob")
class ValueLogProb:
    supports_off_policy = True
    supports_partial_trajectories = False
    needs_reference_policy_value = False

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
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

    def __call__(self, pred_policy, pred_value, sample, training_state):
        target = sample.td_lambda

        with torch.no_grad():
            clipped = torch.clamp(
                target,
                min=pred_value.mean - 2 * pred_value.stddev,
                max=pred_value.mean + 2 * pred_value.stddev,
            )
            target = clipped
        return -self.strength * pred_value.log_prob(target).mean()


@loss_from_string.register("bootstrap_value_mse_loss")
class BootstrapValueMSELoss:
    supports_off_policy = True
    supports_partial_trajectories = True
    needs_reference_policy_value = True

    def __init__(self, strength: float = 1):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        return self.strength * F.mse_loss(pred_value.mean, sample.td_lambda)


@loss_from_string.register("q_mse_loss")
class QMSELoss:
    supports_off_policy = True
    supports_partial_trajectories = True
    needs_reference_policy_value = True

    def __init__(self, discount_factor: float, renormalize: bool = False):
        self.discount_factor = discount_factor

    def __call__(self, pred_policy, pred_value, sample, training_state):
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
