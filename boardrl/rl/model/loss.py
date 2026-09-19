import math
from collections.abc import Callable
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from boardrl.rl.model.utils import js_div, jeffreys_div
from boardrl.schedules import Scheduler
from boardrl.utils import RegisterByName

loss_from_string = RegisterByName()


@dataclass
class LossResult:
    objective: torch.Tensor
    metrics: dict[str, float | torch.Tensor] = field(default_factory=dict)


class Loss:
    """Optimization objective with optional mutable training state."""

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError(f"{type(self).__name__} has no mutable state")


@loss_from_string.register("imitation_ce_loss")
class ImitationCELoss(Loss):
    """Cross-entropy against variable-length target policy logits."""

    supports_off_policy = True
    supports_partial_trajectories = True
    needs_reference_policy_value = False

    def __call__(self, pred_policy, pred_value, sample, training_state):
        assert len(pred_policy) == len(sample.action_distribution)
        targets = list(sample.action_distribution)
        assert all(
            prediction.shape == target.shape
            for prediction, target in zip(pred_policy, targets)
        )

        padding = torch.finfo(pred_policy[0].dtype).min
        predictions = pad_sequence(
            pred_policy,
            batch_first=True,
            padding_value=padding,
        )
        targets = pad_sequence(
            targets,
            batch_first=True,
            padding_value=padding,
        )
        return LossResult(F.cross_entropy(predictions, F.softmax(targets, dim=1)))


@loss_from_string.register("ce_loss")
class CELoss(Loss):
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
        return LossResult(loss / len(sample.action_idx))


@loss_from_string.register("imitation_jeffreys_loss")
class ImitationJeffreysLoss(Loss):
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
        return LossResult(loss / len(sample.action_distribution))


@loss_from_string.register("imitation_js_loss")
class ImitationJSLoss(Loss):
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
        return LossResult(loss / len(sample.action_distribution))


@loss_from_string.register("imitation_mse_loss")
class ImitationMSELoss(Loss):
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
        return LossResult(loss / len(sample.action_distribution))


@loss_from_string.register("imitation_kl_loss")
class ImitationKLLoss(Loss):
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
        return LossResult(loss / len(sample.action_distribution))


@loss_from_string.register("imitation_reverse_kl_loss")
class ImitationReverseKLLoss(Loss):
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
        return LossResult(loss / len(sample.action_distribution))


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

    def state_dict(self):
        return {
            "running": self.running,
            "iter": self.iter,
        }

    def load_state_dict(self, state):
        self.running = state["running"]
        self.iter = state["iter"]


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

    def state_dict(self):
        return {
            "mean": self.running_mean.state_dict(),
            "var": self.running_var.state_dict(),
        }

    def load_state_dict(self, state):
        self.running_mean.load_state_dict(state["mean"])
        self.running_var.load_state_dict(state["var"])


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
class PolicyGradientLoss(Loss):
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
        discount_factor: float | None = None,
        normalizer_alpha: float | None = None,
        drift: str | None = None,
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
        drifts: dict[
            str | None,
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
        ] = {
            None: None,
            "importance_sampling": importance_sampling,
            "ppo": (lambda r, A: ppo(r, A, clip_val=imp_ratio_clip, rectification=0)),
            "ppo-rb": (
                lambda r, A: ppo(
                    r, A, clip_val=imp_ratio_clip, rectification=rectification
                )
            ),
            "spo": (lambda r, A: spo(r, A, imp_ratio_clip)),
        }
        self.drift = drifts[drift]
        self.drift_name = drift
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

    def state_dict(self):
        return {
            "normalizer": (
                None if self.normalizer is None else self.normalizer.state_dict()
            )
        }

    def load_state_dict(self, state):
        normalizer_state = state["normalizer"]
        if self.normalizer is None:
            if normalizer_state is not None:
                raise ValueError("checkpoint expects a policy weight normalizer")
            return
        if normalizer_state is None:
            raise ValueError("checkpoint is missing the policy weight normalizer")
        self.normalizer.load_state_dict(normalizer_state)

    def __call__(self, pred_policy, pred_value, sample, training_state):
        assert len(pred_policy) == len(sample.action_idx)
        metrics = {}

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
                metrics["importance_ratio"] = imp_ratio.mean()
                if self.drift_name in {"ppo", "ppo-rb"}:
                    clipped = ((weight > 0) & (imp_ratio > 1 + self.imp_ratio_clip)) | (
                        (weight < 0) & (imp_ratio < 1 - self.imp_ratio_clip)
                    )
                    metrics["clip_fraction"] = clipped.float().mean()
                weight = self.drift(imp_ratio, weight)

        return LossResult(
            self.strength
            * torch.mean(
                weight * F.cross_entropy(padded, sample.action_idx, reduction="none")
            ),
            metrics,
        )


@loss_from_string.register("kl")
class KLPenalty(Loss):
    """Penalize reverse KL from the current policy to the reference policy."""

    needs_reference_policy_value = True
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float = 0.0):
        self.strength = strength

    def distances(self, pred_policy, sample, training_state):
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
        policy_prob = torch.where(
            mask, safe_log_policy.exp(), torch.zeros_like(safe_log_policy)
        )
        reference_prob = torch.where(
            mask, safe_log_reference.exp(), torch.zeros_like(safe_log_reference)
        )
        batch_size = len(sample.reference_policy)
        kl = (policy_prob * (safe_log_policy - safe_log_reference)).sum() / batch_size
        total_variation = 0.5 * (policy_prob - reference_prob).abs().sum() / batch_size
        return kl, total_variation

    def divergence(self, pred_policy, sample, training_state):
        divergence, _ = self.distances(pred_policy, sample, training_state)
        return divergence

    def __call__(self, pred_policy, pred_value, sample, training_state):
        if self.strength is None or self.strength == 0:
            return LossResult(torch.zeros((1,), device=pred_value.mean.device))

        return LossResult(
            self.strength * self.divergence(pred_policy, sample, training_state)
        )


@loss_from_string.register("adaptive_kl")
class AdaptiveKLPenalty(Loss):
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
        self.max_strength = self.init_strength * 100.0
        self.adaptation_rate = adaptation_rate
        self.deadband = deadband

        self.kl = KLPenalty(init_strength)

        # Useful for logging and inspection.
        self.last_target_kl = target
        self.last_kl = None
        self.last_strength = init_strength

    def state_dict(self):
        return {
            "strength": self.kl.strength,
            "last_target_kl": self.last_target_kl,
            "last_kl": self.last_kl,
            "last_strength": self.last_strength,
        }

    def load_state_dict(self, state):
        self.kl.strength = state["strength"]
        self.last_target_kl = state["last_target_kl"]
        self.last_kl = state["last_kl"]
        self.last_strength = state["last_strength"]

    def update_strength(self, measured_kl: float):
        error = measured_kl - self.target
        strength = self.kl.strength

        if error > self.deadband:
            # The policy is outside its KL budget.
            strength += self.adaptation_rate * error
        else:
            # Recover from an earlier increase, but never below the base
            # strength used to initialize the controller.
            relax_rate = 0.1 * self.adaptation_rate
            strength += relax_rate * (self.init_strength - strength)

        strength = max(self.init_strength, min(self.max_strength, strength))
        self.kl.strength = strength
        self.last_strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        divergence, total_variation = self.kl.distances(
            pred_policy, sample, training_state
        )
        measured_kl = divergence.detach().item()
        measured_total_variation = total_variation.detach().item()

        if training_state.get("update_kl_controller", True):
            self.update_strength(measured_kl)

        self.last_kl = measured_kl
        penalty = self.kl.strength * divergence
        return LossResult(
            penalty,
            metrics={
                "kl": measured_kl,
                "total_variation": measured_total_variation,
                "target": self.target,
                "strength": self.kl.strength,
            },
        )


@loss_from_string.register("z_loss")
class ZLoss(Loss):
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float = 1e-6):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        s = self.strength / len(sample.action_idx)
        return LossResult(s * sum(logit.pow(2).sum() for logit in pred_policy))


@loss_from_string.register("entropy_bonus")
class EntropyBonus(Loss):
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
        probs = torch.where(
            mask, safe_log_probs.exp(), torch.zeros_like(safe_log_probs)
        )
        terms = probs * safe_log_probs
        return LossResult(self.strength * terms.sum() / len(sample.action_idx))


@loss_from_string.register("support_floor")
class SupportFloorPenalty(Loss):
    """Keep every legal action above a small probability floor.

    Unlike Shannon entropy, the log-probability hinge keeps a finite recovery
    gradient for actions whose probability has effectively collapsed to zero.
    It becomes exactly inactive once every legal action reaches
    ``floor_mass / number_of_legal_actions``, so it does not continuously pull
    a sufficiently supported policy toward uniformity.
    """

    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, floor_mass: float = 0.01, strength: float = 0.001):
        if not 0.0 < floor_mass < 1.0:
            raise ValueError("support floor mass must be between zero and one")
        if strength < 0.0:
            raise ValueError("support floor strength must be non-negative")
        self.floor_mass = floor_mass
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        padded = pack_cached(pred_policy, training_state, "pred_policy")
        mask = torch.isfinite(padded)
        action_counts = mask.sum(dim=1)
        log_probs = F.log_softmax(padded, dim=1)
        target_log_probs = math.log(self.floor_mass) - action_counts.float().log()
        shortfall = torch.where(
            mask,
            (target_log_probs[:, None] - log_probs).clamp_min(0.0),
            torch.zeros_like(log_probs),
        )
        per_state = shortfall.sum(dim=1) / action_counts
        violation_fraction = (
            ((shortfall > 0.0).sum(dim=1) / action_counts).float().mean()
        )
        return LossResult(
            self.strength * per_state.mean(),
            metrics={"violation_fraction": violation_fraction},
        )


class LinearRegularizer(Loss):
    """Linearly decay the strength of another policy regularizer."""

    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(
        self,
        regularizer_factory: Callable[[float], Loss],
        start: float,
        end: float = 0.0,
        schedule: Scheduler | None = None,
    ):
        if start < 0.0 or end < 0.0:
            raise ValueError("regularizer strengths must be non-negative")
        self.start = start
        self.end = end
        self.schedule = schedule
        self.regularizer = regularizer_factory(start)

    def strength(self, progress: float) -> float:
        if self.schedule is not None:
            progress = self.schedule.to_schedule(progress)
        else:
            progress = max(0.0, min(1.0, progress))
        return self.start * (1.0 - progress) + self.end * progress

    def __call__(self, pred_policy, pred_value, sample, training_state):
        strength = self.strength(training_state["progress"])
        self.regularizer.strength = strength
        result = self.regularizer(pred_policy, pred_value, sample, training_state)
        result.metrics["strength"] = strength
        return result


@loss_from_string.register("linear_entropy_bonus")
class LinearEntropyBonus(LinearRegularizer):
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, start: float, end: float = 0.0, schedule=None):
        super().__init__(EntropyBonus, start, end, schedule)


class LinearReverseEntropyBonus(LinearRegularizer):
    def __init__(self, start: float, end: float = 0.0, schedule=None):
        super().__init__(ReverseEntropyBonus, start, end, schedule)


class LinearSymmetricUniformKLPenalty(LinearRegularizer):
    def __init__(self, start: float, end: float = 0.0, schedule=None):
        super().__init__(SymmetricUniformKLPenalty, start, end, schedule)


class LinearSupportFloorPenalty(LinearRegularizer):
    def __init__(
        self,
        floor_mass: float = 0.01,
        start: float = 0.001,
        end: float = 0.0,
        schedule=None,
    ):
        super().__init__(
            lambda strength: SupportFloorPenalty(floor_mass, strength),
            start,
            end,
            schedule,
        )


@loss_from_string.register("scheduled_perplexity")
class ScheduledPerplexity(Loss):
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
        regularizer_factory: Callable[[float], Loss] = EntropyBonus,
        schedule: Scheduler | None = None,
    ):
        assert start >= 0.0
        assert end >= 0.0
        assert init_strength > 0.0
        assert 0.0 <= baseline_ratio <= 1.0
        assert adaptation_rate > 0.0
        assert 0.0 <= ppl_beta < 1.0
        assert deadband >= 0.0

        self.start = start
        self.end = end
        self.schedule = schedule

        self.init_strength = init_strength
        self.baseline_strength = init_strength * baseline_ratio

        # Hard clamps derived from init/baseline, not exposed as knobs.
        self.min_strength = self.baseline_strength * 0.1
        self.max_strength = self.init_strength * 10.0

        self.adaptation_rate = adaptation_rate
        self.ppl_beta = ppl_beta
        self.deadband = deadband

        self.regularizer = regularizer_factory(init_strength)
        # Compatibility for checkpoints and callers written before the
        # exploration regularizer became configurable.
        self.entropy = self.regularizer

        self.ppl_ema: float | None = None

        # Useful for logging.
        self.last_target_ppl: float | None = None
        self.last_ppl: float | None = None
        self.last_ppl_ema: float | None = None
        self.last_strength = init_strength

    def state_dict(self):
        return {
            "strength": self.regularizer.strength,
            "ppl_ema": self.ppl_ema,
            "last_target_ppl": self.last_target_ppl,
            "last_ppl": self.last_ppl,
            "last_ppl_ema": self.last_ppl_ema,
            "last_strength": self.last_strength,
        }

    def load_state_dict(self, state):
        self.regularizer.strength = state["strength"]
        self.ppl_ema = state["ppl_ema"]
        self.last_target_ppl = state["last_target_ppl"]
        self.last_ppl = state["last_ppl"]
        self.last_ppl_ema = state["last_ppl_ema"]
        self.last_strength = state["last_strength"]

    @staticmethod
    def perplexity(policy, training_state=None):
        """
        policy: iterable of 1D logits tensors, one per state.
                Each tensor should already contain only legal-action logits.

        Returns raw effective action count ``exp(entropy)``:

            1 = deterministic
            n = uniform over ``n`` legal actions
        """

        state = {} if training_state is None else training_state
        padded = pack_cached(policy, state, "pred_policy").float()
        mask = torch.isfinite(padded)
        action_counts = mask.sum(dim=1)
        logp = F.log_softmax(padded, dim=1)
        safe_logp = torch.where(mask, logp, torch.zeros_like(logp))
        probabilities = torch.where(mask, safe_logp.exp(), torch.zeros_like(logp))
        entropy = -(probabilities * safe_logp).sum(dim=1)
        normalized = entropy.exp()
        values = normalized[action_counts > 1]

        if values.numel() == 0:
            # Degenerate batch: no state with >1 legal action.
            # Return a tensor on a reasonable device.
            first = next(iter(policy))
            return first.new_tensor(0.0)

        return values.mean()

    @staticmethod
    def normalized_perplexity(policy, training_state=None):
        """
        policy: iterable of 1D logits tensors, one per state.
                Each tensor should already contain only legal-action logits.

        Returns normalized perplexity in [0, 1]:

            0 = deterministic
            1 = uniform over legal actions
        """

        state = {} if training_state is None else training_state
        padded = pack_cached(policy, state, "pred_policy").float()
        mask = torch.isfinite(padded)
        action_counts = mask.sum(dim=1)
        logp = F.log_softmax(padded, dim=1)
        safe_logp = torch.where(mask, logp, torch.zeros_like(logp))
        probabilities = torch.where(mask, safe_logp.exp(), torch.zeros_like(logp))
        entropy = -(probabilities * safe_logp).sum(dim=1)
        normalized = (entropy.exp() - 1.0) / (action_counts - 1).clamp_min(1)
        values = normalized[action_counts > 1]

        if values.numel() == 0:
            # Degenerate batch: no state with >1 legal action.
            # Return a tensor on a reasonable device.
            first = next(iter(policy))
            return first.new_tensor(0.0)

        return values.mean()

    def target_ppl(self, progress: float) -> float:
        if self.schedule is not None:
            progress = self.schedule.to_schedule(progress)
        else:
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
                self.ppl_beta * self.ppl_ema + (1.0 - self.ppl_beta) * measured_ppl
            )

        strength = self.regularizer.strength

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
            relax_rate = 0.5 * self.adaptation_rate
            strength += relax_rate * (self.baseline_strength - strength)

        strength = max(self.min_strength, min(self.max_strength, strength))

        self.regularizer.strength = strength

        self.last_strength = strength
        self.last_ppl_ema = self.ppl_ema

    def __call__(self, pred_policy, pred_value, sample, training_state):
        progress = training_state["progress"]
        target = self.target_ppl(progress)

        ppl_tensor = self.normalized_perplexity(pred_policy, training_state)
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

        e = self.regularizer(pred_policy, pred_value, sample, training_state)
        ppl_ema = self.ppl_ema if self.ppl_ema is not None else ppl
        return LossResult(
            e.objective,
            metrics={
                "perplexity": ppl,
                "target": target,
                "ema": ppl_ema,
                "strength": self.regularizer.strength,
            },
        )


@loss_from_string.register("reverse_entropy_bonus")
class ReverseEntropyBonus(Loss):
    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        padded = pack_cached(pred_policy, training_state, "pred_policy")
        mask = torch.isfinite(padded)
        action_counts = mask.sum(dim=1)
        log_probs = F.log_softmax(padded, dim=1)
        safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))

        # KL(U || policy) = -mean_a(log p(a)) - log(number of legal actions).
        # Averaging within each state avoids making the effective strength
        # proportional to its branching factor. The constant makes the loss
        # zero at uniform and does not affect its gradient.
        reverse_kl = -safe_log_probs.sum(dim=1) / action_counts
        reverse_kl = reverse_kl - action_counts.log()
        return LossResult(self.strength * reverse_kl.mean())


@loss_from_string.register("symmetric_uniform_kl")
class SymmetricUniformKLPenalty(Loss):
    """Jeffreys divergence between the policy and the legal-action uniform prior.

    This averages both KL directions. The policy-to-uniform direction is the
    usual entropy regularizer up to a per-state constant, while the
    uniform-to-policy direction retains a recovery gradient for actions whose
    probability has become very small.
    """

    needs_reference_policy_value = False
    supports_off_policy = True
    supports_partial_trajectories = True

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        padded = pack_cached(pred_policy, training_state, "pred_policy")
        mask = torch.isfinite(padded)
        action_counts = mask.sum(dim=1)
        log_probs = F.log_softmax(padded, dim=1)
        safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
        probs = torch.where(
            mask, safe_log_probs.exp(), torch.zeros_like(safe_log_probs)
        )
        log_action_counts = action_counts.float().log()

        forward_kl = (probs * safe_log_probs).sum(dim=1) + log_action_counts
        reverse_kl = -safe_log_probs.sum(dim=1) / action_counts - log_action_counts
        symmetric_kl = 0.5 * (forward_kl + reverse_kl)
        return LossResult(self.strength * symmetric_kl.mean())


@loss_from_string.register("value_mse_loss")
class ValueMSELoss(Loss):
    supports_off_policy = True
    supports_partial_trajectories = False
    needs_reference_policy_value = False

    def __init__(self, strength: float = 1):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        assert pred_value.mean.shape == sample.returns.shape
        return LossResult(self.strength * F.mse_loss(pred_value.mean, sample.returns))


@loss_from_string.register("value_log_prob")
class ValueLogProb(Loss):
    supports_off_policy = True
    supports_partial_trajectories = False
    needs_reference_policy_value = False

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        lp = pred_value.log_prob(sample.returns)
        return LossResult(-self.strength * lp.mean())


@loss_from_string.register("bootstrap_mse_loss")
class BootstrapMSELoss(Loss):
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
        return LossResult(-self.strength * pred_value.log_prob(target).mean())


@loss_from_string.register("bootstrap_value_mse_loss")
class BootstrapValueMSELoss(Loss):
    supports_off_policy = True
    supports_partial_trajectories = True
    needs_reference_policy_value = True

    def __init__(self, strength: float = 1):
        self.strength = strength

    def __call__(self, pred_policy, pred_value, sample, training_state):
        return LossResult(self.strength * F.mse_loss(pred_value.mean, sample.td_lambda))


@loss_from_string.register("bootstrap_value_log_prob_loss")
class BootstrapValueLogProbLoss(Loss):
    """Fit TD(lambda) under the value distribution with a rollout trust region."""

    supports_off_policy = True
    supports_partial_trajectories = True
    needs_reference_policy_value = True

    def __init__(self, strength: float = 1.0, epsilon: float | None = 2.0):
        if epsilon is not None and epsilon <= 0:
            raise ValueError("epsilon must be positive or None")
        self.strength = strength
        self.epsilon = epsilon

    def __call__(self, pred_policy, pred_value, sample, training_state):
        target = sample.td_lambda
        if self.epsilon is None:
            clipped_target = target
            clip_ratio = target.new_zeros(())
        else:
            radius = self.epsilon * sample.reference_value_stddev
            lower = sample.reference_value - radius
            upper = sample.reference_value + radius
            clipped_target = torch.clamp(target, min=lower, max=upper)
            clip_ratio = ((target < lower) | (target > upper)).float().mean()

        return LossResult(
            -self.strength * pred_value.log_prob(clipped_target).mean(),
            {"clip_ratio": clip_ratio},
        )


@loss_from_string.register("q_mse_loss")
class QMSELoss(Loss):
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

        return LossResult(loss / len(sample.action_idx))
