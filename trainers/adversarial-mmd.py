"""Shared-policy self-play with PPO-style Magnetic Mirror Descent regularization."""

from __future__ import annotations

import math

import torch

if __package__:
    from . import adversarial_selfplay, coop
else:
    import adversarial_selfplay
    import coop

from boardrl.rl.model.loss import (
    BootstrapValueLogProbLoss,
    EntropyBonus,
    KLPenalty,
    PolicyGradientLoss,
)
from boardrl.training import Learner, PolicyMetrics, TrainResult, ValueMetrics

TRAINER_NAME = "adversarial-mmd"
ALGORITHM = "shared-policy-magnetic-mirror-descent"


def build_parser():
    parser = adversarial_selfplay.build_parser(description=__doc__, tag=TRAINER_NAME)
    parser.add_argument(
        "--mmd-coefficient",
        type=coop.positive_float,
        default=0.05,
        help="base coefficient c in alpha_t = eta_t = c * sqrt(T / t)",
    )
    parser.add_argument(
        "--mmd-reference-timesteps",
        type=coop.positive_int,
        default=10_000_000,
        help="reference horizon T in the MMD alpha/eta schedule",
    )
    return parser


def mmd_parameters(timesteps, *, coefficient=0.05, reference_timesteps=10_000_000):
    """Return (alpha, eta, reverse-KL strength) from the deep MMD schedule."""

    t = max(int(timesteps), 1)
    scale = math.sqrt(reference_timesteps / t)
    alpha = coefficient * scale
    eta = coefficient * scale
    return alpha, eta, 1.0 / eta


class MMDLearner(Learner):
    """Learner with the paper's timestep-based entropy/reverse-KL schedule."""

    def __init__(
        self,
        *args,
        entropy_loss,
        kl_loss,
        mmd_coefficient,
        mmd_reference_timesteps,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.entropy_loss = entropy_loss
        self.kl_loss = kl_loss
        self.mmd_coefficient = mmd_coefficient
        self.mmd_reference_timesteps = mmd_reference_timesteps
        self.mmd_timesteps = 0
        self.last_alpha = None
        self.last_eta = None
        self.last_kl_strength = None

    def state_dict(self):
        state = super().state_dict()
        state["mmd_timesteps"] = self.mmd_timesteps
        return state

    def load_state_dict(self, state):
        state = dict(state)
        self.mmd_timesteps = int(state.pop("mmd_timesteps", 0))
        super().load_state_dict(state)

    def train(self, samples, *, progress=0.0):
        self.mmd_timesteps += len(samples)
        alpha, eta, kl_strength = mmd_parameters(
            self.mmd_timesteps,
            coefficient=self.mmd_coefficient,
            reference_timesteps=self.mmd_reference_timesteps,
        )
        self.entropy_loss.strength = alpha
        self.kl_loss.strength = kl_strength
        self.last_alpha = alpha
        self.last_eta = eta
        self.last_kl_strength = kl_strength

        result = super().train(samples, progress=progress)
        metrics = dict(result.metrics)
        metrics.update(
            {
                "mmd_alpha": alpha,
                "mmd_eta": eta,
                "mmd_reverse_kl_strength": kl_strength,
                "mmd_timesteps": float(self.mmd_timesteps),
            }
        )
        return TrainResult(metrics)


def make_learner(model, game, args):
    optimizer, lr_schedule = adversarial_selfplay.make_optimizer(model, args)
    alpha, _, kl_strength = mmd_parameters(
        1,
        coefficient=args.mmd_coefficient,
        reference_timesteps=args.mmd_reference_timesteps,
    )
    entropy_loss = EntropyBonus(alpha)
    kl_loss = KLPenalty(kl_strength)
    learner = MMDLearner(
        model,
        optimizer,
        [
            PolicyGradientLoss(
                weight="normalized_gae",
                drift="ppo",
                imp_ratio_clip=args.ppo_clip,
            ),
            entropy_loss,
            kl_loss,
            BootstrapValueLogProbLoss(
                strength=args.value_strength,
                epsilon=args.value_clip_epsilon,
            ),
        ],
        entropy_loss=entropy_loss,
        kl_loss=kl_loss,
        mmd_coefficient=args.mmd_coefficient,
        mmd_reference_timesteps=args.mmd_reference_timesteps,
        batch_size=args.learner_batch_size,
        device=args.device,
        epochs=args.epochs,
        gradient_clip=args.gradient_clip,
        augmentations=game.augmentations,
        batch_metrics=[
            PolicyMetrics(nucleus_threshold=coop.NUCLEUS_THRESHOLD),
            ValueMetrics(),
        ],
        normalize_lr=False,
        lr_schedule=lr_schedule,
    )
    return learner, optimizer


def checkpoint_directory(args):
    return adversarial_selfplay.checkpoint_directory(args, TRAINER_NAME)


def run(args):
    return adversarial_selfplay.run(
        args,
        trainer_name=TRAINER_NAME,
        algorithm=ALGORITHM,
        make_learner=make_learner,
        script_path=__file__,
    )


def main():
    torch.set_float32_matmul_precision("medium")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
