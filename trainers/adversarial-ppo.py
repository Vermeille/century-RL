"""Raw shared-policy self-vs-self PPO baseline."""

from __future__ import annotations

import torch

if __package__:
    from . import adversarial_selfplay, coop
else:
    import adversarial_selfplay
    import coop

from boardrl.rl.model.loss import BootstrapValueLogProbLoss, PolicyGradientLoss
from boardrl.training import Learner, PolicyMetrics, ValueMetrics

TRAINER_NAME = "adversarial-ppo"
ALGORITHM = "shared-policy-self-play-ppo"


def build_parser():
    return adversarial_selfplay.build_parser(description=__doc__, tag=TRAINER_NAME)


def make_learner(model, game, args):
    optimizer, lr_schedule = adversarial_selfplay.make_optimizer(model, args)
    learner = Learner(
        model,
        optimizer,
        [
            PolicyGradientLoss(
                weight="normalized_gae",
                drift="ppo",
                imp_ratio_clip=args.ppo_clip,
            ),
            BootstrapValueLogProbLoss(
                strength=args.value_strength,
                epsilon=args.value_clip_epsilon,
            ),
        ],
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
