"""Connect Four current-vs-reference experiment, formerly YAML configuration."""

import copy
import torch

from boardrl import Inference, RolloutRunner
from boardrl.games import games_library
from boardrl.models import cnn_large, copy_weights
from boardrl.rl.model.loss import AdaptiveKLPenalty, BootstrapValueMSELoss, EntropyBonus, PolicyGradientLoss
from boardrl.training import ComputeReturns, Learner, Pipeline, ReferenceTargets, Select, ToSamples


game = games_library("connectfour")
device = "cuda" if torch.cuda.is_available() else "cpu"
current = cnn_large().to(device)
reference = copy.deepcopy(current).eval()
current_player = Inference(current, batch_size=64)
reference_player = Inference(reference, batch_size=64)
runner = RolloutRunner(game.make_game)
optimizer = torch.optim.AdamW(
    current.parameters(), lr=1e-5, betas=(0.5, 0.999), weight_decay=0.01
)
learner = Learner(
    current,
    optimizer,
    [
        PolicyGradientLoss(weight="normalized_gae", drift="ppo", imp_ratio_clip=0.2),
        EntropyBonus(0.01),
        AdaptiveKLPenalty(
            target=0.005, init_strength=1.0, deadband=0.001, adaptation_rate=0.01
        ),
        BootstrapValueMSELoss(),
    ],
    batch_size=64,
    device=device,
    gradient_clip=5.0,
    augmentations=game.augmentations,
)
prepare = Pipeline(
    ComputeReturns(1.0),
    Select(strategies=[0]),
    ToSamples(),
    ReferenceTargets(
        reference,
        batch_size=64,
        discount=1.0,
        gae_lambda=1.0,
        value_lambda=1.0,
    ),
)

for step in range(80_000):
    with current_player.evaluating(), reference_player.evaluating():
        games = runner.play(
            [current_player.policy(), reference_player.policy()],
            games=64,
            max_steps=500,
        )
    copy_weights(reference, current)
    learner.train(prepare(games), progress=step / 80_000)
