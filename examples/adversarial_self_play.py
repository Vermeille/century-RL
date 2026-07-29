"""Minimal self-vs-self training loop for an adversarial game."""

import copy

import torch

from boardrl import Inference, RolloutRunner
from boardrl.games import games_library
from boardrl.models import cnn
from boardrl.rl.model.loss import BootstrapValueMSELoss, EntropyBonus, PolicyGradientLoss
from boardrl.training import ComputeReturns, Learner, Pipeline, ReferenceTargets, ToSamples


game = games_library("tictactoe")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = cnn().to(device)
reference = copy.deepcopy(model)
inference = Inference(model, batch_size=64)
rollouts = RolloutRunner(game.make_game)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
learner = Learner(
    model,
    optimizer,
    [
        PolicyGradientLoss(weight="normalized_gae", drift="ppo", imp_ratio_clip=0.2),
        EntropyBonus(0.01),
        BootstrapValueMSELoss(),
    ],
    batch_size=64,
    device=device,
    augmentations=game.augmentations,
)
prepare = Pipeline(
    ComputeReturns(1.0),
    ToSamples(),
    ReferenceTargets(
        reference,
        batch_size=64,
        discount=1.0,
        gae_lambda=1.0,
        value_lambda=1.0,
    ),
)

for step in range(8_000):
    reference.load_state_dict(model.state_dict())
    with inference.evaluating():
        player = inference.policy()
        games = rollouts.play([player, player], games=32, max_steps=20)
    learner.train(prepare(games), progress=step / 8_000)
