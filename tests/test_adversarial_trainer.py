from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from boardrl.rl.model.loss import CELoss, ImitationCELoss
from boardrl.games import games_library
from boardrl.rollouts import RolloutRunner
from boardrl.training import TrainingSample
from trainers import coop
from trainers.adversarial import Reservoir, build_parser


class CountingGame:
    def __init__(self, *, num_players):
        self.num_players = num_players
        self.moves = ["move-0", "move-1"]
        self.played = []

    def ended(self):
        return len(self.played) == 2

    def play_idx(self, index):
        self.played.append(index)


def test_ce_loss_imitates_sampled_actions_without_action_distribution():
    policy = [
        torch.tensor([0.0, 2.0], requires_grad=True),
        torch.tensor([1.5, -0.5], requires_grad=True),
    ]
    sample = SimpleNamespace(action_idx=torch.tensor([1, 0]))

    result = CELoss()(policy, None, sample, {})
    expected = (
        F.cross_entropy(policy[0].unsqueeze(0), torch.tensor([1]))
        + F.cross_entropy(policy[1].unsqueeze(0), torch.tensor([0]))
    ) / 2

    assert torch.allclose(result.objective, expected)
    result.objective.backward()
    assert all(logits.grad is not None for logits in policy)


def test_imitation_ce_loss_packs_variable_action_distributions():
    policy = [
        torch.tensor([0.0, 2.0], requires_grad=True),
        torch.tensor([1.5, -0.5, 0.25], requires_grad=True),
    ]
    targets = [
        torch.tensor([2.0, -1.0]),
        torch.tensor([-0.5, 0.0, 1.0]),
    ]
    sample = SimpleNamespace(action_distribution=targets)

    result = ImitationCELoss()(policy, None, sample, {})
    expected = torch.stack(
        [
            F.cross_entropy(prediction[None], target[None].softmax(1))
            for prediction, target in zip(policy, targets)
        ]
    ).mean()

    assert torch.allclose(result.objective, expected)
    result.objective.backward()
    assert all(logits.grad is not None for logits in policy)


def test_reservoir_round_trip_preserves_sampling_state():
    reservoir = Reservoir(3, seed=7)
    reservoir.add(
        [
            TrainingSample(
                state=f"state-{index}",
                action_distribution=torch.tensor([float(index), -float(index)]),
            )
            for index in range(20)
        ]
    )

    restored = Reservoir(3, seed=999)
    restored.load_state_dict(reservoir.state_dict())

    assert restored.seen == 20
    assert [state for state, _ in restored.samples] == [
        state for state, _ in reservoir.samples
    ]
    assert all(
        torch.equal(restored_distribution, distribution)
        for (_, restored_distribution), (_, distribution) in zip(
            restored.samples, reservoir.samples
        )
    )
    assert [
        (sample.state, sample.action_distribution.tolist())
        for sample in restored.sample(2)
    ] == [
        (sample.state, sample.action_distribution.tolist())
        for sample in reservoir.sample(2)
    ]


def test_reservoir_stores_policy_distribution_not_sampled_action():
    action_distribution = torch.tensor([2.0, -1.0, 0.5])
    reservoir = Reservoir(1, seed=7)

    reservoir.add(
        [
            TrainingSample(
                state="state",
                action_idx=1,
                action_distribution=action_distribution,
            )
        ]
    )
    action_distribution[0] = 100.0
    sample = reservoir.sample(1)[0]

    assert not hasattr(sample, "action_idx")
    assert torch.equal(sample.action_distribution, torch.tensor([2.0, -1.0, 0.5]))


def test_reservoir_add_random_uses_exact_quota_without_replacement():
    samples = [
        TrainingSample(
            state=f"state-{index}",
            action_distribution=torch.tensor([float(index), 0.0]),
        )
        for index in range(10)
    ]
    reservoir = Reservoir(20, seed=7)

    inserted = reservoir.add_random(samples, 4)

    assert inserted == 4
    assert reservoir.seen == 4
    assert len(reservoir) == 4
    assert len({state for state, _ in reservoir.samples}) == 4


def test_reservoir_add_random_oversamples_to_exact_quota():
    samples = [
        TrainingSample(
            state=f"state-{index}",
            action_distribution=torch.tensor([float(index), 0.0]),
        )
        for index in range(2)
    ]
    reservoir = Reservoir(20, seed=7)

    inserted = reservoir.add_random(samples, 8)

    assert inserted == 8
    assert reservoir.seen == 8
    assert len(reservoir) == 8
    assert {state for state, _ in reservoir.samples} == {"state-0", "state-1"}


def test_adversarial_trainer_defaults_to_nfsp_connect_four():
    args = build_parser().parse_args([])

    assert args.game == "connectfour"
    assert args.tag == "nfsp"
    assert args.anticipatory == 0.1
    assert args.fixed_training_opponent_fraction == 0.0
    assert args.reservoir_insert_ratio == 4.0
    assert args.adam_beta1 == 0.9
    assert args.adam_beta2 == 0.95
    assert args.adam_eps == 1e-5
    assert args.random_game_depth == 0


def test_random_opening_factory_plays_uniformly_sampled_legal_prefix(monkeypatch):
    depths = iter([0, 1, 3])
    monkeypatch.setattr(coop.random, "randint", lambda start, end: next(depths))
    monkeypatch.setattr(coop.random, "randrange", lambda count: count - 1)
    factory = coop.RandomOpeningGameFactory(CountingGame, max_depth=3)

    games = [factory(num_players=2) for _ in range(3)]

    assert [game.num_players for game in games] == [2, 2, 2]
    assert [game.played for game in games] == [[], [1], [1, 1]]


def test_disabled_random_opening_does_not_consume_rng(monkeypatch):
    monkeypatch.setattr(
        coop.random,
        "randint",
        lambda start, end: pytest.fail("disabled wrapper consumed RNG"),
    )

    game = coop.RandomOpeningGameFactory(CountingGame, max_depth=0)(num_players=2)

    assert game.played == []
