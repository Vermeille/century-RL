from types import SimpleNamespace

import torch
import torch.nn.functional as F

from boardrl.rl.model.loss import CELoss
from boardrl.games import games_library
from boardrl.rollouts import RolloutRunner
from boardrl.training import TrainingSample
from trainers.adversarial import Reservoir, build_parser


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


def test_reservoir_round_trip_preserves_sampling_state():
    reservoir = Reservoir(3, seed=7)
    reservoir.add(
        [
            TrainingSample(state=f"state-{index}", action_idx=index % 2)
            for index in range(20)
        ]
    )

    restored = Reservoir(3, seed=999)
    restored.load_state_dict(reservoir.state_dict())

    assert restored.seen == 20
    assert restored.samples == reservoir.samples
    assert [
        (sample.state, sample.action_idx) for sample in restored.sample(2)
    ] == [
        (sample.state, sample.action_idx) for sample in reservoir.sample(2)
    ]


def test_reservoir_add_random_uses_exact_quota_without_replacement():
    samples = [
        TrainingSample(state=f"state-{index}", action_idx=index)
        for index in range(10)
    ]
    reservoir = Reservoir(20, seed=7)

    inserted = reservoir.add_random(samples, 4)

    assert inserted == 4
    assert reservoir.seen == 4
    assert len(reservoir) == 4
    assert len(set(reservoir.samples)) == 4


def test_reservoir_add_random_oversamples_to_exact_quota():
    samples = [
        TrainingSample(state=f"state-{index}", action_idx=index)
        for index in range(2)
    ]
    reservoir = Reservoir(20, seed=7)

    inserted = reservoir.add_random(samples, 8)

    assert inserted == 8
    assert reservoir.seen == 8
    assert len(reservoir) == 8
    assert set(reservoir.samples) == {("state-0", 0), ("state-1", 1)}


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
