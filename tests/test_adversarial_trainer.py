from types import SimpleNamespace

import torch
import torch.nn.functional as F

from boardrl.training import TrainingSample
from trainers.adversarial import NFSPAveragePolicyLoss, Reservoir, build_parser


def test_nfsp_average_policy_loss_imitates_sampled_actions():
    policy = [
        torch.tensor([0.0, 2.0], requires_grad=True),
        torch.tensor([1.5, -0.5], requires_grad=True),
    ]
    sample = SimpleNamespace(action_idx=torch.tensor([1, 0]))

    result = NFSPAveragePolicyLoss()(policy, None, sample, {})
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


def test_adversarial_trainer_defaults_to_nfsp_connect_four():
    args = build_parser().parse_args([])

    assert args.game == "connectfour"
    assert args.tag == "nfsp"
    assert args.anticipatory == 0.1
