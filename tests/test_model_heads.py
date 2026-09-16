import pytest
import torch

from boardrl.games import games_library
from boardrl.models import architectures, make, make_for_game
from boardrl.rl.model.cnn import PatchTransformerCNNEncoder
from boardrl.rl.model.model import Model, OutcomeValueDistribution
from boardrl.rl.model.transformer import Transformer


def test_transformer_canon_layers_are_identity_initialized():
    model = Transformer(hidden_size=8, num_layers=2, num_heads=2, head_size=4)
    canons = [
        *(
            canon
            for block in model.transformer_blocks
            for canon in (block.canon_a, block.canon_c)
        ),
    ]
    inputs = torch.randn(2, 7, 8)

    for canon in canons:
        expected_weight = torch.zeros_like(canon.weight)
        expected_weight[:, :, canon.kernel_size // 2] = 1
        assert torch.equal(canon.weight, expected_weight)
        assert torch.equal(canon(inputs), inputs)


def test_model_heads_produce_gradients():
    torch.manual_seed(0)
    model = Model(dim=16, num_layers=1, head_size=4, num_heads=4)
    output = model(["abc@0def@1", "xy@0"])

    assert [logits.shape for logits in output.policy] == [
        torch.Size([2]),
        torch.Size([1]),
    ]
    assert output.value.mean.shape == torch.Size([2])

    loss = sum(logits.square().sum() for logits in output.policy)
    loss = loss + output.value.mean.square().sum()
    loss.backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_outcome_value_head_preserves_distribution_when_unbatched():
    torch.manual_seed(0)
    model = Model(
        dim=16,
        num_layers=1,
        head_size=4,
        num_heads=4,
        points_based=False,
    )

    output = model(["abc@0def@1", "xy@0"])
    values = [prediction.value for prediction in output.unbatched()]

    assert isinstance(output.value, torch.distributions.Categorical)
    assert output.value.mean.shape == torch.Size([2])
    assert output.value.stddev.shape == torch.Size([2])
    assert len(values) == 2
    assert all(isinstance(value, torch.distributions.Categorical) for value in values)
    assert all(value.mean.shape == torch.Size([1]) for value in values)


def test_outcome_value_distribution_has_expected_moments_and_log_prob():
    distribution = Model(
        dim=16,
        num_layers=1,
        head_size=4,
        num_heads=4,
        points_based=False,
    )(["abc@0"])
    value = distribution.value

    assert torch.allclose(value.probs, torch.full((1, 3), 1 / 3))
    assert torch.allclose(value.mean, torch.tensor([0.0]))
    assert torch.allclose(value.variance, torch.tensor([2 / 3]))
    assert torch.allclose(
        value.log_prob(torch.tensor([-1.0])),
        torch.log(torch.tensor([1 / 3])),
    )


def test_outcome_value_distribution_projects_fractional_targets():
    distribution = OutcomeValueDistribution(
        logits=torch.log(torch.tensor([[0.2, 0.3, 0.5]]))
    )

    assert torch.allclose(
        distribution.log_prob(torch.tensor([0.25])),
        0.75 * torch.log(torch.tensor([0.3]))
        + 0.25 * torch.log(torch.tensor([0.5])),
    )


def test_game_descriptor_selects_value_head_family():
    outcome_model = make_for_game("toy", games_library("tictactoe"))
    points_model = make_for_game("toy", games_library("thegame"))

    assert outcome_model.rewards.out[2].out_features == 3
    assert points_model.rewards.out[2].out_features == 2


def test_heads_handle_variable_action_counts():
    torch.manual_seed(0)
    model = Model(
        dim=16,
        num_layers=1,
        head_size=4,
        num_heads=4,
    )

    output = model(["state@left@right", "state@only", "state"])

    assert [logits.shape for logits in output.policy] == [
        torch.Size([2]),
        torch.Size([1]),
        torch.Size([0]),
    ]

    loss = sum(logits.square().sum() for logits in output.policy)
    loss = loss + output.value.mean.square().sum()
    loss.backward()

    assert model.to_pred.out[2].weight.grad is not None
    assert model.rewards.out[2].weight.grad is not None
    assert model.rewards.attention.kv.weight.grad is not None
    assert model.rewards.attention.qg.weight.grad is not None


def test_reinit_heads_preserves_backbone_and_resets_both_heads():
    torch.manual_seed(0)
    model = Model(dim=16, num_layers=1, head_size=4, num_heads=4)
    backbone = {
        name: parameter.detach().clone()
        for name, parameter in model.backbone.named_parameters()
    }
    initial_raw_value_scale = model.rewards.raw_value_scale.detach().clone()
    with torch.no_grad():
        for parameter in model.to_pred.parameters():
            parameter.fill_(7)
        for parameter in model.rewards.parameters():
            parameter.fill_(7)

    model.reinit_heads()

    assert all(
        torch.equal(parameter, backbone[name])
        for name, parameter in model.backbone.named_parameters()
    )
    assert torch.equal(
        model.to_pred.norm.weight,
        torch.ones_like(model.to_pred.norm.weight),
    )
    assert torch.count_nonzero(model.to_pred.out[2].weight) > 0
    assert torch.count_nonzero(model.to_pred.out[2].bias) == 0
    assert torch.count_nonzero(model.rewards.query) > 0
    assert torch.equal(
        model.rewards.norm.weight,
        torch.ones_like(model.rewards.norm.weight),
    )
    assert torch.count_nonzero(model.rewards.out[0].weight) > 0
    assert torch.count_nonzero(model.rewards.out[0].bias) == 0
    assert torch.count_nonzero(model.rewards.out[2].weight) == 0
    assert torch.count_nonzero(model.rewards.out[2].bias) == 0
    assert torch.equal(model.rewards.raw_value_scale, initial_raw_value_scale)
    for head in (model.to_pred, model.rewards):
        assert torch.count_nonzero(head.attention.kv.weight != 7) > 0
        assert torch.count_nonzero(head.attention.qg.weight != 7) > 0
        assert torch.count_nonzero(head.attention.fc.weight != 7) > 0


def test_heads_resolve_missing_head_counts_for_all_presets():
    expected_heads = {
        "toy": 8,
        "cnn": 4,
        "minimal-lstm": 4,
        "minimal-gated-cnn": 4,
    }

    for name, num_heads in expected_heads.items():
        model = make(name)
        assert model.to_pred.attention.num_heads == num_heads
        assert model.rewards.attention.num_heads == num_heads


def test_patchformer_backbone_trains_local_and_global_paths():
    torch.manual_seed(0)
    model = make(
        "patchformer-small-p4",
        dim=16,
        num_heads=4,
        head_size=4,
    )

    output = model(["short@a@b", "a considerably longer state@only"])
    loss = sum(logits.square().sum() for logits in output.policy)
    loss = loss + output.value.mean.square().sum()
    loss.backward()

    assert [logits.shape for logits in output.policy] == [
        torch.Size([2]),
        torch.Size([1]),
    ]
    assert model.backbone.encode.local.blocks[0].conv1.weight.grad is not None
    assert model.backbone.encode.downsample.weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in model.backbone.encode.global_context.parameters()
    )


@pytest.mark.parametrize(
    "length,expected_patches",
    [
        (9, [1.0, 2.0]),
        (16, [1.0, 2.0]),
        (17, [1.0, 0.0, 2.0]),
    ],
)
def test_patchformer_patchification_keeps_prefix_and_suffix(
    length, expected_patches
):
    encoder = PatchTransformerCNNEncoder(
        dim=2,
        global_layers=1,
        num_heads=1,
        head_size=2,
        patch_size=8,
    )
    with torch.no_grad():
        encoder.downsample.weight.zero_()
        encoder.downsample.bias.zero_()
        encoder.downsample.weight[0, 0].fill_(1.0)

    local = torch.zeros(1, 2, length)
    local[0, 0, 0] = 1.0
    local[0, 0, -1] = 2.0
    mask = torch.ones(1, length, dtype=torch.bool)

    patches, patch_mask = encoder._patchify(local, mask)

    assert patches.shape == (1, 2, len(expected_patches))
    assert patches[0, 0].tolist() == expected_patches
    assert patch_mask.tolist() == [[True] * len(expected_patches)]


def test_patchformer_scales_increase_capacity_monotonically():
    names = [
        "patchformer-tiny-p4",
        "patchformer-small-p4",
        "patchformer-medium-p4",
        "patchformer-large-p4",
    ]
    models = [make(name) for name in names]
    parameter_counts = [
        sum(parameter.numel() for parameter in model.parameters())
        for model in models
    ]

    assert parameter_counts == sorted(parameter_counts)
    assert len(set(parameter_counts)) == len(parameter_counts)
    assert [len(model.backbone.encode.local.blocks) for model in models] == [
        2,
        2,
        2,
        2,
    ]
    assert [
        len(model.backbone.encode.global_context.transformer_blocks)
        for model in models
    ] == [2, 4, 4, 6]
    assert [model.spec()["head_size"] for model in models] == [8, 16, 32, 32]
    assert [model.spec()["num_heads"] for model in models] == [4, 4, 4, 8]


def test_model_architectures_cover_scale_and_patch_size_product():
    expected = {
        f"patchformer-{scale}-p{patch_size}"
        for scale in ("tiny", "small", "medium", "large")
        for patch_size in (4, 8, 16)
    }

    assert expected <= architectures.keys()


def test_patchformer_compression_width_is_checkpointed():
    model = make("patchformer-small-p8")

    assert model.spec()["backbone_kwargs"] == {
        "patch_size": 8,
        "canon_kernel_size": 7,
    }
    assert model.backbone.encode.patch_size == 8
    assert model.backbone.encode.downsample.kernel_size == (8,)
