import torch

from boardrl.models import make
from boardrl.rl.model.model import Model
from boardrl.rl.model.transformer import Transformer


def test_transformer_canon_layers_are_identity_initialized():
    model = Transformer(hidden_size=8, num_layers=2, num_heads=2, head_size=4)
    canons = [
        model.canon,
        *(
            canon
            for block in model.transformer_blocks
            for canon in (block.canon_a, block.canon_c)
        ),
    ]
    inputs = torch.randn(2, 7, 8)

    for canon in canons:
        expected_weight = torch.zeros_like(canon.weight)
        expected_weight[:, :, -1] = 1
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

    assert model.to_pred.out.weight.grad is not None
    assert model.rewards.out.weight.grad is not None
    assert model.rewards.attention.in_proj_weight.grad is not None


def test_shared_patch_backbone_trains_local_and_global_paths():
    torch.manual_seed(0)
    model = make(
        "shared-patch",
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
    assert model.backbone.encode.local.blocks[0].conv.weight.grad is not None
    assert model.backbone.encode.downsample.weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in model.backbone.encode.global_context.parameters()
    )


def test_shared_patch_scales_increase_capacity_monotonically():
    names = [
        "shared-patch-tiny",
        "shared-patch-small",
        "shared-patch-medium",
        "shared-patch-large",
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


def test_shared_patch_compression_width_is_checkpointed():
    model = make(
        "shared-patch-small",
        backbone_kwargs={"patch_size": 8},
    )

    assert model.spec()["backbone_kwargs"] == {"patch_size": 8}
    assert model.backbone.encode.patch_size == 8
    assert model.backbone.encode.downsample.kernel_size == (8,)
