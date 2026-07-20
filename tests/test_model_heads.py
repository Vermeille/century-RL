import torch

from boardrl.rl.model.model import Model, VariancePreservingAttentionPool


def test_attention_pool_starts_as_normalized_sum():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 8)
    mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
        ]
    )

    pool = VariancePreservingAttentionPool(x.size(-1))
    pooled = pool(x, mask)
    expected = torch.stack(
        [
            x[0, :4].sum(dim=0) / 4**0.5,
            x[1, :2].sum(dim=0) / 2**0.5,
        ]
    )

    assert torch.allclose(pooled, expected)


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
