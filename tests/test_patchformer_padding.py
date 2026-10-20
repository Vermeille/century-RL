import pytest
import torch

from boardrl.rl.model.cnn import PatchTransformerCNNEncoder
from boardrl.rl.model.model import Model


def test_patchification_ignores_masked_tokens_and_their_gradients():
    encoder = PatchTransformerCNNEncoder(
        dim=2, global_layers=1, num_heads=1, head_size=2, patch_size=8
    )
    local = torch.randn(1, 2, 10)
    mask = torch.ones(1, 10, dtype=torch.bool)
    expected, _ = encoder._patchify(local, mask)

    padded = torch.cat((local, torch.full((1, 2, 14), 99.0)), dim=-1)
    padded.requires_grad_()
    padded_mask = torch.arange(24)[None, :] < 10
    actual, patch_mask = encoder._patchify(padded, padded_mask)

    torch.testing.assert_close(actual[..., :2], expected)
    assert patch_mask.tolist() == [[True, True, False]]
    assert torch.count_nonzero(actual[..., 2:]) == 0
    actual.sum().backward()
    assert torch.count_nonzero(padded.grad[..., 10:]) == 0


@pytest.mark.parametrize("patch_size", [4, 8, 16])
def test_patchformer_policy_and_value_do_not_depend_on_other_batch_inputs(patch_size):
    torch.manual_seed(1729)
    model = Model(
        dim=16,
        num_layers=2,
        head_size=4,
        num_heads=4,
        backbone="patch_transformer_cnn",
        backbone_kwargs={"patch_size": patch_size},
    ).eval()
    # Activate the context/value paths and keep padding embeddings nonzero,
    # as they are after the backbone's custom embedding initialization.
    with torch.no_grad():
        model.backbone.encode.tfm_out.fill_(1)
        model.backbone.embed.weight[0].fill_(3)
        model.rewards.out[-1].weight.normal_(std=0.1)
        text = "short\n@a\n@b"
        alone = model([text])
        mixed = model(["a much longer unrelated state\n@c\n@d", text])

    torch.testing.assert_close(alone.policy[0], mixed.policy[1])
    torch.testing.assert_close(alone.value.mean[0], mixed.value.mean[1])
    torch.testing.assert_close(alone.value.stddev[0], mixed.value.stddev[1])
