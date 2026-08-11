import torch
import math
from boardrl.rl.model.transformer import Rotary


def test_rotate_half_involution():
    rot = Rotary(dim=4)
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # B=1,H=1,L=1,D=4
    y = rot.rotate_half(x)
    z = rot.rotate_half(y)
    assert torch.allclose(z, -x)


def test_rotary_forward_matches_manual():
    dim = 8
    rot = Rotary(dim)
    B, H, L = 2, 3, 5
    q = torch.randn(B, H, L, dim)
    k = torch.randn(B, H, L, dim)
    v = torch.randn(B, H, L, dim)
    q_out, k_out, v_out = rot(q, k, v)

    t = torch.arange(L, device=q.device).type_as(rot.inv_freq)
    freqs = torch.einsum("i,j->ij", t, rot.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    def manual(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=x1.ndim - 1)
        return x * cos + rotated * sin

    assert torch.allclose(q_out, manual(q))
    assert torch.allclose(k_out, manual(k))
    assert torch.allclose(v_out, v)
