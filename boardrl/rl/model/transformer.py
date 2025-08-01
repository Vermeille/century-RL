import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DynamicTanh


def normal_init(m, std):
    assert isinstance(m.weight, torch.Tensor)
    nn.init.normal_(m.weight, 0, std)
    if hasattr(m, "bias") and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


def xavier(m):
    assert isinstance(m.weight, torch.Tensor)
    nn.init.xavier_normal_(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


def kaiming(m):
    assert isinstance(m.weight, torch.Tensor)
    nn.init.kaiming_normal_(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, k, v, seq_dim=-2):
        # B H L D
        seq_len = q.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(q.shape[seq_dim], device=q.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(q.device)
            self.cos_cached = emb.cos()[:, :]
            self.sin_cached = emb.sin()[:, :]
        return self.apply_rotary_pos_emb(q, k, v, self.cos_cached, self.sin_cached)

    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in torch < 1.8.0

    def apply_rotary_pos_emb(self, q, k, v, cos, sin):
        return (
            (q * cos) + (self.rotate_half(q) * sin),
            (k * cos) + (self.rotate_half(k) * sin),
            v,
        )


class SelfAttnOp(nn.Module):
    def __init__(self, head_size, num_heads, rotary=False, alibi=False):
        super().__init__()
        self.rotary = None
        self.num_heads = num_heads
        self.head_size = head_size
        if rotary:
            self.rotary = Rotary(head_size)

        if alibi:
            print("WARNING: Using alibi uses a fuckton of memory")
            self.register_buffer(
                "alibi",
                torch.tensor(
                    [
                        1 / ((2**8) ** (1 / num_heads)) ** (h + 1)
                        for h in range(num_heads)
                    ]
                ),
            )
        else:
            self.alibi = None

    def forward(self, q, k, v, attn_mask):
        # q, k, v: B L D
        b, d, h = q.shape[0], self.head_size, self.num_heads
        lq = q.shape[1]
        lk = k.shape[1]
        # BLD->BHLD
        q = q.reshape(b, lq, h, d).permute(0, 2, 1, 3)
        k = k.reshape(b, lk, h, d).permute(0, 2, 1, 3)
        v = v.reshape(b, lk, h, d).permute(0, 2, 1, 3)

        if self.rotary is not None:
            q, k, v = self.rotary(q, k, v)

        attn_mask = attn_mask.unsqueeze(1) & attn_mask.unsqueeze(2)
        attn_mask = attn_mask.unsqueeze(1)
        if self.alibi is not None:
            mask = -torch.abs(
                torch.arange(q.shape[-2]).unsqueeze(1) - torch.arange(k.shape[-2])
            ).to(attn_mask.device) * self.alibi.unsqueeze(-1).unsqueeze(-1)
            attn_mask = torch.where(attn_mask.bool(), mask, float("-inf"))

        att = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False, attn_mask=attn_mask[:, :, :lq, :]
        )
        # BHLD->BLHD
        att = att.permute(0, 2, 1, 3).contiguous().reshape(b, lq, h * d)
        return att


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size, rotary=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv = xavier(
            nn.Linear(hidden_size, head_size * num_heads * 3, bias=True),
        )
        with torch.no_grad():
            self.qkv.weight[: head_size * num_heads, :].copy_(
                self.qkv.weight[head_size * num_heads : head_size * num_heads * 2, :]
            )
        self.fc = xavier(nn.Linear(head_size * num_heads, hidden_size, bias=True))
        # Rotary here is detrimental, it's better to use it in the trunk
        self.attn_op = SelfAttnOp(head_size, num_heads, rotary=rotary, alibi=False)

    def forward(self, x, attn_mask):
        # bld -> (q/k/v)bl(hd)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)

        att = self.attn_op(q, k, v, attn_mask)
        # bhld -> blhd

        return self.fc(att)


class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class Permute(nn.Module):
    def __init__(self, *transpo):
        super().__init__()
        self.transpo = transpo

    def forward(self, x):
        return x.permute(*self.transpo)


class GatedResidual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gating = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, y):
        return torch.sigmoid(self.gating(x)) * y + x


def just_add(x, y):
    return x + y


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size, rotary=False):
        super().__init__()
        self.layer_norm1 = DynamicTanh(hidden_size)
        self.sa = SelfAttention(hidden_size, num_heads, head_size, rotary=rotary)
        self.feed_forward = nn.Sequential(
            DynamicTanh(hidden_size),
            kaiming(
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            ),  # bias is better
            GEGLU(),  # better than GELU
            normal_init(nn.Linear(2 * hidden_size, hidden_size, bias=True), 0.02),
        )
        # GatedResidual is better than just_add. Not sure why.
        self.residual1 = GatedResidual(hidden_size)
        self.residual2 = GatedResidual(hidden_size)

    def forward(self, x, attn_mask):
        x = self.residual1(x, self.sa(self.layer_norm1(x), attn_mask))
        x = self.residual2(x, self.feed_forward(x))
        return x


class ConvTrunkBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size):
        super().__init__()
        self.sa = nn.Sequential(
            DynamicTanh(hidden_size),
            Permute(0, 2, 1),  # bld -> bdl
            xavier(
                nn.Conv1d(
                    hidden_size,
                    num_heads * head_size,
                    7,
                    padding=3,
                    groups=hidden_size,
                )
            ),
            # nn.GELU(),
            normal_init(nn.Conv1d(hidden_size, hidden_size, 1), 0.02),
            Permute(0, 2, 1),  # bdl -> bld
        )
        self.feed_forward = nn.Sequential(
            DynamicTanh(hidden_size),
            kaiming(
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            ),  # bias is better
            GEGLU(),  # better than GELU
            normal_init(nn.Linear(2 * hidden_size, hidden_size, bias=True), 0.02),
        )
        # self.sa[2].weight.data.fill_(1 / 7.0)

    def forward(self, x, attn_mask):
        x = self.sa(x).masked_fill_(~attn_mask.unsqueeze(-1), 0.0) + x
        x = self.feed_forward(x) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_heads,
        head_size,
        num_conv_blocks=0,
        rotary=False,
    ):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                (
                    ConvTrunkBlock(hidden_size, num_heads, head_size)
                    if i < num_conv_blocks
                    else TransformerBlock(
                        hidden_size, num_heads, head_size, rotary=rotary
                    )
                )
                for i in range(num_layers)
            ]
        )

        for m in self.modules():
            if isinstance(m, DynamicTanh):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
                m.alpha.data.fill_(1.0)

    def forward(self, x, attn_mask):
        for i, transformer_block in enumerate(self.transformer_blocks):
            x = transformer_block(x, attn_mask)
        return x
