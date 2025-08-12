import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from boardrl.rl.model.transformer import SelfAttnOp


def normal_init(m, std):
    nn.init.normal_(m.weight, 0, std)
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


def xavier(m):
    nn.init.xavier_normal_(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


def kaiming(m):
    nn.init.kaiming_normal_(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


class GEGLU(nn.Module):
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


class ConvTrunkBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size):
        super().__init__()
        self.sa = nn.Sequential(
            nn.LayerNorm(hidden_size),
            Permute(0, 2, 1),
            xavier(
                nn.Conv1d(
                    hidden_size,
                    num_heads * head_size,
                    7,
                    padding=3,
                    groups=hidden_size,
                )
            ),
            normal_init(nn.Conv1d(hidden_size, hidden_size, 1), 0.02),
            Permute(0, 2, 1),
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden_size),
            kaiming(nn.Linear(hidden_size, 4 * hidden_size, bias=True)),
            GEGLU(),
            normal_init(nn.Linear(2 * hidden_size, hidden_size, bias=True), 0.02),
        )

    def forward(self, x, attn_mask):
        x = self.sa(x).masked_fill_(~attn_mask.unsqueeze(-1), 0.0) + x
        x = self.feed_forward(x) + x
        return x


class MeanPool(nn.Module):
    def forward(self, x, mask):
        mask = mask.unsqueeze(-1)
        return (x * mask.to(x.dtype)).sum(1) / mask.to(x.dtype).sum(1)


class EnergyPool(nn.Module):
    def forward(self, x, mask):
        mask = x.norm(dim=-1, keepdim=True) * mask.unsqueeze(-1)
        mask = mask / (1e-6 + mask.to(x.dtype).sum(1, keepdim=True))
        return (x * mask.to(x.dtype)).sum(1)


class FirstPool(nn.Module):
    def forward(self, x, mask):
        return x[:, 0]


class AttnPool(nn.Module):
    def __init__(self, head_size, num_heads, out_dim):
        super().__init__()
        self.attn = SelfAttnOp(head_size, num_heads)
        self.q = nn.Parameter(torch.randn(1, 1, head_size * num_heads))
        self.proj = nn.Linear(head_size * num_heads, head_size * num_heads * 2)
        self.out = nn.Linear(head_size * num_heads, num_heads * head_size)

    def forward(self, x, mask):
        k, v = self.proj(x).chunk(2, dim=-1)
        out = self.attn(self.q.expand(k.shape[0], -1, -1), k, v, mask)[:, 0]
        return out


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class SinusoidalPositional(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, theta=10000):
        super().__init__()
        self.theta = theta
        self.make_pe(embedding_dim, max_seq_length)

    def make_pe(self, embedding_dim, max_seq_length):
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(self.theta) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, input_ids):
        return input_ids + self.pe[: input_ids.size(1)]


class ScaledSinosoidal(SinusoidalPositional):
    def __init__(self, embedding_dim, max_seq_length, theta=10_000):
        super().__init__(embedding_dim, max_seq_length, theta)
        self.scale_factor = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, input_ids):
        return super().forward(input_ids) * torch.exp(self.scale_factor)
