import torch.nn as nn
import torch.nn.functional as F
from boardrl.rl.model.utils import zero, init


class MaskedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, mask):
        # input: BCL, mask: BL
        mask = mask.unsqueeze(1)
        return super().forward(input * mask) * mask


class ConvBlock1(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.c1 = init(
            MaskedConv1d(
                dim,
                dim,
                groups=max(1, dim // 16),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.c2 = zero(
            MaskedConv1d(
                dim,
                dim,
                groups=max(1, dim // 16),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

    def forward(self, x, mask):
        x = x + self.c2(
            F.gelu(self.c1(self.norm(x.transpose(1, 2)).transpose(1, 2), mask)), mask
        )
        return x


class CNNEncoder(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([ConvBlock1(dim, 5) for _ in range(num_layers)])

    def forward(self, x, mask):
        for m in self.blocks:
            x = m(x, mask)
        return x
