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


class Norm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvBlock1(nn.Module):
    def __init__(self, dim, kernel_size, dilation):
        super().__init__()
        self.norm = Norm(dim)
        self.c1 = init(
            MaskedConv1d(
                dim,
                dim,
                groups=max(1, dim // 32),
                kernel_size=kernel_size,
                padding=(kernel_size // 2) * dilation,
                dilation=dilation,
            )
        )
        self.c2 = init(
            MaskedConv1d(
                dim,
                dim,
                # groups=max(1, dim // 32),
                kernel_size=1,
                padding=0,
            )
        )
        self.c3 = zero(
            MaskedConv1d(
                dim,
                dim,
                # groups=max(1, dim // 32),
                kernel_size=1,
                padding=0,
            )
        )

    def forward(self, x, mask):
        x = x + self.c3(F.gelu(self.c2(self.c1(self.norm(x), mask), mask)), mask)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvBlock1(dim, 5, dilation=2 ** (i % 4)) for i in range(num_layers)]
        )

    def forward(self, x, mask):
        for m in self.blocks:
            x = m(x, mask)
        return x
