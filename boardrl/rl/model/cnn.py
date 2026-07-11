import torch
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


class Norm(nn.RMSNorm):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvBlock1(nn.Module):
    def __init__(self, dim, kernel_size, dilation):
        super().__init__()
        self.norm = Norm(dim)
        self.expand = init(
            MaskedConv1d(
                dim,
                2 * dim,
                kernel_size=1,
                padding=0,
            )
        )
        self.dilated = init(
            MaskedConv1d(
                dim,
                dim,
                groups=dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * dilation // 2,
                dilation=dilation,
            )
        )
        self.local = init(
            MaskedConv1d(
                dim,
                dim,
                groups=dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.project = zero(
            MaskedConv1d(
                dim,
                dim,
                kernel_size=1,
                padding=0,
            )
        )

    def forward(self, x, mask):
        residual = x
        x = self.norm(x)
        x = F.glu(self.expand(x, mask), dim=1)
        x = F.gelu(self.dilated(x, mask))
        x = F.gelu(self.local(x, mask))
        return residual + self.project(x, mask)


class CNNEncoder(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvBlock1(dim, 5, dilation=2 ** (i % 6)) for i in range(num_layers)]
        )

    def forward(self, x, mask):
        for m in self.blocks:
            x = m(x, mask)
        return x
