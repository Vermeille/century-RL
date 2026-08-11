import torch
import torch.nn as nn
import torch.nn.functional as F
from boardrl.rl.model.transformer import Transformer
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


class ConvBlock(nn.Module):
    """One local, dense residual convolution."""

    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.norm = Norm(dim)
        self.conv1 = init(
            MaskedConv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
        )
        self.conv2 = zero(
            MaskedConv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

    def forward(self, x, mask):
        return x + self.conv2(F.gelu(self.conv1(self.norm(x), mask)), mask)


class CNNEncoder(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([ConvBlock(dim) for _ in range(num_layers)])

    def forward(self, x, mask):
        for m in self.blocks:
            x = m(x, mask)
        return x


class PatchTransformerCNNEncoder(nn.Module):
    """Local action parsing plus shared global reasoning at patch resolution."""

    def __init__(
        self,
        dim,
        global_layers,
        num_heads,
        head_size,
        patch_size,
        local_layers=2,
        canon_kernel_size=5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.local = CNNEncoder(dim, local_layers)
        self.downsample = init(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
        )
        self.global_context = Transformer(
            dim,
            global_layers,
            num_heads,
            head_size,
            rotary=False,
            canon_kernel_size=canon_kernel_size,
        )
        self.context_project = init(MaskedConv1d(dim, dim, kernel_size=5, padding=2))
        self.norm = Norm(dim)
        self.tfm_out = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask):
        x = self.norm(x)
        local = self.local(x, mask)  # BDL
        length = local.shape[-1]
        padding = (-length) % self.patch_size
        if padding:
            local_padded = F.pad(local, (0, padding))
            mask_padded = F.pad(mask, (0, padding), value=False)
        else:
            local_padded = local
            mask_padded = mask

        patches = self.downsample(local_padded)  # BDL
        patch_mask = mask_padded.view(
            len(mask),
            -1,
            self.patch_size,
        ).any(dim=-1)
        patches = self.global_context(
            self.norm(patches).transpose(1, 2),  # BLD
            patch_mask,
        ).transpose(1, 2)  # BDL

        context = patches.repeat_interleave(self.patch_size, dim=-1)[..., :length]
        return self.norm(self.context_project(local + self.tfm_out * context, mask))
