import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConvBlock(nn.Module):
    """Depthwise-separable gated convolutional block.

    Architecture: PreNorm -> pointwise (2C) -> GLU -> depthwise conv ->
    Squeeze-Excitation -> pointwise out -> Dropout -> Residual.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        se_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "use odd kernel for SAME padding"

        self.norm = nn.RMSNorm(d_model)
        self.pw_in = nn.Linear(d_model, 2 * d_model, bias=True)

        pad = (kernel_size // 2) * dilation
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=pad,
            dilation=dilation,
            groups=d_model,
            bias=True,
        )

        hidden = max(8, d_model // se_ratio)
        self.se_down = nn.Linear(d_model, hidden, bias=True)
        self.se_up = nn.Linear(hidden, d_model, bias=True)

        self.pw_out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """x: (B,T,C), attn_mask: (B,T) with 1=keep, 0=pad"""
        residual = x
        x = self.norm(x)
        a, b = self.pw_in(x).chunk(2, dim=-1)
        x = a * torch.sigmoid(b)

        x = x.transpose(1, 2)
        x = self.dw(x)
        x = x.transpose(1, 2)

        s = x.mean(dim=1)
        s = F.silu(self.se_down(s))
        s = torch.sigmoid(self.se_up(s))
        x = x * s.unsqueeze(1)

        x = self.pw_out(x)
        x = self.dropout(x)

        if attn_mask is not None:
            m = attn_mask.unsqueeze(-1)
            return residual * (1 - m) + (residual + x) * m
        return residual + x


class GatedCNNEncoder(nn.Module):
    """Stack of :class:`GatedConvBlock` with optional stochastic depth."""

    def __init__(
        self,
        d_model: int = 256,
        n_blocks: int = 10,
        kernel_size: int = 5,
        dilations: tuple = (1, 2, 4, 8, 16),
        se_ratio: int = 4,
        dropout: float = 0.1,
        stochastic_depth: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList()
        self.drop_path_rates = self._make_drop_path_rates(n_blocks, stochastic_depth)

        for i in range(n_blocks):
            d = dilations[i % len(dilations)]
            self.blocks.append(
                GatedConvBlock(
                    d_model,
                    kernel_size,
                    d,
                    se_ratio=se_ratio,
                    dropout=dropout,
                )
            )

        self.out_norm = nn.RMSNorm(d_model)

    @staticmethod
    def _make_drop_path_rates(n: int, sd: float):
        if sd <= 0:
            return [0.0] * n
        return [sd * i / max(1, n - 1) for i in range(n)]

    def forward(self, x, attn_mask=None):
        for block, rate in zip(self.blocks, self.drop_path_rates):
            h = block(x, attn_mask=attn_mask)
            if rate > 0.0 and self.training:
                keep = 1.0 - rate
                if keep < 1.0:
                    shape = (x.size(0),) + (1,) * (h.dim() - 1)
                    mask = (torch.rand(shape, device=h.device) < keep).float() / keep
                    h = x + (h - x) * mask
            x = h
        return self.out_norm(x)
