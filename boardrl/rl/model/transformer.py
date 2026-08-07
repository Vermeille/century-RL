import torch
import torch.nn as nn
import torch.nn.functional as F

from boardrl.rl.model.utils import init, zero


class Canon(nn.Conv1d):
    def __init__(self, dim, kernel_size=5):
        super().__init__(dim, dim, kernel_size, groups=dim, bias=False)
        self.kernel_size = kernel_size
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, :, kernel_size // 2] = 1

    def forward(self, x):
        x = F.pad(x.transpose(1, 2), (self.kernel_size // 2, self.kernel_size // 2))
        return super().forward(x).transpose(1, 2)


class CausalCanon(nn.Conv1d):
    """Depthwise causal token mixing used by the original transformer."""

    def __init__(self, dim, kernel_size=4):
        super().__init__(dim, dim, kernel_size, groups=dim, bias=False)
        self.kernel_size = kernel_size
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, :, -1] = 1

    def forward(self, x):
        x = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        return super().forward(x).transpose(1, 2)


CANONS = {
    "bidirectional": Canon,
    "causal": CausalCanon,
}


def make_canon(name, dim, kernel_size=None):
    canon_cls = CANONS[name]
    return canon_cls(dim) if kernel_size is None else canon_cls(dim, kernel_size)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _cache(self, length, device, dtype):
        if (
            self.cos_cached is None
            or self.cos_cached.shape[0] < length
            or self.cos_cached.device != device
            or self.cos_cached.dtype != dtype
        ):
            t = torch.arange(length, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.float().to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)
        return self.cos_cached[:length], self.sin_cached[:length]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, x, seq_dim=-2):
        length = x.shape[seq_dim]
        cos, sin = self._cache(length, x.device, x.dtype)
        shape = [1] * x.ndim
        shape[seq_dim % x.ndim] = length
        shape[-1] = self.dim
        cos, sin = cos.reshape(shape), sin.reshape(shape)
        return x * cos + self.rotate_half(x) * sin

    def forward(self, q, k, v, seq_dim=-2):
        return self.apply(q, seq_dim), self.apply(k, seq_dim), v


class SelfAttnOp(nn.Module):
    def __init__(self, head_size, num_heads, rotary=False, alibi=False):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.rotary = Rotary(head_size) if rotary else None
        if alibi:
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
        b, lq, lk = q.shape[0], q.shape[2], k.shape[2]
        h, d = self.num_heads, self.head_size
        if self.rotary is not None:
            q, k, v = self.rotary(q, k, v)

        mask = None
        if attn_mask is not None:
            query = attn_mask[:, :lq].bool()
            key = attn_mask[:, :lk].bool()
            mask = (query.unsqueeze(-1) & key.unsqueeze(-2)).unsqueeze(1)

        att = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False
        )  # bhld
        return att


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size, rotary=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv = init(nn.Linear(hidden_size, head_size * num_heads * 4))
        self.fc = zero(nn.Linear(head_size * num_heads, hidden_size, bias=False))
        self.attn_op = SelfAttnOp(head_size, num_heads, rotary=rotary)

    def forward(self, x, attn_mask):
        b, l, h, d = x.shape[0], x.shape[1], self.num_heads, self.head_size
        q, k, v, gate = (
            self.qkv(x).reshape(b, l, 4, h, d).permute(2, 0, 3, 1, 4)
        )  # 4bhld
        att = self.attn_op(q, k, v, attn_mask) * torch.sigmoid(gate)
        att = att.permute(0, 2, 1, 3).contiguous().reshape(b, l, h * d)
        return self.fc(att)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.in_proj = init(nn.Linear(hidden_size, 8 * hidden_size))
        self.out_proj = zero(nn.Linear(4 * hidden_size, hidden_size, bias=False))

    def forward(self, x):
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(value * F.silu(gate))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_size,
        rotary=False,
        canon="bidirectional",
        canon_kernel_size=None,
    ):
        super().__init__()
        self.layer_norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False)
        self.canon_a = make_canon(canon, hidden_size, canon_kernel_size)
        self.sa = SelfAttention(hidden_size, num_heads, head_size, rotary=rotary)
        self.layer_norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False)
        self.canon_c = make_canon(canon, hidden_size, canon_kernel_size)
        self.feed_forward = SwiGLU(hidden_size)

    def forward(self, x, attn_mask):
        mask = attn_mask.unsqueeze(-1)
        x = x.masked_fill(~mask, 0)

        a = self.layer_norm1(x)
        a = self.canon_a(a).masked_fill(~mask, 0)
        x = x + self.sa(a, attn_mask)
        x = x.masked_fill(~mask, 0)

        c = self.layer_norm2(x)
        c = self.canon_c(c).masked_fill(~mask, 0)
        x = x + self.feed_forward(c)
        return x.masked_fill(~mask, 0)


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_heads,
        head_size,
        rotary=False,
        canon="bidirectional",
        canon_kernel_size=None,
    ):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    head_size,
                    rotary=rotary,
                    canon=canon,
                    canon_kernel_size=canon_kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, attn_mask):
        attn_mask = attn_mask.bool()
        for block in self.transformer_blocks:
            x = block(x, attn_mask)
        return x
