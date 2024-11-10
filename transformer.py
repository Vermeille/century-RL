import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(m, std):
    assert isinstance(m.weight, torch.Tensor)
    nn.init.normal_(m.weight, 0, std)
    if hasattr(m, "biais") and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


def xavier(m):
    assert isinstance(m.weight, torch.Tensor)
    nn.init.xavier_normal_(m.weight)
    if hasattr(m, "biais") and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


def kaiming(m):
    assert isinstance(m.weight, torch.Tensor)
    nn.init.kaiming_normal_(m.weight)
    if hasattr(m, "biais") and m.bias is not None:
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


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv = normal_init(
            nn.Linear(hidden_size, head_size * num_heads * 3, bias=True),
            math.sqrt(2 / (5 * hidden_size)),
        )
        self.fc = normal_init(
            nn.Linear(head_size * num_heads, hidden_size, bias=True), 0.0
        )
        self.rotary = Rotary(head_size)

    def forward(self, x):
        b, l, h, d = x.shape[0], x.shape[1], self.num_heads, self.head_size
        # bld -> (q/k/v)bhld
        qkv = self.qkv(x).reshape(b, l, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = self.rotary(q, k, v)
        att = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        # bhld -> blhd
        att = att.permute(0, 2, 1, 3).contiguous().reshape(b, l, h * d)
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


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.sa = SelfAttention(hidden_size, num_heads, head_size)
        self.feed_forward = nn.Sequential(
            kaiming(nn.Linear(hidden_size, 4 * hidden_size, bias=True)),
            Permute(0, 2, 1),
            nn.BatchNorm1d(4 * hidden_size),
            Permute(0, 2, 1),
            GEGLU(),
            normal_init(nn.Linear(2 * hidden_size, hidden_size, bias=True), 0.0),
        )

    def forward(self, x):
        x = self.sa(self.layer_norm1(x)) + x
        x = self.feed_forward(x).add_(x)
        return x


class Transformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, head_size):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, head_size)
                for _ in range(num_layers)
            ]
        )

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
                m.eps = 1e-6

    def forward(self, x):
        for i, transformer_block in enumerate(self.transformer_blocks):
            x = transformer_block(x)
        return x
