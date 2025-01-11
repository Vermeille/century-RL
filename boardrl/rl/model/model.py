from collections import namedtuple
import torch
import torch.nn as nn
from boardrl.rl.model.transformer import Rotary, Transformer


def mask_mean_pool(x, mask):
    # mask: BL1
    # x * mask: BLD * BL1 = BLD => BD
    # mask.sum(1): B1
    mask = mask.unsqueeze(-1)
    return (x * mask.to(x.dtype)).sum(1) / mask.to(x.dtype).sum(1)


def mask_energy_pool(x, mask):
    # mask: BL1
    # x * mask: BLD * BL1 = BLD => BD
    # mask.sum(1): B1
    mask = x.norm(dim=-1, keepdim=True) * mask.unsqueeze(-1)
    mask = mask / mask.to(x.dtype).sum(1, keepdim=True)
    # print( mask.squeeze(2) .sort(descending=True, dim=1) .values.cumsum(dim=1) .le(0.95) .float() .sum(1))
    return (x * mask.to(x.dtype)).sum(1)


def pool_first(x, mask):
    return x[:, 0]


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class PolicyValue:
    def __init__(self, policy, value):
        self.policy = policy
        self.value = value

    def __iter__(self):
        return iter([self.policy, self.value])

    def __len__(self):
        return len(self.policy)

    def unbatched(self):
        return [
            PolicyValue(
                [self.policy[i]],
                torch.distributions.Normal(
                    self.value.mean[i, None], self.value.scale[i, None]
                ),
            )
            for i in range(len(self))
        ]


class SinusoidalPositional(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, embedding_dim, max_seq_length=512, theta=10000):
        super().__init__()
        self.theta = theta
        self.make_pe(embedding_dim, max_seq_length)

    def make_pe(self, embedding_dim, max_seq_length):
        import math

        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(self.theta) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)


class ScaledSinosoidal(SinusoidalPositional):
    """Sinusoidal with scaling (see FLASH paper)."""

    def __init__(self, embedding_dim, max_seq_length, theta=10_000):
        super().__init__(embedding_dim, max_seq_length, theta)
        self.scale_factor = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        if input_ids.shape[1] > self.pe.shape[0]:
            self.make_pe(input_ids.shape[2], input_ids.shape[1])
            self.pe = self.pe.to(input_ids.device)
        return self.scale_factor * self.pe[: input_ids.shape[1], :] + input_ids


class ValueHead(nn.Module):
    def __init__(self, dim, head_size):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.tfblock = Transformer(dim, 1, dim // head_size, head_size)
        self.out = nn.Sequential(
            nn.Linear(dim, 2),
            # Scale(2),
            # B2
        )

    def forward(self, x, attn_mask):
        x = self.proj_in(x)
        x = self.tfblock(x, attn_mask)
        x = mask_mean_pool(x, attn_mask)
        out = self.out(x)
        return out


class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class PolicyHead(nn.Module):
    def __init__(self, dim, head_size):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.tfblock = Transformer(dim, 1, dim // head_size, head_size)
        self.out = nn.Sequential(
            # it looks Detrimental but actually smoothes the gradient norm
            # nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            # BL
        )

    def forward(self, x, attn_mask):
        x = self.proj_in(x)
        x = self.tfblock(x, attn_mask)
        x = self.out(x)
        return x.squeeze(-1)


class RotarySingle(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, seq_dim=-2):
        # B H L D
        seq_len = q.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(q.shape[seq_dim], device=q.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(q.device)
            self.cos_cached = emb.cos()[:, :]
            self.sin_cached = emb.sin()[:, :]
        return self.apply_rotary_pos_emb(q, self.cos_cached, self.sin_cached)

    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in torch < 1.8.0

    def apply_rotary_pos_emb(self, q, cos, sin):
        return (q * cos) + (self.rotate_half(q) * sin)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.zeros(max_len, dim), requires_grad=True)

    def forward(self, x):
        return x + self.pos_enc[: x.shape[1]]


class Model(nn.Module):
    def __init__(self, dim: int, num_layers: int, head_size: int = 64):
        super().__init__()
        self.maxlen = 2048
        self.in_embed = nn.Sequential(
            nn.Embedding(128, dim, padding_idx=0),
            nn.LayerNorm(dim),
            RotarySingle(dim),
            PositionalEncoding(dim, self.maxlen),
        )
        self.in_embed[0].weight.data.normal_(0, 0.02)
        self.encode = Transformer(
            dim, num_layers - 1, dim // head_size, head_size, num_conv_blocks=2
        )
        self.to_pred = PolicyHead(dim, head_size)
        self.rewards = ValueHead(dim, head_size)
        print(self)

    def text_encode(self, txts, maxlen):
        maxlen = min(maxlen, max(len(g) for g in txts))

        def do_pad(l):
            return l + [1] + [0] * (maxlen + 1 - len(l))

        txts = [torch.LongTensor(do_pad([ord(c) for c in txt])) for txt in txts]
        return torch.stack(txts, dim=0).to(self.in_embed[0].weight.device)

    def forward(self, games: list[str], return_hidden=False):
        txt = self.text_encode(games, self.maxlen)
        attn_mask = txt != 0
        enc = self.encode(self.in_embed(txt), attn_mask)
        pred = self.to_pred(enc, attn_mask)
        value = self.rewards(enc, attn_mask)

        moves_pos = [[i for i, c in enumerate(game) if c == "@"] for game in games]

        pred = [
            (pred[i, torch.tensor(moves_pos[i])] if len(moves_pos[i]) else [])
            for i in range(len(games))
        ]

        out = PolicyValue(
            pred,
            torch.distributions.Normal(
                value[:, 0], torch.nn.functional.softplus(value[:, 1])
            ),
        )
        if not return_hidden:
            return out
        else:
            return out, enc


def load_model(model_path):
    ckpt = torch.load(model_path, weights_only=False, map_location="cpu")

    if "config" not in ckpt:
        ckpt["config"] = {"dim": 256, "num_layers": 8}

    model = Model(ckpt["config"]["dim"], ckpt["config"]["num_layers"])
    print(model.load_state_dict(ckpt["model"]))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model
