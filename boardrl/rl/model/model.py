from collections import namedtuple
import torch
import torch.nn as nn
from boardrl.rl.model.transformer import Transformer


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
    return (x * mask.to(x.dtype)).sum(1) / mask.to(x.dtype).sum(1)


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

    def __init__(self, embedding_dim, max_seq_length=5000):
        super().__init__()
        self.make_pe(embedding_dim, max_seq_length)

    def make_pe(self, embedding_dim, max_seq_length):
        import math

        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)


class ScaledSinosoidal(SinusoidalPositional):
    """Sinusoidal with scaling (see FLASH paper)."""

    def __init__(self, embedding_dim, max_seq_length):
        super().__init__(embedding_dim, max_seq_length)
        self.scale_factor = torch.nn.Parameter(
            0.02 * torch.tensor([1.0 / embedding_dim**0.5])
        )

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
        self.tfblock = Transformer(dim, 1, dim // head_size, head_size)
        self.out = nn.Sequential(
            # nn.LayerNorm(dim),
            # nn.GELU(),
            nn.Linear(dim, 2),
            Scale(2),
            # B2
        )

    def forward(self, x, attn_mask):
        x = self.tfblock(x, attn_mask)
        x = mask_energy_pool(x, attn_mask)
        # x = x[:, 0]
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
        self.tfblock = Transformer(dim, 1, dim // head_size, head_size)
        self.out = nn.Sequential(
            # nn.LayerNorm(dim),
            # nn.GELU(),
            nn.Linear(dim, 1),
            # BL
        )

    def forward(self, x, attn_mask):
        x = self.tfblock(x, attn_mask)
        # m = mask_pool(x, attn_mask)  # BD
        x = self.out(x)
        # out = torch.bmm(x, m.unsqueeze(-1)).squeeze(-1)
        # return out
        return x.squeeze(-1)


class Model(nn.Module):
    def __init__(self, dim: int, num_layers: int, head_size: int = 64):
        super().__init__()
        self.maxlen = 2048
        self.in_embed = nn.Sequential(
            nn.Embedding(128, dim, padding_idx=0),
            nn.LayerNorm(dim),
            ScaledSinosoidal(dim, self.maxlen),
        )
        self.in_embed[0].weight.data.normal_(0, 1 / dim**0.5)
        self.encode = Transformer(
            dim, num_layers - 1, dim // head_size, head_size, num_conv_blocks=4
        )
        self.to_pred = PolicyHead(dim, head_size)
        self.rewards = ValueHead(dim, head_size)

    def text_encode(self, txts, maxlen, pad=False):
        def do_pad(l):
            if pad:
                return l + [1] + [0] * (maxlen - len(l) - 1)
            else:
                return l

        txts = [torch.LongTensor(do_pad([ord(c) for c in txt])) for txt in txts]
        return torch.stack(txts, dim=0).to(self.in_embed[0].weight.device)

    def text_embed(self, txts, maxlen, pad=False):
        txts = self.text_encode(txts, maxlen, pad=pad)
        return txts

    def forward(self, games: list[str], samples=None):
        txt = self.text_embed(games, self.maxlen, pad=True)
        attn_mask = txt != 0
        enc = self.encode(self.in_embed(txt), attn_mask)
        pred = self.to_pred(enc, attn_mask)
        value = self.rewards(enc, attn_mask)

        moves_pos = [[i for i, c in enumerate(game) if c == "@"] for game in games]

        pred = [
            (pred[i][torch.tensor(moves_pos[i])] if len(moves_pos[i]) else [])
            for i in range(len(games))
        ]

        return PolicyValue(
            pred,
            torch.distributions.Normal(
                value[:, 0], torch.nn.functional.softplus(value[:, 1])
            ),
        )


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
