import torch
import torch.nn as nn
import torch.nn.functional as F
from boardrl.rl.model.transformer import Transformer


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

    def q_value(self) -> [torch.Tensor]:
        return [
            v + (a - a.mean() if len(a) != 0 else 0)
            for a, v in zip(self.policy, self.value.mean)
        ]




class ValueHead(nn.Module):
    def __init__(self, dim, head_size):
        super().__init__()
        # self.tfblock = Transformer(dim, 1, dim // head_size, head_size)
        self.out = nn.Sequential(
            # nn.LayerNorm(dim),  # Detrimental
            nn.Linear(dim, 2),
            # Scale(2),
            # B2
        )

    def forward(self, x, attn_mask):
        # x = self.tfblock(x, attn_mask)
        # x = mask_mean_pool(x, attn_mask)
        # x = self.pool(x, attn_mask)
        x = x[:, 0]
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
        # self.tfblock = Transformer(dim, 1, dim // head_size, head_size)
        self.out = nn.Sequential(
            # it IS Detrimental
            # nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            # BL
            # nn.LogSoftmax(dim=1), # THIS IS WRONG BECAUSE WE SELECT AFTER
        )

    def forward(self, x, attn_mask):
        # x = self.tfblock(x, attn_mask)
        x = self.out(x)  # BLD
        return x.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(max_len, dim) / dim)

    def forward(self, x):
        return x + self.pos_enc[: x.shape[1]]


class Backbone(nn.Module):
    """Common interface for model backbones."""

    def forward(self, tokens, attn_mask):  # pragma: no cover - interface only
        raise NotImplementedError


class TransformerBackbone(Backbone):
    def __init__(
        self,
        dim,
        num_layers,
        head_size,
        max_len,
        rotary: bool = False,
        rotary_single: bool = True,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Embedding(128, dim, padding_idx=0),
            PositionalEncoding(dim, max_len),
        )
        self.encode = Transformer(
            dim,
            num_layers,
            dim // head_size,
            head_size,
            rotary=rotary,
            rotary_single=rotary_single,
        )

    def forward(self, tokens, attn_mask):
        emb = self.embed(tokens)
        return self.encode(emb, attn_mask)


class LSTMBackbone(Backbone):
    def __init__(self, dim, num_layers, head_size, max_len):
        super().__init__()
        self.embed = nn.Embedding(128, dim, padding_idx=0)
        self.encode = nn.LSTM(dim, dim, num_layers, batch_first=True)

    def forward(self, tokens, attn_mask):
        emb = self.embed(tokens)
        enc, _ = self.encode(emb)
        return enc * attn_mask.unsqueeze(-1)


BACKBONES = {
    "transformer": TransformerBackbone,
    "lstm": LSTMBackbone,
}


class Model(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        head_size: int = 64,
        backbone: str = "transformer",
    ):
        super().__init__()
        self.maxlen = 2048
        self.backbone_name = backbone
        backbone_cls = BACKBONES.get(backbone)
        if backbone_cls is None:
            raise ValueError(f"Unknown backbone {backbone}")
        self.backbone = backbone_cls(dim, num_layers, head_size, self.maxlen)

        self.to_pred = PolicyHead(dim, head_size)
        self.rewards = ValueHead(dim, head_size)

    def text_encode(self, txts, maxlen):
        maxlen = min(maxlen, max(len(g) for g in txts))

        def do_pad(l):
            return l + [1] + [0] * (maxlen + 1 - len(l))

        txts = [torch.LongTensor(do_pad([ord(c) for c in txt])) for txt in txts]
        device = next(self.backbone.parameters()).device
        return torch.stack(txts, dim=0).to(device)

    def forward(self, games: list[str], return_hidden=False):
        txt = self.text_encode(games, self.maxlen)
        attn_mask = txt != 0
        enc = self.backbone(txt, attn_mask)
        policy_logits = self.to_pred(enc, attn_mask)
        value = self.rewards(enc, attn_mask)

        moves_pos = [[i for i, c in enumerate(game) if c == "@"] for game in games]

        pred = []
        for i in range(len(games)):
            if len(moves_pos[i]):
                logits = policy_logits[i, torch.tensor(moves_pos[i])]
                pred.append(F.log_softmax(logits, dim=0))
            else:
                pred.append(policy_logits[i, torch.tensor([], dtype=torch.long)])

        out = PolicyValue(
            pred,
            torch.distributions.Normal(
                value[:, 0],
                torch.nn.functional.softplus(value[:, 1]),
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

    config = ckpt["config"]
    model = Model(
        config.get("dim", 256),
        config.get("num_layers", 8),
        head_size=config.get("head_size", 64),
        backbone=config.get("backbone", "transformer"),
    )
    print(model.load_state_dict(ckpt["model"]))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model
