from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from boardrl.rl.model.transformer import Transformer
from boardrl.rl.model.gated_cnn import GatedCNNEncoder
from boardrl.rl.model.cnn import (
    CNNEncoder,
    PatchTransformerCNNEncoder,
)
from boardrl.rl.model.utils import zero, init


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

    def q_value(self) -> List[torch.Tensor]:
        return [
            v + (a - a.mean() if len(a) != 0 else 0)
            for a, v in zip(self.policy, self.value.mean)
        ]


class ValueHead(nn.Module):
    """Pool the encoded state with one fixed learned cross-attention query."""

    def __init__(self, dim, initial_value_scale=1.0, num_heads=4):
        super().__init__()
        if dim % num_heads:
            raise ValueError("value-head dimension must divide the head count")
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True,
        )
        self.norm = nn.RMSNorm(dim)
        self.out = zero(nn.Linear(dim, 2))
        self.raw_value_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_value_scale)))
        )

    def forward(self, x, attn_mask):
        query = self.query.expand(len(x), -1, -1)
        summary, _ = self.attention(
            query,
            x,
            x,
            key_padding_mask=~attn_mask,
            need_weights=False,
        )
        out = self.out(self.norm(summary[:, 0]))
        value_scale = F.softplus(self.raw_value_scale)
        return value_scale * out


class PolicyHead(nn.Module):
    """Score action tokens directly; global reasoning belongs to the backbone."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.RMSNorm(dim)
        self.out = init(nn.Linear(dim, 1), var_scale=0.1)

    def actions(self, x, mask, positions):
        counts = [len(indices) for indices in positions]
        max_actions = max(counts, default=0)
        if max_actions == 0:
            return [x.new_empty(0) for _ in positions]

        index_matrix = torch.zeros(
            len(positions),
            max_actions,
            dtype=torch.long,
            device=x.device,
        )
        for batch, action_indices in enumerate(positions):
            if action_indices:
                action_positions = torch.tensor(action_indices, device=x.device)
                index_matrix[batch, : len(action_positions)] = action_positions

        features = x.gather(
            1,
            index_matrix.unsqueeze(-1).expand(-1, -1, self.dim),
        )
        logits = self.out(F.gelu(self.norm(features))).squeeze(-1)
        return [row[:count] for row, count in zip(logits, counts)]


class Backbone(nn.Module):
    """Common interface for model backbones."""

    def forward(self, tokens, attn_mask):  # pragma: no cover - interface only
        raise NotImplementedError


class TransformerBackbone(Backbone):
    def __init__(
        self,
        dim,
        num_layers,
        head_size=None,
        max_len=2048,
        num_heads=None,
        rotary: bool = True,
        rotary_single: bool = False,
    ):
        super().__init__()
        if head_size is None and num_heads is None:
            head_size = 64
            num_heads = dim // head_size
        elif head_size is None:
            head_size = dim // num_heads
        elif num_heads is None:
            num_heads = dim // head_size
        self.head_size = head_size
        self.num_heads = num_heads
        embedding = nn.Embedding(128, dim, padding_idx=0)
        embedding.weight.data.normal_(0, 0.1)
        self.embed = nn.Sequential(
            embedding,
            nn.RMSNorm(dim),
        )
        self.encode = Transformer(
            dim,
            num_layers,
            num_heads,
            head_size,
            rotary=rotary,
            rotary_single=rotary_single,
        )

    def forward(self, tokens, attn_mask):
        emb = self.embed(tokens)
        return self.encode(emb, attn_mask)


class LSTMBackbone(Backbone):
    def __init__(self, dim, num_layers, head_size=None, max_len=None, num_heads=None):
        super().__init__()
        assert head_size is None and num_heads is None
        self.embed = nn.Embedding(128, dim, padding_idx=0)
        self.encode = nn.LSTM(
            dim, dim, num_layers, batch_first=True, bidirectional=True
        )

    def forward(self, tokens, attn_mask):
        emb = self.embed(tokens)
        enc, _ = self.encode(emb)
        x0, x1 = torch.chunk(enc, 2, dim=2)
        enc = x0 + x1
        return enc * attn_mask.unsqueeze(-1)


class GatedCNNBackbone(Backbone):
    """Backbone using the :class:`GatedCNNEncoder`."""

    def __init__(self, dim, num_layers, head_size=None, max_len=None, num_heads=None):
        super().__init__()
        assert head_size is None and num_heads is None
        self.embed = nn.Embedding(128, dim, padding_idx=0)
        self.encode = GatedCNNEncoder(d_model=dim, n_blocks=num_layers)

    def forward(self, tokens, attn_mask):
        emb = self.embed(tokens)
        return self.encode(emb, attn_mask)


class CNNBackbone(Backbone):
    """Backbone using the :class:`CNNEncoder`."""

    def __init__(self, dim, num_layers, head_size=None, max_len=None, num_heads=None):
        super().__init__()
        assert head_size is None and num_heads is None
        self.embed = init(nn.Embedding(128, dim, padding_idx=0))
        self.norm = nn.LayerNorm(dim)
        self.encode = CNNEncoder(dim=dim, num_layers=num_layers)

    def forward(self, tokens, attn_mask):
        emb = self.embed(tokens).transpose(1, 2)
        return self.encode(emb, attn_mask).transpose(1, 2)


class PatchTransformerCNNBackbone(Backbone):
    def __init__(
        self,
        dim,
        num_layers,
        head_size=None,
        max_len=None,
        num_heads=None,
        patch_size=4,
    ):
        super().__init__()
        if head_size is None and num_heads is None:
            num_heads = 4
            head_size = dim // num_heads
        elif head_size is None:
            head_size = dim // num_heads
        elif num_heads is None:
            num_heads = dim // head_size
        self.embed = init(nn.Embedding(128, dim, padding_idx=0))
        self.encode = PatchTransformerCNNEncoder(
            dim=dim,
            global_layers=num_layers,
            num_heads=num_heads,
            head_size=head_size,
            patch_size=patch_size,
        )

    def forward(self, tokens, attn_mask):
        emb = self.embed(tokens).transpose(1, 2)
        return self.encode(emb, attn_mask).transpose(1, 2)


BACKBONES = {
    "transformer": TransformerBackbone,
    "lstm": LSTMBackbone,
    "gated_cnn": GatedCNNBackbone,
    "cnn": CNNBackbone,
    "patch_transformer_cnn": PatchTransformerCNNBackbone,
}


class Model(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        head_size: int | None = None,
        num_heads: int | None = None,
        backbone: str = "transformer",
        shared_backbone: bool = True,
        backbone_kwargs: dict | None = None,
    ):
        super().__init__()
        backbone_kwargs = dict(backbone_kwargs or {})
        self.maxlen = 2048
        self._spec = {
            "dim": dim,
            "num_layers": num_layers,
            "head_size": head_size,
            "num_heads": num_heads,
            "backbone": backbone,
            "shared_backbone": shared_backbone,
            "backbone_kwargs": backbone_kwargs,
        }
        self.backbone_name = backbone
        self.shared_backbone = shared_backbone
        backbone_cls = BACKBONES.get(backbone)
        if backbone_cls is None:
            raise ValueError(f"Unknown backbone {backbone}")
        self.backbone = backbone_cls(
            dim,
            num_layers,
            head_size,
            self.maxlen,
            num_heads,
            **backbone_kwargs,
        )
        if not shared_backbone:
            self.policy_backbone = self.backbone
            self.value_backbone = backbone_cls(
                dim,
                num_layers,
                head_size,
                self.maxlen,
                num_heads,
                **backbone_kwargs,
            )
            del self.backbone
        self.to_pred = PolicyHead(dim)
        self.rewards = ValueHead(dim)

    def spec(self):
        """Constructor arguments needed to recreate this model."""
        return dict(self._spec)

    def text_encode(self, txts, maxlen):
        maxlen = min(maxlen, max(len(g) for g in txts))

        def do_pad(l):
            return l + [1] + [0] * (maxlen + 1 - len(l))

        txts = [torch.LongTensor(do_pad([ord(c) for c in txt])) for txt in txts]
        device = next(self.parameters()).device
        return torch.stack(txts, dim=0).to(device)

    def forward(self, games: list[str], return_hidden=False):
        txt = self.text_encode(games, self.maxlen)
        attn_mask = txt != 0
        if self.shared_backbone:
            enc = self.backbone(txt, attn_mask)
            value_enc = enc
        else:
            enc = self.policy_backbone(txt, attn_mask)
            value_enc = self.value_backbone(txt, attn_mask)
        assert enc.shape[:-1] == txt.shape
        assert value_enc.shape[:-1] == txt.shape
        moves_pos = [[i for i, c in enumerate(game) if c == "@"] for game in games]
        pred = self.to_pred.actions(enc, attn_mask, moves_pos)
        value = self.rewards(value_enc, attn_mask)

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


def load_model(model_path, name=None):
    ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
    name = name or next(iter(ckpt["models"]))
    config = ckpt["model_specs"][name]
    state = ckpt["models"][name]
    model = Model(
        config["dim"],
        config["num_layers"],
        config.get("head_size"),
        config.get("num_heads"),
        backbone=config.get("backbone", "transformer"),
        shared_backbone=config.get("shared_backbone", True),
        backbone_kwargs=config.get("backbone_kwargs"),
    )
    model.load_state_dict(state)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model
