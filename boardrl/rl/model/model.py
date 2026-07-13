from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from boardrl.rl.model.transformer import Transformer
from boardrl.rl.model.gated_cnn import GatedCNNEncoder
from boardrl.rl.model.cnn import CNNEncoder
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


class VariancePreservingAttentionPool(nn.Module):
    """Learned pooling initialized as ``sum(x) / sqrt(length)``.

    If the token states are independent with equal variance, a weighted sum
    has variance proportional to ``sum(weights ** 2)``.  Normalizing by that
    quantity preserves the variance while allowing the head to focus on the
    useful parts of the input.
    """

    def __init__(self, dim):
        super().__init__()
        # Uniform attention at initialization recovers the existing pool.
        self.score = zero(nn.Linear(dim, 1, bias=False))

    def forward(self, x, mask):
        mask = mask.bool()
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -torch.inf)
        weights = F.softmax(scores, dim=1)

        weight_norm = weights.square().sum(dim=1, keepdim=True).sqrt()
        weight_norm = weight_norm.clamp_min(1e-6)
        pooled = torch.einsum("bl,bld->bd", weights, x)
        return pooled / weight_norm


class ValueHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = VariancePreservingAttentionPool(dim)
        self.out = nn.Sequential(
            zero(nn.Linear(dim, 2)),
        )

    def forward(self, x, attn_mask):
        x = self.pool(x, attn_mask)
        out = self.out(x)
        return out


class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class PolicyHead(nn.Module):
    def __init__(self, dim, initial_logit_scale=1.0):
        super().__init__()
        self.dim = dim
        self.pool = VariancePreservingAttentionPool(dim)
        self.mean_proj = nn.Linear(dim, dim)
        self.pred_proj = nn.Linear(dim, dim)
        self.action_bias = zero(nn.Linear(dim, 1))
        self.raw_logit_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_logit_scale)))
        )

    def forward(self, x, mask):
        query = self.mean_proj(self.pool(x, mask))
        keys = self.pred_proj(x)
        compatibility = torch.einsum("bd,bld->bl", query, keys)
        compatibility = compatibility / math.sqrt(self.dim)
        logit_scale = F.softplus(self.raw_logit_scale)
        return logit_scale * compatibility + self.action_bias(keys).squeeze(-1)


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
        self.embed = nn.Sequential(
            nn.Embedding(128, dim, padding_idx=0),
            nn.RMSNorm(dim),
        )
        self.embed[0].weight.data.normal_(0, 0.1)
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


BACKBONES = {
    "transformer": TransformerBackbone,
    "lstm": LSTMBackbone,
    "gated_cnn": GatedCNNBackbone,
    "cnn": CNNBackbone,
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
    ):
        super().__init__()
        self.maxlen = 2048
        self.backbone_name = backbone
        self.shared_backbone = shared_backbone
        backbone_cls = BACKBONES.get(backbone)
        if backbone_cls is None:
            raise ValueError(f"Unknown backbone {backbone}")
        self.backbone = backbone_cls(dim, num_layers, head_size, self.maxlen, num_heads)
        if not shared_backbone:
            self.policy_backbone = self.backbone
            self.value_backbone = backbone_cls(
                dim, num_layers, head_size, self.maxlen, num_heads
            )
            del self.backbone
        self.to_pred = PolicyHead(dim)
        self.rewards = ValueHead(dim)

    def _normalize_backbone_state_dict(self, state_dict):
        has_backbone = any(k.startswith("backbone.") for k in state_dict)
        has_policy = any(k.startswith("policy_backbone.") for k in state_dict)
        has_value = any(k.startswith("value_backbone.") for k in state_dict)

        if self.shared_backbone:
            if has_backbone:
                return {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith(("policy_backbone.", "value_backbone."))
                }

            old_prefix = "policy_backbone." if has_policy else "value_backbone."
            if has_policy or has_value:
                normalized = {}
                for key, value in state_dict.items():
                    if key.startswith(old_prefix):
                        suffix = key[len(old_prefix) :]
                        normalized[f"backbone.{suffix}"] = value
                    elif not key.startswith(("policy_backbone.", "value_backbone.")):
                        normalized[key] = value
                return normalized

            return state_dict

        if has_policy or has_value:
            if has_policy and has_value:
                return {
                    k: v for k, v in state_dict.items() if not k.startswith("backbone.")
                }

            old_prefix = "policy_backbone." if has_policy else "value_backbone."
            normalized = {}
            for key, value in state_dict.items():
                if key.startswith(old_prefix):
                    suffix = key[len(old_prefix) :]
                    normalized[f"policy_backbone.{suffix}"] = value
                    normalized[f"value_backbone.{suffix}"] = value.clone()
                elif not key.startswith(("policy_backbone.", "value_backbone.")):
                    normalized[key] = value
            return normalized

        if has_backbone:
            normalized = {}
            for key, value in state_dict.items():
                if key.startswith("backbone."):
                    suffix = key[len("backbone.") :]
                    normalized[f"policy_backbone.{suffix}"] = value
                    normalized[f"value_backbone.{suffix}"] = value.clone()
                else:
                    normalized[key] = value
            return normalized

        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        expanded = self._normalize_backbone_state_dict(state_dict)
        if strict:
            model_state = super().state_dict()
            model_keys = set(model_state)
            extra_keys = sorted(set(expanded) - model_keys)
            if extra_keys:
                expanded = {k: v for k, v in expanded.items() if k in model_keys}
                print(f"Dropped {len(extra_keys)} stale checkpoint keys")
            # Checkpoints created before the learned pooling/scaling heads do
            # not contain these parameters.  Their defaults are deliberately
            # chosen to recover the old normalized-sum behavior as closely as
            # possible, so they can be loaded without invalidating old runs.
            optional_head_keys = tuple(
                key
                for key in model_keys
                if key.startswith(
                    (
                        "to_pred.pool.score.",
                        "to_pred.action_bias.",
                        "to_pred.raw_logit_scale",
                        "rewards.pool.score.",
                    )
                )
            )
            for key in optional_head_keys:
                expanded.setdefault(key, model_state[key])
        return super().load_state_dict(expanded, strict=strict)

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
        policy_logits = self.to_pred(enc, attn_mask)
        value = self.rewards(value_enc, attn_mask)

        moves_pos = [[i for i, c in enumerate(game) if c == "@"] for game in games]

        pred = []
        for i in range(len(games)):
            if len(moves_pos[i]):
                logits = policy_logits[i, torch.tensor(moves_pos[i])]
                pred.append(logits)
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
    config = ckpt["config"]["net"]
    model = Model(
        config["dim"],
        config["num_layers"],
        config.get("head_size"),
        config.get("num_heads"),
        backbone=config.get("backbone", "transformer"),
        shared_backbone=config.get("shared_backbone", True),
    )
    ckpt_state = model._normalize_backbone_state_dict(ckpt["model"])
    model_state = model.state_dict()
    extra_keys = sorted(set(ckpt_state) - set(model_state))
    if extra_keys:
        ckpt_state = {k: v for k, v in ckpt_state.items() if k in model_state}
        print(f"Dropped {len(extra_keys)} stale checkpoint keys from {model_path}")
    print(model.load_state_dict(ckpt_state))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model
