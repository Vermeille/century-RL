from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from centuryrl.rl.model.transformer import Transformer


def mask_pool(x, mask):
    mask = mask.unsqueeze(-1)
    return (x.float() * mask.float()).sum(1) / mask.float().sum(1)


class First(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


PolicyValue = namedtuple("PolicyValue", ["policy", "value"])


class SinusoidalPositional(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, embedding_dim, max_seq_length=5000):
        super().__init__()

        import math

        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)


class ScaledSinosoidal(SinusoidalPositional):
    """Sinusoidal with scaling (see FLASH paper)."""

    def __init__(self, embedding_dim, max_seq_length):
        super().__init__(embedding_dim, max_seq_length)
        self.scale_factor = torch.nn.Parameter(torch.tensor([1.0 / embedding_dim**0.5]))

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
        return self.scale_factor * self.pe[:, : input_ids.shape[1], :] + input_ids


class ValueHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tfblock = Transformer(dim, 1, dim // 64, 64)
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            Squeeze(-1),
            # B
        )

    def forward(self, x, attn_mask):
        x = self.tfblock(x, attn_mask)
        x = mask_pool(x, attn_mask)
        return self.out(x)


class PolicyHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tfblock = Transformer(dim, 1, dim // 64, 64)
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            Squeeze(-1),
            # BL
        )

    def forward(self, x, attn_mask):
        x = self.tfblock(x, attn_mask)
        return self.out(x)


class Model(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.maxlen = 2048
        self.in_embed = nn.Sequential(
            nn.Embedding(128, dim),
            ScaledSinosoidal(dim, self.maxlen),
            # nn.LayerNorm(dim),
        )
        self.in_embed[0].weight.data.normal_(0, 0.02)
        self.encode = Transformer(dim, num_layers - 1, dim // 64, 64)
        self.to_pred = PolicyHead(dim)
        self.rewards = ValueHead(dim)
        self.pretrain_head = nn.Sequential(
            nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, 128)
        )
        self.loss = ImitationLoss()  # PolicyGradientWithBaselineLoss()
        self.pretrain_weight = 0.001

    def text_encode(self, txts, maxlen, pad=False):
        def do_pad(l):
            if pad:
                return l + [1] + [0] * (maxlen - len(l) - 1)
            else:
                return l

        txts = [torch.LongTensor(do_pad([ord(c) for c in txt])) for txt in txts]
        return nn.utils.rnn.pad_sequence(txts, batch_first=True).to(
            self.in_embed[0].weight.device
        )

    def text_embed(self, txts, maxlen, pad=False):
        txts = self.text_encode(txts, maxlen, pad=pad)
        return txts

    def forward(self, games: list[str], samples=None):
        games = [game[: self.maxlen] for game in games]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            txt = self.text_embed(games, max(len(g) for g in games), pad=True)
            attn_mask = txt != 0
            enc = self.encode(self.in_embed(txt), attn_mask)
            pred = self.to_pred(enc, attn_mask).float()
            v_norm = self.rewards(enc, attn_mask).float()

        moves_pos = [[i for i, c in enumerate(game) if c == "@"] for game in games]

        pred = [pred[i][torch.tensor(moves_pos[i])] for i in range(len(games))]

        if samples is not None:
            pretrain_loss = F.cross_entropy(
                self.pretrain_head(enc[:, :-1, :].float()).transpose(1, 2),
                txt[:, 1:],
                ignore_index=0,
            )

            policy_loss = self.loss(
                pred,
                samples.action,
                pred_value=v_norm.detach(),
                returns=samples.returns,
            )
            v_loss = F.mse_loss(samples.returns, v_norm)
            losses = {
                "policy": policy_loss.item(),
                "value": v_loss.item(),
                "pretrain": pretrain_loss.item(),
            }

            loss = policy_loss + v_loss + self.pretrain_weight * pretrain_loss
            return loss, losses
        else:
            return PolicyValue(pred, v_norm)  # undo normalization?


class PolicyGradientLoss:
    def __call__(self, logits, pred_value, sample):
        loss = F.cross_entropy(logits, sample.action, reduction="none")
        policy_loss = (sample.returns * loss).mean()
        return policy_loss


class PolicyGradientWithBaselineLoss:
    def __call__(self, logits, action, **kwargs):
        pred_value, returns = kwargs.pop("pred_value"), kwargs.pop("returns")
        assert len(pred_value) == len(returns)
        assert len(logits) == len(returns)
        advantage = returns - pred_value

        loss = 0
        for adv, logit, act in zip(advantage, logits, action):
            loss += adv * F.cross_entropy(logit, act)
        return loss / len(returns)


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
