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


class Model(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.maxlen = 2048
        self.in_embed = nn.Embedding(128, dim)
        self.in_embed.weight.data.normal_(0, 0.02)
        self.encode = Transformer(dim, num_layers, dim // 64, 64)
        self.to_pred = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            Squeeze(-1),
            # BL
        )

        self.rewards = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            Squeeze(-1),
            # B
        )
        self.pretrain_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128))
        self.loss = PolicyGradientWithBaselineLoss()

    def text_encode(self, txts, maxlen, pad=False):
        def do_pad(l):
            if pad:
                return l + [0] * (maxlen - len(l))
            else:
                return l

        txts = [torch.LongTensor(do_pad([ord(c) for c in txt])) for txt in txts]
        return nn.utils.rnn.pad_sequence(txts, batch_first=True).to(
            self.in_embed.weight.device
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
            pred = self.to_pred(enc).float()
            v_norm = self.rewards(mask_pool(enc, attn_mask)).float()

        moves_pos = [
            [i + 1 for i, c in enumerate(game[:-1]) if c == "@"] for game in games
        ]

        pred = [pred[i][torch.tensor(moves_pos[i])] for i in range(len(games))]

        if samples is not None:
            pretrain_loss = F.cross_entropy(
                self.pretrain_head(enc[:, 1:, :].float()).transpose(1, 2), txt[:, :-1]
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
            loss = policy_loss + v_loss + 10 * pretrain_loss
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
