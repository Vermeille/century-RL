from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from centuryrl.rl.model.transformer import Transformer, Permute
from centuryrl.rl.model.utils import js_div, jeffreys_div


def mask_pool(x, mask):
    # mask: BL1
    # x * mask: BLD * BL1 = BLD => BD
    # mask.sum(1): B1
    mask = mask.unsqueeze(-1)
    return (x * mask.to(x.dtype)).sum(1) / mask.to(x.dtype).sum(1)


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
        return self.scale_factor * self.pe[: input_ids.shape[1], :] + input_ids


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
            ScaledSinosoidal(dim, self.maxlen),
            # nn.LayerNorm(dim),
        )
        self.in_embed[0].weight.data.normal_(0, 1 / dim**0.5)
        self.encode = Transformer(dim, num_layers - 1, dim // head_size, head_size)
        self.to_pred = PolicyHead(dim)
        self.rewards = ValueHead(dim)
        self.pretrain_head = nn.Sequential(
            nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, 128)
        )
        self.loss = ImitationLoss("kl")
        self.pretrain_weight = 0
        print(self)

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
        # games = [game.replace("\n", "") for game in games]
        # games = self.tokenize.encode_batch(games)
        txt = self.text_embed(games, self.maxlen, pad=True)
        attn_mask = txt != 0
        enc = self.encode(self.in_embed(txt), attn_mask)
        pred = self.to_pred(enc, attn_mask)
        v_norm = self.rewards(enc, attn_mask)

        moves_pos = [[i for i, c in enumerate(game) if c == "@"] for game in games]

        pred = [pred[i][torch.tensor(moves_pos[i])] for i in range(len(games))]

        if samples is not None:
            assert len(pred) == len(samples.action)
            assert len(games) == len(pred)
            pretrain_loss = F.cross_entropy(
                self.pretrain_head(enc[:, :-1, :]).transpose(1, 2),
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
            return PolicyValue(pred, v_norm)


class ImitationLoss:
    def __init__(self, function=None):
        """
        Imiation Loss: Loss function for imitation learning.
        Args:
            function (str): loss function to use. One of "cross_entropy", "kl", "reverse_kl", "jeffreys", "js"
        """
        self.function = function

    def __call__(self, logits, action, returns, **kwargs):
        """
        Calculate the loss for imitation learning.
        Args:
            logits (list[torch.Tensor]): List of logit tensors. Shape: (batch_size, num_classes).
            action (list[torch.Tensor]): List of action tensors. Shape: (batch_size) if discrete, (batch_size, num_classes) if continuous, logits.
            returns (torch.Tensor): Returns for each sample in the batch. Shape: (batch_size).
        """
        assert len(logits) == len(action)
        loss = 0
        for logit, act, r in zip(logits, action, returns.abs()):
            assert logit.shape == act.shape
            assert logit.ndim == 1
            # print(F.softmax(logit, dim=0), F.softmax(act, dim=0))
            # print(logit, act)
            logit = logit.unsqueeze(0)
            act = act.unsqueeze(0)
            print(F.softmax(logit, dim=1), F.softmax(act, dim=1))
            if self.function == "cross_entropy":
                loss += F.cross_entropy(logit, F.softmax(act, dim=1))
            elif self.function == "kl":
                loss += F.kl_div(
                    F.log_softmax(logit, dim=1),
                    F.log_softmax(act, dim=1),
                    reduction="batchmean",
                    log_target=True,
                )
            elif self.function == "reverse_kl":
                loss += F.kl_div(
                    F.log_softmax(act, dim=1),
                    F.log_softmax(logit, dim=1),
                    reduction="batchmean",
                    log_target=True,
                )
            elif self.function == "jeffreys":
                loss += jeffreys_div(logit, act)
            elif self.function == "js":
                loss += js_div(logit, act)
            elif self.function == "mse":
                loss += F.mse_loss(logit, act)
            else:
                raise ValueError(f"Unknown loss function: {self.function}")
        return loss / len(action)


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
