import torch.multiprocessing as mp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer
from tqdm import tqdm

import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from engine import *


class Illegal(BaseException):
    pass


class AlternativeEncoder(nn.Module):
    def __init__(self, n_layers, dim):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.GroupNorm(1, dim),
                            nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim),
                        ),
                        nn.Sequential(
                            nn.GroupNorm(1, dim),
                            nn.Conv1d(dim, dim * 4, 1),
                            nn.ReLU(True),
                            nn.Conv1d(dim * 4, dim, 1),
                        ),
                    ]
                )
                for _ in range(n_layers)
            ]
        )

        for layer in self.layers:
            layer[0][-1].weight.data.zero_()
            layer[1][-1].weight.data.zero_()

    def forward(self, x):
        x = x.transpose(2, 1)  # BLC -> BCL
        for m in self.layers:
            x = m[0](x).add_(x)
            x = m[1](x).add_(x)
        x = x.transpose(2, 1)  # BCL -> BLC
        return x


class AttentionPool1d(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.query = nn.Parameter(torch.randn(embed_dim) / math.sqrt(embed_dim))
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # BLC -> LBC
        x, _ = F.multi_head_attention_forward(
            query=self.query.expand(1, x.shape[1], x.shape[2]),
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0) + x.mean(dim=0)


class Pool(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class First(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.seq(x)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Model(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.maxlen = 768
        self.in_embed = nn.Embedding(128, dim)
        self.in_embed.weight.data.normal_(0, 0.02)
        self.encode = nn.Sequential(
            nn.LayerNorm(dim),
            # AlternativeEncoder(4, dim),
            #nn.TransformerEncoder( nn.TransformerEncoderLayer( dim, dim // 64, dim * 4, norm_first=True, batch_first=True), num_layers=num_layers,),
            Transformer(dim, num_layers, dim // 64, 64),
        )
        self.to_pred = nn.Sequential(
            # AttentionPool1d(dim, dim // 32, dim),
            FFN(dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            Squeeze(-1),
            # BL
        )

        self.rewards = nn.Sequential(  # AttentionPool1d(dim, dim // 32, dim),
            First(),
            FFN(dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            Squeeze(-1),
            # B
        )
        self.pretrain_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128))
        self.normalizer = RunningNormalizer()
        self.loss = PolicyGradientWithBaselineLoss()

    def text_encode(self, txts, maxlen, pad=False):
        def do_pad(l):
            if pad:
                return l + [0] * (maxlen - len(l))
            else:
                return l

        txts = [
            torch.LongTensor(do_pad([ord(c) for c in txt])) for txt in txts
        ]
        return nn.utils.rnn.pad_sequence(txts, batch_first=True).to(
            self.in_embed.weight.device
        )

    def text_embed(self, txts, maxlen, pad=False):
        txts = self.text_encode(txts, maxlen, pad=pad)
        return txts

    def forward(self, games, samples=None):
        games = [game[:self.maxlen] for game in games]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            txt = self.text_embed(games, self.maxlen, pad=True)
            enc = self.encode(self.in_embed(txt))
            pred = self.to_pred(enc).float()
            v_norm = self.rewards(enc).float()

        moves_pos = [[i for i, c in enumerate(game) if game[i] == '@'] for game in games]

        if samples is not None:
            pretrain_loss = F.cross_entropy(self.pretrain_head(enc[:, :-1, :].float()).transpose(1, 2), txt[:, 1:])
            samples.action = torch.tensor([moves_pos[i][a] for i, a in enumerate(samples.action)]).to(device=pred.device)
            pred = self.mask_logits(pred, moves_pos)
            returns_norm = self.normalizer(samples.returns)

            policy_loss = self.loss(pred, samples.action, pred_value=v_norm.detach(), returns=returns_norm)
            print(self.normalizer.undo(v_norm))
            v_loss = F.mse_loss(returns_norm, v_norm)
            losses = {"policy": policy_loss.item(), "value": v_loss.item(), "pretrain": pretrain_loss.item()}
            loss = policy_loss + v_loss + 1 * pretrain_loss
            return loss, losses
        else:
            assert not self.training
            pred = [
                pred[i][torch.tensor(moves_pos[i])] for i in range(len(games))
            ]
            return pred, v_norm  # undo normalization?

    @staticmethod
    def mask_logits(logits, moves_pos):
        mask = torch.full_like(logits, False, dtype=torch.bool)
        indexes = [
            (b, n)
            for b in range(len(moves_pos))
            for n in moves_pos[b]
        ]
        xs, ys = zip(*indexes)
        mask[torch.tensor(xs), torch.tensor(ys)] = True
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class PolicyGradientLoss:
    def __init__(self):
        self.normalizer = RunningNormalizer()

    def __call__(self, logits, pred_value, sample):
        loss = F.cross_entropy(logits, sample.action, reduction="none")
        policy_loss = (self.normalizer(sample.returns) * loss).mean()
        return policy_loss


class PolicyGradientWithBaselineLoss:
    def __call__(self, logits, action, **kwargs):
        pred_value, returns = kwargs.pop('pred_value'), kwargs.pop('returns')
        loss = F.cross_entropy(logits, action, reduction="none")
        policy_loss = ((returns - pred_value) * loss).mean()
        return policy_loss


class RunningNormalizer:
    def __init__(self):
        self.std = None
        self.mean = None

    def __call__(self, x):
        if self.std is None:
            self.std = x.std().item()
            self.mean = x.mean().item()
        else:
            self.std = 0.99 * self.std + 0.01 * x.std().item()
            self.mean = 0.99 * self.mean + 0.01 * x.mean().item()
        return (x - self.mean) / self.std

    def undo(self, x):
        return x * self.std + self.mean


class TrainingSample:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def collate(samples):
        return TrainingSample(
            **{
                k: collate([getattr(s, k) for s in samples]) for k in samples[0].__dict__
            }
        )

    def to(self, *args, **kwargs):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(*args, **kwargs)
        return self

def collate(xs):
    if isinstance(xs[0], (int, float)):
        return torch.tensor(xs)
    if isinstance(xs[0], torch.Tensor):
        return torch.cat(xs)
    return xs


class GamesData:
    def __init__(self, data):
        self.data = data

    def to_trainset(self):
        out = []
        for d in self.data:
            hist = d["history"]
            rewards = [0] * (len(hist) - 1)
            for i in range(len(hist) - 1):
                rewards[i] = float(
                    hist[i + 1].current_diff_points - hist[i].current_diff_points
                )

            for i, log in enumerate(hist[:-1]):
                out.append(
                    TrainingSample(
                        state=log.state,
                        moves=log.moves,
                        action=log.action_idx,
                        score=d["history"][-1].current_diff_points,
                        returns=sum(rewards[i:]),
                        current_diff_points=log.current_diff_points,
                    )
                )
        return out

    def avg_len(self):
        return sum(len(d["history"]) for d in self.data) / len(self.data)

    def avg_points(self):
        return sum(d["history"][-1].my_points for d in self.data) / (len(self.data))

    def stats_cause(self):
        proper = sum(1 for d in self.data if d["history"][-1].cause == "proper")
        toolong = sum(1 for d in self.data if d["history"][-1].cause == "toolong")
        n = len(self.data)
        return {"proper": proper / n, "toolong": toolong / n}

    def avg_move_summary(self):
        movs = {}
        for typ in "HRVA":
            v = sum(
                1
                for d in self.data
                for h in d["history"][:-1]
                if h.moves[h.action_idx][0] == typ
            )
            movs[typ] = v / len(self.data)
        return movs

    def prompt_size(self):
        v = sum(
            sum(len(h.state) for h in d["history"][:-1]) / len(d["history"][:-1])
            for d in self.data
        ) / len(self.data)
        return v

    def metrics(self):
        return {
            "prompt_size": int(self.prompt_size()),
            "avg_len": self.avg_len(),
            "avg_points": self.avg_points(),
            "causes": self.stats_cause(),
            "avg_move_summary": self.avg_move_summary(),
        }

    def print_short_history(self):
        for g in self.data:
            h = g['history'][:-1]
            print(''.join(s.moves[s.action_idx][0] for s in h))

    def dump(self):
        with open("game.txt", "w") as f:
            for i, d in enumerate(self.data):
                print(f"== GAME {i} ==", file=f)
                for log in d["history"][:-1]:
                    print(log.state, file=f)
                    print(">", log.moves[log.action_idx], ",".join(log.notes), file=f)
                    print(file=f)
                log = d["history"][-1]
                print(log.state, file=f)
                print("END", ",".join(log.notes), file=f)
                print(file=f)


def chunk(data, size):
    i = 0
    while i + size < len(data):
        yield data[i : i + size]
        i += size


def flatten(list_of_lists):
    out = []
    for l in list_of_lists:
        out += l
    return out


def autobatch(model, input, bs=None):
    if bs is None:
        bs = len(input)
    assert bs > 0

    try:
        with torch.no_grad():
            return flatten([model(x) for x in chunk(input, bs)])
    except Exception as e:
        print(e)
        return autobatch(model, input, bs // 2)


@torch.no_grad()
def pit(model1, model2, n_games, max_len, device):
    model1.eval()
    model2.eval()
    if device is not None:
        model1.to(device)
        model2.to(device)
    strategy1 = ArgmaxStrategy()
    strategy2 = RandomBuyStrategy()

    won = 0
    for g_i in tqdm(range(n_games), desc="pit"):
        g = Game()
        for i_mov in range(max_len):
            if g.ended():
                break

            if i_mov % 2 == 0:
                mov, _ = strategy1(g, model1)
            else:
                mov, _ = strategy2(g, model2)
            g.play_str(mov)
        won += g.diff_points_for(0)
    return won / n_games


class Record:
    def __init__(self, game: Game, action: str):
        self.state = game.display_with_moves()
        self.moves = game.moves[:]
        self.action_idx = self.moves.index(action)
        self.current_diff_points = game.diff_points()
        self.my_points = game.points()
        self.notes = []


class EndState:
    def __init__(self, game: Game, player: int):
        self.cause = "proper" if game.ended() else "toolong"
        self.state = game.display(force=player)
        self.my_points = game.points_for(player)
        self.current_diff_points = game.diff_points_for(player)
        self.notes = []


@torch.no_grad()
def self_play(strategies, n_games, max_len):
    data = [{"history": []} for _ in range(n_games * 2)]

    for i in tqdm(range(n_games), desc="playing games"):
        g = Game()

        for i_mov in range(max_len):
            if g.ended():
                break

            mov, debug = strategies[g.current_player()](g)

            rec = Record(g, mov)
            rec.notes += [str(x) for x in debug]
            data[i * 2 + g.current_player()]["history"].append(rec)

            g.play_str(mov)

        data[i * 2]["history"].append(EndState(g, 0))
        data[i * 2 + 1]["history"].append(EndState(g, 1))

    return GamesData(data)


import math


from visdom import Visdom


def load(model, file):
    try:
        ckpt = torch.load(file, map_location="cpu")
        d = model.state_dict()
        for k in d.keys():
            if k in d and k in ckpt:
                try:
                    d[k].copy_(ckpt[k])
                except Exception as e:
                    print(e)
        print("loaded")
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    from collections import Counter
    import sys
    import yaml
    from easydict import EasyDict

    config = EasyDict(yaml.safe_load(open(sys.argv[1])))

    m = Model(config.net.dim, config.net.num_layers)
    # m = torch.compile(m)
    if len(sys.argv) >= 3:
        m.load_state_dict(torch.load(sys.argv[2]))

    m.to(config.device)

    opt = torch.optim.AdamW(m.parameters(), lr=config.train.lr, weight_decay=1e-4)

    print("#parameters", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    viz = Visdom(env=f"{config.tag}-lr={config.train.lr}")
    viz.close()
    # self play
    ii = 0
    for epoch in range(3000):
        print("EPOCH", epoch)
        data = [
            self_play(
                [PolicySamplingStrategy(m), PolicySamplingStrategy(m)],
                config.self_play.num_games,
                config.self_play.max_len,
            )
        ]
        data = GamesData(flatten([d.data for d in data]))
        data.dump()
        data.print_short_history()
        metrics = data.metrics()

        trainset = data.to_trainset()

        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    viz.line(
                        torch.tensor([vv]),
                        torch.tensor([ii]),
                        win=k + "." + kk,
                        update="append",
                        opts={"title": k + "." + kk},
                    )
            else:
                viz.line(
                    torch.tensor([v]),
                    torch.tensor([ii]),
                    win=k,
                    update="append",
                    opts={"title": k},
                )
        print(metrics)
        print(len(trainset), "samples")
        m.train()

        previous_model = copy.deepcopy(m)
        for e in range(config.train.gradient_epochs):
            random.shuffle(trainset)
            for batch in chunk(trainset, config.train.batch_size):
                samples = TrainingSample.collate(batch).to(config.device)
                opt.zero_grad()
                loss, losses = m(samples.state, samples)
                loss.backward()
                opt.step()
                ii += 1
                if ii % config.train.show_every == 0:
                    print("lr", opt.param_groups[0]["lr"])
                    for k, v in losses.items():
                        viz.line(
                            torch.tensor([v]),
                            torch.tensor([ii]),
                            win="loss" + k,
                            update="append",
                            opts={"title": "loss." + k},
                        )
                    viz.line(
                        torch.tensor([epoch]),
                        torch.tensor([ii]),
                        win="epoch",
                        update="append",
                        opts=dict(title="epoch"),
                    )
            print()
        if epoch % config.pit.every == 0:
            win_rate = pit(
                m,
                previous_model,
                config.pit.num_games,
                config.pit.max_len,
                config.device,
            )
            viz.line(
                torch.tensor([win_rate]),
                torch.tensor([epoch]),
                win="win_rate",
                update="append",
                opts=dict(title="win_rate"),
            )
        del previous_model

        if epoch % 10 == 0:
            torch.save(m.state_dict(), f"rl-{epoch}.pth")
