import torch.multiprocessing as mp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, x):
        x = x.transpose(2, 1)
        for m in self.layers:
            x = m[0](x).add_(x)
            x = m[1](x).add_(x)
        x = x.transpose(2, 1)
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
        return (x.squeeze(0) + x.mean(dim=0))


class Pool(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)

class First(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxlen = 768
        dim = 512
        self.in_embed = nn.Embedding(128, dim)
        self.in_embed.weight.data.normal_(0, 1/math.sqrt(dim))
        self.pos_enc_out = nn.Parameter(
            torch.randn(self.maxlen, dim) * 0 / math.sqrt(dim)
        )
        self.pos_enc = nn.Parameter(torch.randn(self.maxlen, dim) * 0 / math.sqrt(dim))
        self.encode = nn.Sequential(
            nn.LayerNorm(dim),
            #AlternativeEncoder(4, dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    dim, dim // 32, dim * 4, norm_first=True, batch_first=True
                ),
                num_layers=4,
                #norm=nn.LayerNorm(dim),
            ),
            # nn.ReLU(True),
            nn.LayerNorm(dim),
        )
        self.to_pred = nn.Sequential(
            First(),
            nn.LayerNorm(dim),
            #AttentionPool1d(dim, dim // 32, dim),
            nn.ReLU(True), nn.Linear(dim, 128)
        )

        self.rewards = nn.Sequential(  # AttentionPool1d(dim, dim // 32, dim),
            nn.LayerNorm(dim), First(),
            nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, 1)
        )
        self.normalizer = RunningNormalizer()

    def text_encode(self, txts, maxlen, pad=False):

        def do_pad(l):
            if pad:
                return l + [0] * (maxlen - len(l))
            else:
                return l

        txts = [
            torch.LongTensor(do_pad([ord(c) for c in txt][:maxlen])) for txt in txts
        ]
        return nn.utils.rnn.pad_sequence(txts, batch_first=True).to(
            self.in_embed.weight.device
        )

    def text_embed(self, txts, maxlen, pos, pad=False):
        txts = self.text_encode(txts, maxlen, pad=pad)
        txts = self.in_embed(txts)
        return txts + pos[: min(txts.shape[1], maxlen)]

    def forward(self, games, outs=None, win=None):
        enc = self.encode(self.text_embed(games, self.maxlen, self.pos_enc, pad=True))
        #enc = enc + self.pos_enc_out[: enc.shape[1]]
        pred = self.to_pred(enc)
        v = self.rewards(enc).squeeze(1)

        if outs is not None:
            out = torch.tensor(outs, device=pred.device)
            loss = nn.functional.cross_entropy(pred, out, reduction="none")
            win = torch.tensor(win, device=loss.device, dtype=torch.float)
            policy_loss = (self.normalizer(win) * loss).mean()

            # v_loss = F.mse_loss(v, win)
            print(v)
            print(win)
            v_loss = F.mse_loss(v, win)
            losses = {"policy": policy_loss.item(), "value": v_loss.item()}
            loss = policy_loss + v_loss
            return loss, losses
        else:
            assert not self.training
            return pred, torch.sigmoid(v)

class RunningNormalizer:
    def __init__(self):
        self.val = 0.0

    def __call__(self, x):
        self.val = 0.95 * self.val + 0.05 * x.std().item()
        return x / self.val

class GamesData:

    def __init__(self, data):
        self.data = data

    def to_trainset(self):
        out = []
        for d in self.data:
            game_len = len(d['history']) // 2
            for i, log in enumerate(d["history"][:-2]):
                out.append(
                    [log["state"], log["moves"].index(log["action"]), 0.95**(game_len - i //2) * log["winner"]]
                )
        return out

    def avg_len(self):
        return sum(len(d["history"]) for d in self.data) / len(self.data)

    def avg_points(self):
        return sum(max(d["p0"], d["p1"]) for d in self.data) / (len(self.data))

    def stats_cause(self):
        illeg = sum(1 for d in self.data if d["cause"] == "illegal")
        proper = sum(1 for d in self.data if d["cause"] == "proper")
        toolong = sum(1 for d in self.data if d["cause"] == "toolong")
        n = len(self.data)
        return {"illegal": illeg / n, "proper": proper / n, "toolong": toolong / n}

    def avg_move_summary(self):
        movs = {}
        for typ in "HRVA":
            v = sum(
                1
                for d in self.data
                for h in d["history"]
                if len(h.get("action", "")) > 0 and h["action"][0] == typ
            )
            movs[typ] = v / len(self.data)
        return movs

    def prompt_size(self):
        v = sum(
            sum(len(h["state"]) for h in d["history"]) / len(d["history"])
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

    def dump(self):
        with open("game.txt", "w") as f:
            for i, d in enumerate(self.data):
                print(f"== GAME {i} ==", file=f)
                for log in d["history"]:
                    print(log["state"], file=f)
                    print(">", log.get("action", ""), ",".join(log["notes"]), file=f)
                    print(file=f)


def chunk(data, size):
    i = 0
    while i < len(data):
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


def self_play(model, n_games, max_len, device):
    model.eval()
    print(device)
    if device is not None:
        model.to(device)
    data = [{"history": []} for _ in range(n_games)]
    running = [True for _ in range(n_games)]
    games = [Game() for _ in range(n_games)]

    strategy = PolicyGuidedMCMCStrategy(budget=5)

    for i_mov in tqdm(range(max_len), desc="playing moves"):
        if not any(running):
            break

        for i in range(n_games):
            if not running[i]:
                continue

            g = games[i]
            log = {"state": g.display_with_moves(), "notes": []}
            with torch.no_grad():
                mov, debug = strategy(g, model)
            log["notes"] += []  # [str(x) for x in debug]
            log["moves"] = g.moves

            log["action"] = mov
            data[i]["history"].append(log)
            try:
                g.play_str(mov)
            except Illegal:
                running[i] = False
                log["notes"].append("illegal")
                data[i]["winner"] = 1 - g.state
                data[i]["cause"] = "illegal"
                data[i]["p0"] = g.p0.points()
                data[i]["p1"] = g.p1.points()

            if g.ended():
                data[i]["history"].append({"state": g.display(), "notes": []})
                data[i]["history"].append(
                    {"state": g.display(force=1 - g.state), "notes": []}
                )
                running[i] = False
                data[i]["winner"] = 0 if g.p0.points() > g.p1.points() else 1
                data[i]["cause"] = "proper"
                data[i]["p0"] = g.p0.points()
                data[i]["p1"] = g.p1.points()

    for i in range(n_games):
        if running[i]:
            data[i]["history"].append(
                {"state": games[i].display(), "notes": [], "action": ""}
            )
            data[i]["history"].append(
                {
                    "state": games[i].display(force=1 - g.state),
                    "notes": [],
                    "action": "",
                }
            )
            data[i]["winner"] = 0 if games[i].p0.points() > games[i].p1.points() else 1
            data[i]["cause"] = "toolong"
            data[i]["p0"] = games[i].p0.points()
            data[i]["p1"] = games[i].p1.points()

        hist = data[i]["history"]
        points = abs(data[i]["p0"] - data[i]["p1"])
        for j in range(len(hist)):
            if j % 2 == data[i]["winner"]:
                hist[j]["winner"] = points
            else:
                hist[j]["winner"] = -points
            hist[j]["notes"] += ["reward: " + str(hist[j]["winner"])]

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
    import time

    device = "cpu"
    m = Model()
    # m = torch.compile(m)
    if len(sys.argv) >= 3:
        m.load_state_dict(torch.load(sys.argv[2]))

    m.to(device)
    num_games = 32
    max_len = 200
    EPOCHS = 1

    opt = torch.optim.AdamW(m.parameters(), lr=float(sys.argv[1]))

    print("#parameters", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    viz = Visdom(env="century-rl-3")
    viz.close()
    # self play
    mp.set_start_method("spawn")
    ii = 0
    for epoch in range(3000):
        print("EPOCH", epoch)
        data = [self_play(m, num_games, max_len, device)]
        data = GamesData(flatten([d.data for d in data]))
        data.dump()
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
        if len(trainset) > 0:
            print(Counter(list(zip(*trainset))[1]))
        m.train()

        for e in range(EPOCHS):
            random.shuffle(trainset)
            for batch in chunk(trainset, 4):
                X, Y, W = zip(*batch)
                opt.zero_grad()
                loss, losses = m(X, Y, W)
                loss.backward()
                opt.step()
                ii += 1
                if ii % 10 == 0:
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
                        opts=dict(title='epoch')
                    )
            print()

        if epoch % 10 == 0:
            torch.save(m.state_dict(), f"rl-{epoch}.pth")
