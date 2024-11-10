import torch.multiprocessing as mp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer, TransformerBlock
from tqdm import tqdm

import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from engine import Game, RandomBuyStrategy, PolicySamplingStrategy


class Illegal(BaseException):
    pass


class Pool(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class First(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


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
            # nn.LayerNorm(dim),
            Transformer(dim, num_layers, dim // 64, 64),
        )
        self.to_pred = nn.Sequential(
            # AttentionPool1d(dim, dim // 32, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            Squeeze(-1),
            # BL
        )

        self.rewards = nn.Sequential(  # AttentionPool1d(dim, dim // 32, dim),
            Pool(),
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

    def forward(self, games, samples=None):
        games = [game[: self.maxlen] for game in games]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            txt = self.text_embed(games, self.maxlen, pad=True)
            enc = self.encode(self.in_embed(txt))
            pred = self.to_pred(enc).float()
            v_norm = self.rewards(enc).float()

        moves_pos = [
            [i + 1 for i, c in enumerate(game[:-1]) if c == "@"] for game in games
        ]

        pred = [pred[i][torch.tensor(moves_pos[i])] for i in range(len(games))]

        if samples is not None:
            pretrain_loss = 1 * F.cross_entropy(
                self.pretrain_head(enc[:, :-1, :].float()).transpose(1, 2), txt[:, 1:]
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
            loss = policy_loss + v_loss + 1 * pretrain_loss
            return loss, losses
        else:
            return pred, v_norm  # undo normalization?


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


class TrainingSample:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def collate(samples):
        return TrainingSample(
            **{
                k: collate([getattr(s, k) for s in samples])
                for k in samples[0].__dict__
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
        def discount(rews):
            d = 0.98
            return sum(d**i * r for r in rews)

        out = []
        for d in self.data:
            end = d["history"][-1]
            hist = d["history"]
            rewards = [0] * (len(hist) - 1)
            for i in range(len(hist) - 1):
                rewards[i] = (
                    float(
                        hist[i + 1].current_diff_points
                        - hist[i].current_diff_points
                        - 1
                    )
                    / 30
                )
            # if hist[-1].cause == "toolong": rewards[-1] -= 4

            for i, log in enumerate(hist[:-1]):
                out.append(
                    TrainingSample(
                        state=log.state,
                        moves=log.moves,
                        action=log.action_idx,
                        score=d["history"][-1].current_diff_points,
                        returns=discount(rewards[i:]),
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
        import crayons

        colorized = {
            "A": str(crayons.red("A")),
            "H": str(crayons.green("H")),
            "R": str(crayons.yellow("R")),
            "V": str(crayons.white("V")),
        }
        for g in self.data:
            h = g["history"]
            print(
                "".join(colorized[s.moves[s.action_idx][0]] for s in h[:-1]),
                h[-1].my_points,
            )

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
def pit(strategies, n_games, max_len):
    n_players = len(strategies)
    dat = self_play(strategies, n_games, max_len)
    dat.print_short_history()
    return sum(
        int(d["history"][-1].current_diff_points >= 0) for d in dat.data[::n_players]
    ) / len(dat.data[::n_players])


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
    n_players = len(strategies)
    data = [{"history": []} for _ in range(n_games * n_players)]

    for i in tqdm(range(n_games), desc="playing games"):
        g = Game()

        for i_mov in range(max_len):
            if g.ended():
                break

            mov, debug = strategies[g.current_player()](g)

            rec = Record(g, mov)
            rec.notes += [str(x) for x in debug]
            data[i * n_players + g.current_player()]["history"].append(rec)

            g.play_str(mov)

        for p in range(n_players):
            data[i * n_players + p]["history"].append(EndState(g, p))

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


def warm_batchnorm(m):
    m.train()
    strategy = RandomBuyStrategy()

    for _ in range(50):
        prompts = []
        g = Game()
        for _ in range(32):
            if g.ended():
                break
            prompts.append(g.display_with_moves())
            strategy(g)
        m(prompts)


import random
import numpy as np


def smart_mix(old, new):
    random.shuffle(old)
    random.shuffle(new)
    out = []
    i_old, i_new = 0, 0
    for j in np.linspace(0, 1, len(old) + len(new)):
        if i_old == len(old):
            out += new[i_new:]
            break

        if i_new == len(new):
            out = old[i_old:] + out
            break

        if random.random() > j:
            out.append(old[i_old])
            i_old += 1
        else:
            out.append(new[i_new])
            i_new += 1
    return out


import time


def main():
    from collections import Counter
    import sys
    import yaml
    from easydict import EasyDict

    config = EasyDict(yaml.safe_load(open(sys.argv[1])))

    m = Model(config.net.dim, config.net.num_layers)
    # m = torch.compile(m)
    m.to(config.device)
    if len(sys.argv) >= 3:
        m.load_state_dict(torch.load(sys.argv[2]), map_location=config.device)
    else:
        warm_batchnorm(m)

    opt = torch.optim.AdamW(m.parameters(), lr=config.train.lr, weight_decay=1e-4)

    print("#parameters", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    viz = Visdom(env=f"{config.tag}-lr={config.train.lr}")
    viz.close()
    # self play
    ii = 0
    old_trainset = []
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

        new_trainset = data.to_trainset()

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
        print(len(new_trainset), "samples")
        m.train()

        # trainset = smart_mix( old_trainset * (config.train.gradient_epochs // 2), new_trainset * (config.train.gradient_epochs // 2),)
        trainset = new_trainset
        now = time.time()
        opt.zero_grad()
        for batch in chunk(trainset, config.train.batch_size):
            samples = TrainingSample.collate(batch).to(config.device)
            loss, losses = m(samples.state, samples)
            (loss / (len(trainset) // config.train.batch_size)).backward()
            ii += 1
            if ii % config.train.show_every == 0:
                print(losses)
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
        opt.step()
        print("throughput", len(trainset) / (time.time() - now))
        grad_mag = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=100.0)
        viz.line(
            torch.tensor([grad_mag.item()]),
            torch.tensor([ii]),
            win="grad_mag",
            update="append",
            opts=dict(title="grad_mag"),
        )
        old_trainset = new_trainset
        print()
        if epoch % config.pit.every == 0:
            win_rate = pit(
                [PolicySamplingStrategy(m), RandomBuyStrategy()],
                config.pit.num_games,
                config.pit.max_len,
            )
            viz.line(
                torch.tensor([win_rate]),
                torch.tensor([epoch]),
                win="win_rate",
                update="append",
                opts=dict(title="win_rate"),
            )

        if epoch % 10 == 0:
            torch.save(m.state_dict(), f"rl-{epoch}.pth")


if __name__ == "__main__":
    main()
