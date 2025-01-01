from collections import defaultdict
import random
import torch
import numpy as np
import time
from visdom import Visdom
from tqdm import tqdm

from centuryrl.rl.model import Model
from centuryrl.rl.model.loss import loss_from_string
from centuryrl.century.strategies import strategy_from_string
from centuryrl.rl.eval.selfplay import self_play, pit
import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})


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
            elif isinstance(v[0], torch.Tensor):
                self.__dict__[k] = [x.to(*args, **kwargs) for x in v]
        return self


def collate(xs):
    if isinstance(xs[0], (int, float)):
        return torch.tensor(xs)
    if isinstance(xs[0], torch.Tensor):
        try:
            return torch.stack(xs, dim=0)
        except RuntimeError:
            return xs
    return xs


def discount(rews):
    d = 0.98
    return sum(d**i * r for i, r in enumerate(rews))


def to_trainset(games_data):
    out = []
    for hist in games_data.data:
        end = hist[-1]
        rewards = [0] * (len(hist) - 1)
        for i in range(len(hist) - 1):
            rewards[i] = hist[i + 1].current_diff_points - hist[i].current_diff_points

        for i, log in enumerate(hist[:-1]):
            out.append(
                TrainingSample(
                    round=float(i),
                    state=log.state,
                    moves=log.moves,
                    action_idx=log.action_idx,
                    action_distribution=log.action_distribution,
                    score=float(end.current_diff_points),
                    returns=discount(rewards[i:]),
                    current_diff_points=float(log.current_diff_points),
                )
            )
    return out


class GamesData:
    def __init__(self, data):
        self.data = data

    def avg_len(self):
        return sum(len(history) for history in self.data) / len(self.data)

    def avg_points(self):
        return sum(history[-1].my_points for history in self.data) / (len(self.data))

    def stats_cause(self):
        proper = sum(1 for history in self.data if history[-1].cause == "proper")
        toolong = sum(1 for history in self.data if history[-1].cause == "toolong")
        n = len(self.data)
        return {"proper": proper / n, "toolong": toolong / n}

    def avg_move_summary(self):
        movs = {}
        for typ in "HRVA":
            v = sum(
                1
                for history in self.data
                for h in history[:-1]
                if h.moves[h.action_idx][0] == typ
            )
            movs[typ] = v / len(self.data)
        return movs

    def prompt_size(self):
        avg = sum(
            sum(len(h.state) for h in history[:-1]) / len(history[:-1])
            for history in self.data
        ) / len(self.data)
        min_length = min(len(h.state) for history in self.data for h in history[:-1])
        max_length = max(len(h.state) for history in self.data for h in history[:-1])
        return {"avg": avg, "min": min_length, "max": max_length}

    def buy_rank(self):
        rank = []
        for history in self.data:
            for h in history[:-1]:
                if h.moves[h.action_idx][0] == "V":
                    victory_cards = [
                        l for l in h.state.split("\n") if "->" in l and l[0] == "V"
                    ]
                    victories_points = [int(v.split("->")[1]) for v in victory_cards]
                    buy_idx = int(h.moves[h.action_idx].split(" ")[0][1:])
                    this_points = victories_points[buy_idx]
                    victories_points.sort(key=lambda x: -x)
                    this_rank = victories_points.index(this_points)
                    rank.append(this_rank)
        print(rank)
        return sum(rank) / len(rank)

    def metrics(self):
        return {
            "prompt_size": self.prompt_size(),
            "avg_len": self.avg_len(),
            "avg_points": self.avg_points(),
            "causes": self.stats_cause(),
            "avg_move_summary": self.avg_move_summary(),
            "buy_rank": self.buy_rank(),
        }

    def metrics_to_visdom(self, viz, epoch):
        metrics = self.metrics()
        avg_move_summary = metrics.pop("avg_move_summary")
        viz.line(
            Y=torch.tensor(
                [[sum(avg_move_summary[k] for k in "AHRV"[: i + 1]) for i in range(4)]]
            ),
            X=torch.tensor([[epoch] * 4]),
            opts=dict(
                title="avg_move_summary",
                fillarea=True,
                legend=["A", "H", "R", "V"],
                xlabel="epoch",
                ylabel="freq",
            ),
            update="append",
            win="avg_move_summary",
        )

        lenghts = metrics.pop("prompt_size")
        viz.line(
            Y=torch.tensor([[lenghts[k] for k in ["avg", "min", "max"]]]),
            X=torch.tensor([[epoch] * 3]),
            opts=dict(
                title="prompt_size",
                legend=["avg", "min", "max"],
                xlabel="epoch",
                ylabel="length",
            ),
            update="append",
            win="prompt_size",
        )
        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    viz.line(
                        torch.tensor([vv]),
                        torch.tensor([epoch]),
                        win=k + "." + kk,
                        update="append",
                        opts={"title": k + "." + kk},
                    )
            else:
                viz.line(
                    torch.tensor([v]),
                    torch.tensor([epoch]),
                    win=k,
                    update="append",
                    opts={"title": k},
                )

    def print_short_history(self):
        import crayons

        colorized = {
            "A": str(crayons.red("A")),
            "H": str(crayons.green("H")),
            "R": str(crayons.yellow("R")),
            "V": str(crayons.white("V")),
        }
        for h in self.data:
            print(
                "".join(colorized[s.moves[s.action_idx][0]] for s in h[:-1]),
                h[-1].my_points,
            )

    def dump(self):
        with open("game.txt", "w") as f:
            for i, history in enumerate(self.data):
                print(f"== GAME {i} ==", file=f)
                for log in history[:-1]:
                    print(log.state, file=f)
                    print(">", log.moves[log.action_idx], ",".join(log.notes), file=f)
                    print(file=f)
                log = history[-1]
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


import copy


class Visualizer:
    def __init__(self, tag):
        self.viz = Visdom(
            env=tag,
            server="https://visdom.vermeille.fr",
            port=443,
        )
        self.viz.close()

    def push(self, name, value, epoch):
        self.viz.line(
            torch.tensor([value]),
            torch.tensor([epoch]),
            win=name,
            update="append",
            opts=dict(title=name),
        )


class Trainer:
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.model = Model(**self.config.net)
        self.model.to(config.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=config.train.lr)
        self.policy_loss = loss_from_string(config.train.loss.policy)
        self.value_loss = loss_from_string(config.train.loss.value)
        self.viz = Visualizer(f"{config.tag}-lr={config.train.lr}")
        self.epoch = 0

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path)
            self.model.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["opt"])

    def _log_pit(self):
        print("PIT: ", " VS ".join(self.config.pit.strategies))
        pit_results = pit(
            [
                strategy_from_string(s, model=self.model)
                for s in self.config.pit.strategies
            ],
            self.config.pit.num_games,
            self.config.pit.max_len,
        )
        GamesData(pit_results.games).print_short_history()
        self.viz.push("pit.win_rate", pit_results.win_rate(0), self.epoch)
        self.viz.push("pit.avg_points", pit_results.my_avg_points(0), self.epoch)

    def _train_epoch(self, data):
        self.model.train()
        now = time.time()
        grad_pct = 1 / self.config.train.gradient_epochs
        for grad_ep in range(self.config.train.gradient_epochs):
            indices = torch.randperm(len(data))
            batch_pct = 1 / (len(indices) // self.config.train.batch_size)
            for b_i, batch in enumerate(
                tqdm(
                    chunk(indices, self.config.train.batch_size),
                    desc=f"epoch {self.epoch}",
                )
            ):
                with torch.no_grad():
                    samples = TrainingSample.collate(
                        [copy.deepcopy(data[bi]) for bi in batch]
                    ).to(self.config.device)
                self.opt.zero_grad()
                total_losses = defaultdict(float)
                policy, value = self.model(samples.state, samples)
                policy_loss = self.policy_loss(policy, value, samples)
                value_loss = self.value_loss(policy, value, samples)
                loss = policy_loss + value_loss
                loss.backward()
                total_losses["policy"] += policy_loss.item() / len(data) * len(batch)
                total_losses["value"] += value_loss.item() / len(data) * len(batch)

                #for p in self.model.parameters(): p.grad.data *= len(batch) / len(data)

                grad_mag = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )
                self.opt.step()

                print(total_losses)
                for k, v in total_losses.items():
                    self.viz.push(f"loss.{k}", v, self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct)
                self.viz.push("grad_mag", grad_mag.item(), self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct)
        print(
            "throughput",
            len(data) * self.config.train.gradient_epochs / (time.time() - now),
        )
        print()

    def _save_model(self):
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "epoch": self.epoch,
                "config": self.config,
            },
            f"rl-{self.epoch}.pth",
        )

    def _run_episode(self):
        print("SELF PLAY: ", " VS ".join(self.config.self_play.strategies))
        data = self_play(
            [
                strategy_from_string(s, model=self.model)
                for s in self.config.self_play.strategies
            ],
            self.config.self_play.num_games,
            self.config.self_play.max_len,
        )
        data = GamesData(flatten(data))
        data.dump()
        data.print_short_history()
        data.metrics_to_visdom(self.viz.viz, self.epoch)
        return data

    def train(self):
        print("#parameters", sum(p.numel() for p in self.model.parameters()) / 1e6, "M")
        for epoch in range(3000):
            self.epoch = epoch

            print("EPOCH", epoch)
            if epoch % self.config.pit.every == 0:
                self._log_pit()

            if epoch % self.config.train.save_every == 0:
                self._save_model()

            data = self._run_episode()
            trainset = to_trainset(data)

            print(len(trainset), "samples")
            self._train_epoch(trainset)


def main():
    import sys
    import yaml
    from easydict import EasyDict

    with open(sys.argv[1]) as f:
        config = EasyDict(yaml.safe_load(f))
    Trainer(config, sys.argv[2] if len(sys.argv) > 2 else None).train()


if __name__ == "__main__":
    main()
