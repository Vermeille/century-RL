import crayons
import torch


class Metrics:
    def __init__(self, data):
        self.data = data

    def avg_len(self):
        return sum(len(player[0]) for player in self.data) / len(self.data)

    def avg_points(self):
        return [
            sum(players[player][-1].my_points for players in self.data)
            / (len(self.data))
            for player in range(len(self.data[0]))
        ]

    def stats_cause(self):
        proper = sum(
            1
            for players in self.data
            for history in players
            if history[-1].cause == "proper"
        )
        toolong = sum(
            1
            for players in self.data
            for history in players
            if history[-1].cause == "toolong"
        )
        n = sum(1 for players in self.data for history in players)
        return {"proper": proper / n, "toolong": toolong / n}

    def avg_move_summary(self):
        movs = {}
        for typ in "HRVA":
            v = sum(
                1
                for players in self.data
                for history in players
                for h in history[:-1]
                if h.moves[h.action_idx][0] == typ
            )
            movs[typ] = v / len(self.data)
        return movs

    def prompt_size(self):
        avg = sum(
            sum(len(h.state) for h in history[:-1]) / len(history[:-1])
            for players in self.data
            for history in players
        ) / sum(1 for players in self.data for history in players)
        min_length = min(
            len(h.state)
            for players in self.data
            for history in players
            for h in history[:-1]
        )
        max_length = max(
            len(h.state)
            for players in self.data
            for history in players
            for h in history[:-1]
        )
        return {"avg": avg, "min": min_length, "max": max_length}

    def buy_rank(self):
        rank = [[] for _ in range(len(self.data[0]))]
        for players in self.data:
            for player in range(len(players)):
                for h in players[player][:-1]:
                    if h.moves[h.action_idx][0] == "V":
                        victory_cards = [
                            l for l in h.state.split("\n") if ">" in l and l[0] == "V"
                        ]
                        victories_points = [int(v.split(">")[1]) for v in victory_cards]
                        buy_idx = int(h.moves[h.action_idx].split(" ")[0][1:])
                        this_points = victories_points[buy_idx]
                        victories_points.sort(key=lambda x: -x)
                        this_rank = victories_points.index(this_points)
                        rank[player].append(this_rank)
        mean_ranks = [sum(rankp) / len(rankp) for rankp in rank if rankp]
        return mean_ranks

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
        viz.visdom(
            "line",
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

        lengths = metrics.pop("prompt_size")
        viz.visdom(
            "line",
            Y=torch.tensor([[lengths[k] for k in ["avg", "min", "max"]]]),
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
                    viz.push(
                        f"{k}.{kk}",
                        vv,
                        epoch,
                    )
            else:
                viz.push(
                    k,
                    v,
                    epoch,
                )

    def print_short_history(self):
        colorized = {
            "A": str(crayons.red("A")),
            "H": str(crayons.green("H")),
            "R": str(crayons.yellow("R")),
            "V": str(crayons.white("V")),
        }
        for players in self.data:
            for h in players:
                print(
                    "".join(colorized[s.moves[s.action_idx][0]] for s in h[:-1]),
                    h[-1].my_points,
                )
            print("--")

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
