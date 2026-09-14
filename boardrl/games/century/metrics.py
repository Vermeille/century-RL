import crayons  # type: ignore[import-untyped]
from boardrl.metrics import GameMetrics


class Metrics(GameMetrics):
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
                            line
                            for line in h.state.split("\n")
                            if ">" in line and line[0] == "V"
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
            "sensitivity": self.data.sensitivity(),
        }

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
