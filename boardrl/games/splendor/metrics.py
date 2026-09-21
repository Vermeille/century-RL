import re

from boardrl.games.semantics import PointScores
from boardrl.metrics import GameMetrics, GroupedTraceMetrics, TraceMetrics


_PLAYER_LINE = re.compile(
    r"^P(?P<player>\d+)\*? S(?P<score>\d+) D(?P<developments>\d+) .* H(?P<reserved>.*)$"
)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _chosen_moves(traces):
    for trace in traces:
        for record in trace[:-1]:
            if not hasattr(record, "moves") or not hasattr(record, "action_idx"):
                continue
            yield record.moves[record.action_idx]


def _terminal_player_state(trace):
    terminal = trace[-1]
    prefix = f"P{trace.seat_id}"
    for line in terminal.state.splitlines():
        if not line.startswith(prefix):
            continue
        match = _PLAYER_LINE.match(line)
        if match:
            reserved = match.group("reserved")
            return {
                "developments": int(match.group("developments")),
                "reserved": 0 if reserved == "-" else len(reserved.split()),
            }
    return {"developments": 0, "reserved": 0}


class SplendorTraceMetrics(TraceMetrics):
    def behavior_metrics(self):
        moves = list(_chosen_moves(self.traces))
        main = [move for move in moves if move.startswith(("T:", "R:", "B:"))]
        takes = [move for move in main if move.startswith("T:")]
        reserves = [move for move in main if move.startswith("R:")]
        buys = [move for move in main if move.startswith("B:")]
        market_buys = [move for move in buys if not move.startswith("B:H")]

        double_takes = [
            move
            for move in takes
            if len(move[2:]) == 2 and move[2] == move[3]
        ]
        blind_reserves = [move for move in reserves if move.endswith(".D")]
        reserved_buys = [move for move in buys if move.startswith("B:H")]
        gold_buys = [move for move in buys if "~" in move]
        gold_spent = sum(len(move.split("~", 1)[1]) for move in gold_buys)
        discards = [move for move in moves if move.startswith("D:")]

        tier_counts = {tier: 0 for tier in (1, 2, 3)}
        for move in market_buys:
            tier_counts[int(move[2])] += 1

        return {
            "main_action_mix": {
                "take": _rate(len(takes), len(main)),
                "reserve": _rate(len(reserves), len(main)),
                "buy": _rate(len(buys), len(main)),
            },
            "take_double_same_share": _rate(len(double_takes), len(takes)),
            "reserve_blind_share": _rate(len(blind_reserves), len(reserves)),
            "buy_from_reserve_share": _rate(len(reserved_buys), len(buys)),
            "buy_with_gold_share": _rate(len(gold_buys), len(buys)),
            "gold_per_buy": _rate(gold_spent, len(buys)),
            "discard_per_main_action": _rate(len(discards), len(main)),
            "market_buy_tier_share": {
                str(tier): _rate(tier_counts[tier], len(market_buys))
                for tier in (1, 2, 3)
            },
        }

    def endgame_metrics(self):
        terminal_states = [_terminal_player_state(trace) for trace in self.traces]
        developments = [state["developments"] for state in terminal_states]
        reserved = [state["reserved"] for state in terminal_states]
        main_actions = [
            sum(
                move.startswith(("T:", "R:", "B:"))
                for move in _chosen_moves([trace])
            )
            for trace in self.traces
        ]

        return {
            "avg_final_developments": sum(developments) / len(developments),
            "avg_final_reserved": sum(reserved) / len(reserved),
            "points_per_main_action": sum(
                float(trace[-1].my_points) / max(actions, 1)
                for trace, actions in zip(self.traces, main_actions)
            )
            / len(self.traces),
        }

    def metrics(self):
        return super().metrics() | {
            "behavior": self.behavior_metrics(),
            "endgame": self.endgame_metrics(),
        }


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            terminal = game.by_seat[0][-1]
            print(terminal.state, terminal.my_points)
            print()

    def metrics(self):
        terminal_games = sum(game.by_seat[0][-1].terminal for game in self.data)
        return {
            "terminal_rate": terminal_games / len(self.data),
            "avg_game_actions": sum(
                max(len(trace) - 1, 0)
                for game in self.data
                for trace in game.by_seat
            )
            / len(self.data),
            "strategy": GroupedTraceMetrics(
                self.data.by_strategy,
                SplendorTraceMetrics,
                scores=PointScores(),
            ).metrics(),
        }
