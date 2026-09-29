from boardrl.games.semantics import PointScores
from boardrl.games.take5.game import card_points
from boardrl.metrics import GameMetrics, GroupedTraceMetrics, TraceMetrics


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _chosen_move(record):
    return record.moves[record.action_idx]


def _moves(traces):
    for trace in traces:
        for record in trace[:-1]:
            if hasattr(record, "moves") and hasattr(record, "action_idx"):
                yield record


def _stack_cards(state, stack_index):
    prefix = f"S{stack_index}:"
    for line in state.splitlines():
        if line.startswith(prefix):
            values = line[len(prefix) :].split()
            return [int(value) for value in values]
    return []


def _card_bucket(card):
    if card > 100:
        return "101_plus"
    lower = ((card - 1) // 10) * 10 + 1
    return f"{lower}_{lower + 9}"


class Take5TraceMetrics(TraceMetrics):
    def behavior_metrics(self):
        records = list(_moves(self.traces))
        played_cards = []
        taken_stacks = []
        fitting_cards = []

        for record in records:
            move = _chosen_move(record)
            if move.startswith("S"):
                stack = _stack_cards(record.state, int(move[1:]))
                taken_stacks.append(stack)
            else:
                card = int(move)
                played_cards.append(card)
                stack_tops = []
                for line in record.state.splitlines():
                    if line.startswith("S") and ":" in line:
                        label, cards = line.split(":", 1)
                        if label[1:].isdigit() and cards.split():
                            stack_tops.append(int(cards.split()[-1]))
                fitting_cards.append(
                    any(top < card for top in stack_tops)
                )

        decisions = len(records)
        card_buckets = {
            f"{lower}_{lower + 9}": 0 for lower in range(1, 101, 10)
        }
        card_buckets["101_plus"] = 0
        for card in played_cards:
            card_buckets[_card_bucket(card)] += 1

        return {
            "action_mix": {
                "play_card": _rate(len(played_cards), decisions),
                "take_stack": _rate(len(taken_stacks), decisions),
            },
            "take_stacks_per_played_card": _rate(
                len(taken_stacks), len(played_cards)
            ),
            "avg_played_card": _rate(sum(played_cards), len(played_cards)),
            "played_card_bucket_share": {
                bucket: _rate(count, len(played_cards))
                for bucket, count in card_buckets.items()
            },
            "current_stack_fit_share": _rate(
                sum(fitting_cards), len(fitting_cards)
            ),
            "avg_cards_in_taken_stack": _rate(
                sum(len(stack) for stack in taken_stacks), len(taken_stacks)
            ),
            "avg_penalty_on_taken_stack": _rate(
                sum(card_points(card) for stack in taken_stacks for card in stack),
                len(taken_stacks),
            ),
        }

    def metrics(self):
        return super().metrics() | {"behavior": self.behavior_metrics()}


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for player in game:
                print([_chosen_move(record) for record in player[:-1]])
            print("--")

    def metrics(self):
        if not self.data:
            return {
                "terminal_rate": 0.0,
                "avg_game_rounds": 0.0,
                "avg_game_actions": 0.0,
                "strategy": {},
            }

        terminal_games = sum(game.by_seat[0][-1].terminal for game in self.data)
        return {
            "terminal_rate": terminal_games / len(self.data),
            "avg_game_rounds": sum(game.by_seat[0][-1].round for game in self.data)
            / len(self.data),
            "avg_game_actions": sum(
                max(len(trace) - 1, 0)
                for game in self.data
                for trace in game.by_seat
            )
            / len(self.data),
            "strategy": GroupedTraceMetrics(
                self.data.by_strategy,
                Take5TraceMetrics,
                scores=PointScores(),
            ).metrics(),
        }
