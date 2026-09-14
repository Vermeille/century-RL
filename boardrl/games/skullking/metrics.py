"""Game-specific metrics for Skull King self-play.

Skull King is primarily a bidding game: before every round each player predicts
exactly how many tricks they will take. These metrics focus on bid calibration
and on whether self-play develops non-trivial bidding behaviour.

Completed-round outcomes are reconstructed from the textual observations that
are already stored in rollout records. During the bidding phase of round N,
``Tricks`` still contains the final trick counts from round N-1; the terminal
state contains the final round's trick counts. This avoids adding recorder-only
state to the game implementation.
"""

from __future__ import annotations

from boardrl.metrics import GameMetrics


def _field(state: str, name: str) -> str:
    prefix = f"{name}:"
    for line in state.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise ValueError(f"Could not find {name!r} in Skull King state")


def _round_number(state: str) -> int:
    return int(_field(state, "Round").split("/", 1)[0])


def _phase(state: str) -> str:
    return _field(state, "Phase")


def _tricks(state: str) -> list[int]:
    return [int(value) for value in _field(state, "Tricks").split()]


def _chosen_move(record) -> str:
    return record.moves[record.action_idx]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _completed_rounds(game):
    """Yield ``(round_number, bids, tricks)`` for completed rounds in a trace."""

    num_players = len(game)
    bids: dict[int, list[int | None]] = {}
    outcomes: dict[int, list[int]] = {}

    for player, trace in enumerate(game):
        for record in trace[:-1]:
            if _phase(record.state) != "bid":
                continue

            round_number = _round_number(record.state)
            bids.setdefault(round_number, [None] * num_players)[player] = int(
                _chosen_move(record)
            )

            # A bid observation for round N is made before start_round_ resets
            # tricks, so it still exposes the result of round N-1.
            if round_number > 1:
                outcomes[round_number - 1] = _tricks(record.state)

        end = trace[-1]
        if _phase(end.state) == "ended":
            outcomes[_round_number(end.state)] = _tricks(end.state)

    for round_number in sorted(bids):
        round_bids = bids[round_number]
        round_tricks = outcomes.get(round_number)
        if round_tricks is None or any(bid is None for bid in round_bids):
            continue
        yield round_number, [int(bid) for bid in round_bids], round_tricks


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for player in game:
                print([_chosen_move(record) for record in player[:-1]])
            print("--")

    def metrics(self):
        avg_len = sum(len(history) for game in self.data for history in game) / (
            len(self.data) * self.data.num_players()
        )

        won_bets: list[float] = []
        abs_bet_errors: list[float] = []
        bet_diff_equal: list[float] = []
        abs_bet_diff_equal: list[float] = []
        zero_bets: list[float] = []
        zero_bet_wins: list[float] = []
        overbets: list[float] = []
        underbets: list[float] = []

        num_players = self.data.num_players()
        for game in self.data:
            for round_number, bids, tricks in _completed_rounds(game):
                equal_bid = round_number / num_players
                for bid, taken in zip(bids, tricks):
                    won = bid == taken
                    won_bets.append(float(won))
                    abs_bet_errors.append(abs(bid - taken))

                    diff_equal = bid - equal_bid
                    bet_diff_equal.append(diff_equal)
                    abs_bet_diff_equal.append(abs(diff_equal))

                    is_zero = bid == 0
                    zero_bets.append(float(is_zero))
                    if is_zero:
                        zero_bet_wins.append(float(won))

                    overbets.append(float(bid > taken))
                    underbets.append(float(bid < taken))

        return {
            "avg_len": avg_len,
            "sensitivity": self.data.sensitivity(),
            "won_bet_ratio": _mean(won_bets),
            "avg_abs_bet_error": _mean(abs_bet_errors),
            # Equal distribution is exactly N/P as requested: round number N
            # divided by player count P. Signed difference measures whether the
            # policy systematically bids above/below that neutral share.
            "bet_diff_equal": _mean(bet_diff_equal),
            "abs_bet_diff_equal": _mean(abs_bet_diff_equal),
            "zero_bet_ratio": _mean(zero_bets),
            "zero_bet_win_ratio": _mean(zero_bet_wins),
            "overbet_ratio": _mean(overbets),
            "underbet_ratio": _mean(underbets),
        }
