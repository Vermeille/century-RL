"""Role-aware outcomes and behavior from public Hnefatafl observations."""

from collections import Counter

from boardrl.games.hnefatafl.game import CARDINAL, CORNERS, SIZE, Move
from boardrl.games.semantics import OutcomeScores
from boardrl.metrics import GameMetrics, GroupedTraceMetrics, TraceMetrics


END_REASONS = (
    "king_escaped", "king_captured", "stalemate", "move_limit",
    "truncated", "unknown",
)


def _rate(total, count):
    return total / count if count else 0.0


def _field(state, name):
    prefix = name + ":"
    return next(line[len(prefix):] for line in state.splitlines() if line.startswith(prefix))


class Observation:
    def __init__(self, state):
        lines = state.splitlines()
        rows = lines[2:2 + SIZE]
        if len(rows) != SIZE or any(
            len(row) != SIZE
            or any(cell not in ("A", "D", "K", ".", "T", "X") for cell in row)
            for row in rows
        ):
            raise ValueError("Incomplete Hnefatafl board")
        self.board = {
            (row, col): cell
            for row, cells in enumerate(rows)
            for col, cell in enumerate(cells) if cell in ("A", "D", "K")
        }
        self.captured = tuple(map(int, _field(state, "C").split()))
        if len(self.captured) != 2 or min(self.captured) < 0:
            raise ValueError("Invalid Hnefatafl capture counts")
        self.king = next((position for position, piece in self.board.items() if piece == "K"), None)

    @classmethod
    def parse(cls, state):
        try:
            return cls(state)
        except (ValueError, IndexError, StopIteration):
            return None


def _end_reason(end):
    if not end.terminal:
        return "truncated"
    try:
        reason = _field(end.state, "End")
    except StopIteration:
        return "unknown"
    return reason if reason in END_REASONS else "unknown"


class HnefataflTraceMetrics(TraceMetrics):
    def behavior_metrics(self):
        counts = Counter()
        for trace in self.traces:
            observations = [Observation.parse(record.state) for record in trace]
            for index, record in enumerate(trace[:-1]):
                before, after = observations[index:index + 2]
                if before is None:
                    continue
                move = Move.from_text(record.moves[record.action_idx])
                source, target = move.source, move.destination
                counts["actions"] += 1
                counts["king_moves"] += before.board[source] == "K"
                counts["distance"] += sum(abs(a - b) for a, b in zip(source, target))
                counts["legal_moves"] += len(record.moves)
                if after is None:
                    continue
                # Consecutive observations in one seat trace bracket its move
                # and possibly the opponent's. Only this seat removes soldiers
                # belonging to its opponent, so attribution remains exact.
                captures = after.captured[1 - record.player] - before.captured[1 - record.player]
                counts["capture_observations"] += 1
                counts["captures"] += captures
                counts["capture_actions"] += captures > 0
                counts["multi_capture_actions"] += captures > 1
        actions = counts["actions"]
        capture_observations = counts["capture_observations"]
        return {
            "observation_coverage": _rate(
                actions, sum(max(len(trace) - 1, 0) for trace in self.traces),
            ),
            "king_move_share": _rate(counts["king_moves"], actions),
            "avg_move_distance": _rate(counts["distance"], actions),
            "avg_legal_moves": _rate(counts["legal_moves"], actions),
            "capture_observation_coverage": _rate(capture_observations, actions),
            "capture_share": _rate(counts["capture_actions"], capture_observations),
            "multi_capture_share": _rate(counts["multi_capture_actions"], capture_observations),
            "avg_captures": _rate(counts["captures"], capture_observations),
        }

    def metrics(self):
        return super().metrics() | {"behavior": self.behavior_metrics()}


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def metrics(self):
        ends = [game.by_seat[0][-1] for game in self.data]
        count = len(ends)
        reasons = Counter(_end_reason(end) for end in ends)
        observations = [Observation.parse(end.state) for end in ends]
        observations = [observation for observation in observations if observation is not None]
        kings = [observation for observation in observations if observation.king is not None]
        attacker_wins = sum(
            end.terminal and end.episodic_utility == (1 if end.player == 0 else -1)
            for end in ends
        )
        defender_wins = sum(
            end.terminal and end.episodic_utility == (1 if end.player == 1 else -1)
            for end in ends
        )
        return {
            "terminal_rate": _rate(sum(end.terminal for end in ends), count),
            "attacker_win_rate": _rate(attacker_wins, count),
            "defender_win_rate": _rate(defender_wins, count),
            "draw_rate": _rate(
                sum(end.terminal and end.episodic_utility == 0 for end in ends), count,
            ),
            "end_reason": {reason: _rate(reasons[reason], count) for reason in END_REASONS},
            "observation_coverage": _rate(len(observations), count),
            "material": {
                "avg_attackers_remaining": _rate(
                    sum(list(observation.board.values()).count("A") for observation in observations),
                    len(observations),
                ),
                "avg_defenders_remaining": _rate(
                    sum(list(observation.board.values()).count("D") for observation in observations),
                    len(observations),
                ),
                "avg_attackers_captured": _rate(
                    sum(observation.captured[0] for observation in observations), len(observations),
                ),
                "avg_defenders_captured": _rate(
                    sum(observation.captured[1] for observation in observations), len(observations),
                ),
            },
            "king": {
                "on_board_rate": _rate(len(kings), len(observations)),
                "avg_corner_distance": _rate(
                    sum(min(
                        sum(abs(a - b) for a, b in zip(observation.king, corner))
                        for corner in CORNERS
                    ) for observation in kings), len(kings),
                ),
                "avg_adjacent_attackers": _rate(
                    sum(
                        observation.board.get((observation.king[0] + dr, observation.king[1] + dc)) == "A"
                        for observation in kings for dr, dc in CARDINAL
                    ), len(kings),
                ),
            },
            "strategy": GroupedTraceMetrics(
                self.data.by_strategy, HnefataflTraceMetrics, scores=OutcomeScores(),
            ).metrics() if count else {},
        }

    def print_short_history(self):
        for game in self.data:
            print(game.by_seat[0][-1].state)
