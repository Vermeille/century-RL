"""Behavior and outcome metrics reconstructed from public rollout observations."""

from collections import Counter

from boardrl.games.semantics import PointScores
from boardrl.games.traversal.game import EXPLORER, SENTINEL, VANGUARD, Piece, neighbors
from boardrl.games.traversal.terrain import SIZE, TERRAIN_TYPES
from boardrl.metrics import GameMetrics, GroupedTraceMetrics, TraceMetrics


KINDS = {kind.symbol: kind for kind in (EXPLORER, VANGUARD, SENTINEL)}
END_REASONS = (
    "sentinel", "two_finished", "elimination", "repetition", "stalemate",
    "move_limit_score", "move_limit_advancement", "move_limit_draw", "truncated", "unknown",
)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _field(state, name, default="unknown"):
    prefix = name + ": "
    return next(
        (line[len(prefix):] for line in state.splitlines() if line.startswith(prefix)),
        default,
    )


def _position(coordinate):
    row, col = int(coordinate[1:]) - 1, ord(coordinate[0]) - ord("a")
    if not 0 <= row < SIZE or not 0 <= col < SIZE:
        raise ValueError(f"Invalid Traversal coordinate: {coordinate}")
    return row, col


def _end_reason(end):
    if not end.terminal:
        return "truncated"
    reason = _field(end.state, "Status")
    return reason if reason in END_REASONS else "unknown"


def _piece(token):
    if len(token) != 2 or token[0] not in ("0", "1"):
        raise ValueError(f"Invalid Traversal piece: {token}")
    return Piece(int(token[0]), KINDS[token[1]])


class Observation:
    def __init__(self, state):
        lines = state.splitlines()
        height_start = lines.index("Terrain (height):") + 2
        piece_start = lines.index("Pieces:") + 2
        height_rows = [line.split() for line in lines[height_start:height_start + SIZE]]
        piece_rows = [line.split() for line in lines[piece_start:piece_start + SIZE]]
        for rows in (height_rows, piece_rows):
            if len(rows) != SIZE or any(
                len(row) != SIZE + 1 or int(row[0]) != index + 1
                for index, row in enumerate(rows)
            ):
                raise ValueError("Incomplete Traversal board")
        self.heights = tuple(tuple(map(int, row[1:])) for row in height_rows)
        self.board = {
            (row, col): _piece(token)
            for row, cells in enumerate(piece_rows)
            for col, token in enumerate(cells[1:]) if token != "."
        }
        self.finished = [
            _piece(token) for token in _field(state, "Finished", "-").split()
            if token != "-"
        ]
        self.preset = _field(state, "Terrain")

    @classmethod
    def parse(cls, state):
        try:
            return cls(state)
        except (ValueError, IndexError, KeyError):
            return None


class TraversalTraceMetrics(TraceMetrics):
    def behavior_metrics(self):
        counts = Counter()
        piece_actions = Counter()
        direction = Counter()
        finish_types = Counter()
        for trace in self.traces:
            for record in trace[:-1]:
                observation = Observation.parse(record.state)
                if observation is None:
                    continue
                source_text, target_text = record.moves[record.action_idx].split(">")
                source = _position(source_text)
                piece = observation.board[source]
                finish = target_text == "off"
                target = (
                    (SIZE if piece.player else -1, source[1])
                    if finish else _position(target_text)
                )
                captured = observation.board.get(target)
                capture = captured is not None and captured.player != piece.player
                counts["actions"] += 1
                piece_actions[piece.kind.name.lower()] += 1
                counts["captures"] += capture
                counts["finishes"] += finish
                forward = (target[0] - source[0]) * (1 if piece.player else -1)
                counts["forward_rows"] += forward
                direction[
                    "forward" if forward > 0 else "backward" if forward < 0 else "lateral"
                ] += 1
                if finish:
                    finish_types[piece.kind.name.lower()] += 1

                capture_available = finish_available = sentinel_available = False
                for move in record.moves:
                    start, end = move.split(">")
                    if end == "off":
                        finish_available = True
                        sentinel_available |= (
                            not observation.finished
                            and observation.board[_position(start)].kind.first_finish_wins
                        )
                    else:
                        occupant = observation.board.get(_position(end))
                        capture_available |= (
                            occupant is not None and occupant.player != piece.player
                        )
                counts["capture_available"] += capture_available
                counts["finish_available"] += finish_available
                counts["sentinel_available"] += sentinel_available
                counts["sentinel_converted"] += (
                    finish and not observation.finished and piece.kind.first_finish_wins
                )

                if capture:
                    delta = (
                        observation.heights[source[0]][source[1]]
                        - observation.heights[target[0]][target[1]]
                    )
                    counts["downhill_captures"] += delta > 0
                    counts["uphill_captures"] += delta < 0
                    allies = [
                        observation.board[position]
                        for position in neighbors(target)
                        if position != source and position in observation.board
                    ]
                    counts["supported_captures"] += any(
                        ally.player == piece.player for ally in allies
                    )
                    counts["reinforced_targets"] += any(
                        ally.player == captured.player for ally in allies
                    )

        actions, captures, finishes = counts["actions"], counts["captures"], counts["finishes"]
        return {
            "observation_coverage": _rate(
                actions, sum(max(len(trace) - 1, 0) for trace in self.traces),
            ),
            "piece_share": {
                kind.name.lower(): _rate(piece_actions[kind.name.lower()], actions)
                for kind in KINDS.values()
            },
            "capture_share": _rate(captures, actions),
            "finish_share": _rate(finishes, actions),
            "direction_share": {
                name: _rate(direction[name], actions)
                for name in ("forward", "backward", "lateral")
            },
            "avg_forward_rows": _rate(counts["forward_rows"], actions),
            "opportunity": {
                "capture_available_rate": _rate(counts["capture_available"], actions),
                "capture_conversion_rate": _rate(captures, counts["capture_available"]),
                "finish_available_rate": _rate(counts["finish_available"], actions),
                "finish_conversion_rate": _rate(finishes, counts["finish_available"]),
                "sentinel_win_available_rate": _rate(counts["sentinel_available"], actions),
                "sentinel_win_conversion_rate": _rate(
                    counts["sentinel_converted"], counts["sentinel_available"],
                ),
            },
            "combat": {
                "downhill_capture_share": _rate(counts["downhill_captures"], captures),
                "uphill_capture_share": _rate(counts["uphill_captures"], captures),
                "supported_capture_share": _rate(counts["supported_captures"], captures),
                "reinforced_target_share": _rate(counts["reinforced_targets"], captures),
            },
            "finish_type_share": {
                kind.name.lower(): _rate(finish_types[kind.name.lower()], finishes)
                for kind in KINDS.values()
            },
        }

    def metrics(self):
        ends = [trace[-1] for trace in self.traces]
        wins = [end for end in ends if end.terminal and end.episodic_utility > 0]
        causes = Counter(_end_reason(end) for end in wins)
        return super().metrics() | {
            "behavior": self.behavior_metrics(),
            "avg_points": _rate(sum(end.my_points for end in ends), len(ends)),
            "win_cause": {
                reason: _rate(causes[reason], len(wins))
                for reason in END_REASONS if reason != "truncated"
            },
        }


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def terrain_metrics(self, ends):
        observations = [Observation.parse(end.state) for end in ends]
        observations = [observation for observation in observations if observation is not None]
        presets = Counter(observation.preset for observation in observations)
        ranges, slopes = [], []
        for observation in observations:
            board = observation.heights
            heights = [height for row in board for height in row]
            ranges.append(max(heights) - min(heights))
            slopes.extend(
                abs(board[row][col] - board[r][c])
                for row in range(SIZE) for col in range(SIZE)
                for r, c in ((row + 1, col), (row, col + 1)) if r < SIZE and c < SIZE
            )
        return {
            "observation_coverage": _rate(len(observations), len(ends)),
            "preset_share": {
                name: _rate(presets[name], len(observations))
                for name in (*TERRAIN_TYPES, "flat", "unknown")
            },
            "unique_map_share": _rate(
                len({observation.heights for observation in observations}), len(observations),
            ),
            "avg_height_range": _rate(sum(ranges), len(ranges)),
            "avg_adjacent_height_difference": _rate(sum(slopes), len(slopes)),
        }

    def metrics(self):
        ends = [game.by_seat[0][-1] for game in self.data]
        count = len(ends)
        reasons = Counter(_end_reason(end) for end in ends)
        return {
            "terminal_rate": _rate(sum(end.terminal for end in ends), count),
            "draw_rate": _rate(
                sum(end.terminal and end.episodic_utility == 0 for end in ends), count,
            ),
            "avg_game_actions": _rate(
                sum(max(len(trace) - 1, 0) for game in self.data for trace in game.by_seat),
                count,
            ),
            "end_reason": {reason: _rate(reasons[reason], count) for reason in END_REASONS},
            "terrain": self.terrain_metrics(ends),
            "strategy": GroupedTraceMetrics(
                self.data.by_strategy, TraversalTraceMetrics, scores=PointScores(),
            ).metrics() if count else {},
        }

    def print_short_history(self):
        for game in self.data:
            print(game.by_seat[0][-1].state)
