import re

from boardrl.games.semantics import OutcomeScores
from boardrl.metrics import GameMetrics, GroupedTraceMetrics, TraceMetrics


_BOARD_ROW = re.compile(r"^([1-5]) ((?:[0-4][.#OX](?: [0-4][.#OX]){4}))$")
_COORD = re.compile(r"^[a-e][1-5]$")
_DIRECTIONS = {
    "u": (0, -1),
    "d": (0, 1),
    "l": (-1, 0),
    "r": (1, 0),
    "ul": (-1, -1),
    "ur": (1, -1),
    "dl": (-1, 1),
    "dr": (1, 1),
}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _coord_xy(coord):
    return ord(coord[0]) - ord("a"), int(coord[1]) - 1


def _step(coord, direction):
    x, y = _coord_xy(coord)
    dx, dy = _DIRECTIONS[direction]
    x += dx
    y += dy
    if not 0 <= x < 5 or not 0 <= y < 5:
        raise ValueError(f"Santorini direction {direction!r} leaves the board from {coord}")
    return f"{chr(ord('a') + x)}{y + 1}"


def _resolve_action(move):
    """Return (source, destination, build) for absolute or relative encodings."""
    if move.startswith("P:"):
        return None

    movement, *build_parts = move.split("+")
    source, destination_token = movement.split(">")
    destination = (
        destination_token
        if _COORD.fullmatch(destination_token)
        else _step(source, destination_token)
    )

    build = None
    if build_parts:
        build_token = build_parts[0]
        build = (
            build_token
            if _COORD.fullmatch(build_token)
            else _step(destination, build_token)
        )
    return source, destination, build


def _parse_board(state):
    board = {}
    for line in state.splitlines():
        match = _BOARD_ROW.fullmatch(line)
        if not match:
            continue
        y = int(match.group(1)) - 1
        for x, cell in enumerate(match.group(2).split(" ")):
            coord = f"{chr(ord('a') + x)}{y + 1}"
            board[coord] = (int(cell[0]), cell[1])
    return board


def _adjacent(first, second):
    ax, ay = _coord_xy(first)
    bx, by = _coord_xy(second)
    return max(abs(ax - bx), abs(ay - by)) == 1


def _inner_board(coord):
    x, y = _coord_xy(coord)
    return 1 <= x <= 3 and 1 <= y <= 3


def _chosen_move(record):
    if not hasattr(record, "moves") or not hasattr(record, "action_idx"):
        return None
    return record.moves[record.action_idx]


def _last_play_move(trace):
    for record in reversed(trace[:-1]):
        move = _chosen_move(record)
        if move is not None and not move.startswith("P:"):
            return move
    return None


class SantoriniTraceMetrics(TraceMetrics):
    def behavior_metrics(self):
        play_states = 0
        move_options = 0
        winning_threat_states = 0
        winning_threat_converted = 0

        height_moves = {"up": 0, "flat": 0, "down": 0}
        inner_destinations = 0

        worker_height_sum = 0.0
        worker_height_states = 0

        builds = 0
        build_levels = {1: 0, 2: 0, 3: 0, 4: 0}
        vacated_square_builds = 0
        opponent_blocking_builds = 0

        for trace in self.traces:
            own_mark = "O" if trace.seat_id == 0 else "X"
            opponent_mark = "X" if own_mark == "O" else "O"

            for record in trace[:-1]:
                chosen = _chosen_move(record)
                if chosen is None or chosen.startswith("P:"):
                    continue

                board = _parse_board(record.state)
                if len(board) != 25:
                    continue

                source, destination, build = _resolve_action(chosen)
                play_states += 1

                own_heights = [
                    height
                    for height, occupant in board.values()
                    if occupant == own_mark
                ]
                if own_heights:
                    worker_height_sum += sum(own_heights) / len(own_heights)
                    worker_height_states += 1

                legal_play_actions = [
                    resolved
                    for move in record.moves
                    if not move.startswith("P:")
                    for resolved in [_resolve_action(move)]
                    if resolved is not None
                ]
                move_options += len(
                    {(src, dst) for src, dst, _build in legal_play_actions}
                )
                win_available = any(
                    legal_build is None
                    for _src, _dst, legal_build in legal_play_actions
                )
                if win_available:
                    winning_threat_states += 1
                    if build is None:
                        winning_threat_converted += 1

                source_height = board[source][0]
                destination_height = board[destination][0]
                delta = destination_height - source_height
                height_moves["up" if delta > 0 else "down" if delta < 0 else "flat"] += 1
                inner_destinations += int(_inner_board(destination))

                if build is None:
                    continue

                builds += 1
                build_height = board[build][0]
                resulting_height = build_height + 1
                build_levels[resulting_height] += 1
                vacated_square_builds += int(build == source)

                blocks_opponent = False
                for coord, (opponent_height, occupant) in board.items():
                    if occupant != opponent_mark or not _adjacent(coord, build):
                        continue
                    accessible_before = build_height <= opponent_height + 1
                    inaccessible_after = (
                        resulting_height == 4
                        or resulting_height > opponent_height + 1
                    )
                    if accessible_before and inaccessible_after:
                        blocks_opponent = True
                        break
                opponent_blocking_builds += int(blocks_opponent)

        return {
            "mobility": {
                "avg_move_options": _rate(move_options, play_states),
                "winning_threat_state_rate": _rate(
                    winning_threat_states, play_states
                ),
                "winning_threat_conversion_rate": _rate(
                    winning_threat_converted, winning_threat_states
                ),
            },
            "position": {
                "avg_worker_height": _rate(
                    worker_height_sum, worker_height_states
                ),
                "inner_board_destination_share": _rate(
                    inner_destinations, play_states
                ),
            },
            "movement": {
                "height_mix": {
                    name: _rate(count, play_states)
                    for name, count in height_moves.items()
                },
            },
            "building": {
                "result_level_share": {
                    "1": _rate(build_levels[1], builds),
                    "2": _rate(build_levels[2], builds),
                    "3": _rate(build_levels[3], builds),
                    "dome": _rate(build_levels[4], builds),
                },
                "vacated_square_share": _rate(vacated_square_builds, builds),
                "opponent_blocking_share": _rate(
                    opponent_blocking_builds, builds
                ),
            },
        }

    def win_cause_metrics(self):
        wins = 0
        climb_wins = 0
        immobilization_wins = 0

        for trace in self.traces:
            terminal = trace[-1]
            if getattr(terminal, "my_points", 0) <= 0:
                continue
            wins += 1
            last_move = _last_play_move(trace)
            resolved = _resolve_action(last_move) if last_move is not None else None
            if resolved is not None and resolved[2] is None:
                climb_wins += 1
            else:
                immobilization_wins += 1

        return {
            "climb_share": _rate(climb_wins, wins),
            "immobilization_share": _rate(immobilization_wins, wins),
        }

    def metrics(self):
        return super().metrics() | {
            "behavior": self.behavior_metrics(),
            "win_cause": self.win_cause_metrics(),
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
                SantoriniTraceMetrics,
                scores=OutcomeScores(),
            ).metrics(),
        }
