from boardrl.games.game2048.game import SLIDES
from boardrl.games.game2048.versus import SPAWN_CELLS
from boardrl.metrics import GameMetrics, Range


def tiles_from_state(state):
    rows = state.split("Board:\n", 1)[1].splitlines()[:4]
    return [int(cell) if cell != "." else 0 for row in rows for cell in row.split()]


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def metrics(self):
        # One endpoint per game; seat 0 always owns the sliding role, even with rotation.
        traces = [game.by_seat[0] for game in self.data]
        ends = [trace[-1] for trace in traces]
        boards = [tiles_from_state(end.state) for end in ends]
        max_tiles = [max(board) for board in boards]
        records = [record for trace in traces for record in trace[:-1]]
        spawns = [
            record for trace in self.data.all_traces() for record in trace[:-1]
            if record.moves[record.action_idx] in SPAWN_CELLS
        ]
        gains = [
            after.my_points - before.my_points
            for trace in traces for before, after in zip(trace, trace[1:])
        ]
        game_count = len(ends) or 1
        move_count = len(records) or 1
        spawn_count = len(spawns) or 1
        wins = [end.terminal and max_tile >= 2048 for end, max_tile in zip(ends, max_tiles)]
        return {
            "score": Range([end.my_points for end in ends] or [0]),
            "moves": Range([end.round for end in ends] or [0]),
            "max_tile": Range(max_tiles or [0]),
            "empty_cells": Range([board.count(0) for board in boards] or [0]),
            "win_rate": sum(wins) / game_count,
            "terminal_rate": sum(end.terminal for end in ends) / game_count,
            "truncated_rate": sum(end.truncated for end in ends) / game_count,
            "stuck_rate": sum(
                end.terminal and "Status: stuck" in end.state for end in ends
            ) / game_count,
            "tile_reached": {
                str(tile): sum(value >= tile for value in max_tiles) / game_count
                for tile in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
            },
            "direction_share": {
                move: sum(record.moves[record.action_idx] == move for record in records)
                / move_count
                for move in SLIDES
            },
            "merge_move_rate": sum(gain > 0 for gain in gains) / move_count,
            "score_per_move": sum(gains) / move_count,
            "spawn_actions": Range([
                sum(len(trace) - 1 for trace in game.by_seat[1:])
                for game in self.data
            ] or [0]),
            "spawn_position_share": {
                move.removeprefix("spawn "):
                sum(record.moves[record.action_idx] == move for record in spawns) / spawn_count
                for move in SPAWN_CELLS
            },
        }
