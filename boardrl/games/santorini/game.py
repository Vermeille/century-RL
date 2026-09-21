import random


class Santorini:
    SIZE = 5
    DOME = 4
    WORKERS_PER_PLAYER = 2
    SETUP_ACTIONS = 4
    DIRECTIONS = {
        "u": (0, -1),
        "d": (0, 1),
        "l": (-1, 0),
        "r": (1, 0),
        "ul": (-1, -1),
        "ur": (1, -1),
        "dl": (-1, 1),
        "dr": (1, 1),
    }
    DELTA_TO_DIRECTION = {delta: direction for direction, delta in DIRECTIONS.items()}

    def __init__(self, num_players: int = 2):
        assert num_players == 2
        self.num_players = num_players
        self.heights = [0 for _ in range(self.SIZE * self.SIZE)]
        self.workers = [
            [None for _ in range(self.WORKERS_PER_PLAYER)] for _ in range(2)
        ]
        self.turn = 0
        self.moves = []
        self._winner = None
        self._refresh_moves()

    @classmethod
    def _coord(cls, index: int) -> str:
        x = index % cls.SIZE
        y = index // cls.SIZE
        return f"{chr(ord('a') + x)}{y + 1}"

    @classmethod
    def _index(cls, coord: str) -> int:
        assert len(coord) == 2
        x = ord(coord[0]) - ord("a")
        y = int(coord[1]) - 1
        assert 0 <= x < cls.SIZE
        assert 0 <= y < cls.SIZE
        return y * cls.SIZE + x

    @classmethod
    def _direction(cls, source: int, destination: int) -> str:
        sx, sy = source % cls.SIZE, source // cls.SIZE
        dx, dy = destination % cls.SIZE, destination // cls.SIZE
        return cls.DELTA_TO_DIRECTION[(dx - sx, dy - sy)]

    @classmethod
    def _step(cls, source: int, direction: str) -> int:
        dx, dy = cls.DIRECTIONS[direction]
        x = source % cls.SIZE + dx
        y = source // cls.SIZE + dy
        assert 0 <= x < cls.SIZE and 0 <= y < cls.SIZE
        return y * cls.SIZE + x

    @classmethod
    def _neighbors(cls, index: int):
        x = index % cls.SIZE
        y = index // cls.SIZE
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                xx = x + dx
                yy = y + dy
                if 0 <= xx < cls.SIZE and 0 <= yy < cls.SIZE:
                    yield yy * cls.SIZE + xx

    def copy(self):
        g = Santorini()
        g.heights = self.heights[:]
        g.workers = [workers[:] for workers in self.workers]
        g.turn = self.turn
        g.moves = self.moves[:]
        g._winner = self._winner
        return g

    def round(self):
        return max(self.turn - self.SETUP_ACTIONS, 0) // 2

    def current_player(self):
        if self.turn < self.WORKERS_PER_PLAYER:
            return 0
        if self.turn < self.SETUP_ACTIONS:
            return 1
        return (self.turn - self.SETUP_ACTIONS) % 2

    def _setup(self):
        return self.turn < self.SETUP_ACTIONS

    def _occupied(self):
        return {
            position
            for workers in self.workers
            for position in workers
            if position is not None
        }

    def _setup_moves(self):
        occupied = self._occupied()
        return [
            f"P:{self._coord(index)}"
            for index in range(self.SIZE * self.SIZE)
            if index not in occupied
        ]

    def _normal_moves(self, player):
        occupied = self._occupied()
        moves = []

        for source in self.workers[player]:
            assert source is not None
            source_height = self.heights[source]

            for destination in self._neighbors(source):
                if destination in occupied:
                    continue
                destination_height = self.heights[destination]
                if destination_height == self.DOME:
                    continue
                if destination_height > source_height + 1:
                    continue

                move_direction = self._direction(source, destination)
                move_prefix = f"{self._coord(source)}>{move_direction}"

                # Reaching level 3 from level 2 wins immediately; there is no build.
                if source_height == 2 and destination_height == 3:
                    moves.append(move_prefix)
                    continue

                occupied_after_move = (occupied - {source}) | {destination}
                for build in self._neighbors(destination):
                    if build in occupied_after_move:
                        continue
                    if self.heights[build] == self.DOME:
                        continue
                    build_direction = self._direction(destination, build)
                    moves.append(f"{move_prefix}+{build_direction}")

        return moves

    def _refresh_moves(self):
        if self._winner is not None:
            self.moves = []
            return

        if self._setup():
            self.moves = self._setup_moves()
            return

        self.moves = self._normal_moves(self.current_player())
        if not self.moves:
            self._winner = 1 - self.current_player()

    def display(self, force=-1):
        if force == -1:
            player = self.current_player()
        else:
            assert force in (0, 1)
            player = force

        phase = "setup" if self._setup() else "play"
        worker_marks = {}
        for owner, workers in enumerate(self.workers):
            mark = "O" if owner == 0 else "X"
            for position in workers:
                if position is not None:
                    worker_marks[position] = mark

        lines = [f">{player} {phase}", "   a  b  c  d  e"]
        for y in range(self.SIZE):
            cells = []
            for x in range(self.SIZE):
                index = y * self.SIZE + x
                occupant = worker_marks.get(index, ".")
                height = self.heights[index]
                if height == self.DOME:
                    occupant = "#"
                cells.append(f"{height}{occupant}")
            lines.append(f"{y + 1} " + " ".join(cells))
        return "\n".join(lines) + "\n"

    def display_with_moves(self):
        return self.display() + "\n".join(f"@{move}" for move in self.moves)

    def play_str(self, move):
        assert not self.ended()
        assert move in self.moves
        player = self.current_player()

        if self._setup():
            position = self._index(move[2:])
            worker = self.workers[player].index(None)
            self.workers[player][worker] = position
            self.turn += 1
            self._refresh_moves()
            return

        movement, *build_parts = move.split("+")
        source_coord, move_direction = movement.split(">")
        source = self._index(source_coord)
        destination = self._step(source, move_direction)
        source_height = self.heights[source]
        destination_height = self.heights[destination]

        worker = self.workers[player].index(source)
        self.workers[player][worker] = destination
        self.turn += 1

        if source_height == 2 and destination_height == 3:
            self._winner = player
            self.moves = []
            return

        assert len(build_parts) == 1
        build = self._step(destination, build_parts[0])
        self.heights[build] += 1
        self._refresh_moves()

    def ended(self):
        return self._winner is not None

    def winner(self):
        return self._winner

    def points_for(self, player):
        winner = self.winner()
        if winner is None:
            return 0
        return 1 if winner == player else -1

    def points(self):
        return self.points_for(self.current_player())

    diff_points = points
    diff_points_for = points_for

    def play_idx(self, index):
        return self.play_str(self.moves[index])

    def simulate_to_end(self):
        while not self.ended():
            self.play_str(random.choice(self.moves))
