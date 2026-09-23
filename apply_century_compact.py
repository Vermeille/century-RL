from pathlib import Path


path = Path("boardrl/games/century/engine.pyx")
text = path.read_text()

replacements = [
    (
        """    cdef public Player p0
    cdef public Player p1
    cdef public Player p2
    cdef public Player p3
    cdef public Player p4
""",
        """    cdef public list players
""",
    ),
    (
        """        self.goal_cards = goal_cards
        self.p0 = Player()
        self.p0.stock = Stock.cfrom_str('YYY')
        self.p1 = Player()
        self.p1.stock = Stock.cfrom_str('YYYY')
        self.p2 = Player()
        self.p2.stock = Stock.cfrom_str('YYYY')
        self.p3 = Player()
        self.p3.stock = Stock.cfrom_str('YYYR')
        self.p4 = Player()
        self.p4.stock = Stock.cfrom_str('YYYR')

        self.num_players = num_players
""",
        """        self.goal_cards = goal_cards
        cdef int i
        cdef list starting_stocks = ['YYY', 'YYYY', 'YYYY', 'YYYR', 'YYYR']
        self.players = [Player() for _ in range(num_players)]
        for i in range(num_players):
            (<Player>self.players[i]).stock = Stock.cfrom_str(starting_stocks[i])

        self.num_players = num_players
""",
    ),
    (
        """    def copy(self, randomize=True):
        g = Century(empty=True)
        g.p0 = self.p0.copy()
        g.p1 = self.p1.copy()
        g.p2 = self.p2.copy()
        g.p3 = self.p3.copy()
        g.p4 = self.p4.copy()
        g.victory = self.victory.copy(randomize)
        g.action = self.action.copy(randomize)
        g.turn = self.turn
        g.moves = self.moves.copy()
        g.goal_cards = self.goal_cards
        g.num_players = self.num_players
        return g

    cpdef Player get_player(self, int index):
        assert index >= 0 and index < self.num_players
        if index == 0:
            return self.p0
        elif index == 1:
            return self.p1
        elif index == 2:
            return self.p2
        elif index == 3:
            return self.p3
        elif index == 4:
            return self.p4
        return -1
""",
        """    def copy(self, randomize=True):
        g = Century(empty=True)
        g.players = [player.copy() for player in self.players]
        g.victory = self.victory.copy(randomize)
        g.action = self.action.copy(randomize)
        g.turn = self.turn
        g.moves = self.moves
        g.goal_cards = self.goal_cards
        g.num_players = self.num_players
        return g

    cpdef Player get_player(self, int index):
        assert index >= 0 and index < self.num_players
        return <Player>self.players[index]
""",
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match, found {count}: {old.splitlines()[0]!r}")
    text = text.replace(old, new)

path.write_text(text)
