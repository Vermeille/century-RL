# cython: profile=False
# cython: language_level=3
# cython: linetrace=False
import torch
import cython
from cpython cimport array
import array
import copy
import random
from random import choice as rndchoice
from typing import Tuple, List
from libc.math cimport sqrt, log
from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cpython.unicode cimport PyUnicode_DecodeLatin1


cdef random_buy_fast(Game g):
    moves = g.moves
    for mov in moves:
        if mov[0] == "V":
            return mov

    return random.choice(moves)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int fast_sample(x):
    assert x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1)
    x = x.detach().cpu().contiguous()
    x = x.numpy() if x.ndim == 1 else x[0].numpy()
    cdef float[:] x_view = x
    cdef float* x_ = &x_view[0]
    cdef float total = 0.
    cdef float r
    cdef float acc = 0
    cdef int i
    cdef int n = x.shape[0]

    for i in range(n):
        total += x_[i]

    r = rand() / RAND_MAX * total
    for i in range(n):
        acc += x_[i]
        if acc >= r:
            return i
    print(x)
    assert False, ("Should not reach here. Called fast_sample on "
        "an invalid distribution (all zeros or negative values)")


class Illegal(BaseException):
    pass


cdef class Stock:
    cdef int Y
    cdef int R
    cdef int G
    cdef int B

    def __init__(self):
        self.Y = 0
        self.R = 0
        self.G = 0
        self.B = 0

    def __str__(self):
        return self.str()

    cdef str str(self):
        #return 'Y' * self.Y + 'R' * self.R + 'G' * self.G + 'B' * self.B
        out = []
        for i in range(self.Y):
            out.append('Y')
        for i in range(self.R):
            out.append('R')
        for i in range(self.G):
            out.append('G')
        for i in range(self.B):
            out.append('B')
        return ''.join(out)

    cdef Stock ccopy(self):
        o = Stock()
        Stock.iadd(o, self)
        return o

    def copy(self):
        return Stock.ccopy(self)

    @staticmethod
    cdef Stock cfrom_str(s: str):
        cdef Py_UCS4 c
        stock = Stock()
        for c in s:
            if c not in 'YRGB':
                raise Illegal()

            #stock.__dict__[c] += 1
            if c == 'Y':
                stock.Y += 1
            elif c == 'R':
                stock.R += 1
            elif c == 'G':
                stock.G += 1
            elif c == 'B':
                stock.B += 1
        return stock


    def __contains__(self, ref: Stock):
        return self.contains(ref)

    cdef int contains(self, ref: Stock):
        return (self.Y >= ref.Y and self.R >= ref.R and self.G >= ref.G
                and self.B >= ref.B)

    cdef int size(self) nogil:
        return self.Y + self.R + self.G + self.B

    def __len__(self):
        return self.size()

    def __sub__(self, ref: Stock):
        return Stock.sub(self, ref)

    cdef inline Stock sub(self, ref: Stock):
        if not self.contains(ref):
            raise Illegal()
        out = Stock()
        out.Y = self.Y - ref.Y
        out.R = self.R - ref.R
        out.G = self.G - ref.G
        out.B = self.B - ref.B
        return out

    cdef Stock add(self, ref: Stock):
        out = Stock()
        out.Y = self.Y + ref.Y
        out.R = self.R + ref.R
        out.G = self.G + ref.G
        out.B = self.B + ref.B
        return out

    def __add__(self, ref: Stock):
        return Stock.add(self, ref)

    cdef void iadd(self: Stock, ref: Stock) nogil:
        self.Y += ref.Y
        self.R += ref.R
        self.G += ref.G
        self.B += ref.B

    def __iadd__(self, ref: Stock):
        Stock.iadd(self, ref)
        return self

    cdef int isub(self, ref: Stock) except 0:
        if not self.contains(ref):
            raise Illegal()
        self.Y -= ref.Y
        self.R -= ref.R
        self.G -= ref.G
        self.B -= ref.B
        return 1

    def __isub__(self, ref: Stock):
        Stock.isub(self, ref)
        return self

    cdef points(self):
        return self.R + self.G + self.B

    cdef trim(self):
        while self.size() > 10:
            #m = max(list(self.__dict__.items()), key=lambda x: x[1])[0]
            m = max(list({
                'Y': self.Y,
                'R': self.R,
                'G': self.G,
                'B': self.B
            }.items()),
                    key=lambda x: x[1])[0]
            if m == 'Y':
                self.Y -= 1
            elif m == 'R':
                self.R -= 1
            elif m == 'G':
                self.G -= 1
            elif m == 'B':
                self.B -= 1


cdef class ActionCard:
    cdef Stock from_
    cdef Stock to_
    cdef list str_cache

    def __init__(self, from_, to_):
        self.from_ = (from_
                      if isinstance(from_, Stock) else Stock.cfrom_str(from_))
        self.to_ = (to_ if isinstance(to_, Stock) else Stock.cfrom_str(to_))
        self.str_cache = [self.from_.str() + '->' + self.to_.str()]
        if self.from_.size() > 0:
            needed = self.from_.ccopy()
            gen = self.to_.ccopy()
            while needed.size() <= 10:
                self.str_cache.append(f'{needed}->{gen}')
                Stock.iadd(needed, self.from_)
                Stock.iadd(gen, self.to_)

    def __str__(self):
        return self.str_cache[0]

    def takes(self):
        return self.from_

    def gives(self):
        return self.to_

    __repr__ = __str__

    @staticmethod
    def from_str(s):
        f, t = s.split('->')
        return ActionCard(f, t)

    def gen_move(self, stock: Stock):
        if self.from_.size() == 0:
            yield self.str_cache[0]
            return
        i = 1
        needed = self.from_.ccopy()
        while stock.contains(needed):
            yield self.str_cache[i]
            Stock.iadd(needed, self.from_)
            i += 1

    def allows(self, from_: Stock, to_: Stock):
        if self.from_.size() == 0:
            return self.to_.contains(to_)

        cdef Stock gen_
        from_ = from_.ccopy()
        gen = Stock()
        while from_.contains(self.from_):
            Stock.isub(from_, self.from_)
            Stock.iadd(gen, self.to_)

        return gen.contains(to_)



cdef class VictoryCard:
    cdef public int points
    cdef public Stock cost

    def __init__(self, cost, points):
        self.points = points
        self.cost = (cost if isinstance(cost, Stock) else Stock.cfrom_str(cost))

    def __str__(self):
        return self.cost.str() + '->' + str(self.points)

    __repr__ = __str__

    @staticmethod
    def from_str(s):
        c, p = s.split('->')
        return VictoryCard(c, int(p))


joker2_moves = [
    'Y->R',
    'R->G',
    'G->B',

    'YY->RR',
    'YR->RG',
    'YG->RB',

    # 'RY->GR', Already covered by YR->RG
    'RR->GG',
    'RG->GB',

    # 'GY->BR', Already covered by YG->RB
    # 'GR->BG', Already covered by RG->GB
    'GG->BB',

    'Y->G',
    'R->B',
]

joker3_moves = joker2_moves + [
    # Y->R
    'YYY->RRR',
    'YYR->RRG',
    'YYG->RRB',

    'YRR->RGG',
    'YRG->RGB',

    'YGG->RBB',

    'YY->RG',
    'YR->RB',

    # R->G
    'RRR->GGG',
    'RRG->GGB',

    'RGG->GBB',

    'YR->GG',
    'RR->GB',

    # G->B
    'GGG->BBB',

    'YG->GB',
    'RG->BB',

    #
    'Y->B',
]

class Joker(ActionCard):

    def __init__(self, n):
        assert n <= 3
        if n == 2:
            moves = joker2_moves
        elif n == 3:
            moves = joker3_moves
        self.n = n
        self.instances = [ActionCard.from_str(move) for move in moves]

    def __str__(self):
        return 'X' * self.n

    def allows(self, from_, to_):
        for ins in self.instances:
            if ins.allows(from_, to_):
                return True
        return False

    def gen_move(self, stock):
        moves = []
        for ins in self.instances:
            if ins.takes() in stock:
                moves.append(str(ins))
        return moves


cdef class VictoryPile:
    cdef public list pile

    def __init__(self, empty=False):
        if empty:
            return
        self.pile = [
            VictoryCard.from_str('YRGB->12'),
            VictoryCard.from_str('YRGGGB->18'),
            VictoryCard.from_str('YGGB->12'),
            VictoryCard.from_str('GGBBB->18'),
            VictoryCard.from_str('GGGGG->15'),
            VictoryCard.from_str('YYBBB->14'),
            VictoryCard.from_str('YYRB->9'),
            VictoryCard.from_str('YYGGG->11'),
            VictoryCard.from_str('RRGGBB->19'),
            VictoryCard.from_str('RRRGG->12'),
            VictoryCard.from_str('RRGG->10'),
            VictoryCard.from_str('BBBB->16'),
            VictoryCard.from_str('RRRR->8'),
            VictoryCard.from_str('RRRRR->10'),
            VictoryCard.from_str('YYRR->6'),
            VictoryCard.from_str('YYGGBB->17'),
            VictoryCard.from_str('YYBB->10'),
            VictoryCard.from_str('RRRBB->14'),
            VictoryCard.from_str('YRRRGB->16'),
            VictoryCard.from_str('YYGG->8'),
            VictoryCard.from_str('RRGB->12'),
            VictoryCard.from_str('GGBB->14'),
            VictoryCard.from_str('GGGBB->17'),
            VictoryCard.from_str('RRGGG->13'),
            VictoryCard.from_str('GGGG->12'),
            VictoryCard.from_str('YRGBBB->20'),
            VictoryCard.from_str('RRBBB->16'),
            VictoryCard.from_str('YYYRR->7'),
            VictoryCard.from_str('YYYGG->9'),
            VictoryCard.from_str('YYRRGG->13'),
            VictoryCard.from_str('YYRRR->8'),
            VictoryCard.from_str('YYYRGB->14'),
            VictoryCard.from_str('RRBB->12'),
            VictoryCard.from_str('YYRRBB->15'),
            VictoryCard.from_str('YYYBB->11'),
            VictoryCard.from_str('BBBBB->20'),
        ]
        random.shuffle(self.pile)
        # FIXME add coins

    def copy(self, randomize=True):
        v = VictoryPile(empty=True)
        v.pile = self.pile[:]
        if randomize:
            random.shuffle(v.pile[5:])
        return v

    cpdef visible(self):
        return self.pile[:5]

    def __str__(self):
        return '\n'.join([f'V{i} {p}' for i, p in enumerate(self.visible())])

    def take(self, idx: int) -> VictoryCard:
        if idx >= min(5, len(self.pile)):
            raise Illegal()

        v = self.pile[idx]
        del self.pile[idx]
        return v


cdef class ActionPile:
    cdef public list pile
    cdef public list on_cards

    def __init__(self, empty=False):
        if empty:
            return
        self.pile = [
            ActionCard.from_str('RRR->GGYY'),
            ActionCard.from_str('RR->BYY'),
            ActionCard.from_str('RRR->GGG'),
            ActionCard.from_str('RRR->GGG'),
            ActionCard.from_str('RR->GG'),
            ActionCard.from_str('YR->B'),
            ActionCard.from_str('->YYY'),
            ActionCard.from_str('YYG->BB'),
            ActionCard.from_str('G->RR'),
            ActionCard.from_str('YYY->RRR'),
            ActionCard.from_str('B->RRR'),
            ActionCard.from_str('RRR->BGY'),
            ActionCard.from_str('BB->YRGGG'),
            ActionCard.from_str('YYYYY->BB'),
            ActionCard.from_str('GG->YYRRR'),
            ActionCard.from_str('->YR'),
            ActionCard.from_str('B->YYYG'),
            ActionCard.from_str('G->YRR'),
            ActionCard.from_str('R->YYY'),
            ActionCard.from_str('->B'),
            ActionCard.from_str('G->YYYYR'),
            ActionCard.from_str('YYYY->GB'),
            ActionCard.from_str('B->YRG'),
            ActionCard.from_str('->G'),
            ActionCard.from_str('->GY'),
            ActionCard.from_str('RR->YYYG'),
            ActionCard.from_str('YYYY->GG'),
            ActionCard.from_str('YYYYY->GGG'),
            ActionCard.from_str('B->RRYY'),
            ActionCard.from_str('YYY->RG'),
            ActionCard.from_str('YYY->B'),
            ActionCard.from_str('GG->BB'),
            ActionCard.from_str('->YYYY'),
            ActionCard.from_str('YY->G'),
            ActionCard.from_str('GG->RRB'),
            ActionCard.from_str('GG->YYRB'),
            ActionCard.from_str('->RYY'),
            Joker(3),
            ActionCard.from_str('->RR'),
            ActionCard.from_str('GGG->BBB'),
            ActionCard.from_str('BB->RRRGG'),
            ActionCard.from_str('RRR->BB'),
            ActionCard.from_str('YY->RR'),
            ActionCard.from_str('B->GG'),
        ]
        random.shuffle(self.pile)
        self.on_cards = [Stock() for _ in range(6)]

    def copy(self, randomize=True):
        a = ActionPile(empty=True)
        a.pile = self.pile[:]
        if randomize:
            random.shuffle(self.pile[6:])
        a.on_cards = [s.copy() for s in self.on_cards]
        return a

    def visible(self)->[Tuple[ActionCard, Stock]]:
        return list(zip(self.pile[:6], self.on_cards))

    cdef list cvisible(self):
        return list(zip(self.pile[:6], self.on_cards))

    def __str__(self):
        return '\n'.join([
            f'A{i} {p[0]} {"X" * i}->{p[1]}'
            for i, p in enumerate(self.visible())
        ])

    def take(self, idx, bonus: str) -> Tuple[ActionCard, Stock]:
        if idx >= min(6, len(self.pile)):
            raise Illegal()

        if len(bonus) != idx:
            raise Illegal()

        for i, b in enumerate(str(bonus)):
            self.on_cards[i] += Stock.cfrom_str(b)

        a = self.pile.pop(idx)

        s = self.on_cards.pop(idx)
        self.on_cards += [Stock()]

        return a, s


cdef class Player:
    victory: [VictoryCard]
    cdef public list hand
    discard: [ActionCard]
    cdef public Stock stock

    def __init__(self, empty=False):
        if empty:
            return
        self.victory = []
        self.hand = []
        self.hand = [
            ActionCard.from_str('->YY'),
            Joker(2),
        ]
        self.discard = []
        self.stock = Stock()

    def copy(self):
        p = Player(empty=True)
        p.victory = self.victory[:]
        p.hand = self.hand[:]
        p.discard = self.discard[:]
        p.stock = self.stock.copy()
        return p

    def points(self):
        return self.victory_points() + self.stock.points()

    def victory_points(self):
        return sum(p.points for p in self.victory)

    cdef has_finished(self, int goal_cards):
        return len(self.victory) >= goal_cards

    def reload(self):
        self.hand += self.discard
        self.discard = []

    def play(self, idx, from_, to_):
        if idx >= len(self.hand):
            raise Illegal()

        c = self.hand[idx]
        if not c.allows(from_, to_):
            raise Illegal()
        self.stock -= from_
        self.stock += to_

        self.discard.append(c)
        del self.hand[idx]
        return 1

    def buy_victory(self, v):
        self.stock -= v.cost
        self.victory.append(v)

    def new_card(self, c):
        self.hand.append(c)

    def display(self, hidden=False):
        lines = []
        lines.append(f'V {len(self.victory)}')

        lines += ['S ' + self.stock.str()]

        if not hidden:
            for i, h in enumerate(self.hand):
                lines.append(f'H{i} {h}')

            for i, d in enumerate(self.discard):
                lines.append(f'D{i} {d}')

        return '\n'.join(lines)


cdef class Game:
    cdef public Player p0
    cdef public Player p1
    cdef public Player p2
    cdef public Player p3
    cdef public Player p4
    victory: VictoryPile
    cdef public ActionPile action
    cdef int turn
    cdef public list[str] moves
    cdef int goal_cards
    cdef public int num_players

    def __init__(self, empty=False, int goal_cards=-1, int num_players=2):
        if empty:
            return

        if goal_cards == -1:
            goal_cards = 6 if num_players <= 3 else 5

        self.goal_cards = goal_cards
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
        self.victory = VictoryPile()
        self.action = ActionPile()
        self.turn = 0
        self.moves = self.gen_move()

    def copy(self, randomize=True):
        g = Game(empty=True)
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

    cdef rank(self, int[5] ranking):
        ranking[0] = 0
        ranking[1] = 1
        ranking[2] = 2
        ranking[3] = 3
        ranking[4] = 4

        cdef int i, j
        cdef int tmp
        for i in range(self.num_players):
            for j in range(0, self.num_players - i - 1):
                if self.points_for(ranking[j + 1]) > self.points_for(ranking[j]):
                    tmp = ranking[j]
                    ranking[j] = ranking[j + 1]
                    ranking[j + 1] = tmp


    def simulate_to_end(self: Game, strategy, cut: int=30):
        cdef int i
        cdef list moves
        for i in range(cut):
            if self.ended():
                break
            distr = strategy(self)[0]
            self.play_idx(torch.multinomial(distr, 1).item())

    cpdef int current_player(self):
        return self.turn % self.num_players

    cpdef int diff_points(self):
        return self.diff_points_for(self.current_player())

    cpdef int diff_points_for(self, int me):
        cdef int[5] ranking
        assert me <= self.num_players

        if self.num_players == 1:
            return self.points_for(0)

        if self.num_players == 2:
            return (
                self.points_for(me)
                - self.points_for(1 - me)
            )

        self.rank(ranking)
        if me == ranking[0]:
            return self.points_for(ranking[0]) - self.points_for(ranking[1])
        else:
            return self.points_for(me) - self.points_for(ranking[0])

    cpdef points_for(self, int me):
        # FIXME:: this should be .points() but it seems to drive to just play with cubes?
        return self.get_player(me).victory_points()

    cpdef points(self):
        return self.points_for(self.current_player())

    def display(self, int force=-1) -> str:
        cdef int p
        if force != -1:
            p = force
        else:
            p = self.current_player()
        lines = [f'{self.turn // self.num_players:4}']

        lines.append('_Board')
        lines.append(str(self.victory))
        lines.append(str(self.action))

        for i in range(1, self.num_players):
            lines.append(f'_Him {i} {self.points_for((p + i) % self.num_players)}')
            lines.append(self.get_player((p + i) % self.num_players).display(hidden=True))

        lines.append(f'_Me {self.points_for(p)}')
        lines.append(self.get_player(p).display(hidden=False))

        return '\n'.join(lines)

    def display_with_moves(self, int force=-1) -> str:
        out = self.display(force=force) + '\n_Moves\n'
        if force == -1 or force == self.current_player():
            out += '\n'.join(['@'+ mov for mov in self.moves])
        return out

    def buy_action(self, p, idx, give, take):
        a, s = self.action.take(idx, give)
        take = Stock.cfrom_str(take)
        if take not in s:
            raise Illegal()
        p.new_card(a)
        p.stock -= Stock.cfrom_str(give)
        p.stock += take

    cpdef int play_idx(self, idx: int) except 0:
        return self.play_str(self.moves[idx])

    cpdef int play_str(self, s: str) except 0:
        p = self.get_player(self.current_player())

        if s == '':
            raise Illegal()

        if s == 'R':
            p.reload()
        elif s[0] == 'H':
            try:
                hx, action = s.split(' ')
                from_, to_ = action.split('->')
                idx = int(hx[1:])
            except:
                raise Illegal()
            p.play(idx, Stock.cfrom_str(from_), Stock.cfrom_str(to_))
        elif s[0] == 'V':
            try:
                idx = int(s[1:])
            except:
                raise Illegal()
            v = self.victory.take(idx)
            p.buy_victory(v)
        elif s[0] == 'A':
            try:
                a, bonus = s.split(' ')
                give, take = bonus.split('->')
                idx = int(a[1:])
            except:
                raise Illegal()
            self.buy_action(p, idx, give, take)
        else:
            raise Illegal()

        p.stock.trim()

        self.turn += 1

        self.moves = self.gen_move()
        return 1

    cpdef int ended(self: Game):
        cdef int i

        for i in range(self.num_players):
            if self.get_player(i).has_finished(self.goal_cards):
                return 1
        return 0

    cpdef int winner(self: Game):
        return self.max_points().index

    cpdef list[str] gen_move(self):
        cdef int i
        cdef Player p
        cdef ActionCard h, a
        cdef VictoryCard v
        cdef Stock gain
        cdef list moves

        if self.ended():
            return []

        p = self.get_player(self.current_player())

        moves = []

        if len(p.discard) > 0:
            moves.append('R')

        for i in range(5):
            if i >= len(self.victory.pile):
                continue
            v = self.victory.pile[i]
            if p.stock.contains(v.cost):
                moves.append(f'V{i}')

        for i in range(6):
            if i >= len(self.action.pile):
                continue

            a = self.action.pile[i]
            gain = self.action.on_cards[i]
            if p.stock.size() <= i:
                # Can't put cubes on previous cards
                continue
            give = p.stock.str()[:i]
            moves.append(f'A{i} {give}->{gain}')

        i = 0
        for h in p.hand:
            for m in h.gen_move(p.stock):
                x = f'H{i} {m}'
                moves.append(x)
            i += 1

        return moves

    cpdef round(self: Game):
        return self.turn // self.num_players

