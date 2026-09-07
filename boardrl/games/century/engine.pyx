# cython: profile=False
# cython: language_level=3
# cython: binding=True
# cython: linetrace=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import torch
cimport cython
import copy
import random
from typing import Tuple, List
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.stdio cimport sprintf
from cpython.unicode cimport PyUnicode_DecodeLatin1


cpdef random_buy_fast(Century g):
    moves = g.moves
    for mov in moves:
        if mov[0] == "V":
            return mov

    return random.choice(moves)


class Illegal(Exception):
    pass


@cython.profile(False)
cdef Stock make_stock():
    cdef Stock s = Stock()
    s.Y = 0
    s.R = 0
    s.G = 0
    s.B = 0
    return s


@cython.final
cdef class Stock:
    """Compact cube stock for Century.

    Fields Y, R, G, B store counts for yellow, red, green, blue.
    Provides helpers to add/subtract, prefix selection and string forms.
    """
    cdef int Y
    cdef int R
    cdef int G
    cdef int B

    cpdef str to_str(self):
        if self.Y + self.R + self.G + self.B == 0:
            return ""

        cdef char buff[32]
        cdef int pos = 0

        if self.Y > 0:
            if self.Y > 1:
                pos += sprintf(buff + pos, "%d", self.Y)
            buff[pos] = 'Y'
            pos += 1
        if self.R > 0:
            if self.R > 1:
                pos += sprintf(buff + pos, "%d", self.R)
            buff[pos] = 'R'
            pos += 1
        if self.G > 0:
            if self.G > 1:
                pos += sprintf(buff + pos, "%d", self.G)
            buff[pos] = 'G'
            pos += 1
        if self.B > 0:
            if self.B > 1:
                pos += sprintf(buff + pos, "%d", self.B)
            buff[pos] = 'B'
            pos += 1

        return PyUnicode_DecodeLatin1(buff, pos, NULL)

    @cython.profile(False)
    cpdef inline Stock ccopy(self):
        o = make_stock()
        Stock.iadd(o, self)
        return o

    @staticmethod
    cdef inline Stock cfrom_str_(s: str):
        cdef Py_UCS4 c
        cdef Stock stock
        stock = make_stock()
        for c in s:
            if c not in 'YRGB':
                raise Illegal()

            if c == 'Y':
                stock.Y += 1
            elif c == 'R':
                stock.R += 1
            elif c == 'G':
                stock.G += 1
            elif c == 'B':
                stock.B += 1
        return stock

    @staticmethod
    cdef inline Stock cfrom_str(s: str):
        cdef Stock stock = make_stock()
        cdef str num_buffer = ''
        cdef int i, count
        cdef int length = len(s)

        i = 0
        while i < length:
            if s[i].isdigit():
                num_buffer += s[i]
                i += 1
                continue

            count = int(num_buffer) if num_buffer else 1
            num_buffer = ''

            if s[i] not in 'YRGB':
                raise Illegal()

            if s[i] == 'Y':
                stock.Y += count
            elif s[i] == 'R':
                stock.R += count
            elif s[i] == 'G':
                stock.G += count
            elif s[i] == 'B':
                stock.B += count

            i += 1

        return stock

    def __iter__(self):
        for i in range(self.Y):
            yield 'Y'
        for i in range(self.R):
            yield 'R'
        for i in range(self.G):
            yield 'G'
        for i in range(self.B):
            yield 'B'

    @cython.profile(False)
    cpdef inline int contains(Stock self, Stock ref) nogil:
        return (self.Y >= ref.Y and self.R >= ref.R and self.G >= ref.G
                and self.B >= ref.B)

    @cython.profile(False)
    cdef inline int size(self) noexcept nogil:
        return self.Y + self.R + self.G + self.B

    cpdef int weighted_value(self):
        return self.Y + 2 * self.R + 3 * self.G + 4 * self.B

    @cython.profile(False)
    cdef inline Stock sub(self, ref: Stock):
        if not self.contains(ref):
            raise Illegal()
        cdef Stock out
        out = make_stock()
        out.Y = self.Y - ref.Y
        out.R = self.R - ref.R
        out.G = self.G - ref.G
        out.B = self.B - ref.B
        return out

    @cython.profile(False)
    cdef inline Stock add(self, ref: Stock):
        cdef Stock out
        out = make_stock()
        out.Y = self.Y + ref.Y
        out.R = self.R + ref.R
        out.G = self.G + ref.G
        out.B = self.B + ref.B
        return out

    @cython.profile(False)
    cdef void iadd(self: Stock, ref: Stock) nogil:
        self.Y += ref.Y
        self.R += ref.R
        self.G += ref.G
        self.B += ref.B

    @cython.profile(False)
    cdef int isub(self, ref: Stock) except 0:
        if not self.contains(ref):
            raise Illegal()
        self.Y -= ref.Y
        self.R -= ref.R
        self.G -= ref.G
        self.B -= ref.B
        return 1

    def __iadd__(self, ref: Stock):
        Stock.iadd(self, ref)
        return self

    def __isub__(self, ref: Stock):
        Stock.isub(self, ref)
        return self

    @cython.profile(False)
    cdef int points(self):
        return self.R + self.G + self.B

    cdef void trim(self) nogil:
        """Reduce total cubes to 10 by removing from the largest color."""
        cdef int idx
        while self.size() > 10:
            idx = 0  # 0:Y, 1:R, 2:G, 3:B
            if self.R > (self.Y if idx == 0 else self.R):
                idx = 1
            if self.G > (self.R if idx == 1 else (self.Y if idx == 0 else self.G)):
                idx = 2
            if self.B > (self.G if idx == 2 else (self.R if idx == 1 else self.Y)):
                idx = 3

            if idx == 0:
                self.Y -= 1
            elif idx == 1:
                self.R -= 1
            elif idx == 2:
                self.G -= 1
            else:
                self.B -= 1

    @cython.profile(False)
    cdef inline Stock _prefix(self, int n):
        """Return a Stock with the first ``n`` cubes in Y->R->G->B order."""
        cdef Stock out = make_stock()
        cdef int take

        if n <= 0:
            return out

        take = self.Y if self.Y < n else n
        out.Y = take
        n -= take
        if n == 0:
            return out

        take = self.R if self.R < n else n
        out.R = take
        n -= take
        if n == 0:
            return out

        take = self.G if self.G < n else n
        out.G = take
        n -= take
        if n == 0:
            return out

        take = self.B if self.B < n else n
        out.B = take
        return out

    cpdef Stock prefix(self, int n):
        """Return a Stock with the first n cubes in Y->R->G->B order."""
        return self._prefix(n)

@cython.profile(False)
cdef inline void prefix_into_stock(Stock src, int n, Stock out):
    out.Y = 0
    out.R = 0
    out.G = 0
    out.B = 0

    if n <= 0:
        return

    cdef int take
    take = src.Y if src.Y < n else n
    out.Y = take
    n -= take
    if n == 0:
        return

    take = src.R if src.R < n else n
    out.R = take
    n -= take
    if n == 0:
        return

    take = src.G if src.G < n else n
    out.G = take
    n -= take
    if n == 0:
        return

    take = src.B if src.B < n else n
    out.B = take


cdef class ActionCard:
    """Action card transforming cubes: from_ -> to_.

    gen_move returns strings like 'YY>R' or 'YYRR>RGBB' for multiples.
    """
    cdef Stock from_
    cdef Stock to_
    cdef list str_cache

    def __init__(self, from_, to_):
        self.from_ = (from_
                      if isinstance(from_, Stock) else Stock.cfrom_str(from_))
        self.to_ = (to_ if isinstance(to_, Stock) else Stock.cfrom_str(to_))
        self.str_cache = [self.from_.to_str() + '>' + self.to_.to_str()]
        if self.from_.size() > 0:
            needed = self.from_.ccopy()
            gen = self.to_.ccopy()
            while needed.size() <= 10:
                self.str_cache.append(f'{needed.to_str()}>{gen.to_str()}')
                Stock.iadd(needed, self.from_)
                Stock.iadd(gen, self.to_)

    def __str__(self):
        return self.str_cache[0]

    cpdef takes(self):
        return self.from_

    cpdef gives(self):
        return self.to_

    __repr__ = __str__

    @staticmethod
    cdef from_str(s):
        f, t = s.split('>')
        return ActionCard(f, t)

    cpdef list gen_move(self, Stock stock):
        """Enumerate all valid multiples of this action for a given stock.

        Returns move strings from an internal cache for speed and clarity.
        """
        cdef list moves
        cdef int k, t
        # If the action is free (no cost), it can always be played once.
        if self.from_.size() == 0:
            return [self.str_cache[0]]

        # Compute the maximum number of times this action can be applied given the stock,
        # using integer division per color. k is upper-bounded by available cubes for each
        # required color (colors with 0 requirement are ignored).
        k = 1_000_000  # large sentinel; real counts are tiny
        if self.from_.Y:
            t = stock.Y // self.from_.Y
            if t < k:
                k = t
        if self.from_.R:
            t = stock.R // self.from_.R
            if t < k:
                k = t
        if self.from_.G:
            t = stock.G // self.from_.G
            if t < k:
                k = t
        if self.from_.B:
            t = stock.B // self.from_.B
            if t < k:
                k = t

        if k <= 0:
            return []

        # str_cache[1] corresponds to 1x application, up to precomputed bound.
        # The cache is built up to a safe limit; append desired range.
        # Slice cache from 1..k (inclusive) and return a new list
        t = min(k + 1, len(self.str_cache))
        return list(self.str_cache[1:t])

    cpdef allows(self, from_: Stock, to_: Stock):
        """Check if repeatedly applying the card maps from_ to a superset of to_."""
        if self.from_.size() == 0:
            return self.to_.contains(to_)

        cdef Stock remain = from_.ccopy()
        cdef Stock gen = make_stock()
        while remain.contains(self.from_):
            Stock.isub(remain, self.from_)
            Stock.iadd(gen, self.to_)

        return gen.contains(to_)



cdef class VictoryCard:
    cdef public int points
    cdef public Stock cost

    def __init__(self, cost, points):
        self.points = points
        self.cost = (cost if isinstance(cost, Stock) else Stock.cfrom_str(cost))

    def __str__(self):
        return self.cost.to_str() + '>' + str(self.points)

    __repr__ = __str__

    @staticmethod
    cdef from_str(s):
        c, p = s.split('>')
        return VictoryCard(c, int(p))


joker2_moves = [
    'Y>R',
    'R>G',
    'G>B',

    'YY>RR',
    'YR>RG',
    'YG>RB',

    # 'RY>GR', Already covered by YR>RG
    'RR>GG',
    'RG>GB',

    # 'GY>BR', Already covered by YG>RB
    # 'GR>BG', Already covered by RG>GB
    'GG>BB',

    'Y>G',
    'R>B',
]

joker3_moves = joker2_moves + [
    # Y>R
    'YYY>RRR',
    'YYR>RRG',
    'YYG>RRB',

    'YRR>RGG',
    'YRG>RGB',

    'YGG>RBB',

    'YY>RG',
    'YR>RB',

    # R>G
    'RRR>GGG',
    'RRG>GGB',

    'RGG>GBB',

    'YR>GG',
    'RR>GB',

    # G>B
    'GGG>BBB',

    'YG>GB',
    'RG>BB',

    #
    'Y>B',
]

cdef class Joker(ActionCard):
    cdef int n
    cdef list instances

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

    cpdef allows(self, from_: Stock, to_: Stock):
        for ins in self.instances:
            if ins.allows(from_, to_):
                return True
        return False

    cpdef list gen_move(self, Stock stock):
        cdef list moves = []
        for ins in self.instances:
            if stock.contains(ins.takes()):
                moves.append(str(ins))
        return moves


cdef class VictoryPile:
    cdef public list pile

    def __init__(self, empty=False):
        if empty:
            return
        self.pile = [
            VictoryCard.from_str('YRGB>12'),
            VictoryCard.from_str('YRGGGB>18'),
            VictoryCard.from_str('YGGB>12'),
            VictoryCard.from_str('GGBBB>18'),
            VictoryCard.from_str('GGGGG>15'),
            VictoryCard.from_str('YYBBB>14'),
            VictoryCard.from_str('YYRB>9'),
            VictoryCard.from_str('YYGGG>11'),
            VictoryCard.from_str('RRGGBB>19'),
            VictoryCard.from_str('RRRGG>12'),
            VictoryCard.from_str('RRGG>10'),
            VictoryCard.from_str('BBBB>16'),
            VictoryCard.from_str('RRRR>8'),
            VictoryCard.from_str('RRRRR>10'),
            VictoryCard.from_str('YYRR>6'),
            VictoryCard.from_str('YYGGBB>17'),
            VictoryCard.from_str('YYBB>10'),
            VictoryCard.from_str('RRRBB>14'),
            VictoryCard.from_str('YRRRGB>16'),
            VictoryCard.from_str('YYGG>8'),
            VictoryCard.from_str('RRGB>12'),
            VictoryCard.from_str('GGBB>14'),
            VictoryCard.from_str('GGGBB>17'),
            VictoryCard.from_str('RRGGG>13'),
            VictoryCard.from_str('GGGG>12'),
            VictoryCard.from_str('YRGBBB>20'),
            VictoryCard.from_str('RRBBB>16'),
            VictoryCard.from_str('YYYRR>7'),
            VictoryCard.from_str('YYYGG>9'),
            VictoryCard.from_str('YYRRGG>13'),
            VictoryCard.from_str('YYRRR>8'),
            VictoryCard.from_str('YYYRGB>14'),
            VictoryCard.from_str('RRBB>12'),
            VictoryCard.from_str('YYRRBB>15'),
            VictoryCard.from_str('YYYBB>11'),
            VictoryCard.from_str('BBBBB>20'),
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
            ActionCard.from_str('RRR>GGYY'),
            ActionCard.from_str('RR>BYY'),
            ActionCard.from_str('RRR>GGG'),
            ActionCard.from_str('RRR>GGG'),
            ActionCard.from_str('RR>GG'),
            ActionCard.from_str('YR>B'),
            ActionCard.from_str('>YYY'),
            ActionCard.from_str('YYG>BB'),
            ActionCard.from_str('G>RR'),
            ActionCard.from_str('YYY>RRR'),
            ActionCard.from_str('B>RRR'),
            ActionCard.from_str('RRR>BGY'),
            ActionCard.from_str('BB>YRGGG'),
            ActionCard.from_str('YYYYY>BB'),
            ActionCard.from_str('GG>YYRRR'),
            ActionCard.from_str('>YR'),
            ActionCard.from_str('B>YYYG'),
            ActionCard.from_str('G>YRR'),
            ActionCard.from_str('R>YYY'),
            ActionCard.from_str('>B'),
            ActionCard.from_str('G>YYYYR'),
            ActionCard.from_str('YYYY>GB'),
            ActionCard.from_str('B>YRG'),
            ActionCard.from_str('>G'),
            ActionCard.from_str('>GY'),
            ActionCard.from_str('RR>YYYG'),
            ActionCard.from_str('YYYY>GG'),
            ActionCard.from_str('YYYYY>GGG'),
            ActionCard.from_str('B>RRYY'),
            ActionCard.from_str('YYY>RG'),
            ActionCard.from_str('YYY>B'),
            ActionCard.from_str('GG>BB'),
            ActionCard.from_str('>YYYY'),
            ActionCard.from_str('YY>G'),
            ActionCard.from_str('GG>RRB'),
            ActionCard.from_str('GG>YYRB'),
            ActionCard.from_str('>RYY'),
            #Joker(3),  # FIXME: this card gnenerates too many moves so we mute it for now
            ActionCard.from_str('>RR'),
            ActionCard.from_str('GGG>BBB'),
            ActionCard.from_str('BB>RRRGG'),
            ActionCard.from_str('RRR>BB'),
            ActionCard.from_str('YY>RR'),
            ActionCard.from_str('B>GG'),
        ]
        random.shuffle(self.pile)
        self.on_cards = [make_stock() for _ in range(6)]

    def copy(self, randomize=True):
        a = ActionPile(empty=True)
        a.pile = self.pile[:]
        if randomize:
            random.shuffle(self.pile[6:])
        a.on_cards = [s.ccopy() for s in self.on_cards]
        return a

    def visible(self)->[Tuple[ActionCard, Stock]]:
        return list(zip(self.pile[:6], self.on_cards))

    cdef list cvisible(self):
        return list(zip(self.pile[:6], self.on_cards))

    def __str__(self):
        return '\n'.join(
            [f'A{i} {p[0]} {"X" * i}>{p[1].to_str()}' for i, p in enumerate(self.visible())]
        )

    cpdef Tuple[ActionCard, Stock] take(self, int idx, Stock bonus):
        if idx >= min(6, len(self.pile)):
            raise Illegal()

        if bonus.size() != idx:
            raise Illegal()

        # Distribute one cube over each previous card according to the
        # Y->R->G->B order encoded in bonus without constructing temporary Stocks.
        cdef int i
        cdef int y = bonus.Y
        cdef int r = bonus.R
        cdef int g = bonus.G
        cdef int b = bonus.B
        cdef Stock s_i
        for i in range(idx):
            s_i = self.on_cards[i]
            if y:
                s_i.Y += 1
                y -= 1
            elif r:
                s_i.R += 1
                r -= 1
            elif g:
                s_i.G += 1
                g -= 1
            elif b:
                s_i.B += 1
                b -= 1

        a = self.pile.pop(idx)

        s = self.on_cards.pop(idx)
        self.on_cards += [make_stock()]

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
            ActionCard.from_str('>YY'),
            Joker(2),
        ]
        self.discard = []
        self.stock = make_stock()

    def copy(self):
        p = Player(empty=True)
        p.victory = self.victory[:]
        p.hand = self.hand[:]
        p.discard = self.discard[:]
        p.stock = self.stock.ccopy()
        return p

    def points(self):
        return self.victory_points() + self.stock.points()

    def victory_points(self):
        return sum(p.points for p in self.victory)

    cpdef int victory_count(self):
        return len(self.victory)

    cpdef int discard_count(self):
        return len(self.discard)

    cdef has_finished(self, int goal_cards):
        return len(self.victory) >= goal_cards

    def reload(self):
        self.hand += self.discard
        self.discard = []

    cpdef int play(self, int idx, Stock from_, Stock to_) except 0:
        if idx >= len(self.hand):
            raise Illegal()

        c = self.hand[idx]
        if not c.allows(from_, to_):
            raise Illegal()
        self.stock -= from_
        self.stock.iadd(to_)

        self.discard.append(c)
        del self.hand[idx]
        return 1

    def buy_victory(self, v):
        self.stock -= v.cost
        self.victory.append(v)

    def new_card(self, c):
        self.hand.append(c)

    cpdef display(self, int hidden=False):
        cdef list lines = []
        cdef int i
        cdef ActionCard h, d
        lines.append(f'V {len(self.victory)}')

        lines += ['S ' + self.stock.to_str()]

        if not hidden:
            for i, h in enumerate(self.hand):
                lines.append(f'H{i} {h}')

            for i, d in enumerate(self.discard):
                lines.append(f'D{i} {d}')

        return '\n'.join(lines)


cdef class Century:
    """Century environment.

    Manages players, piles, turn counter and legal move generation/playing.
    Moves are represented as strings like 'R' (reload), 'Vi', 'Ai X>Y', 'Hi X>Y'.
    """
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
        assert 0 < num_players <= 5
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

    @cython.profile(True)
    def copy(self, randomize=True):
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

    cpdef list visible_victory(self):
        return self.victory.visible()

    cpdef int goal_card_count(self):
        return self.goal_cards

    cpdef Stock preview_stock(self, str s):
        """Return post-move stock without mutating state or revealing new cards."""
        cdef Player p = self.get_player(self.current_player())
        cdef Stock out = p.stock.ccopy()
        cdef Stock from_stock, to_stock
        cdef VictoryCard v
        cdef str hx, action, from_, to_, a, bonus, give, take
        cdef int idx

        if s not in self.moves:
            raise Illegal()
        if s == 'R':
            return out
        if s[0] == 'H':
            hx, action = s.split(' ')
            from_, to_ = action.split('>')
            from_stock = Stock.cfrom_str(from_)
            to_stock = Stock.cfrom_str(to_)
            Stock.isub(out, from_stock)
            Stock.iadd(out, to_stock)
        elif s[0] == 'V':
            idx = int(s[1:])
            v = self.victory.pile[idx]
            Stock.isub(out, v.cost)
        elif s[0] == 'A':
            a, bonus = s.split(' ')
            give, take = bonus.split('>')
            from_stock = Stock.cfrom_str(give)
            to_stock = Stock.cfrom_str(take)
            Stock.isub(out, from_stock)
            Stock.iadd(out, to_stock)
        else:
            raise Illegal()

        out.trim()
        return out

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


    def simulate_to_end(self: Century, cut: int=30):
        cdef int i
        cdef list moves
        for i in range(cut):
            if self.ended():
                break
            action = random_buy_fast(self)
            self.play_str(action)

    cpdef int current_player(self) noexcept:
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
        lines = [f'{self.current_player()} {self.round():4}']

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
        assert force == -1 or force == self.current_player()
        out += '\n'.join(['@' + mov for mov in self.moves])
        return out

    cpdef void buy_action(self, Player p, int idx, Stock give, Stock take):
        cdef ActionCard  a
        cdef Stock s
        a, s = self.action.take(idx, give)
        if not s.contains(take):
            raise Illegal()
        p.new_card(a)
        p.stock -= give
        p.stock.iadd(take)

    cpdef int play_idx(self, idx: int) except 0:
        return self.play_str(self.moves[idx])

    cpdef int play_str(self, s: str) except 0:
        """Apply a move string to the current game state.

        Accepts 'R', 'V{i}', 'A{i} GIVE>TAKE', 'H{i} FROM>TO'.
        """
        p = self.get_player(self.current_player())

        if s == '':
            raise Illegal()

        if s == 'R':
            p.reload()
        elif s[0] == 'H':
            try:
                hx, action = s.split(' ')
                from_, to_ = action.split('>')
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
                give, take = bonus.split('>')
                idx = int(a[1:])
            except:
                raise Illegal()
            self.buy_action(p, idx, Stock.cfrom_str(give), Stock.cfrom_str(take))
        else:
            raise Illegal()

        p.stock.trim()

        self.turn += 1

        self.moves = self.gen_move()
        return 1

    cpdef int ended(self: Century):
        cdef int i

        for i in range(self.num_players):
            if self.get_player(i).has_finished(self.goal_cards):
                return 1
        return 0

    cpdef list[str] gen_move(self):
        """Generate all legal moves for the current player as strings."""
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

        cdef int len_v = len(self.victory.pile)
        for i in range(min(5, len_v)):
            v = self.victory.pile[i]
            if p.stock.contains(v.cost):
                moves.append(f'V{i}')

        cdef Stock give_tmp = make_stock()
        cdef int len_a = len(self.action.pile)
        for i in range(min(6, len_a)):

            a = self.action.pile[i]
            gain = self.action.on_cards[i]
            if p.stock.size() < i:
                # Can't put cubes on previous cards
                continue
            prefix_into_stock(p.stock, i, give_tmp)
            moves.append(f'A{i} {give_tmp.to_str()}>{gain.to_str()}')

        i = 0
        for h in p.hand:
            for m in h.gen_move(p.stock):
                x = f'H{i} {m}'
                moves.append(x)
            i += 1

        return moves

    cpdef round(self: Century):
        return self.turn // self.num_players
