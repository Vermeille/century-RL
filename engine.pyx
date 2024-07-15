#cython: language_level=3, profile=True
import torch
import cython
from cpython cimport array
import array
import copy
import random
from random import choice as rndchoice
from typing import Tuple, List
from libc.math cimport sqrt, log

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

    def __str__(self) -> str:
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


    def __contains__(self, ref: 'Stock'):
        return (self.Y >= ref.Y and self.R >= ref.R and self.G >= ref.G
                and self.B >= ref.B)

    cdef int size(self) nogil:
        return self.Y + self.R + self.G + self.B

    def __len__(self) -> int:
        return self.Y + self.R + self.G + self.B

    def __sub__(self, ref: 'Stock'):
        return Stock.sub(self, ref)

    cdef inline Stock sub(self, ref: 'Stock'):
        if not ref in self:
            raise Illegal()
        out = Stock()
        out.Y = self.Y - ref.Y
        out.R = self.R - ref.R
        out.G = self.G - ref.G
        out.B = self.B - ref.B
        return out

    cdef Stock add(self, ref: 'Stock'):
        out = Stock()
        out.Y = self.Y + ref.Y
        out.R = self.R + ref.R
        out.G = self.G + ref.G
        out.B = self.B + ref.B
        return out

    def __add__(self, ref: 'Stock'):
        return Stock.add(self, ref)

    cdef void iadd(self: 'Stock', ref: 'Stock') nogil:
        self.Y += ref.Y
        self.R += ref.R
        self.G += ref.G
        self.B += ref.B

    def __iadd__(self, ref: 'Stock'):
        Stock.iadd(self, ref)
        return self

    cdef int isub(self, ref: 'Stock') except 0:
        if not ref in self:
            raise Illegal()
        self.Y -= ref.Y
        self.R -= ref.R
        self.G -= ref.G
        self.B -= ref.B
        return 1

    def __isub__(self, ref: 'Stock'):
        Stock.isub(self, ref)
        return self

    def points(self):
        return self.R + self.G + self.B

    def trim(self):
        while len(self) > 10:
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
        self.str_cache = [str(self.from_) + '->' + str(self.to_)]
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
        while needed in stock:
            yield self.str_cache[i]
            Stock.iadd(needed, self.from_)
            i += 1

    def allows(self, from_: Stock, to_: Stock):
        if self.from_.size() == 0:
            return to_ in self.to_

        cdef Stock gen_
        from_ = from_.ccopy()
        gen = Stock()
        while self.from_ in from_:
            Stock.isub(from_, self.from_)
            Stock.iadd(gen, self.to_)

        return to_ in gen



cdef class VictoryCard:
    cdef public int points
    cdef public Stock cost

    def __init__(self, cost, points):
        self.points = points
        self.cost = (cost if isinstance(cost, Stock) else Stock.cfrom_str(cost))

    def __str__(self):
        return str(self.cost) + '->' + str(self.points)

    __repr__ = __str__

    @staticmethod
    def from_str(s):
        c, p = s.split('->')
        return VictoryCard(c, int(p))


class Joker(ActionCard):

    def __init__(self, n):
        assert n <= 3
        self.n = n
        self.instances = [
            ActionCard.from_str('Y->R'),
            ActionCard.from_str('R->G'),
            ActionCard.from_str('G->B'),
        ]
        start = 0
        for i in range(2, n + 1):
            new = []
            for ins in self.instances[start:]:
                for base in self.instances[:3]:
                    # parallel upgrades
                    new.append(
                        ActionCard(ins.takes() + base.takes(),
                                   ins.gives() + base.gives()))
            if i == 2:
                new.append(ActionCard.from_str('Y->G'))
                new.append(ActionCard.from_str('R->B'))
            if i == 3:
                new.append(ActionCard.from_str('Y->B'))
            start = len(self.instances)
            self.instances += new
            n -= 1

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

    def __init__(self):
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

    def copy(self):
        v = copy.copy(self)
        v.pile = self.pile[:]
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

    def __init__(self):
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

    def copy(self):
        a = copy.copy(self)
        a.pile = self.pile[:]
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

    def take(self, idx, bonus: Stock()) -> Tuple[ActionCard, Stock]:
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
        return sum(p.points for p in self.victory) + self.stock.points()

    def has_finished(self):
        return len(self.victory) >= 5

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

        lines += ['S ' + str(self.stock)]

        if not hidden:
            for i, h in enumerate(self.hand):
                lines.append(f'H{i} {h}')

            for i, d in enumerate(self.discard):
                lines.append(f'D{i} {d}')

        return '\n'.join(lines)


cdef enum State:
    P0_TURN = 0
    P1_TURN = 1
    ENDED = 2
    FAILED = 3

cdef class Game:
    cdef public Player p0
    cdef public Player p1
    victory: VictoryPile
    cdef public ActionPile action
    cdef public State state
    cdef int turn

    def __init__(self, empty=False):
        if empty:
            return
        self.p0 = Player()
        self.p0.stock += Stock.cfrom_str('YYY')
        self.p1 = Player()
        self.p1.stock += Stock.cfrom_str('YYYY')

        self.victory = VictoryPile()
        self.action = ActionPile()
        self.state = State.P0_TURN
        self.turn = 0

    def copy(self):
        g = Game(empty=True)
        g.p0 = self.p0.copy()
        g.p1 = self.p1.copy()
        g.victory = self.victory.copy()
        g.action = self.action.copy()
        g.state = self.state
        g.turn = self.turn
        return g

    def simulate_to_end(self: 'Game', cut: int=30) -> int:
        cdef int i
        cdef list moves
        for i in range(cut):
            moves = self.gen_move()
            self.play_str(rndchoice(moves))
            if self.ended():
                break
        return 0 if self.p0.points() > self.p1.points() else 1

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def gen_good_move(self, int num_sims=1000, list propals=[]):
        cdef i
        cdef int idx
        cdef int n_moves
        cdef str move
        cdef list moves
        cdef Game g2
        cdef int winner
        cdef int[:] points
        cdef int[:] sims
        cdef int best_idx
        cdef float best_score
        cdef float score
        moves = propals + self.gen_move()
        n_moves = len(moves)
        points = array.array('i', [0] * n_moves)
        sims = array.array('i', [0] * n_moves)
        for idx in range(n_moves):
            for i in range(num_sims):
                move = moves[idx]
                try:
                    g2 = self.copy()
                    g2.play_str(move)
                except Illegal:
                    points[idx] = -1000000
                    continue
                winner = g2.simulate_to_end()
                p = abs(g2.p0.points() - g2.p1.points())
                points[idx] += (1 if winner == self.state else -1) * p
                sims[idx] += 1

        best_idx = 0
        best_score = -2000000
        for idx in range(0, n_moves):
            if sims[idx] == 0:
                continue
            score = float(points[idx]) / sims[idx]# + sqrt(2*log(num_sims)/sims[idx])
            if score > best_score:
                best_idx = idx
                best_score = score
        #print(moves, list(points), list(sims), moves[best_idx])
        return moves[best_idx]

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def gen_neural_move(self, nn, list propals=[], float T=1):
        cdef i
        cdef int idx
        cdef int n_moves
        cdef str move
        cdef list moves
        cdef Game g2
        cdef int winner
        cdef int best_idx
        cdef float best_score
        cdef float score
        moves = propals + self.gen_move()
        n_moves = len(moves)
        prompts = []
        for idx in range(n_moves):
            move = moves[idx]
            try:
                player = self.state
                g2 = self.copy()
                g2.play_str(move)
                prompts.append(g2.display(player))
            except Illegal:
                prompts.append('')

        pred_r = nn(prompts)[1].cpu() * T
        for idx in range(n_moves):
            if prompts[idx] == '':
                pred_r[idx] = -1000
            #print(idx, pred_r[idx].item())
            #print(prompts[idx])
            #print('====')

        probs = torch.nn.functional.softmax(pred_r, 0)
        return moves[torch.multinomial(probs, 1)], list(zip(moves,
            pred_r.tolist(), probs.tolist()))

    def display(self, force=-1):
        if force != -1:
            p = force
        else:
            if self.state == State.P0_TURN:
                p = 0
            else:
                p = 1
        lines = [f'{self.turn:4}']
        if p == 0:
            lines.append(f'_Me {self.p0.points()}')
            lines.append(self.p0.display(hidden=False))
            lines.append(f'_Him {self.p1.points()}')
            lines.append(self.p1.display(hidden=True))
        elif p == 1:
            lines.append(f'_Me {self.p1.points()}')
            lines.append(self.p1.display(hidden=False))
            lines.append(f'_Him {self.p0.points()}')
            lines.append(self.p0.display(hidden=True))
        else:
            assert False, f"can't display the game for state {self.state}"
        lines.append('_Board')
        lines.append(str(self.victory))
        lines.append(str(self.action))
        return '\n'.join(lines)

    def buy_action(self, p, idx, give, take):
        a, s = self.action.take(idx, give)
        take = Stock.cfrom_str(take)
        if take not in s:
            raise Illegal()
        p.new_card(a)
        p.stock -= Stock.cfrom_str(give)
        p.stock += take

    cpdef int play_str(self, s: str) except 0:
        if self.state == State.P0_TURN:
            p = self.p0
        elif self.state == State.P1_TURN:
            p = self.p1
        else:
            assert False, f"can't play for a game in state {self.state}"

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

        if self.ended():
            self.state = State.ENDED

        if self.state == State.P0_TURN:
            self.state = State.P1_TURN
        elif self.state == State.P1_TURN:
            self.state = State.P0_TURN
        self.turn += 1

        return 1

    cpdef int ended(self: 'Game'):
        return self.p0.has_finished() or self.p1.has_finished()

    cpdef gen_move(self):
        cdef int i
        cdef Player p
        cdef ActionCard h, a
        cdef VictoryCard v
        cdef Stock gain
        cdef list moves
        if self.state == State.P0_TURN:
            p = self.p0
        elif self.state == State.P1_TURN:
            p = self.p1
        else:
            assert False, f"can't generate a move for a gam in state {self.state}"

        moves = []

        if len(p.discard) > 0 or True:
            moves.append('R')

        i = 0
        for h in p.hand:
            for m in h.gen_move(p.stock):
                x = f'H{i} {m}'
                moves.append(x)
            i += 1


        for i in range(min(5, len(self.victory.pile))):
            v = self.victory.pile[i]
            if v.cost in p.stock:
                moves.append(f'V{i}')

        for i in range(min(6, len(self.action.pile))):
            a = self.action.pile[i]
            gain = self.action.on_cards[i]
            if p.stock.size() <= i:
                continue
            #give = ''.join(random.sample(str(p.stock), i))
            give = str(p.stock)[:i]
            moves.append(f'A{i} {give}->{gain}')

        return moves


