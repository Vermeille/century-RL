import itertools
import random
from dataclasses import dataclass, field

from boardrl.games.splendor.data import CARDS_BY_TIER, COLORS, GOLD, NOBLES


MAX_TOKENS = 10
MAX_RESERVED = 3
PRESTIGE_TARGET = 15


@dataclass
class ReservedCard:
    card: object
    public: bool


@dataclass
class Player:
    tokens: dict = field(default_factory=lambda: {color: 0 for color in (*COLORS, GOLD)})
    purchased: list = field(default_factory=list)
    reserved: list = field(default_factory=list)
    nobles: list = field(default_factory=list)

    def bonuses(self):
        result = {color: 0 for color in COLORS}
        for card in self.purchased:
            result[card.bonus] += 1
        return result

    def prestige(self):
        return sum(card.points for card in self.purchased) + sum(
            noble.points for noble in self.nobles
        )

    def token_total(self):
        return sum(self.tokens.values())


class Splendor:
    """Base-game Splendor.

    A normal turn is one main RL action. Token returns above the ten-token
    limit and ambiguous noble visits are represented as one compact follow-up
    action each, only when required.
    """

    def __init__(self, num_players: int = 2):
        if not 2 <= num_players <= 4:
            raise ValueError("Splendor supports 2 to 4 players")
        self.num_players = num_players
        colored_supply = {2: 4, 3: 5, 4: 7}[num_players]
        self.bank = {color: colored_supply for color in COLORS}
        self.bank[GOLD] = 5
        self.players = [Player() for _ in range(num_players)]

        self.decks = {}
        self.market = {}
        for tier in (1, 2, 3):
            deck = list(CARDS_BY_TIER[tier])
            random.shuffle(deck)
            self.decks[tier] = deck
            self.market[tier] = [deck.pop() for _ in range(4)]

        nobles = list(NOBLES)
        random.shuffle(nobles)
        self.nobles = nobles[: num_players + 1]

        self.turn = 0
        self.phase = "main"
        self.final_round = False
        self._ended = False
        self._stalemate = False
        self.moves = []
        self._refresh_moves()

    def copy(self):
        game = object.__new__(Splendor)
        game.num_players = self.num_players
        game.bank = self.bank.copy()
        game.players = []
        for player in self.players:
            clone = Player(
                tokens=player.tokens.copy(),
                purchased=list(player.purchased),
                reserved=[
                    ReservedCard(reserved.card, reserved.public)
                    for reserved in player.reserved
                ],
                nobles=list(player.nobles),
            )
            game.players.append(clone)
        game.decks = {tier: list(deck) for tier, deck in self.decks.items()}
        game.market = {tier: list(row) for tier, row in self.market.items()}
        game.nobles = list(self.nobles)
        game.turn = self.turn
        game.phase = self.phase
        game.final_round = self.final_round
        game._ended = self._ended
        game._stalemate = self._stalemate
        game.moves = list(self.moves)
        return game

    def current_player(self):
        return self.turn % self.num_players

    def round(self):
        return self.turn // self.num_players

    def prestige_for(self, player):
        return self.players[player].prestige()

    def purchased_count_for(self, player):
        return len(self.players[player].purchased)

    def points_for(self, player):
        return self.prestige_for(player)

    def points(self):
        return self.points_for(self.current_player())

    def _terminal_winners(self):
        if not self._ended or self._stalemate:
            return ()
        scores = [self.prestige_for(player) for player in range(self.num_players)]
        best_score = max(scores)
        tied = [player for player, score in enumerate(scores) if score == best_score]
        fewest_cards = min(self.purchased_count_for(player) for player in tied)
        return tuple(
            player
            for player in tied
            if self.purchased_count_for(player) == fewest_cards
        )

    def winners(self):
        return self._terminal_winners()

    def winner(self):
        winners = self._terminal_winners()
        return winners[0] if len(winners) == 1 else None

    def diff_points_for(self, player):
        if self._stalemate:
            return 0
        mine = self.prestige_for(player)
        best_other = max(
            self.prestige_for(other)
            for other in range(self.num_players)
            if other != player
        )
        diff = mine - best_other
        if self._ended and diff == 0:
            winners = self._terminal_winners()
            if len(winners) == 1:
                return 0.01 if player == winners[0] else -0.01
        return diff

    def diff_points(self):
        return self.diff_points_for(self.current_player())

    def ended(self):
        return self._ended

    def stalemate(self):
        return self._stalemate

    @staticmethod
    def _card_text(card):
        cost = "".join(
            f"{color}{amount}"
            for color, amount in zip(COLORS, card.cost)
            if amount
        )
        return f"{card.bonus}{card.points}[{cost or '-'}]"

    @staticmethod
    def _noble_text(noble):
        need = "".join(
            f"{color}{amount}"
            for color, amount in zip(COLORS, noble.requirement)
            if amount
        )
        return f"N{noble.id}[{need}]"

    @staticmethod
    def _counts_text(counts, colors):
        return "".join(f"{color}{counts[color]}" for color in colors)

    def display(self, force=-1):
        viewer = self.current_player() if force == -1 else force
        if not 0 <= viewer < self.num_players:
            raise ValueError("invalid viewer")

        lines = [
            f">{self.current_player()} {self.phase} v{viewer}",
            f"Bank {self._counts_text(self.bank, (*COLORS, GOLD))}",
            "Nobles " + " ".join(self._noble_text(noble) for noble in self.nobles),
        ]
        for tier in (3, 2, 1):
            cards = []
            for slot, card in enumerate(self.market[tier]):
                cards.append(
                    f"{slot}={self._card_text(card)}" if card is not None else f"{slot}=-"
                )
            lines.append(f"T{tier}({len(self.decks[tier])}) " + " ".join(cards))

        order = [viewer] + [
            player for player in range(self.num_players) if player != viewer
        ]
        for player_index in order:
            player = self.players[player_index]
            reserved = []
            for slot, item in enumerate(player.reserved):
                if player_index == viewer or item.public:
                    value = self._card_text(item.card)
                else:
                    value = "?"
                reserved.append(f"{slot}={value}")
            lines.append(
                f"P{player_index}{'*' if player_index == self.current_player() else ''} "
                f"S{player.prestige()} D{len(player.purchased)} "
                f"T{self._counts_text(player.tokens, (*COLORS, GOLD))} "
                f"C{self._counts_text(player.bonuses(), COLORS)} "
                f"H{' '.join(reserved) if reserved else '-'}"
            )
        return "\n".join(lines) + "\n"

    def display_with_moves(self):
        return self.display() + "\n".join(f"@{move}" for move in self.moves)

    def _draw_market_replacement(self, tier, slot):
        self.market[tier][slot] = self.decks[tier].pop() if self.decks[tier] else None

    def _take_moves(self):
        available = [color for color in COLORS if self.bank[color] > 0]
        moves = []
        if available:
            take_count = min(3, len(available))
            for colors in itertools.combinations(available, take_count):
                moves.append("T:" + "".join(colors))
        for color in COLORS:
            if self.bank[color] >= 4:
                moves.append(f"T:{color}{color}")
        return moves

    def _reserve_moves(self, player):
        if len(player.reserved) >= MAX_RESERVED:
            return []
        moves = []
        for tier in (1, 2, 3):
            for slot, card in enumerate(self.market[tier]):
                if card is not None:
                    moves.append(f"R:{tier}.{slot}")
            if self.decks[tier]:
                moves.append(f"R:{tier}.D")
        return moves

    def _gold_patterns(self, player, card):
        bonuses = player.bonuses()
        needs = [
            max(card.cost[i] - bonuses[color], 0)
            for i, color in enumerate(COLORS)
        ]
        ranges = [range(need + 1) for need in needs]
        patterns = []
        for gold_by_color in itertools.product(*ranges):
            if sum(gold_by_color) > player.tokens[GOLD]:
                continue
            if any(
                needs[i] - gold_by_color[i] > player.tokens[color]
                for i, color in enumerate(COLORS)
            ):
                continue
            pattern = "".join(
                color * gold_by_color[i] for i, color in enumerate(COLORS)
            )
            patterns.append(pattern)
        return patterns

    def _buy_moves(self, player):
        moves = []
        for tier in (1, 2, 3):
            for slot, card in enumerate(self.market[tier]):
                if card is None:
                    continue
                for gold_pattern in self._gold_patterns(player, card):
                    suffix = f"~{gold_pattern}" if gold_pattern else ""
                    moves.append(f"B:{tier}.{slot}{suffix}")
        for slot, reserved in enumerate(player.reserved):
            for gold_pattern in self._gold_patterns(player, reserved.card):
                suffix = f"~{gold_pattern}" if gold_pattern else ""
                moves.append(f"B:H{slot}{suffix}")
        return moves

    def _discard_moves(self, player):
        excess = player.token_total() - MAX_TOKENS
        if excess <= 0:
            return []

        colors = (*COLORS, GOLD)
        out = []

        def visit(index, left, chosen):
            if index == len(colors):
                if left == 0:
                    out.append("D:" + "".join(chosen))
                return
            color = colors[index]
            limit = min(player.tokens[color], left)
            for count in range(limit + 1):
                visit(index + 1, left - count, chosen + [color] * count)

        visit(0, excess, [])
        return out

    def _eligible_nobles(self, player):
        bonuses = player.bonuses()
        return [
            noble
            for noble in self.nobles
            if all(
                bonuses[color] >= noble.requirement[i]
                for i, color in enumerate(COLORS)
            )
        ]

    def _main_moves(self):
        player = self.players[self.current_player()]
        return self._take_moves() + self._reserve_moves(player) + self._buy_moves(player)

    def _refresh_moves(self):
        if self._ended:
            self.moves = []
        elif self.phase == "main":
            self.moves = self._main_moves()
            if not self.moves:
                # Extremely defensive play can exhaust every colored token while
                # all players have three unbuyable reserves. The physical rules
                # then have no state-changing action; terminate the fixed point
                # as a draw rather than let RL exploit an infinite pass loop.
                self._stalemate = True
                self._ended = True
        elif self.phase == "discard":
            self.moves = self._discard_moves(self.players[self.current_player()])
        elif self.phase == "noble":
            player = self.players[self.current_player()]
            self.moves = [f"N:{noble.id}" for noble in self._eligible_nobles(player)]
        else:
            raise RuntimeError(f"unknown phase {self.phase}")

    def _after_main_action(self):
        player = self.players[self.current_player()]
        if player.token_total() > MAX_TOKENS:
            self.phase = "discard"
            self._refresh_moves()
            return
        self._after_discard()

    def _after_discard(self):
        player = self.players[self.current_player()]
        eligible = self._eligible_nobles(player)
        if len(eligible) > 1:
            self.phase = "noble"
            self._refresh_moves()
            return
        if eligible:
            noble = eligible[0]
            self.nobles.remove(noble)
            player.nobles.append(noble)
        self._finish_turn()

    def _finish_turn(self):
        player = self.current_player()
        if self.prestige_for(player) >= PRESTIGE_TARGET:
            self.final_round = True

        self.turn += 1
        self.phase = "main"
        if self.final_round and self.turn % self.num_players == 0:
            self._ended = True
            self.moves = []
            return
        self._refresh_moves()

    def _play_take(self, move):
        colors = move[2:]
        for color in colors:
            assert color in COLORS
            assert self.bank[color] > 0
        if len(colors) == 2 and colors[0] == colors[1]:
            assert self.bank[colors[0]] >= 4
        for color in colors:
            self.bank[color] -= 1
            self.players[self.current_player()].tokens[color] += 1
        self._after_main_action()

    def _play_reserve(self, move):
        ref = move[2:]
        tier_text, slot_text = ref.split(".")
        tier = int(tier_text)
        player = self.players[self.current_player()]
        assert len(player.reserved) < MAX_RESERVED

        if slot_text == "D":
            assert self.decks[tier]
            card = self.decks[tier].pop()
            public = False
        else:
            slot = int(slot_text)
            card = self.market[tier][slot]
            assert card is not None
            public = True
            self._draw_market_replacement(tier, slot)

        player.reserved.append(ReservedCard(card, public))
        if self.bank[GOLD] > 0:
            self.bank[GOLD] -= 1
            player.tokens[GOLD] += 1
        self._after_main_action()

    def _parse_buy(self, move):
        body = move[2:]
        ref, *gold = body.split("~", 1)
        return ref, gold[0] if gold else ""

    def _play_buy(self, move):
        ref, gold_pattern = self._parse_buy(move)
        player = self.players[self.current_player()]

        if ref.startswith("H"):
            slot = int(ref[1:])
            card = player.reserved[slot].card
            market_ref = None
        else:
            tier_text, slot_text = ref.split(".")
            tier = int(tier_text)
            slot = int(slot_text)
            card = self.market[tier][slot]
            assert card is not None
            market_ref = (tier, slot)

        plans = self._gold_patterns(player, card)
        assert gold_pattern in plans

        bonuses = player.bonuses()
        gold_by_color = {color: gold_pattern.count(color) for color in COLORS}
        for i, color in enumerate(COLORS):
            need = max(card.cost[i] - bonuses[color], 0)
            colored_spent = need - gold_by_color[color]
            player.tokens[color] -= colored_spent
            self.bank[color] += colored_spent
        gold_spent = len(gold_pattern)
        player.tokens[GOLD] -= gold_spent
        self.bank[GOLD] += gold_spent

        if market_ref is None:
            player.reserved.pop(slot)
        else:
            tier, slot = market_ref
            self._draw_market_replacement(tier, slot)
        player.purchased.append(card)
        self._after_main_action()

    def _play_discard(self, move):
        player = self.players[self.current_player()]
        colors = move[2:]
        assert len(colors) == player.token_total() - MAX_TOKENS
        for color in colors:
            assert player.tokens[color] > 0
            player.tokens[color] -= 1
            self.bank[color] += 1
        assert player.token_total() == MAX_TOKENS
        self._after_discard()

    def _play_noble(self, move):
        noble_id = int(move[2:])
        eligible = self._eligible_nobles(self.players[self.current_player()])
        noble = next(noble for noble in eligible if noble.id == noble_id)
        self.nobles.remove(noble)
        self.players[self.current_player()].nobles.append(noble)
        self._finish_turn()

    def play_str(self, move):
        assert not self._ended
        assert move in self.moves
        if self.phase == "discard":
            self._play_discard(move)
        elif self.phase == "noble":
            self._play_noble(move)
        elif move.startswith("T:"):
            self._play_take(move)
        elif move.startswith("R:"):
            self._play_reserve(move)
        elif move.startswith("B:"):
            self._play_buy(move)
        else:
            raise AssertionError(f"unknown move {move}")

    def play_idx(self, index):
        self.play_str(self.moves[index])

    def simulate_to_end(self, max_steps=2000):
        for _ in range(max_steps):
            if self.ended():
                return
            self.play_str(random.choice(self.moves))
        raise RuntimeError("Splendor random playout did not terminate")
