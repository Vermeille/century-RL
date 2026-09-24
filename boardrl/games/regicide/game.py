from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations


SUITS = ("H", "D", "C", "S")
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
RANK_VALUE = {
    "A": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 10,
    "Q": 15,
    "K": 20,
}
ENEMY_HP = {"J": 20, "Q": 30, "K": 40}
ENEMY_ATTACK = {"J": 10, "Q": 15, "K": 20}
HAND_LIMIT = {1: 8, 2: 7, 3: 6, 4: 5}
JOKERS_IN_TAVERN = {1: 0, 2: 0, 3: 1, 4: 2}

ATTACK = "attack"
DEFEND = "defend"


@dataclass(frozen=True, slots=True)
class Card:
    rank: str
    suit: str = ""

    @property
    def value(self) -> int:
        return 0 if self.is_joker else RANK_VALUE[self.rank]

    @property
    def is_joker(self) -> bool:
        return self.rank == "X"

    @property
    def code(self) -> str:
        return "JKR" if self.is_joker else f"{self.rank}{self.suit}"

    def __str__(self) -> str:
        return self.code


JOKER = Card("X")
CARDS = tuple(Card(rank, suit) for rank in RANKS for suit in SUITS) + (JOKER,)
CARD_ID = {card: index for index, card in enumerate(CARDS)}
CARD_BY_CODE = {card.code: card for card in CARDS}


def encode_cards(cards) -> bytearray:
    return bytearray(CARD_ID[card] for card in cards)


def decode_cards(ids):
    return (CARDS[card_id] for card_id in ids)


def card_sort_key(card: Card):
    if card.is_joker:
        return (99, 99)
    return (RANKS.index(card.rank), SUITS.index(card.suit))


def parse_card(code: str) -> Card:
    try:
        return CARD_BY_CODE[code]
    except KeyError as exc:
        raise ValueError(f"Invalid card: {code}") from exc


def attack_move(cards: tuple[Card, ...] | list[Card]) -> str:
    cards = sorted(cards, key=card_sort_key)
    return "+".join(card.code for card in cards)


def discard_move(cards: tuple[Card, ...] | list[Card]) -> str:
    cards = sorted(cards, key=card_sort_key)
    return "+".join(card.code for card in cards)


class Regicide:
    """Regicide with all mutable card collections stored as byte IDs."""

    __slots__ = (
        "num_players",
        "hand_limit",
        "tavern",
        "hands",
        "castle",
        "enemy",
        "discard",
        "battle_cards",
        "curplay",
        "_round",
        "phase",
        "enemy_damage",
        "spade_shield",
        "immunity_lifted",
        "consecutive_passes",
        "defense_remaining",
        "defeated_hp",
        "_won",
        "_lost",
        "known_tavern_prefix",
        "solo_jokers",
        "_solo_joker_available_this_phase",
        "moves",
    )

    def __init__(self, num_players: int = 2):
        if num_players not in HAND_LIMIT:
            raise ValueError("Regicide supports 1 to 4 players.")

        self.num_players = num_players
        self.hand_limit = HAND_LIMIT[num_players]

        self.tavern = encode_cards(
            Card(rank, suit)
            for rank in RANKS[:10]
            for suit in SUITS
        )
        self.tavern.extend([CARD_ID[JOKER]] * JOKERS_IN_TAVERN[num_players])
        random.shuffle(self.tavern)

        self.hands = [bytearray() for _ in range(num_players)]
        for player in range(num_players):
            for _ in range(self.hand_limit):
                self.hands[player].append(self.tavern.pop())

        kings = encode_cards(Card("K", suit) for suit in SUITS)
        queens = encode_cards(Card("Q", suit) for suit in SUITS)
        jacks = encode_cards(Card("J", suit) for suit in SUITS)
        random.shuffle(kings)
        random.shuffle(queens)
        random.shuffle(jacks)
        self.castle = bytearray((*kings, *queens, *jacks))
        self.enemy: Card | None = CARDS[self.castle.pop()]

        self.discard = bytearray()
        self.battle_cards = bytearray()

        self.curplay = 0
        self._round = 0
        self.phase = ATTACK
        self.enemy_damage = 0
        self.spade_shield = 0
        self.immunity_lifted = False
        self.consecutive_passes = 0
        self.defense_remaining = 0

        self.defeated_hp = 0
        self._won = False
        self._lost = False
        self.known_tavern_prefix = bytearray()

        self.solo_jokers = 2 if num_players == 1 else 0
        self._solo_joker_available_this_phase = num_players == 1

        self.moves: list[str] = []
        self._refresh_moves()

    def current_player(self) -> int:
        return self.curplay

    def round(self) -> int:
        return self._round

    def ended(self) -> bool:
        return self._won or self._lost

    def won(self) -> bool:
        return self._won

    def solo_medal(self) -> str | None:
        if self.num_players != 1 or not self._won:
            return None
        return {2: "gold", 1: "silver", 0: "bronze"}[self.solo_jokers]

    def points_for(self, player: int) -> int:
        del player
        current = 0
        if self.enemy is not None:
            current = min(self.enemy_damage, ENEMY_HP[self.enemy.rank])
        return self.defeated_hp + current

    def points(self) -> int:
        return self.points_for(self.curplay)

    def diff_points_for(self, player: int) -> int:
        return self.points_for(player)

    def diff_points(self) -> int:
        return self.points()

    def play_idx(self, idx: int):
        return self.play_str(self.moves[idx])

    def simulate_to_end(self):
        for _ in range(10_000):
            if self.ended():
                return
            if not self.moves:
                self._lost = True
                return
            self.play_str(random.choice(self.moves))
        raise RuntimeError("Regicide random playout did not terminate after 10,000 actions.")

    def copy(self, *, randomize: bool = False) -> "Regicide":
        g = Regicide.__new__(Regicide)
        g.num_players = self.num_players
        g.hand_limit = self.hand_limit
        g.tavern = self.tavern.copy()
        g.hands = [hand.copy() for hand in self.hands]
        g.castle = self.castle.copy()
        g.enemy = self.enemy
        g.discard = self.discard.copy()
        g.battle_cards = self.battle_cards.copy()
        g.curplay = self.curplay
        g._round = self._round
        g.phase = self.phase
        g.enemy_damage = self.enemy_damage
        g.spade_shield = self.spade_shield
        g.immunity_lifted = self.immunity_lifted
        g.consecutive_passes = self.consecutive_passes
        g.defense_remaining = self.defense_remaining
        g.defeated_hp = self.defeated_hp
        g._won = self._won
        g._lost = self._lost
        g.known_tavern_prefix = self.known_tavern_prefix.copy()
        g.solo_jokers = self.solo_jokers
        g._solo_joker_available_this_phase = self._solo_joker_available_this_phase

        if randomize and not g.ended():
            g._randomize_hidden_state()

        g.moves = self.moves if not randomize else g.gen_moves()
        return g

    def display(self, force: int = -1) -> str:
        viewer = self.curplay if force == -1 else force
        if viewer not in range(self.num_players):
            raise ValueError(f"Invalid viewer/player: {viewer}")

        if self.enemy is None:
            enemy_line = "Enemy: none"
        else:
            hp = ENEMY_HP[self.enemy.rank]
            atk = ENEMY_ATTACK[self.enemy.rank]
            effective_atk = self.effective_enemy_attack()
            enemy_line = (
                f"Enemy: {self.enemy.code} "
                f"HP:{max(0, hp - self.enemy_damage)}/{hp} "
                f"ATK:{effective_atk}/{atk} "
                f"Immunity:{'off' if self.immunity_lifted else self.enemy.suit}"
            )

        castle_counts = {
            rank: sum(CARDS[card_id].rank == rank for card_id in self.castle)
            for rank in ("J", "Q", "K")
        }
        hand_sizes = " ".join(str(len(hand)) for hand in self.hands)
        hand = " ".join(
            card.code
            for card in sorted(decode_cards(self.hands[viewer]), key=card_sort_key)
        )
        battle = " ".join(card.code for card in decode_cards(self.battle_cards))
        known_top = " ".join(card.code for card in decode_cards(self.known_tavern_prefix))

        lines = [
            f"Round: {self._round}, Phase: {self.phase}, Active: P{self.curplay}",
            enemy_line,
            f"J={castle_counts['J']} Q={castle_counts['Q']} K={castle_counts['K']}",
            f"Tavern: {len(self.tavern)}",
            f"KnownTavernTop: {known_top or '-'}",
            f"Discard: {len(self.discard) or '-'}",
            f"Battle: {battle or '-'}",
            f"HandSizes: {hand_sizes}",
            f"Hand(P{viewer}): {hand or '-'}",
            f"Passes: {self.consecutive_passes}",
        ]
        if self.phase == DEFEND and self.enemy is not None:
            lines.append(f"DamageToAbsorb: {self.defense_remaining}")
        if self.num_players == 1:
            lines.append(f"SoloJokers: {self.solo_jokers}")
        return "\n".join(lines) + "\n"

    def display_with_moves(self) -> str:
        return self.display() + "\n".join(f"@{move}" for move in self.moves)

    def gen_moves(self) -> list[str]:
        if self.ended():
            return []
        if self.phase == ATTACK:
            return self._attack_moves()
        return self._defense_moves()

    def _refresh_moves(self):
        self.moves = self.gen_moves()
        if not self.moves and not self.ended():
            self._lost = True
            self.moves = []

    def _attack_moves(self) -> list[str]:
        hand = list(decode_cards(self.hands[self.curplay]))
        moves: set[str] = set()

        non_jokers = [card for card in hand if not card.is_joker]
        for card in non_jokers:
            moves.add(attack_move((card,)))

        aces = [card for card in non_jokers if card.rank == "A"]
        for ace in aces:
            for card in non_jokers:
                if card == ace:
                    continue
                moves.add(attack_move((ace, card)))

        by_rank: dict[str, list[Card]] = {}
        for card in non_jokers:
            if card.rank == "A":
                continue
            by_rank.setdefault(card.rank, []).append(card)

        for cards in by_rank.values():
            if len(cards) < 2:
                continue
            value = cards[0].value
            max_count = min(len(cards), 10 // value)
            for count in range(2, max_count + 1):
                for combo in combinations(cards, count):
                    moves.add(attack_move(combo))

        if self.num_players > 1 and any(card.is_joker for card in hand):
            for player in range(self.num_players):
                moves.add(f"joker:P{player}")

        if self._can_pass():
            moves.add("pass")

        if (
            self.num_players == 1
            and self.solo_jokers > 0
            and self._solo_joker_available_this_phase
        ):
            moves.add("solo-joker")

        return sorted(moves)

    def _can_pass(self) -> bool:
        if self.num_players <= 1:
            return False
        return self.consecutive_passes < self.num_players - 1

    def _defense_moves(self) -> list[str]:
        if self.enemy is None or self.defense_remaining <= 0:
            return []

        hand = list(decode_cards(self.hands[self.curplay]))
        moves: set[str] = set()
        for count in range(1, len(hand) + 1):
            for cards in combinations(hand, count):
                if sum(card.value for card in cards) >= self.defense_remaining:
                    moves.add(discard_move(cards))

        if (
            self.num_players == 1
            and self.solo_jokers > 0
            and self._solo_joker_available_this_phase
        ):
            moves.add("solo-joker")

        return sorted(moves)

    def play_str(self, move: str):
        if self.ended():
            raise ValueError("Cannot play after the game has ended.")
        if move not in self.moves:
            raise ValueError(f"Illegal move: {move}. Legal moves: {self.moves}")

        if move == "solo-joker":
            self._play_solo_joker()
        elif self.phase == ATTACK:
            self._play_attack(move)
        else:
            self._play_defense(move)

        if not self.ended():
            self._refresh_moves()

    def _play_attack(self, move: str):
        if move == "pass":
            self.consecutive_passes += 1
            self._solo_joker_available_this_phase = False
            self._enter_defense()
            return

        if move.startswith("joker:P"):
            target = int(move.removeprefix("joker:P"))
            self._play_multiplayer_joker(target)
            return

        cards = [parse_card(code) for code in move.removeprefix("play:").split("+")]
        self._remove_cards_from_hand(self.curplay, cards)
        self.battle_cards.extend(CARD_ID[card] for card in cards)
        self.consecutive_passes = 0
        self._solo_joker_available_this_phase = False

        attack = sum(card.value for card in cards)
        suits = {card.suit for card in cards}

        if "H" in suits and self._power_active("H"):
            self._heart_power(attack)
        if "D" in suits and self._power_active("D"):
            self._diamond_power(attack)

        if "S" in suits:
            self.spade_shield += attack

        damage = attack
        if "C" in suits and self._power_active("C"):
            damage *= 2

        self.enemy_damage += damage
        assert self.enemy is not None
        if self.enemy_damage >= ENEMY_HP[self.enemy.rank]:
            self._defeat_enemy()
        else:
            self._enter_defense()

    def _play_multiplayer_joker(self, target: int):
        self._remove_cards_from_hand(self.curplay, [JOKER])
        self.battle_cards.append(CARD_ID[JOKER])
        self.consecutive_passes = 0
        self.immunity_lifted = True
        self.curplay = target
        self._round += 1
        self.phase = ATTACK
        self._solo_joker_available_this_phase = False

    def _play_solo_joker(self):
        if self.num_players != 1:
            raise ValueError("Solo Joker is only available in solo games.")
        if self.solo_jokers <= 0 or not self._solo_joker_available_this_phase:
            raise ValueError("No Solo Joker is available now.")

        self.discard.extend(self.hands[0])
        self.hands[0].clear()
        self._draw_to_player(0, self.hand_limit)
        self.solo_jokers -= 1
        self._solo_joker_available_this_phase = False

        if self.phase == DEFEND and self.defense_remaining > 0:
            if sum(CARDS[card_id].value for card_id in self.hands[0]) < self.defense_remaining:
                self._lost = True

    def _enter_defense(self):
        self.phase = DEFEND
        self.defense_remaining = self.effective_enemy_attack()
        self._solo_joker_available_this_phase = self.num_players == 1 and self.solo_jokers > 0

        if self.defense_remaining <= 0:
            self._finish_turn()
            return

        if sum(CARDS[card_id].value for card_id in self.hands[self.curplay]) < self.defense_remaining:
            if not (
                self.num_players == 1
                and self.solo_jokers > 0
                and self._solo_joker_available_this_phase
            ):
                self._lost = True

    def _play_defense(self, move: str):
        cards = [parse_card(code) for code in move.removeprefix("discard:").split("+")]
        total = sum(card.value for card in cards)
        if total < self.defense_remaining:
            raise ValueError(
                f"Defense only absorbs {total}, but {self.defense_remaining} is required."
            )

        self._remove_cards_from_hand(self.curplay, cards)
        self.discard.extend(CARD_ID[card] for card in cards)
        self._solo_joker_available_this_phase = False
        self.defense_remaining = 0
        self._finish_turn()

    def _finish_turn(self):
        self._round += 1
        self.curplay = (self.curplay + 1) % self.num_players
        self.phase = ATTACK
        self.defense_remaining = 0
        self._solo_joker_available_this_phase = self.num_players == 1 and self.solo_jokers > 0

    def _defeat_enemy(self):
        assert self.enemy is not None
        enemy = self.enemy
        hp = ENEMY_HP[enemy.rank]
        perfect = self.enemy_damage == hp

        self.defeated_hp += hp

        if perfect:
            enemy_id = CARD_ID[enemy]
            self.tavern.append(enemy_id)
            self.known_tavern_prefix.insert(0, enemy_id)
        else:
            self.discard.append(CARD_ID[enemy])

        self.discard.extend(self.battle_cards)
        self.battle_cards.clear()

        if not self.castle:
            self.enemy = None
            self._won = True
            self._round += 1
            self.moves = []
            return

        self.enemy = CARDS[self.castle.pop()]
        self.enemy_damage = 0
        self.spade_shield = 0
        self.immunity_lifted = False
        self.consecutive_passes = 0
        self._round += 1
        self.phase = ATTACK
        self._solo_joker_available_this_phase = self.num_players == 1 and self.solo_jokers > 0

    def _power_active(self, suit: str) -> bool:
        assert self.enemy is not None
        return self.immunity_lifted or self.enemy.suit != suit

    def effective_enemy_attack(self) -> int:
        if self.enemy is None:
            return 0
        shield = self.spade_shield
        if self.enemy.suit == "S" and not self.immunity_lifted:
            shield = 0
        return max(0, ENEMY_ATTACK[self.enemy.rank] - shield)

    def _heart_power(self, amount: int):
        if amount <= 0 or not self.discard:
            return

        random.shuffle(self.discard)
        count = min(amount, len(self.discard))
        recovered = self.discard[-count:]
        del self.discard[-count:]
        self.tavern[0:0] = recovered

    def _diamond_power(self, amount: int):
        remaining = amount
        player = self.curplay
        consecutive_full = 0

        while remaining > 0 and self.tavern:
            if len(self.hands[player]) < self.hand_limit:
                self._draw_to_player(player, 1)
                remaining -= 1
                consecutive_full = 0
            else:
                consecutive_full += 1
                if consecutive_full >= self.num_players:
                    break
            player = (player + 1) % self.num_players

    def _draw_to_player(self, player: int, count: int):
        for _ in range(count):
            if not self.tavern or len(self.hands[player]) >= self.hand_limit:
                return
            card_id = self.tavern.pop()
            if self.known_tavern_prefix:
                expected = self.known_tavern_prefix[0]
                if card_id == expected:
                    del self.known_tavern_prefix[0]
                else:
                    raise AssertionError("Known Tavern prefix is inconsistent with Tavern.")
            self.hands[player].append(card_id)

    def _remove_cards_from_hand(self, player: int, cards: list[Card]):
        hand = self.hands[player]
        for card in cards:
            try:
                hand.remove(CARD_ID[card])
            except ValueError as exc:
                raise ValueError(f"Player P{player} does not hold {card}.") from exc

    def _randomize_hidden_state(self):
        known_count = len(self.known_tavern_prefix)
        if known_count:
            known_suffix = self.tavern[-known_count:]
            expected = bytearray(reversed(self.known_tavern_prefix))
            if known_suffix != expected:
                raise AssertionError("Known Tavern prefix is inconsistent with Tavern.")
            unknown_tavern = self.tavern[:-known_count]
        else:
            known_suffix = bytearray()
            unknown_tavern = self.tavern.copy()

        random.shuffle(unknown_tavern)
        self.tavern = bytearray((*unknown_tavern, *known_suffix))

        for rank in ("J", "Q", "K"):
            indices = [i for i, card_id in enumerate(self.castle) if CARDS[card_id].rank == rank]
            cards = [self.castle[i] for i in indices]
            random.shuffle(cards)
            for i, card_id in zip(indices, cards):
                self.castle[i] = card_id