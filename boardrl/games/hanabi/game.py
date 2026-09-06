import random
from dataclasses import dataclass


RANK_MULTIPLICITIES = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
FULL_COLORS = ("R", "Y", "G", "B", "W")
MINI_COLORS = ("R", "Y")


@dataclass(frozen=True, order=True)
class Card:
    color: str
    rank: int

    def __str__(self) -> str:
        return f"{self.color}{self.rank}"


@dataclass
class CardKnowledge:
    colors: set[str]
    ranks: set[int]

    @classmethod
    def unknown(cls, colors, ranks):
        return cls(set(colors), set(ranks))

    def copy(self):
        return CardKnowledge(set(self.colors), set(self.ranks))

    def allows(self, card: Card) -> bool:
        return card.color in self.colors and card.rank in self.ranks

    def reveal_color(self, true_color: str, hinted_color: str):
        if true_color == hinted_color:
            self.colors.intersection_update({hinted_color})
        else:
            self.colors.discard(hinted_color)

    def reveal_rank(self, true_rank: int, hinted_rank: int):
        if true_rank == hinted_rank:
            self.ranks.intersection_update({hinted_rank})
        else:
            self.ranks.discard(hinted_rank)


class HanabiMode:
    name = ""
    colors = ()
    ranks = (1, 2, 3, 4, 5)
    max_information_tokens = 0
    max_life_tokens = 0

    def hand_size(self, num_players: int) -> int:
        raise NotImplementedError


class FullHanabi(HanabiMode):
    name = "full"
    colors = FULL_COLORS
    max_information_tokens = 8
    max_life_tokens = 3

    def hand_size(self, num_players: int) -> int:
        return 5 if num_players <= 3 else 4


class MiniHanabi(HanabiMode):
    """DeepMind Hanabi-Small used by MARL benchmarks."""

    name = "mini"
    colors = MINI_COLORS
    max_information_tokens = 3
    max_life_tokens = 1

    def hand_size(self, num_players: int) -> int:
        return 2


MODE_BY_NAME = {mode.name: mode for mode in (FullHanabi(), MiniHanabi())}


class Hanabi:
    def __init__(self, num_players: int = 2, mode: str = "full"):
        if not 2 <= num_players <= 5:
            raise ValueError("Hanabi supports 2 to 5 players.")
        if mode not in MODE_BY_NAME:
            raise ValueError(f"mode must be one of {sorted(MODE_BY_NAME)}, got {mode!r}")

        self.num_players = num_players
        self.mode = mode
        self.game_mode = MODE_BY_NAME[mode]
        self.colors = tuple(self.game_mode.colors)
        self.ranks = tuple(self.game_mode.ranks)
        self.max_information_tokens = self.game_mode.max_information_tokens
        self.max_life_tokens = self.game_mode.max_life_tokens
        self.hand_size = self.game_mode.hand_size(num_players)

        self.deck = self._make_deck()
        random.shuffle(self.deck)
        self.hands = [[] for _ in range(num_players)]
        self.knowledge = [[] for _ in range(num_players)]
        for _ in range(self.hand_size):
            for player in range(num_players):
                self.hands[player].append(self.deck.pop())
                self.knowledge[player].append(self._unknown_knowledge())

        self.fireworks = {color: 0 for color in self.colors}
        self.discard = []
        self.information_tokens = self.max_information_tokens
        self.life_tokens = self.max_life_tokens
        self.curplay = 0
        self._round = 0
        self.final_turns_left = None
        self.moves = self.gen_moves()

    def _make_deck(self) -> list[Card]:
        return [
            Card(color, rank)
            for color in self.colors
            for rank in self.ranks
            for _ in range(RANK_MULTIPLICITIES[rank])
        ]

    def _unknown_knowledge(self) -> CardKnowledge:
        return CardKnowledge.unknown(self.colors, self.ranks)

    @property
    def max_score(self) -> int:
        return len(self.colors) * len(self.ranks)

    def current_player(self) -> int:
        return self.curplay

    def round(self) -> int:
        return self._round

    def score(self) -> int:
        return sum(self.fireworks.values())

    def points(self) -> int:
        return self.score()

    def points_for(self, player: int) -> int:
        if player not in range(self.num_players):
            raise ValueError(f"Invalid player: {player}")
        return self.points()

    diff_points = points
    diff_points_for = points_for

    def won(self) -> bool:
        return self.score() == self.max_score

    def ended(self) -> bool:
        return self.life_tokens <= 0 or self.won() or self.final_turns_left == 0

    def gen_moves(self) -> list[str]:
        if self.ended():
            return []

        moves = [f"play:{i}" for i in range(len(self.hands[self.curplay]))]
        if self.information_tokens < self.max_information_tokens:
            moves.extend(f"discard:{i}" for i in range(len(self.hands[self.curplay])))
        if self.information_tokens > 0:
            moves.extend(self._hint_moves())
        return moves

    def _hint_moves(self) -> list[str]:
        moves = []
        for offset in range(1, self.num_players):
            target = (self.curplay + offset) % self.num_players
            hand = self.hands[target]
            present_colors = {card.color for card in hand}
            present_ranks = {card.rank for card in hand}
            moves.extend(
                f"hint:P+{offset}:color:{color}"
                for color in self.colors
                if color in present_colors
            )
            moves.extend(
                f"hint:P+{offset}:rank:{rank}"
                for rank in self.ranks
                if rank in present_ranks
            )
        return moves

    def display(self, force: int = -1) -> str:
        viewer = self.curplay if force == -1 else force
        if viewer not in range(self.num_players):
            raise ValueError(f"Invalid viewer: {viewer}")

        final = "-" if self.final_turns_left is None else str(self.final_turns_left)
        fireworks = " ".join(f"{color}{self.fireworks[color]}" for color in self.colors)
        discard = " ".join(str(card) for card in self.discard) or "-"
        acting_offset = (self.curplay - viewer) % self.num_players
        acting = "self" if acting_offset == 0 else f"P+{acting_offset}"
        lines = [
            f"Hanabi {self.mode} | Round: {self._round} | Turn: {acting}",
            (
                f"Score: {self.score()}/{self.max_score} | "
                f"Info: {self.information_tokens}/{self.max_information_tokens} | "
                f"Lives: {self.life_tokens}/{self.max_life_tokens} | "
                f"Deck: {len(self.deck)} | Final: {final}"
            ),
            f"Fireworks: {fireworks}",
            f"Discard: {discard}",
        ]
        for offset in range(self.num_players):
            player = (viewer + offset) % self.num_players
            lines.append(self._display_hand(player, viewer, offset))
        return "\n".join(lines) + "\n"

    def _display_hand(self, player: int, viewer: int, offset: int) -> str:
        cards = []
        for card, knowledge in zip(self.hands[player], self.knowledge[player]):
            visible_card = "??" if player == viewer else str(card)
            cards.append(f"{visible_card}[{self._knowledge_text(knowledge)}]")
        label = "Self" if offset == 0 else f"P+{offset}"
        return f"{label}: " + (" ".join(cards) if cards else "-")

    def _knowledge_text(self, knowledge: CardKnowledge) -> str:
        colors = "".join(color for color in self.colors if color in knowledge.colors) or "-"
        ranks = "".join(str(rank) for rank in self.ranks if rank in knowledge.ranks) or "-"
        return f"{colors}|{ranks}"

    def display_with_moves(self) -> str:
        return self.display() + "\n".join(f"@{move}" for move in self.moves)

    def play_idx(self, idx: int):
        return self.play_str(self.moves[idx])

    def play_str(self, move: str):
        if move not in self.moves:
            raise ValueError(f"Illegal move: {move}. Legal moves: {self.moves}")

        if move.startswith("play:"):
            self._play_card(int(move.split(":", 1)[1]))
        elif move.startswith("discard:"):
            self._discard_card(int(move.split(":", 1)[1]))
        else:
            self._give_hint(move)

        if self.ended():
            self.moves = []
        else:
            self.moves = self.gen_moves()

    def _play_card(self, index: int):
        player = self.curplay
        card = self._remove_from_hand(player, index)
        if card.rank == self.fireworks[card.color] + 1:
            self.fireworks[card.color] = card.rank
            if card.rank == self.ranks[-1]:
                self.information_tokens = min(
                    self.max_information_tokens, self.information_tokens + 1
                )
        else:
            self.discard.append(card)
            self.life_tokens -= 1

        if self.life_tokens <= 0 or self.won():
            self._round += 1
            return
        self._draw_card(player)
        self._finish_turn()

    def _discard_card(self, index: int):
        player = self.curplay
        card = self._remove_from_hand(player, index)
        self.discard.append(card)
        self.information_tokens = min(
            self.max_information_tokens, self.information_tokens + 1
        )
        self._draw_card(player)
        self._finish_turn()

    def _give_hint(self, move: str):
        _, target_text, kind, value = move.split(":")
        offset = int(target_text[2:])
        target = (self.curplay + offset) % self.num_players
        self.information_tokens -= 1

        if kind == "color":
            for card, knowledge in zip(self.hands[target], self.knowledge[target]):
                knowledge.reveal_color(card.color, value)
        else:
            rank = int(value)
            for card, knowledge in zip(self.hands[target], self.knowledge[target]):
                knowledge.reveal_rank(card.rank, rank)
        self._finish_turn()

    def _remove_from_hand(self, player: int, index: int) -> Card:
        self.knowledge[player].pop(index)
        return self.hands[player].pop(index)

    def _draw_card(self, player: int):
        if not self.deck:
            return
        self.hands[player].append(self.deck.pop())
        self.knowledge[player].append(self._unknown_knowledge())
        if not self.deck and self.final_turns_left is None:
            # Count the current turn as the +1, then leave one final turn per player.
            self.final_turns_left = self.num_players + 1

    def _finish_turn(self):
        self._round += 1
        if self.final_turns_left is not None:
            self.final_turns_left -= 1
            if self.final_turns_left == 0:
                return
        self.curplay = (self.curplay + 1) % self.num_players

    def copy(self, randomize: bool = False):
        copied = object.__new__(Hanabi)
        copied.num_players = self.num_players
        copied.mode = self.mode
        copied.game_mode = self.game_mode
        copied.colors = self.colors
        copied.ranks = self.ranks
        copied.max_information_tokens = self.max_information_tokens
        copied.max_life_tokens = self.max_life_tokens
        copied.hand_size = self.hand_size
        copied.deck = self.deck[:]
        copied.hands = [hand[:] for hand in self.hands]
        copied.knowledge = [
            [knowledge.copy() for knowledge in hand_knowledge]
            for hand_knowledge in self.knowledge
        ]
        copied.fireworks = self.fireworks.copy()
        copied.discard = self.discard[:]
        copied.information_tokens = self.information_tokens
        copied.life_tokens = self.life_tokens
        copied.curplay = self.curplay
        copied._round = self._round
        copied.final_turns_left = self.final_turns_left
        if randomize and not copied.ended():
            copied._randomize_hidden_state()
        copied.moves = copied.gen_moves()
        return copied

    def _randomize_hidden_state(self):
        # Every hand is visible to at least one player in Hanabi. A randomized
        # copy must therefore preserve every hand exactly; only the draw pile is
        # unseen by all players and may be resampled.
        random.shuffle(self.deck)

    def simulate_to_end(self):
        steps = 0
        while not self.ended():
            if not self.moves:
                raise RuntimeError("Non-terminal Hanabi state has no legal moves")
            self.play_str(random.choice(self.moves))
            steps += 1
            if steps > 1000:
                raise RuntimeError("Hanabi playout exceeded 1000 turns")
        return self.points()
