import random
from dataclasses import dataclass

from boardrl.games.compact import IndexedByteArray, NamedByteCounts


RANK_MULTIPLICITIES = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
FULL_COLORS = ("R", "Y", "G", "B", "W")
MINI_COLORS = ("R", "Y")


@dataclass(frozen=True, order=True, slots=True)
class Card:
    color: str
    rank: int

    def __str__(self) -> str:
        return f"{self.color}{self.rank}"


CARDS = tuple(Card(color, rank) for color in FULL_COLORS for rank in range(1, 6))


class CardArray(IndexedByteArray):
    __slots__ = ()
    VALUES = CARDS
    ID_BY_VALUE = {card: index for index, card in enumerate(CARDS)}


class Fireworks(NamedByteCounts):
    __slots__ = ()
    KEYS = FULL_COLORS
    INDEX = {color: index for index, color in enumerate(KEYS)}


@dataclass(slots=True)
class CardKnowledge:
    colors: tuple[str, ...]
    ranks: tuple[int, ...]
    hinted_color: str | None = None
    hinted_rank: int | None = None

    @classmethod
    def unknown(cls, colors, ranks):
        return cls(tuple(colors), tuple(ranks))

    def copy(self):
        return CardKnowledge(
            self.colors,
            self.ranks,
            self.hinted_color,
            self.hinted_rank,
        )

    def allows(self, card: Card) -> bool:
        return card.color in self.colors and card.rank in self.ranks

    def reveal_color(self, true_color: str, hinted_color: str):
        if true_color == hinted_color:
            self.colors = (hinted_color,)
            self.hinted_color = hinted_color
        else:
            self.colors = tuple(color for color in self.colors if color != hinted_color)

    def reveal_rank(self, true_rank: int, hinted_rank: int):
        if true_rank == hinted_rank:
            self.ranks = (hinted_rank,)
            self.hinted_rank = hinted_rank
        else:
            self.ranks = tuple(rank for rank in self.ranks if rank != hinted_rank)


@dataclass(frozen=True, slots=True)
class ActionRecord:
    actor: int
    kind: str
    target: int | None = None
    clue_kind: str | None = None
    clue_value: str | int | None = None
    affected: tuple[int, ...] = ()
    card: Card | None = None
    success: bool | None = None


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
    __slots__ = (
        "num_players",
        "mode",
        "game_mode",
        "colors",
        "ranks",
        "max_information_tokens",
        "max_life_tokens",
        "hand_size",
        "deck",
        "hands",
        "knowledge",
        "fireworks",
        "discard",
        "information_tokens",
        "life_tokens",
        "curplay",
        "_round",
        "final_turns_left",
        "last_action",
        "moves",
    )

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
        self.hands = [CardArray() for _ in range(num_players)]
        self.knowledge = [[] for _ in range(num_players)]
        for _ in range(self.hand_size):
            for player in range(num_players):
                self.hands[player].append(self.deck.pop())
                self.knowledge[player].append(self._unknown_knowledge())

        self.fireworks = Fireworks()
        self.discard = CardArray()
        self.information_tokens = self.max_information_tokens
        self.life_tokens = self.max_life_tokens
        self.curplay = 0
        self._round = 0
        self.final_turns_left = None
        self.last_action: ActionRecord | None = None
        self.moves = self.gen_moves()

    def _make_deck(self) -> CardArray:
        return CardArray(
            Card(color, rank)
            for color in self.colors
            for rank in self.ranks
            for _ in range(RANK_MULTIPLICITIES[rank])
        )

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
        return sum(self.fireworks[color] for color in self.colors)

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

        moves = [f"p {i}" for i in range(len(self.hands[self.curplay]))]
        if self.information_tokens < self.max_information_tokens:
            moves.extend(f"d {i}" for i in range(len(self.hands[self.curplay])))
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
                f"h p{offset} c{color}"
                for color in self.colors
                if color in present_colors
            )
            moves.extend(
                f"h p{offset} r{rank}"
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
        acting = self._player_label(self.curplay, viewer)
        lines = [
            f"{self.mode} r{self._round} t {acting}",
            (
                f"s {self.score()}/{self.max_score} "
                f"i {self.information_tokens}/{self.max_information_tokens} "
                f"l {self.life_tokens}/{self.max_life_tokens} "
                f"d {len(self.deck)} f {final}"
            ),
            f"fw {fireworks}",
            f"dc {discard}",
            f"la {self._last_action_text(viewer)}",
        ]
        for offset in range(self.num_players):
            player = (viewer + offset) % self.num_players
            lines.append(self._display_hand(player, viewer, offset))
        return "\n".join(lines) + "\n"

    def _player_label(self, player: int, viewer: int) -> str:
        offset = (player - viewer) % self.num_players
        return "me" if offset == 0 else f"p{offset}"

    def _last_action_text(self, viewer: int) -> str:
        action = self.last_action
        if action is None:
            return "-"

        actor = self._player_label(action.actor, viewer)
        if action.kind == "play":
            outcome = "ok" if action.success else "x"
            return f"{actor} p {action.card} {outcome}"
        if action.kind == "discard":
            return f"{actor} d {action.card}"

        target = self._player_label(action.target, viewer)  # type: ignore[arg-type]
        clue = (
            f"c{action.clue_value}"
            if action.clue_kind == "color"
            else f"r{action.clue_value}"
        )
        affected = " ".join(str(index) for index in action.affected)
        return f"{actor} h {target} {clue} {affected}"

    def _display_hand(self, player: int, viewer: int, offset: int) -> str:
        cards = []
        for card, knowledge in zip(self.hands[player], self.knowledge[player]):
            visible_card = "?" if player == viewer else str(card)
            cards.append(
                f"{visible_card}/{self._possible_text(knowledge)}/{self._hinted_text(knowledge)}"
            )
        label = "me" if offset == 0 else f"p{offset}"
        return f"{label} " + (" ".join(cards) if cards else "-")

    def _possible_text(self, knowledge: CardKnowledge) -> str:
        colors = "".join(color for color in self.colors if color in knowledge.colors) or "-"
        ranks = "".join(str(rank) for rank in self.ranks if rank in knowledge.ranks) or "-"
        return colors + ranks

    def _hinted_text(self, knowledge: CardKnowledge) -> str:
        color = knowledge.hinted_color or ""
        rank = "" if knowledge.hinted_rank is None else str(knowledge.hinted_rank)
        return color + rank or "-"

    def display_with_moves(self) -> str:
        return self.display() + "\n".join(f"@{move}" for move in self.moves)

    def play_idx(self, idx: int):
        return self.play_str(self.moves[idx])

    def play_str(self, move: str):
        if move not in self.moves:
            raise ValueError(f"Illegal move: {move}. Legal moves: {self.moves}")

        parts = move.split()
        if parts[0] == "p":
            self._play_card(int(parts[1]))
        elif parts[0] == "d":
            self._discard_card(int(parts[1]))
        else:
            self._give_hint(parts)

        if self.ended():
            self.moves = []
        else:
            self.moves = self.gen_moves()

    def _play_card(self, index: int):
        player = self.curplay
        card = self._remove_from_hand(player, index)
        success = card.rank == self.fireworks[card.color] + 1
        if success:
            self.fireworks[card.color] = card.rank
            if card.rank == self.ranks[-1]:
                self.information_tokens = min(
                    self.max_information_tokens, self.information_tokens + 1
                )
        else:
            self.discard.append(card)
            self.life_tokens -= 1
        self.last_action = ActionRecord(
            actor=player,
            kind="play",
            card=card,
            success=success,
        )

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
        self.last_action = ActionRecord(actor=player, kind="discard", card=card)
        self._draw_card(player)
        self._finish_turn()

    def _give_hint(self, parts: list[str]):
        _, target_text, clue = parts
        offset = int(target_text[1:])
        target = (self.curplay + offset) % self.num_players
        kind = "color" if clue[0] == "c" else "rank"
        value = clue[1:] if kind == "color" else int(clue[1:])
        self.information_tokens -= 1

        affected = []
        if kind == "color":
            for index, (card, knowledge) in enumerate(
                zip(self.hands[target], self.knowledge[target])
            ):
                knowledge.reveal_color(card.color, value)  # type: ignore[arg-type]
                if card.color == value:
                    affected.append(index)
        else:
            for index, (card, knowledge) in enumerate(
                zip(self.hands[target], self.knowledge[target])
            ):
                knowledge.reveal_rank(card.rank, value)  # type: ignore[arg-type]
                if card.rank == value:
                    affected.append(index)

        self.last_action = ActionRecord(
            actor=self.curplay,
            kind="hint",
            target=target,
            clue_kind=kind,
            clue_value=value,
            affected=tuple(affected),
        )
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
            self.final_turns_left = self.num_players + 1

    def _finish_turn(self):
        self._round += 1
        if self.final_turns_left is not None:
            self.final_turns_left -= 1
            if self.final_turns_left == 0:
                return
        self.curplay = (self.curplay + 1) % self.num_players

    def copy(self, randomize: bool = False):
        copied = Hanabi.__new__(Hanabi)
        copied.num_players = self.num_players
        copied.mode = self.mode
        copied.game_mode = self.game_mode
        copied.colors = self.colors
        copied.ranks = self.ranks
        copied.max_information_tokens = self.max_information_tokens
        copied.max_life_tokens = self.max_life_tokens
        copied.hand_size = self.hand_size
        copied.deck = self.deck.copy()
        copied.hands = [hand.copy() for hand in self.hands]
        copied.knowledge = [
            [knowledge.copy() for knowledge in hand_knowledge]
            for hand_knowledge in self.knowledge
        ]
        copied.fireworks = self.fireworks.copy()
        copied.discard = self.discard.copy()
        copied.information_tokens = self.information_tokens
        copied.life_tokens = self.life_tokens
        copied.curplay = self.curplay
        copied._round = self._round
        copied.final_turns_left = self.final_turns_left
        copied.last_action = self.last_action
        if randomize and not copied.ended():
            copied._randomize_hidden_state()
        copied.moves = self.moves
        return copied

    def _randomize_hidden_state(self):
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