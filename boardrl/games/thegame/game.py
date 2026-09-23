import random


MESSAGE_MOVES = tuple("ABCDEFGHIJ")
MESSAGE_BY_ID = ("", *MESSAGE_MOVES)
MESSAGE_ID = {message: index for index, message in enumerate(MESSAGE_BY_ID)}
MESSAGE_SET = frozenset(MESSAGE_MOVES)
PLAYED_CARD_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEF"
GAME_MODES = {
    "strict",
    "free",
    "omni",
    "strict_message_before_draw",
    "free_message_before_draw",
    "strict_message_after_draw",
    "free_message_after_draw",
}
PLAYING_PHASE = "playing"
AFTER_DRAW_MESSAGE_PHASE = "after_draw_message"


def _card_buffer(values=(), *, max_value: int):
    """Use one byte/card for the normal game, retain large custom variants."""
    if max_value <= 256:
        return bytearray(values)
    return list(values)


class GameMode:
    name = ""
    has_messages = False
    displays_all_information = False

    def randomize_hidden_state(self, game):
        random.shuffle(game.deck)

    def moves_after_minimum_cards(self, game):
        raise NotImplementedError

    def after_minimum_cards_reached(self, game, player):
        raise NotImplementedError

    def play_x(self, game, player):
        raise ValueError("This turn state does not allow ending with x.")

    def play_message(self, game, player, message):
        raise ValueError("This turn state does not allow messages.")


class Strict(GameMode):
    name = "strict"

    def moves_after_minimum_cards(self, game):
        return []

    def after_minimum_cards_reached(self, game, player):
        game.finish_turn(player)


class Free(GameMode):
    name = "free"

    def moves_after_minimum_cards(self, game):
        return game.card_moves() + ["x"]

    def after_minimum_cards_reached(self, game, player):
        game.moves = self.moves_after_minimum_cards(game)
        if game.moves == ["x"]:
            self.play_x(game, player)

    def play_x(self, game, player):
        game.finish_turn(player)


class Omni(Free):
    """Free play with a complete view of the players and draw pile."""

    name = "omni"
    displays_all_information = True

    def randomize_hidden_state(self, game):
        hidden_deck = game.deck[:-10]
        random.shuffle(hidden_deck)
        game.deck[:-10] = hidden_deck


class StrictMessageBeforeDraw(GameMode):
    name = "strict_message_before_draw"
    has_messages = True

    def moves_after_minimum_cards(self, game):
        return game.message_moves()

    def after_minimum_cards_reached(self, game, player):
        game.moves = self.moves_after_minimum_cards(game)

    def play_message(self, game, player, message):
        game.record_message(player, message)
        game.finish_turn(player)


class FreeMessageBeforeDraw(StrictMessageBeforeDraw):
    name = "free_message_before_draw"

    def moves_after_minimum_cards(self, game):
        return game.card_moves() + game.message_moves()


class StrictMessageAfterDraw(GameMode):
    name = "strict_message_after_draw"
    has_messages = True

    def moves_after_minimum_cards(self, game):
        return []

    def after_minimum_cards_reached(self, game, player):
        game.draw_then_wait_for_message(player)


class FreeMessageAfterDraw(Free):
    name = "free_message_after_draw"
    has_messages = True

    def play_x(self, game, player):
        game.draw_then_wait_for_message(player)


GAME_MODE_BY_NAME = {
    mode.name: mode
    for mode in (
        Strict(),
        Free(),
        Omni(),
        StrictMessageBeforeDraw(),
        FreeMessageBeforeDraw(),
        StrictMessageAfterDraw(),
        FreeMessageAfterDraw(),
    )
}


class TheGame:
    """A simplified version of The Game with byte-backed card state."""

    __slots__ = (
        "max_value",
        "mode",
        "game_mode",
        "deck",
        "piles",
        "hands",
        "initial_hand_size",
        "_round",
        "turn",
        "action",
        "curplay",
        "num_players",
        "_turn_phase",
        "_played_cards",
        "moves",
        "_last_messages",
    )

    def __init__(
        self,
        num_players: int = 2,
        max_value: int = 100,
        mode: str = "strict",
    ):
        assert 0 < num_players <= 5, "The Game supports 1 to 5 players."
        if mode not in GAME_MODES:
            raise ValueError(f"mode must be one of {sorted(GAME_MODES)}, got {mode!r}")

        self.max_value = max_value
        self.mode = mode
        self.game_mode = GAME_MODE_BY_NAME[mode]
        self.deck = _card_buffer(range(2, max_value), max_value=max_value)
        random.shuffle(self.deck)

        self.piles = (
            bytearray((1, 1, max_value, max_value))
            if max_value <= 255
            else [1, 1, max_value, max_value]
        )

        self.hands = [
            _card_buffer(max_value=max_value) for _ in range(num_players)
        ]
        self.initial_hand_size = 6 if num_players > 3 else 7
        for p in range(num_players):
            for _ in range(self.initial_hand_size):
                self.hands[p].append(self.deck.pop())

        self._round = 0
        self.turn = 0  # compatibility alias retained for old probes/tests
        self.action = 0
        self.curplay = 0
        self.num_players = num_players
        self._turn_phase = PLAYING_PHASE
        self._played_cards = 0
        self.moves = self.gen_moves()
        self._last_messages = bytearray(num_players)

    def copy(self, randomize=False):
        g = TheGame.__new__(TheGame)
        g.max_value = self.max_value
        g.mode = self.mode
        g.game_mode = self.game_mode
        g.deck = self.deck.copy()
        if randomize:
            g.game_mode.randomize_hidden_state(g)
        g.piles = self.piles.copy()
        g.hands = [hand.copy() for hand in self.hands]
        g.initial_hand_size = self.initial_hand_size
        g._round = self._round
        g.turn = self.turn
        g.action = self.action
        g.curplay = self.curplay
        g.num_players = self.num_players
        g._turn_phase = self._turn_phase
        g._played_cards = self._played_cards
        g.moves = self.moves
        g._last_messages = self._last_messages.copy()
        return g

    def round(self):
        return self._round

    def current_player(self) -> int:
        return self.curplay

    def display(self, force=-1) -> str:
        if force == -1:
            p = self.curplay
        else:
            assert force in range(self.num_players)
            p = force
        pile_info = " ".join(f"{val}" for val in self.piles)
        played_cards_memory = self.played_cards_memory()
        hand_lines = [f"Hand: {' '.join(str(c) for c in self.hands[p])}"]
        deck_line = ""
        if self.game_mode.displays_all_information:
            for offset in range(1, self.num_players):
                player = (p + offset) % self.num_players
                hand_lines.append(
                    f"Hand: {' '.join(str(c) for c in self.hands[player])}"
                )
            next_cards = self.deck[-10:][::-1]
            deck_line = f"Deck: {' '.join(str(c) for c in next_cards)}\n"
        hands_line = "\n".join(hand_lines)
        msg_line = ""
        if self.has_messages():
            order = [((p + i) % self.num_players) for i in range(1, self.num_players)]
            rel_msgs = [MESSAGE_BY_ID[self._last_messages[i]] for i in order]
            msg_line = f"Msgs: {''.join(rel_msgs)}\n"
        return (
            f"Round: {self._round}, Action: {self.action}\n"
            f"Piles: {pile_info}\n"
            f"Cards: {len(self.deck)}\n"
            f"Mem: {played_cards_memory}\n"
            f"{hands_line}\n"
            f"{deck_line}"
            f"{msg_line}"
        )

    def played_cards_memory(self) -> str:
        highest_decade = max(0, (self.max_value - 1) // 10)
        return " ".join(
            self._played_decade_memory(decade)
            for decade in range(highest_decade, -1, -1)
        )

    def _played_decade_memory(self, decade: int) -> str:
        start = decade * 10
        return (
            f"{decade}"
            f"{self._played_card_symbol(start + 9)}"
            f"{self._played_card_symbol(start + 4)}"
        )

    def _played_card_symbol(self, highest_card: int) -> str:
        value = sum(
            1 << bit
            for bit in range(5)
            if self._played_cards & (1 << (highest_card - bit))
        )
        return PLAYED_CARD_SYMBOLS[value]

    def display_with_moves(self) -> str:
        board = self.display()
        return board + "\n".join([f"@{m}" for m in self.moves])

    def gen_moves(self) -> list[str]:
        if self._turn_phase == AFTER_DRAW_MESSAGE_PHASE:
            return self.message_moves()
        if self.needs_more_cards_this_turn():
            return self.card_moves()
        return self.game_mode.moves_after_minimum_cards(self)

    def has_messages(self) -> bool:
        return self.game_mode.has_messages

    def needs_more_cards_this_turn(self) -> bool:
        return self.action < self.min_actions()

    def min_actions(self) -> int:
        return 2 if self.deck else 1

    def card_moves(self) -> list[str]:
        moves = []
        ascending_indices = (0, 1)
        descending_indices = (2, 3)
        hand = self.hands[self.curplay]

        for card in hand:
            for pile_idx in ascending_indices:
                top_val = self.piles[pile_idx]
                if card >= top_val or (top_val - card == 10):
                    moves.append(f"{card}->{pile_idx}")

            for pile_idx in descending_indices:
                top_val = self.piles[pile_idx]
                if card <= top_val or (card - top_val == 10):
                    moves.append(f"{card}->{pile_idx}")

        return moves

    def message_moves(self) -> list[str]:
        return list(MESSAGE_MOVES)

    def draw_to_hand(self, player: int):
        while self.deck and len(self.hands[player]) < self.initial_hand_size:
            self.hands[player].append(self.deck.pop())

    def finish_turn(self, player: int):
        self.draw_to_hand(player)
        self.pass_to_next_player()

    def draw_then_wait_for_message(self, player: int):
        self.draw_to_hand(player)
        self._turn_phase = AFTER_DRAW_MESSAGE_PHASE
        self.moves = self.gen_moves()

    def pass_to_next_player(self):
        self.curplay = (self.curplay + 1) % self.num_players
        self.skip_empty_hands_after_deck_empty()
        self.action = 0
        self._turn_phase = PLAYING_PHASE
        self.moves = self.gen_moves()
        self._round += 1
        self.turn = self._round

    def skip_empty_hands_after_deck_empty(self):
        if self.deck:
            return
        for _ in range(self.num_players):
            if self.hands[self.curplay]:
                return
            self._last_messages[self.curplay] = 0
            self.curplay = (self.curplay + 1) % self.num_players

    def record_message(self, player: int, message: str):
        self._last_messages[player] = MESSAGE_ID[message]

    def play_str(self, move: str):
        if move not in self.moves:
            raise ValueError(f"Illegal move: {move}. Legal moves: {self.moves}")

        player = self.curplay

        if move == "x":
            self.game_mode.play_x(self, player)
            return
        if move in MESSAGE_SET:
            self.play_message(player, move)
            return

        card_str, pile_str = move.split("->")
        card = int(card_str)
        pile_idx = int(pile_str)

        self.hands[player].remove(card)
        self._played_cards |= 1 << card
        self.piles[pile_idx] = card
        self.action += 1

        if self.needs_more_cards_this_turn():
            self.moves = self.card_moves()
        else:
            self.game_mode.after_minimum_cards_reached(self, player)

    def play_message(self, player: int, message: str):
        if self._turn_phase == AFTER_DRAW_MESSAGE_PHASE:
            self.record_message(player, message)
            self.pass_to_next_player()
        else:
            self.game_mode.play_message(self, player, message)

    def ended(self) -> bool:
        return (not self.deck and not any(self.hands)) or (
            bool(self.hands[self.curplay]) and not self.moves
        )

    def won(self) -> bool:
        return not self.deck and not any(self.hands)

    def points(self) -> int:
        return self.max_value - (len(self.deck) + sum(len(h) for h in self.hands))

    def points_for(self, player: int) -> int:
        return self.points()

    diff_points = points
    diff_points_for = points_for

    def play_idx(self, idx: int):
        return self.play_str(self.moves[idx])


if __name__ == "__main__":
    game = TheGame(num_players=2)
    print(game.display_with_moves())
    possible = game.gen_moves()
    print("Possible moves:", possible)
    if possible:
        move = possible[0]
        print(f"Playing move: {move}")
        game.play_str(move)
    print(game.display_with_moves())