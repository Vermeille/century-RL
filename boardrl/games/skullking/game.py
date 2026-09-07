import random


SUITS = "rybB"


def fresh_deck():
    return (
        [f"{color}:{number}" for color in SUITS for number in range(1, 15)]
        + ["_:E"] * 5
        + ["_:P"] * 5
        + ["_:M"] * 2
        + ["_:K"]
        + ["_:T"]
    )


def _card(card: str):
    color, value = card.split(":")
    return color, value


def _rank(card: str) -> str:
    """Return the effective rank for special cards, including Tigress choices."""
    _, value = _card(card)
    if value == "T=P":
        return "P"
    if value == "T=E":
        return "E"
    return value


def beats(top: str, candidate: str) -> bool:
    """Return whether candidate directly beats top.

    Character cards are non-transitive, so complete tricks are resolved from
    all played cards by :func:`_winning_play` rather than by repeated pairwise
    comparison.
    """
    top_color, _ = _card(top)
    candidate_color, _ = _card(candidate)
    top_value = _rank(top)
    candidate_value = _rank(candidate)

    if top_value == "E":
        return candidate_value != "E"
    if candidate_value == "E":
        return False

    if top_value == "P":
        return candidate_value == "K"
    if top_value == "M":
        return candidate_value == "P"
    if top_value == "K":
        return candidate_value == "M"

    if candidate_value in {"P", "M", "K"}:
        return True

    if candidate_color == top_color:
        return int(candidate_value) > int(top_value)
    if candidate_color == "B":
        return True
    return False


def can_play(color: str, hand: list[str]) -> list[bool]:
    """Return which physical cards may be played when ``color`` is led."""
    if color not in SUITS:
        return [True] * len(hand)

    has_lead_suit = any(_card(card)[0] == color for card in hand)
    playable = []
    for card in hand:
        card_color, _ = _card(card)
        playable.append(card_color == "_" or not has_lead_suit or card_color == color)
    return playable


def _winning_play(trick: list[tuple[int, str]], lead_suit: str) -> tuple[int, str]:
    """Resolve a complete or partial trick from all cards played so far."""
    assert trick

    def first_with_rank(rank: str):
        return next((play for play in trick if _rank(play[1]) == rank), None)

    pirate = first_with_rank("P")
    mermaid = first_with_rank("M")
    skull_king = first_with_rank("K")

    # Mermaid beats Skull King even if a Pirate is also present. Otherwise
    # Skull King beats Pirates, Pirates beat Mermaids, and characters beat
    # numbered cards. First played wins ties between equal characters.
    if mermaid is not None and skull_king is not None:
        return mermaid
    if skull_king is not None:
        return skull_king
    if pirate is not None:
        return pirate
    if mermaid is not None:
        return mermaid

    numbered = [play for play in trick if _rank(play[1]) != "E"]
    if not numbered:
        return trick[0]

    black = [play for play in numbered if _card(play[1])[0] == "B"]
    candidates = black or [play for play in numbered if _card(play[1])[0] == lead_suit]
    if not candidates:
        # Only reachable in a partial trick before a suit is established.
        return numbered[0]
    return max(candidates, key=lambda play: int(_card(play[1])[1]))


class SkullKing:
    def __init__(self, num_players: int = 4, num_rounds: int = 10):
        assert 2 <= num_players <= 8
        assert 1 <= num_rounds <= 10
        self.num_players = num_players
        self.num_rounds = num_rounds
        self.the_points = [0] * num_players
        self.bids = []
        self.tricks = [0] * num_players
        self.round_ = 1
        self.round_starter = 0
        self.current_player_ = 0
        self.phase = "bid"
        self.current_color = "_"
        self.current_top = "_:E"
        self.current_winner = 0
        self._lead_suit_pending = True
        self._trick = []
        self._played_tricks: list[list[tuple[int, str]]] = []
        self.captured = [[] for _ in range(num_players)]
        self._captured_tricks = [[] for _ in range(num_players)]
        self.round_points = [0] * num_players
        self.hands = [[] for _ in range(num_players)]
        self.deck = []
        self._deal()
        self.moves = self.gen_moves()

    def _cards_per_player(self) -> int:
        if self.num_players == 8:
            return min(self.round_, 8)
        return self.round_

    def _deal(self) -> None:
        self.deck = fresh_deck()
        random.shuffle(self.deck)
        cards_per_player = self._cards_per_player()
        assert cards_per_player * self.num_players <= len(self.deck)
        self.hands = [[] for _ in range(self.num_players)]
        for _ in range(cards_per_player):
            for hand in self.hands:
                hand.append(self.deck.pop())

    def start_round_(self):
        self.phase = "play"
        self.tricks = [0] * self.num_players
        self.captured = [[] for _ in range(self.num_players)]
        self._captured_tricks = [[] for _ in range(self.num_players)]
        self._played_tricks = []
        self._trick = []
        self.current_player_ = self.round_starter
        self.current_color = "_"
        self.current_top = "_:E"
        self.current_winner = self.current_player_
        self._lead_suit_pending = True

    def _finish_round(self) -> None:
        cards_dealt = self._cards_per_player()
        self.round_points = [0] * self.num_players
        for player, bid in enumerate(self.bids):
            taken = self.tricks[player]
            if bid == taken:
                points = 10 * cards_dealt if bid == 0 else 20 * taken
                points += self._bonus_points(player)
            elif bid == 0:
                points = -10 * cards_dealt
            else:
                points = -10 * abs(bid - taken)
            self.round_points[player] = points
            self.the_points[player] += points

        if self.round_ == self.num_rounds:
            self.round_ += 1
            self.phase = "ended"
            self.moves = []
            return

        self.round_ += 1
        self.round_starter = (self.round_starter + 1) % self.num_players
        self.phase = "bid"
        self.bids = []
        # Bids are collected in seat order only as an implementation detail;
        # they remain hidden until all players have committed.
        self.current_player_ = 0
        self._deal()
        self.moves = self.gen_moves()

    def _bonus_points(self, player: int) -> int:
        cards = self.captured[player]
        bonus = sum(
            20 if card == "B:14" else 10 for card in cards if card.endswith(":14")
        )
        for trick, winner_card in self._captured_tricks[player]:
            winner_rank = _rank(winner_card)
            if winner_rank == "P":
                bonus += 20 * sum(_rank(card) == "M" for card in trick)
            elif winner_rank == "K":
                bonus += 30 * sum(_rank(card) == "P" for card in trick)
            elif winner_rank == "M":
                bonus += 40 * sum(_rank(card) == "K" for card in trick)
        return bonus

    def ended(self):
        return self.phase == "ended"

    def gen_moves(self) -> list[str]:
        if self.ended():
            return []
        if self.phase == "bid":
            return [str(i) for i in range(self._cards_per_player() + 1)]
        if self.phase == "play":
            hand = self.hands[self.current_player_]
            color = self.current_color if not self._lead_suit_pending else "_"
            moves = []
            for card, can in zip(hand, can_play(color, hand)):
                if not can:
                    continue
                if card == "_:T":
                    moves.extend(["_:T=P", "_:T=E"])
                else:
                    moves.append(card)
            # Identical physical cards have identical consequences and should
            # not become duplicate policy actions.
            return list(dict.fromkeys(moves))
        return []

    def _complete_trick(self) -> None:
        winner = self.current_winner
        cards = [card for _, card in self._trick]
        self.tricks[winner] += 1
        self.captured[winner].extend(cards)
        self._captured_tricks[winner].append((cards, self.current_top))
        self._played_tricks.append(self._trick[:])

        if all(not hand for hand in self.hands):
            self._finish_round()
            return

        self.current_player_ = winner
        self.current_color = "_"
        self.current_top = "_:E"
        self.current_winner = winner
        self._lead_suit_pending = True
        self._trick = []
        self.moves = self.gen_moves()

    def _update_lead(self, card: str) -> None:
        card_color, _ = _card(card)
        rank = _rank(card)
        if not self._lead_suit_pending:
            return
        if rank == "E":
            return
        self._lead_suit_pending = False
        self.current_color = card_color if card_color in SUITS else "_"

    def play_str(self, move: str):
        assert not self.ended()
        p = self.current_player_

        if self.phase == "bid":
            assert move in self.moves
            self.bids.append(int(move))
            self.current_player_ = (p + 1) % self.num_players
            if len(self.bids) == self.num_players:
                self.start_round_()
            self.moves = self.gen_moves()
            return

        assert self.phase == "play"
        assert move in self.moves, f"Invalid move {move} for player {p}"
        hand_card = "_:T" if move.startswith("_:T=") else move
        self.hands[p].remove(hand_card)
        self._trick.append((p, move))
        self._update_lead(move)

        self.current_winner, self.current_top = _winning_play(
            self._trick, self.current_color
        )

        if len(self._trick) == self.num_players:
            self._complete_trick()
            return

        self.current_player_ = (p + 1) % self.num_players
        self.moves = self.gen_moves()

    def display(self, force=-1) -> str:
        if force == -1:
            player = self.current_player_
        else:
            assert force in range(self.num_players)
            player = force
        round_number = min(self.round_, self.num_rounds)
        lead = self.current_color if not self._lead_suit_pending else "-"
        visible_bids = self.bids if self.phase != "bid" else []
        current_trick = " ".join(f"{p}:{card}" for p, card in self._trick) or "-"
        history = (
            " | ".join(
                " ".join(f"{p}:{card}" for p, card in trick)
                for trick in self._played_tricks
            )
            or "-"
        )
        return (
            f"Round: {round_number}/{self.num_rounds}\n"
            f"Phase: {self.phase}\n"
            f"Current player: {self.current_player_}\n"
            f"Scores: {' '.join(map(str, self.the_points))}\n"
            f"Bids: {' '.join(map(str, visible_bids)) or '-'}\n"
            f"Tricks: {' '.join(map(str, self.tricks))}\n"
            f"Lead: {lead}\n"
            f"Top: {self.current_top}\n"
            f"Trick: {current_trick}\n"
            f"History: {history}\n"
            f"Hand: {' '.join(self.hands[player])}"
        )

    def display_with_moves(self) -> str:
        return (
            self.display() + "\nMoves\n" + "\n".join(f"@{move}" for move in self.moves)
        )

    def copy(self):
        game = SkullKing.__new__(SkullKing)
        game.num_players = self.num_players
        game.num_rounds = self.num_rounds
        game.the_points = self.the_points[:]
        game.bids = self.bids[:]
        game.tricks = self.tricks[:]
        game.round_ = self.round_
        game.round_starter = self.round_starter
        game.current_player_ = self.current_player_
        game.phase = self.phase
        game.current_color = self.current_color
        game.current_top = self.current_top
        game.current_winner = self.current_winner
        game._lead_suit_pending = self._lead_suit_pending
        game._trick = self._trick[:]
        game._played_tricks = [trick[:] for trick in self._played_tricks]
        game.captured = [cards[:] for cards in self.captured]
        game._captured_tricks = [
            [(cards[:], winner_card) for cards, winner_card in tricks]
            for tricks in self._captured_tricks
        ]
        game.round_points = self.round_points[:]
        game.hands = [hand[:] for hand in self.hands]
        game.deck = self.deck[:]
        game.moves = self.moves[:]
        return game

    def round(self) -> int:
        return self.round_

    def current_player(self) -> int:
        return self.current_player_

    def winner(self):
        if not self.ended():
            return None
        return max(range(self.num_players), key=self.points_for)

    def points_for(self, player: int) -> int:
        return self.the_points[player]

    def points(self) -> int:
        return self.points_for(self.current_player_)

    def diff_points(self) -> int:
        return self.diff_points_for(self.current_player_)

    def diff_points_for(self, player: int) -> int:
        opponents = [score for i, score in enumerate(self.the_points) if i != player]
        return self.points_for(player) - max(opponents)

    def simulate_to_end(self) -> None:
        while not self.ended():
            self.play_str(random.choice(self.moves))

    def play_idx(self, idx: int) -> None:
        self.play_str(self.moves[idx])
