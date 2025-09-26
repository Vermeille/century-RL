import random


class TheGame:
    """
    A simplified version of 'The Game':
      - 4 piles: 2 ascending (indices 0 and 1) and 2 descending (indices 2 and 3).
      - Each pile starts at 1 (for ascending) or 100 (for descending).
      - There's a deck of cards from 2 to 99 (inclusive).
      - Each player has a hand of cards. We'll track only the current player's turn.
    Can configure the max_value (100 by default)
    """

    def __init__(
        self,
        num_players: int = 2,
        max_value: int = 100,
        more_than_two_actions: bool = False,
        messages: bool = False,
    ):
        assert 0 < num_players <= 5, "The Game supports 1 to 5 players."
        # Build the deck of 2..max_value-1
        self.max_value = max_value
        self.more_than_two_actions = messages or more_than_two_actions
        # Optional end-of-turn letters A..J instead of 'x'.
        self.messages = messages
        self.deck = list(range(2, max_value))
        random.shuffle(self.deck)

        # Initialize the 4 piles
        # For simplicity:
        #   piles[0], piles[1] = ascending, start at 1
        #   piles[2], piles[3] = descending, start at max_value
        self.piles = [1, 1, max_value, max_value]

        # Deal initial hands
        # (In the real game, it can be 6 or 7 cards depending on the player count.)
        self.hands = [[] for _ in range(num_players)]
        self.initial_hand_size = 6 if num_players > 3 else 7
        for p in range(num_players):
            for _ in range(self.initial_hand_size):
                self.hands[p].append(self.deck.pop())

        # Current player index
        self._round = 0
        self.action = 0
        self.curplay = 0
        self.num_players = num_players
        self.moves = self.gen_moves()
        # Track last message each player sent when ending their turn
        self._last_messages = ["" for _ in range(num_players)]

    def copy(self, randomize=False):
        """
        Returns a deep copy of the game state.
        """
        g = TheGame(
            num_players=self.num_players,
            max_value=self.max_value,
            more_than_two_actions=self.more_than_two_actions,
            messages=self.messages,
        )
        g.max_value = self.max_value
        g.deck = self.deck[:]
        if randomize:
            random.shuffle(g.deck)
        g.piles = self.piles[:]
        g.hands = [h[:] for h in self.hands]
        g._round = self._round
        g.action = self.action
        g.curplay = self.curplay
        g.num_players = self.num_players
        # Recompute legal moves from the copied state to avoid stale moves
        g.moves = g.gen_moves()
        g._last_messages = self._last_messages[:]
        return g

    def round(self):
        return self._round

    def current_player(self) -> int:
        return self.curplay

    def display(self, force=-1) -> str:
        """
        Returns a string showing:
          - Pile states
          - Current player's hand
          - Which player's turn it is
        """
        if force == -1:
            p = self.curplay
        else:
            assert force in range(self.num_players)
            p = force
        pile_info = " ".join(f"{val}" for val in self.piles)
        hand_info = " ".join(str(c) for c in self.hands[p])
        msg_line = ""
        if self.messages:
            # Show last messages from other players in order relative to current viewer.
            # Order: next player, then clockwise, excluding the viewer.
            order = [((p + i) % self.num_players) for i in range(1, self.num_players)]
            rel_msgs = [self._last_messages[i] for i in order]
            msg_line = f"Msgs: {''.join(rel_msgs)}\n"
        return (
            f"Round: {self._round}, Action: {self.action}\n"
            f"Piles: {pile_info}\n"
            f"Cards: {len(self.deck)}\n"
            f"Hand: {hand_info}\n"
            f"{msg_line}"
        )

    def display_with_moves(self) -> str:
        board = self.display()
        return board + "\n".join([f"@{m}" for m in self.moves])

    def gen_moves(self) -> list[str]:
        """
        Generates all legal moves for the current player as a list of strings.
        We'll use the format 'card->pileIndex'.
          e.g. '42->0' means 'play card 42 onto pile 0'.
        """
        moves = []
        ascending_indices = [0, 1]
        descending_indices = [2, 3]

        hand = self.hands[self.curplay]

        for card in hand:
            for pile_idx in ascending_indices:
                top_val = self.piles[pile_idx]
                # Ascending rule:
                #   card >= top_val OR (top_val - card == 10) for the "jump back by 10"
                if card >= top_val or (top_val - card == 10):
                    moves.append(f"{card}->{pile_idx}")

            for pile_idx in descending_indices:
                top_val = self.piles[pile_idx]
                # Descending rule:
                #   card <= top_val OR (card - top_val == 10) for the "jump up by 10"
                if card <= top_val or (card - top_val == 10):
                    moves.append(f"{card}->{pile_idx}")

        # After the minimum required actions have been played this turn,
        # the player may optionally end their turn with the special move 'x'.
        # Minimum is 2 when the deck still has cards, otherwise 1.
        min_actions = 2 if self.deck else 1
        if self.action >= min_actions:
            if self.messages:
                moves.extend(list("ABCDEFGHIJ"))
            else:
                moves.append("x")

        return moves

    def play_str(self, move: str):
        """
        Parses a move string like '42->0'. If it's not in gen_moves(), raise an exception.
        Otherwise, perform the move and (optionally) draw a card (simplified).
        """
        if move not in self.moves:
            raise ValueError(f"Illegal move: {move}. Legal moves: {self.moves}")

        p = self.curplay

        def next_player():
            while self.deck and len(self.hands[p]) < self.initial_hand_size:
                self.hands[p].append(self.deck.pop())
            # Next player
            self.curplay = (self.curplay + 1) % self.num_players
            self.action = 0
            self.moves = self.gen_moves()
            self._round += 1

        # Handle explicit end-of-turn move
        if move == "x":
            # Draw back up to hand size if possible
            next_player()
            return
        if self.messages and move in set("ABCDEFGHIJ"):
            # Record the letter from current player to be shown to others
            self._last_messages[p] = move
            next_player()
            return
        # Parse the move
        #   expecting 'card->pileIndex'
        card_str, pile_str = move.split("->")
        card = int(card_str)
        pile_idx = int(pile_str)

        # Execute the move:
        #  1) Remove the card from the current player's hand
        self.hands[p].remove(card)

        #  2) Update the pile
        self.piles[pile_idx] = card

        #  3) Turn progression
        # Increase the count of actions taken this turn.
        self.action += 1

        min_actions = 2 if self.deck else 1
        if self.action >= min_actions and not self.more_than_two_actions:
            next_player()
            return

        # Otherwise, stay on the same player; allow continuing plays or 'x'
        self.moves = self.gen_moves()
        # If there are no further playable card moves, auto-end the turn (draw and pass).
        if not self.messages and self.moves == ["x"]:
            next_player()

    def ended(self) -> bool:
        """
        The game ends when the deck is empty or the current player can't make any moves.
        """
        return (
            len(self.deck) == 0 and sum(len(h) for h in self.hands) == 0
        ) or not self.moves

    def points(self) -> int:
        """
        Returns the score for the current player.
        """
        # Yes, there should be a -2, but I absolutely can't stand the max score
        # being 98. It's very frustrating.
        return self.max_value - (len(self.deck) + sum(len(h) for h in self.hands))

    def points_for(self, player: int) -> int:
        return self.points()

    diff_points = points
    diff_points_for = points_for

    def play_idx(self, idx: int):
        """
        Plays the move at the given index.
        """
        return self.play_str(self.moves[idx])


# ------------------------
# Example usage:
if __name__ == "__main__":
    game = TheGame(num_players=2)
    print(game.display_with_moves())

    # Generate possible moves for the current player
    possible = game.gen_moves()
    print("Possible moves:", possible)

    # Try a move (take the first possible move)
    if possible:
        move = possible[0]
        print(f"Playing move: {move}")
        game.play_str(move)

    # Display after the move
    print(game.display_with_moves())
