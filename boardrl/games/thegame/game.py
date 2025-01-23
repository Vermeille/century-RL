import random
from boardrl.utils import fast_sample


class TheGame:
    """
    A simplified version of 'The Game':
      - 4 piles: 2 ascending (indices 0 and 1) and 2 descending (indices 2 and 3).
      - Each pile starts at 1 (for ascending) or 100 (for descending).
      - There's a deck of cards from 2 to 99 (inclusive).
      - Each player has a hand of cards. We'll track only the current player's turn.
    """

    def __init__(self, num_players=2):
        assert 0 < num_players <= 5, "The Game supports 1 to 5 players."
        # Build the deck of 2..99
        self.deck = list(range(2, 100))
        random.shuffle(self.deck)

        # Initialize the 4 piles
        # For simplicity:
        #   piles[0], piles[1] = ascending, start at 1
        #   piles[2], piles[3] = descending, start at 100
        self.piles = [1, 1, 100, 100]

        # Deal initial hands
        # (In the real game, it can be 6 or 7 cards depending on the player count.)
        self.hands = [[] for _ in range(num_players)]
        initial_hand_size = 6 if num_players > 3 else 7
        for p in range(num_players):
            for _ in range(initial_hand_size):
                self.hands[p].append(self.deck.pop())

        # Current player index
        self.turn = 0
        self.num_players = num_players
        self.moves = self.gen_moves()

    def copy(self, randomize=False):
        """
        Returns a deep copy of the game state.
        """
        g = TheGame(num_players=self.num_players)
        g.deck = self.deck[:]
        if randomize:
            random.shuffle(g.deck)
        g.piles = self.piles[:]
        g.hands = [h[:] for h in self.hands]
        g.turn = self.turn
        g.moves = self.moves[:]
        return g

    def round(self):
        return int(self.turn // (2 * self.num_players))

    def current_player(self) -> int:
        return self.round() % self.num_players

    def display(self, force=-1) -> str:
        """
        Returns a string showing:
          - Pile states
          - Current player's hand
          - Which player's turn it is
        """
        if force == -1:
            p = self.current_player()
        else:
            assert force in range(self.num_players)
            p = force
        pile_info = ", ".join(
            f"{'asc' if i<2 else 'desc'}:{val}" for i, val in enumerate(self.piles)
        )
        hand_info = " ".join(str(c) for c in self.hands[p])
        return (
            f"Round: {self.round()}, Action: {self.turn % 2}\n"
            f"Piles: {pile_info}\n"
            f"Cards: {len(self.deck)}\n"
            f"Hand: {hand_info}\n"
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

        hand = self.hands[self.current_player()]

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

        return moves

    def play_str(self, move: str):
        """
        Parses a move string like '42->0'. If it's not in gen_moves(), raise an exception.
        Otherwise, perform the move and (optionally) draw a card (simplified).
        """
        if move not in self.moves:
            raise ValueError(f"Illegal move: {move}. Legal moves: {self.moves}")

        p = self.current_player()
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

        #  3) Optionally draw a card from the deck if available after the second player's move
        if self.deck and self.turn % 2 == 1:
            self.hands[p].append(self.deck.pop())
            self.hands[p].append(self.deck.pop())

        # Move to the next player (if you want multi-player rotation)
        self.turn += 1
        self.moves = self.gen_moves()

    def ended(self) -> bool:
        """
        The game ends when the deck is empty or the current player can't make any moves.
        """
        return not self.deck or not self.moves

    def points(self) -> int:
        """
        Returns the score for the current player.
        """
        return 99 - (len(self.deck) + sum(len(h) for h in self.hands))

    def points_for(self, player: int) -> int:
        return self.points()

    diff_points = points
    diff_points_for = points_for

    def play_idx(self, idx: int):
        """
        Plays the move at the given index.
        """
        return self.play_str(self.moves[idx])

    def play_distribution(self, x):
        idx = fast_sample(x)
        move = self.moves[idx]
        return self.play_str(move)


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
