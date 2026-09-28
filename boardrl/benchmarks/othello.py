from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from boardrl.benchmarks.gtp import GTPEngine
from boardrl.games.openspiel.game import OpenSpielGame


@dataclass(frozen=True)
class OthelloMatchResult:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0


class ModelMoveSelector:
    def __init__(self, model, *, temperature: float = 0.0):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.model = model
        self.temperature = temperature

    def select(self, game: OpenSpielGame) -> str:
        with torch.no_grad():
            logits = self.model([game.display_with_moves()]).policy[0].detach()
        if len(logits) != len(game.moves):
            raise ValueError(
                f"model returned {len(logits)} logits for {len(game.moves)} legal moves"
            )
        if self.temperature == 0:
            index = int(torch.argmax(logits).item())
        else:
            probabilities = torch.softmax(logits / self.temperature, dim=0)
            index = int(torch.multinomial(probabilities, 1).item())
        return game.moves[index]


class OthelloEngineBenchmark:
    """Paired Othello evaluation against a GTP-compatible external engine."""

    def __init__(
        self,
        model,
        engine: GTPEngine,
        *,
        temperature: float = 0.0,
        opening_plies: int = 8,
        seed: int = 0,
    ):
        if opening_plies < 0:
            raise ValueError("opening_plies must be non-negative")
        self.selector = ModelMoveSelector(model, temperature=temperature)
        self.engine = engine
        self.opening_plies = opening_plies
        self.rng = random.Random(seed)

    def _opening(self) -> tuple[str, ...]:
        game = OpenSpielGame("othello")
        moves = []
        for _ in range(self.opening_plies):
            if game.ended():
                break
            move = self.rng.choice(game.moves)
            moves.append(move)
            game.play_str(move)
        return tuple(moves)

    @staticmethod
    def _color(player: int) -> str:
        return "b" if player == 0 else "w"

    def _play(self, agent_player: int, opening: tuple[str, ...]) -> float:
        game = OpenSpielGame("othello")
        self.engine.clear_board()

        for move in opening:
            player = game.current_player()
            self.engine.play(self._color(player), move)
            game.play_str(move)

        while not game.ended():
            player = game.current_player()
            color = self._color(player)
            if player == agent_player:
                move = self.selector.select(game)
                self.engine.play(color, move)
            else:
                move = self.engine.genmove(color)
                if move not in game.moves:
                    raise ValueError(
                        f"engine returned illegal move {move!r}; legal moves: {game.moves}"
                    )
            game.play_str(move)
        return game.points_for(agent_player)

    def run(self, *, pairs: int) -> OthelloMatchResult:
        if pairs <= 0:
            raise ValueError("pairs must be positive")
        wins = draws = losses = 0
        for _ in range(pairs):
            opening = self._opening()
            for agent_player in (0, 1):
                result = self._play(agent_player, opening)
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    draws += 1
        return OthelloMatchResult(wins=wins, draws=draws, losses=losses)
