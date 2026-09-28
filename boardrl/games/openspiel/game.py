from __future__ import annotations

import random


def _require_pyspiel():
    try:
        import pyspiel
    except ImportError as exc:
        raise RuntimeError(
            "OpenSpiel games require the optional 'open-spiel' package. "
            "Install it with `uv pip install open-spiel`."
        ) from exc
    return pyspiel


class OpenSpielGame:
    """Expose a sequential OpenSpiel game through Century's game protocol.

    Chance nodes are resolved internally, so rollout code only ever observes
    player decision states. Imperfect-information games expose information-state
    strings; perfect-information games expose observations (which are usually a
    much more compact board representation than full action history).
    """

    __slots__ = (
        "game_name",
        "num_players",
        "_game",
        "_state",
        "_information_state",
        "_turn",
        "_actions",
        "moves",
    )

    def __init__(
        self,
        game_name: str,
        num_players: int = 2,
        *,
        spiel_game=None,
        state=None,
        information_state: bool | None = None,
        resolve_chance: bool = True,
    ):
        self.game_name = game_name
        self._game = spiel_game or _require_pyspiel().load_game(game_name)
        self.num_players = self._game.num_players()
        if self.num_players != num_players:
            raise ValueError(
                f"OpenSpiel game {game_name!r} has {self.num_players} players, "
                f"but Century requested {num_players}"
            )

        if information_state is None:
            pyspiel = _require_pyspiel()
            information_state = (
                self._game.get_type().information
                == pyspiel.GameType.Information.IMPERFECT_INFORMATION
            )
        self._information_state = information_state
        self._state = self._game.new_initial_state() if state is None else state.clone()
        self._turn = 0
        if resolve_chance:
            self._resolve_chance()
        self._refresh_moves()

    @classmethod
    def from_state(cls, state, *, information_state: bool):
        game = state.get_game()
        return cls(
            game.get_type().short_name,
            num_players=game.num_players(),
            spiel_game=game,
            state=state,
            information_state=information_state,
            resolve_chance=False,
        )

    @property
    def information_state(self) -> bool:
        return self._information_state

    @property
    def action_ids(self) -> tuple[int, ...]:
        return tuple(self._actions)

    def _resolve_chance(self) -> None:
        while self._state.is_chance_node():
            outcomes = self._state.chance_outcomes()
            if not outcomes:
                raise RuntimeError("OpenSpiel chance node has no outcomes")
            actions, probabilities = zip(*outcomes)
            self._state.apply_action(random.choices(actions, weights=probabilities, k=1)[0])

    def _action_label(self, action: int) -> str:
        label = self._state.action_to_string(self._state.current_player(), action)
        if "@" in label or "\n" in label:
            raise ValueError(
                f"OpenSpiel action label {label!r} cannot be represented by Century"
            )
        return label

    def _refresh_moves(self) -> None:
        if self.ended() or self._state.is_chance_node():
            self._actions = []
            self.moves = []
            return
        self._actions = list(self._state.legal_actions())
        self.moves = [self._action_label(action) for action in self._actions]
        if len(set(self.moves)) != len(self.moves):
            raise ValueError(
                f"OpenSpiel game {self.game_name!r} produced duplicate action labels"
            )

    def current_player(self) -> int:
        return self._state.current_player()

    def round(self) -> int:
        return self._turn // self.num_players

    def display(self, force: int = -1) -> str:
        player = force
        if player == -1:
            player = 0 if self.ended() else self.current_player()
        if not 0 <= player < self.num_players:
            raise ValueError(f"invalid viewer {player}")
        if self._information_state:
            return self._state.information_state_string(player)
        return self._state.observation_string(player)

    def display_with_moves(self) -> str:
        state = self.display()
        moves = "\n".join(f"@{move}" for move in self.moves)
        return f"{state}\n{moves}" if moves else state

    def play_str(self, move: str) -> None:
        try:
            index = self.moves.index(move)
        except ValueError as exc:
            raise ValueError(f"illegal move {move!r}; legal moves: {self.moves}") from exc
        self.play_idx(index)

    def play_idx(self, index: int) -> None:
        if self.ended():
            raise ValueError("cannot play a terminal OpenSpiel state")
        self._state.apply_action(self._actions[index])
        self._turn += 1
        self._resolve_chance()
        self._refresh_moves()

    def ended(self) -> bool:
        return self._state.is_terminal()

    def winner(self):
        if not self.ended():
            return None
        returns = self._state.returns()
        best = max(returns)
        winners = [player for player, value in enumerate(returns) if value == best]
        return winners[0] if len(winners) == 1 else None

    def points_for(self, player: int) -> float:
        return float(self._state.returns()[player]) if self.ended() else 0.0

    def points(self) -> float:
        return self.points_for(self.current_player()) if not self.ended() else 0.0

    def diff_points_for(self, player: int) -> float:
        mine = self.points_for(player)
        opponents = (
            self.points_for(other)
            for other in range(self.num_players)
            if other != player
        )
        return mine - max(opponents, default=mine)

    def diff_points(self) -> float:
        return self.diff_points_for(self.current_player()) if not self.ended() else 0.0

    def copy(self):
        copied = OpenSpielGame.__new__(OpenSpielGame)
        copied.game_name = self.game_name
        copied.num_players = self.num_players
        copied._game = self._game
        copied._state = self._state.clone()
        copied._information_state = self._information_state
        copied._turn = self._turn
        copied._actions = self._actions.copy()
        copied.moves = self.moves.copy()
        return copied

    def simulate_to_end(self):
        while not self.ended():
            self.play_idx(random.randrange(len(self.moves)))
        return self._state.returns()
