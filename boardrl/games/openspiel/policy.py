"""Validate and convert model logits for exact, legal-action evaluation."""

from boardrl.games.openspiel.game import OpenSpielGame, _require_pyspiel
from boardrl.games.openspiel.representation import PlayerStateText


def _require_policy_modules():
    _require_pyspiel()
    try:
        from open_spiel.python import policy as policy_lib
        from open_spiel.python.algorithms import exploitability
    except ImportError as exc:
        raise RuntimeError(
            "OpenSpiel evaluation requires the optional 'open-spiel' package. "
            "Install it with `uv pip install open-spiel`."
        ) from exc
    return policy_lib, exploitability


class OpenSpielPolicy:
    """Map batched legal-action probabilities to OpenSpiel action IDs."""

    def __init__(
        self,
        spiel_game,
        predict,
        *,
        information_state: bool,
        text_type=PlayerStateText,
    ):
        self.game = spiel_game
        self.player_ids = list(range(spiel_game.num_players()))
        self.predict = predict
        self.information_state = information_state
        self.text_type = text_type
        self._cache: dict[str, tuple[float, ...]] = {}

    def precompute(self, states, *, batch_size=64):
        """Fill this evaluation's cache from batched, player-visible inputs."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        pending = {}
        for state in states:
            view = OpenSpielGame.from_state(
                state,
                information_state=self.information_state,
                text_type=self.text_type,
            )
            pending.setdefault(view.display_with_moves(), len(view.action_ids))
        items = list(pending.items())
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            predictions = self.predict(
                [text for text, _ in batch], [count for _, count in batch]
            )
            for (text, _), probabilities in zip(batch, predictions, strict=True):
                self._cache[text] = tuple(probabilities)

    def action_probabilities(self, state, player_id=None):
        view = OpenSpielGame.from_state(
            state,
            information_state=self.information_state,
            text_type=self.text_type,
        )
        text = view.display_with_moves()
        probabilities = self._cache.get(text)
        if probabilities is None:
            probabilities = tuple(self.predict([text], [len(view.action_ids)])[0])
            self._cache[text] = probabilities
        return dict(zip(view.action_ids, probabilities))

    def __call__(self, state, player_id=None):
        return self.action_probabilities(state, player_id)

    @property
    def evaluated_states(self) -> int:
        return len(self._cache)
