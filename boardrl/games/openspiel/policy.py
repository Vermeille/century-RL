"""Validate and convert model logits for exact, legal-action evaluation."""

import math

import torch

from boardrl.games.openspiel.game import OpenSpielGame, _require_pyspiel


class ModelPolicy:
    def __init__(self, model, temperature=1.0):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        self.model = model
        self.temperature = temperature

    def probabilities(self, logits, action_count):
        logits = logits.detach().double().cpu()
        if logits.ndim != 1 or len(logits) != action_count or action_count == 0:
            raise ValueError("policy output does not match OpenSpiel legal actions")
        if (
            torch.isnan(logits).any()
            or torch.isposinf(logits).any()
            or not torch.isfinite(logits).any()
        ):
            raise ValueError(
                "policy logits must define a finite probability distribution"
            )
        if self.temperature == 0:
            probabilities = torch.zeros_like(logits)
            probabilities[torch.argmax(logits)] = 1.0
        else:
            probabilities = torch.softmax(
                (logits - logits.max()) / self.temperature, dim=0
            )
        if not torch.isfinite(probabilities).all():
            raise ValueError(
                "policy logits must define a finite probability distribution"
            )
        return probabilities.numpy()

    def predict(self, texts, action_counts):
        with torch.no_grad():
            predictions = self.model(texts).policy
        if len(predictions) != len(texts):
            raise ValueError("policy output does not match evaluation batch")
        return [
            self.probabilities(logits, count)
            for logits, count in zip(predictions, action_counts, strict=True)
        ]


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
    """Adapt a Century model to OpenSpiel's policy protocol."""

    def __init__(
        self,
        spiel_game,
        model,
        *,
        information_state: bool,
        temperature: float = 1.0,
    ):
        policy_lib, _ = _require_policy_modules()
        self._policy = policy_lib.Policy(
            spiel_game, list(range(spiel_game.num_players()))
        )
        self.game = spiel_game
        self.player_ids = self._policy.player_ids
        self.model = model
        self.information_state = information_state
        self.temperature = temperature
        self._model_policy = ModelPolicy(model, temperature)
        self._cache: dict[str, tuple[float, ...]] = {}

    def _probabilities(self, logits, action_count):
        return tuple(self._model_policy.probabilities(logits, action_count))

    def precompute(self, states, *, batch_size=64):
        """Fill this evaluation's cache from batched, player-visible inputs."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        pending = {}
        for state in states:
            view = OpenSpielGame.from_state(
                state, information_state=self.information_state
            )
            pending.setdefault(view.display_with_moves(), len(view.action_ids))
        items = list(pending.items())
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                predictions = self.model([text for text, _ in batch]).policy
                if len(predictions) != len(batch):
                    raise ValueError(
                        "Century policy output does not match evaluation batch"
                    )
                for (text, count), logits in zip(batch, predictions):
                    self._cache[text] = self._probabilities(logits, count)

    def action_probabilities(self, state, player_id=None):
        view = OpenSpielGame.from_state(
            state,
            information_state=self.information_state,
        )
        text = view.display_with_moves()
        probabilities = self._cache.get(text)
        if probabilities is None:
            with torch.no_grad():
                logits = self.model([text]).policy[0]
            probabilities = self._probabilities(logits, len(view.action_ids))
            self._cache[text] = probabilities
        return dict(zip(view.action_ids, probabilities))

    def __call__(self, state, player_id=None):
        return self.action_probabilities(state, player_id)

    @property
    def evaluated_states(self) -> int:
        return len(self._cache)
