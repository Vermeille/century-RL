from __future__ import annotations

from dataclasses import dataclass

import torch

from boardrl.games.openspiel.game import OpenSpielGame, _require_pyspiel


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


@dataclass(frozen=True)
class NashConvResult:
    nash_conv: float
    exploitability: float
    player_improvements: tuple[float, ...]
    evaluated_states: int


class CenturyOpenSpielPolicy:
    """Adapt a Century model to OpenSpiel's policy protocol."""

    def __init__(
        self,
        spiel_game,
        model,
        *,
        information_state: bool,
        temperature: float = 1.0,
    ):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        policy_lib, _ = _require_policy_modules()
        self._policy = policy_lib.Policy(spiel_game, list(range(spiel_game.num_players())))
        self.game = spiel_game
        self.player_ids = self._policy.player_ids
        self.model = model
        self.information_state = information_state
        self.temperature = temperature
        self._cache: dict[str, tuple[float, ...]] = {}

    def action_probabilities(self, state, player_id=None):
        view = OpenSpielGame.from_state(
            state,
            information_state=self.information_state,
        )
        text = view.display_with_moves()
        probabilities = self._cache.get(text)
        if probabilities is None:
            with torch.no_grad():
                logits = self.model([text]).policy[0].detach().float().cpu()
            if len(logits) != len(view.action_ids):
                raise ValueError(
                    "Century policy output does not match OpenSpiel legal actions: "
                    f"{len(logits)} logits for {len(view.action_ids)} moves"
                )
            if self.temperature == 0:
                probs = torch.zeros_like(logits)
                probs[torch.argmax(logits)] = 1.0
            else:
                probs = torch.softmax(logits / self.temperature, dim=0)
            probabilities = tuple(float(value) for value in probs.tolist())
            self._cache[text] = probabilities
        return dict(zip(view.action_ids, probabilities))

    def __call__(self, state, player_id=None):
        return self.action_probabilities(state, player_id)

    @property
    def evaluated_states(self) -> int:
        return len(self._cache)


def nash_conv(
    model,
    game_name: str = "leduc_poker",
    *,
    temperature: float = 1.0,
    use_cpp_br: bool = True,
) -> NashConvResult:
    pyspiel = _require_pyspiel()
    _, exploitability_module = _require_policy_modules()
    game = pyspiel.load_game(game_name)
    information_state = (
        game.get_type().information
        == pyspiel.GameType.Information.IMPERFECT_INFORMATION
    )
    policy = CenturyOpenSpielPolicy(
        game,
        model,
        information_state=information_state,
        temperature=temperature,
    )
    result = exploitability_module.nash_conv(
        game,
        policy,
        return_only_nash_conv=False,
        use_cpp_br=use_cpp_br,
    )
    value = float(result.nash_conv)
    return NashConvResult(
        nash_conv=value,
        exploitability=value / game.num_players(),
        player_improvements=tuple(float(x) for x in result.player_improvements),
        evaluated_states=policy.evaluated_states,
    )
