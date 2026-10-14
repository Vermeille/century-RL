import pytest
import torch

pyspiel = pytest.importorskip("pyspiel")

from boardrl.evaluation import Evaluation
from boardrl.games import games_library
from boardrl.games.openspiel.game import OpenSpielGame
from boardrl.games.openspiel.leduc.representation import LeducStateText
from boardrl.games.semantics import PointDeltaRewards, PointScores
from boardrl.models import make_for_game
from boardrl.rollouts import EndState, GameTrace, PlayerTrace, Record, Rollouts
from boardrl.rollouts import RolloutRunner
from boardrl.training import ReferenceTargets, Select, ToSamples


def test_leduc_rotates_actual_strategies_and_preserves_cached_value_ownership():
    desc = games_library("leduc")

    class TaggedStrategy:
        def __init__(self, identity):
            self.identity = identity

        async def __call__(self, game):
            # Always call/check so both players act in both betting rounds.
            logits = torch.full((len(game.moves),), -torch.inf)
            logits[game.action_ids.index(1)] = 0.0
            return logits, {
                "identity": self.identity,
                "reference_policy": logits,
                "reference_value": float(self.identity + 1),
                "reference_value_stddev": 1.0,
                "reference_max_q": float(self.identity + 1),
            }

    class CachedOnlyModel:
        def __call__(self, states):
            raise AssertionError("complete cached predictions must not be recomputed")

    games = RolloutRunner(desc.make_game, progress=False, outcome=desc.outcome).play(
        [TaggedStrategy(0), TaggedStrategy(1)],
        games=32,
        max_steps=20,
        rotate=True,
    )
    desc.rewards.make_returns(1.0)(games)
    for index, game in enumerate(games):
        assert sum(trace[-1].my_points for trace in game) == 0.0
        for trace in game:
            assert trace.strategy_id == (trace.seat_id + index) % 2
            for record in trace[:-1]:
                assert record.player == trace.seat_id
                assert record.metadata["identity"] == trace.strategy_id
                assert record.returns == pytest.approx(
                    2 * desc.rewards.scale * trace[-1].my_points
                )

    for identity in (0, 1):
        selected = Select(strategies=[identity])(games)
        assert sum(game[0].seat_id == 0 for game in selected) == 16
        samples = ToSamples()(selected)
        ReferenceTargets(
            CachedOnlyModel(), batch_size=64, discount=1.0,
            gae_lambda=1.0, value_lambda=1.0, reuse_rollout_predictions=True,
        )(samples)
        for sample in samples:
            assert sample.reference_value == identity + 1
            assert sample.td_lambda == pytest.approx(sample.returns)
            assert sample.gae == pytest.approx(sample.returns - sample.reference_value)
            if not sample.next.terminal:
                assert sample.next.reference_value == identity + 1


def _trace(actions, seat_to_strategy=(0, 1)):
    desc = games_library("leduc")
    state = pyspiel.load_game("leduc_poker").new_initial_state()
    traces = [PlayerTrace(seat, strategy) for seat, strategy in enumerate(seat_to_strategy)]
    for action in actions:
        if not state.is_chance_node():
            view = OpenSpielGame.from_state(
                state, information_state=True, text_type=LeducStateText
            )
            traces[state.current_player()].append(
                Record(view, torch.zeros(len(view.moves)), view.action_ids.index(action), {})
            )
        state.apply_action(action)
    view = OpenSpielGame.from_state(
        state, information_state=True, text_type=LeducStateText
    )
    for seat, trace in enumerate(traces):
        trace.append(EndState(view, seat, outcome=desc.outcome))
    return GameTrace(traces)


def test_leduc_descriptor_uses_scaled_chip_rewards_and_information_states():
    desc = games_library("leduc")
    assert issubclass(type(desc.scores), PointScores)
    assert type(desc.rewards) is PointDeltaRewards
    assert desc.rewards.scale == 0.1
    game = desc.make_game()
    assert game.information_state is True
    assert game.display() == game._state.information_state_string(game.current_player())
    assert "[Round1:" in game.display() and "[Round2:" in game.display()
    model = make_for_game("toy", desc)
    assert model.spec()["points_based"] is True
    assert model.value_distribution.output_size == 2


def test_leduc_rewards_apply_configured_scale_and_preserve_native_payoffs():
    desc = games_library("leduc")
    games = Rollouts([
        _trace([0, 4, 2, 0]),
        _trace([0, 4, 2, 2, 1, 5, 2, 2, 1], seat_to_strategy=(1, 0)),
    ])
    desc.rewards.make_returns(1.0)(games)
    assert games[0].by_strategy[0][-1].my_points == 1.0
    assert games[1].by_strategy[0][-1].my_points == 13.0
    for game in games:
        for trace in game:
            payoff = trace[-1].my_points
            # The point difference is twice the native zero-sum chip payoff.
            expected_return = 2 * desc.rewards.scale * payoff
            assert all(record.returns == expected_return for record in trace[:-1])
            assert sum(record.reward for record in trace[:-1]) == expected_return
    evaluation = Evaluation(("agent", "opponent"), games, outcome=desc.outcome)
    assert evaluation.win_rate() == 1.0
    assert desc.rewards.evaluation_score(evaluation) == 14.0


def test_leduc_payoff_metrics_follow_strategy_rotation_and_exclude_truncation():
    desc = games_library("leduc")
    games = Rollouts([
        _trace([0, 4, 2, 0]),
        _trace([0, 4, 2, 2, 1, 5, 2, 2, 1], seat_to_strategy=(1, 0)),
        _trace([0, 1, 1, 1, 2, 1, 1]),  # Showdown draw.
        _trace([0, 4, 2]),  # Truncated with no settled payoff.
    ])
    metrics = desc.make_metrics(games).metrics()
    assert metrics["terminal_rate"] == 0.75
    for identity, total in [("0", 14.0), ("1", -14.0)]:
        group = metrics["strategy"][identity]
        assert group["payoff_games"] == 3
        assert group["total_payoff"] == total
        assert group["avg_payoff"] == pytest.approx(total / 3)
    truncated = desc.make_metrics(Rollouts([games[-1]])).metrics()["strategy"]["0"]
    assert truncated["payoff_games"] == 0
    assert truncated["total_payoff"] == truncated["avg_payoff"] == 0.0
    assert desc.make_metrics(Rollouts([])).metrics() == {"games": 0}


def test_leduc_information_states_preserve_recall_and_hide_opponent_cards():
    information_state = games_library("leduc").make_game().information_state
    keys_by_text = {}
    text_by_key = {}
    recall_by_text = {}
    hidden_cards_by_key = {}

    def visit(state, histories=((), ())):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                visit(state.child(action), histories)
            return
        player = state.current_player()
        view = OpenSpielGame.from_state(
            state, information_state=information_state, text_type=LeducStateText
        )
        text = view.display_with_moves()
        key = (
            player, state.private_card(player), state.public_card(),
            tuple(state.round1()), tuple(state.round2()),
        )
        assert keys_by_text.setdefault(text, key) == key
        assert text_by_key.setdefault(key, text) == text
        assert recall_by_text.setdefault(text, histories[player]) == histories[player]
        hidden_cards_by_key.setdefault(key, set()).add(state.private_card(1 - player))
        assert view.action_ids == tuple(state.legal_actions())
        assert text.count("@") == len(view.moves)
        for action in state.legal_actions():
            next_histories = list(histories)
            next_histories[player] += ((text, action),)
            visit(state.child(action), tuple(next_histories))

    visit(pyspiel.load_game("leduc_poker").new_initial_state())
    assert len(keys_by_text) == 936
    assert all(len(cards) > 1 for cards in hidden_cards_by_key.values())


def test_leduc_information_state_distinguishes_histories_merged_by_observation():
    game = pyspiel.load_game("leduc_poker")
    states = []
    for actions in ([0, 1, 1, 1, 2, 1, 2, 2], [0, 1, 2, 2, 1, 2, 2]):
        state = game.new_initial_state()
        for action in actions:
            state.apply_action(action)
        states.append(state)
    assert states[0].current_player() == states[1].current_player() == 1
    assert states[0].observation_string(1) == states[1].observation_string(1)
    views = [
        OpenSpielGame.from_state(
            state, information_state=True, text_type=LeducStateText
        )
        for state in states
    ]
    assert views[0].display_with_moves() != views[1].display_with_moves()
