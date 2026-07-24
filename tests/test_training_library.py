import torch

from boardrl.checkpoints import Checkpoints
from boardrl.games import games_library
from boardrl.games.strategies import RandomStrategy
from boardrl.models import toy
from boardrl.rl.model.loss import ImitationCELoss
from boardrl.rl.model import load_model
from boardrl.rollouts import RolloutRunner
from boardrl.training import ComputeReturns, Learner, Pipeline, Select, ToSamples, TrainingSample


def test_rollout_pipeline_is_composable():
    game = games_library("tictactoe")
    results = RolloutRunner(game.make_game, progress=False).play(
        [RandomStrategy(), RandomStrategy()], games=2, max_steps=2
    )

    samples = Pipeline(
        ComputeReturns(1.0), Select(strategies=[0]), ToSamples()
    )(results)

    assert samples
    assert all(sample.next is not None for sample in samples)


def test_rollout_factory_owns_matchmaking():
    game = games_library("tictactoe")
    calls = []

    def lineup(index):
        calls.append(index)
        return [RandomStrategy(), RandomStrategy()]

    results = RolloutRunner(game.make_game, progress=False).play(
        lineup, games=3, max_steps=1
    )

    assert calls == [0, 1, 2]
    assert len(results) == 3


def test_checkpoints_support_multiple_models(tmp_path):
    current = toy()
    average = toy()
    optimizer = torch.optim.AdamW(current.parameters())
    checkpoints = Checkpoints(tmp_path, keep=2)

    path = checkpoints.save(
        4,
        {"current": current, "average": average},
        optimizers={"current": optimizer},
        metadata={"algorithm": "nfsp"},
    )
    restored_current = toy()
    restored_average = toy()
    payload = checkpoints.load(
        path,
        models={"current": restored_current, "average": restored_average},
    )

    assert payload["step"] == 4
    assert payload["metadata"]["algorithm"] == "nfsp"
    for expected, restored in zip(current.parameters(), restored_current.parameters()):
        assert torch.equal(expected, restored)


def test_checkpoint_can_be_loaded_as_a_rollout_model(tmp_path):
    model = toy()
    path = Checkpoints(tmp_path).save(1, {"current": model})

    loaded = load_model(path)

    assert loaded.spec() == model.spec()


def test_learner_accepts_pure_imitation_samples_without_returns():
    model = toy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    learner = Learner(
        model,
        optimizer,
        [ImitationCELoss()],
        batch_size=1,
        device="cpu",
    )
    sample = TrainingSample(
        state="board\n@left\n@right",
        action_idx=0,
        action_distribution=torch.tensor([2.0, -1.0]),
        next=None,
    )

    result = learner.train([sample])

    assert result.samples == 1
    assert result.batches == 1


def test_normalized_learner_handles_partial_batch():
    model = toy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    learner = Learner(
        model,
        optimizer,
        [ImitationCELoss()],
        batch_size=4,
        device="cpu",
        normalize_lr=True,
    )
    sample = TrainingSample(
        state="board\n@left\n@right",
        action_idx=0,
        action_distribution=torch.tensor([2.0, -1.0]),
        next=None,
    )

    result = learner.train([sample])

    assert result.samples == 1
    assert result.batches == 1
