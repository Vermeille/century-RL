import pytest

from trainers.arena import load_trainer_module


@pytest.mark.parametrize("game", ["leduc", "phantom_ttt"])
@pytest.mark.parametrize(
    "trainer", ["adversarial-advshape", "adversarial-ppo", "adversarial-mmd"]
)
@pytest.mark.parametrize("arena", ["0", "1"])
def test_launchers_rate_during_training_then_evaluate_exactly(
    launch, game, trainer, arena
):
    result, events = launch(
        arguments=["--game", game, "--tag", "tag with spaces"],
        TRAINER=trainer,
        ARENA=arena,
    )
    assert result.returncode == 0, result.stderr
    expected = ["checkpoint-dir", "training", "exploitability"]
    if arena == "1":
        expected = [
            "checkpoint-dir",
            "arena-start",
            "training",
            "arena-finished",
            "exploitability",
        ]
    assert [event["event"] for event in events] == expected
    training = next(event["args"] for event in events if event["event"] == "training")
    assert training[:2] == ["python", f"trainers/{trainer}.py"]
    args = load_trainer_module(trainer).build_parser().parse_args(training[2:])
    assert args.game == game
    assert args.tag == "tag with spaces"
    assert "--sampled-evaluation" not in training
    assert events[-1]["args"][:3] == [
        "python",
        "evaluate_nashconv.py",
        "--training-run",
    ]
    if arena == "1":
        assert events[1]["args"][1] == events[-1]["args"][3]


def test_launcher_preserves_game_and_evaluation_overrides(launch):
    result, events = launch(
        "train_openspiel.sh",
        ["--game", "abrupt_dark_hex", "--opponent-eval-strategy", "longest_move"],
    )
    assert result.returncode == 0, result.stderr
    training = next(event["args"] for event in events if event["event"] == "training")
    args = load_trainer_module().build_parser().parse_args(training[2:])
    assert args.game == "abrupt_dark_hex"
    assert args.opponent_eval_strategy == "longest_move"


def test_benchmarks_run_leduc_then_phantom_with_per_game_settings(launch):
    result, events = launch(
        "train_openspiel_benchmarks.sh",
        ["--tag", "benchmark tag", "--seed", "7", "--evaluation-games", "32"],
    )
    assert result.returncode == 0, result.stderr
    assert [event["event"] for event in events] == [
        "checkpoint-dir",
        "arena-start",
        "training",
        "arena-finished",
        "exploitability",
    ] * 2
    parser = load_trainer_module().build_parser()
    runs = [
        parser.parse_args(event["args"][2:])
        for event in events
        if event["event"] == "training"
    ]
    leduc, phantom = runs
    assert [run.game for run in runs] == ["leduc", "phantom_ttt"]
    assert (leduc.steps, leduc.evaluation_every, leduc.perplexity_start) == (
        2000,
        25,
        2.0,
    )
    assert (phantom.steps, phantom.evaluation_every, phantom.perplexity_start) == (
        800,
        20,
        4.0,
    )
    assert phantom.environment_perplexity_start == 5.0
    assert phantom.environment_perplexity_end == 4.0
    for run in runs:
        assert run.tag == "benchmark tag"
        assert run.seed == 7
        assert run.evaluation_games == 32


@pytest.mark.parametrize("script", ["train_openspiel.sh", "train_openspiel_benchmarks.sh"])
@pytest.mark.parametrize(
    "status",
    [
        {"TRAINING_STATUS": "12"},
        {"ARENA_STATUS": "7"},
        {"TRAINING_STATUS": "130", "ARENA": "0"},
    ],
)
def test_failed_or_interrupted_training_does_not_start_exploitability(
    launch, script, status
):
    result, events = launch(script, **status)
    assert result.returncode == int(
        status.get("TRAINING_STATUS", status.get("ARENA_STATUS"))
    )
    assert "exploitability" not in [event["event"] for event in events]
    assert sum(event["event"] == "training" for event in events) == 1


def test_final_exploitability_can_be_disabled_independently(launch):
    result, events = launch(FINAL_EXPLOITABILITY="0")
    assert result.returncode == 0
    assert [event["event"] for event in events] == [
        "checkpoint-dir",
        "arena-start",
        "training",
        "arena-finished",
    ]


def test_final_exploitability_failure_is_reported(launch):
    result, events = launch(EXACT_STATUS="9")
    assert result.returncode == 9
    assert events[-1]["event"] == "exploitability"
