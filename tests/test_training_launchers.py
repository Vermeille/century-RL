from pathlib import Path

import pytest

from trainers import coop
from trainers.arena import load_trainer_module


TRAINERS = ("adversarial-advshape", "adversarial-ppo", "adversarial-mmd")
SELECTABLE = {"2048", "connectfour", "openspiel"}
COOPERATIVE = {"hanabi", "regicide", "the_game"}

# Existing recipes: game, steps, GAE/value lambdas, inference/learner batches,
# rollout games, evaluation games.
RECIPES = {
    "2048": ("2048versus,nolimits=True", 1800, 0, 1, 256, 128, 128, 128),
    "century": ("century", 600, 0.5, 0.8, 256, 128, 64, 128),
    "connectfour": ("connectfour", 1800, 0, 1, 1024, 2048, 512, 512),
    "hanabi": ("hanabi,mode=mini", 900, 0.9, 0.9, 1024, 384, 128, 512),
    "openspiel": ("leduc", 2000, 0.8, 0.9, 1024, 256, 1024, 512),
    "regicide": ("regicide", 900, 0.1, 0.9, 128, 96, 128, 512),
    "rps": ("rps", 600, 0.2, 0.5, 256, 128, 128, 256),
    "santorini": ("santorini", 600, 0.2, 0.5, 256, 128, 128, 256),
    "skullking": ("skullking,num_players=2,num_rounds=5", 600, 0.5, 0.9, 256, 128, 128, 128),
    "splendor": ("splendor", 600, 0.5, 0.8, 256, 128, 64, 128),
    "take5": ("take5,num_players=2", 1800, 0, 1, 1024, 2048, 512, 512),
    "the_game": ("thegame,mode=omni", 1800, 0, 1, 1024, 384, 128, 512),
}
CASES = [
    pytest.param(game, trainer, id=f"{game}-{trainer}")
    for game in RECIPES
    for trainer in (
        TRAINERS if game in SELECTABLE
        else ("coop",) if game in COOPERATIVE
        else ("adversarial-advshape",)
    )
]


def parsed_training(events, trainer):
    training = next(event for event in events if event["event"] == "training")
    assert training["args"][:2] == ["python", f"trainers/{trainer}.py"]
    parser = coop.build_parser() if trainer == "coop" else load_trainer_module(trainer).build_parser()
    return parser.parse_args(training["args"][2:]), training


def game_arguments(game):
    return ["--game", "leduc"] if game == "openspiel" else []


@pytest.mark.parametrize("game,trainer", CASES)
def test_launchers_preserve_game_recipes(launch, game, trainer):
    result, events = launch(
        f"train_{game}.sh", game_arguments(game), TRAINER=trainer,
    )
    assert result.returncode == 0, result.stderr
    args, training = parsed_training(events, trainer)
    assert (
        args.game, args.steps, args.gae_lambda, args.value_lambda,
        args.inference_batch_size, args.learner_batch_size,
        args.rollout_games, args.evaluation_games,
    ) == RECIPES[game]
    learning_rate = 4e-4 if game in COOPERATIVE else 2e-5 if game == "openspiel" else 6e-4
    assert args.learning_rate == learning_rate
    assert args.tag == ("YOLO" if game in COOPERATIVE else "splendor" if game == "splendor" else trainer)
    assert args.seed == 1
    assert args.trackio and args.no_progress
    assert training["extras"] == ["--extra", "trackio"]
    assert Path(training["cwd"]).name == "launch workspace"
    assert any(event["event"] == "arena-start" for event in events) == (trainer != "coop")


@pytest.mark.parametrize("game,trainer", CASES)
def test_cli_overrides_win_and_arena_uses_identical_arguments(launch, game, trainer):
    result, events = launch(
        f"train_{game}.sh",
        [
            "--game", "leduc", "--steps", "11", "--gae-lambda", "0.63",
            "--seed", "19", "--tag", "CLI tag with spaces",
            "--checkpoint-root", "/tmp/checkpoints with spaces",
        ],
        TRAINER=trainer, TAG="environment tag", SEED="7",
    )
    assert result.returncode == 0, result.stderr
    args, training = parsed_training(events, trainer)
    assert (args.game, args.steps, args.gae_lambda, args.seed, args.tag) == (
        "leduc", 11, 0.63, 19, "CLI tag with spaces",
    )
    assert args.checkpoint_root == Path("/tmp/checkpoints with spaces")
    if trainer != "coop":
        directory = next(event["args"] for event in events if event["event"] == "checkpoint-dir")
        assert directory[directory.index("--") + 1:] == training["args"][2:]
        assert directory[directory.index("--trainer") + 1] == trainer


@pytest.mark.parametrize("game,trainer", CASES)
@pytest.mark.parametrize("arena", ["0", "1"])
def test_trackio_and_arena_can_be_disabled_independently(launch, game, trainer, arena):
    result, events = launch(
        f"train_{game}.sh", game_arguments(game),
        TRAINER=trainer, TRACKIO="0", ARENA=arena, FINAL_EXPLOITABILITY="0",
    )
    assert result.returncode == 0, result.stderr
    args, _ = parsed_training(events, trainer)
    assert not args.trackio
    assert all(event["extras"] == [] for event in events)
    assert any(event["event"] == "arena-start" for event in events) == (
        arena == "1" and trainer != "coop"
    )
    assert not any(event["event"] == "exploitability" for event in events)


@pytest.mark.parametrize("game,trainer", CASES)
def test_help_bypasses_arena_and_final_evaluation(launch, game, trainer):
    result, events = launch(
        f"train_{game}.sh", ["--help"], TRAINER=trainer,
    )
    assert result.returncode == 0, result.stderr
    assert [event["event"] for event in events] == ["help"]


@pytest.mark.parametrize("game", sorted(SELECTABLE))
def test_unknown_trainer_fails_before_starting_any_process(launch, game):
    result, events = launch(f"train_{game}.sh", [], TRAINER="unknown")
    assert result.returncode == 2
    assert "unsupported TRAINER: unknown" in result.stderr
    assert events == []


@pytest.mark.parametrize("game", sorted(SELECTABLE))
@pytest.mark.parametrize("trainer", TRAINERS)
def test_existing_environment_overrides(launch, game, trainer):
    result, events = launch(
        f"train_{game}.sh", game_arguments(game), TRAINER=trainer,
        TAG="environment tag", SEED="17",
        AGENT_PPL_START="2.7", AGENT_PPL_END="1.2",
        ENVIRONMENT_PPL_START="3.7", ENVIRONMENT_PPL_END="1.7",
        AGENT_THRESHOLD="0.83", ENVIRONMENT_THRESHOLD="0.61",
        MMD_COEFFICIENT="0.08", MMD_REFERENCE_TIMESTEPS="123456",
    )
    assert result.returncode == 0, result.stderr
    args, _ = parsed_training(events, trainer)
    assert args.tag == "environment tag"
    assert args.seed == (1 if game == "connectfour" else 17)
    if trainer == "adversarial-advshape":
        assert (args.perplexity_start, args.perplexity_end) == (2.7, 1.2)
        assert (args.environment_perplexity_start, args.environment_perplexity_end) == (3.7, 1.7)
        assert (args.agent_threshold, args.environment_threshold) == (0.83, 0.61)
    elif trainer == "adversarial-mmd":
        assert (args.mmd_coefficient, args.mmd_reference_timesteps) == (0.08, 123456)
    elif game != "2048":
        assert (args.perplexity_start, args.perplexity_end) == (2.7, 1.2)
    else:
        assert (args.perplexity_start, args.perplexity_end) == (2.5, 1.5)


@pytest.mark.parametrize("cli", [[], ["--tag", "CLI tag", "--perplexity-start", "3.3"]])
def test_dotenv_is_loaded_and_exported_before_recipe_overrides(launch, cli):
    result, events = launch(
        "train_century.sh", cli,
        dotenv="TRACKIO=0\nTAG='dotenv tag'\nAGENT_PPL_START=4.7\nLAUNCH_TEST_MARKER=exported\n",
        TRACKIO="1", TAG="process tag", AGENT_PPL_START="2.7",
    )
    assert result.returncode == 0, result.stderr
    args, training = parsed_training(events, "adversarial-advshape")
    assert not args.trackio
    assert args.tag == ("CLI tag" if cli else "dotenv tag")
    assert args.perplexity_start == (3.3 if cli else 4.7)
    assert training["marker"] == "exported"


def test_explicit_trackio_request_includes_optional_dependency(launch):
    result, events = launch("train_regicide.sh", ["--trackio"], TRACKIO="0")
    assert result.returncode == 0, result.stderr
    args, training = parsed_training(events, "coop")
    assert args.trackio
    assert training["extras"] == ["--extra", "trackio"]


@pytest.mark.parametrize("script", ["train_century.sh", "train_openspiel.sh"])
def test_checkpoint_resolution_failure_does_not_start_training(launch, script):
    result, events = launch(script, ["--game", "leduc"], CHECKPOINT_STATUS="23")
    assert result.returncode == 23
    assert [event["event"] for event in events] == ["checkpoint-dir"]
