import json
from pathlib import Path

import pytest

pytest.importorskip("pyspiel")

import evaluate_nashconv as cli
from boardrl.checkpoints import Checkpoints
from boardrl.games import games_library
from boardrl.games.openspiel.result import NashConvResult
from boardrl.models import make_for_game
from boardrl.run import RunInfo


def save_run(directory, game="phantom_ttt"):
    RunInfo(
        Path("trainers/adversarial-advshape.py"),
        {
            "game": game,
            "eval_temperature": 0.7,
            "inference_batch_size": 13,
            "device": "cpu",
        },
        None,
        "",
    ).save(directory)


@pytest.mark.parametrize("override", [False, True])
def test_training_run_uses_final_step_and_recorded_settings(
    tmp_path,
    monkeypatch,
    capsys,
    override,
):
    save_run(tmp_path)
    for name in ("step-2.pth", "step-10.pth", "best-999.pth"):
        (tmp_path / name).touch()
    calls = {}
    policy = object()

    def load(checkpoint, **kwargs):
        calls["load"] = (checkpoint, kwargs)
        return policy

    def evaluate(model, game, **kwargs):
        assert model is policy
        calls["evaluate"] = (game, kwargs)
        return NashConvResult(0.6, 0.3, (0.2, 0.4), 7)

    monkeypatch.setattr(cli, "load_model", load)
    monkeypatch.setattr(cli, "nash_conv", evaluate)
    argv = ["--training-run", str(tmp_path)]
    output = tmp_path / "exploitability.json"
    game_spec = "phantom_ttt"
    model, temperature, batch_size, device = "agent", 0.7, 13, "cpu"
    if override:
        output = tmp_path / "explicit.json"
        game_spec = "abrupt_phantom_ttt"
        model, temperature, batch_size, device = "environment", 0.0, 5, "cuda:1"
        argv += [
            "--game",
            game_spec,
            "--model",
            model,
            "--temperature",
            str(temperature),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "--output",
            str(output),
        ]
    cli.main(argv)
    assert calls["load"] == (
        tmp_path / "step-10.pth",
        {"name": model, "device": device},
    )
    game, options = calls["evaluate"]
    assert game == game_spec
    assert str(options["spiel_game"]) == str(cli.resolve_game(game))
    assert options["temperature"] == temperature
    assert options["batch_size"] == batch_size
    assert options["backend"].traverser_name == (
        "AbruptPtttTraverser" if override else "PtttTraverser"
    )
    report = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == report
    assert report["checkpoint"] == str(tmp_path / "step-10.pth")
    assert report["model"] == model
    assert report["temperature"] == temperature
    assert report["exploitability"] == 0.3


@pytest.mark.parametrize("game", ["tictactoe", "othello"])
def test_training_run_skips_games_without_registered_exact_evaluation(
    tmp_path,
    monkeypatch,
    capsys,
    game,
):
    save_run(tmp_path, game)

    def forbid_load(*args, **kwargs):
        pytest.fail("unsupported final evaluation must not load a model")

    monkeypatch.setattr(cli, "load_model", forbid_load)
    cli.main(["--training-run", str(tmp_path)])
    assert "no registered exact evaluator" in capsys.readouterr().err
    assert not (tmp_path / "exploitability.json").exists()


def test_training_run_without_final_checkpoint_fails_clearly(tmp_path, capsys):
    save_run(tmp_path)
    (tmp_path / "best-99.pth").touch()
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--training-run", str(tmp_path)])
    assert "no step checkpoint found" in capsys.readouterr().err


@pytest.mark.parametrize("training_run", [False, True])
def test_exact_leduc_cli_evaluates_saved_policy(tmp_path, capsys, training_run):
    model = make_for_game("toy", games_library("leduc"))
    checkpoint = Checkpoints(tmp_path).save(0, {"agent": model})
    if training_run:
        save_run(tmp_path, "leduc")
        argv = ["--training-run", str(tmp_path)]
    else:
        argv = [
            str(checkpoint),
            "--game",
            "leduc",
            "--model",
            "agent",
            "--device",
            "cpu",
        ]
    cli.main(argv)
    result = json.loads(capsys.readouterr().out)
    assert result["evaluated_states"] == 936
    assert result["exploitability"] >= 0
    assert result["nash_conv"] == 2 * result["exploitability"]
    assert result["checkpoint"] == str(checkpoint)
