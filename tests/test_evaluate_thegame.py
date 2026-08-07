import pytest

from evaluate_thegame import build_parser, score_summary


def test_evaluation_defaults_to_reliable_sample_size():
    args = build_parser().parse_args([])

    assert args.game == "thegame,mode=strict"
    assert args.games == 1_000
    assert args.temperature == 0.02
    assert args.temperatures is None


def test_evaluation_accepts_temperature_sweep():
    args = build_parser().parse_args(["--temperatures", "0.001", "0.02", "0.1"])

    assert args.temperatures == [0.001, 0.02, 0.1]


def test_evaluation_accepts_gumbel_search_sweep():
    args = build_parser().parse_args(
        ["--gumbel-evals", "2", "--gumbel-q-scales", "1", "4"]
    )

    assert args.gumbel_evals == 2
    assert args.gumbel_q_scales == [1.0, 4.0]


def test_score_summary_reports_confidence_interval():
    result = score_summary([1.0, 2.0, 3.0])

    assert result["games"] == 3
    assert result["mean"] == 2.0
    assert result["stdev"] == 1.0
    assert result["ci95"] == pytest.approx(
        [2.0 - 1.96 / 3**0.5, 2.0 + 1.96 / 3**0.5]
    )
