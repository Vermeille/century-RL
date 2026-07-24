import pytest

from evaluate_thegame import build_parser, score_summary


def test_evaluation_defaults_to_reliable_sample_size():
    args = build_parser().parse_args([])

    assert args.games == 1_000
    assert args.temperature == 0.02


def test_score_summary_reports_confidence_interval():
    result = score_summary([1.0, 2.0, 3.0])

    assert result["games"] == 3
    assert result["mean"] == 2.0
    assert result["stdev"] == 1.0
    assert result["ci95"] == pytest.approx(
        [2.0 - 1.96 / 3**0.5, 2.0 + 1.96 / 3**0.5]
    )
