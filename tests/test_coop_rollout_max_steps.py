import pytest

from trainers.coop import build_parser


def test_coop_rollout_max_steps_defaults_to_existing_limit():
    args = build_parser().parse_args([])

    assert args.rollout_max_steps == 5_000


def test_coop_rollout_max_steps_is_configurable():
    args = build_parser().parse_args(["--rollout-max-steps", "123"])

    assert args.rollout_max_steps == 123


def test_coop_rollout_max_steps_must_be_positive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--rollout-max-steps", "0"])
