from trainers.fightbot import build_parser


def test_fightbot_uses_outcome_value_likelihood_by_default():
    args = build_parser().parse_args([])

    assert args.value_lambda == 1.0
    assert args.value_clip_epsilon is None
    assert args.opponent_bot == "tactical_random"
    assert args.opponent_eval_strategy == "tactical_random"
