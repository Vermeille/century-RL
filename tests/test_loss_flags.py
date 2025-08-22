from boardrl.rl.model.loss import loss_from_string


def test_losses_define_reference_flag():
    for name, (cls, _) in loss_from_string.registry.items():
        assert hasattr(cls, "needs_reference_policy_value"), f"{name} missing flag"
