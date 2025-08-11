"""Integration tests verifying the RL model with an LSTM backbone."""

import pytest
from boardrl.experiments.model import (
    model_learns_policy_and_value,
    model_learns_from_context,
)
from boardrl.rl.model.model import Model


pytestmark = pytest.mark.slow


def test_lstm_model_learns_policy_and_value():
    model_learns_policy_and_value(
        Model(dim=32, num_layers=2, head_size=8, backbone="lstm")
    )


def test_lstm_model_learns_from_context():
    model_learns_from_context(
        Model(dim=64, num_layers=3, head_size=8, backbone="lstm")
    )
