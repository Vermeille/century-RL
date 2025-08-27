"""Integration tests verifying the RL model on simple tasks."""

import pytest
import torch
from boardrl.experiments.model import (
    model_learns_policy_and_value,
    model_learns_from_context,
    model_dyck,
)
from boardrl.rl.model.model import Model


pytestmark = pytest.mark.slow

def test_model_learns_policy_and_value():
    torch.manual_seed(0)
    model_learns_policy_and_value(
        Model(dim=32, num_layers=2, head_size=8, num_heads=4)
    )


def test_model_learns_from_context():
    torch.manual_seed(0)
    model_learns_from_context(
        Model(dim=64, num_layers=3, head_size=8, num_heads=8)
    )


def test_model_dyck():
    torch.manual_seed(0)
    model_dyck(Model(64, 4, 32, 2))
