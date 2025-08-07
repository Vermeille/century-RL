"""Integration tests verifying the RL model on simple tasks."""

from boardrl.experiments.model import (
    model_learns_policy_and_value,
    model_learns_from_context,
    model_dyck,
)
from boardrl.rl.model.model import Model


def test_model_learns_policy_and_value():
    model_learns_policy_and_value(Model(dim=32, num_layers=2, head_size=8))


def test_model_learns_from_context():
    model_learns_from_context(Model(dim=64, num_layers=3, head_size=8))


def test_model_dyck():
    model_dyck(Model(64, 4, 32))
