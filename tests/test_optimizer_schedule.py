import math

import pytest
import torch

from boardrl.training import (
    CosineWarmupDecay,
    LearningRateScheduler,
    LinearWarmupDecay,
    Scheduler,
)


def make_schedule(initial_lr=1.0, iterations=100):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=initial_lr)
    return LinearWarmupDecay(optimizer, end=1.0, warmup=5 / iterations), optimizer


def test_scheduler_maps_a_normalized_window_and_shape():
    schedule = Scheduler(start=0.2, end=0.8, shape="linear")

    assert schedule.to_schedule(0.0) == 0.0
    assert schedule.to_schedule(0.2) == 0.0
    assert schedule.to_schedule(0.5) == pytest.approx(0.5)
    assert schedule.to_schedule(1.0) == 1.0


def test_scheduler_supports_warmup_and_value_interpolation():
    schedule = Scheduler(
        start=0.5,
        end=1.0,
        warmup=0.2,
        start_value=1.0,
        end_value=0.1,
    )

    assert schedule.to_schedule(0.0) == 0.0
    assert schedule.to_schedule(0.1) == 0.5
    assert schedule.to_schedule(0.4) == 1.0
    assert schedule.to_schedule(0.75) == pytest.approx(0.55)
    assert schedule.to_schedule(1.0) == 0.1


def test_scheduler_can_be_defined_with_step_boundaries():
    schedule = Scheduler.from_steps(
        total_steps=101,
        start_step=20,
        end_step=80,
        warmup_steps=10,
    )

    assert schedule.to_schedule(0.0) == 0.0
    assert schedule.to_schedule(0.1) == 0.0
    assert schedule.to_schedule(0.5) == pytest.approx(0.5)
    assert schedule.to_schedule(0.8) == 1.0


def test_lr_schedule_decays_to_zero_after_warmup():
    schedule, optimizer = make_schedule()

    assert schedule.step(0.0) == 0.0

    assert schedule.step(0.05) == 1.0

    assert schedule.step(1.0) == 0.0


def test_lr_schedule_clamps_after_final_iteration():
    schedule, optimizer = make_schedule()

    assert schedule.step(1.5) == 0.0
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_lr_schedule_can_delay_decay_until_a_start_iteration():
    _, optimizer = make_schedule(initial_lr=2.0, iterations=100)
    schedule = LinearWarmupDecay(
        optimizer,
        warmup=0.1,
        start=0.2,
        end=1.0,
    )

    assert schedule.step(0.0) == 0.0
    assert schedule.step(0.05) == 1.0
    assert schedule.step(0.1) == 2.0
    assert schedule.step(0.19) == 2.0
    assert schedule.step(0.2) == 2.0
    assert schedule.step(0.3) == pytest.approx(2.0 * 0.875)


def test_lr_schedule_supports_explicit_warmup_and_floor():
    _, optimizer = make_schedule(initial_lr=2.0, iterations=100)
    schedule = LinearWarmupDecay(
        optimizer,
        end=1.0,
        warmup=0.1,
        min_scale=0.1,
    )

    assert schedule.step(0.0) == 0.0

    assert schedule.step(0.05) == 1.0

    assert schedule.step(0.1) == 2.0

    assert schedule.step(1.0) == 0.2

    assert schedule.step(1.5) == 0.2


def test_cosine_lr_schedule_uses_half_cosine_decay():
    _, optimizer = make_schedule()
    schedule = CosineWarmupDecay(optimizer, end=1.0, warmup=0, min_scale=0.2)

    assert schedule.step(0.0) == 1.0
    assert math.isclose(schedule.step(0.5), 0.6)
    assert schedule.step(1.0) == 0.2


def test_learning_rate_scheduler_accepts_shape_by_name():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    schedule = LearningRateScheduler(optimizer, shape="cosine", min_scale=0.2)

    assert schedule.step(0.5) == pytest.approx(0.6)
