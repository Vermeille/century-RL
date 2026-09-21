import math

import pytest

from boardrl.training import Scheduler


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


def test_scheduler_supports_delayed_decay_and_warmup():
    schedule = Scheduler(
        warmup=0.1,
        start=0.2,
        end=1.0,
        start_value=1.0,
        end_value=0.0,
    )

    assert schedule.to_schedule(0.0) == 0.0
    assert schedule.to_schedule(0.05) == 0.5
    assert schedule.to_schedule(0.1) == 1.0
    assert schedule.to_schedule(0.19) == 1.0
    assert schedule.to_schedule(0.2) == 1.0
    assert schedule.to_schedule(0.3) == pytest.approx(0.875)


def test_scheduler_clamps_to_final_value():
    schedule = Scheduler(start_value=1.0, end_value=0.1)

    assert schedule.to_schedule(1.0) == 0.1
    assert schedule.to_schedule(1.5) == 0.1


def test_cosine_scheduler_uses_half_cosine_interpolation():
    schedule = Scheduler(
        shape="cosine",
        start_value=1.0,
        end_value=0.2,
    )

    assert schedule.to_schedule(0.0) == 1.0
    assert math.isclose(schedule.to_schedule(0.5), 0.6)
    assert schedule.to_schedule(1.0) == 0.2
