import torch

from boardrl.training import LinearWarmupDecay


def make_schedule(initial_lr=1.0, iterations=100):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=initial_lr)
    return LinearWarmupDecay(optimizer, steps=iterations), optimizer


def test_lr_schedule_decays_to_zero_after_warmup():
    schedule, optimizer = make_schedule()

    assert schedule.step(0) == 0.0

    assert schedule.step(5) == 1.0

    assert schedule.step(100) == 0.0


def test_lr_schedule_clamps_after_final_iteration():
    schedule, optimizer = make_schedule()

    assert schedule.step(150) == 0.0
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_lr_schedule_supports_explicit_warmup_and_floor():
    _, optimizer = make_schedule(initial_lr=2.0, iterations=100)
    schedule = LinearWarmupDecay(optimizer, steps=100, warmup=10, min_scale=0.1)

    assert schedule.step(0) == 0.0

    assert schedule.step(5) == 1.0

    assert schedule.step(10) == 2.0

    assert schedule.step(100) == 0.2

    assert schedule.step(150) == 0.2
