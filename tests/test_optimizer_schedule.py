from types import SimpleNamespace

from main import Optimizer


def make_schedule(initial_lr=1.0, iterations=100):
    opt = Optimizer.__new__(Optimizer)
    opt.opt = SimpleNamespace(param_groups=[{"lr": None}])
    opt._initial_lr = initial_lr
    opt._total_iterations = iterations
    opt._warmup_epochs = None
    opt._min_lr_scale = 0.0
    opt.current_lr = initial_lr
    return opt


def test_lr_schedule_decays_to_zero_after_warmup():
    opt = make_schedule()

    opt._update_lr(0)
    assert opt.current_lr == 0.0

    opt._update_lr(5)
    assert opt.current_lr == 1.0

    opt._update_lr(100)
    assert opt.current_lr == 0.0


def test_lr_schedule_clamps_after_final_iteration():
    opt = make_schedule()

    opt._update_lr(150)
    assert opt.current_lr == 0.0
    assert opt.opt.param_groups[0]["lr"] == 0.0


def test_lr_schedule_supports_explicit_warmup_and_floor():
    opt = make_schedule(initial_lr=2.0, iterations=100)
    opt._warmup_epochs = 10
    opt._min_lr_scale = 0.1

    opt._update_lr(5)
    assert opt.current_lr == 1.0

    opt._update_lr(10)
    assert opt.current_lr == 2.0

    opt._update_lr(100)
    assert opt.current_lr == 0.2
