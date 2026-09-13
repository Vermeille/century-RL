from types import SimpleNamespace
import signal

import torch

import boardrl.training.learner as learner_module
from boardrl.training import Learner
from boardrl.training.cuda_pause import CudaOffloadPause


class _RecordingModule:
    def __init__(self, calls):
        self.calls = calls

    def to(self, device):
        self.calls.append(("module", str(device)))
        return self


class _RecordingTensor:
    def __init__(self, calls, name, device):
        self.calls = calls
        self.name = name
        self.device = torch.device(device)

    def to(self, device):
        self.calls.append((self.name, str(device)))
        self.device = torch.device(device)
        return self


def test_ctrl_z_offloads_stops_and_restores_cuda_state(monkeypatch):
    calls = []
    model = _RecordingModule(calls)
    reference = _RecordingModule(calls)
    optimizer_step = _RecordingTensor(calls, "step", "cpu")
    optimizer_moment = _RecordingTensor(calls, "moment", "cuda")
    optimizer = SimpleNamespace(
        state={0: {"step": optimizer_step, "exp_avg": optimizer_moment}}
    )
    pause = CudaOffloadPause((model, reference), optimizer, device="cuda")

    real_is_tensor = torch.is_tensor
    monkeypatch.setattr(
        torch,
        "is_tensor",
        lambda value: value is optimizer_step
        or value is optimizer_moment
        or real_is_tensor(value),
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: calls.append(("synchronize", str(device))),
    )

    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: calls.append(("empty_cache", None)),
    )
    monkeypatch.setattr(
        "boardrl.training.cuda_pause.os.getpgrp",
        lambda: 456,
    )
    monkeypatch.setattr(
        "boardrl.training.cuda_pause.os.killpg",
        lambda pgrp, sig: calls.append(("killpg", (pgrp, sig))),
    )

    pause._request_pause(None, None)
    pause.service()

    assert calls == [
        ("synchronize", "cuda"),
        ("module", "cpu"),
        ("module", "cpu"),
        ("step", "cpu"),
        ("moment", "cpu"),
        ("empty_cache", None),
        ("killpg", (456, signal.SIGSTOP)),
        ("module", "cuda"),
        ("module", "cuda"),
        ("step", "cpu"),
        ("moment", "cuda"),
    ]


def test_ctrl_z_pause_is_disabled_for_cpu(monkeypatch):
    calls = []
    pause = CudaOffloadPause(
        (_RecordingModule(calls),),
        SimpleNamespace(state={}),
        device="cpu",
    )
    monkeypatch.setattr(
        signal,
        "signal",
        lambda *args: calls.append(args),
    )

    with pause:
        pause._pause_requested = True
        pause.service()

    assert calls == []


def test_ctrl_z_offloads_multiple_optimizer_states(monkeypatch):
    calls = []
    first_moment = _RecordingTensor(calls, "first", "cuda")
    second_moment = _RecordingTensor(calls, "second", "cuda")
    first_optimizer = SimpleNamespace(state={0: {"moment": first_moment}})
    second_optimizer = SimpleNamespace(state={0: {"moment": second_moment}})
    pause = CudaOffloadPause(
        (),
        first_optimizer,
        device="cuda",
        extra_optimizers=(second_optimizer,),
    )

    monkeypatch.setattr(
        torch,
        "is_tensor",
        lambda value: value is first_moment or value is second_moment,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    pause._offload()
    pause._restore()

    assert calls == [
        ("first", "cpu"),
        ("second", "cpu"),
        ("first", "cuda"),
        ("second", "cuda"),
    ]


def test_ctrl_z_handler_is_scoped_to_the_training_context(monkeypatch):
    calls = []
    previous = object()
    pause = CudaOffloadPause(
        (_RecordingModule(calls),),
        SimpleNamespace(state={}),
        device="cuda",
    )
    monkeypatch.setattr(signal, "getsignal", lambda sig: previous)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda sig, handler: calls.append((sig, handler)),
    )

    with pause:
        pass

    assert calls == [
        (signal.SIGTSTP, pause._request_pause),
        (signal.SIGTSTP, previous),
    ]


def test_learner_owns_pause_lifecycle_and_auxiliary_modules(monkeypatch):
    calls = []

    class RecordingPause:
        def __init__(self, modules, optimizer, *, device):
            calls.append(("init", tuple(modules), optimizer, torch.device(device)))

        def __enter__(self):
            calls.append(("enter",))

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append(("exit", exc_type, exc_value, traceback))

        def service(self):
            calls.append(("service",))

    monkeypatch.setattr(learner_module, "CudaOffloadPause", RecordingPause)
    model = torch.nn.Linear(2, 2)
    reference = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    learner = Learner(
        model,
        optimizer,
        [],
        batch_size=1,
        device="cuda",
        offload_modules=(reference,),
    )

    with learner:
        learner.safe_point()

    assert calls == [
        ("init", (model, reference), optimizer, torch.device("cuda")),
        ("enter",),
        ("service",),
        ("exit", None, None, None),
    ]
