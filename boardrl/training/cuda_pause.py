"""Cooperative CUDA offloading controlled by ``Ctrl-Z``."""

from __future__ import annotations

import os
import signal

import torch


class CudaOffloadPause:
    """Move live training state to CPU while a CUDA process is paused.

    ``SIGTSTP`` first requests a safe pause. The learner calls :meth:`service`
    at boundaries where CUDA work has completed; training
    state is migrated there, then ``SIGSTOP`` hands the job back to the shell.
    Resuming the job with ``fg`` continues after the stop and restores CUDA.
    """

    def __init__(self, modules, optimizer, *, device):
        self.modules = tuple(modules)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.signal = getattr(signal, "SIGTSTP", None)
        self.enabled = self.device.type == "cuda" and self.signal is not None
        self._pause_requested = False
        self._previous_handler = None
        self._optimizer_devices = []

    def __enter__(self):
        if self.enabled:
            self._previous_handler = signal.getsignal(self.signal)
            signal.signal(self.signal, self._request_pause)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            signal.signal(self.signal, self._previous_handler)

    def _request_pause(self, signum, frame):
        del signum, frame
        self._pause_requested = True

    def service(self):
        """Offload at a safe point and block until the shell continues the job."""

        if not self.enabled or not self._pause_requested:
            return

        self._pause_requested = False
        print(
            "Ctrl-Z: moving CUDA training state to CPU; use `fg` to resume.",
            flush=True,
        )
        self._offload()
        torch.cuda.empty_cache()
        print("Ctrl-Z: CUDA cache released; stopping the job now.", flush=True)
        os.killpg(os.getpgrp(), signal.SIGSTOP)

        while True:
            print(f"Ctrl-Z: restoring training state to {self.device}.", flush=True)
            try:
                self._restore()
            except torch.cuda.OutOfMemoryError as error:
                self._offload(remember_optimizer_devices=False)
                torch.cuda.empty_cache()
                print(
                    f"Ctrl-Z: CUDA restore failed ({error}); returning to the "
                    "shell. Free VRAM and use `fg` to retry.",
                    flush=True,
                )
                os.killpg(os.getpgrp(), signal.SIGSTOP)
                continue

            print("Ctrl-Z: resumed.", flush=True)
            self._optimizer_devices = []
            return

    def _offload(self, *, remember_optimizer_devices=True):
        torch.cuda.synchronize(self.device)
        for module in self.modules:
            module.to(torch.device("cpu"))
        if remember_optimizer_devices:
            self._optimizer_devices = []
        for state in self.optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    if remember_optimizer_devices:
                        self._optimizer_devices.append((state, name, value.device))
                    state[name] = value.to(torch.device("cpu"))

    def _restore(self):
        for module in self.modules:
            module.to(self.device)
        for state, name, device in self._optimizer_devices:
            state[name] = state[name].to(device)
