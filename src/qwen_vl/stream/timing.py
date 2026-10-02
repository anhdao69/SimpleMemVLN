"""Opt-in synchronized diagnostic timing, separate from headline benchmarks."""

from contextlib import contextmanager, nullcontext
import time
import torch


class StageTimer:
    def __init__(self):
        self.seconds = {}

    @contextmanager
    def measure(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.seconds[name] = (
                self.seconds.get(name, 0.0) + time.perf_counter() - start
            )


def timed(model, name):
    timer = getattr(model, "_stage_timer", None)
    return nullcontext() if timer is None else timer.measure(name)
