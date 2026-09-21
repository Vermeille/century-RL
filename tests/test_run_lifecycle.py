from types import SimpleNamespace

import pytest

from boardrl.run import trackio_run


def test_trackio_run_always_finishes_the_sink():
    finished = []
    sink = SimpleNamespace(finish=lambda: finished.append(True))

    with pytest.raises(RuntimeError, match="training failed"):
        with trackio_run(
            project="test",
            name="run",
            factory=lambda **kwargs: sink,
        ) as current:
            assert current is sink
            raise RuntimeError("training failed")

    assert finished == [True]
