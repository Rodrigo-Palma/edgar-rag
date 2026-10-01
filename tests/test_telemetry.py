import json
import logging

import pytest

from edgar_rag.telemetry import StageTimer, log_request


class TickingClock:
    """A clock that moves forward by ``step`` seconds each time it is read."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def test_each_stage_is_timed_by_the_clock():
    timer = StageTimer(clock=TickingClock(0.25))

    with timer.measure("embed"):
        pass

    assert timer.seconds() == {"embed": 0.25}


def test_a_stage_that_fails_is_still_timed():
    """The slow call that ended in an error is the one an operator looks for."""
    timer = StageTimer(clock=TickingClock(2.0))

    with pytest.raises(RuntimeError), timer.measure("generate"):
        raise RuntimeError("model down")

    assert timer.seconds() == {"generate": 2.0}


def test_a_stage_run_twice_adds_up():
    timer = StageTimer(clock=TickingClock(0.5))

    for _ in range(2):
        with timer.measure("gate"):
            pass

    assert timer.seconds() == {"gate": 1.0}


def test_the_reading_is_a_copy_the_caller_cannot_change():
    timer = StageTimer(clock=TickingClock(1.0))
    with timer.measure("embed"):
        pass

    timer.seconds()["embed"] = 99.0

    assert timer.seconds() == {"embed": 1.0}


def test_a_request_is_logged_as_one_line_of_json(caplog):
    with caplog.at_level(logging.INFO, logger="edgar_rag.telemetry"):
        log_request({"path": "/ask", "status": 200, "stages": {"embed": 0.1}})

    assert len(caplog.records) == 1
    line = caplog.records[0].getMessage()
    assert "\n" not in line
    assert json.loads(line) == {"path": "/ask", "status": 200, "stages": {"embed": 0.1}}
