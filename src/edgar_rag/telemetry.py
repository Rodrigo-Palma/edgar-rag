"""What one request cost, written as one line of JSON.

A gate that quietly fell back to cosine, or a generation that took a minute,
is invisible without this. One line per request keeps the log greppable and
loadable into a table as is; money is derived from it offline, from the token
counts, not computed in the service.
"""

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

logger = logging.getLogger(__name__)

Clock = Callable[[], float]


class StageTimer:
    """Seconds spent in each named stage of one request.

    Safe to read from another thread: a request that ran out of time is
    answered and logged while its worker may still be finishing a stage.
    """

    def __init__(self, clock: Clock = time.perf_counter) -> None:
        self._clock = clock
        self._seconds: dict[str, float] = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time the block, including a block that raises."""
        started = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - started
            with self._lock:
                self._seconds[stage] = self._seconds.get(stage, 0.0) + elapsed

    def seconds(self) -> dict[str, float]:
        """A copy of the time per stage, rounded to a tenth of a millisecond."""
        with self._lock:
            return {stage: round(spent, 4) for stage, spent in self._seconds.items()}


def log_request(fields: Mapping[str, object]) -> None:
    logger.info(json.dumps(dict(fields), separators=(",", ":"), default=str))
