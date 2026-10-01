"""Test doubles for the EDGAR client: a clock that never sleeps and a fake SEC.

A plain module for the same reason as ``fakes.py``: test files import these by
name, which importing ``conftest`` does not support reliably.
"""

from collections.abc import Callable

import httpx

Handler = Callable[[httpx.Request], httpx.Response]


class FakeClock:
    """Time that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class RecordingTransport(httpx.MockTransport):
    """Answers with ``handler`` and records the URL and clock time of every request."""

    def __init__(self, handler: Handler, clock: FakeClock) -> None:
        self.requests: list[httpx.Request] = []
        self.times: list[float] = []

        def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            self.times.append(clock.now)
            return handler(request)

        super().__init__(record)

    @property
    def urls(self) -> list[str]:
        return [str(request.url) for request in self.requests]
