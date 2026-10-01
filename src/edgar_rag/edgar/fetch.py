"""GET from EDGAR within the SEC's fair access rules.

The SEC allows ten requests per second per sender and answers anything faster
with 429 or a temporary 403. This module keeps every request at least
``MIN_REQUEST_INTERVAL_SECONDS`` after the previous one, honours ``Retry-After``,
and gives up after ``MAX_ATTEMPTS``. Bodies are read as a stream and abandoned as
soon as they pass the size cap, so a runaway response never sits in memory.

Time comes from an injected ``Clock`` so the tests can check the pacing without
sleeping.
"""

import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from edgar_rag.edgar.errors import EdgarError

MIN_REQUEST_INTERVAL_SECONDS = 0.15  # at most 6.7 requests per second, under the SEC's 10
MAX_ATTEMPTS = 3
FIRST_BACKOFF_SECONDS = 1.0  # doubled on each retry the server did not time itself
MAX_RETRY_AFTER_SECONDS = 60.0
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30.0
# 403 is how EDGAR also refuses an undeclared sender, which waiting does not fix:
# it is only retried when the server says when to come back.
ALWAYS_RETRIED = frozenset({429, 500, 502, 503, 504})
RETRIED_WITH_RETRY_AFTER = frozenset({403})


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class Throttle:
    """Keeps consecutive requests at least ``interval`` seconds apart."""

    def __init__(self, clock: Clock, interval: float = MIN_REQUEST_INTERVAL_SECONDS) -> None:
        self._clock = clock
        self._interval = interval
        self._last_start: float | None = None

    def wait(self) -> None:
        if self._last_start is not None:
            elapsed = self._clock.monotonic() - self._last_start
            if elapsed < self._interval:
                self._clock.sleep(self._interval - elapsed)
        self._last_start = self._clock.monotonic()


@dataclass(frozen=True, slots=True)
class _Retry:
    delay: float
    cause: str


class Fetcher:
    """Throttled, retried and size-capped GETs that return the body as bytes."""

    def __init__(
        self,
        http: httpx.Client,
        headers: dict[str, str],
        clock: Clock,
        max_bytes: int = MAX_DOWNLOAD_BYTES,
    ) -> None:
        self._http = http
        self._headers = dict(headers)
        self._clock = clock
        self._throttle = Throttle(clock)
        self._max_bytes = max_bytes

    def get(self, url: str) -> bytes:
        """Download ``url``.

        Raises:
            EdgarError: on a refusal that retrying cannot fix, after the last
                attempt, or when the body passes the size cap.
        """
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle.wait()
            outcome = self._attempt(url, attempt)
            if isinstance(outcome, bytes):
                return outcome
            if attempt == MAX_ATTEMPTS:
                raise EdgarError(f"{url} failed after {MAX_ATTEMPTS} attempts: {outcome.cause}")
            self._clock.sleep(outcome.delay)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _attempt(self, url: str, attempt: int) -> bytes | _Retry:
        try:
            with self._http.stream("GET", url, headers=self._headers) as response:
                retry = _retry_for(response, attempt)
                if retry is not None:
                    return retry
                if not response.is_success:
                    raise EdgarError(f"{url} answered {response.status_code}")
                return self._read_capped(url, response)
        except httpx.TransportError as error:
            return _Retry(_backoff(attempt), f"{type(error).__name__}: {error}")
        except httpx.HTTPError as error:
            raise EdgarError(f"could not read {url}: {error}") from error

    def _read_capped(self, url: str, response: httpx.Response) -> bytes:
        declared = response.headers.get("Content-Length", "")
        if declared.isdigit() and int(declared) > self._max_bytes:
            raise EdgarError(f"{url} declares {declared} bytes, over the {self._max_bytes} cap")
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            # Checked while reading, so a body that lies about its length still stops here
            if len(body) > self._max_bytes:
                raise EdgarError(f"{url} passed the {self._max_bytes} byte cap while downloading")
        return bytes(body)


def _retry_for(response: httpx.Response, attempt: int) -> _Retry | None:
    status = response.status_code
    retry_after = _retry_after_seconds(response)
    if status in RETRIED_WITH_RETRY_AFTER and retry_after is None:
        return None
    if status not in ALWAYS_RETRIED | RETRIED_WITH_RETRY_AFTER:
        return None
    if retry_after is not None and retry_after > MAX_RETRY_AFTER_SECONDS:
        raise EdgarError(
            f"{response.url} asks to wait {retry_after:.0f} s, "
            f"longer than the {MAX_RETRY_AFTER_SECONDS:.0f} s this client waits"
        )
    delay = retry_after if retry_after is not None else _backoff(attempt)
    return _Retry(delay, f"answered {status}")


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Read ``Retry-After`` in seconds; the HTTP-date form falls back to backoff."""
    value = response.headers.get("Retry-After", "").strip()
    if not value.isdigit():
        return None
    return float(value)


def _backoff(attempt: int) -> float:
    return float(FIRST_BACKOFF_SECONDS * 2 ** (attempt - 1))
