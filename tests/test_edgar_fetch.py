"""Fair access to EDGAR: pacing, retries and the download cap, on a fake clock."""

import itertools

import httpx
import pytest

from edgar_rag.edgar.errors import EdgarError
from edgar_rag.edgar.fetch import (
    MAX_ATTEMPTS,
    MIN_REQUEST_INTERVAL_SECONDS,
    Fetcher,
    SystemClock,
)
from tests.edgar_fakes import FakeClock, RecordingTransport

URL = "https://data.sec.gov/submissions/CIK0000320193.json"
HEADERS = {"User-Agent": "Test Runner tests@ledgerworks.io"}


def _fetcher(handler, *, max_bytes: int = 1_000) -> tuple[Fetcher, RecordingTransport, FakeClock]:
    clock = FakeClock()
    transport = RecordingTransport(handler, clock)
    fetcher = Fetcher(httpx.Client(transport=transport), HEADERS, clock, max_bytes=max_bytes)
    return fetcher, transport, clock


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"{}")


def test_every_request_carries_the_declared_user_agent():
    fetcher, transport, _ = _fetcher(_ok)

    assert fetcher.get(URL) == b"{}"
    assert transport.requests[0].headers["User-Agent"] == HEADERS["User-Agent"]


def test_consecutive_requests_are_spaced_by_the_minimum_interval():
    fetcher, transport, clock = _fetcher(_ok)

    for _ in range(4):
        fetcher.get(URL)

    gaps = [later - earlier for earlier, later in itertools.pairwise(transport.times)]
    assert gaps == pytest.approx([MIN_REQUEST_INTERVAL_SECONDS] * 3)
    assert clock.sleeps == pytest.approx([MIN_REQUEST_INTERVAL_SECONDS] * 3)


def test_a_request_after_a_long_pause_does_not_wait():
    fetcher, _, clock = _fetcher(_ok)

    fetcher.get(URL)
    clock.now += 5.0
    fetcher.get(URL)

    assert clock.sleeps == []


def test_429_with_retry_after_is_retried_once_after_that_wait():
    # Adversarial case 22
    statuses = iter([429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        headers = {"Retry-After": "1"} if status == 429 else {}
        return httpx.Response(status, content=b"{}", headers=headers)

    fetcher, transport, clock = _fetcher(handler)

    assert fetcher.get(URL) == b"{}"
    assert len(transport.requests) == 2
    assert clock.sleeps == [1.0]
    assert transport.times[1] - transport.times[0] >= MIN_REQUEST_INTERVAL_SECONDS


def test_a_retry_after_shorter_than_the_interval_still_keeps_the_interval():
    statuses = iter([429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), headers={"Retry-After": "0"})

    fetcher, transport, _ = _fetcher(handler)
    fetcher.get(URL)

    assert transport.times[1] - transport.times[0] == pytest.approx(MIN_REQUEST_INTERVAL_SECONDS)


def test_a_server_that_keeps_refusing_is_given_up_after_the_last_attempt():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    fetcher, transport, clock = _fetcher(handler)

    with pytest.raises(EdgarError, match=f"after {MAX_ATTEMPTS} attempts"):
        fetcher.get(URL)
    assert len(transport.requests) == MAX_ATTEMPTS
    # No Retry-After: exponential backoff between attempts, none after the last
    assert clock.sleeps == [1.0, 2.0]


def test_403_with_retry_after_is_a_rate_limit_and_is_retried():
    statuses = iter([403, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), headers={"Retry-After": "2"})

    fetcher, transport, _ = _fetcher(handler)

    fetcher.get(URL)
    assert len(transport.requests) == 2


def test_403_without_retry_after_is_a_refusal_and_is_not_retried():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Undeclared Automated Tool")

    fetcher, transport, clock = _fetcher(handler)

    with pytest.raises(EdgarError, match="answered 403"):
        fetcher.get(URL)
    assert len(transport.requests) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("status", [301, 404])
def test_a_status_other_than_success_is_an_error_without_retry(status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers={"Location": "https://elsewhere.test/"})

    fetcher, transport, _ = _fetcher(handler)

    with pytest.raises(EdgarError, match=f"answered {status}"):
        fetcher.get(URL)
    assert len(transport.requests) == 1


def test_a_retry_after_longer_than_the_client_waits_is_an_error_not_a_hang():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "3600"})

    fetcher, transport, clock = _fetcher(handler)

    with pytest.raises(EdgarError, match="3600 s"):
        fetcher.get(URL)
    assert clock.sleeps == []


def test_a_retry_after_given_as_a_date_falls_back_to_backoff():
    statuses = iter([429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            next(statuses), headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
        )

    fetcher, _, clock = _fetcher(handler)
    fetcher.get(URL)

    assert clock.sleeps == [1.0]


def test_a_dropped_connection_is_retried():
    calls = iter([httpx.ConnectError("reset"), None])

    def handler(request: httpx.Request) -> httpx.Response:
        error = next(calls)
        if error is not None:
            raise error
        return httpx.Response(200, content=b"ok")

    fetcher, transport, _ = _fetcher(handler)

    assert fetcher.get(URL) == b"ok"
    assert len(transport.requests) == 2


def test_a_body_over_the_cap_is_abandoned_while_reading():
    chunks_sent: list[int] = []

    def endless():
        for _ in range(100):
            chunks_sent.append(1)
            yield b"x" * 400

    def handler(request: httpx.Request) -> httpx.Response:
        # No Content-Length: the cap has to hold during the read, not after it
        return httpx.Response(200, content=endless())

    fetcher, _, _ = _fetcher(handler, max_bytes=1_000)

    with pytest.raises(EdgarError, match="cap while downloading"):
        fetcher.get(URL)
    assert len(chunks_sent) == 3


def test_a_declared_length_over_the_cap_is_refused_before_reading():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 10, headers={"Content-Length": "999999"})

    fetcher, _, _ = _fetcher(handler, max_bytes=1_000)

    with pytest.raises(EdgarError, match="declares 999999 bytes"):
        fetcher.get(URL)


def test_a_body_exactly_at_the_cap_is_kept():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 1_000)

    fetcher, _, _ = _fetcher(handler, max_bytes=1_000)

    assert len(fetcher.get(URL)) == 1_000


def test_a_body_that_fails_to_decode_is_an_edgar_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not gzip", headers={"Content-Encoding": "gzip"})

    fetcher, _, _ = _fetcher(handler)

    with pytest.raises(EdgarError, match="could not read"):
        fetcher.get(URL)


def test_the_system_clock_moves_forward():
    clock = SystemClock()
    before = clock.monotonic()
    clock.sleep(0)
    assert clock.monotonic() >= before
