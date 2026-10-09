"""The autouse ``no_network`` fixture refuses every way a test could reach a host."""

import asyncio
import socket

import httpx
import pytest

EXTERNAL = "https://example.com/"


def test_a_sync_httpx_request_to_an_external_host_fails():
    with pytest.raises(AssertionError, match="tried to reach"), httpx.Client() as client:
        client.get(EXTERNAL)


def test_an_async_httpx_request_to_an_external_host_fails():
    async def fetch() -> None:
        async with httpx.AsyncClient() as client:
            await client.get(EXTERNAL)

    with pytest.raises(AssertionError, match="tried to reach"):
        asyncio.run(fetch())


def test_a_raw_socket_connect_fails():
    with (
        pytest.raises(AssertionError, match="tried to open a socket"),
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as raw,
    ):
        raw.connect(("192.0.2.1", 80))  # TEST-NET-1, RFC 5737


def test_a_raw_socket_connect_ex_fails():
    with (
        pytest.raises(AssertionError, match="tried to open a socket"),
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as raw,
    ):
        raw.connect_ex(("192.0.2.1", 80))  # TEST-NET-1, RFC 5737
