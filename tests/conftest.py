import httpx
import pytest

from edgar_rag.domain import Chunk
from edgar_rag.index import CorpusIndex
from tests.fakes import one_filing_index


@pytest.fixture
def index() -> CorpusIndex:
    """One filing, ``EXAMPLE``, with a passage on each axis."""
    chunks = (
        Chunk(
            chunk_id="Item 1#0", item="Item 1", title="Business", text="The Company designs phones."
        ),
        Chunk(
            chunk_id="Item 1A#0",
            item="Item 1A",
            title="Risk Factors",
            text="Supply chains may fail.",
        ),
    )
    return one_filing_index(chunks, [[1.0, 0.0], [0.0, 1.0]])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that reaches for a real socket through httpx.

    Every outbound call goes through an injected client, so a test that forgot
    to give one a ``MockTransport`` would otherwise quietly call a local
    Ollama when one happens to be running.
    """

    def refuse(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a test tried to reach {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
