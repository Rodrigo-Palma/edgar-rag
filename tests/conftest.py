import numpy as np
import pytest

from edgar_rag.chunking import Chunk
from edgar_rag.index import build_index


class FakeEmbedder:
    """Embeds by lookup, so a test can place a question next to a passage."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed(self, texts: tuple[str, ...]) -> np.ndarray:
        return np.asarray([self._table[text] for text in texts], dtype=np.float32)


class FakeGenerator:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture
def index():
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
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return build_index({"company": "Example Inc"}, chunks, vectors)
