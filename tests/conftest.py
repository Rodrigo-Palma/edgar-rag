import numpy as np
import pytest

from edgar_rag.chunking import Chunk
from edgar_rag.index import FilingIndex, build_index


@pytest.fixture
def index() -> FilingIndex:
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
