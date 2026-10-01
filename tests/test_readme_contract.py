"""The JSON the README shows is the JSON the service returns.

The README once promised ``answer``, ``confidence`` and ``degraded`` while the
API returned ``text`` and ``retrieval_score`` and no ``degraded`` at all. This
test reads every JSON example in the README and compares it, key by key and
type by type, with a response serialised by the real endpoint. Changing the
response without the README, or the README without the response, fails here.
"""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edgar_rag.answer import AbstentionReason
from edgar_rag.api import (
    app,
    provide_embedder,
    provide_gate,
    provide_generator,
    provide_index,
    provide_settings,
)
from edgar_rag.config import ServiceSettings
from edgar_rag.gate import CosineGate
from tests.fakes import FakeEmbedder, FakeGenerator

README = Path(__file__).resolve().parents[1] / "README.md"
JSON_BLOCK = re.compile(r"```json\n(.*?)```", re.DOTALL)
QUESTION = "what does the company design?"
# Filing metadata: which keys it has depends on what was ingested, so the
# contract is only that it is an object of strings.
OPEN_OBJECTS = frozenset({"source"})


def _examples() -> list[dict[str, object]]:
    blocks = [json.loads(block) for block in JSON_BLOCK.findall(README.read_text("utf-8"))]
    return [block for block in blocks if isinstance(block, dict) and "abstained" in block]


def _example(*, abstained: bool) -> dict[str, object]:
    matching = [example for example in _examples() if example["abstained"] is abstained]
    assert matching, f"the README shows no response with abstained={abstained}"
    return matching[0]


def _served(index, *, min_score: float) -> dict[str, object]:
    app.dependency_overrides = {
        provide_settings: lambda: ServiceSettings(),
        provide_index: lambda: index,
        provide_embedder: lambda: FakeEmbedder({QUESTION: [1.0, 0.0]}),
        provide_generator: lambda: FakeGenerator("The Company designs phones [1]."),
        provide_gate: lambda: CosineGate(min_score),
    }
    try:
        response = TestClient(app).post("/ask", json={"question": QUESTION, "top_k": 2})
    finally:
        app.dependency_overrides = {}
    assert response.status_code == 200
    body: dict[str, object] = response.json()
    return body


def _kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _shape(value: object, key: str = "") -> object:
    """Keys and JSON types, recursively; the values themselves do not matter."""
    if isinstance(value, dict):
        if key in OPEN_OBJECTS:
            assert all(isinstance(item, str) for item in value.values()), key
            return "object of strings"
        return {name: _shape(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value[:1]]
    return _kind(value)


def test_the_readme_shows_one_answer_and_one_abstention():
    assert {example["abstained"] for example in _examples()} == {True, False}


@pytest.mark.parametrize(
    ("abstained", "min_score"), [(False, 0.5), (True, 1.1)], ids=["answer", "abstention"]
)
def test_the_readme_example_has_the_shape_of_a_served_response(index, abstained, min_score):
    served = _served(index, min_score=min_score)
    assert served["abstained"] is abstained

    assert _shape(_example(abstained=abstained)) == _shape(served)


def test_the_readme_abstention_uses_a_published_reason():
    reason = _example(abstained=True)["reason"]

    assert reason in {member.value for member in AbstentionReason}
