"""Test doubles for the ports the answering core depends on.

They live in a plain module rather than in ``conftest.py`` because test files
import them by name, and importing ``conftest`` as a module is the pattern that
breaks when pytest's rootdir or import mode changes. Fixtures stay in
``conftest.py``; classes a test constructs itself live here.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from edgar_rag.gate import GateDecision
from edgar_rag.index import ScoredChunk


class FakeEmbedder:
    """Embeds by lookup, so a test can place a question next to a passage.

    A text missing from the table raises ``KeyError``: a test that embeds
    something it did not plan for should fail loudly, not get a zero vector.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = dict(table)
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        batch = tuple(texts)
        self.calls.append(batch)
        return np.asarray([self._table[text] for text in batch], dtype=np.float32)


class FakeGenerator:
    """Replies with a fixed text and records every prompt it was given."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class FakeGate:
    """Returns a fixed decision and records what it was asked to judge.

    ``calls`` lets a test assert the gate ran before generation, or did not run
    at all.
    """

    def __init__(self, decision: GateDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[str, tuple[ScoredChunk, ...]]] = []

    @classmethod
    def admitting(cls, confidence: float = 1.0) -> "FakeGate":
        return cls(GateDecision(admitted=True, confidence=confidence, reason="fake: admitted"))

    @classmethod
    def rejecting(cls, confidence: float = 0.0) -> "FakeGate":
        return cls(GateDecision(admitted=False, confidence=confidence, reason="fake: rejected"))

    def admits(self, question: str, passages: tuple[ScoredChunk, ...]) -> GateDecision:
        self.calls.append((question, passages))
        return self.decision


class FixedNonce:
    """A nonce source that always returns the same value.

    The prompt delimiters and the refusal token carry a per-request nonce, so a
    test that asserts on the prompt or on a refusal needs to know it in
    advance. Calling the instance returns the nonce; ``calls`` counts how many
    were drawn, which is how a test checks that one request draws exactly one.
    """

    def __init__(self, value: str = "0badc0de") -> None:
        if not value or not value.isalnum():
            raise ValueError("a nonce must be a non-empty alphanumeric string")
        self.value = value
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self.value
