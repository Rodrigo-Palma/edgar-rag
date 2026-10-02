"""Record what the models answered, and answer from the recording without them.

A tape (a cassette directory) holds, per model call, the request and the
reply::

    <tape>/meta.json        the models and digests, Ollama version, options
    <tape>/embed.jsonl      {"key", "request", "response": {"vector": base64 float32}}
    <tape>/generate.jsonl   {"key", "request", "response": {text, tokens, seconds}}
    <tape>/brier.jsonl      {"key", "request", "response": {status, body}}

The key is the SHA-256 of the canonical JSON (sorted keys, no spaces, UTF-8)
of exactly what reaches the model: the embedding input after lower-casing,
the prompt with its deterministic nonce and the generation options, the brier
request body. Two calls share a key only when the model would see the same
bytes, so a changed prompt, option or passage is a miss, never a stale hit.

The embedder and the generator work the same way: with a ``live`` model each
answers from the tape when it can and otherwise calls the model and appends
the reply; without one it is a replay, and a miss raises ``TapeMiss`` telling
the reader to re-record locally. Brier is replay only: its gate was removed
(ADR-0014), and the tape of the frozen run that scored it is where its scores
come from. Vectors are stored as the raw float32 bytes, so a
replayed search ranks exactly as the recorded one did.

Rows are appended as they are recorded, so a run that stops halfway keeps
what it paid for, and ``Tape.save`` rewrites each file sorted by key.
"""

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import httpx
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ValidationError

from edgar_rag.domain import (
    Embedder,
    EmbedderSpec,
    GateDecision,
    Generation,
    Generator,
    IndexedFiling,
    NotRecorded,
    ScoredChunk,
)
from edgar_rag.lfs import is_pointer, pointer_message
from edgar_rag.models import GENERATION_OPTIONS

META_FILE = "meta.json"
EMBED = "embed"
GENERATE = "generate"
BRIER = "brier"
BRIER_PATH = "/decide"
KINDS = (EMBED, GENERATE, BRIER)
GENERATION_KEY_OPTIONS: dict[str, object] = {**GENERATION_OPTIONS, "think": False}
"""The generation options a tape is keyed by: those sent to Ollama, thinking off."""
RE_RECORD = (
    "re-record locally: `edgar-rag eval run --mode record` with Ollama running, "
    "then commit the tape"
)


class TapeMiss(NotRecorded):
    """Raised when a replay is asked for a call the tape never recorded."""


def canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tape_key(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of what the model receives."""
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _Store:
    """One kind of call: its rows by key, and the file new rows are appended to."""

    path: Path
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def read(cls, path: Path) -> "_Store":
        store = cls(path)
        if path.exists():
            content = path.read_bytes()
            if is_pointer(content):
                raise ValueError(pointer_message(path))
            for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    store.rows[str(row["key"])] = row
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{path.name}:{number} is not a tape row: {error}") from error
        return store

    def get(self, key: str) -> dict[str, Any] | None:
        return self.rows.get(key)

    def put(self, key: str, request: Mapping[str, Any], response: Mapping[str, Any]) -> None:
        row = {"key": key, "request": dict(request), "response": dict(response)}
        self.rows[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{canonical(row)}\n")

    def save(self) -> None:
        if not self.rows:
            return
        lines = (f"{canonical(self.rows[key])}\n" for key in sorted(self.rows))
        self.path.write_text("".join(lines), encoding="utf-8")


@dataclass(slots=True)
class Tape:
    """The recorded calls of one tape directory, and what recorded them."""

    root: Path
    meta: dict[str, Any]
    stores: dict[str, _Store]

    @classmethod
    def open(cls, root: Path) -> "Tape":
        """Read the tape under ``root``; an absent one is empty.

        Raises:
            ValueError: when a file in it is not a tape.
        """
        meta_path = root / META_FILE
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stores = {kind: _Store.read(root / f"{kind}.jsonl") for kind in KINDS}
        return cls(root=root, meta=meta, stores=stores)

    def has(self, kind: str) -> bool:
        return bool(self.stores[kind].rows)

    def save(self, meta: Mapping[str, Any] | None = None) -> None:
        """Rewrite every file sorted by key, and the meta when given."""
        self.root.mkdir(parents=True, exist_ok=True)
        for store in self.stores.values():
            store.save()
        if meta is not None:
            self.meta = dict(meta)
        text = json.dumps(self.meta, sort_keys=True, indent=1, ensure_ascii=False)
        (self.root / META_FILE).write_text(f"{text}\n", encoding="utf-8")

    def lookup(self, kind: str, request: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None]:
        key = tape_key({"kind": kind, **request})
        row = self.stores[kind].get(key)
        return key, None if row is None else row["response"]

    def record(
        self, kind: str, key: str, request: Mapping[str, Any], response: Mapping[str, Any]
    ) -> None:
        self.stores[kind].put(key, request, response)


def _miss(kind: str, key: str, what: str) -> TapeMiss:
    return TapeMiss(f"no {kind} recorded for {what} (key {key[:12]}); {RE_RECORD}")


def encode_vector(vector: NDArray[np.float32]) -> str:
    little_endian = np.asarray(vector, dtype="<f4")
    return base64.b64encode(little_endian.tobytes()).decode("ascii")


def decode_vector(text: str) -> NDArray[np.float32]:
    return np.frombuffer(base64.b64decode(text), dtype="<f4").astype(np.float32)


@dataclass(frozen=True, slots=True)
class TapedEmbedder:
    """An ``Embedder`` answered from a tape, and by ``live`` on a miss when there is one."""

    tape: Tape
    spec: EmbedderSpec
    live: Embedder | None = None

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("nothing to embed")
        rows = [self._one(text) for text in texts]
        return np.vstack(rows).astype(np.float32)

    def _one(self, text: str) -> NDArray[np.float32]:
        sent = text.lower() if self.spec.lowercase else text
        request = {"model": self.spec.model, "lowercase": self.spec.lowercase, "input": sent}
        key, response = self.tape.lookup(EMBED, request)
        if response is not None:
            return decode_vector(str(response["vector"]))
        if self.live is None:
            raise _miss(EMBED, key, repr(text[:60]))
        vector: NDArray[np.float32] = np.asarray(self.live.embed((text,)), dtype=np.float32)[0]
        self.tape.record(EMBED, key, request, {"vector": encode_vector(vector)})
        return vector


@dataclass(frozen=True, slots=True)
class TapedGenerator:
    """A ``Generator`` answered from a tape, and by ``live`` on a miss when there is one.

    ``model`` and ``options`` are part of the key: the same prompt sent to
    another model, or with another seed, is another call.
    """

    tape: Tape
    model: str
    options: Mapping[str, object]
    live: Generator | None = None

    def request(self, prompt: str) -> dict[str, Any]:
        return {"model": self.model, "options": dict(self.options), "prompt": prompt}

    def generate(self, prompt: str) -> Generation:
        request = self.request(prompt)
        key, response = self.tape.lookup(GENERATE, request)
        if response is not None:
            return Generation(
                text=str(response["text"]),
                prompt_tokens=response["prompt_tokens"],
                completion_tokens=response["completion_tokens"],
                seconds=float(response["seconds"]),
            )
        if self.live is None:
            raise _miss(GENERATE, key, "this prompt")
        generation = self.live.generate(prompt)
        self.tape.record(
            GENERATE,
            key,
            request,
            {
                "text": generation.text,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "seconds": round(generation.seconds, 4),
            },
        )
        return generation


class TapedBrierScores:
    """The brier scores of a frozen run, read back from its tape.

    ``BrierGate`` was removed after the headline run (ADR-0014), but that run
    scored it, and its tape keeps every request and reply. This replays them,
    so the frozen run reproduces with all three gates' scores; it never calls
    a model, and a request the tape does not hold raises ``TapeMiss``.

    It decides as the removed gate did with a threshold of 0: each passage is
    looked up on its own, the score is the most confident "yes", a reply that
    is not a valid answer leaves its passage unjudged and the decision
    degraded, and no passage at all is a rejection scored 0.
    """

    def __init__(self, tape: Tape) -> None:
        self._tape = tape

    def admits(
        self, question: str, passages: tuple[ScoredChunk, ...], filing: IndexedFiling
    ) -> GateDecision:
        if not passages:
            return GateDecision(
                False, 0.0, "retrieval returned nothing to judge", scores={BRIER: 0.0}
            )
        confidences = (self._confidence(question, scored.chunk.text) for scored in passages)
        judged = [confidence for confidence in confidences if confidence is not None]
        if not judged:
            raise TapeMiss(f"no passage has a valid brier reply on the tape: {question!r}")
        best = round(max(judged), 4)
        return GateDecision(
            admitted=True,
            confidence=best,
            reason=f"recorded brier confidence {best:.3f}",
            degraded=len(judged) < len(passages),
            scores={BRIER: best},
        )

    def _confidence(self, question: str, passage: str) -> float | None:
        """The recorded probability of "yes"; ``None`` when the reply was not an answer."""
        call = {"path": BRIER_PATH, "body": brier_request(question, passage)}
        key, response = self._tape.lookup(BRIER, call)
        if response is None:
            raise _miss(BRIER, key, f"POST {BRIER_PATH}")
        if response.get("status") != httpx.codes.OK:
            return None
        try:
            reply = _BrierReply.model_validate(response.get("body"))
        except ValidationError:
            return None
        return reply.answers[0].probabilities[1]


def brier_request(question: str, passage: str) -> dict[str, Any]:
    """The body the brier gate sent for one passage: what its tape rows are keyed by."""
    return {
        "state": passage,
        "questions": [
            {
                "name": "relevance",
                "kind": "bool",
                "prompt": f"Does the passage answer this question: {question}",
                "options": ["no", "yes"],
            }
        ],
    }


_Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)]


class _BrierAnswer(BaseModel):
    # One probability per option, in the order sent: ["no", "yes"]
    probabilities: Annotated[list[_Probability], Field(min_length=2, max_length=2)]


class _BrierReply(BaseModel):
    answers: Annotated[list[_BrierAnswer], Field(min_length=1)]
