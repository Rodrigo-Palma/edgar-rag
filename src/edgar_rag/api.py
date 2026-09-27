"""The service: one endpoint that answers, and one that reports readiness."""

from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from edgar_rag.answer import answer_question
from edgar_rag.config import Settings, get_settings
from edgar_rag.embeddings import (
    Embedder,
    Generator,
    ModelError,
    OllamaEmbedder,
    OllamaGenerator,
)
from edgar_rag.gate import BrierGate, CosineGate, RelevanceGate
from edgar_rag.index import FilingIndex

app = FastAPI(title="edgar-rag", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=4, ge=1, le=10)


def provide_settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(provide_settings)]


def provide_index(settings: SettingsDep) -> FilingIndex:
    """Load the index, or tell the caller the service has nothing to answer from."""
    try:
        return FilingIndex.load(Path(settings.index_dir))
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def provide_embedder(settings: SettingsDep) -> Embedder:
    return OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)


def provide_generator(settings: SettingsDep) -> Generator:
    return OllamaGenerator(settings.ollama_base_url, settings.generation_model)


@app.get("/health")
def health(settings: SettingsDep) -> dict[str, object]:
    try:
        index = FilingIndex.load(Path(settings.index_dir))
    except FileNotFoundError:
        return {"status": "no index", "indexed_filing": None, "chunks": 0}
    return {"status": "ready", "indexed_filing": index.source, "chunks": len(index.chunks)}


def provide_gate(settings: SettingsDep) -> RelevanceGate:
    """Cosine on its own, or brier with cosine behind it."""
    cosine = CosineGate(min_score=settings.min_retrieval_score)
    if not settings.brier_url:
        return cosine
    return BrierGate(
        url=settings.brier_url,
        min_confidence=settings.brier_min_confidence,
        fallback=cosine,
    )


@app.post("/ask")
def ask(
    request: AskRequest,
    settings: SettingsDep,
    index: Annotated[FilingIndex, Depends(provide_index)],
    embedder: Annotated[Embedder, Depends(provide_embedder)],
    generator: Annotated[Generator, Depends(provide_generator)],
    gate: Annotated[RelevanceGate, Depends(provide_gate)],
) -> dict[str, object]:
    try:
        answer = answer_question(
            question=request.question,
            index=index,
            embedder=embedder,
            generator=generator,
            gate=gate,
            top_k=request.top_k,
        )
    except ModelError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {**asdict(answer), "source": index.source}
