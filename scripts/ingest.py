"""Ingest one filing: download, split, embed, save the index.

Usage:
    uv run python scripts/ingest.py --cik 320193
"""

import argparse
import sys
from pathlib import Path

from edgar_rag.chunking import chunk_sections
from edgar_rag.config import get_settings
from edgar_rag.edgar.client import EdgarError, fetch_latest_filing
from edgar_rag.edgar.parse import html_to_text, split_into_sections
from edgar_rag.embeddings import ModelError, OllamaEmbedder
from edgar_rag.index import build_index

EMBED_BATCH_SIZE = 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cik", type=int, required=True, help="company CIK, e.g. 320193 for Apple")
    parser.add_argument("--form", default="10-K")
    args = parser.parse_args()

    settings = get_settings()
    try:
        filing = fetch_latest_filing(args.cik, settings.edgar_user_agent, args.form)
    except EdgarError as error:
        print(f"EDGAR: {error}", file=sys.stderr)
        return 1

    sections = split_into_sections(html_to_text(filing.html))
    chunks = chunk_sections(sections)
    print(
        f"{filing.company} {filing.form} {filing.filing_date}: "
        f"{len(sections)} sections, {len(chunks)} chunks"
    )

    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    try:
        vectors = _embed_all(embedder, tuple(chunk.text for chunk in chunks))
    except (ModelError, ValueError) as error:
        print(f"embedding: {error}", file=sys.stderr)
        return 1

    index = build_index(
        source={
            "company": filing.company,
            "form": filing.form,
            "filing_date": filing.filing_date,
            "url": filing.document_url,
        },
        chunks=chunks,
        vectors=vectors,
    )
    index.save(Path(settings.index_dir))
    print(f"index written to {settings.index_dir}")
    return 0


def _embed_all(embedder, texts: tuple[str, ...]):
    import numpy as np

    batches = [
        texts[start : start + EMBED_BATCH_SIZE] for start in range(0, len(texts), EMBED_BATCH_SIZE)
    ]
    return np.vstack([embedder.embed(batch) for batch in batches])


if __name__ == "__main__":
    raise SystemExit(main())
