# edgar-rag

Question answering over SEC filings that **cites the passage it used** and **declines to answer**
when the retrieved evidence is too far from the question.

The interesting part is not the retrieval. It is the refusal: the service decides whether it is
able to answer *before* it asks a language model, because a model handed passages that do not
contain the answer will produce one anyway, and the caller cannot tell.

## What runs today

- Ingest one filing straight from EDGAR: download, split on its `Item N` headings, chunk, embed.
- `POST /ask`: retrieve, check the best cosine score against a threshold, answer with numbered
  citations or abstain.
- `GET /health`: which filing is indexed, and how many passages.
- Embeddings and generation run locally through Ollama, so a full ingest costs nothing.

## Running it

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env        # the SEC requires a real contact in EDGAR_USER_AGENT
ollama pull nomic-embed-text && ollama pull qwen3:32b

python scripts/ingest.py --cik 320193       # Apple's latest 10-K
uvicorn edgar_rag.api:app --reload
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What does the company identify as its principal competitive factors?"}'
```

## Layout

| Path | What lives there |
| --- | --- |
| `src/edgar_rag/edgar/` | EDGAR client and the 10-K parser |
| `src/edgar_rag/chunking.py` | sections into overlapping passages |
| `src/edgar_rag/embeddings.py` | Ollama embedding and generation |
| `src/edgar_rag/index.py` | vector index, cosine search, disk format |
| `src/edgar_rag/answer.py` | retrieval threshold, prompt, citations, abstention |
| `src/edgar_rag/api.py` | the service |

## Tests

```bash
python -m pytest
```

The tests carry no network and no model: a fake embedder places a question next to a passage by
construction, which is what makes the abstention path testable at all.

## Measured on the first filing

Apple's latest 10-K, 17 sections, 200 passages, embedded locally in about a minute.

| Question | Outcome |
| --- | --- |
| Principal competitive factors | answered in 6.0s, 3 citations, best score 0.70 |
| Who won the 1998 World Cup | abstained in 0.06s, best score 0.50, the model was never called |
| R&D spending | abstained: retrieval scored 0.62 but returned the wrong passages |

That third row is the useful one. The filing does discuss R&D, in 7 of the 200 passages, and the
retriever did not surface any of them, so the answer was refused rather than invented. The
refusal worked; the retrieval is what needs the next pass, and a golden set is how that gets
measured instead of guessed.

## Not there yet

Eval gate in CI over a golden set, cost and latency logging, semantic cache, drift monitor, and a
deployed instance. Those are the point of the project; this is the vertical slice they attach to.
