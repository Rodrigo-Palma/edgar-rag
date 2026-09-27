# edgar-rag

Question answering over SEC filings that **cites the passage it used** and **declines to answer**
when the retrieved evidence is too far from the question.

The interesting part is not the retrieval. It is the refusal: the service decides whether it is
able to answer *before* it asks a language model, because a model handed passages that do not
contain the answer will produce one anyway, and the caller cannot tell.

## What runs today

- Ingest one filing straight from EDGAR: download, split on its `Item N` headings, chunk, embed.
- `POST /ask`: retrieve, put the passages through a relevance gate, answer with numbered
  citations or abstain.
- Two gates, chosen by configuration. `CosineGate` compares the question to the passage and is
  free. `BrierGate` asks a [calibrated decision model](https://github.com/Rodrigo-Palma) whether
  the passage answers *this* question, and falls back to cosine if that service is down.
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

That third row was the useful one, and it has since been fixed. See below.

## The embedder was throwing away every proper noun

Chasing the R&D miss led to the encoder rather than the retriever. On Ollama 0.18.0 with
`nomic-embed-text`, **every capitalised token collapses onto a single vector**. Measured directly
against the API:

```
cos("Apple", "Cat")        = 1.0000      cos("apple", "petrobras")  = 0.4075
cos("Apple", "Zebra")      = 1.0000      cos("apple", "vale")       = 0.3355
```

Two sentences differing only in a company name came back byte for byte identical. A filing is
made of proper nouns, so this was discarding exactly the words that say what a passage is about.

Lower-casing the text before embedding is one line in `OllamaEmbedder`, and it has to apply to
both the index and the query or they stop agreeing. Re-indexing the same filing with it:

| Question | best score before | after | top passage |
| --- | --- | --- | --- |
| R&D spending | 0.574 | **0.635** | Item 1A → **Item 7**, which does contain the R&D figures |
| What the company designs | 0.683 | 0.742 | Item 1, unchanged |
| State of incorporation | 0.608 | 0.646 | Item 2 → Item 1 |
| Supply chain risks | 0.774 | 0.809 | Item 1A, unchanged |

The R&D question no longer abstains, and it abstained for the right reason before: retrieval
genuinely was not finding the passage.

## What the model gate buys

Ten questions against the same filing, five the filing answers and five it does not. The five it
does not are deliberately hard: not "who won the league in 1998" but "what were the revenues in
1994" and "how many employees does Petrobras have", which share almost all their vocabulary with
the filing. A cosine gate cannot reject those by construction; it only measures that the words
are nearby.

| gate | answers the answerable | wrongly answers the rest |
| --- | --- | --- |
| cosine ≥ 0.55 | 5/5 | **4/5** |
| brier ≥ 0.7 | 4/5 | **2/5** |

The model gate halves the wrong admissions, at the cost of one question it should have answered.
What it rejects is telling: Petrobras at confidence 0.046 and the airline question at 0.066, both
invisible to cosine. What it still admits wrongly is equally telling: both are wrong-*year*
questions, and a year is one token inside a long mean-pooled vector. That is the same weakness the
model reports about itself on held-out data, so the failure in production is the one its own
metrics predicted.

Run it:

```bash
python scripts/evaluate_gates.py          # needs a brier service on :8100
```

## Not there yet

Eval gate in CI over a golden set, cost and latency logging, semantic cache, drift monitor, and a
deployed instance. Those are the point of the project; this is the vertical slice they attach to.
