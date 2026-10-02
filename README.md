# edgar-rag

Question answering over SEC filings that **cites the passage it used** and
**declines to answer** when the evidence does not support one.

The retrieval is not the interesting part. The refusal is: a language model
handed four passages that do not contain the answer will produce an answer
anyway, fluent and wrong, and the caller has no way to tell. So this service
decides whether it is *able* to answer before it asks the model, and every
answer it does give names the passage it came from.

```
  "How much did the company        ┌──────────────────────────────┐
   spend on R&D?"                  │ EDGAR: download the 10-K      │
         │                         │ split on its Item N headings  │
         │                         │ chunk on sentence boundaries  │
         │                         │ 17 sections → 232 passages    │
         │                         └──────────────┬───────────────┘
         ▼                                         ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ retrieve: cosine over locally-embedded passages, top 4         │
  └───────────────────────────────┬───────────────────────────────┘
                                   ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ RELEVANCE GATE ── does this passage answer THIS question?      │
  │                                                                │
  │   CosineGate   free, and cannot tell 1994 from 2024            │
  │   BrierGate    a calibrated model, degrades to cosine if down  │
  └──────────┬────────────────────────────────┬───────────────────┘
             │ admitted                        │ refused
             ▼                                 ▼
  ┌────────────────────────────┐   ┌──────────────────────────────┐
  │ generate, and REQUIRE a    │   │ abstain, with the reason and  │
  │ [n] marker pointing at a   │   │ the confidence that produced  │
  │ real passage, or abstain   │   │ it. The model is never called.│
  └────────────┬───────────────┘   └──────────────────────────────┘
               ▼
    answer + citations, each a quote from the window
    of the passage that actually overlaps the question
```

## The guarantee, and what enforces it

Every claim traceable to a filing. That sentence is worth nothing unless
something checks it, and for a while nothing did: an answer that cited no
passage at all was returned as an answer. Now the markers are parsed, matched
against the passages actually retrieved, and an answer with no valid marker
becomes an abstention whose reason is *"the answer cited no passage, so it
could not be checked"*.

The citations quote the window around the sentence with the most overlap with
the question, not the first 400 characters of the chunk, which is rarely the
part the answer used.

The filing text is untrusted input. It arrives inside `<passages>` delimiters,
labelled as untrusted document content, with the refusal token stripped out of
it so a filing cannot make the service refuse, and marker-shaped text in the
source neutralised so it cannot fake a citation.

## Measured on a real filing

Apple's latest 10-K: 17 sections, 232 passages, 232/232 unique ids, embedded
locally in about a minute.

| Question | Outcome |
|---|---|
| Principal competitive factors | **answered**, 3 citations, best score 0.70, 6.0s |
| Who won the 1998 World Cup | **abstained** in 0.06s, best score 0.50, model never called |
| How much was spent on R&D | **abstained** after generation, 19.6s |

The third row is the honest one and it is worth being precise about. Fixing the
embedder moved that question's retrieval from 0.574 to 0.635 and pulled the
right section into the top 4. It still abstains: the model reads the passages
and reports the figure is not in them. Retrieval improved; the answer did not
become available.

## The embedder was discarding every proper noun

Chasing that R&D miss led to the encoder, not the retriever. On Ollama 0.18.0
with `nomic-embed-text`, **every capitalised token collapses onto one vector**:

```
cos("Apple", "Cat")   = 1.0000        cos("apple", "petrobras") = 0.4075
cos("Apple", "Zebra") = 1.0000        cos("apple", "vale")      = 0.3355
cos("Apple", "apple") = 0.4706   ← lower-casing is NOT being applied
```

Two sentences differing only in a company name came back byte for byte
identical, and 287 distinct texts produced 119 distinct vectors. A filing is
made of proper nouns, so this was throwing away exactly the words that say what
a passage is about.

Lower-casing before embedding is one line, and it has to apply to the index and
the query or they stop agreeing. Re-indexing the same filing:

| Question | before | after | top passage moved |
|---|---|---|---|
| R&D spending | 0.574 | **0.635** | Item 1A → **Item 7**, which has the R&D figures |
| What the company designs | 0.683 | 0.742 | Item 1, unchanged |
| State of incorporation | 0.608 | 0.646 | Item 2 → Item 1 |
| Supply chain risks | 0.774 | 0.809 | Item 1A, unchanged |

This is [ollama/ollama#15609](https://github.com/ollama/ollama/issues/15609), a
regression at v0.14.0 whose cause is `BasicTokenizer` preprocessing lost in the
HF→gguf conversion. The issue frames it as a non-ASCII problem; it is wider than
that, and a 10-K is the proof, with **18.3% of its words starting with a
capital** against 2.9% non-ASCII.

## What the model gate buys

Ten questions, five the filing answers and five it does not. The five it does
not are deliberately hard: not "who won the league in 1998" but "what were the
revenues in 1994" and "how many employees does Petrobras have", which share
almost all their vocabulary with the filing. **A cosine gate cannot reject those
by construction** — it measures that the words are nearby, and they are.

Comparing two detectors at one threshold each says nothing, so both are swept:

```
  cosine                             brier
  ────────────────────────────       ────────────────────────────
  thr   admits   wrongly admits      thr   admits   wrongly admits
  0.45   5/5        4/5              0.20   5/5        3/5
  0.55   5/5        4/5              0.40   5/5        1/5   ←
  0.60   5/5        3/5              0.50   3/5        1/5
  0.65   4/5        1/5              0.60   3/5        1/5
  0.70   3/5        1/5              0.70   3/5        0/5   ←
```

At equal recall the model gate is better at both operating points:

| accepts all 5 answerable | cosine admits **4** of 5 negatives | brier admits **1** |
|---|---|---|
| accepts 3 of 5 answerable | cosine admits **1** | brier admits **0** |

**This is n=5 per class and it is not inference.** With five paired
observations the smallest two-sided p-value reachable is 0.0625, so no
arrangement of these ten questions could have produced a significant result.
What the table shows is a direction, on questions chosen to be hard for cosine.

What the model rejects is telling: Petrobras at confidence 0.046 and the airline
question at 0.066, both invisible to cosine. What it still admits wrongly is
equally telling: the survivors are wrong-*year* questions, and a year is one
token inside a long mean-pooled vector. That is the same weakness the model
reports about itself on held-out data, where wrong-year accuracy is 0.000
[0.000, 0.299] — so the failure in production is the one its own metrics
predicted, which is the most useful thing a metric can do.

```bash
python scripts/evaluate_gates.py     # needs a brier service on :8100
```

## Install and run

```bash
uv sync --frozen            # runtime and dev tools, exactly as locked in uv.lock
make check                  # lint, types, import contracts, tests with coverage
cp .env.example .env        # the SEC requires a real contact in EDGAR_RAG_EDGAR_USER_AGENT
ollama pull nomic-embed-text && ollama pull qwen3:32b

uv run edgar-rag ingest --cik 320193     # Apple's latest 10-K
uv run edgar-rag serve                   # serves on 127.0.0.1:8000
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What does the company identify as its principal competitive factors?"}'
```

An answer, as the service returned it for Apple's 10-K (long strings trimmed
to `...`, nothing else changed):

```json
{
  "question": "What does the company identify as its principal competitive factors?",
  "text": "The company identifies the following principal competitive factors: ... maintain a competitive advantage [1]. ... aggressive pricing and low cost structures ... [2]. ...",
  "citations": [
    {
      "marker": 1,
      "item": "Item 1A",
      "title": "Risk Factors",
      "quote": "on the Company's competitive advantage and materially adversely affect its business, ...",
      "score": 0.6804
    }
  ],
  "abstained": false,
  "reason": null,
  "detail": "best passage scored 0.680",
  "retrieval_score": 0.6804,
  "gate_score": 0.6804,
  "degraded": false,
  "source": {
    "company": "Apple Inc.",
    "form": "10-K",
    "filing_date": "2025-10-31",
    "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
  }
}
```

An abstention has no `text`, and says which check withheld the answer:

```json
{
  "question": "Who won the 1998 World Cup?",
  "text": null,
  "citations": [],
  "abstained": true,
  "reason": "gate_rejected",
  "detail": "No passage in this filing is close enough to the question, so the model was not asked. (best passage scored 0.405, below the 0.55 threshold)",
  "retrieval_score": 0.4051,
  "gate_score": 0.4051,
  "degraded": false,
  "source": {
    "company": "Apple Inc.",
    "form": "10-K",
    "filing_date": "2025-10-31",
    "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
  }
}
```

`reason` is one of `gate_rejected`, `out_of_period`, `model_declined`,
`no_valid_citation`, `unsupported_claim` or `out_of_scope`, and is `null` on an
answer. Today the pipeline emits the first, third and fourth; the others are
reserved for the period guard, the citation support check and multi-filing
scope, so adding them does not change the response. `gate_score` is the gate's
own confidence: cosine similarity for the cosine gate, a probability for the
model gate.

`degraded` is true when the gate decided on part of its evidence: the brier
service was unreachable and cosine answered in its place, or it could not judge
some of the passages. A fallback is never reported as a model decision. The
full schema is served at `/openapi.json`, and `tests/test_readme_contract.py`
fails if the examples above stop matching a real response.

## Operating it

The service reads the index once at startup and shares one HTTP client across
requests. At most `EDGAR_RAG_MAX_CONCURRENT_GENERATIONS` (2) generations run at
once; a request that needs another gets `503` with `Retry-After` immediately
rather than waiting in a queue, and a question the gate rejects is answered
whatever the load. A request still running after
`EDGAR_RAG_REQUEST_TIMEOUT_SECONDS` (90) gets `504`. `/health` reports the
loaded filing, a fingerprint of the index and whether Ollama answers, probing
it at most every 10 seconds.

Every request writes one line of JSON to stderr with its status, total and
per-stage seconds (`embed`, `search`, `gate`, `generate`), `reason`,
`degraded`, and the prompt and completion tokens of the generation (null when
the model was not asked, or did not report a count). The question itself is
not logged. The generator runs with `temperature=0`, `seed=0`, thinking off
and an 8192-token context, so the same prompt should get the same answer;
that is pinned, not proved, until the evaluation measures it.

There is no authentication and no rate limit, so it binds to `127.0.0.1` by
default. Both are needed before it listens anywhere else.

## Layout

| Path | What lives there |
|---|---|
| `src/edgar_rag/cli.py` | the `edgar-rag` command: `ingest`, `serve`, `eval power` |
| `src/edgar_rag/service/` | the service: composition at startup, limits, the JSON contract, error mapping |
| `src/edgar_rag/eval/` | evaluation statistics: metrics, cluster bootstrap, numeric matching, power |
| `src/edgar_rag/ingest.py` | a downloaded filing into an index: parse, chunk, embed in batches |
| `src/edgar_rag/answer.py` | the `Answerer`: retrieve, gate, generate, check the citations, or abstain |
| `src/edgar_rag/config.py` | settings for the service, the ingestion and the evaluation |
| `src/edgar_rag/prompt.py` | the generation prompt and the untrusted-text guard |
| `src/edgar_rag/citations.py` | which passages an answer cites, and the quote shown for each |
| `src/edgar_rag/chunking.py` | sections into passages, cut on sentence boundaries |
| `src/edgar_rag/gate.py` | the relevance gate: cosine, model, and the fallback |
| `src/edgar_rag/index.py` | vector index, cosine search, disk format |
| `src/edgar_rag/models.py` | Ollama embedding and generation, lower-cased and pinned |
| `src/edgar_rag/edgar/` | EDGAR client, XBRL facts and the 10-K parser |
| `src/edgar_rag/telemetry.py` | per-stage timing and the one JSON line per request |
| `src/edgar_rag/domain.py` | the values the pipeline passes around, and the ports it calls |

The table runs top to bottom in import order: a module imports only from rows
below its own layer, and the answering core reaches the models, the gate and
the index only through the ports in `domain.py`. `make imports` enforces both.

No test reaches the network or a model: a fake embedder places a question next
to a passage by construction, which is what makes the abstention path testable
at all.

## Not there yet

An eval gate in CI over a golden set built from XBRL `companyfacts` (the SEC
publishes the numbers, so the labels can be generated rather than hand-written),
token counts in the request log, a semantic cache, a drift monitor, and a
deployed instance. Those are the point of the project. This is the vertical slice they
attach to, and it is honest about which of its numbers are measurements and
which are anecdotes.

## Licence

MIT
