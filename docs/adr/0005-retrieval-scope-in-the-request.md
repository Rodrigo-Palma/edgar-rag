# 0005. The retrieval scope is explicit in the request

- Status: Accepted (retroactive, decided with [ADR-0004](0004-index-format-shard-per-filing.md) on 2026-10-01)
- Date: 2026-10-02

## Context

With many filings in one index, a question has to be answered from the right
one. Searching every filing would let a passage of another company, or of
another year of the same company, win on vocabulary alone, which is exactly
the false answer the project exists to prevent. Inferring the company and year
from the question is entity linking, a separate problem with its own errors.

## Decision

`AskRequest` (`src/edgar_rag/service/schemas.py`) requires `cik` and takes an
optional `fiscal_year`; together they are a `Scope`. `CorpusIndex.search`
searches only the shard that scope resolves to (the latest indexed year when
`fiscal_year` is left out). A scope with no indexed filing abstains with
`out_of_scope` before anything is embedded or generated.

## Consequences

- A passage of another filing can never be retrieved, so "the right company,
  the wrong filing" is not an error this service can make.
- The caller has to know the CIK. That is a real cost for a person typing a
  question, and acceptable for the programmatic client this is.
- A question about another company asked against a filing is still possible,
  and is one of the evaluation's unanswerable kinds (`other_company`).
- The period guard judges a question against the filing the scope resolved to
  ([ADR-0001](0001-abstain-before-generating.md)).

## Enforced by

- [`test_a_search_never_returns_a_passage_of_another_filing`](../../tests/test_corpus_index.py)
- [`test_a_question_is_never_answered_from_another_company_s_filing`](../../tests/test_answerer.py)
- [`test_a_scope_without_a_fiscal_year_is_the_latest_one_indexed`](../../tests/test_corpus_index.py)
- [`test_a_question_about_a_company_not_indexed_abstains_as_out_of_scope`](../../tests/test_api.py) and [`test_a_scope_with_no_indexed_filing_abstains_before_anything_runs`](../../tests/test_answerer.py)
- [`test_every_request_the_readme_sends_is_one_the_service_accepts`](../../tests/test_readme_contract.py): the README's requests name a filing.
