# Architecture decision records

One page each, in the format of Michael Nygard's
[Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).
Copy [0000-template.md](0000-template.md) for a new one. Records marked
*retroactive* document a decision the code already made, written down after the
fact with the evidence that led to it.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-abstain-before-generating.md) | Abstain before generating; the relevance gate is a port | Accepted (retroactive) |
| [0002](0002-exact-numpy-search-no-vector-database.md) | Exact NumPy search on disk, no vector database | Accepted (retroactive) |
| [0003](0003-lowercase-embedding-input.md) | Lower-case embedding input to work around ollama#15609 | Accepted (retroactive) |
| [0004](0004-index-format-shard-per-filing.md) | Index format 2: one shard per filing under a manifest that pins the embedder | Accepted (retroactive) |
| [0005](0005-retrieval-scope-in-the-request.md) | The retrieval scope is explicit in the request | Accepted (retroactive) |
| [0006](0006-closed-abstention-reasons.md) | Abstention reasons are a closed set, and the answer carries the gate score | Accepted (retroactive) |
| [0007](0007-golden-set-from-xbrl.md) | Generate the golden set from XBRL companyfacts and the filing text | Accepted (retroactive) |
| [0008](0008-eval-runs-the-production-pipeline.md) | The evaluation runs the production pipeline | Accepted (retroactive) |
| [0009](0009-ci-eval-gate-replayed.md) | CI eval gate: a replayed dev split against a committed baseline | Accepted (retroactive) |
| [0010](0010-injected-http-clients.md) | Reach models through injected HTTP clients, without a framework | Accepted (retroactive) |
| [0011](0011-composition-root-in-an-app-factory.md) | Compose the service once in an app factory; no DI framework | Accepted (retroactive) |
| [0012](0012-bounded-generations-one-log-line.md) | Bound concurrent generations and log one JSON line per request | Accepted (retroactive) |
| [0013](0013-filing-text-is-untrusted.md) | Filing text is untrusted input | Accepted |
| [0014](0014-remove-brier-default-to-no-gate.md) | Remove the brier gate and default to no gate | Accepted |
| [0015](0015-one-generation-serves-every-arm.md) | One generation per question serves every evaluation arm | Accepted |
| [0016](0016-local-only-no-hosted-instance.md) | Local only, no hosted instance | Accepted |

Numbers follow the order the decisions were planned in, not the order they
were written.
