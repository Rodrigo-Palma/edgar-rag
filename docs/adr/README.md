# Architecture decision records

One page each, in the format of Michael Nygard's
[Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).
Copy [0000-template.md](0000-template.md) for a new one. Records marked
*retroactive* document a decision the code already made, written down after the
fact with the evidence that led to it.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-abstain-before-generating.md) | Abstain before generating; the relevance gate is a port | Accepted (retroactive) |
| [0003](0003-lowercase-embedding-input.md) | Lower-case embedding input to work around ollama#15609 | Accepted (retroactive) |
| [0013](0013-filing-text-is-untrusted.md) | Filing text is untrusted input | Accepted |

Numbers are reserved in the order the decisions were planned, so gaps are
records not yet written, not records that were deleted.
