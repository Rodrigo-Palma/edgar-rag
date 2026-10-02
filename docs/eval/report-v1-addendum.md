# Addendum to report v1

Written by hand after the headline run, in the commit that freezes it. It adds
two things sections 6 and 7 of the [protocol](protocol.md) require of the
report and that `src/edgar_rag/eval/report.py`, frozen since the protocol, does
not print. It changes no number of [report-v1.md](report-v1.md); every figure
below is copied from it. Recorded as DV-02 in
[deviations-v1.md](deviations-v1.md).

## Headline sentence (protocol, section 7)

FAR(A) is 13 of 300, under the 5 p.p. margin, so the report reads H1 as "not
attainable" and section 7 fixes the headline sentence. Its numbers come from
the report's arm table, and the 99% interval of the recall cost from its
confirmatory comparisons:

> With no gate, the model answered 13 of 300 unanswerable questions (FAR 4.3%,
> Wilson 95% [2.5%, 7.3%]). No pre-generation gate can remove more false
> answers than the model gives, so H1 was not attainable on this set. What the
> gate changes is cost: arm F spends 6.59 generator seconds per question
> against 13.21 for arm A, at a recall cost of 0.6% [99% interval +0.0 to
> +2.2 p.p.].

Defaults follow section 9: H1 was not met, so the default gate is `none` (arm
A), and `period+cosine` is documented as a latency option.

## Which parts of the report are confirmatory

Section 6 names H1 (FAR(A) - FAR(F) and F's recall cost), H2 (FAR(E) - FAR(F),
paired) and, as descriptive by construction, arm D on `wrong_year`. Section 9
adds the gate-only AUROC difference as condition 2 of the brier rule.
Everything else is exploratory, and the report does not say so; read it with
these labels:

| report section | status |
|---|---|
| Thresholds | method, not a result |
| Six arms: false answers and recall cost of arms A, E and F | inputs of H1 and H2 (confirmatory) |
| Six arms: the rows of arms B, C and D, and for every arm the columns answered correct, numeric accuracy, citation support, gate recall and generator s / question | exploratory |
| Confirmatory comparisons (H1, H2) | confirmatory |
| False answers by subtype | descriptive (section 4); arm D on `wrong_year` descriptive by construction |
| Gate only: AUROC(brier) - AUROC(cosine) | condition 2 of the brier rule (section 9), not a hypothesis |
| Gate only: per-gate AUROC, FAR at R90, realised recall | exploratory |
| Risk and coverage | exploratory |
| Narrative questions | exploratory |
| Determinism | exploratory |
| Operating cost | exploratory |
