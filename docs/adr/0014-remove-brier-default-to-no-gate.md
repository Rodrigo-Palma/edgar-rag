# 0014. Remove the brier gate and default to no gate

- Status: Accepted
- Date: 2026-10-02

## Context

The [protocol](../eval/protocol.md) fixed, before the headline run, what its
results decide (section 9). Brier stays as an optional plugin only if all three
hold: (1) H2 met, the upper bound of the 99% interval of FAR(E) - FAR(F) below
0; (2) the lower bound of the 95% interval of the gate-only ΔAUROC (C - B)
above 0; (3) E's recall cost not above F's by more than 2 p.p. The default gate
is `period+cosine` (F) if H1 is met and `none` (A) otherwise.

The run is `eval/runs/v1` (code `d8d23b9f9610`, eval split, 20 companies,
6192 cases), reported in [report-v1.md](../eval/report-v1.md). Copied from it:

| | value |
|---|---|
| arm A, false answers on unanswerable | 13/300 = 4.3% [2.5%, 7.3%] |
| arm E, false answers / recall cost | 5/300 = 1.7% [0.7%, 3.8%] / 3/180 = 1.7% [0.6%, 4.8%] |
| arm F, false answers / recall cost | 3/300 = 1.0% [0.3%, 2.9%] / 1/180 = 0.6% [0.1%, 3.1%] |
| H1, FAR(A) - FAR(F) | +3.3 p.p., 99.0% CI [+1.3, +5.7] p.p.; reading: not attainable |
| recall cost of F | +0.6 p.p., 99.0% CI [+0.0, +2.2] p.p. |
| H2, FAR(E) - FAR(F), paired | +0.7 p.p., 99.0% CI [+0.0, +2.7] p.p.; not met |
| gate only, AUROC cosine | 0.834 [0.820, 0.849], n 3076/3076, 20 companies |
| gate only, AUROC brier | 0.732 [0.713, 0.755], n 3076/3076, 20 companies |
| AUROC(brier) - AUROC(cosine) | -0.101, 95% CI [-0.124, -0.080]; MDE at 80% power about 0.031 |
| generator s / question | A 13.21, F 6.59 |

Conditions 1 and 2 fail (condition 2 with the whole interval below 0: brier
ranks worse than cosine), so brier is removed. FAR(A) is under the 5 p.p.
margin, so H1 is not attainable, F did not meet it, and the default is `none`.
The headline sentence and the confirmatory labels are in the
[addendum](../eval/report-v1-addendum.md).

## Decision

`BrierGate` leaves `src/edgar_rag/gate.py`, `brier` and `period+brier` leave
`EDGAR_RAG_GATE`, and `EDGAR_RAG_BRIER_URL`, `EDGAR_RAG_BRIER_MIN_CONFIDENCE`
and `EDGAR_RAG_BRIER_SHA` leave the settings; a setting naming a brier gate
stops the service at startup. `ServiceSettings.gate` defaults to `none`.

`period+cosine` stays as a latency option: on the run it spent 6.59 generator
seconds per question against 13.21 for no gate, at a recall cost of +0.6 p.p.
(99% CI [+0.0, +2.2]). Its threshold, `COSINE_R90` in
`src/edgar_rag/config.py`, is **0.7329**: the R90 of
`edgar_rag.eval.metrics.threshold_at_recall` over the cosine scores of the
3748 answerable golden questions of all 24 companies, fitted after the report.
The 3076 of the eval split come from `eval/runs/v1/cases.jsonl`; the 672 of the
dev split come from the CI replay of the committed index, whose 1344 dev cosine
scores equal those of the full 48-filing index (the pilot run) case for case.
To reproduce, with no model:

```sh
make cosine-threshold
# cosine R90 over 3748 answerable golden questions, 24 companies: 0.7329
```

The frozen brier scores stay published. `eval/runs/v1/cases.jsonl` keeps them,
`report-v1.md` prints arms C and E and the AUROC difference beside the others,
and `make eval` regenerates it unchanged. A replay of the run still scores
brier: `TapedBrierScores` in `src/edgar_rag/eval/replay.py` reads the replies
from the run's tape (`git lfs pull --include="eval/runs/v1/tape/**"
--exclude=""`) and never calls a model.

## Consequences

- The service needs nothing but Ollama and the index, and by default every
  question costs a generation. An operator who wants the saving sets
  `EDGAR_RAG_GATE=period+cosine`; `make demo` does, to show a decline before
  generation.
- 0.7329 is fitted in sample, on both splits, after the report. The report's
  F row was measured at the cross-fitted thresholds (fold 0: 0.7376, fold 1:
  0.7247), so it describes the method, not this exact number, and nothing
  measures 0.7329 out of sample. Refit it when the golden set, the index or the
  embedder changes.
- Raising the threshold from 0.55 to 0.7329 moved the CI baseline. In the dev
  gate-only tier, arms B and F each lose 23 answerable questions and gain 225
  correct declines (68 off_domain, 141 other_company, 16 unreported_concept),
  and B 23 more on wrong_year; end to end, each gains 1 other_company decline.
  `eval/ci/baseline.json` was judged again in the same commit.
- Recording can no longer score brier. A future run that wants it back restores
  the gate from the history (`d9112ac`) and goes through a new protocol.
- `GateError` is removed: no gate raises it, and it comes back with the gate
  that needs it (the history at `d9112ac` keeps its 502 handler). `degraded`
  stays, because the brier replay sets it and the v1 run records it.

## Enforced by

- [`test_the_default_gate_is_none`](../../tests/test_config.py) and [`test_the_default_gate_leaves_the_refusal_to_the_model`](../../tests/test_api.py): the default.
- [`test_the_default_cosine_threshold_is_the_r90_fitted_after_the_headline_run`](../../tests/test_config.py): the threshold the service reads.
- [`test_a_gate_that_is_not_one_of_the_three_is_refused_at_startup`](../../tests/test_config.py): a brier gate stops the service.
- [`test_brier_scores_on_a_frozen_tape_are_replayed`](../../tests/eval/test_eval_commands.py) and [`test_a_recording_never_scores_brier_even_on_a_tape_that_holds_it`](../../tests/eval/test_eval_commands.py): brier is replay only.
- The CI job "report reproduces": `make eval` leaves `docs/eval/` unchanged.
- Not enforced by a test: that 0.7329 is what `make cosine-threshold` prints,
  since the dev half needs the CI replay first.
