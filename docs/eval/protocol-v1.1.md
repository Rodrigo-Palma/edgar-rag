# Measurement protocol v1.1: citation support (pre-registered)

- Written: 2026-10-09, before the v1.1 citation check existed and before any
  check, old or new, was replayed over the v1 generations with references
  stripped from the passages.
- What had been seen when this was written: the outcome counts the v1 run
  recorded (117 answered, 19 `no_valid_citation`, 3 `unsupported_claim`, 381
  `model_declined` among the 520 generated cases), the number of cases each
  arm answers (section 4), and the five probe answers of
  [issue #10](https://github.com/Rodrigo-Palma/edgar-rag/issues/10), which
  are made-up sentences, not v1 generations. Nothing else.
- This file is frozen once committed. A change after the first measurement
  commit is a deviation, written in the report, never edited in here.

## 1. Scope

Issues [#10](https://github.com/Rodrigo-Palma/edgar-rag/issues/10) and
[#11](https://github.com/Rodrigo-Palma/edgar-rag/issues/11) are correctness
defects of the citation check (`src/edgar_rag/citations.py`): reference
numbers back figures, a spelled-out figure is not read, and the abstention
detail repeats the figure it withheld.
[Issue #13](https://github.com/Rodrigo-Palma/edgar-rag/issues/13) asks how
often the first two happened in the v1 run.

This measurement is **exploratory and descriptive**. It changes no number of
[report-v1.md](report-v1.md), no headline, and neither H1 nor H2: those were
read on the outcomes the v1.0.0 code recorded, and a new headline needs a new
pre-registered run. The v1.1 fix ships whatever the counts below turn out to
be, because it is a correctness fix, not a tuning step: no threshold or rule
of it is chosen by looking at these numbers.

## 2. Pinned inputs

| input | pin |
|---|---|
| frozen run `eval/runs/v1/cases.jsonl` | sha256 `1c107cf223665d8e1aabdda0687a8202b0066210e663c5ee6c4f34d5e04a9989` |
| generation tape `eval/runs/v1/tape/generate.jsonl` (plain git) | sha256 `9059408046575d30dc0e3776d2e87f3915fb020b8151428a12c3f3c105c52d87` |
| tape meta `eval/runs/v1/tape/meta.json` | sha256 `5435d4a2146cddbc776179eab793f2bab16baa84fc080823200bf95525ed8c3d` |
| arms and thresholds | as in report-v1: cross-fitted R90, `src/edgar_rag/eval/arms.py` |

The script that measures refuses to run when the sha256 of `cases.jsonl` is
not the one in this table.

## 3. Passages, without a model or a network

The passages a check needs are the ones the model was shown. They are read
back from the prompt recorded in the tape (each passage is written there as
`[n] (item) text`), matched to its case by the nonce the evaluation derives
from the case id and by the generation text. Two checks make the
reconstruction trustworthy before anything is counted, and the script stops
if either fails:

1. The item of every reconstructed passage equals the item of the chunk id the
   case recorded in `retrieved`, in order.
2. Replaying the **v1.0.0 rules** over the reconstructed passages reproduces
   the outcome the run recorded (`answered`, `no_valid_citation`,
   `unsupported_claim`) for every one of the cases with text.

## 4. Population and size

The cases with text are the generated cases whose outcome is `answered`,
`no_valid_citation` or `unsupported_claim`. Per arm, a case belongs to the arm
when the arm's gate admits it (the masks of report-v1):

| arm | answered | with text | Wilson 95% upper bound at 0 of answered |
|---|---|---|---|
| A | 117 | 139 | 3.18% |
| B | 97 | 116 | 3.81% |
| C | 99 | 120 | 3.74% |
| D | 117 | 139 | 3.18% |
| E | 99 | 120 | 3.74% |
| F | 97 | 116 | 3.81% |

The bounds come from `edgar_rag.eval.metrics.wilson_interval(0, n)`, the same
function the report uses; the measurement script recomputes them and prints
them in the report. With n near 100, a count of zero still leaves a rate of
about 3 to 4% possible. That is the resolution of this measurement, and the
report states it next to every zero.

## 5. What is measured

All counts are k/n per arm with a Wilson 95% interval, labelled exploratory.

**M1, reference numbers as support.** Over the answered cases: how many the
v1.0.0 rules withhold as `unsupported_claim` once the item label is no longer
read and `Note N`, `page N`, `Item N`, `Exhibit N` and `Section N` (with
plurals) are removed from the cited passages. Every other v1.0.0 rule stays as
it was, so M1 isolates the reference numbers.

**M2, spelled-out figures.** Over the answered cases: how many state their
figures only in words (a number written in words followed by a scale word,
`percent` or `dollars`, and no amount in digits in any cited sentence). These
were never checked by v1.0.0.

**M3, the delta of the v1.1 check.** Over the cases with text: the outcome the
v1.1 check gives against the outcome the run recorded, as a transition table
(`answered` to withheld, withheld to `answered`, withheld for another reason).
Every case that changes outcome is listed by id.

## 6. Output and reproduction

`scripts/measure_citation_support.py` writes
`docs/eval/report-v1.1-citations.md`. `make eval` runs it after the v1 report,
so the CI job "report reproduces" fails when the committed report is not what
the frozen run and the code produce.

## 7. What this does not establish

- It does not measure how often the v1 answers were **wrong**: an answer the
  new check withholds may state a correct figure the passage prints in a form
  the check cannot match, and an answer it accepts may still paraphrase
  wrongly without digits.
- It says nothing about another model, prompt or corpus.
- The dev split replayed by `make eval-ci` is a regression gate, not part of
  this measurement; if the v1.1 check changes its baseline, the change is shown
  failing first and re-baselined in a commit of its own that lists the cases.
