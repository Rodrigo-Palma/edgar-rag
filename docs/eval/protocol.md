# Evaluation protocol v1 (pre-registered)

- Written: 2026-10-01, before any question of the eval split was sent to a
  generator, a gate or brier. `edgar-rag eval run --split eval` refuses to start
  without `--protocol`, and the sha256 of this file goes in the run's manifest.
- What had been seen of the eval split when this was written: its questions,
  labels and counts (they are built from XBRL and the filing text, with no
  model in the loop), and nothing else. Every model output quoted below comes
  from the dev split.
- This file is frozen once committed. A change after the headline run starts is
  a deviation and is listed in the run's `deviations.md`, never edited in here.

## 1. Question

Does a cheap gate in front of the generator (the period guard plus a cosine
threshold, arm F) remove enough answers to unanswerable questions to be worth
its recall cost, compared with letting the model refuse on its own (arm A)?
And does the optional brier model, in place of cosine, do better than cosine
(arm E against arm F)?

The two answers decide the shipped defaults (section 9). Everything else in the
report is secondary or exploratory, and labelled so.

## 2. Pinned inputs

| input | pin |
|---|---|
| golden set `eval/golden/v1.jsonl` | sha256 `00419f11876cacad11f41ca51c9284b071d1bc562c139a1b3c7e42c23a3882bf` |
| narratives `eval/golden/narrative-v1.jsonl` | sha256 `8140c88aec9622818dfd5e4d0e4889ba56ccc4a8d8573d5f45d38b8a242a7455` |
| filings `eval/filings.lock.json` (48 10-Ks) | sha256 `8b03e4a452d5cc5fba19f4bb56392e78cf3d2661f1ec9c7ceca87b3193b29f8b` |
| roster `eval/companies.toml` | sha256 `507c033893888dafc7c3e05c7fc1aed02d3835a4cf7e7b73248a0853d476943e` |
| prompt `src/edgar_rag/prompt.py` | sha256 `501c43f1d33f9e242a9f0f733e1671d901d61835e17244e590286f5bf85313f9` |
| index of the 48 filings | digest `bbe4812361d6c95e` (the one the dev pilot ran on) |
| generator | `qwen3:32b`, digest `030ee887880fc378860c2dd35101da424377520441ae4bfe7be6deff8ade7840`; `temperature=0`, `seed=0`, `think=false`, `num_ctx=8192` |
| embedder | `nomic-embed-text`, digest `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`, input lower-cased |
| runtime | Ollama 0.18.0 on an Apple M3 Max (36 GB), macOS |
| retrieval | `top_k=4`, scope `(cik, fiscal_year)` of the target filing |
| brier code | `d70e7df9b0b4f765e2d533c7bc9dd532155916e5` |
| brier weights `models/brier.npz` | sha256 `594e88c8304959f35881e946adfac52a87fc56b049b722bf6c47de676ad9586c` (not under version control; see section 12) |
| bootstrap | B = 5,000 company resamples, stratified by fold, percentile, seed 20261001 |

## 3. Units, splits, folds and arms

- Unit: one golden question, asked against the filing its `Scope` names.
- Splits by company. The dev split (AAPL, KO, CAT, PFE) never enters a
  headline number. The eval split has 20 companies in two cross-fitting folds,
  drawn with seed 20261001:
  - fold 0: AMZN, COST, CVX, DE, GOOGL, JPM, NVDA, UNP, V, WMT
  - fold 1: BA, BAC, JNJ, LLY, MCD, MSFT, NEE, PEP, PG, UNH
- Realised counts on the eval split: gate-only tier 3,076 answerable and 3,076
  unanswerable (769 per negative subtype); end-to-end tier 180 answerable and
  300 unanswerable, exactly 9 and 15 per company, 75 per subtype
  (`wrong_year`, `other_company`, `unreported_concept`, `off_domain`); 40
  hand-written narratives (20 answerable, 20 not) over 18 of the companies.
- Arms ([ADR-0015](../adr/0015-one-generation-serves-every-arm.md): one
  generation per question serves all of them): **A** no gate, the model's own
  `REFUSE-{nonce}` only; **B** cosine; **C** brier; **D** period guard; **E**
  period guard and brier; **F** period guard and cosine.
- Thresholds of the learned gates (cosine, brier): the score that admits 90%
  of the answerable questions of the gate-only tier (R90), fitted on the
  opposite fold. D has no threshold.

## 4. Metrics

- **Primary: FAR**, the share of the 300 unanswerable end-to-end questions an
  arm answers (does not abstain on).
- **Co-primary, safety: recall cost**, the share of the 180 answerable
  end-to-end questions that arm A answers correctly and the gated arm declines.
  "Correctly" means the number is within tolerance (relative 0.5%, or the
  rounding of the digits printed) and the cited passage contains the gold value.
- Secondary: numeric accuracy among answered; citation support among answered;
  FAR per subtype (n = 75, descriptive); gate-only AUROC and FAR at R90
  (3,076/3,076); risk against coverage; generator seconds per question by arm;
  determinism (30 repeated generations); latency P50/P95 by stage.
- Narratives are reported apart and never pooled: correct decision on all 40;
  expected `Item` cited only on answered answerable narratives whose filing
  passes the split sanity rule of section 11 (D-02).

## 5. Intervals, and why 99% nominal

The two confirmatory hypotheses share a familywise 5% by Bonferroni, so each
interval is meant to cover 97.5%. With 10 companies per fold, the percentile
company bootstrap undercovers. Measured with `edgar-rag eval power` (600
replicates, company effect sd 0.5 logit, the report's own estimator):

| design | 97.5% nominal | 99% nominal |
|---|---|---|
| H1, nested difference, 5 to 12.5 p.p. | coverage 0.937 to 0.960 | coverage 0.957 to 0.972 |
| H2, paired, null | one-sided size 2.2% (target 1.25%) | one-sided size 1.0% |
| H2, paired, -2 to -8 p.p. | coverage 0.942 to 0.972 | coverage 0.968 to 0.985 |

**Every confirmatory interval is the 99% nominal company-bootstrap percentile
interval**, which is the report's default (`DEFAULT_CONFIRMATORY_CONFIDENCE`).
Its coverage, 0.957 to 0.985 across both designs, is close to the 97.5% wanted,
and the one-sided size that matters for H2 drops to 1.0%. Rates elsewhere
carry Wilson 95% intervals; the gate-only AUROC difference carries a 95%
bootstrap interval, as the brier rule in section 9 specifies.

## 6. Confirmatory hypotheses and decision rules

### H1: the cheap gate pays for itself

**Met if and only if both hold:**

1. the lower bound of the 99% interval of FAR(A) - FAR(F) on the 300
   unanswerable questions is **above 5 p.p.**; and
2. the upper bound of the 99% interval of F's recall cost on the 180 answerable
   questions is **at most 12 p.p.**

The report states one of four readings of the first condition, decided
mechanically by `h1_reading` in `src/edgar_rag/eval/report.py`:

| reading | condition |
|---|---|
| above the margin | lower bound > 5 p.p. |
| not attainable | FAR(A) point estimate < 5 p.p. (section 7) |
| below the margin | upper bound < 5 p.p. |
| inconclusive at this n | the interval contains 5 p.p. |

Why the margin goes on the lower bound. An earlier draft read "a difference of
at least 5 p.p. with a lower bound above 0". Because F can only remove answers
A gave, the difference is non-negative per question, so "lower bound above 0"
holds for almost any gate that removes anything, and the rule reduces to a
point estimate of at least 5 p.p. Simulated through the report's estimator
(600 replicates, 99%):

| true difference | point >= 5 and lower > 0 | lower > 5 |
|---|---|---|
| 3.0 p.p. | 0.07 | 0.00 |
| 4.0 p.p. | 0.25 | 0.00 |
| 5.0 p.p. | 0.56 | 0.01 |
| 7.5 p.p. | 0.94 | 0.15 |
| 10.0 p.p. | 1.00 | 0.58 |

The first rule passes a gate that removes 4 p.p. a quarter of the time and
sits at a coin flip on the margin itself: it is not a test. The margin rule has
a size of 1% at a true 5 p.p. and needs a true effect of about 11.5 p.p. for
80% power (0.81 there, section 8). The cost is accepted and stated in advance: a gate that truly
removes between 5 and about 11 p.p. will usually read "inconclusive at this
n", and the report must print it with those words, not as a success.

Measured property of condition 2, declared rather than corrected: with 9
answerable questions per company, the 99% upper bound of the recall cost
passes the 12 p.p. ceiling 2.7% of the time when the true cost is exactly 12
p.p. (1,000 replicates; nominal one-sided 0.5%), so the ceiling is somewhat
lenient. Its power: 0.95 when the true cost is 4 p.p., 0.68 at 6 p.p., 0.33 at
8 p.p. A cosine gate at R90 blocks about 10% of answerable questions, so a
recall cost near 6 to 9 p.p. is expected when arm A is right on most of them,
and condition 2 can fail with the gate working exactly as designed. That is a
property of the 12 p.p. ceiling, fixed before this run, and it is reported as
such.

### H2: brier beats cosine behind the guard

**Met if and only if** the upper bound of the 99% interval of FAR(E) - FAR(F),
paired on the same 300 unanswerable questions, is **below 0**. Discordant
counts both ways and the exact McNemar p-value are printed beside it; they do
not decide.

Power assumption, part of the hypothesis: per question, E answers where F
declines (E worse) at 1 p.p. (`--worse-rate 0.01`), and E declines where F
answers (E better) at 1 p.p. plus the net improvement.

### Descriptive, by construction

D on `wrong_year`: the guard and the label share one definition of the period
a filing covers, so its FAR there is low by construction. Its recall cost is
the number of interest.

Everything not named in this section is exploratory and is labelled
exploratory in the report.

## 7. If the model already refuses: pre-committed reporting

On the dev split the 8B model answered 1 of 100 unanswerable questions with no
gate, and the 32B pilot answered 0 of 100 (Wilson 95% [0.0%, 3.7%], section
10). If the eval split looks the same, H1 cannot be met by any gate, since
FAR(A) - FAR(F) is at most FAR(A). This outcome is decided now, so it cannot be dressed up later:

- "Near zero" means FAR(A) point estimate below 5 p.p., that is 14 or fewer of
  the 300. The report's H1 reading is then "not attainable", printed by the
  code.
- The headline sentence is fixed: *"With no gate, the model answered k of 300
  unanswerable questions (FAR x%, Wilson 95% [l, u]). No pre-generation gate
  can remove more false answers than the model gives, so H1 was not
  attainable on this set. What the gate changes is cost: arm F spends s_F
  generator seconds per question against s_A for arm A, at a recall cost of r
  [99% interval]."* The numbers come from the report's arm table.
- Defaults follow section 9 without exception: F did not meet H1, so the
  default gate becomes `none` and `period+cosine` is documented as a latency
  option.
- Not allowed in that case: switching to a relative reduction ("the gate
  removes 100% of the remaining false answers"); promoting a subtype, a fold
  or the gate-only tier to a confirmatory claim; testing H1 against zero;
  re-running with another model, prompt or threshold target to obtain a
  larger FAR(A). Any of these may appear only as exploratory, under that label.
- If FAR(A) is 5 p.p. or more, the four readings of section 6 apply as written.

## 8. Minimum detectable effects with the realised counts

`uv run edgar-rag eval power --n-companies 20 --neg-per-company 15
--per-subtype 75 --positives 180 --gate-per-class 3076 --worse-rate 0.01
--calibrated-confidence 0.99` (seed 20261001, 600 replicates x 5,000
resamples):

| comparison | design | result |
|---|---|---|
| FAR of one arm, n = 300 | Wilson 95% | ±3.4 p.p. at 10%, ±4.5 p.p. at 20% |
| FAR per subtype, n = 75 | Wilson 95% | ±7.1 p.p. at 10%, ±8.9 p.p. at 20% (descriptive only) |
| recall cost or accuracy, n = 180 | Wilson 95% | ±5.2 p.p. at 85%, ±6.6 p.p. at 70% |
| H1, FAR(A) - FAR(F), lower bound > 5 p.p. | nested, 99% | MDE 11.5 p.p. at 80% power; half-width ±3.2 p.p. at 5 p.p., ±5.1 at 11.5 |
| H2, FAR(E) - FAR(F), upper bound < 0 | paired, 99% | MDE 5.0 p.p. (power 0.82; 0.69 at 4 p.p.); size 1.0% |
| ΔAUROC (C - B), gate-only 3,076/3,076 at AUROC 0.8 | 95%, r = 0.5 | MDE 0.016 i.i.d.; 0.030 with a company shift of sd 0.5 |

Printed in the report next to the comparisons: a difference in FAR between
arms below about 5 p.p. is "not distinguishable at this n", in those words.

## 9. Decisions the results drive (not reopened after the run)

Brier stays as an **optional plugin** if and only if all three hold:

1. H2 met (upper bound of the 99% interval of FAR(E) - FAR(F) below 0);
2. ΔAUROC (C - B) on the eval split gate-only tier with the lower bound of its
   95% company-bootstrap interval above 0;
3. E's recall cost not above F's by more than 2 p.p. (point estimates).

Otherwise `BrierGate` is removed from the package (PT-18) and its frozen scores
stay in the report as a negative result, with the same prominence. The
interval in condition 2 is the plan's 95%; its coverage with 10 companies per
fold was not measured, and it is probably lenient like the others. Since all
three conditions are required, a lenient condition 2 can only keep brier when
H2 also holds.

The shipped default gate is `period+cosine` (F) if H1 is met, and `none` (A)
otherwise, with the README saying that the pre-generation gate did not pay for
itself and is offered as a latency saving.

**If brier is not running when the headline run starts**, arms C and E have no
scores and are printed as not run, H2 is reported as **not run** (not as not
met), and the brier rule above cannot be satisfied, so brier is removed. Brier
scores may not be added to the frozen run afterwards: the decision to add them
would be made after seeing A and F. The pre-flight in section 11 exists to keep
this from happening for a technical reason.

## 10. Pilot on the dev split (qwen3:32b)

Run on 2026-10-01 and 02, dev split only: all 1,344 dev golden cases scored by
the period guard and cosine, the 200 end-to-end cases (100 answerable, 100
unanswerable, 4 companies) generated with the pinned generator, and 10
determinism repeats. Code `cd482b9`; the manifest says dirty only because a
log file sat untracked in the checkout. Index `bbe4812361d6c95e`, brier not
plugged in. The run directory is kept, not committed, at
`data/eval/pilot-32b-dev-e2e` in the main checkout (`cases.jsonl` sha256
`4ed35d14cc817452f95f70e459918f0b00db72df852eccee03f3273ee28b55f9`).

| measure | qwen3:32b pilot | qwen3:8b smoke, same split |
|---|---|---|
| arm A FAR | 0/100 = 0.0%, Wilson 95% [0.0%, 3.7%] | 1/100 = 1.0% [0.2%, 5.4%] |
| arm A answered correctly, of 100 answerable | 36/100 [27.3%, 45.8%] | 41/100 [31.9%, 50.8%] |
| arm A outcome on the 100 answerable | 54 answered, 42 declined by the model, 4 withheld by the citation check | 56 answered |
| F recall cost (in-sample R90) | 3/100 [1.0%, 8.5%] | 4/100 [1.6%, 9.8%] |
| generation P50 / P95 | 14.0 s / 18.1 s (n = 174, see below); 14.2 s / 18.5 s on all 200 | 3.4 s / 4.4 s |
| mean generator seconds, answered / declined | 16.5 s / 13.4 s | 3.8 s / 3.2 s |
| prompt tokens, answered / declined | 1,787 / 1,539 | 1,782 / 1,536 |
| same prompt twice, identical text | 10/10 | 10/10 |

Twenty-six of the 200 generations (the 41st to the 66th) ran while a brier
gate-only check shared the GPU; their median was 16.4 s. The first row of
latency excludes them.

Fallback rule of the plan: the P50 of 14.0 s is under 23 s, so the
end-to-end tier keeps 180 answerable and 300 unanswerable questions.

What the pilot says about H1: four dev companies are not twenty eval
companies, but the 32B answered none of the 100 unanswerable dev questions.
Section 7 is the expected branch, not a remote one, and it was written
before the pilot finished. The H1 rule, the 99% level and section 7 come from
the simulations of sections 5 and 6 and from the 8B smoke known beforehand;
nothing in this protocol was changed after the pilot's numbers were seen.

Brier pre-flight, also on the dev split only: with brier served as in section
11, a gate-only run scored all 1,344 dev cases with brier (no failure, no
fallback), the manifest recorded brier ready at `d70e7df`, and the three gates
together took 0.16 s median per question. The dev numbers it printed (brier
AUROC 0.679 against cosine 0.863, in-sample, 4 companies) decide nothing; the
brier rule of section 9 is read on the eval split only.

Budget for the headline run, from these measurements:

| step | n | unit | total |
|---|---|---|---|
| generation (480 end-to-end, 40 narratives, 30 repeats) | 550 | 14.2 s mean | 2 h 10 min |
| embedding, search and the three gates, every eval case | 6,192 | 0.17 s | 18 min |
| ingest, only if the index has to be rebuilt | 48 filings | | at most 15 min |
| **total** | | | **about 2 h 45 min, under the 4 h ceiling** |

The plan budgeted 3 h 30 min (20 s per generation, 0.1 s per gate-only
question without brier).

## 11. Procedure, stopping and rerun rules

Pre-flight, before any eval-split call (none of these reads an eval outcome):

1. `git status --short` is empty, and the pins of section 2 hold:
   `shasum -a 256` of the four `eval/` files and `src/edgar_rag/prompt.py`;
   `ollama show qwen3:32b` digest; `ollama --version` is 0.18.0.
2. Frozen code: `git diff --stat <commit adding this file> HEAD -- src/edgar_rag/{answer,prompt,citations,amounts,gate,period,models,chunking,domain,index}.py src/edgar_rag/eval/{arms,grading,numeric,textmatch,runner,summary,metrics,bootstrap,report,golden}.py eval/golden eval/filings.lock.json eval/companies.toml`
   prints nothing. The one change allowed in advance is the Git LFS pointer
   check of PT-15 in `src/edgar_rag/index.py`, which can only raise on a
   pointer file. Any other difference is written in `deviations.md` with its
   reason before the eval split is touched.
3. The index loads with digest `bbe4812361d6c95e` and 48 filings. A different
   digest is investigated and written in `deviations.md` before the run.
4. CI on the run's commit is green, including the PT-15 replay of the dev
   cassettes, which misses on any change to the prompt.
5. The pilot's tape replays: `edgar-rag eval run --split dev --mode replay
   --tape <main checkout>/data/eval/pilot-32b-dev-e2e/tape --index-dir <the
   run's index> --out data/eval/pilot-replay` finishes with no tape miss and the same outcome
   for each of the 200 end-to-end cases. A miss means the prompt, the
   retrieved passages or the embedder input changed since this protocol.
6. Logs go under `data/` or outside the checkout. The manifest marks a run
   dirty for any untracked file, and a dirty run is not valid.
7. Brier is served from `~/dev/pessoal/brier` at `d70e7df`, with
   `models/brier.npz` at the sha256 of section 2, and `GET /ready` answers.
   The repo's `.venv` points at a pre-move path and no longer runs; on
   2026-10-01 this ran it without touching the repo:
   `uv run --no-project --python 3.13 --with-editable . --with uvicorn uvicorn brier.api:app --host 127.0.0.1 --port 8100`.

The run, one command, from a clean checkout:

```sh
EDGAR_RAG_BRIER_URL=http://127.0.0.1:8100 \
EDGAR_RAG_BRIER_SHA=d70e7df9b0b4f765e2d533c7bc9dd532155916e5 \
uv run --frozen edgar-rag eval run --split eval --narratives --repeat 30 \
  --mode record --generation-model qwen3:32b \
  --protocol docs/eval/protocol.md --out eval/runs/v1
uv run --frozen edgar-rag eval report --run eval/runs/v1 --out docs/eval/report-v1.md
```

A run is valid when its manifest shows: `repo_dirty` false; `protocol_sha256`
equal to this file's; the golden and narrative hashes of section 2; the
generator digest, options and Ollama version of section 2; `top_k` 4; gates
`period`, `cosine`, `brier` with brier ready; 6,192 cases (6,152 golden and 40
narratives); 30 repeats.

Stopping and reruns:

- The first valid, complete run is the headline. It is not rerun because of
  what it shows.
- A technical failure (crash, out of memory, Ollama restart, tape miss, a
  manifest that fails the checks above) discards the attempt. The cause goes in
  `deviations.md`, and the same command runs again from scratch into a fresh
  run directory and tape. No code changes between attempts except the fix of
  that failure, itself listed as a deviation.
- After a third failed attempt the round stops, and the failure is the
  published result for v1.
- The run prints outcomes as it goes. Nobody acts on them: the rules above
  leave nothing to decide while it runs.
- The report is generated once, with its defaults (99% confirmatory, B =
  5,000, seed 20261001). Other levels may appear only as exploratory.

Split sanity for the `Item` metric (D-02), applied mechanically by
`split_is_sane` in `src/edgar_rag/eval/grading.py`: the filing has at least 2
items besides the full-filing fallback, no item holds more than 50% of the
text, and the expected item is among them. The report prints the n included
and the filings excluded with the reason.

## 12. Leakage check against brier

brier's training rows are produced only by `brier.tasks.generate` at
`d70e7df`, which composes literal lists (8 subject domains, 5 companies, 5
years, 12 hedges, 6 neutral templates); training reads no file and no filing.
So its whole question space can be enumerated, and was: 200 instances of the
`answerable` template (5 companies, 8 subjects, 5 years), 8 of `relevance`,
and the fixed `tone` and `subject` prompts.

| check | result |
|---|---|
| eval-split questions (6,152 golden and 40 narratives) naming a brier company (Apple, Petrobras, Vale, Siemens, Toyota), word boundary | 0 |
| golden questions, any split, equal to a brier question after normalising case, punctuation and spacing | 0 of 7,536 |
| golden templates (92, with company and year as slots) equal to a brier template after masking company, year and subject | 0 |
| golden questions containing brier's wrapper "does the passage answer this question" | 0 |
| eval gold values within 0.5% of a number in brier's passages | 32 questions (MSFT net income 88,136 against "88.2 billion dollars", and similar); coincidences of magnitude, since brier scores relevance and never states a value |

Closest template pair: golden "What was {company}'s {label} in fiscal {year}?"
and brier's inner question "what was {company}'s {subject} in {year}?". They
differ by the word "fiscal" and by the slot vocabulary (brier's subjects are
quarterly revenue, headcount, litigation, rainfall, attendance, infections,
emissions, downtime; none is an XBRL concept of the golden set). Brier receives
every golden question wrapped in its own "Does the passage answer this
question: ..." prompt by design of the plugin; that is its interface, not
eval data reaching its training.

What could not be verified:

- The weights file is not under version control and stores no commit. Its
  modification time (2026-09-26 22:04:59 -03) is 45 seconds before commit
  `7b08d07`, and `d70e7df` only rewrote the README, so the weights most likely
  come from the code at `7b08d07`, equal to `d70e7df` outside the README. This
  is inferred from timestamps, not proven; the sha256 in section 2 pins the
  file that will run.
- brier has no lock file, so its dependency versions at run time are whatever
  resolves then.
- brier and the cosine gate share the frozen `nomic-embed-text` encoder, whose
  pre-training data is unknown and may include SEC filings. That affects both
  gates alike and does not bias H2 towards either.

## 13. Deviations from the plan, decided before the run

- **D-01, roster.** The exclusion rule became: a company is replaced when
  either of its two 10-Ks (fiscal 2024 and 2025) is missing under the CIK that
  `company_tickers.json` lists, or a report date falls in another calendar year
  than its fiscal year. XOM is out (the ticker now maps to ExxonMobil Holdings
  Corp, CIK 2115436, a new registrant with no 10-K) and HD is out (fiscal 2024
  report date 2025-02-02); LLY and PEP came in from the reserve list, in order.
  Applied before any generation, blind to results. Energy is left with CVX
  alone; no hypothesis is by sector.
- **D-02, item split.** `split_into_sections` mislabels some filings. The
  `Item` metric is reported only where the split passes the rule of section
  11. The plan expected this to exclude JPM fiscal 2025 and MCD only. Applied
  to the pinned index before the run, the same rule fails 15 of the 48
  filings (both JPM, MCD, PEP, CVX, UNP, JNJ and DE filings, and PFE fiscal
  2024) and excludes 4 of the 20 answerable narratives: JNJ, CVX, DE and UNP,
  all fiscal 2025. The `Item` metric therefore has at most 16 narratives. The
  rule is kept as written; it was fixed before this count was taken, and
  loosening it now would be choosing the denominator. No headline metric
  depends on the item label.
- **Confidence level.** The plan specified 97.5% intervals; this protocol uses
  99% nominal for the reason in section 5.
- **H1 rule.** The plan's wording was ambiguous between "lower bound above 0"
  and "lower bound above 5 p.p."; section 6 fixes the second, with the reason.
- **Fallback for the budget.** Not triggered: the pilot P50 is below 23 s
  (section 10), so the end-to-end tier keeps 180 and 300.

## 14. Known limitations, stated in advance

- Questions are generated from XBRL templates; the 40 narratives are the only
  hand-written check and are reported apart.
- The generator may have seen these filings in pre-training. Fiscal 2024 and
  2025 filings, and the requirement that the cited passage print the gold
  value, limit that risk and do not remove it.
- The period guard and the `wrong_year` label share one definition of the
  period a filing covers.
- Twenty companies are few clusters: Wilson intervals assume independent
  questions, and the percentile bootstrap needed the 99% recalibration above.
- Results hold for 10-K filings of large US companies, on this hardware and
  runtime, and nothing broader.
