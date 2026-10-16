# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The v1.1 counts (0/117 answers and 0/22 withheld under arm A, 139
  generations with text) pool the golden and the hand-written narrative
  questions, as protocol-v1.1 does, while the README said the narratives are
  never pooled and report-v1 gives 101 answered under arm A. The v1.1 report
  now opens with a population table by kind (117 = 101 golden + 16
  narrative), and the README says the narratives stay out of the v1 metrics
  only.
- The README's guarantee table said every number is copied from the
  evaluation reports. `scripts/check_readme_numbers.py` checks that each
  number appears in a report, not that it sits on the right row, arm or
  column: two numbers swapped between arms pass. The row, the `make readme`
  help and the checker's docstring now say what is checked.
- The README cited ollama/ollama#15609 for the collapse of capitalised tokens,
  but the issue reports only the non-ASCII case; the capitalised case was
  measured here. The README now says so and that the collapse no longer
  reproduces on Ollama 0.35.1. ADR-0003 revisits the lower-casing when the
  collapse stops reproducing, not when the issue closes, and records the
  0.35.1 measurement. The embedder still lower-cases by default.

## [1.1.2] - 2026-10-16

A patch to the figure-in-words parser of 1.1.1, and dependency bumps. No
measured case changes.

### Fixed

- A whole number that ends in a scale word now takes a fraction.
  `number_in_words` writes 250,000.0 as `two hundred fifty thousand point
  zero`, and 1.1.1 marked that unreadable: the rule meant to refuse a scale
  word on both sides of `point` (`one million point five billion`) also
  refused one on the whole part alone. It now refuses only the two together.
  The round-trip property test pins the counterexample, and fractions of
  zeros are a parametrized case.

The fix does not change a measured outcome. `make eval-ci` judges every case
of the dev split as in `eval/ci/baseline.json`, under every arm, and
`docs/eval` (the citation report included) is unchanged.

### Changed

- Runtime dependencies: fastapi 0.142.2 to 0.143.0, numpy 2.5.3 to 2.5.4,
  pydantic 2.13.5 to 2.14.1 (pydantic-core 2.46.5 to 2.50.1).
- Development dependencies: hypothesis 6.168.4 to 6.169.0, ruff 0.16.10 to
  0.17.0.
- Docker image: `astral-sh/uv` 0.12.23 to 0.13.1, and a refreshed
  `python:3.13-slim` digest. Dependabot no longer proposes a new Python minor
  for the image, which follows `UV_PYTHON` in CI; the two move together.
- CI: `astral-sh/setup-uv` 10.2.0 to 10.3.0.

## [1.1.1] - 2026-10-09

A patch to the citation check of 1.1.0, whose README promised that figures in
words are checked, and whose parser read some of them wrong.

### Fixed

- A figure in words is read whole. 1.1.0 read `one point five billion
  dollars` as five billion, so a passage printing $5 billion backed it; it did
  not read `two and a half billion` at all, and split `two thousand five
  hundred dollars` into two figures. Words now compose (`hundred`, scale words
  stepping down or multiplying, `point`, `a half`), and words that do not
  compose are a figure with no value, which backs nothing and nothing backs.
- The `no_network` test fixture also refuses `socket.connect_ex`.

### Added

- Two positive controls in `docs/eval/report-v1.1-citations.md`, not
  pre-registered. A reference number put in place of a cited figure is
  accepted by v1.0.0 in 88/88 and withheld by the M1 rules in 55/88, so the
  M1 zero bounds support that comes only from a reference number. Figures
  rewritten in words are counted by M2 in 85/85.
- A "Deviations from the protocol" section in the same report: the
  percentage rule and the 3-digit rescale floor came from issue #10, not from
  the protocol's scope, and each alone changes 0 of 139 outcomes.

### Changed

- The rescale rule is named for what it counts: 3 digits, trailing zeros
  included, not 3 significant digits. The README says so, and lists the
  round-count and `(in percent)` cases among the limitations.
- The report's arm tables collapse arms that admit the same cases.

The counts of M1, M2 and M3 are unchanged.

## [1.1.0] - 2026-10-09

v1.1 changes the citation check, not the gate. The headline numbers of
[report-v1.md](docs/eval/report-v1.md) were measured with v1.0.0 and stand.

### Fixed

- The citation check no longer counts the item label, or the passage's
  references to its own notes, pages, items, exhibits and sections, as support
  for a figure. A percentage is backed only by a percentage, and a number the
  passage prints without a scale word is rescaled only when it has at least 3
  significant digits ([#10](https://github.com/Rodrigo-Palma/edgar-rag/issues/10)).
- A figure in words that names a scale, `percent` or `dollars` now needs a
  marker and is held to the cited passage like one in digits
  ([#10](https://github.com/Rodrigo-Palma/edgar-rag/issues/10)).
- An `unsupported_claim` abstention names the sentence and its markers, never
  the figure it withheld ([#11](https://github.com/Rodrigo-Palma/edgar-rag/issues/11)).

### Added

- `docs/eval/protocol-v1.1.md`, committed before the measurement, and
  `docs/eval/report-v1.1-citations.md`, rebuilt by `make eval`: the v1.0.0 and
  v1.1 checks replayed over the 139 v1 generations with text, offline. No
  outcome changes in any arm (0/117 answered cases under arm A, exploratory,
  Wilson 95% upper bound 3.2%); a positive control that changes one digit per
  answer is withheld in 102/105
  ([#13](https://github.com/Rodrigo-Palma/edgar-rag/issues/13)).

### Changed

- The `no_network` test fixture also refuses httpx's async transport and
  `socket.connect` to an internet address.
- `astral-sh/uv` image 0.12.23, `hypothesis` 6.168.4.

## [1.0.0] - 2026-10-02

First public release.

### Added

- A FastAPI service that answers questions about one SEC 10-K at a time,
  cites the passages it used, and abstains with a closed set of reasons when
  the filing does not support an answer.
- Exact NumPy search over a sharded on-disk index, Ollama for embeddings and
  generation, a nonce-delimited prompt that treats filing text as untrusted,
  and a citation check on every answer.
- An evaluation harness that runs the production pipeline: a golden set built
  from XBRL, a pre-registered protocol, six gate arms as masks over one
  generation, and a CI gate that replays the dev split from a recorded tape.
- The pre-registered headline round (`qwen3:32b`, 20 companies): H1 not
  attainable, H2 not met; the brier relevance model was removed and the
  default gate is `none` ([ADR-0014](docs/adr/0014-remove-brier-default-to-no-gate.md)).

[Unreleased]: https://github.com/Rodrigo-Palma/edgar-rag/compare/v1.1.2...HEAD
[1.1.2]: https://github.com/Rodrigo-Palma/edgar-rag/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/Rodrigo-Palma/edgar-rag/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Rodrigo-Palma/edgar-rag/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Rodrigo-Palma/edgar-rag/releases/tag/v1.0.0
