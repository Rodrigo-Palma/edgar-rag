"""Build the golden set from the pinned snapshots, without the network.

    python -m edgar_rag.eval.build fetch          # once, with the network
    python -m edgar_rag.eval.build build          # writes eval/golden/v1.*
    python -m edgar_rag.eval.build build --check  # rebuilds, compares bytes
    python -m edgar_rag.eval.build stats          # counts and drops

``build`` reads ``eval/companies.toml``, ``eval/filings.lock.json`` and
``eval/snapshots/``, checks every snapshot against its SHA-256 in the lock and
every CIK against the SEC's ticker map, generates the candidate pools
(``candidates``), samples the tiers (``selection``) and writes:

* ``eval/golden/v1.jsonl``: one ``GoldenCase`` per line, sorted by id;
* ``eval/golden/v1.stats.json``: counts by split, class, subtype and company,
  the facts dropped by subtype, concept and reason, and the input hashes.

It also validates the hand-written ``eval/golden/narrative-v1.jsonl`` against
the snapshots. The same inputs always give the same bytes; ``--check`` is how
CI holds that.
"""

import argparse
import collections
import json
import random
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import XMLParsedAsHTMLWarning

from edgar_rag.chunking import chunk_sections
from edgar_rag.edgar.parse import split_into_sections
from edgar_rag.edgar.xbrl import parse_companyfacts
from edgar_rag.eval import candidates
from edgar_rag.eval.candidates import CompanyData, Drops, FilingText
from edgar_rag.eval.golden import GoldenCase, NarrativeCase, dumps_cases, load_cases
from edgar_rag.eval.selection import (
    E2eQuota,
    by_company_and_kind,
    select_e2e,
    select_gate_negatives,
)
from edgar_rag.eval.snapshot import (
    FilingsLock,
    LockedFiling,
    Roster,
    SnapshotPaths,
    canonical_json,
    check_roster_ciks,
    load_roster,
    read_companyfacts_snapshot,
    read_lock,
    read_text_snapshot,
    read_tickers,
    reported_concepts,
    sha256,
)
from edgar_rag.eval.textmatch import PrintedNumbers, contains_phrase, normalize_words

E2E_QUOTAS = {
    "dev": E2eQuota(positives=25, negatives=25),
    "eval": E2eQuota(positives=9, negatives=15),
}
# Names in the relevance model's training data, and its question template: an
# eval-split question carrying either would measure memory, not relevance.
BRIER_NAMES = ("Apple", "Petrobras", "Vale", "Siemens", "Toyota")
BRIER_TEMPLATE = "does the passage answer this question"
GOLDEN_NAME = "v1.jsonl"
STATS_NAME = "v1.stats.json"
NARRATIVE_NAME = "narrative-v1.jsonl"


@dataclass(frozen=True, slots=True)
class EvalPaths:
    root: Path

    @property
    def roster(self) -> Path:
        return self.root / "companies.toml"

    @property
    def lock(self) -> Path:
        return self.root / "filings.lock.json"

    @property
    def snapshots(self) -> SnapshotPaths:
        return SnapshotPaths(self.root / "snapshots")

    @property
    def golden(self) -> Path:
        return self.root / "golden" / GOLDEN_NAME

    @property
    def stats(self) -> Path:
        return self.root / "golden" / STATS_NAME

    @property
    def narrative(self) -> Path:
        return self.root / "golden" / NARRATIVE_NAME


@dataclass(frozen=True, slots=True)
class Built:
    lock: FilingsLock
    cases: tuple[GoldenCase, ...]
    golden: bytes
    stats: bytes


def indexed_text(text: str) -> str:
    """The filing text as the index holds it: its sections, cut into passages."""
    return "\n\n".join(chunk.text for chunk in chunk_sections(split_into_sections(text)))


def folds(roster: Roster, lock: FilingsLock) -> dict[str, int]:
    """Cross-fitting fold of each eval-split company: half in each, by seed."""
    tickers = sorted(company.ticker for company in lock.companies if company.split == "eval")
    random.Random(roster.fold_seed).shuffle(tickers)
    half = len(tickers) // 2
    return {ticker: 0 if position < half else 1 for position, ticker in enumerate(tickers)}


def load_corpus(paths: EvalPaths) -> tuple[Roster, FilingsLock, tuple[CompanyData, ...]]:
    """Read and check every input of the build.

    Raises:
        ValueError: when a snapshot differs from the lock or a CIK from the SEC's map.
    """
    roster = load_roster(paths.roster)
    lock = read_lock(paths.lock)
    snapshots = paths.snapshots
    if sha256(snapshots.tickers.read_bytes()) != lock.company_tickers_sha256:
        raise ValueError("the company_tickers.json snapshot does not match the lock")
    check_roster_ciks(roster, read_tickers(snapshots))
    fold_of = folds(roster, lock)
    companies = []
    for company in lock.companies:
        trimmed = read_companyfacts_snapshot(snapshots, company)
        filings = tuple(
            _filing_text(filing, read_text_snapshot(snapshots, filing))
            for filing in lock.filings
            if filing.ticker == company.ticker
        )
        companies.append(
            CompanyData(
                company=company,
                facts=parse_companyfacts(trimmed),
                concepts=reported_concepts(trimmed),
                filings=filings,
                fold=fold_of.get(company.ticker),
            )
        )
    return roster, lock, tuple(companies)


def _filing_text(filing: LockedFiling, text: str) -> FilingText:
    return FilingText(
        filing=filing,
        indexed=PrintedNumbers.of(indexed_text(text)),
        full=PrintedNumbers.of(text),
        words=normalize_words(text),
    )


def candidate_pool(companies: Sequence[CompanyData], drops: Drops) -> list[GoldenCase]:
    pool: list[GoldenCase] = []
    for company in companies:
        peers = tuple(peer for peer in companies if peer.company.split == company.company.split)
        pool.extend(candidates.positives(company, drops))
        pool.extend(candidates.wrong_years(company, drops))
        pool.extend(candidates.other_companies(company, peers, drops))
        pool.extend(candidates.unreported_concepts(company, drops))
        pool.extend(candidates.off_domain(company, drops))
    return pool


def select(
    pool: Sequence[GoldenCase],
    lock: FilingsLock,
    seed: int,
    quotas: Mapping[str, E2eQuota] = E2E_QUOTAS,
) -> tuple[GoldenCase, ...]:
    """The golden set: every positive, balanced negatives, the e2e tier marked."""
    selected: list[GoldenCase] = []
    for split, quota in quotas.items():
        tickers = [company.ticker for company in lock.companies if company.split == split]
        in_split = [case for case in pool if case.split == split]
        pools = by_company_and_kind(in_split)
        e2e = select_e2e(pools, tickers, quota, seed)
        found_positives = [case for case in in_split if case.answerable]
        negatives = select_gate_negatives(pools, tickers, len(found_positives), e2e, seed)
        keep = {case.id for case in found_positives} | negatives
        selected.extend(
            case.model_copy(update={"e2e": case.id in e2e}) for case in in_split if case.id in keep
        )
    return tuple(sorted(selected, key=lambda case: (case.split, case.ticker, case.id)))


def check_leakage(cases: Sequence[GoldenCase]) -> None:
    """Refuse eval-split questions that name brier's companies or use its template.

    Raises:
        ValueError: listing the first offending ids.
    """
    names = [name.lower() for name in BRIER_NAMES]
    offending = [
        case.id
        for case in cases
        if BRIER_TEMPLATE in case.question.lower()
        or (
            case.split == "eval"
            and any(contains_phrase(normalize_words(case.question), n) for n in names)
        )
    ]
    if offending:
        raise ValueError(
            f"{len(offending)} questions leak brier's names or template: {offending[:5]}"
        )


def build(paths: EvalPaths, quotas: Mapping[str, E2eQuota] = E2E_QUOTAS) -> Built:
    """Everything ``build`` writes, in memory.

    Raises:
        ValueError: when an input is inconsistent or a pool cannot fill its quota.
    """
    roster, lock, companies = load_corpus(paths)
    drops = Drops()
    pool = candidate_pool(companies, drops)
    cases = select(pool, lock, roster.sample_seed, quotas)
    check_leakage(cases)
    golden = dumps_cases(cases).encode("utf-8")
    stats = summarize(cases, pool, drops, lock)
    stats["inputs"] = {
        "companies_toml_sha256": sha256(paths.roster.read_bytes()),
        "filings_lock_sha256": sha256(paths.lock.read_bytes()),
    }
    stats["golden_sha256"] = sha256(golden)
    return Built(lock=lock, cases=cases, golden=golden, stats=canonical_json(stats))


def _count(cases: Sequence[GoldenCase], key: Callable[[GoldenCase], str]) -> dict[str, int]:
    return dict(sorted(collections.Counter(key(case) for case in cases).items()))


def _cell(case: GoldenCase) -> str:
    return f"{case.split}/{'positive' if case.answerable else case.negative_kind}"


def summarize(
    cases: Sequence[GoldenCase], pool: Sequence[GoldenCase], drops: Drops, lock: FilingsLock
) -> dict[str, object]:
    e2e = [case for case in cases if case.e2e]
    positives = [case for case in cases if case.answerable]
    return {
        "gate_only": _count(cases, _cell),
        "e2e": _count(e2e, _cell),
        "e2e_by_company": _count(
            e2e, lambda case: f"{case.split}/{case.ticker}/{_cell(case).split('/')[1]}"
        ),
        "gate_only_by_company": _count(
            cases, lambda case: f"{case.ticker}/{_cell(case).split('/')[1]}"
        ),
        "folds": _count(
            [c for c in cases if c.fold is not None], lambda c: f"fold{c.fold}/{c.ticker}"
        ),
        "positive_facts": len({(c.accession, c.concept, c.period_year) for c in positives}),
        "positives_by_concept": _count(positives, lambda case: case.template.split("/")[0]),
        "pool": _count(pool, _cell),
        "drops": {
            f"{kind}/{concept}/{reason}": n
            for (kind, concept, reason), n in sorted(drops.counts.items())
        },
        "drops_by_reason": _count_pairs(drops),
        "companies": [{"ticker": c.ticker, "cik": c.cik, "split": c.split} for c in lock.companies],
        "excluded": [exclusion.model_dump(mode="json") for exclusion in lock.excluded],
    }


def _count_pairs(drops: Drops) -> dict[str, int]:
    totals: collections.Counter[str] = collections.Counter()
    for (kind, _, reason), count in drops.counts.items():
        totals[f"{kind}/{reason}"] += count
    return dict(sorted(totals.items()))


def check_narratives(paths: EvalPaths, lock: FilingsLock) -> list[str]:
    """Problems with the hand-written questions; empty when every one holds.

    An answerable question's terms must all occur in its expected item, and an
    unanswerable one's terms nowhere in its filing.
    """
    if not paths.narrative.is_file():
        return [f"{paths.narrative.name} is missing"]
    problems = []
    for case in load_cases(paths.narrative, NarrativeCase):
        try:
            filing = lock.filing(case.ticker, case.fiscal_year)
        except KeyError as error:
            problems.append(f"{case.id}: {error}")
            continue
        if (filing.accession, filing.cik) != (case.accession, case.cik):
            problems.append(f"{case.id}: accession or CIK differs from the lock")
            continue
        text = read_text_snapshot(paths.snapshots, filing)
        problems.extend(f"{case.id}: {problem}" for problem in _narrative_problems(case, text))
    return problems


def _narrative_problems(case: NarrativeCase, text: str) -> list[str]:
    if not case.answerable:
        words = normalize_words(text)
        return [
            f"{term!r} occurs in the filing"
            for term in case.absent_terms
            if contains_phrase(words, term)
        ]
    items = [section for section in split_into_sections(text) if section.item == case.expected_item]
    if not items:
        return [f"no section {case.expected_item}"]
    words = normalize_words("\n".join(section.text for section in items))
    return [
        f"{term!r} is not in {case.expected_item}"
        for term in case.evidence_terms
        if not contains_phrase(words, term)
    ]


def _write_or_check(paths: EvalPaths, built: Built, *, check: bool) -> int:
    outputs = {paths.golden: built.golden, paths.stats: built.stats}
    if not check:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(f"wrote {len(built.cases)} cases, sha256 {sha256(built.golden)}")
        return 0
    stale = [
        path.name
        for path, content in outputs.items()
        if not path.is_file() or path.read_bytes() != content
    ]
    if stale:
        print(f"the rebuild differs from the committed files: {', '.join(stale)}", file=sys.stderr)
        return 1
    print(f"{len(built.cases)} cases rebuilt byte for byte, sha256 {sha256(built.golden)}")
    return 0


def run_build(paths: EvalPaths, *, check: bool, quotas: Mapping[str, E2eQuota] = E2E_QUOTAS) -> int:
    try:
        built = build(paths, quotas)
    except (ValueError, KeyError, OSError) as error:
        print(f"build failed: {error}", file=sys.stderr)
        return 1
    status = _write_or_check(paths, built, check=check)
    problems = check_narratives(paths, built.lock)
    for problem in problems:
        print(f"narrative: {problem}", file=sys.stderr)
    return status or (1 if problems else 0)


def format_stats(stats: dict[str, Any]) -> str:
    """The stats file as aligned text, one section per table."""
    lines = []
    for section in (
        "gate_only",
        "e2e",
        "e2e_by_company",
        "folds",
        "positives_by_concept",
        "drops_by_reason",
        "drops",
    ):
        table = stats.get(section, {})
        lines.append(f"\n{section}")
        if isinstance(table, dict):
            lines.extend(f"  {key:60s} {value:>6}" for key, value in table.items())
    lines.append(f"\npositive facts: {stats.get('positive_facts')}")
    for exclusion in stats.get("excluded", []):
        lines.append(f"excluded: {exclusion}")
    lines.append(f"golden sha256: {stats.get('golden_sha256')}")
    return "\n".join(lines)


def run_stats(paths: EvalPaths) -> int:
    stats = json.loads(paths.stats.read_bytes())
    cases = load_cases(paths.golden, GoldenCase)
    if _count(cases, _cell) != stats["gate_only"]:
        print("the stats file does not describe the golden file", file=sys.stderr)
        return 1
    print(format_stats(stats))
    return 0


def run_fetch(paths: EvalPaths, cache: Path) -> int:  # pragma: no cover - needs the network
    import httpx

    from edgar_rag.config import IngestSettings
    from edgar_rag.edgar.client import EdgarClient
    from edgar_rag.edgar.fetch import REQUEST_TIMEOUT_SECONDS, Fetcher, SystemClock
    from edgar_rag.edgar.user_agent import validate_user_agent
    from edgar_rag.eval.acquire import acquire

    user_agent = validate_user_agent(IngestSettings().edgar_user_agent)
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    clock = SystemClock()
    with (
        httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False) as http,
        EdgarClient(user_agent, http=http, clock=clock, cache_dir=cache) as client,
    ):
        acquire(
            client,
            Fetcher(http, headers, clock),
            clock,
            load_roster(paths.roster),
            snapshots=paths.snapshots.root,
            lock_path=paths.lock,
            log=print,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m edgar_rag.eval.build", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--root", type=Path, default=Path("eval"), help="the eval directory")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="write the golden set from the snapshots")
    build_parser.add_argument("--check", action="store_true", help="compare instead of writing")
    commands.add_parser("stats", help="print counts and drops")
    fetch_parser = commands.add_parser("fetch", help="download and pin the inputs (network)")
    fetch_parser.add_argument(
        "--cache", type=Path, default=Path("data/cache"), help="download cache"
    )
    args = parser.parse_args(argv)
    paths = EvalPaths(args.root)
    with warnings.catch_warnings():
        # iXBRL documents are XHTML; parsing them as HTML is what the index does too
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        if args.command == "build":
            return run_build(paths, check=args.check)
        if args.command == "stats":
            return run_stats(paths)
        return run_fetch(paths, args.cache)


if __name__ == "__main__":
    raise SystemExit(main())
