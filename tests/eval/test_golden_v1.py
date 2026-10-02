"""The committed golden set v1: the counts the plan fixes, and a byte-for-byte rebuild."""

import gzip
import json
from pathlib import Path

from edgar_rag.eval.build import EvalPaths, build, check_leakage, check_narratives
from edgar_rag.eval.golden import NEGATIVE_KINDS, GoldenCase, NarrativeCase, load_cases
from edgar_rag.eval.templates import SECTOR_CONCEPTS

REPO_EVAL = Path(__file__).parents[2] / "eval"
SNAPSHOTS = REPO_EVAL / "snapshots" / "companyfacts"


def test_the_committed_golden_set_matches_the_plan():
    cases = load_cases(REPO_EVAL / "golden" / "v1.jsonl", GoldenCase)
    narratives = load_cases(REPO_EVAL / "golden" / "narrative-v1.jsonl", NarrativeCase)

    def count(predicate):
        return sum(1 for case in cases if predicate(case))

    eval_e2e = [c for c in cases if c.e2e and c.split == "eval"]
    assert sum(c.answerable for c in eval_e2e) == 180
    for kind in NEGATIVE_KINDS:
        assert sum(c.negative_kind == kind for c in eval_e2e) == 75
        assert count(lambda c, k=kind: c.e2e and c.split == "dev" and c.negative_kind == k) == 25
    assert count(lambda c: c.answerable) >= 2000
    assert count(lambda c: c.answerable) == count(lambda c: not c.answerable)
    assert len({c.ticker for c in cases if c.split == "eval"}) == 20
    assert len({c.ticker for c in cases if c.split == "dev"}) == 4
    assert "AAPL" in {c.ticker for c in cases if c.split == "dev"}
    check_leakage(cases)
    assert sum(n.answerable for n in narratives) == 20 and len(narratives) == 40


def test_the_committed_golden_set_rebuilds_byte_for_byte_from_the_snapshots():
    built = build(EvalPaths(REPO_EVAL))

    assert built.golden == (REPO_EVAL / "golden" / "v1.jsonl").read_bytes()
    assert built.stats == (REPO_EVAL / "golden" / "v1.stats.json").read_bytes()
    assert check_narratives(EvalPaths(REPO_EVAL), built.lock) == []


def test_every_sector_concept_is_a_tag_some_company_of_the_golden_set_reports():
    reported = set()
    for path in SNAPSHOTS.glob("*.json.gz"):
        reported |= set(json.loads(gzip.decompress(path.read_bytes()))["usGaapConcepts"])

    assert {concept.tag for concept in SECTOR_CONCEPTS} <= reported
