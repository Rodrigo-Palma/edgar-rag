"""The build end to end: offline, byte for byte, and refusing what it cannot vouch for."""

import json

import pytest

from edgar_rag.eval.build import (
    BRIER_NAMES,
    EvalPaths,
    build,
    check_leakage,
    check_narratives,
    main,
    run_build,
)
from edgar_rag.eval.golden import NEGATIVE_KINDS, GoldenCase, NarrativeCase, dumps_cases, load_cases
from edgar_rag.eval.selection import E2eQuota
from edgar_rag.eval.snapshot import read_lock
from tests.eval.golden_fakes import DEV, EVAL, FakeCompany, write_eval_root

SMALL = {"dev": E2eQuota(positives=2, negatives=4), "eval": E2eQuota(positives=2, negatives=4)}


def _narratives(root, lock):
    filing = lock.filing("CCC", 2025)
    ref = {"ticker": "CCC", "cik": filing.cik, "fiscal_year": 2025, "accession": filing.accession}
    cases = [
        NarrativeCase(
            id="n1",
            question="How are results discussed?",
            answerable=True,
            expected_item="Item 7",
            evidence_terms=("results of operations",),
            **ref,
        ),
        NarrativeCase(
            id="n2", question="Write a poem?", answerable=False, absent_terms=("poem",), **ref
        ),
    ]
    (root / "golden").mkdir(exist_ok=True)
    (root / "golden" / "narrative-v1.jsonl").write_text(dumps_cases(cases))


@pytest.fixture
def root(tmp_path):
    lock = write_eval_root(tmp_path)
    _narratives(tmp_path, lock)
    return tmp_path


def test_the_build_balances_classes_and_marks_the_e2e_tier(root):
    built = build(EvalPaths(root), SMALL)
    stats = json.loads(built.stats)

    for split in ("dev", "eval"):
        positives = stats["gate_only"][f"{split}/positive"]
        assert positives == 96
        assert sum(stats["gate_only"][f"{split}/{kind}"] for kind in NEGATIVE_KINDS) == positives
        assert stats["e2e"][f"{split}/positive"] == 4
        assert sum(stats["e2e"][f"{split}/{kind}"] for kind in NEGATIVE_KINDS) == 8
    assert stats["golden_sha256"] in built.stats.decode()
    assert {case.fold for case in built.cases if case.split == "eval"} == {0, 1}


def test_writing_then_checking_is_byte_identical_and_a_change_is_caught(root, capsys):
    paths = EvalPaths(root)

    assert run_build(paths, check=False, quotas=SMALL) == 0
    assert run_build(paths, check=True, quotas=SMALL) == 0
    assert "rebuilt byte for byte" in capsys.readouterr().out

    lines = paths.golden.read_text().splitlines()
    paths.golden.write_text("\n".join(lines[1:]) + "\n")
    assert run_build(paths, check=True, quotas=SMALL) == 1
    assert "v1.jsonl" in capsys.readouterr().err


def test_every_written_line_validates_and_lines_are_sorted_by_split_company_and_id(root):
    paths = EvalPaths(root)
    run_build(paths, check=False, quotas=SMALL)

    cases = load_cases(paths.golden, GoldenCase)

    assert len(cases) == 2 * 96 * 2
    keys = [(case.split, case.ticker, case.id) for case in cases]
    assert keys == sorted(keys)


def test_a_tampered_snapshot_fails_the_build(root, capsys):
    paths = EvalPaths(root)
    paths.snapshots.companyfacts("AAA").write_bytes(b"\x1f\x8b broken")

    assert run_build(paths, check=True, quotas=SMALL) == 1
    assert "build failed" in capsys.readouterr().err


def test_a_pool_too_small_for_its_quota_fails_the_build(root):
    with pytest.raises(ValueError, match="e2e"):
        build(EvalPaths(root), {"dev": E2eQuota(positives=99, negatives=4)})


def test_an_eval_question_naming_a_brier_company_is_refused(tmp_path):
    toyota = FakeCompany("TTT", 1007, "Toyota", "eval", base=59)
    lock = write_eval_root(tmp_path, DEV + (EVAL[0], toyota))
    _narratives(tmp_path, lock)

    with pytest.raises(ValueError, match="leak"):
        build(EvalPaths(tmp_path), SMALL)
    assert "Toyota" in BRIER_NAMES


def test_the_brier_template_is_refused_in_any_split():
    case = GoldenCase.model_validate(
        {
            "id": "x",
            "split": "dev",
            "e2e": False,
            "ticker": "A",
            "cik": 1,
            "fiscal_year": 2025,
            "accession": "0000000001-26-000001",
            "template": "t",
            "answerable": False,
            "negative_kind": "off_domain",
            "question": "Does the passage answer this question?",
        }
    )
    with pytest.raises(ValueError, match="leak"):
        check_leakage([case])


def test_narratives_are_checked_against_their_filing(root):
    paths = EvalPaths(root)
    lock = read_lock(paths.lock)

    assert check_narratives(paths, lock) == []
    filing = lock.filing("CCC", 2025)
    ref = {"ticker": "CCC", "cik": filing.cik, "fiscal_year": 2025, "accession": filing.accession}
    broken = [
        NarrativeCase(
            id="a",
            question="q?x",
            answerable=True,
            expected_item="Item 7",
            evidence_terms=("blockchain",),
            **ref,
        ),
        NarrativeCase(
            id="b",
            question="q?x",
            answerable=True,
            expected_item="Item 4",
            evidence_terms=("goods",),
            **ref,
        ),
        NarrativeCase(
            id="c", question="q?x", answerable=False, absent_terms=("durable goods",), **ref
        ),
        NarrativeCase(
            id="d",
            question="q?x",
            answerable=False,
            absent_terms=("x",),
            **{**ref, "fiscal_year": 2019},
        ),
        NarrativeCase(
            id="e",
            question="q?x",
            answerable=False,
            absent_terms=("x",),
            **{**ref, "accession": lock.filing("CCC", 2024).accession},
        ),
    ]
    paths.narrative.write_text(dumps_cases(broken))

    problems = check_narratives(paths, lock)

    assert [problem.split(":")[0] for problem in problems] == ["a", "b", "c", "d", "e"]


def test_a_missing_narrative_file_is_a_problem(root):
    paths = EvalPaths(root)
    paths.narrative.unlink()

    assert check_narratives(paths, read_lock(paths.lock)) == ["narrative-v1.jsonl is missing"]


def test_stats_prints_the_tables_and_refuses_a_stale_file(root, capsys, monkeypatch):
    paths = EvalPaths(root)
    run_build(paths, check=False, quotas=SMALL)
    capsys.readouterr()

    assert main(["--root", str(root), "stats"]) == 0
    out = capsys.readouterr().out
    assert "gate_only" in out and "dev/positive" in out and "golden sha256" in out

    stats = json.loads(paths.stats.read_text())
    stats["gate_only"]["dev/positive"] += 1
    paths.stats.write_text(json.dumps(stats))
    assert main(["--root", str(root), "stats"]) == 1
