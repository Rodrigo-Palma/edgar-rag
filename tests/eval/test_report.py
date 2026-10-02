import random

import pytest

from edgar_rag.eval.records import (
    BrierInfo,
    ModelInfo,
    RepeatRecord,
    Run,
    RunManifest,
    read_run,
    write_jsonl,
    write_manifest,
)
from edgar_rag.eval.report import ReportOptions, build_report
from tests.eval.harness_fakes import record

KINDS = ("wrong_year", "other_company", "unreported_concept", "off_domain")
FAST = ReportOptions(resamples=200)


def _records(*, folds: bool, brier: bool, seed: int = 7):
    """Eight companies, each with answerable and unanswerable cases scored by every gate."""
    rng = random.Random(seed)
    rows = []
    for company in range(8):
        fold = company % 2 if folds else None
        for i in range(12):
            answerable = i < 6
            kind = None if answerable else KINDS[i % 4]
            cosine = rng.uniform(0.5, 1.0) if answerable else rng.uniform(0.2, 0.8)
            answered = rng.random() < (0.9 if answerable else 0.4)
            right = answerable and answered and rng.random() < 0.8
            rows.append(
                record(
                    f"c{company}-{i:02d}",
                    answerable=answerable,
                    cosine=round(cosine, 4),
                    period=0.0 if kind == "wrong_year" else 1.0,
                    brier=round(rng.random(), 4) if brier else None,
                    outcome="answered" if answered else "model_declined",
                    fold=fold,
                    ticker=f"T{company}",
                    negative_kind=kind,
                    numeric_correct=right if answered and answerable else None,
                    citation_supported=right if answered and answerable else None,
                    generator_seconds=2.5,
                    prompt_tokens=900,
                    completion_tokens=30 if answered else 5,
                    stages={"embed": 0.02, "search": 0.001, "gate": 0.0, "generate": 2.5},
                )
            )
    return tuple(rows)


def _manifest(*, split="eval", brier=True, mode="record", cases=96, repeats=0):
    return RunManifest(
        split=split,
        tier="e2e",
        mode=mode,
        limit=None,
        repo_sha="0123456789abcdef",
        repo_dirty=False,
        golden_sha256="a" * 64,
        narrative_sha256=None,
        protocol_sha256="b" * 64 if split == "eval" else None,
        index_digest="c" * 16,
        index_filings=16,
        embedding=ModelInfo(name="nomic-embed-text", digest="0a109f422b47"),
        generation=ModelInfo(name="qwen3:8b", digest="500a1f067a9f"),
        generation_options={"temperature": 0, "seed": 0, "num_ctx": 8192, "think": False},
        ollama_version="0.18.0",
        top_k=4,
        gates=("period", "cosine", "brier") if brier else ("period", "cosine"),
        brier=BrierInfo(sha="d70e7df", ready=None) if brier else None,
        cases=cases,
        repeats=repeats,
        date="2026-10-01",
        hardware="Apple M3 Max",
        wall_seconds=120.0,
    )


def _run(*, folds=True, brier=True, split="eval", mode="record", repeats=(), extra=()):
    cases = _records(folds=folds, brier=brier) + tuple(extra)
    manifest = _manifest(split=split, brier=brier, mode=mode, cases=len(cases))
    return Run(manifest=manifest, cases=cases, repeats=tuple(repeats))


def test_the_report_is_the_same_text_every_time():
    run = _run()

    assert build_report(run, FAST) == build_report(run, FAST)


def test_every_arm_has_a_row_with_n_and_companies():
    text = build_report(_run(), FAST)

    for label in ("A no gate", "B cosine", "C brier", "D period guard", "E period", "F period"):
        assert f"| {label}" in text
    assert "48 unanswerable and 48 answerable questions, 8 companies" in text
    assert "cross-fitted R90" in text


def test_each_comparison_prints_its_interval_and_mde():
    text = build_report(_run(), FAST)

    assert "H1, FAR(A) - FAR(F) on 48 negatives" in text
    assert "H2, FAR(E) - FAR(F), paired" in text
    assert "97.5% CI" in text
    assert text.count("MDE at 80% power") >= 4
    assert "AUROC(brier) - AUROC(cosine)" in text
    assert "not distinguishable at this n" in text


def test_the_confirmatory_level_is_an_option():
    text = build_report(_run(), ReportOptions(confirmatory_confidence=0.99, resamples=200))

    assert "99.0% CI" in text


def test_without_brier_its_arms_are_printed_as_not_run():
    text = build_report(_run(brier=False), FAST)

    assert "| C brier | not run" in text
    assert "| E period guard + brier | not run" in text
    assert "H2, FAR(E) - FAR(F): not run" in text
    assert "not plugged in; arms C and E not run" in text


def test_a_dev_run_says_it_is_not_a_headline_number_and_fits_in_sample():
    text = build_report(_run(folds=False, split="dev"), FAST)

    assert "Dev split: development and CI only" in text
    assert "in-sample R90 (no folds)" in text


def test_a_replayed_run_does_not_report_replay_times_as_latency():
    text = build_report(_run(mode="replay"), FAST)

    assert "Replayed run: stage times are replay times" in text
    assert "| generate |" not in text


def test_latency_and_cost_are_reported_for_a_recorded_run():
    text = build_report(_run(), FAST)

    assert "| generate | 96 | 2.500 | 2.500 |" in text
    assert "| declined after generating |" in text


def test_narratives_and_determinism_are_reported_apart():
    narrative = record(
        "narrative:answerable:01",
        answerable=True,
        cosine=0.9,
        kind="narrative",
        fold=0,
        ticker="T0",
        cited_items=("Item 1A",),
        expected_item="Item 1A",
        item_split_sane=True,
    )
    repeats = [RepeatRecord(id="c0-00", identical=True, first_seconds=2, second_seconds=2)]

    text = build_report(_run(extra=(narrative,), repeats=repeats), FAST)

    assert "## Narrative questions" in text
    assert "Expected item cited" in text and "1/1 = 100.0%" in text
    assert "Same prompt sent twice, text identical: 1/1" in text


def test_the_report_never_uses_an_em_dash():
    assert chr(0x2014) not in build_report(_run(), FAST)


def test_a_run_directory_round_trips_and_reports_the_same(tmp_path):
    run = _run()
    write_manifest(tmp_path, run.manifest)
    write_jsonl(tmp_path / "cases.jsonl", reversed(run.cases))
    (tmp_path / "deviations.md").write_text("Fold seed changed.\n")

    read = read_run(tmp_path)

    assert read.cases == tuple(sorted(run.cases, key=lambda r: r.id))
    text = build_report(read, FAST)
    assert "Fold seed changed." in text
    assert text == build_report(read_run(tmp_path), FAST)


def test_a_run_whose_cases_do_not_match_its_manifest_is_refused(tmp_path):
    run = _run()
    write_manifest(tmp_path, run.manifest)
    write_jsonl(tmp_path / "cases.jsonl", run.cases[:3])

    with pytest.raises(ValueError, match="lists 96 cases"):
        read_run(tmp_path)
