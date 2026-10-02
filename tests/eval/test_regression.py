"""The CI evaluation gate: how a replayed run is judged against its baseline."""

import json

import pytest

from edgar_rag import cli
from edgar_rag.eval.records import CASES_FILE, ModelInfo, RunManifest, write_jsonl, write_manifest
from edgar_rag.eval.regression import (
    NET_CHANGE_LIMIT,
    Baseline,
    compare,
    dumps,
    judge,
    judge_all,
    loads,
    render,
)
from edgar_rag.eval.regression_command import served_cosine_threshold
from tests.eval.harness_fakes import record

THRESHOLD = 0.55


def _correct(case_id: str, **fields):
    graded = {"numeric_correct": True, "citation_supported": True, **fields}
    return record(case_id, answerable=True, cosine=0.8, **graded)


def _wrong_year(case_id: str, *, period: float = 0.0, outcome: str = "answered"):
    return record(
        case_id,
        answerable=False,
        cosine=0.8,
        period=period,
        outcome=outcome,
        negative_kind="wrong_year",
    )


def _baseline(records, threshold: float = THRESHOLD, golden: str = "a" * 64) -> Baseline:
    return Baseline(
        cosine_threshold=threshold,
        golden_sha256=golden,
        index_digest="c" * 16,
        generation=ModelInfo(name="qwen3:8b", digest="500a"),
        cases=judge_all(records, threshold),
    )


def test_a_correct_answer_the_gates_admit_is_right_under_every_arm():
    assert judge(_correct("p"), THRESHOLD).right == ("A", "B", "D", "F")


def test_a_wrong_year_the_model_answered_is_right_only_where_the_guard_stops_it():
    judged = judge(_wrong_year("n"), THRESHOLD)

    assert (judged.tier, judged.label, judged.right) == ("e2e", "wrong_year", ("D", "F"))


def test_an_answer_without_the_gold_value_is_wrong_even_when_answered():
    judged = judge(_correct("p", numeric_correct=False), THRESHOLD)

    assert judged.right == ()


def test_a_gate_only_case_is_judged_on_admission_and_arm_a_is_not_judged():
    low = record(
        "g", answerable=True, cosine=0.4, generated=False, outcome="scored_only", e2e=False
    )

    judged = judge(low, THRESHOLD)

    assert (judged.tier, judged.right) == ("gate-only", ("D",))


def test_narratives_are_not_part_of_the_ci_tier():
    narrative = _correct("n", kind="narrative")

    assert judge_all([narrative, _correct("p")], THRESHOLD)[0].id == "p"
    assert len(judge_all([narrative], THRESHOLD)) == 0


def test_an_end_to_end_case_whose_generation_disappears_is_judged_end_to_end_as_worse():
    """A pipeline that stops asking the model must not move cases to the gate-only tier."""
    skipped = _correct("p", generated=False, outcome="scored_only")

    judged = judge(skipped, THRESHOLD)
    verdict = compare(_baseline([_correct("p")]), _baseline([skipped]))

    assert (judged.tier, judged.right) == ("e2e", ())
    assert verdict.worse[("e2e", "answerable", "A")] == ["p"]


def _negative(case_id: str, kind: str, outcome: str = "model_declined"):
    return record(case_id, answerable=False, cosine=0.8, outcome=outcome, negative_kind=kind)


def test_a_worsening_spread_over_unanswerable_kinds_fails_on_their_aggregate():
    """Mutation E1: 2 off-domain and 2 other-company questions newly answered."""
    kinds = ("off_domain", "off_domain", "other_company", "other_company")
    before = [_negative(f"n{i}", kind) for i, kind in enumerate(kinds)]
    after = [_negative(f"n{i}", kind, outcome="answered") for i, kind in enumerate(kinds)]

    verdict = compare(_baseline(before), _baseline(after))

    assert not verdict.passed
    assert any(
        "net worsening of 4 cases in e2e unanswerable under arm F" in f for f in verdict.failures
    )
    assert not any("off_domain" in failure for failure in verdict.failures)


def test_answerable_cases_get_no_aggregate_row():
    verdict = compare(_baseline([_correct("p")]), _baseline([_correct("p", numeric_correct=False)]))

    assert {group[1] for group in verdict.worse} == {"answerable"}


def test_an_outcome_change_that_keeps_the_case_right_is_reported_not_failed():
    before = [_negative(f"n{i}", "off_domain") for i in range(5)]
    after = [_negative(f"n{i}", "off_domain", outcome="no_valid_citation") for i in range(5)]

    verdict = compare(_baseline(before), _baseline(after))
    text = render(verdict, _baseline(after))

    assert verdict.passed
    assert verdict.silent_transitions == 5
    assert "| model_declined | no_valid_citation | 5 |" in text


def test_the_same_run_passes_with_nothing_changed():
    cases = [_correct("p"), _wrong_year("n")]

    verdict = compare(_baseline(cases), _baseline(cases))

    assert verdict.passed
    assert not verdict.worse and not verdict.better


def _guard_off(count: int, total: int = 5) -> tuple[Baseline, Baseline]:
    """``count`` of ``total`` wrong-year answers the guard no longer stops."""
    before = [_wrong_year(f"n{i}") for i in range(total)]
    after = [_wrong_year(f"n{i}", period=1.0 if i < count else 0.0) for i in range(total)]
    return _baseline(before), _baseline(after)


def test_a_net_worsening_at_the_limit_fails_naming_the_group():
    baseline, current = _guard_off(NET_CHANGE_LIMIT)

    verdict = compare(baseline, current)

    assert not verdict.passed
    assert any(
        f"net worsening of {NET_CHANGE_LIMIT} cases in e2e wrong_year under arm D" in failure
        for failure in verdict.failures
    )


def test_a_worsening_below_the_limit_passes_and_is_still_listed():
    baseline, current = _guard_off(NET_CHANGE_LIMIT - 1)

    verdict = compare(baseline, current)

    assert verdict.passed
    assert verdict.worse[("e2e", "wrong_year", "D")] == ["n0", "n1"]


def test_cases_that_got_better_offset_cases_that_got_worse_in_the_same_group():
    before = [_wrong_year(f"n{i}", period=1.0 if i == 4 else 0.0) for i in range(5)]
    after = [_wrong_year(f"n{i}", period=1.0 if i < NET_CHANGE_LIMIT else 0.0) for i in range(5)]

    verdict = compare(_baseline(before), _baseline(after))

    assert verdict.passed


def test_an_improvement_at_the_limit_fails_until_the_baseline_records_it():
    baseline, current = _guard_off(NET_CHANGE_LIMIT)

    verdict = compare(current, baseline)

    assert "net improvement" in verdict.failures[0]
    assert "make eval-ci-baseline" in verdict.failures[0]


def test_a_baseline_judged_at_another_threshold_is_stale():
    cases = [_correct("p")]

    verdict = compare(_baseline(cases, threshold=0.5), _baseline(cases))

    assert "judged at cosine 0.5" in verdict.failures[0]


@pytest.mark.parametrize(
    ("current", "golden"),
    [([_correct("p"), _correct("q")], "a" * 64), ([_correct("p")], "f" * 64)],
    ids=["new-case", "golden-changed"],
)
def test_cases_the_baseline_did_not_judge_make_it_stale(current, golden):
    verdict = compare(_baseline([_correct("p")]), _baseline(current, golden=golden))

    assert "not the ones the baseline judged" in verdict.failures[0]


def test_a_baseline_round_trips_with_one_case_per_line():
    baseline = _baseline([_correct("p"), _wrong_year("n")])

    text = dumps(baseline)

    assert loads(text) == baseline
    assert json.loads(text)["cases"][0]["id"] == "n"
    assert sum(line.startswith('  {"id"') for line in text.splitlines()) == 2


def test_the_rendered_verdict_names_what_got_worse_and_why_it_failed():
    baseline, current = _guard_off(NET_CHANGE_LIMIT)

    text = render(compare(baseline, current), current)

    assert "| e2e | wrong_year | D | 3 | 0 | +3 |" in text
    assert "`n0`" in text
    assert "FAIL: net worsening of 3 cases" in text
    assert "95% Wilson" in text


def test_the_rendered_pass_says_nothing_changed():
    cases = [_correct("p")]

    text = render(compare(_baseline(cases), _baseline(cases)), _baseline(cases))

    assert "Every case judged as in the baseline" in text
    assert text.rstrip().endswith(f"no group changed by a net {NET_CHANGE_LIMIT} cases or more.")


# --- the command -----------------------------------------------------------


def _manifest(cases: int, mode: str = "replay") -> RunManifest:
    return RunManifest(
        split="dev",
        tier="e2e",
        mode=mode,
        limit=None,
        repo_sha="0123456789abcdef",
        repo_dirty=False,
        golden_sha256="a" * 64,
        narrative_sha256=None,
        protocol_sha256=None,
        index_digest="c" * 16,
        index_filings=8,
        embedding=ModelInfo(name="nomic-embed-text", digest="0a10"),
        generation=ModelInfo(name="qwen3:8b", digest="500a"),
        generation_options={},
        ollama_version=None,
        top_k=4,
        gates=("period", "cosine"),
        brier=None,
        cases=cases,
        repeats=0,
        date="2026-10-01",
        hardware="test",
        wall_seconds=1.0,
    )


def _write_run(directory, records, mode: str = "replay"):
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / CASES_FILE, records)
    write_manifest(directory, _manifest(len(records), mode))
    return directory


def _ci(*args) -> int:
    return cli.main(["eval", "ci", *map(str, args)])


def test_the_command_writes_a_baseline_and_then_passes_against_it(tmp_path, capsys):
    run = _write_run(tmp_path / "run", [_correct("p"), _wrong_year("n")])
    baseline = tmp_path / "baseline.json"

    assert _ci("--run", run, "--baseline", baseline, "--write") == 0
    assert _ci("--run", run, "--baseline", baseline) == 0
    assert "PASS" in capsys.readouterr().out
    assert loads(baseline.read_text()).cosine_threshold == served_cosine_threshold()


def test_the_command_fails_on_a_regression(tmp_path, capsys):
    before = _write_run(tmp_path / "before", [_wrong_year(f"n{i}") for i in range(3)])
    after = _write_run(tmp_path / "after", [_wrong_year(f"n{i}", period=1.0) for i in range(3)])
    baseline = tmp_path / "baseline.json"
    _ci("--run", before, "--baseline", baseline, "--write")

    assert _ci("--run", after, "--baseline", baseline) == 1
    assert "FAIL: net worsening of 3 cases in e2e wrong_year" in capsys.readouterr().out


def test_the_command_warns_about_outcome_changes_that_keep_cases_right(tmp_path, capsys):
    before = _write_run(tmp_path / "before", [_negative("n", "off_domain")])
    after = _write_run(tmp_path / "after", [_negative("n", "off_domain", "no_valid_citation")])
    baseline = tmp_path / "baseline.json"
    _ci("--run", before, "--baseline", baseline, "--write")

    assert _ci("--run", after, "--baseline", baseline) == 0
    assert "::warning title=eval gate::1 cases changed outcome" in capsys.readouterr().out


def test_the_command_judges_only_a_replay(tmp_path, capsys):
    run = _write_run(tmp_path / "run", [_correct("p")], mode="record")

    assert _ci("--run", run, "--baseline", tmp_path / "b.json") == 1
    assert "judges a replay" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "message"), [(None, "no baseline"), ("{}", "is not a baseline")]
)
def test_the_command_says_what_is_wrong_with_the_baseline(tmp_path, capsys, content, message):
    run = _write_run(tmp_path / "run", [_correct("p")])
    baseline = tmp_path / "baseline.json"
    if content is not None:
        baseline.write_text(content)

    assert _ci("--run", run, "--baseline", baseline) == 1
    assert message in capsys.readouterr().err


def test_the_command_refuses_a_directory_that_is_not_a_run(tmp_path, capsys):
    assert _ci("--run", tmp_path, "--baseline", tmp_path / "b.json") == 1
    assert "run:" in capsys.readouterr().err
