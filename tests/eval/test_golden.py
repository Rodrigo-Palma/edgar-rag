"""The golden set's schema refuses an inconsistent label and writes canonical bytes."""

import json
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgar_rag.eval.golden import GoldenCase, NarrativeCase, dumps_cases, load_cases

POSITIVE = {
    "id": "pos:AAA:2025:revenue:y0:p0",
    "split": "eval",
    "fold": 1,
    "e2e": True,
    "ticker": "AAA",
    "cik": 1001,
    "fiscal_year": 2025,
    "accession": "0000001001-26-002025",
    "question": "What was Alder Works's total revenue in fiscal 2025?",
    "template": "revenue/0",
    "answerable": True,
    "concept": "us-gaap:Revenues",
    "period_year": 2025,
    "period_end": "2025-12-31",
    "expected_value": "11175000000",
    "unit": "USD",
}
NEGATIVE = {
    **{key: POSITIVE[key] for key in ("split", "fold", "e2e", "ticker", "cik", "fiscal_year")},
    "id": "off_domain:AAA:2025:poem",
    "accession": POSITIVE["accession"],
    "question": "Write a short poem about Alder Works.",
    "template": "off_domain/poem",
    "answerable": False,
    "negative_kind": "off_domain",
}


def test_a_positive_keeps_its_value_exact():
    case = GoldenCase.model_validate(POSITIVE)

    assert case.expected_value == Decimal("11175000000")
    assert case.period_end == date(2025, 12, 31)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"negative_kind": "wrong_year"}, "no negative kind"),
        ({"expected_value": None}, "needs concept"),
        ({"period_end": None}, "needs the period"),
        ({"split": "dev"}, "carry a fold"),
        ({"asked_company": "BBB"}, "asked_company"),
        ({"surprise": 1}, "Extra inputs"),
    ],
)
def test_an_inconsistent_positive_is_refused(changes, message):
    with pytest.raises(ValidationError, match=message):
        GoldenCase.model_validate({**POSITIVE, **changes})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"negative_kind": None}, "needs its negative kind"),
        ({"expected_value": "1"}, "no expected value"),
        ({"negative_kind": "other_company"}, "asked_company"),
    ],
)
def test_an_inconsistent_negative_is_refused(changes, message):
    with pytest.raises(ValidationError, match=message):
        GoldenCase.model_validate({**NEGATIVE, **changes})


def test_lines_are_canonical_json_with_decimals_as_strings():
    line = dumps_cases([GoldenCase.model_validate(POSITIVE)])

    assert line.endswith("}\n") and line.count("\n") == 1
    payload = json.loads(line)
    assert list(payload) == sorted(payload)
    assert payload["expected_value"] == "11175000000"
    assert ", " not in line and ": " not in line


def test_cases_round_trip_through_a_file(tmp_path):
    cases = [GoldenCase.model_validate(POSITIVE), GoldenCase.model_validate(NEGATIVE)]
    path = tmp_path / "cases.jsonl"
    path.write_text(dumps_cases(cases))

    assert load_cases(path, GoldenCase) == tuple(cases)


def test_a_bad_line_is_named(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(dumps_cases([GoldenCase.model_validate(POSITIVE)]) + "\n{}\n")

    with pytest.raises(ValueError, match="cases.jsonl:3"):
        load_cases(path, GoldenCase)


def test_a_repeated_id_is_refused(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(dumps_cases([GoldenCase.model_validate(POSITIVE)] * 2))

    with pytest.raises(ValueError, match="repeated id"):
        load_cases(path, GoldenCase)


def test_a_narrative_carries_the_evidence_its_label_needs():
    base = {
        "id": "narrative:1",
        "ticker": "AAA",
        "cik": 1001,
        "fiscal_year": 2025,
        "accession": POSITIVE["accession"],
        "question": "What does the company make?",
    }
    NarrativeCase.model_validate(
        {**base, "answerable": True, "expected_item": "Item 1", "evidence_terms": ["goods"]}
    )
    NarrativeCase.model_validate({**base, "answerable": False, "absent_terms": ["poem"]})
    with pytest.raises(ValidationError, match="expected_item and evidence_terms"):
        NarrativeCase.model_validate({**base, "answerable": True, "expected_item": "Item 1"})
    with pytest.raises(ValidationError, match="absent_terms"):
        NarrativeCase.model_validate({**base, "answerable": False, "expected_item": "Item 1"})
