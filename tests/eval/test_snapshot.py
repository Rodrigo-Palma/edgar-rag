"""The roster, lock and snapshots: read back exactly, refused when tampered with."""

import gzip
import json

import pytest

from edgar_rag.eval.snapshot import (
    SnapshotPaths,
    check_roster_ciks,
    gzip_bytes,
    load_roster,
    read_companyfacts_snapshot,
    read_lock,
    read_text_snapshot,
    reported_concepts,
    trim_companyfacts,
    xz_bytes,
)
from tests.eval.golden_fakes import DEV, EVAL, companyfacts, roster_toml, write_eval_root


def test_the_roster_reads_companies_and_reserves_in_order(tmp_path):
    path = tmp_path / "companies.toml"
    path.write_text(roster_toml(DEV, reserves=EVAL))

    roster = load_roster(path)

    assert [spec.ticker for spec in roster.companies] == ["AAA", "BBB"]
    assert [spec.split for spec in roster.reserves] == [None, None]
    assert roster.fiscal_years == (2024, 2025)
    assert [spec.ticker for spec in roster.everyone()] == ["AAA", "BBB", "CCC", "DDD"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (roster_toml(DEV + DEV), "twice"),
        (roster_toml(DEV).replace('split = "dev"', 'split = "test"', 1), "unknown split"),
        (roster_toml(DEV).replace("fold_seed = 7", ""), "missing"),
    ],
)
def test_a_broken_roster_is_refused(tmp_path, text, message):
    path = tmp_path / "companies.toml"
    path.write_text(text)

    with pytest.raises(ValueError, match=message):
        load_roster(path)


def test_a_cik_the_sec_maps_elsewhere_is_named(tmp_path):
    path = tmp_path / "companies.toml"
    path.write_text(roster_toml(DEV))
    roster = load_roster(path)

    check_roster_ciks(roster, {"AAA": 1001, "BBB": 1002})
    with pytest.raises(ValueError, match="BBB: listed 1002, SEC maps 9"):
        check_roster_ciks(roster, {"AAA": 1001, "BBB": 9})


def test_trimming_keeps_the_shape_the_10k_facts_and_every_concept_name():
    trimmed = trim_companyfacts(companyfacts(DEV[0]), ["Revenues", "Deposits"])
    payload = json.loads(trimmed)

    assert list(payload["facts"]["us-gaap"]) == ["Revenues"]
    assert reported_concepts(trimmed) == {
        "AccountsPayableCurrent",
        "Assets",
        "EarningsPerShareDiluted",
        "Revenues",
    }
    facts = payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    assert {fact["form"] for fact in facts} == {"10-K"}


def test_compression_is_deterministic():
    assert gzip_bytes(b"same") == gzip_bytes(b"same")
    assert gzip.decompress(gzip_bytes(b"same")) == b"same"
    assert xz_bytes(b"same") == xz_bytes(b"same")


def test_snapshots_read_back_and_a_tampered_one_is_refused(tmp_path):
    lock = write_eval_root(tmp_path)
    paths = SnapshotPaths(tmp_path / "snapshots")
    filing = lock.filings[0]

    assert read_lock(tmp_path / "filings.lock.json") == lock
    assert "Total revenues" in read_text_snapshot(paths, filing)
    assert read_companyfacts_snapshot(paths, lock.companies[0])

    paths.text(filing).write_bytes(xz_bytes(b"edited"))
    with pytest.raises(ValueError, match="SHA-256"):
        read_text_snapshot(paths, filing)
    paths.companyfacts("AAA").write_bytes(gzip_bytes(b"{}"))
    with pytest.raises(ValueError, match="does not match"):
        read_companyfacts_snapshot(paths, lock.companies[0])


def test_the_lock_finds_filings_and_companies(tmp_path):
    lock = write_eval_root(tmp_path)

    assert lock.filing("CCC", 2024).accession == EVAL[0].accession(2024)
    assert lock.company("DDD").split == "eval"
    with pytest.raises(KeyError):
        lock.filing("CCC", 2019)
    with pytest.raises(KeyError):
        lock.company("ZZZ")
