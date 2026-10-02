"""Pinning the inputs: the right 10-Ks, reserves for exclusions, never faster than the pace."""

import itertools
import json

import httpx
import pytest

from edgar_rag.edgar.client import EdgarClient
from edgar_rag.edgar.errors import EdgarError
from edgar_rag.edgar.fetch import MIN_REQUEST_INTERVAL_SECONDS, Fetcher
from edgar_rag.edgar.xbrl import parse_companyfacts
from edgar_rag.eval.acquire import (
    accessions_by_fiscal_year,
    acquire,
    check_unchanged,
    fetch_tickers,
    pin_roster,
)
from edgar_rag.eval.snapshot import SnapshotPaths, load_roster, read_lock, read_text_snapshot
from tests.edgar_fakes import FakeClock, RecordingTransport
from tests.eval.golden_fakes import (
    DEV,
    EVAL,
    FakeCompany,
    FakeSec,
    companyfacts,
    roster_toml,
    submissions,
)

USER_AGENT = "Test Runner tests@ledgerworks.io"
HEADERS = {"User-Agent": USER_AGENT}
LATE = FakeCompany(
    "LTE", 1009, "Larch Late", "dev", base=53, report_month=1, report_day=31, report_year_offset=1
)


def _world(tmp_path, companies, reserves=(), **overrides):
    clock = FakeClock()
    transport = RecordingTransport(FakeSec(companies + reserves, **overrides), clock)
    http = httpx.Client(transport=transport)
    client = EdgarClient(USER_AGENT, http=http, clock=clock)
    fetcher = Fetcher(http, HEADERS, clock)
    path = tmp_path / "companies.toml"
    path.write_text(roster_toml(companies, reserves))
    return client, fetcher, clock, transport, load_roster(path)


def test_acquire_pins_both_10ks_and_writes_snapshots_matching_the_lock(tmp_path):
    client, fetcher, clock, transport, roster = _world(tmp_path, DEV)

    lock = acquire(
        client,
        fetcher,
        clock,
        roster,
        snapshots=tmp_path / "snapshots",
        lock_path=tmp_path / "lock.json",
        log=lambda _: None,
    )

    assert [(f.ticker, f.fiscal_year) for f in lock.filings] == [
        ("AAA", 2024),
        ("AAA", 2025),
        ("BBB", 2024),
        ("BBB", 2025),
    ]
    assert lock.filing("AAA", 2025).accession == DEV[0].accession(2025)
    assert read_lock(tmp_path / "lock.json") == lock
    text = read_text_snapshot(SnapshotPaths(tmp_path / "snapshots"), lock.filing("BBB", 2024))
    assert "Total revenues" in text
    gaps = [later - earlier for earlier, later in itertools.pairwise(transport.times)]
    assert min(gaps) >= MIN_REQUEST_INTERVAL_SECONDS - 1e-9


def test_a_company_whose_report_year_differs_from_its_fiscal_year_is_replaced(tmp_path):
    client, _, _, _, roster = _world(tmp_path, (DEV[0], LATE), reserves=(EVAL[0],))
    logged = []

    pinned, excluded = pin_roster(client, roster, logged.append)

    assert [p.company.ticker for p in pinned] == ["AAA", "CCC"]
    assert pinned[1].company.split == "dev"
    assert excluded[0].ticker == "LTE"
    assert "fy 2024 but report date 2025-01-31" in excluded[0].reason
    assert excluded[0].replaced_by == "CCC"
    assert any("excluded LTE" in line for line in logged)


def test_running_out_of_reserves_stops_the_run(tmp_path):
    client, _, _, _, roster = _world(tmp_path, (LATE,))

    with pytest.raises(EdgarError, match="no reserve left"):
        pin_roster(client, roster, lambda _: None)


def test_a_10k_that_left_the_recent_list_is_found_in_the_nearest_older_page(tmp_path):
    alder = DEV[0]
    recent = submissions(alder, recent_years=(2025,))
    recent["filings"]["files"] = [
        {
            "name": "CIK0000001001-submissions-002.json",
            "filingCount": 1,
            "filingFrom": "2020-01-01",
            "filingTo": "2023-12-31",
        },
        {
            "name": "CIK0000001001-submissions-001.json",
            "filingCount": 1,
            "filingFrom": "2025-02-16",
            "filingTo": "2025-12-31",
        },
    ]
    older = submissions(alder, recent_years=(2024,))["filings"]["recent"]
    base = "https://data.sec.gov/submissions/"
    client, _, _, transport, roster = _world(
        tmp_path,
        (alder,),
        **{
            f"{base}CIK0000001001.json": recent,
            # the 10-K was filed on 2025-02-15, the day between the two pages' ranges
            f"{base}CIK0000001001-submissions-001.json": {
                **older,
                "accessionNumber": [alder.accession(2024)],
            },
        },
    )

    pinned, _ = pin_roster(client, roster, lambda _: None)

    assert [f.accession for f in pinned[0].filings] == [
        alder.accession(2024),
        alder.accession(2025),
    ]
    assert not any("submissions-002" in url for url in transport.urls)


def test_a_document_whose_bytes_changed_since_the_lock_is_refused(tmp_path):
    client, fetcher, clock, _, roster = _world(tmp_path, DEV)
    kwargs = {
        "snapshots": tmp_path / "s",
        "lock_path": tmp_path / "lock.json",
        "log": lambda _: None,
    }
    lock = acquire(client, fetcher, clock, roster, **kwargs)
    first = lock.filings[0]
    tampered = lock.model_copy(
        update={"filings": (first.model_copy(update={"document_sha256": "0" * 64}),)}
    )
    pinned, _ = pin_roster(client, roster, lambda _: None)

    with pytest.raises(EdgarError, match=first.accession):
        check_unchanged(tampered, pinned)
    check_unchanged(lock, pinned)


def test_a_cik_the_ticker_map_disagrees_with_stops_before_any_filing(tmp_path):
    url = "https://www.sec.gov/files/company_tickers.json"
    payload = {
        "0": {"cik_str": 77, "ticker": "AAA", "title": "x"},
        "1": {"cik_str": 1002, "ticker": "BBB", "title": "y"},
    }
    client, fetcher, clock, transport, roster = _world(tmp_path, DEV, **{url: payload})

    with pytest.raises(ValueError, match="AAA: listed 1001, SEC maps 77"):
        acquire(
            client, fetcher, clock, roster, snapshots=tmp_path, lock_path=tmp_path / "l", log=print
        )
    assert transport.urls == [url]


def test_a_malformed_ticker_map_is_an_edgar_error(tmp_path):
    url = "https://www.sec.gov/files/company_tickers.json"
    _, fetcher, clock, _, _ = _world(tmp_path, DEV, **{url: b"[1, 2]"})

    with pytest.raises(EdgarError, match="malformed"):
        fetch_tickers(fetcher, clock)


def test_each_fiscal_year_maps_to_the_accession_whose_facts_carry_it():
    facts = parse_companyfacts(companyfacts(DEV[0]))

    by_year = accessions_by_fiscal_year(facts)

    assert by_year[2024] == [DEV[0].accession(2024)]
    assert json.dumps(sorted(by_year)) == json.dumps(list(range(2017, 2026)))
