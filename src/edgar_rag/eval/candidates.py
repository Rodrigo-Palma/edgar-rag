"""Every question the snapshots can label, and every fact dropped on the way.

The pools here are larger than the golden set; ``selection`` samples from them.
Each candidate is checked against the filing it is scoped to:

* a positive is kept only when the value its filing's XBRL reported is printed
  in the filing's indexed text (``PrintedNumbers.prints``), so a parser that
  loses a table is not later scored as a retrieval failure;
* ``wrong_year``: the question asks for a year four to seven years before the
  filing's, and no value the company ever reported for that year may be
  printed anywhere in the filing (``may_print``, the loose rule);
* ``other_company``: the question names another company of the same split and
  asks for a year the filing covers; that company's value may not be printed;
* ``unreported_concept``: the company never reported the concept in any year
  and none of its phrases occurs in the filing;
* ``off_domain``: the question's marker phrase does not occur in the filing.

A dropped fact is counted once per reason (not once per phrasing) in ``Drops``.
"""

import collections
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal

from edgar_rag.edgar.xbrl import (
    PERIOD_END_TOLERANCE_DAYS,
    CompanyFacts,
    Fact,
    XbrlError,
    period_end,
    reported_fact,
)
from edgar_rag.eval.golden import GoldenCase
from edgar_rag.eval.snapshot import LockedCompany, LockedFiling
from edgar_rag.eval.templates import (
    OFF_DOMAIN,
    POSITIVE_CONCEPTS,
    SECTOR_CONCEPTS,
    SECTOR_PHRASINGS,
    OffDomainQuestion,
    PositiveConcept,
    SectorConcept,
    fill,
)
from edgar_rag.eval.textmatch import PrintedNumbers, contains_phrase

COVERED_YEARS_BACK = (0, 1)  # the filing's year and the comparative year before it
WRONG_YEARS_BACK = (4, 5, 6, 7)


@dataclass(frozen=True, slots=True)
class FilingText:
    """What a filing prints, read once: the indexed passages and the whole text."""

    filing: LockedFiling
    indexed: PrintedNumbers
    full: PrintedNumbers
    words: str


@dataclass(frozen=True, slots=True)
class CompanyData:
    company: LockedCompany
    facts: CompanyFacts
    concepts: frozenset[str]
    filings: tuple[FilingText, ...]
    fold: int | None

    def filing_for(self, fiscal_year: int) -> FilingText | None:
        for text in self.filings:
            if text.filing.fiscal_year == fiscal_year:
                return text
        return None


@dataclass
class Drops:
    """Facts left out, by subtype, concept and reason."""

    counts: collections.Counter[tuple[str, str, str]] = field(default_factory=collections.Counter)

    def add(self, kind: str, concept: str, reason: str) -> None:
        self.counts[(kind, concept, reason)] += 1


def case_id(*parts: object) -> str:
    return ":".join(str(part) for part in parts)


def _base(company: CompanyData, text: FilingText) -> dict[str, object]:
    return {
        "split": company.company.split,
        "fold": company.fold,
        "e2e": False,
        "ticker": company.company.ticker,
        "cik": company.company.cik,
        "fiscal_year": text.filing.fiscal_year,
        "accession": text.filing.accession,
    }


def filed_value(
    company: CompanyData, concept: PositiveConcept, filing: LockedFiling, years_back: int
) -> Fact | None:
    """The value ``filing`` reported for ``concept``, trying its tags in order.

    Raises:
        XbrlError: when the filing reported two values for the period.
    """
    for tag in concept.tags:
        fact = reported_fact(
            company.facts,
            concept=tag,
            unit=concept.unit,
            accession=filing.accession,
            report_date=filing.report_date,
            years_before=years_back,
        )
        if fact is not None:
            return fact
    return None


def positives(company: CompanyData, drops: Drops) -> Iterator[GoldenCase]:
    for text in company.filings:
        for concept in POSITIVE_CONCEPTS:
            for years_back in COVERED_YEARS_BACK:
                fact = _verified_positive(company, text, concept, years_back, drops)
                if fact is not None:
                    yield from _positive_cases(company, text, concept, years_back, fact)


def _verified_positive(
    company: CompanyData,
    text: FilingText,
    concept: PositiveConcept,
    years_back: int,
    drops: Drops,
) -> Fact | None:
    try:
        fact = filed_value(company, concept, text.filing, years_back)
    except XbrlError:
        drops.add("positive", concept.key, "conflicting_values")
        return None
    if fact is None:
        drops.add("positive", concept.key, "no_xbrl_fact")
    elif fact.value == 0:
        drops.add("positive", concept.key, "zero_value")
    elif not text.indexed.prints(fact.value, per_share=concept.per_share):
        drops.add("positive", concept.key, "not_in_indexed_text")
    else:
        return fact
    return None


def _positive_cases(
    company: CompanyData,
    text: FilingText,
    concept: PositiveConcept,
    years_back: int,
    fact: Fact,
) -> Iterator[GoldenCase]:
    year = text.filing.fiscal_year - years_back
    ticker, fiscal_year = company.company.ticker, text.filing.fiscal_year
    for number, phrasing in enumerate(concept.phrasings):
        yield GoldenCase.model_validate(
            {
                **_base(company, text),
                "id": case_id(
                    "pos", ticker, fiscal_year, concept.key, f"y{years_back}", f"p{number}"
                ),
                "question": fill(phrasing, company=company.company.name, year=year),
                "template": f"{concept.key}/{number}",
                "answerable": True,
                "concept": f"us-gaap:{fact.concept}",
                "period_year": year,
                "period_end": fact.end,
                "expected_value": fact.value,
                "unit": fact.unit,
            }
        )


def values_for_period(
    company: CompanyData, concept: PositiveConcept, filing: LockedFiling, years_back: int
) -> frozenset[Decimal]:
    """Every value any 10-K of the company reported for that fiscal year."""
    target = period_end(filing.report_date, years_back)
    tolerance = timedelta(days=PERIOD_END_TOLERANCE_DAYS)
    return frozenset(
        fact.value
        for tag in concept.tags
        for fact in company.facts.of(tag)
        if fact.unit == concept.unit
        and abs(fact.end - target) <= tolerance
        and (fact.is_instant or fact.is_annual)
    )


def wrong_years(company: CompanyData, drops: Drops) -> Iterator[GoldenCase]:
    for text in company.filings:
        for concept in POSITIVE_CONCEPTS:
            for years_back in WRONG_YEARS_BACK:
                values = values_for_period(company, concept, text.filing, years_back)
                if not values:
                    drops.add("wrong_year", concept.key, "no_value_for_year")
                elif any(text.full.may_print(v, per_share=concept.per_share) for v in values):
                    drops.add("wrong_year", concept.key, "value_in_text")
                else:
                    yield from _wrong_year_cases(company, text, concept, years_back)


def _wrong_year_cases(
    company: CompanyData, text: FilingText, concept: PositiveConcept, years_back: int
) -> Iterator[GoldenCase]:
    year = text.filing.fiscal_year - years_back
    ticker, fiscal_year = company.company.ticker, text.filing.fiscal_year
    for number, phrasing in enumerate(concept.phrasings):
        yield GoldenCase.model_validate(
            {
                **_base(company, text),
                "id": case_id("wrong_year", ticker, fiscal_year, concept.key, year, f"p{number}"),
                "question": fill(phrasing, company=company.company.name, year=year),
                "template": f"{concept.key}/{number}",
                "answerable": False,
                "negative_kind": "wrong_year",
                "concept": f"us-gaap:{concept.tags[0]}",
                "period_year": year,
            }
        )


def other_companies(
    company: CompanyData, peers: tuple[CompanyData, ...], drops: Drops
) -> Iterator[GoldenCase]:
    """Questions about each peer's figures, scoped to this company's filings."""
    for text in company.filings:
        for peer in peers:
            if peer.company.ticker == company.company.ticker:
                continue
            peer_filing = peer.filing_for(text.filing.fiscal_year)
            if peer_filing is None:
                continue
            for concept in POSITIVE_CONCEPTS:
                for years_back in COVERED_YEARS_BACK:
                    yield from _other_company_cases(
                        company, text, peer, peer_filing, concept, years_back, drops
                    )


def _other_company_cases(
    company: CompanyData,
    text: FilingText,
    peer: CompanyData,
    peer_filing: FilingText,
    concept: PositiveConcept,
    years_back: int,
    drops: Drops,
) -> Iterator[GoldenCase]:
    try:
        fact = filed_value(peer, concept, peer_filing.filing, years_back)
    except XbrlError:
        fact = None
    if fact is None or fact.value == 0:
        drops.add("other_company", concept.key, "no_value_for_asked_company")
        return
    if text.full.may_print(fact.value, per_share=concept.per_share):
        drops.add("other_company", concept.key, "value_in_text")
        return
    year = text.filing.fiscal_year - years_back
    ticker, fiscal_year = company.company.ticker, text.filing.fiscal_year
    for number, phrasing in enumerate(concept.phrasings):
        yield GoldenCase.model_validate(
            {
                **_base(company, text),
                "id": case_id(
                    "other_company",
                    ticker,
                    fiscal_year,
                    peer.company.ticker,
                    concept.key,
                    f"y{years_back}",
                    f"p{number}",
                ),
                "question": fill(phrasing, company=peer.company.name, year=year),
                "template": f"{concept.key}/{number}",
                "answerable": False,
                "negative_kind": "other_company",
                "concept": f"us-gaap:{fact.concept}",
                "period_year": year,
                "asked_company": peer.company.ticker,
            }
        )


def unreported_concepts(company: CompanyData, drops: Drops) -> Iterator[GoldenCase]:
    for text in company.filings:
        for concept in SECTOR_CONCEPTS:
            if concept.tag in company.concepts:
                drops.add("unreported_concept", concept.key, "reported_by_company")
            elif any(contains_phrase(text.words, phrase) for phrase in concept.absent_phrases):
                drops.add("unreported_concept", concept.key, "phrase_in_text")
            else:
                yield from _unreported_cases(company, text, concept)


def _unreported_cases(
    company: CompanyData, text: FilingText, concept: SectorConcept
) -> Iterator[GoldenCase]:
    ticker, fiscal_year = company.company.ticker, text.filing.fiscal_year
    for years_back in COVERED_YEARS_BACK:
        year = text.filing.fiscal_year - years_back
        for number, phrasing in enumerate(SECTOR_PHRASINGS):
            question = fill(phrasing, company=company.company.name, year=year, label=concept.label)
            yield GoldenCase.model_validate(
                {
                    **_base(company, text),
                    "id": case_id(
                        "unreported_concept",
                        ticker,
                        fiscal_year,
                        concept.key,
                        f"y{years_back}",
                        f"p{number}",
                    ),
                    "question": question,
                    "template": f"sector/{number}",
                    "answerable": False,
                    "negative_kind": "unreported_concept",
                    "concept": f"us-gaap:{concept.tag}",
                    "period_year": year,
                }
            )


def off_domain(company: CompanyData, drops: Drops) -> Iterator[GoldenCase]:
    for text in company.filings:
        for question in OFF_DOMAIN:
            if contains_phrase(text.words, question.marker):
                drops.add("off_domain", question.key, "marker_in_text")
                continue
            yield _off_domain_case(company, text, question)


def _off_domain_case(
    company: CompanyData, text: FilingText, question: OffDomainQuestion
) -> GoldenCase:
    ticker = company.company.ticker
    return GoldenCase.model_validate(
        {
            **_base(company, text),
            "id": case_id("off_domain", ticker, text.filing.fiscal_year, question.key),
            "question": fill(question.text, company=company.company.name),
            "template": f"off_domain/{question.key}",
            "answerable": False,
            "negative_kind": "off_domain",
        }
    )
