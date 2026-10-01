"""Typed XBRL facts from EDGAR's ``companyfacts`` API.

The trap in this data is the ``fy`` field. It labels the filing a fact came
from, not the period the fact measures: Apple's 10-K for fiscal 2024 reports
revenue for 2022, 2023 and 2024, and all three carry ``fy=2024``. A question
about "revenue in fiscal 2023" built from ``fy`` would quote the 2024 number.

So a fact is tied to a year here only through its dates. ``reported_fact``
takes the filing's ``reportDate`` (the end of the period the filing covers),
moves back whole years from it, and keeps the fact whose period ends there. A
flow (revenue, net income) must also span about a year, which drops the
three-month fourth-quarter figures some 10-Ks tag with the same end date.
"""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Annotated, NoReturn

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from edgar_rag.edgar.errors import EdgarError

# Fiscal years of 52 or 53 weeks run 363 to 371 days from start to end.
ANNUAL_MIN_DAYS = 350
ANNUAL_MAX_DAYS = 380
# A 52/53-week year ends on a weekday, so its end drifts a few days each year.
PERIOD_END_TOLERANCE_DAYS = 7


class XbrlError(EdgarError):
    """Raised when company facts are malformed or a period has two different values."""


@dataclass(frozen=True, slots=True)
class Fact:
    """One reported value.

    ``start`` is ``None`` for a balance at a point in time (total assets) and
    set for an amount over a period (revenue). ``fiscal_year`` and
    ``fiscal_period`` are EDGAR's ``fy`` and ``fp``: they describe the filing,
    and nothing here reads them to decide which year a value belongs to.
    """

    taxonomy: str
    concept: str
    unit: str
    value: Decimal
    start: date | None
    end: date
    fiscal_year: int | None
    fiscal_period: str | None
    form: str
    accession: str
    filed: date

    @property
    def is_instant(self) -> bool:
        return self.start is None

    @property
    def duration_days(self) -> int | None:
        return None if self.start is None else (self.end - self.start).days

    @property
    def is_annual(self) -> bool:
        days = self.duration_days
        return days is not None and ANNUAL_MIN_DAYS <= days <= ANNUAL_MAX_DAYS


@dataclass(frozen=True, slots=True)
class CompanyFacts:
    cik: int
    entity_name: str
    facts: tuple[Fact, ...]

    def of(self, concept: str, *, taxonomy: str = "us-gaap") -> tuple[Fact, ...]:
        """Every fact of ``concept``, in every unit, in EDGAR's order."""
        return tuple(
            fact for fact in self.facts if fact.concept == concept and fact.taxonomy == taxonomy
        )


def _number_only(value: object) -> object:
    # Lax Decimal would read "12" as 12; a reported value is a JSON number or nothing
    if isinstance(value, bool) or not isinstance(value, int | Decimal):
        raise ValueError(f"{value!r} is not a number")
    return value


ReportedValue = Annotated[Decimal, BeforeValidator(_number_only), Field(allow_inf_nan=False)]


class _RawFact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: date | None = None
    end: date
    val: ReportedValue
    accn: str
    fy: int | None = None
    fp: str | None = None
    form: str
    filed: date


class _RawConcept(BaseModel):
    model_config = ConfigDict(extra="ignore")

    units: dict[str, list[_RawFact]]


class _RawCompanyFacts(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cik: int
    entity_name: str = Field(alias="entityName")
    facts: dict[str, dict[str, _RawConcept]] = Field(default_factory=dict)


def parse_companyfacts(content: bytes) -> CompanyFacts:
    """Validate and type a ``CIK##########.json`` from the companyfacts API.

    Decimal values are kept exact; ``NaN`` and ``Infinity`` are refused.

    Raises:
        XbrlError: when the payload is not JSON or not the documented shape.
    """
    try:
        payload = json.loads(content, parse_float=Decimal, parse_constant=_refuse_constant)
        raw = _RawCompanyFacts.model_validate(payload)
    except (ValueError, ValidationError) as error:
        raise XbrlError(f"the company facts are malformed: {error}") from error
    facts = tuple(
        Fact(
            taxonomy=taxonomy,
            concept=concept,
            unit=unit,
            value=fact.val,
            start=fact.start,
            end=fact.end,
            fiscal_year=fact.fy,
            fiscal_period=fact.fp,
            form=fact.form,
            accession=fact.accn,
            filed=fact.filed,
        )
        for taxonomy, concepts in raw.facts.items()
        for concept, described in concepts.items()
        for unit, unit_facts in described.units.items()
        for fact in unit_facts
    )
    return CompanyFacts(cik=raw.cik, entity_name=raw.entity_name, facts=facts)


def period_end(report_date: date, years_before: int = 0) -> date:
    """The end of the fiscal year ``years_before`` years ahead of ``report_date``.

    Approximate by design: matching allows ``PERIOD_END_TOLERANCE_DAYS``.
    """
    if years_before < 0:
        raise ValueError("years_before counts back from the report date and cannot be negative")
    year = report_date.year - years_before
    # 29 February has no twin in most years; the tolerance absorbs the day
    day = min(report_date.day, 28) if report_date.month == 2 else report_date.day
    return report_date.replace(year=year, day=day)


def reported_fact(
    company: CompanyFacts,
    *,
    concept: str,
    unit: str,
    accession: str,
    report_date: date,
    years_before: int = 0,
    taxonomy: str = "us-gaap",
) -> Fact | None:
    """The value a pinned filing reported for one fiscal year, or ``None``.

    ``report_date`` is the filing's period end; ``years_before=0`` asks for the
    year the filing covers and ``1`` for the comparative year before it. Only
    facts from ``accession`` count, so a later restatement never replaces what
    this filing said. A balance must be an instant; a flow must span a year.

    Raises:
        XbrlError: when the filing reported two different values for that period.
    """
    target = period_end(report_date, years_before)
    tolerance = timedelta(days=PERIOD_END_TOLERANCE_DAYS)
    matches = [
        fact
        for fact in company.of(concept, taxonomy=taxonomy)
        if fact.unit == unit
        and fact.accession == accession
        and abs(fact.end - target) <= tolerance
        and (fact.is_instant or fact.is_annual)
    ]
    if not matches:
        return None
    values = {fact.value for fact in matches}
    if len(values) > 1:
        raise XbrlError(
            f"{accession} reports {len(values)} values of {taxonomy}:{concept} "
            f"for the period ending near {target}: {sorted(values)}"
        )
    return matches[0]


def _refuse_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not a reported value")
