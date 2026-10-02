"""Question wording: one slot per varying thing, no leak of the relevance model's template."""

import re

from edgar_rag.eval.templates import (
    OFF_DOMAIN,
    POSITIVE_CONCEPTS,
    SECTOR_PHRASINGS,
    fill,
    possessive,
)

ALL_PHRASINGS = (
    [p for concept in POSITIVE_CONCEPTS for p in concept.phrasings]
    + list(SECTOR_PHRASINGS)
    + [question.text for question in OFF_DOMAIN]
)


def test_every_positive_concept_has_four_distinct_phrasings_with_both_slots():
    assert len(POSITIVE_CONCEPTS) == 12
    for concept in POSITIVE_CONCEPTS:
        assert len(set(concept.phrasings)) == 4
        assert all("{company}" in p and "{year}" in p for p in concept.phrasings)


def test_no_phrasing_writes_a_year_or_the_relevance_model_template():
    for phrasing in ALL_PHRASINGS:
        assert not re.search(r"\b(19|20)\d{2}\b", phrasing), phrasing
        assert "does the passage answer" not in phrasing.lower()


def test_there_are_forty_off_domain_questions_half_naming_the_company():
    assert len(OFF_DOMAIN) == 40
    assert sum(question.names_company for question in OFF_DOMAIN) == 20
    assert len({question.key for question in OFF_DOMAIN}) == 40


def test_the_possessive_does_not_double_an_existing_one():
    assert possessive("Apple") == "Apple's"
    assert possessive("McDonald's") == "McDonald's"
    assert fill("What was {company}'s revenue in {year}?", company="McDonald's", year=2024) == (
        "What was McDonald's revenue in 2024?"
    )
    assert fill("Tell me a joke about {company}.", company="Visa") == "Tell me a joke about Visa."
    assert (
        fill("{company}'s {label}?", company="Visa", label="net premiums") == "Visa's net premiums?"
    )
