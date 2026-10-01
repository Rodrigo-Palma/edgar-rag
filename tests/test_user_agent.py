"""The User-Agent the SEC requires: a real sender with a reachable address (M4)."""

import pytest
from pydantic import ValidationError

from edgar_rag.config import IngestSettings
from edgar_rag.edgar.user_agent import validate_user_agent

DECLARED = "Jane Analyst jane.analyst@ledgerworks.io"


def test_a_name_and_an_email_is_accepted_and_trimmed():
    assert validate_user_agent(f"  {DECLARED}  ") == DECLARED


@pytest.mark.parametrize(
    "user_agent",
    [
        "Your Name your.email@example.com",  # the .env.example placeholder
        "abcde",
        "Jane Analyst",
        "jane.analyst@ledgerworks.io",  # an address with no sender name
        "Jane jane@example.org",
        "Jane jane@example.net",
        "Jane jane@mail.example.com",
        "Jane jane@ledgerworks.test",
        "Jane jane@ledgerworks.invalid",
        "Jane jane@localhost",
        "Jane your.email@ledgerworks.io",
        "Your Name jane@ledgerworks.io",
        "Jane jane@ledgerworks.io\r\nX-Injected: 1",
        "Jané jane@ledgerworks.io",
        "Jane " + "a" * 200 + "@ledgerworks.io",
    ],
)
def test_a_placeholder_or_undeclared_sender_is_rejected(user_agent):
    with pytest.raises(ValueError):
        validate_user_agent(user_agent)


@pytest.mark.parametrize("user_agent", ["Your Name your.email@example.com", "abcde"])
def test_settings_refuse_the_placeholder_user_agent(user_agent):
    # Adversarial case 20: the template copied as is must not reach EDGAR
    with pytest.raises(ValidationError, match="edgar_user_agent"):
        IngestSettings(edgar_user_agent=user_agent)


def test_settings_keep_a_declared_user_agent():
    assert IngestSettings(edgar_user_agent=DECLARED).edgar_user_agent == DECLARED
