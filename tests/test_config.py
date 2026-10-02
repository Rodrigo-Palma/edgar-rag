from pathlib import Path

import pytest
from pydantic import ValidationError

from edgar_rag.config import IngestSettings, ServiceSettings


@pytest.fixture(autouse=True)
def no_env_file(monkeypatch, tmp_path):
    """A developer's own .env must not decide what these tests see."""
    monkeypatch.chdir(tmp_path)


def test_the_service_starts_without_an_edgar_user_agent():
    """The service never calls EDGAR, so it must not demand the SEC contact."""
    settings = ServiceSettings()

    assert settings.index_dir == Path("data/index")


def test_the_service_binds_to_the_local_machine_by_default():
    assert ServiceSettings().host == "127.0.0.1"


def test_the_service_limits_generations_and_request_time_by_default():
    settings = ServiceSettings()

    assert settings.max_concurrent_generations == 2
    assert settings.request_timeout_seconds == 90.0


def test_every_variable_carries_the_project_prefix(monkeypatch):
    monkeypatch.setenv("EDGAR_RAG_INDEX_DIR", "/srv/index")
    monkeypatch.setenv("EDGAR_RAG_GATE", "cosine")
    monkeypatch.setenv("INDEX_DIR", "/ignored")

    settings = ServiceSettings()

    assert settings.index_dir == Path("/srv/index")
    assert settings.gate == "cosine"


def test_the_default_gate_is_the_period_guard_in_front_of_cosine():
    """The default needs nothing but the index, so anyone can run it."""
    assert ServiceSettings().gate == "period+cosine"


@pytest.mark.parametrize(
    "gate", ["", "period", "cosine+period", "PERIOD+COSINE", "brier", "period+brier"]
)
def test_a_gate_that_is_not_one_of_the_three_is_refused_at_startup(gate):
    """Brier's gates were removed (ADR-0014): a setting naming one stops the service."""
    with pytest.raises(ValidationError):
        ServiceSettings(gate=gate)


@pytest.mark.parametrize("limit", [0, -1])
def test_at_least_one_generation_must_be_allowed(limit):
    with pytest.raises(ValidationError):
        ServiceSettings(max_concurrent_generations=limit)


def test_ingestion_still_requires_the_sec_contact():
    with pytest.raises(ValidationError):
        IngestSettings()


def test_ingestion_reads_the_contact_from_its_prefixed_variable(monkeypatch):
    monkeypatch.setenv("EDGAR_RAG_EDGAR_USER_AGENT", "Jane Doe jane@firm.io")

    assert IngestSettings().edgar_user_agent == "Jane Doe jane@firm.io"
