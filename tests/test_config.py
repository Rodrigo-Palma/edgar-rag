from pathlib import Path

import pytest
from pydantic import ValidationError

from edgar_rag.config import EvalSettings, IngestSettings, ServiceSettings


@pytest.fixture(autouse=True)
def no_env_file(monkeypatch, tmp_path):
    """A developer's own .env must not decide what these tests see."""
    monkeypatch.chdir(tmp_path)


def test_the_service_starts_without_an_edgar_user_agent():
    """The service never calls EDGAR, so it must not demand the SEC contact."""
    settings = ServiceSettings()

    assert settings.index_dir == Path("data/index")
    assert settings.brier_url is None


def test_the_service_binds_to_the_local_machine_by_default():
    assert ServiceSettings().host == "127.0.0.1"


def test_the_service_limits_generations_and_request_time_by_default():
    settings = ServiceSettings()

    assert settings.max_concurrent_generations == 2
    assert settings.request_timeout_seconds == 90.0


def test_every_variable_carries_the_project_prefix(monkeypatch):
    monkeypatch.setenv("EDGAR_RAG_INDEX_DIR", "/srv/index")
    monkeypatch.setenv("EDGAR_RAG_GATE", "period+brier")
    monkeypatch.setenv("EDGAR_RAG_BRIER_URL", "http://brier.test:8100")
    monkeypatch.setenv("INDEX_DIR", "/ignored")

    settings = ServiceSettings()

    assert settings.index_dir == Path("/srv/index")
    assert settings.gate == "period+brier"
    assert str(settings.brier_url) == "http://brier.test:8100/"


def test_an_empty_brier_url_is_not_accepted_as_a_sentinel():
    """``None`` means no brier; an empty string used to mean the same, silently."""
    with pytest.raises(ValidationError):
        ServiceSettings(gate="brier", brier_url="")


def test_a_brier_url_that_is_not_a_url_is_refused_at_startup():
    with pytest.raises(ValidationError):
        ServiceSettings(gate="brier", brier_url="localhost:8100")


def test_the_default_gate_is_the_period_guard_in_front_of_cosine():
    """The default needs nothing but the index, so anyone can run it."""
    assert ServiceSettings().gate == "period+cosine"


@pytest.mark.parametrize("gate", ["brier", "period+brier"])
def test_a_brier_gate_without_a_brier_url_is_refused_at_startup(gate):
    with pytest.raises(ValidationError, match="needs EDGAR_RAG_BRIER_URL"):
        ServiceSettings(gate=gate)


@pytest.mark.parametrize("gate", ["none", "cosine", "period+cosine"])
def test_a_brier_url_the_gate_would_ignore_is_refused_at_startup(gate):
    """Setting the URL used to switch brier on; ignoring it now would be silent."""
    with pytest.raises(ValidationError, match="does not use it"):
        ServiceSettings(gate=gate, brier_url="http://brier.test")


@pytest.mark.parametrize("gate", ["", "period", "cosine+period", "PERIOD+COSINE", "brier+period"])
def test_a_gate_that_is_not_one_of_the_five_is_refused_at_startup(gate):
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


def test_evaluation_runs_without_brier_unless_its_url_is_set(monkeypatch):
    monkeypatch.delenv("EDGAR_RAG_BRIER_URL", raising=False)
    assert EvalSettings().brier_url is None

    monkeypatch.setenv("EDGAR_RAG_BRIER_URL", "http://127.0.0.1:8100")
    monkeypatch.setenv("EDGAR_RAG_BRIER_SHA", "d70e7df")
    settings = EvalSettings()

    assert str(settings.brier_url) == "http://127.0.0.1:8100/"
    assert settings.brier_sha == "d70e7df"
