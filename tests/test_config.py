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
    monkeypatch.setenv("EDGAR_RAG_BRIER_URL", "http://brier.test:8100")
    monkeypatch.setenv("INDEX_DIR", "/ignored")

    settings = ServiceSettings()

    assert settings.index_dir == Path("/srv/index")
    assert str(settings.brier_url) == "http://brier.test:8100/"


def test_an_empty_brier_url_is_not_accepted_as_a_sentinel():
    """``None`` means no brier; an empty string used to mean the same, silently."""
    with pytest.raises(ValidationError):
        ServiceSettings(brier_url="")


def test_a_brier_url_that_is_not_a_url_is_refused_at_startup():
    with pytest.raises(ValidationError):
        ServiceSettings(brier_url="localhost:8100")


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


def test_evaluation_reads_the_cosine_threshold():
    assert EvalSettings(min_retrieval_score=0.4).min_retrieval_score == 0.4
