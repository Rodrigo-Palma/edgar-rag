import tomllib
from pathlib import Path

from edgar_rag import __version__
from edgar_rag.api import app

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_service_reports_the_version_declared_in_pyproject():
    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]

    assert __version__ == declared
    assert app.version == declared
