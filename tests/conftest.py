from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings, get_settings, load_settings
from app.main import create_app

SAMPLE_DOCS = PROJECT_ROOT / "data" / "documents"


@pytest.fixture(autouse=True)
def isolated_data_dirs(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep every test away from the real ./data folders.

    `app.deps.build_components` builds the vector store from the global settings, not from the
    Settings a test passes to `create_app`. Without this, upload tests would index their files
    (notes.md, escape.md ...) into your real data/vectorstore.
    """
    root = tmp_path_factory.mktemp("isolated-data")
    for name, folder in (("VECTOR_DIR", "vectorstore"), ("SESSIONS_DIR", "sessions"), ("DOCUMENTS_DIR", "documents")):
        monkeypatch.setenv(name, str(root / folder))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings pointing every folder at a fresh temp dir."""
    return replace(
        load_settings(),
        data_dir=tmp_path,
        sessions_dir=tmp_path / "sessions",
        documents_dir=tmp_path / "documents",
        vector_dir=tmp_path / "vectorstore",
    )


@pytest.fixture()
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
