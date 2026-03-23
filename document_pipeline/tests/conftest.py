"""Global test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _set_pipeline_env(monkeypatch):
    """Ensure pipeline env vars are set for tests."""
    monkeypatch.setenv("ACCEPTED_FILETYPES", "pdf,docx,pptx,xlsx,csv,md")
    monkeypatch.setenv("MAX_WORKERS", "4")
    monkeypatch.setenv("RETENTION_COUNT", "31")
    monkeypatch.setenv("EXTRACTION_MODEL", "gpt-5-mini")
    monkeypatch.setenv("EXTRACTION_MAX_TOKENS", "16000")
    monkeypatch.setenv("EXTRACTION_TEMPERATURE", "")
    monkeypatch.setenv("VISION_DPI_SCALE", "2.0")
    monkeypatch.setenv("DENSE_TABLE_DESCRIPTION_MODEL", "gpt-5-mini")
    monkeypatch.setenv("DENSE_TABLE_DESCRIPTION_MAX_TOKENS", "4000")
    monkeypatch.setenv("DENSE_TABLE_DESCRIPTION_TEMPERATURE", "")
    monkeypatch.setenv("CONTENT_CHUNKING_MODEL", "gpt-5-mini")
    monkeypatch.setenv("CONTENT_CHUNKING_MAX_TOKENS", "4000")
    monkeypatch.setenv("CONTENT_CHUNKING_TEMPERATURE", "")
    monkeypatch.setenv("CONTENT_PREP_MAX_CHUNK_TOKENS", "8191")
