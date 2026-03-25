"""Tests for ingestion.utils.config."""

from unittest.mock import patch

import pytest

from ingestion.utils.config import (
    get_accepted_filetypes,
    get_api_key,
    get_auth_mode,
    get_data_source_path,
    get_database_config,
    get_database_schema,
    get_dense_table_description_max_prompt_tokens,
    get_enrichment_max_retries,
    get_enrichment_retry_delay,
    get_finalization_embedding_batch_size,
    get_finalization_embedding_model,
    get_finalization_max_classification_pages,
    get_finalization_max_retries,
    get_finalization_retry_delay,
    get_llm_endpoint,
    get_max_workers,
    get_oauth_config,
    get_retention_count,
    get_stage_model_config,
    get_storage_master_dir,
    get_storage_push_to_postgres,
    get_vision_dpi_scale,
    get_xlsx_classification_max_retries,
    get_xlsx_classification_retry_delay,
    get_xlsx_sheet_token_limit,
    load_config,
)


def test_load_config_calls_load_dotenv():
    """load_config delegates to load_dotenv."""
    with patch("ingestion.utils.config.load_dotenv") as mock:
        load_config()
        mock.assert_called_once()


def test_get_auth_mode_missing(monkeypatch):
    """Raises when not set."""
    monkeypatch.delenv("AUTH_MODE", raising=False)
    with pytest.raises(ValueError, match="AUTH_MODE"):
        get_auth_mode()


def test_get_auth_mode_oauth(monkeypatch):
    """Accepts oauth."""
    monkeypatch.setenv("AUTH_MODE", "oauth")
    assert get_auth_mode() == "oauth"


def test_get_auth_mode_api_key(monkeypatch):
    """Accepts api_key."""
    monkeypatch.setenv("AUTH_MODE", "api_key")
    assert get_auth_mode() == "api_key"


def test_get_auth_mode_case_insensitive(monkeypatch):
    """Handles uppercase."""
    monkeypatch.setenv("AUTH_MODE", "OAuth")
    assert get_auth_mode() == "oauth"


def test_get_auth_mode_invalid(monkeypatch):
    """Raises on invalid value."""
    monkeypatch.setenv("AUTH_MODE", "bad")
    with pytest.raises(ValueError, match="AUTH_MODE"):
        get_auth_mode()


def test_get_oauth_config_complete(monkeypatch):
    """Returns config when all fields are set."""
    monkeypatch.setenv("OAUTH_TOKEN_ENDPOINT", "https://auth/token")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("OAUTH_SCOPE", "read")
    config = get_oauth_config()
    assert config["token_endpoint"] == "https://auth/token"
    assert config["client_id"] == "id"
    assert config["client_secret"] == "secret"
    assert config["scope"] == "read"


def test_get_oauth_config_missing_fields(monkeypatch):
    """Raises when required fields are missing."""
    monkeypatch.delenv("OAUTH_TOKEN_ENDPOINT", raising=False)
    monkeypatch.delenv("OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("OAUTH_CLIENT_SECRET", raising=False)
    with pytest.raises(ValueError, match="OAuth requires"):
        get_oauth_config()


def test_get_oauth_config_optional_scope(monkeypatch):
    """Scope defaults to empty string."""
    monkeypatch.setenv("OAUTH_TOKEN_ENDPOINT", "https://auth/token")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "secret")
    monkeypatch.delenv("OAUTH_SCOPE", raising=False)
    config = get_oauth_config()
    assert config["scope"] == ""


def test_get_api_key(monkeypatch):
    """Returns the key when set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_api_key() == "sk-test"


def test_get_api_key_missing(monkeypatch):
    """Raises when key is not set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key()


def test_get_llm_endpoint(monkeypatch):
    """Returns endpoint when set."""
    monkeypatch.setenv("LLM_ENDPOINT", "https://custom/v1")
    assert get_llm_endpoint() == "https://custom/v1"


def test_get_llm_endpoint_missing(monkeypatch):
    """Raises when not set."""
    monkeypatch.delenv("LLM_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="LLM_ENDPOINT"):
        get_llm_endpoint()


def test_get_data_source_path(monkeypatch, tmp_path):
    """Returns path when it exists as a directory."""
    monkeypatch.setenv("DATA_SOURCE_PATH", str(tmp_path))
    assert get_data_source_path() == str(tmp_path)


def test_get_data_source_path_missing(monkeypatch):
    """Raises when not set."""
    monkeypatch.delenv("DATA_SOURCE_PATH", raising=False)
    with pytest.raises(ValueError, match="DATA_SOURCE_PATH"):
        get_data_source_path()


def test_get_data_source_path_not_a_directory(monkeypatch, tmp_path):
    """Raises when path is not a directory."""
    fake = str(tmp_path / "nonexistent")
    monkeypatch.setenv("DATA_SOURCE_PATH", fake)
    with pytest.raises(ValueError, match="not a directory"):
        get_data_source_path()


def test_get_database_config(monkeypatch):
    """Returns config dict when all required fields are set."""
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "testdb")
    monkeypatch.setenv("DB_USER", "dev")
    monkeypatch.setenv("DB_PASSWORD", "secret")
    config = get_database_config()
    assert config["host"] == "localhost"
    assert config["port"] == "5432"
    assert config["dbname"] == "testdb"
    assert config["user"] == "dev"
    assert config["password"] == "secret"


def test_get_database_config_missing_host(monkeypatch):
    """Raises when DB_HOST is not set."""
    monkeypatch.delenv("DB_HOST", raising=False)
    with pytest.raises(ValueError, match="DB_HOST"):
        get_database_config()


def test_get_database_config_optional_password(monkeypatch):
    """Password defaults to empty string."""
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "testdb")
    monkeypatch.setenv("DB_USER", "dev")
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    config = get_database_config()
    assert config["password"] == ""


def test_get_database_schema_defaults_public(monkeypatch):
    """Schema defaults to public when unset."""
    monkeypatch.delenv("DB_SCHEMA", raising=False)
    assert get_database_schema() == "public"


def test_get_database_schema_rejects_invalid_name(monkeypatch):
    """Schema names are restricted to safe SQL identifiers."""
    monkeypatch.setenv("DB_SCHEMA", "bad-name")
    with pytest.raises(ValueError, match="DB_SCHEMA"):
        get_database_schema()


def test_get_accepted_filetypes(monkeypatch):
    """Parses comma-separated list from env."""
    monkeypatch.setenv("ACCEPTED_FILETYPES", "pdf,docx,csv")
    result = get_accepted_filetypes()
    assert result == frozenset({"pdf", "docx", "csv"})


def test_get_accepted_filetypes_missing(monkeypatch):
    """Raises when not set."""
    monkeypatch.delenv("ACCEPTED_FILETYPES", raising=False)
    with pytest.raises(ValueError, match="ACCEPTED_FILETYPES"):
        get_accepted_filetypes()


def test_get_max_workers(monkeypatch):
    """Reads worker count from env."""
    monkeypatch.setenv("MAX_WORKERS", "8")
    assert get_max_workers() == 8


def test_get_max_workers_missing(monkeypatch):
    """Raises when not set."""
    monkeypatch.delenv("MAX_WORKERS", raising=False)
    with pytest.raises(ValueError, match="MAX_WORKERS"):
        get_max_workers()


def test_get_retention_count(monkeypatch):
    """Reads retention count from env."""
    monkeypatch.setenv("RETENTION_COUNT", "7")
    assert get_retention_count() == 7


def test_get_retention_count_missing(monkeypatch):
    """Raises when not set."""
    monkeypatch.delenv("RETENTION_COUNT", raising=False)
    with pytest.raises(ValueError, match="RETENTION_COUNT"):
        get_retention_count()


def test_get_storage_master_dir_defaults_under_project(monkeypatch):
    """Storage masters default under the project storage directory."""
    monkeypatch.delenv("STORAGE_MASTER_DIR", raising=False)
    assert get_storage_master_dir().endswith("/storage/masters")


def test_get_storage_master_dir_uses_env(monkeypatch):
    """Custom storage master dir is returned as-is."""
    monkeypatch.setenv("STORAGE_MASTER_DIR", "/tmp/storage-masters")
    assert get_storage_master_dir() == "/tmp/storage-masters"


def test_get_storage_push_to_postgres_false_by_default(monkeypatch):
    """Storage push defaults to disabled."""
    monkeypatch.delenv("STORAGE_PUSH_TO_POSTGRES", raising=False)
    assert get_storage_push_to_postgres() is False


def test_get_storage_push_to_postgres_accepts_true(monkeypatch):
    """Boolean-like true values enable Postgres sync."""
    monkeypatch.setenv("STORAGE_PUSH_TO_POSTGRES", "yes")
    assert get_storage_push_to_postgres() is True


def test_get_storage_push_to_postgres_rejects_invalid(monkeypatch):
    """Invalid boolean-like values raise."""
    monkeypatch.setenv("STORAGE_PUSH_TO_POSTGRES", "sometimes")
    with pytest.raises(ValueError, match="STORAGE_PUSH_TO_POSTGRES"):
        get_storage_push_to_postgres()


def test_get_stage_model_config_complete(monkeypatch):
    """Returns config when all fields are set."""
    monkeypatch.setenv("CLASSIFICATION_MODEL", "gpt-5")
    monkeypatch.setenv("CLASSIFICATION_MAX_TOKENS", "500")
    monkeypatch.setenv("CLASSIFICATION_TEMPERATURE", "0.7")
    config = get_stage_model_config("classification")
    assert config["model"] == "gpt-5"
    assert config["max_tokens"] == 500
    assert config["temperature"] == 0.7


def test_get_stage_model_config_missing_model(monkeypatch):
    """Raises when model is not set."""
    monkeypatch.delenv("STARTUP_MODEL", raising=False)
    monkeypatch.setenv("STARTUP_MAX_TOKENS", "50")
    monkeypatch.setenv("STARTUP_TEMPERATURE", "0")
    with pytest.raises(ValueError, match="STARTUP_MODEL"):
        get_stage_model_config("startup")


def test_get_stage_model_config_missing_max_tokens(
    monkeypatch,
):
    """Raises when max_tokens is not set."""
    monkeypatch.setenv("STARTUP_MODEL", "gpt-4")
    monkeypatch.delenv("STARTUP_MAX_TOKENS", raising=False)
    monkeypatch.setenv("STARTUP_TEMPERATURE", "0")
    with pytest.raises(ValueError, match="STARTUP_MAX_TOKENS"):
        get_stage_model_config("startup")


def test_get_stage_model_config_no_temperature(
    monkeypatch,
):
    """Temperature is None when not set."""
    monkeypatch.setenv("STARTUP_MODEL", "gpt-4")
    monkeypatch.setenv("STARTUP_MAX_TOKENS", "50")
    monkeypatch.delenv("STARTUP_TEMPERATURE", raising=False)
    config = get_stage_model_config("startup")
    assert config["temperature"] is None


def test_get_stage_model_config_case_insensitive(
    monkeypatch,
):
    """Stage name is uppercased."""
    monkeypatch.setenv("EXTRACTION_MODEL", "gpt-4-vision")
    monkeypatch.setenv("EXTRACTION_MAX_TOKENS", "4000")
    monkeypatch.setenv("EXTRACTION_TEMPERATURE", "0")
    config = get_stage_model_config("extraction")
    assert config["model"] == "gpt-4-vision"


def test_get_vision_dpi_scale(monkeypatch):
    """Reads DPI scale from env."""
    monkeypatch.setenv("VISION_DPI_SCALE", "3.0")
    assert get_vision_dpi_scale() == 3.0


def test_get_vision_dpi_scale_missing(monkeypatch):
    """Raises when not set."""
    monkeypatch.delenv("VISION_DPI_SCALE", raising=False)
    with pytest.raises(ValueError, match="VISION_DPI_SCALE"):
        get_vision_dpi_scale()


def test_get_xlsx_sheet_token_limit(monkeypatch):
    """Reads inline XLSX sheet token limit from env."""
    monkeypatch.setenv("XLSX_SHEET_TOKEN_LIMIT", "12000")
    assert get_xlsx_sheet_token_limit() == 12000


def test_get_xlsx_sheet_token_limit_default(monkeypatch):
    """Uses the default inline XLSX sheet token limit."""
    monkeypatch.delenv("XLSX_SHEET_TOKEN_LIMIT", raising=False)
    assert get_xlsx_sheet_token_limit() == 50000


def test_get_xlsx_classification_max_retries_default(monkeypatch):
    """Uses the default XLSX classification retry count."""
    monkeypatch.delenv("XLSX_CLASSIFICATION_MAX_RETRIES", raising=False)
    assert get_xlsx_classification_max_retries() == 3


def test_get_xlsx_classification_max_retries_override(monkeypatch):
    """Reads XLSX classification retry count from env."""
    monkeypatch.setenv("XLSX_CLASSIFICATION_MAX_RETRIES", "5")
    assert get_xlsx_classification_max_retries() == 5


def test_get_xlsx_classification_retry_delay_default(monkeypatch):
    """Uses the default XLSX classification retry delay."""
    monkeypatch.delenv(
        "XLSX_CLASSIFICATION_RETRY_DELAY_SECONDS", raising=False
    )
    assert get_xlsx_classification_retry_delay() == 2.0


def test_get_xlsx_classification_retry_delay_override(monkeypatch):
    """Reads XLSX classification retry delay from env."""
    monkeypatch.setenv("XLSX_CLASSIFICATION_RETRY_DELAY_SECONDS", "1.5")
    assert get_xlsx_classification_retry_delay() == 1.5


def test_get_dense_table_description_max_prompt_tokens_default(
    monkeypatch,
):
    """Uses the default dense table prompt budget when unset."""
    monkeypatch.delenv(
        "DENSE_TABLE_DESCRIPTION_MAX_PROMPT_TOKENS",
        raising=False,
    )
    assert get_dense_table_description_max_prompt_tokens() == 12000


def test_get_dense_table_description_max_prompt_tokens_override(
    monkeypatch,
):
    """Reads the dense table prompt budget override from env."""
    monkeypatch.setenv("DENSE_TABLE_DESCRIPTION_MAX_PROMPT_TOKENS", "6400")
    assert get_dense_table_description_max_prompt_tokens() == 6400


def test_get_enrichment_max_retries_default(monkeypatch):
    """Uses the default enrichment retry count."""
    monkeypatch.delenv("ENRICHMENT_MAX_RETRIES", raising=False)
    assert get_enrichment_max_retries() == 3


def test_get_enrichment_max_retries_override(monkeypatch):
    """Reads enrichment retry count from env."""
    monkeypatch.setenv("ENRICHMENT_MAX_RETRIES", "5")
    assert get_enrichment_max_retries() == 5


def test_get_enrichment_retry_delay_default(monkeypatch):
    """Uses the default enrichment retry delay."""
    monkeypatch.delenv("ENRICHMENT_RETRY_DELAY_SECONDS", raising=False)
    assert get_enrichment_retry_delay() == 2.0


def test_get_enrichment_retry_delay_override(monkeypatch):
    """Reads enrichment retry delay from env."""
    monkeypatch.setenv("ENRICHMENT_RETRY_DELAY_SECONDS", "1.5")
    assert get_enrichment_retry_delay() == 1.5


def test_get_finalization_max_retries_default(monkeypatch):
    """Uses the default finalization retry count."""
    monkeypatch.delenv("FINALIZATION_MAX_RETRIES", raising=False)
    assert get_finalization_max_retries() == 3


def test_get_finalization_max_retries_override(monkeypatch):
    """Reads finalization retry count from env."""
    monkeypatch.setenv("FINALIZATION_MAX_RETRIES", "4")
    assert get_finalization_max_retries() == 4


def test_get_finalization_retry_delay_default(monkeypatch):
    """Uses the default finalization retry delay."""
    monkeypatch.delenv("FINALIZATION_RETRY_DELAY_SECONDS", raising=False)
    assert get_finalization_retry_delay() == 2.0


def test_get_finalization_retry_delay_override(monkeypatch):
    """Reads finalization retry delay from env."""
    monkeypatch.setenv("FINALIZATION_RETRY_DELAY_SECONDS", "1.25")
    assert get_finalization_retry_delay() == 1.25


def test_get_finalization_embedding_batch_size_default(monkeypatch):
    """Uses the default finalization embedding batch size."""
    monkeypatch.delenv("FINALIZATION_EMBEDDING_BATCH_SIZE", raising=False)
    assert get_finalization_embedding_batch_size() == 100


def test_get_finalization_embedding_batch_size_override(monkeypatch):
    """Reads finalization embedding batch size from env."""
    monkeypatch.setenv("FINALIZATION_EMBEDDING_BATCH_SIZE", "25")
    assert get_finalization_embedding_batch_size() == 25


def test_get_finalization_embedding_model_default(monkeypatch):
    """Uses the default finalization embedding model."""
    monkeypatch.delenv("FINALIZATION_EMBEDDING_MODEL", raising=False)
    assert get_finalization_embedding_model() == "text-embedding-3-large"


def test_get_finalization_embedding_model_override(monkeypatch):
    """Reads finalization embedding model from env."""
    monkeypatch.setenv(
        "FINALIZATION_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    assert get_finalization_embedding_model() == "text-embedding-3-small"


def test_get_finalization_max_classification_pages_default(monkeypatch):
    """Uses the default finalization classification page cap."""
    monkeypatch.delenv(
        "FINALIZATION_MAX_CLASSIFICATION_PAGES",
        raising=False,
    )
    assert get_finalization_max_classification_pages() == 100


def test_get_finalization_max_classification_pages_override(monkeypatch):
    """Reads finalization classification page cap from env."""
    monkeypatch.setenv("FINALIZATION_MAX_CLASSIFICATION_PAGES", "75")
    assert get_finalization_max_classification_pages() == 75
