"""Pipeline configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"
PROJECT_ROOT = ENV_PATH.parent


def load_config() -> None:
    """Load .env file into process environment.

    Safe to call multiple times — load_dotenv does not
    overwrite existing env vars by default.

    Params:
        None

    Returns:
        None

    Example:
        >>> load_config()
    """
    load_dotenv(ENV_PATH)


def get_auth_mode() -> str:
    """Get the authentication mode from AUTH_MODE env var.

    Params:
        None

    Returns:
        str — "oauth" or "api_key"

    Example:
        >>> get_auth_mode()
        "api_key"
    """
    mode = os.getenv("AUTH_MODE", "")
    if not mode:
        raise ValueError("AUTH_MODE is required")
    mode = mode.lower()
    if mode not in ("oauth", "api_key"):
        raise ValueError(
            f"AUTH_MODE must be 'oauth' or 'api_key', " f"got '{mode}'"
        )
    return mode


def get_oauth_config() -> dict:
    """Get OAuth configuration from environment variables.

    All fields are required when AUTH_MODE=oauth.

    Params:
        None

    Returns:
        dict with keys: token_endpoint, client_id,
        client_secret, scope (optional)

    Example:
        >>> cfg = get_oauth_config()
        >>> cfg["token_endpoint"]
        "https://auth.example.com/token"
    """
    config = {
        "token_endpoint": os.getenv("OAUTH_TOKEN_ENDPOINT", ""),
        "client_id": os.getenv("OAUTH_CLIENT_ID", ""),
        "client_secret": os.getenv("OAUTH_CLIENT_SECRET", ""),
        "scope": os.getenv("OAUTH_SCOPE", ""),
    }
    missing = [
        key
        for key in (
            "token_endpoint",
            "client_id",
            "client_secret",
        )
        if not config[key]
    ]
    if missing:
        raise ValueError(f"OAuth requires: {', '.join(missing)}")
    return config


def get_api_key() -> str:
    """Get the API key from OPENAI_API_KEY env var.

    Params:
        None

    Returns:
        str — the API key

    Example:
        >>> get_api_key()
        "sk-..."
    """
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY is required")
    return key


def get_llm_endpoint() -> str:
    """Get the LLM API base URL. Returns: str."""
    value = os.getenv("LLM_ENDPOINT", "")
    if not value:
        raise ValueError("LLM_ENDPOINT is required")
    return value


def get_accepted_filetypes() -> frozenset:
    """Get the set of accepted file extensions.

    Reads ACCEPTED_FILETYPES env var as a comma-separated list.

    Returns:
        frozenset of lowercase extension strings

    Example:
        >>> get_accepted_filetypes()
        frozenset({'pdf', 'docx', ...})
    """
    raw = _require_env("ACCEPTED_FILETYPES")
    return frozenset(
        ext.strip().lower() for ext in raw.split(",") if ext.strip()
    )


def get_max_workers() -> int:
    """Get the max worker threads for parallel file processing.

    Reads MAX_WORKERS env var.

    Returns:
        int — number of worker threads

    Example:
        >>> get_max_workers()
        4
    """
    return int(_require_env("MAX_WORKERS"))


def get_retention_count() -> int:
    """Get the retention count for logs and archives.

    Reads RETENTION_COUNT env var.

    Returns:
        int — number of files to keep

    Example:
        >>> get_retention_count()
        31
    """
    return int(_require_env("RETENTION_COUNT"))


def _require_env(name: str) -> str:
    """Get required env var or raise. Params: name. Returns: str."""
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"{name} is required")
    return value


def get_vision_dpi_scale() -> float:
    """Get the DPI scale factor for vision page rendering.

    Reads VISION_DPI_SCALE env var. Higher values produce
    sharper images but larger payloads.

    Returns:
        float — DPI multiplier (e.g. 2.0)

    Example:
        >>> get_vision_dpi_scale()
        2.0
    """
    return float(_require_env("VISION_DPI_SCALE"))


def get_xlsx_sheet_token_limit() -> int:
    """Get the inline token limit for a serialized XLSX sheet.

    Reads XLSX_SHEET_TOKEN_LIMIT env var and defaults to
    50000 when unset.

    Returns:
        int — max estimated content tokens before a sheet
        likely needs dense-table extraction

    Example:
        >>> get_xlsx_sheet_token_limit()
        50000
    """
    return int(os.getenv("XLSX_SHEET_TOKEN_LIMIT", "50000"))


def get_xlsx_classification_max_retries() -> int:
    """Get the max retry attempts for XLSX sheet classification.

    Reads XLSX_CLASSIFICATION_MAX_RETRIES env var and
    defaults to 3 when unset.

    Returns:
        int — retry attempts before the workbook fails

    Example:
        >>> get_xlsx_classification_max_retries()
        3
    """
    return int(os.getenv("XLSX_CLASSIFICATION_MAX_RETRIES", "3"))


def get_content_prep_max_chunk_tokens() -> int:
    """Get the max token size for content chunks.

    Reads CONTENT_PREP_MAX_CHUNK_TOKENS env var. Chunks
    exceeding this limit are split via LLM breakpoint detection.

    Returns:
        int — max tokens per chunk (default matches
        text-embedding-3-large limit)

    Example:
        >>> get_content_prep_max_chunk_tokens()
        8191
    """
    return int(_require_env("CONTENT_PREP_MAX_CHUNK_TOKENS"))


def get_dense_table_description_max_prompt_tokens() -> int:
    """Get the max prompt budget for dense table description calls.

    Reads DENSE_TABLE_DESCRIPTION_MAX_PROMPT_TOKENS and defaults
    to 12000 when unset.

    Returns:
        int — approximate prompt token budget before the dense
        table description flow switches to batching or fallback

    Example:
        >>> get_dense_table_description_max_prompt_tokens()
        12000
    """
    return int(os.getenv("DENSE_TABLE_DESCRIPTION_MAX_PROMPT_TOKENS", "12000"))


def get_xlsx_classification_retry_delay() -> float:
    """Get the base backoff delay for XLSX classification retries.

    Reads XLSX_CLASSIFICATION_RETRY_DELAY_SECONDS env var
    and defaults to 2.0 when unset.

    Returns:
        float — seconds multiplied by the current attempt

    Example:
        >>> get_xlsx_classification_retry_delay()
        2.0
    """
    return float(os.getenv("XLSX_CLASSIFICATION_RETRY_DELAY_SECONDS", "2.0"))


def get_enrichment_max_retries() -> int:
    """Get max retries for enrichment page calls.

    Returns: int — default 3.
    """
    return int(os.getenv("ENRICHMENT_MAX_RETRIES", "3"))


def get_enrichment_retry_delay() -> float:
    """Get base backoff delay for enrichment retries.

    Returns: float — default 2.0.
    """
    return float(os.getenv("ENRICHMENT_RETRY_DELAY_SECONDS", "2.0"))


def get_finalization_max_retries() -> int:
    """Get max retries for finalization LLM and embedding calls.

    Returns: int — default 3.
    """
    return int(os.getenv("FINALIZATION_MAX_RETRIES", "3"))


def get_finalization_retry_delay() -> float:
    """Get base backoff delay for finalization retries.

    Returns: float — default 2.0.
    """
    return float(os.getenv("FINALIZATION_RETRY_DELAY_SECONDS", "2.0"))


def get_finalization_embedding_batch_size() -> int:
    """Get chunk embedding batch size for finalization.

    Returns: int — default 100.
    """
    return int(os.getenv("FINALIZATION_EMBEDDING_BATCH_SIZE", "100"))


def get_finalization_chunk_summary_batch_size() -> int:
    """Get chunk-summary LLM batch size for finalization.

    Returns: int — default 10.
    """
    return int(os.getenv("FINALIZATION_CHUNK_SUMMARY_BATCH_SIZE", "10"))


def get_finalization_metadata_page_count() -> int:
    """Get the page count sent to metadata extraction prompts.

    Returns: int — default 5.
    """
    return int(os.getenv("FINALIZATION_METADATA_PAGE_COUNT", "5"))


def get_finalization_embedding_model() -> str:
    """Get the embedding model used by finalization.

    Returns: str — default "text-embedding-3-large".
    """
    return os.getenv(
        "FINALIZATION_EMBEDDING_MODEL",
        "text-embedding-3-large",
    )


def get_finalization_max_classification_pages() -> int:
    """Get max pages sent to LLM section classification.

    Returns: int — default 100.
    """
    return int(os.getenv("FINALIZATION_MAX_CLASSIFICATION_PAGES", "100"))


def get_finalization_context_chain_depth() -> int:
    """Get max XLSX context-sheet chain depth.

    Returns: int — default 3.
    """
    return int(os.getenv("FINALIZATION_CONTEXT_CHAIN_DEPTH", "3"))


def get_finalization_degradation_signal_threshold() -> int:
    """Get the number of degradation signals that fail a file.

    Returns: int — default 3.
    """
    return int(os.getenv("FINALIZATION_DEGRADATION_SIGNAL_THRESHOLD", "3"))


def get_pdf_classification_max_retries() -> int:
    """Get max retries for PDF page continuation classification.

    Returns: int — default 5.
    """
    return int(os.getenv("PDF_CLASSIFICATION_MAX_RETRIES", "5"))


def get_pdf_classification_retry_delay() -> float:
    """Get base backoff delay for PDF classification retries.

    Returns: float — default 2.0.
    """
    return float(os.getenv("PDF_CLASSIFICATION_RETRY_DELAY_SECONDS", "2.0"))


def get_pdf_vision_max_retries() -> int:
    """Get max retries for PDF vision extraction calls.

    Returns: int — default 3.
    """
    return int(os.getenv("PDF_VISION_MAX_RETRIES", "3"))


def get_pdf_vision_retry_delay() -> float:
    """Get base backoff delay for PDF vision retries.

    Returns: float — default 2.0.
    """
    return float(os.getenv("PDF_VISION_RETRY_DELAY_SECONDS", "2.0"))


def get_docx_classification_max_retries() -> int:
    """Get max retries for DOCX page continuation classification.

    Returns: int — default 5.
    """
    return int(os.getenv("DOCX_CLASSIFICATION_MAX_RETRIES", "5"))


def get_docx_classification_retry_delay() -> float:
    """Get base backoff delay for DOCX classification retries.

    Returns: float — default 2.0.
    """
    return float(os.getenv("DOCX_CLASSIFICATION_RETRY_DELAY_SECONDS", "2.0"))


def get_docx_vision_max_retries() -> int:
    """Get max retries for DOCX vision extraction calls.

    Returns: int — default 3.
    """
    return int(os.getenv("DOCX_VISION_MAX_RETRIES", "3"))


def get_docx_vision_retry_delay() -> float:
    """Get base backoff delay for DOCX vision retries.

    Returns: float — default 2.0.
    """
    return float(os.getenv("DOCX_VISION_RETRY_DELAY_SECONDS", "2.0"))


def get_pptx_classification_max_retries() -> int:
    """Get max retries for PPTX slide classification.

    Returns: int — default 3.
    """
    return int(os.getenv("PPTX_CLASSIFICATION_MAX_RETRIES", "3"))


def get_pptx_classification_retry_delay() -> float:
    """Get base backoff delay for PPTX classification retries.

    Returns: float — default 2.0.
    """
    return float(os.getenv("PPTX_CLASSIFICATION_RETRY_DELAY_SECONDS", "2.0"))


def get_pptx_vision_max_retries() -> int:
    """Get max retries for PPTX vision extraction calls.

    Returns: int — default 3.
    """
    return int(os.getenv("PPTX_VISION_MAX_RETRIES", "3"))


def get_pptx_vision_retry_delay() -> float:
    """Get base backoff delay for PPTX vision retries.

    Returns: float — default 2.0.
    """
    return float(os.getenv("PPTX_VISION_RETRY_DELAY_SECONDS", "2.0"))


def get_xlsx_vision_max_retries() -> int:
    """Get max retries for XLSX vision extraction calls.

    Returns: int — default 3.
    """
    return int(os.getenv("XLSX_VISION_MAX_RETRIES", "3"))


def get_xlsx_vision_retry_delay() -> float:
    """Get base backoff delay for XLSX vision retries.

    Returns: float — default 2.0.
    """
    return float(os.getenv("XLSX_VISION_RETRY_DELAY_SECONDS", "2.0"))


def get_data_source_path() -> str:
    """Get the base path for data source folders.

    Reads DATA_SOURCE_PATH and validates it exists as a directory.

    Params:
        None

    Returns:
        str — absolute path to the data sources root

    Example:
        >>> get_data_source_path()
        "/data/sources"
    """
    path = _require_env("DATA_SOURCE_PATH")
    if not Path(path).is_dir():
        raise ValueError(f"DATA_SOURCE_PATH is not a directory: {path}")
    return path


def get_database_config() -> dict:
    """Get PostgreSQL connection parameters from environment.

    Reads DB_HOST, DB_PORT, DB_NAME, DB_USER, and
    optional DB_PASSWORD.

    Params:
        None

    Returns:
        dict with keys: host, port, dbname, user, password

    Example:
        >>> get_database_config()
        {"host": "localhost", "port": "5432", ...}
    """
    return {
        "host": _require_env("DB_HOST"),
        "port": _require_env("DB_PORT"),
        "dbname": _require_env("DB_NAME"),
        "user": _require_env("DB_USER"),
        "password": os.getenv("DB_PASSWORD", ""),
    }


def get_database_schema() -> str:
    """Get the PostgreSQL schema name. Returns: str."""
    schema = os.getenv("DB_SCHEMA", "public").strip()
    if not schema:
        raise ValueError("DB_SCHEMA cannot be empty")
    if not schema.replace("_", "").isalnum():
        raise ValueError(
            "DB_SCHEMA must contain only letters, numbers, and underscores"
        )
    if schema[0].isdigit():
        raise ValueError("DB_SCHEMA cannot start with a number")
    return schema


def get_storage_master_dir() -> str:
    """Get the directory where canonical storage CSVs are written."""
    return os.getenv(
        "STORAGE_MASTER_DIR",
        str(PROJECT_ROOT / "storage" / "masters"),
    )


def get_storage_push_to_postgres() -> bool:
    """Get whether storage masters should be pushed into PostgreSQL."""
    raw = os.getenv("STORAGE_PUSH_TO_POSTGRES", "false").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ValueError("STORAGE_PUSH_TO_POSTGRES must be a boolean-like value")


def get_stage_model_config(stage: str) -> dict:
    """Get model config for a pipeline stage.

    Reads {STAGE}_MODEL, {STAGE}_MAX_TOKENS, and
    {STAGE}_TEMPERATURE env vars. Stage name is
    uppercased automatically. MODEL and MAX_TOKENS are
    required. TEMPERATURE is optional — omit or leave
    blank for models that don't support it (e.g. o-series).

    Params:
        stage: Pipeline stage name
            (e.g. "startup", "classification")

    Returns:
        dict with keys: model, max_tokens,
        temperature (float or None)

    Example:
        >>> get_stage_model_config("startup")
        {"model": "gpt-4.1-mini", "max_tokens": 50, "temperature": 0.0}
    """
    prefix = stage.upper()
    temp_raw = os.getenv(f"{prefix}_TEMPERATURE", "")
    temperature = float(temp_raw) if temp_raw else None
    return {
        "model": _require_env(f"{prefix}_MODEL"),
        "max_tokens": int(_require_env(f"{prefix}_MAX_TOKENS")),
        "temperature": temperature,
    }
