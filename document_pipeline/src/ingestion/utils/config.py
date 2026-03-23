"""Pipeline configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"


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

    Reads XLSX_SHEET_TOKEN_LIMIT env var.

    Returns:
        int — max estimated content tokens before a sheet
        likely needs dense-table extraction

    Example:
        >>> get_xlsx_sheet_token_limit()
        12000
    """
    return int(_require_env("XLSX_SHEET_TOKEN_LIMIT"))


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
