"""
Environment Configuration - Centralized settings from environment variables.

Provides the single source of truth for all research pipeline configuration.
Reads from environment variables at import time and exposes them through
the Config class. Used by connections, agents, and monitoring.
"""

import logging
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class Config:
    """Application configuration loaded from environment variables at import time."""

    BASE_URL: str = os.getenv("AZURE_BASE_URL") or "https://api.openai.com/v1"

    DB_HOST: str = os.getenv("DB_HOST", "")
    DB_PORT: str = os.getenv("DB_PORT", "")
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_SCHEMA: str = os.getenv("DB_SCHEMA", "public")

    OAUTH_URL: str = os.getenv("OAUTH_URL", "")
    OAUTH_CLIENT_ID: str = os.getenv("OAUTH_CLIENT_ID", "")
    OAUTH_CLIENT_SECRET: str = os.getenv("OAUTH_CLIENT_SECRET", "")

    MODEL_SMALL: str = os.getenv("RESEARCH_MODEL_SMALL", "")
    MODEL_LARGE: str = os.getenv("RESEARCH_MODEL_LARGE", "")
    MODEL_EMBEDDING: str = os.getenv("RESEARCH_MODEL_EMBEDDING", "")

    MODEL_SMALL_PROMPT_COST: float = float(
        os.getenv("RESEARCH_MODEL_SMALL_PROMPT_COST") or 0
    )
    MODEL_SMALL_COMPLETION_COST: float = float(
        os.getenv("RESEARCH_MODEL_SMALL_COMPLETION_COST") or 0
    )
    MODEL_LARGE_PROMPT_COST: float = float(
        os.getenv("RESEARCH_MODEL_LARGE_PROMPT_COST") or 0
    )
    MODEL_LARGE_COMPLETION_COST: float = float(
        os.getenv("RESEARCH_MODEL_LARGE_COMPLETION_COST") or 0
    )
    MODEL_EMBEDDING_PROMPT_COST: float = float(
        os.getenv("RESEARCH_MODEL_EMBEDDING_PROMPT_COST") or 0
    )
    MODEL_EMBEDDING_COMPLETION_COST: float = float(
        os.getenv("RESEARCH_MODEL_EMBEDDING_COMPLETION_COST") or 0
    )

    MAX_HISTORY_LENGTH: int = int(os.getenv("RESEARCH_MAX_HISTORY_LENGTH") or 10)
    MAX_DATA_SOURCES_PER_QUERY: int = int(
        os.getenv("RESEARCH_MAX_DATA_SOURCES_PER_QUERY") or 5
    )

    LOG_LEVEL: str = os.getenv("RESEARCH_LOG_LEVEL", "INFO")

    PROCESS_MONITOR_MODEL_NAME: str = os.getenv(
        "RESEARCH_PROCESS_MONITOR_MODEL_NAME", ""
    )

    @classmethod
    def validate_required_environment(cls) -> bool:
        """Check that all required environment variables are set.

        Returns:
            True if all required values are present, False otherwise.
        """
        required = {
            "DB_HOST": cls.DB_HOST,
            "DB_PORT": cls.DB_PORT,
            "DB_NAME": cls.DB_NAME,
            "DB_USER": cls.DB_USER,
            "RESEARCH_MODEL_SMALL": cls.MODEL_SMALL,
            "RESEARCH_MODEL_LARGE": cls.MODEL_LARGE,
            "RESEARCH_MODEL_EMBEDDING": cls.MODEL_EMBEDDING,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            logger.error(
                "Missing required environment variables: %s", ", ".join(missing)
            )
            return False
        logger.info("All required configuration values are set")
        return True

    @classmethod
    def get_database_params(cls) -> dict:
        """Build connection parameters dict for PostgreSQL.

        Returns:
            Dict with host, port, dbname, user, password keys.
        """
        return {
            "host": cls.DB_HOST,
            "port": cls.DB_PORT,
            "dbname": cls.DB_NAME,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD,
        }

    @classmethod
    def get_model_settings(cls, capability: str) -> dict:
        """Get model name and cost configuration for a capability tier.

        Args:
            capability: One of "small", "large", or "embedding".

        Returns:
            Dict with name, prompt_token_cost, completion_token_cost keys.

        Raises:
            ValueError: If capability is not a recognized tier.
        """
        configs = {
            "small": (
                cls.MODEL_SMALL,
                cls.MODEL_SMALL_PROMPT_COST,
                cls.MODEL_SMALL_COMPLETION_COST,
                "minimal",
            ),
            "large": (
                cls.MODEL_LARGE,
                cls.MODEL_LARGE_PROMPT_COST,
                cls.MODEL_LARGE_COMPLETION_COST,
                "medium",
            ),
            "embedding": (
                cls.MODEL_EMBEDDING,
                cls.MODEL_EMBEDDING_PROMPT_COST,
                cls.MODEL_EMBEDDING_COMPLETION_COST,
                None,
            ),
        }
        if capability not in configs:
            raise ValueError(
                f"Unknown capability: {capability}. Use: small, large, embedding"
            )
        name, prompt_cost, completion_cost, reasoning_effort = configs[capability]
        result = {
            "name": name,
            "prompt_token_cost": prompt_cost,
            "completion_token_cost": completion_cost,
        }
        if reasoning_effort is not None:
            result["reasoning_effort"] = reasoning_effort
        return result


config = Config()
