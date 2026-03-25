"""
Token Provider - API authentication for LLM access.

Supports two modes: local development using OPENAI_API_KEY from environment,
or production using OAuth client credentials flow.
"""

import logging
import os
import time

import requests

from ..utils.config import config

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 180
_MAX_RETRY_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2


def fetch_oauth_token() -> tuple[str, dict]:
    """Get API token - returns OPENAI_API_KEY if set, otherwise fetches OAuth token.

    Returns:
        Tuple of (token, auth_info) where auth_info contains method and details.

    Raises:
        ValueError: If neither OPENAI_API_KEY nor OAuth settings are configured.
        requests.exceptions.RequestException: If OAuth request fails after retries.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.debug("Using OPENAI_API_KEY from environment")
        return api_key, {"method": "api_key_local", "source": "OPENAI_API_KEY"}

    oauth_url = config.OAUTH_URL
    client_id = config.OAUTH_CLIENT_ID
    client_secret = config.OAUTH_CLIENT_SECRET

    if not all([oauth_url, client_id, client_secret]):
        raise ValueError(
            "Missing OPENAI_API_KEY or OAuth settings (URL, client ID, secret)"
        )

    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }

    start_time = time.time()

    with requests.Session() as session:
        for attempt_num in range(1, _MAX_RETRY_ATTEMPTS + 1):
            attempt_start = time.time()

            try:
                response = session.post(
                    oauth_url, data=payload, timeout=_REQUEST_TIMEOUT
                )
                response.raise_for_status()

                token_data = response.json()
                if not isinstance(token_data, dict):
                    raise ValueError("OAuth response is not a JSON object")

                token = token_data.get("access_token")
                if not token:
                    raise ValueError("OAuth token not found in response")

                logger.debug(
                    "OAuth token acquired in %.2fs (attempt %d/%d)",
                    time.time() - attempt_start,
                    attempt_num,
                    _MAX_RETRY_ATTEMPTS,
                )

                return str(token), {
                    "method": "oauth",
                    "client_id": client_id,
                }

            except (requests.exceptions.RequestException, ValueError) as exc:
                logger.warning(
                    "OAuth attempt %d/%d failed after %.2f seconds: %s",
                    attempt_num,
                    _MAX_RETRY_ATTEMPTS,
                    time.time() - attempt_start,
                    exc,
                )

                if attempt_num == _MAX_RETRY_ATTEMPTS:
                    logger.error(
                        "OAuth failed after %d attempts (%.2fs total)",
                        _MAX_RETRY_ATTEMPTS,
                        time.time() - start_time,
                    )
                    raise

                time.sleep(_RETRY_DELAY_SECONDS)

    raise ValueError("OAuth token acquisition failed")
