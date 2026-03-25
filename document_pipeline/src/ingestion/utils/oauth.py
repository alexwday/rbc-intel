"""OAuth 2.0 client credentials flow with lazy token refresh."""

import logging
import time

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

REFRESH_BUFFER_SECONDS = 300
REQUEST_TIMEOUT = 30


def _should_retry_with_body_credentials(response) -> bool:
    """Check whether a 400 response indicates auth fallback. Returns: bool."""
    if response.status_code != 400:
        return False
    try:
        error_data = response.json()
    except ValueError:
        return False

    error_value = str(error_data.get("error", "")).lower()
    description = str(error_data.get("error_description", "")).lower()
    return (
        error_value in {"invalid_client", "unauthorized_client"}
        or "basic" in description
        or "client credential" in description
        or "client_secret" in description
        or "client_id" in description
    )


class OAuthClient:
    """Manages OAuth token lifecycle with on-demand refresh.

    Fetches tokens via client credentials grant. Caches the
    token and refreshes automatically when it nears expiry.

    Params:
        config: dict with keys token_endpoint, client_id,
            client_secret, scope (from get_oauth_config)
        verify_ssl: Whether to verify SSL certificates

    Example:
        >>> client = OAuthClient(
        ...     config={"token_endpoint": "https://...",
        ...             "client_id": "id",
        ...             "client_secret": "secret",
        ...             "scope": ""},
        ... )
        >>> token = client.get_token()
    """

    def __init__(self, config: dict, verify_ssl: bool = True):
        self.token_endpoint = config["token_endpoint"]
        self.client_id = config["client_id"]
        self.client_secret = config["client_secret"]
        self.scope = config.get("scope", "")
        self.verify_ssl = verify_ssl
        self._access_token = ""
        self._expires_at = 0.0

    def get_token(self) -> str:
        """Get a valid access token, refreshing if needed.

        Params:
            None

        Returns:
            str — the bearer token

        Example:
            >>> client.get_token()
            "eyJhbGciOi..."
        """
        if self.is_expired():
            self._fetch_token()
        return self._access_token

    def is_expired(self) -> bool:
        """Check if token is missing or near expiry.

        Params:
            None

        Returns:
            bool — True if token needs refresh

        Example:
            >>> client.is_expired()
            True
        """
        if not self._access_token:
            return True
        remaining = self._expires_at - time.time()
        return remaining <= REFRESH_BUFFER_SECONDS

    def _fetch_token(self) -> None:
        """Fetch a new token via client credentials grant."""
        logger.info(
            "Fetching OAuth token from %s",
            self.token_endpoint,
        )

        data = {"grant_type": "client_credentials"}
        if self.scope:
            data["scope"] = self.scope

        auth = HTTPBasicAuth(self.client_id, self.client_secret)
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(
            self.token_endpoint,
            data=data,
            auth=auth,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            verify=self.verify_ssl,
        )

        if _should_retry_with_body_credentials(response):
            logger.info("Basic auth failed, retrying with body credentials")
            data["client_id"] = self.client_id
            data["client_secret"] = self.client_secret
            response = requests.post(
                self.token_endpoint,
                data=data,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                verify=self.verify_ssl,
            )

        response.raise_for_status()
        token_data = response.json()

        self._access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._expires_at = time.time() + expires_in

        logger.info(
            "OAuth token obtained (expires in %ds)",
            expires_in,
        )
