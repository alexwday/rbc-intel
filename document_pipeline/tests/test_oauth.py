"""Tests for ingestion.utils.oauth."""

from unittest.mock import MagicMock, patch

import pytest

from ingestion.utils.oauth import (
    OAuthClient,
    _should_retry_with_body_credentials,
)

SAMPLE_CONFIG = {
    "token_endpoint": "https://auth.example.com/token",
    "client_id": "test-id",
    "client_secret": "test-secret",
    "scope": "read",
}


def _mock_token_response(
    status_code=200,
    expires_in=3600,
    error="",
    error_description="",
):
    """Build a mock requests.Response. Returns: MagicMock."""
    resp = MagicMock()
    resp.status_code = status_code
    if error:
        resp.json.return_value = {
            "error": error,
            "error_description": error_description,
        }
    else:
        resp.json.return_value = {
            "access_token": "tok-abc",
            "expires_in": expires_in,
        }
    resp.raise_for_status = MagicMock()
    return resp


def test_get_token_fetches_on_first_call():
    """First call fetches a new token."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = _mock_token_response()
        token = client.get_token()
    assert token == "tok-abc"
    mock.assert_called_once()


def test_get_token_returns_cached():
    """Subsequent calls return cached token."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = _mock_token_response()
        client.get_token()
        token = client.get_token()
    assert token == "tok-abc"
    assert mock.call_count == 1


def test_get_token_refreshes_when_expired():
    """Refreshes when token is near expiry."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = _mock_token_response(expires_in=1)
        client.get_token()
        assert client.is_expired()
        client.get_token()
    assert mock.call_count == 2


def test_is_expired_true_when_no_token():
    """Expired when no token exists."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    assert client.is_expired() is True


def test_is_expired_false_when_fresh():
    """Not expired when token was just fetched."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = _mock_token_response(expires_in=7200)
        client.get_token()
    assert client.is_expired() is False


def test_is_expired_true_when_near_expiry():
    """Expired when within buffer window."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = _mock_token_response(expires_in=1)
        client.get_token()
    assert client.is_expired() is True


def test_fetch_token_with_scope():
    """Includes scope in request when set."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = _mock_token_response()
        client.get_token()
    call_data = mock.call_args.kwargs["data"]
    assert call_data["scope"] == "read"


def test_fetch_token_without_scope():
    """Omits scope when empty."""
    config = {**SAMPLE_CONFIG, "scope": ""}
    client = OAuthClient(config=config)
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = _mock_token_response()
        client.get_token()
    call_data = mock.call_args.kwargs["data"]
    assert "scope" not in call_data


def test_fetch_token_fallback_to_body_credentials():
    """Falls back to body credentials on 400."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    fail_resp = _mock_token_response(
        status_code=400,
        error="invalid_client",
        error_description="Basic auth rejected client credentials",
    )
    ok_resp = _mock_token_response(status_code=200)
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.side_effect = [fail_resp, ok_resp]
        token = client.get_token()
    assert token == "tok-abc"
    assert mock.call_count == 2
    retry_data = mock.call_args_list[1].kwargs["data"]
    assert retry_data["client_id"] == "test-id"
    assert retry_data["client_secret"] == "test-secret"


def test_fetch_token_does_not_retry_on_non_auth_400():
    """Does not retry unrelated 400 responses."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    resp = _mock_token_response(
        status_code=400,
        error="invalid_scope",
        error_description="Requested scope is not allowed",
    )
    resp.raise_for_status.side_effect = Exception("400")
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = resp
        with pytest.raises(Exception, match="400"):
            client.get_token()
    assert mock.call_count == 1


def test_fetch_token_raises_on_failure():
    """Raises on non-200 response."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    resp = _mock_token_response(status_code=401)
    resp.raise_for_status.side_effect = Exception("401")
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = resp
        with pytest.raises(Exception, match="401"):
            client.get_token()


def test_fetch_token_default_expiry():
    """Defaults to 3600s if expires_in is missing."""
    client = OAuthClient(config=SAMPLE_CONFIG)
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"access_token": "tok"}
    resp.raise_for_status = MagicMock()
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = resp
        client.get_token()
    assert client.is_expired() is False


def test_verify_ssl_passed_to_request():
    """SSL verify flag is passed to requests.post."""
    client = OAuthClient(config=SAMPLE_CONFIG, verify_ssl=False)
    with patch("ingestion.utils.oauth.requests.post") as mock:
        mock.return_value = _mock_token_response()
        client.get_token()
    assert mock.call_args.kwargs["verify"] is False


def test_should_retry_with_body_credentials_false_for_non_json():
    """Non-JSON 400s do not trigger auth fallback."""
    resp = MagicMock()
    resp.status_code = 400
    resp.json.side_effect = ValueError("not json")
    assert _should_retry_with_body_credentials(resp) is False
