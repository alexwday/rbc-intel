"""Tests for ingestion.utils.llm."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from ingestion.utils.llm import LLMClient


def _setup_api_key_env(monkeypatch):
    """Set env vars for API key auth mode."""
    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_ENDPOINT", "https://api.test.com/v1")
    monkeypatch.setenv("STARTUP_MODEL", "gpt-4")
    monkeypatch.setenv("STARTUP_MAX_TOKENS", "50")
    monkeypatch.setenv("STARTUP_TEMPERATURE", "0")


def _setup_oauth_env(monkeypatch):
    """Set env vars for OAuth auth mode."""
    monkeypatch.setenv("AUTH_MODE", "oauth")
    monkeypatch.setenv("OAUTH_TOKEN_ENDPOINT", "https://auth/token")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("OAUTH_SCOPE", "")
    monkeypatch.setenv("LLM_ENDPOINT", "https://api.test.com/v1")
    monkeypatch.setenv("STARTUP_MODEL", "gpt-4")
    monkeypatch.setenv("STARTUP_MAX_TOKENS", "50")
    monkeypatch.setenv("STARTUP_TEMPERATURE", "0")


@patch("ingestion.utils.llm.OpenAI")
def test_init_with_api_key(mock_openai, monkeypatch):
    """Creates static OpenAI client with API key."""
    _setup_api_key_env(monkeypatch)
    client = LLMClient()
    mock_openai.assert_called_once_with(
        api_key="sk-test",
        base_url="https://api.test.com/v1",
    )
    assert client.auth_mode == "api_key"


@patch("ingestion.utils.llm.OAuthClient")
def test_init_with_oauth(mock_oauth_cls, monkeypatch):
    """Creates OAuthClient when mode is oauth."""
    _setup_oauth_env(monkeypatch)
    client = LLMClient()
    mock_oauth_cls.assert_called_once()
    assert client.auth_mode == "oauth"


@patch("ingestion.utils.llm.OpenAI")
def test_get_client_static(mock_openai, monkeypatch):
    """Returns static client for api_key mode."""
    _setup_api_key_env(monkeypatch)
    client = LLMClient()
    result = client.get_client()
    assert result == mock_openai.return_value


@patch("ingestion.utils.llm.OpenAI")
@patch("ingestion.utils.llm.OAuthClient")
def test_get_client_oauth(mock_oauth_cls, mock_openai, monkeypatch):
    """Creates new client with token for oauth mode."""
    _setup_oauth_env(monkeypatch)
    mock_oauth_cls.return_value.get_token.return_value = "tok"
    client = LLMClient()
    mock_openai.reset_mock()
    result = client.get_client()
    mock_openai.assert_called_once_with(
        api_key="tok",
        base_url="https://api.test.com/v1",
    )
    assert result == mock_openai.return_value


@patch("ingestion.utils.llm.OpenAI")
def test_call_basic(mock_openai, monkeypatch):
    """Makes a basic chat completion call."""
    _setup_api_key_env(monkeypatch)
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {"choices": []}
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = mock_completion
    client = LLMClient()
    result = client.call(
        messages=[{"role": "user", "content": "hi"}],
        stage="startup",
    )
    assert result == {"choices": []}
    create_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert create_kwargs["model"] == "gpt-4"
    assert create_kwargs["max_completion_tokens"] == 50


@patch("ingestion.utils.llm.OpenAI")
def test_call_with_tools(mock_openai, monkeypatch):
    """Passes tools when provided."""
    _setup_api_key_env(monkeypatch)
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {}
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = mock_completion
    client = LLMClient()
    tools = [{"type": "function", "function": {}}]
    client.call(
        messages=[{"role": "user", "content": "hi"}],
        stage="startup",
        tools=tools,
    )
    create_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert create_kwargs["tools"] == tools


@patch("ingestion.utils.llm.OpenAI")
def test_call_with_tool_choice(mock_openai, monkeypatch):
    """Passes tool_choice when provided."""
    _setup_api_key_env(monkeypatch)
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {}
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = mock_completion
    client = LLMClient()
    tools = [{"type": "function", "function": {}}]
    client.call(
        messages=[{"role": "user", "content": "hi"}],
        stage="startup",
        tools=tools,
        tool_choice="required",
    )
    create_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert create_kwargs["tool_choice"] == "required"


@patch("ingestion.utils.llm.OpenAI")
def test_call_uses_stage_config(mock_openai, monkeypatch):
    """Uses model config from the specified stage."""
    _setup_api_key_env(monkeypatch)
    monkeypatch.setenv("CLASSIFICATION_MODEL", "gpt-5")
    monkeypatch.setenv("CLASSIFICATION_MAX_TOKENS", "500")
    monkeypatch.setenv("CLASSIFICATION_TEMPERATURE", "0.7")
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {}
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = mock_completion
    client = LLMClient()
    client.call(
        messages=[{"role": "user", "content": "hi"}],
        stage="classification",
    )
    create_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert create_kwargs["model"] == "gpt-5"
    assert create_kwargs["max_completion_tokens"] == 500


@patch("ingestion.utils.llm.OpenAI")
def test_call_debug_logging_omits_none_temperature(
    mock_openai,
    monkeypatch,
    caplog,
):
    """Debug logging omits temp when the stage temperature is unset."""
    _setup_api_key_env(monkeypatch)
    monkeypatch.setenv("ENRICHMENT_MODEL", "gpt-5-mini")
    monkeypatch.setenv("ENRICHMENT_MAX_TOKENS", "4000")
    monkeypatch.delenv("ENRICHMENT_TEMPERATURE", raising=False)
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {}
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = mock_completion
    client = LLMClient()

    with caplog.at_level(logging.DEBUG):
        client.call(
            messages=[{"role": "user", "content": "hi"}],
            stage="enrichment",
            tools=[{"type": "function", "function": {}}],
            context="sample.pdf page 1/2",
        )

    assert "sample.pdf page 1/2" in caplog.text
    assert "temp=" not in caplog.text


@patch("ingestion.utils.llm.OpenAI")
def test_test_connection_success(mock_openai, monkeypatch):
    """test_connection returns True when tool call is returned."""
    _setup_api_key_env(monkeypatch)
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"name": "ping"}}]}}
        ]
    }
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = mock_completion
    client = LLMClient()
    assert client.test_connection() is True


@patch("ingestion.utils.llm.OpenAI")
def test_test_connection_no_tool_call(mock_openai, monkeypatch):
    """test_connection raises when model returns no tool call."""
    _setup_api_key_env(monkeypatch)
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = mock_completion
    client = LLMClient()
    with pytest.raises(RuntimeError, match="tool call"):
        client.test_connection()


@patch("ingestion.utils.llm.OpenAI")
def test_test_connection_failure(mock_openai, monkeypatch):
    """test_connection raises on API failure."""
    _setup_api_key_env(monkeypatch)
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.side_effect = Exception(
        "connection failed"
    )
    client = LLMClient()
    with pytest.raises(Exception, match="connection failed"):
        client.test_connection()
