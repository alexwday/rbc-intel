"""LLM connector with swappable auth (OAuth or API key)."""

import logging

from openai import OpenAI

from .oauth import OAuthClient
from ..utils.config import (
    get_api_key,
    get_auth_mode,
    get_llm_endpoint,
    get_oauth_config,
    get_stage_model_config,
)
from ..utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM client with pluggable auth.

    Supports OAuth (token auto-refreshes before each call)
    or static API key, controlled by AUTH_MODE env var.

    Params:
        None — configuration loaded from environment

    Example:
        >>> client = LLMClient()
        >>> result = client.call(
        ...     messages=[{"role": "user", "content": "hi"}],
        ...     stage="classification",
        ... )
    """

    def __init__(self):
        self.auth_mode = get_auth_mode()
        self.endpoint = get_llm_endpoint()
        self.oauth_client = None
        self.static_client = None

        if self.auth_mode == "oauth":
            oauth_cfg = get_oauth_config()
            self.oauth_client = OAuthClient(config=oauth_cfg)
            logger.info("LLM client configured with OAuth")
        else:
            api_key = get_api_key()
            self.static_client = OpenAI(
                api_key=api_key,
                base_url=self.endpoint,
            )
            logger.info("LLM client configured with API key")

    def get_client(self) -> OpenAI:
        """Build or return an OpenAI client with current auth.

        Params:
            None

        Returns:
            OpenAI — configured client instance

        Example:
            >>> client = LLMClient()
            >>> openai_client = client.get_client()
        """
        if self.static_client:
            return self.static_client
        token = self.oauth_client.get_token()
        return OpenAI(
            api_key=token,
            base_url=self.endpoint,
        )

    def call(
        self,
        messages: list,
        stage: str = "startup",
        tools: list | None = None,
        tool_choice: str | None = None,
    ) -> dict:
        """Make an LLM tool-calling request.

        Model and max_tokens are read from env vars based
        on the stage name ({STAGE}_MODEL, {STAGE}_MAX_TOKENS).

        Params:
            messages: List of message dicts
                (e.g. [{"role": "user", "content": "..."}])
            stage: Pipeline stage name for model config
                (e.g. "startup", "classification")
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice constraint
                (e.g. "required", "auto", "none")

        Returns:
            dict — the full API response as a dict

        Example:
            >>> resp = client.call(
            ...     messages=[{"role": "user", "content": "hi"}],
            ...     stage="classification",
            ... )
        """
        client = self.get_client()
        model_config = get_stage_model_config(stage)

        kwargs = {
            "model": model_config["model"],
            "messages": messages,
            "max_completion_tokens": model_config["max_tokens"],
        }
        if model_config["temperature"] is not None:
            kwargs["temperature"] = model_config["temperature"]
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        logger.debug(
            "LLM call: stage=%s, model=%s, "
            "max_tokens=%d, temp=%s, "
            "messages=%d, tools=%d",
            stage,
            model_config["model"],
            model_config["max_tokens"],
            model_config["temperature"],
            len(messages),
            len(tools) if tools else 0,
        )

        response = client.chat.completions.create(**kwargs)
        return response.model_dump()

    def test_connection(self) -> bool:
        """Validate LLM connectivity with a tool-calling request.

        Loads the startup_health_check prompt and verifies
        the model returns an actual tool call.

        Params:
            None

        Returns:
            bool — True if the call succeeds

        Example:
            >>> client.test_connection()
            True
        """
        prompt = load_prompt("startup_health_check")
        messages = []
        if prompt.get("system_prompt"):
            messages.append(
                {"role": "system", "content": prompt["system_prompt"]}
            )
        messages.append({"role": "user", "content": prompt["user_prompt"]})
        try:
            response = self.call(
                messages=messages,
                stage=prompt["stage"],
                tools=prompt.get("tools"),
                tool_choice=prompt.get("tool_choice"),
            )
            choices = response.get("choices", [])
            tool_calls = (
                choices[0].get("message", {}).get("tool_calls")
                if choices
                else None
            )
            if not tool_calls:
                raise RuntimeError("LLM did not return a tool call")
            logger.info("LLM connection test passed")
            return True
        except Exception:
            logger.error("LLM connection test failed")
            raise
