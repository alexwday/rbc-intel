"""
LLM Connector - OpenAI API client with retry logic.

Provides the interface to OpenAI-compatible APIs for all research pipeline
LLM calls. Handles streaming and non-streaming completions, embeddings,
tool calls, and token usage tracking.
"""

import logging
import time
from typing import Any, Iterator, Optional

from openai import OpenAI, OpenAIError

from ..utils.config import config

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.INFO)

_REQUEST_TIMEOUT = 180
_MAX_RETRY_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2


class OpenAIConnectorError(Exception):
    """Exception class for OpenAI connector errors."""


UsageDetails = Optional[dict[str, Any]]


def calculate_token_cost(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_token_cost: float,
    completion_token_cost: float,
) -> float:
    """Calculate total cost based on token usage and per-token costs.

    Args:
        prompt_tokens: Number of prompt tokens used.
        completion_tokens: Number of completion tokens used.
        prompt_token_cost: Cost per 1K prompt tokens in USD.
        completion_token_cost: Cost per 1K completion tokens in USD.

    Returns:
        Total cost in USD.
    """
    prompt_cost = (prompt_tokens / 1000) * prompt_token_cost
    completion_cost = (completion_tokens / 1000) * completion_token_cost
    return prompt_cost + completion_cost


def _build_usage_details_from_response(
    api_response: Any,
    model_name: str,
    prompt_token_cost: float,
    completion_token_cost: float,
    response_time_ms: int,
) -> UsageDetails:
    """Build usage details dict from API response."""
    if not hasattr(api_response, "usage") or not api_response.usage:
        return None

    prompt_tokens = api_response.usage.prompt_tokens or 0
    completion_tokens = api_response.usage.completion_tokens or 0
    cost = calculate_token_cost(
        prompt_tokens, completion_tokens, prompt_token_cost, completion_token_cost
    )
    return {
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": cost,
        "response_time_ms": response_time_ms,
    }


def _execute_embedding_request(client: OpenAI, params: dict) -> Any:
    """Execute an embedding API call with the given parameters."""
    embedding_params = {
        "input": params.get("input"),
        "model": params.get("model"),
        "dimensions": params.get("dimensions"),
        "timeout": params.get("timeout", _REQUEST_TIMEOUT),
    }
    embedding_params = {k: v for k, v in embedding_params.items() if v is not None}
    return client.embeddings.create(**embedding_params)


def execute_llm_call(
    oauth_token: str,
    prompt_token_cost: float = 0,
    completion_token_cost: float = 0,
    **params,
) -> Any:
    """Execute an OpenAI API call with automatic retry on failure.

    Args:
        oauth_token: OAuth token or OpenAI API key.
        prompt_token_cost: Cost per 1K prompt tokens in USD.
        completion_token_cost: Cost per 1K completion tokens in USD.
        **params: OpenAI API parameters (model, messages, stream, tools, etc.).

    Returns:
        API response and usage details (non-streaming) or chunk iterator (streaming).

    Raises:
        OpenAIConnectorError: If the call fails after all retry attempts.
    """
    call_start_time = time.time()
    base_url = config.BASE_URL
    client = OpenAI(api_key=oauth_token, base_url=base_url)
    logger.info("Connecting to OpenAI API at %s", base_url)

    if "timeout" not in params:
        params["timeout"] = _REQUEST_TIMEOUT

    is_embedding = params.pop("is_embedding", False)
    is_streaming = params.get("stream", False) if not is_embedding else False
    if is_streaming:
        stream_options = params.get("stream_options") or {}
        params["stream_options"] = {**stream_options, "include_usage": True}

    # Newer models (gpt-5-*, o-series) require max_completion_tokens
    # instead of max_tokens and do not support temperature.
    # They also support reasoning_effort to control thinking depth.
    if not is_embedding:
        model_lower = (params.get("model") or "").lower()
        if any(
            model_lower.startswith(p)
            for p in ("gpt-5", "o1", "o3", "o4")
        ):
            if "max_tokens" in params:
                params["max_completion_tokens"] = params.pop("max_tokens")
            params.pop("temperature", None)

    # Strip None values — APIs reject null for optional params
    params = {k: v for k, v in params.items() if v is not None}

    model_name = params.get("model", "unknown")
    reasoning = params.get("reasoning_effort", "not set")
    logger.info(
        "LLM call: model=%s, reasoning_effort=%s, keys=%s",
        model_name,
        reasoning,
        sorted(params.keys()),
    )
    last_exception = None

    for attempt_num in range(1, _MAX_RETRY_ATTEMPTS + 1):
        start_time = time.time()

        try:
            if is_embedding:
                return _execute_embedding_request(client, params)

            api_response = client.chat.completions.create(**params)

            if is_streaming:
                return _stream_response_with_usage(
                    stream_iterator=api_response,
                    model_name=model_name,
                    prompt_token_cost=prompt_token_cost,
                    completion_token_cost=completion_token_cost,
                    call_start_time=call_start_time,
                )

            return api_response, _build_usage_details_from_response(
                api_response,
                model_name,
                prompt_token_cost,
                completion_token_cost,
                int((time.time() - start_time) * 1000),
            )

        except (ValueError, TypeError, KeyError) as exc:
            logger.error(
                "LLM call failed with non-retryable error: %s: %s",
                type(exc).__name__,
                exc,
            )
            raise OpenAIConnectorError(
                f"Non-retryable error in OpenAI API call: {exc}"
            ) from exc

        except (OSError, RuntimeError, OpenAIError) as exc:
            last_exception = exc
            logger.warning(
                "LLM call attempt %d/%d failed after %.2f seconds: %s: %s",
                attempt_num,
                _MAX_RETRY_ATTEMPTS,
                time.time() - start_time,
                type(exc).__name__,
                exc,
            )

            if attempt_num < _MAX_RETRY_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS)

    logger.error("LLM call failed after %d attempts", _MAX_RETRY_ATTEMPTS)
    raise OpenAIConnectorError(
        f"Failed to complete OpenAI API call: {last_exception}"
    ) from last_exception


def _stream_response_with_usage(
    stream_iterator: Iterator,
    model_name: str,
    prompt_token_cost: float,
    completion_token_cost: float,
    call_start_time: float,
) -> Iterator:
    """Wrap the OpenAI stream iterator to handle usage statistics.

    Yields:
        Response chunks followed by a usage details dict.
    """
    final_usage_data = None
    total_response_time_ms = 0

    try:
        for chunk in stream_iterator:
            yield chunk
            if hasattr(chunk, "usage") and chunk.usage:
                final_usage_data = chunk.usage
    except GeneratorExit:
        return
    finally:
        total_response_time_ms = int((time.time() - call_start_time) * 1000)

    if final_usage_data:
        prompt_tokens = final_usage_data.prompt_tokens or 0
        completion_tokens = final_usage_data.completion_tokens or 0
        cost = calculate_token_cost(
            prompt_tokens,
            completion_tokens,
            prompt_token_cost,
            completion_token_cost,
        )
        yield {
            "usage_details": {
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "response_time_ms": total_response_time_ms,
            }
        }
    else:
        logger.warning(
            "Stream for model '%s' finished without usage data after %dms",
            model_name,
            total_response_time_ms,
        )
        yield {
            "usage_details": {
                "model": model_name,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0,
                "response_time_ms": total_response_time_ms,
                "warning": "usage_data_missing",
            }
        }
