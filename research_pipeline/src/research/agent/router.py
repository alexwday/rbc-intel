"""Router agent that selects the processing path via an LLM tool call."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..connections.llm import execute_llm_call
from ..utils.config import config
from ..utils.input_sanitizer import format_conversation_history_for_prompt
from ..utils.prompt_loader import get_prompt

MODEL_CAPABILITY = "small"
MODEL_MAX_TOKENS = 4096
MODEL_TEMPERATURE = 0.0

logger = logging.getLogger(__name__)


class RouterError(Exception):
    """Exception raised for router-related errors."""


def _get_router_model_settings() -> Dict[str, Any]:
    """Return model settings from config based on capability tier."""
    return config.get_model_settings(MODEL_CAPABILITY)


def _build_router_messages(
    system_prompt: str,
    user_prompt_template: str,
    conversation: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Construct the messages list for the LLM call.

    Raises:
        RouterError: If the user prompt template is missing.
    """
    if not user_prompt_template:
        raise RouterError(
            "user_prompt not found in database for agent/router. "
            "Please ensure the prompt is configured in the prompts table."
        )

    conversation_context = format_conversation_history_for_prompt(conversation)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt_template.replace(
                "{{conversation}}", conversation_context
            ),
        },
    ]


def _parse_router_tool_response(response: Any) -> Dict[str, Any]:
    """Extract and validate tool call arguments from the LLM response.

    Raises:
        RouterError: If the response is invalid or tool call parsing fails.
    """
    if not response or not hasattr(response, "choices") or not response.choices:
        raise RouterError("Invalid or empty response received from LLM")

    message = response.choices[0].message
    if not message or not message.tool_calls:
        content_returned = (
            message.content if message and message.content else "No content"
        )
        logger.warning(
            "Expected tool call but received content: %s...",
            content_returned[:100],
        )
        raise RouterError(
            "No tool call received in response, content returned instead."
        )

    tool_call = message.tool_calls[0]
    function_name = getattr(
        getattr(tool_call, "function", None), "name", None
    )
    if function_name != "route_query":
        raise RouterError(f"Unexpected function call: {function_name}")

    arguments = tool_call.function.arguments
    if isinstance(arguments, dict):
        parsed_arguments = arguments
    else:
        try:
            parsed_arguments = json.loads(arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RouterError(
                f"Invalid JSON in tool arguments: "
                f"{tool_call.function.arguments}"
            ) from exc

    if not isinstance(parsed_arguments, dict):
        raise RouterError("Tool arguments must be a JSON object")

    return parsed_arguments


VALID_ROUTES = {"direct_response", "database_research"}


def _validate_routing_function_name(arguments: Dict[str, Any]) -> str:
    """Validate the routing function name string.

    Raises:
        RouterError: If function_name is missing or invalid.
    """
    function_name = arguments.get("function_name")
    if function_name is None:
        raise RouterError("Missing 'function_name' in tool arguments")
    if function_name not in VALID_ROUTES:
        raise RouterError(
            f"Invalid function_name: {function_name}, "
            f"expected 'direct_response' or 'database_research'"
        )
    return function_name


def generate_routing_decision(
    conversation: Dict[str, Any],
    token: str,
    available_data_sources: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Return the routing decision from the model using a tool call.

    Args:
        conversation: Conversation payload with ``messages``.
        token: Authentication token for API access.
        available_data_sources: Optional data source configurations.

    Returns:
        Routing decision dict with ``function_name`` and usage details.

    Raises:
        RouterError: If the routing decision cannot be determined.
    """
    try:
        system_prompt, tools, user_prompt_template = get_prompt(
            "agent",
            "router",
            inject_fiscal=True,
            inject_data_sources=True,
            available_data_sources=available_data_sources,
        )
        model_settings = _get_router_model_settings()
        messages = _build_router_messages(
            system_prompt, user_prompt_template, conversation
        )

        response, usage_details = execute_llm_call(
            oauth_token=token,
            model=model_settings["name"],
            messages=messages,
            max_tokens=MODEL_MAX_TOKENS,
            temperature=MODEL_TEMPERATURE,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "route_query"},
            },
            stream=False,
            prompt_token_cost=model_settings["prompt_token_cost"],
            completion_token_cost=model_settings["completion_token_cost"],
            reasoning_effort=model_settings.get("reasoning_effort"),
        )

        arguments = _parse_router_tool_response(response)
        function_name = _validate_routing_function_name(arguments)

        logger.info("Routing decision: %s", function_name)

        return {"function_name": function_name}, usage_details

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        logger.error(
            "Error getting routing decision: %s", str(exc), exc_info=True
        )
        raise RouterError(
            f"Failed to get routing decision: {exc}"
        ) from exc
