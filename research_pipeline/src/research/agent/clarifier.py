"""
Clarifier Agent — context assessment and research scope detection.

Determines:
1. Whether to ask clarifying questions or proceed with research
2. Whether the query is source-wide (requires checking ALL files) or selective
3. For source-wide queries, whether to request user approval for extended research
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from ..connections.llm import execute_llm_call
from ..utils.config import config
from ..utils.input_sanitizer import format_conversation_history_for_prompt
from ..utils.prompt_loader import get_prompt

MODEL_CAPABILITY = "small"
MODEL_MAX_TOKENS = 4096
MODEL_TEMPERATURE = 0.0

logger = logging.getLogger(__name__)


class ClarifierError(Exception):
    """Exception raised for clarifier-related errors."""


def _get_clarifier_model_settings() -> Dict[str, Any]:
    """Return model settings for the clarifier LLM."""
    return config.get_model_settings(MODEL_CAPABILITY)


def _build_clarifier_messages(
    system_prompt: str,
    user_prompt_template: str,
    conversation: Dict[str, Any],
) -> list:
    """Build the messages payload for the LLM call.

    Raises:
        ClarifierError: If no user prompt template is provided.
    """
    if not user_prompt_template:
        raise ClarifierError(
            "user_prompt not found in database for agent/clarifier. "
            "Please ensure the prompt is configured in the prompts table."
        )

    messages = [{"role": "system", "content": system_prompt}]

    conversation_context = format_conversation_history_for_prompt(conversation)
    user_content = user_prompt_template.replace(
        "{{conversation}}", conversation_context
    )

    messages.append({"role": "user", "content": user_content})
    return messages


def _parse_clarifier_tool_response(response: Any) -> Dict[str, Any]:
    """Extract and validate tool call arguments from the LLM response.

    Raises:
        ClarifierError: If the response is invalid or tool call parsing fails.
    """
    if (
        not response
        or not hasattr(response, "choices")
        or not response.choices
    ):
        raise ClarifierError("Invalid or empty response received from LLM")

    message = getattr(response.choices[0], "message", None)
    tool_calls = getattr(message, "tool_calls", None)
    if not message or not tool_calls:
        content_returned = (
            message.content if message and message.content else "No content"
        )
        logger.warning(
            "Expected tool call but received content: %s...",
            content_returned[:100],
        )
        raise ClarifierError(
            "No tool call received in response, content returned instead."
        )

    tool_call = tool_calls[0]

    if tool_call.function.name != "make_clarifier_decision":
        raise ClarifierError(
            f"Unexpected function call: {tool_call.function.name}"
        )

    try:
        return json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        raise ClarifierError(
            f"Invalid JSON in tool arguments: "
            f"{tool_call.function.arguments}"
        ) from exc


def _normalize_bool_flag(value: Any) -> bool:
    """Normalize bool-like values from tool arguments."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _validate_clarifier_decision(
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate and normalize decision fields from tool arguments.

    Raises:
        ClarifierError: If required fields are missing.
    """
    action = arguments.get("action")
    output = arguments.get("output")

    if not action:
        raise ClarifierError("Missing 'action' in tool arguments")

    if not output:
        raise ClarifierError("Missing 'output' in tool arguments")

    valid_actions = {
        "ask_clarification",
        "request_deep_research_approval",
        "proceed_with_research",
    }
    if action not in valid_actions:
        logger.warning(
            "Unexpected action '%s', defaulting to proceed_with_research",
            action,
        )
        action = "proceed_with_research"

    return {
        "action": action,
        "output": output,
        "is_db_wide": _normalize_bool_flag(
            arguments.get("is_db_wide", False)
        ),
        "deep_research_approved": _normalize_bool_flag(
            arguments.get("deep_research_approved", False)
        ),
    }


def generate_clarifier_decision(
    conversation: Dict[str, Any],
    token: str,
    available_data_sources: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Determine the clarifier decision for a conversation.

    Args:
        conversation: Conversation payload with a ``messages`` list.
        token: Authentication token for LLM access.
        available_data_sources: Available data source configurations.

    Returns:
        Decision dictionary and optional LLM usage details.

    Raises:
        ClarifierError: If there is an error in the clarification process.
    """
    try:
        system_prompt, tools, user_prompt_template = get_prompt(
            "agent",
            "clarifier",
            inject_fiscal=True,
            inject_data_sources=True,
            available_data_sources=available_data_sources,
        )
        model_settings = _get_clarifier_model_settings()
        messages = _build_clarifier_messages(
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
                "function": {"name": "make_clarifier_decision"},
            },
            stream=False,
            prompt_token_cost=model_settings["prompt_token_cost"],
            completion_token_cost=model_settings["completion_token_cost"],
            reasoning_effort=model_settings.get("reasoning_effort"),
        )

        arguments = _parse_clarifier_tool_response(response)
        decision = _validate_clarifier_decision(arguments)

        logger.info(
            "Clarifier decision: action=%s, is_db_wide=%s, "
            "deep_research_approved=%s",
            decision["action"],
            decision["is_db_wide"],
            decision["deep_research_approved"],
        )

        return decision, usage_details

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        logger.error(
            "Error clarifying research needs: %s",
            str(exc),
            exc_info=True,
        )
        raise ClarifierError(
            f"Failed to clarify research needs: {exc}"
        ) from exc
