"""
Input Sanitizer - Conversation history validation and formatting.

Validates message structure, keeps only user/assistant messages, enforces
history length limits, and formats conversations into prompt-ready strings.
"""

import logging
from typing import Any

from .config import config

logger = logging.getLogger(__name__)

_ALLOWED_ROLES = ("user", "assistant")


def sanitize_conversation_history(conversation: Any) -> dict[str, Any]:
    """Validate and filter conversation to user/assistant messages only.

    Args:
        conversation: Raw input as list of messages or dict with 'messages' key.

    Returns:
        Dict with 'messages' containing filtered message dicts, plus metadata:
        - original_count: Number of messages before filtering
        - system_filtered: Whether any system messages were removed
        - final_count: Number of messages after filtering and truncation

    Raises:
        ValueError: If conversation format is invalid or messages is not a list.
    """
    if isinstance(conversation, list):
        messages = conversation
    elif isinstance(conversation, dict) and "messages" in conversation:
        messages = conversation["messages"]
    else:
        raise ValueError(
            "Invalid conversation format; expected list or dict with 'messages'."
        )

    if not isinstance(messages, list):
        raise ValueError("Conversation messages must be provided as a list.")

    original_count = len(messages)
    system_filtered = False
    filtered = []
    for msg in messages:
        if not isinstance(msg, dict):
            logger.warning("Skipping non-dict message: %s", msg)
            continue

        role, content = msg.get("role"), msg.get("content")
        if role is None or content is None:
            logger.warning("Skipping message missing required fields: %s", msg)
            continue

        if role not in _ALLOWED_ROLES:
            if role == "system":
                system_filtered = True
            continue

        filtered.append({"role": role, "content": content})

    final_messages = filtered[-config.MAX_HISTORY_LENGTH :]
    return {
        "messages": final_messages,
        "original_count": original_count,
        "system_filtered": system_filtered,
        "final_count": len(final_messages),
    }


def format_conversation_history_for_prompt(conversation: dict[str, Any]) -> str:
    """Convert conversation messages to '[ROLE]: content' text format.

    Args:
        conversation: Dict with 'messages' key containing message dicts.

    Returns:
        Newline-separated string of formatted messages for prompt injection.

    Raises:
        ValueError: If conversation['messages'] is not a list.
    """
    messages = conversation.get("messages") if isinstance(conversation, dict) else None
    if not messages:
        return "No conversation history available."
    if not isinstance(messages, list):
        raise ValueError("conversation['messages'] must be provided as a list.")

    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            logger.warning("Skipping non-dict message: %s", msg)
            continue
        parts.append(
            f"[{msg.get('role', 'unknown').upper()}]: {msg.get('content', '')}"
        )

    return "\n\n".join(parts) if parts else "No conversation history available."
