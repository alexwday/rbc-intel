"""Summarizer agent for generating final research summaries."""

import json
import logging
from typing import Any, Dict, Generator, List, Optional

from ..connections.llm import execute_llm_call
from ..utils.config import config
from ..utils.prompt_loader import get_prompt

MODEL_CAPABILITY = "large"
MODEL_MAX_TOKENS = 16384
MODEL_TEMPERATURE = 0.0

logger = logging.getLogger(__name__)


class SummarizerError(Exception):
    """Exception raised for summarizer-related errors."""


def _get_model_settings() -> Dict[str, Any]:
    """Return model settings pulled from the configured capability tier."""
    return config.get_model_settings(MODEL_CAPABILITY)


def _format_aggregated_research_context(
    aggregated_detailed_research: Dict[str, str],
    available_data_sources: Dict[str, Any],
) -> str:
    """Format aggregated research findings into a context string."""
    lines = ["Aggregated Detailed Research Findings:", ""]

    if not aggregated_detailed_research:
        lines.append(
            "No detailed research findings were provided or generated."
        )
        return "\n".join(lines)

    for ds_name, research_text in (
        aggregated_detailed_research or {}
    ).items():
        ds_display_name = available_data_sources.get(ds_name, {}).get(
            "display_name", ds_name
        )
        lines.append(f"=== Findings from: {ds_display_name} ===")
        lines.append(research_text or "")
        lines.append("")

    return "\n".join(lines).rstrip()


def _format_reference_index_context(
    reference_index: Dict[str, Dict[str, Any]],
) -> str:
    """Format the reference index into a context string."""
    lines = ["Available References:"]

    for ref_id, ref_data in reference_index.items():
        doc_name = ref_data.get("doc_name", "Unknown")
        page = ref_data.get("page", 1)
        lines.append(f"[REF:{ref_id}] = {doc_name} - Page {page}")

    return "\n".join(lines)


def _build_user_message(
    user_prompt_template: str,
    research_statement: Optional[str],
) -> str:
    """Build the user message content for the summary request.

    Raises:
        SummarizerError: If user_prompt_template is not provided.
    """
    if not user_prompt_template:
        raise SummarizerError(
            "user_prompt not found in database for agent/summarizer. "
            "Please ensure the prompt is configured in the prompts table."
        )

    user_content = user_prompt_template.replace(
        "{{research_statement}}",
        research_statement or "No specific research query",
    )

    return user_content


def _build_messages(
    system_prompt: str,
    user_prompt_template: str,
    aggregated_detailed_research: Dict[str, str],
    available_data_sources: Dict[str, Any],
    summary_context: Optional[Dict[str, Any]],
) -> list:
    """Build the complete messages list for the LLM call."""
    messages = [{"role": "system", "content": system_prompt}]

    research_context = _format_aggregated_research_context(
        aggregated_detailed_research, available_data_sources
    )
    messages.append({"role": "system", "content": research_context})

    logger.debug(
        "RESEARCH_INPUT [summarizer]: Aggregated research context:\n%s",
        (
            research_context[:20000]
            if len(research_context) > 20000
            else research_context
        ),
    )

    if summary_context:
        reference_index = summary_context.get("reference_index")
        if reference_index:
            ref_context = _format_reference_index_context(reference_index)
            messages.append({"role": "system", "content": ref_context})
            logger.debug(
                "RESEARCH_INPUT [summarizer]: Reference index:\n%s",
                json.dumps(reference_index, indent=2, default=str),
            )

        research_statement = summary_context.get("research_statement")
    else:
        research_statement = None

    user_content = _build_user_message(
        user_prompt_template, research_statement
    )
    messages.append({"role": "user", "content": user_content})

    return messages


def stream_research_summary(
    aggregated_detailed_research: Dict[str, str],
    token: Optional[str],
    available_data_sources: Dict[str, Any],
    summary_context: Optional[Dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """Stream the final response based on aggregated detailed research.

    Yields:
        str | dict: Streaming content chunks followed by usage details.

    Raises:
        SummarizerError: If there is an error generating the summary.
    """
    final_usage_details = None

    try:
        model_settings = _get_model_settings()
        system_prompt, _, user_prompt_template = get_prompt(
            "agent",
            "summarizer",
            inject_fiscal=True,
            inject_data_sources=True,
            available_data_sources=available_data_sources,
        )

        messages = _build_messages(
            system_prompt,
            user_prompt_template,
            aggregated_detailed_research,
            available_data_sources,
            summary_context,
        )

        llm_stream = execute_llm_call(
            oauth_token=token,
            model=model_settings["name"],
            messages=messages,
            max_tokens=MODEL_MAX_TOKENS,
            temperature=MODEL_TEMPERATURE,
            stream=True,
            prompt_token_cost=model_settings["prompt_token_cost"],
            completion_token_cost=model_settings["completion_token_cost"],
            reasoning_effort=model_settings.get("reasoning_effort"),
        )

        collected_output: List[str] = []

        for item in llm_stream:
            if isinstance(item, dict) and "usage_details" in item:
                final_usage_details = item
                continue

            choices = getattr(item, "choices", None)
            if choices:
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    collected_output.append(content)
                    yield content

        logger.debug(
            "RESEARCH_OUTPUT [summarizer]: Final summary output:\n%s",
            "".join(collected_output),
        )

        if final_usage_details:
            yield final_usage_details
        else:
            logger.warning("Usage details not found in summary stream")
            yield {
                "usage_details": {
                    "error": "Usage data missing from stream"
                }
            }

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        logger.error(
            "Error generating streaming summary: %s",
            str(exc),
            exc_info=True,
        )
        raise SummarizerError(
            f"Failed to generate streaming summary: {exc}"
        ) from exc
