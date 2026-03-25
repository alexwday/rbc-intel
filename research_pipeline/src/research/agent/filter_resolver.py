"""
Filter Resolver Agent — resolves per-source subfolder filters after data
source selection.

After the planner selects which data sources to query, the filter resolver
makes a single LLM call covering all selected sources that have filter
metadata configured in the registry.  For each source it either:

- Auto-resolves filter values from the research statement context
- Generates a clarification question for the user

If the user pre-provided filter values in the API request, those are used
directly and the LLM call is skipped for that source.

The resolver receives filter labels, descriptions, and available values
from the registry and documents table so the LLM can make informed
decisions.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..connections.llm import execute_llm_call
from ..utils.config import config
from ..utils.prompt_loader import get_prompt
from .tools.database_metadata import (
    fetch_filter_values_for_source,
    get_filter_metadata_for_sources,
)

MODEL_CAPABILITY = "small"
MODEL_MAX_TOKENS = 4096
MODEL_TEMPERATURE = 0.0

logger = logging.getLogger(__name__)


class FilterResolverError(Exception):
    """Exception raised for filter resolver errors."""


def _get_model_settings() -> Dict[str, Any]:
    """Return model settings for the filter resolver LLM."""
    return config.get_model_settings(MODEL_CAPABILITY)


def _needs_resolution(
    selected_sources: List[str],
    filter_metadata: Dict[str, Dict[str, Any]],
    pre_provided_filters: Optional[Dict[str, Dict[str, str]]],
) -> Dict[str, Dict[str, Any]]:
    """Determine which sources still need filter resolution.

    Args:
        selected_sources: Data sources selected by the planner.
        filter_metadata: Per-source filter metadata from registry.
        pre_provided_filters: Filters already provided by the API caller,
            keyed by data_source → {filter_1: val, ...}.

    Returns:
        Dict of sources needing resolution, with their filter metadata
        and available values.  Sources fully covered by pre_provided_filters
        are excluded.
    """
    needs_resolve: Dict[str, Dict[str, Any]] = {}

    for ds_name in selected_sources:
        if ds_name not in filter_metadata:
            continue

        ds_filters = filter_metadata[ds_name]["filters"]
        pre = (pre_provided_filters or {}).get(ds_name, {})

        # Check if all configured filter levels are pre-provided
        all_provided = all(
            pre.get(fkey) for fkey in ds_filters
        )
        if all_provided:
            continue

        # Fetch available values for this source
        available_values = fetch_filter_values_for_source(ds_name)

        needs_resolve[ds_name] = {
            "display_name": filter_metadata[ds_name]["display_name"],
            "filters": ds_filters,
            "available_values": available_values,
            "pre_provided": pre,
        }

    return needs_resolve


def _build_filter_context_xml(
    sources_to_resolve: Dict[str, Dict[str, Any]],
) -> str:
    """Build XML context describing sources and their filters for the LLM."""
    lines = ["<DATA_SOURCES_WITH_FILTERS>"]

    for ds_name, ds_info in sources_to_resolve.items():
        lines.append(
            f'<DATA_SOURCE id="{ds_name}" '
            f'name="{ds_info["display_name"]}">'
        )

        for fkey, fmeta in ds_info["filters"].items():
            pre_val = ds_info["pre_provided"].get(fkey, "")
            available = ds_info["available_values"].get(fkey, [])

            lines.append(f"  <FILTER key=\"{fkey}\">")
            lines.append(f"    <LABEL>{fmeta['label']}</LABEL>")
            lines.append(
                f"    <DESCRIPTION>{fmeta['description']}</DESCRIPTION>"
            )
            if available:
                lines.append(
                    f"    <AVAILABLE_VALUES>"
                    f"{', '.join(available)}"
                    f"</AVAILABLE_VALUES>"
                )
            if pre_val:
                lines.append(
                    f"    <PRE_PROVIDED_VALUE>{pre_val}</PRE_PROVIDED_VALUE>"
                )
            lines.append("  </FILTER>")

        lines.append("</DATA_SOURCE>")

    lines.append("</DATA_SOURCES_WITH_FILTERS>")
    return "\n".join(lines)


def _parse_filter_resolver_response(response: Any) -> Dict[str, Any]:
    """Extract and validate tool call arguments from the LLM response.

    Raises:
        FilterResolverError: If the response is invalid or parsing fails.
    """
    if (
        not response
        or not hasattr(response, "choices")
        or not response.choices
    ):
        raise FilterResolverError(
            "Invalid or empty response received from LLM"
        )

    message = getattr(response.choices[0], "message", None)
    tool_calls = getattr(message, "tool_calls", None)
    if not message or not tool_calls:
        content_returned = (
            message.content if message and message.content else "No content"
        )
        logger.warning(
            "Expected tool call but received content: %s...",
            content_returned[:200],
        )
        raise FilterResolverError(
            "No tool call received in response, content returned instead."
        )

    tool_call = tool_calls[0]

    if tool_call.function.name != "resolve_filters":
        raise FilterResolverError(
            f"Unexpected function call: {tool_call.function.name}"
        )

    try:
        return json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        raise FilterResolverError(
            f"Invalid JSON in tool arguments: "
            f"{tool_call.function.arguments}"
        ) from exc


class FilterResolution:
    """Result of the filter resolution step."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self.resolved_filters: Dict[str, Dict[str, str]] = {}
        self.needs_clarification: bool = False
        self.clarification_message: str = ""

    def get_filters_for_source(
        self, data_source: str
    ) -> Optional[Dict[str, str]]:
        """Return resolved filters for a specific data source, or None."""
        return self.resolved_filters.get(data_source)


def resolve_filters(
    research_statement: str,
    selected_sources: List[str],
    token: str,
    pre_provided_filters: Optional[Dict[str, Dict[str, str]]] = None,
    process_monitor: Optional[Any] = None,
) -> Tuple[FilterResolution, Optional[Dict[str, Any]]]:
    """Resolve per-source subfolder filters for the selected data sources.

    Makes a single LLM call covering all sources that need resolution.
    Sources with no filter metadata or with fully pre-provided filters
    are skipped.

    Args:
        research_statement: The research statement from the clarifier.
        selected_sources: Data sources selected by the planner.
        token: Authentication token for LLM access.
        pre_provided_filters: Filters already provided by the API caller,
            keyed by data_source → {filter_1: val, ...}.
        process_monitor: Optional process monitor for instrumentation.

    Returns:
        Tuple of (FilterResolution, usage_details).

    Raises:
        FilterResolverError: If filter resolution fails.
    """
    resolution = FilterResolution()
    usage_details = None

    # Copy pre-provided filters into the resolution
    if pre_provided_filters:
        for ds_name, ds_filters in pre_provided_filters.items():
            if ds_name in selected_sources:
                resolution.resolved_filters[ds_name] = dict(ds_filters)

    # Check which sources need LLM resolution
    filter_metadata = get_filter_metadata_for_sources(selected_sources)

    if not filter_metadata:
        logger.info("No sources have filters configured — skipping resolution")
        return resolution, usage_details

    sources_to_resolve = _needs_resolution(
        selected_sources, filter_metadata, pre_provided_filters
    )

    if not sources_to_resolve:
        logger.info(
            "All filters pre-provided — skipping LLM resolution"
        )
        return resolution, usage_details

    logger.info(
        "Resolving filters for %d source(s): %s",
        len(sources_to_resolve),
        list(sources_to_resolve.keys()),
    )

    if process_monitor:
        process_monitor.start_stage("filter_resolver")

    try:
        system_prompt, tools, user_prompt_template = get_prompt(
            "agent",
            "filter_resolver",
            inject_fiscal=True,
        )

        if not user_prompt_template:
            raise FilterResolverError(
                "user_prompt not found for agent/filter_resolver"
            )

        filter_context = _build_filter_context_xml(sources_to_resolve)

        user_content = user_prompt_template.replace(
            "{{research_statement}}", research_statement
        ).replace("{{filter_context}}", filter_context)

        model_settings = _get_model_settings()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response, usage_details = execute_llm_call(
            oauth_token=token,
            model=model_settings["name"],
            messages=messages,
            max_tokens=MODEL_MAX_TOKENS,
            temperature=MODEL_TEMPERATURE,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "resolve_filters"},
            },
            stream=False,
            prompt_token_cost=model_settings["prompt_token_cost"],
            completion_token_cost=model_settings["completion_token_cost"],
            reasoning_effort=model_settings.get("reasoning_effort"),
        )

        arguments = _parse_filter_resolver_response(response)

        # Process the LLM's decisions
        action = arguments.get("action", "apply_filters")
        source_decisions = arguments.get("source_filters", [])

        if action == "ask_user":
            resolution.needs_clarification = True
            resolution.clarification_message = arguments.get(
                "clarification_message", ""
            )
            logger.info(
                "Filter resolver needs user clarification: %s",
                resolution.clarification_message[:200],
            )
        else:
            for decision in source_decisions:
                ds_name = decision.get("data_source", "")
                if ds_name not in selected_sources:
                    continue

                ds_filters = {}
                for fkey in ("filter_1", "filter_2", "filter_3"):
                    val = decision.get(fkey)
                    if val:
                        ds_filters[fkey] = val

                if ds_filters:
                    existing = resolution.resolved_filters.get(ds_name, {})
                    existing.update(ds_filters)
                    resolution.resolved_filters[ds_name] = existing

            logger.info(
                "Filter resolver resolved filters: %s",
                resolution.resolved_filters,
            )

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        logger.error(
            "Error resolving filters: %s", str(exc), exc_info=True
        )
        raise FilterResolverError(
            f"Failed to resolve filters: {exc}"
        ) from exc
    finally:
        if process_monitor:
            process_monitor.end_stage("filter_resolver")
            if usage_details:
                process_monitor.add_llm_call_details_to_stage(
                    "filter_resolver", usage_details
                )

    return resolution, usage_details
