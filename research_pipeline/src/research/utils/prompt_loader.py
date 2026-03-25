"""
Prompt Loader - Database-backed prompt management with context injection.

Fetches versioned prompts from PostgreSQL at startup, caches them in memory,
and supports runtime context injection for fiscal dates and data source
availability. Used by all agents during initialization.

INDEXING CONVENTION (applies across the research pipeline):
- Data source selection (planner): 0-indexed (LLM tool call convention for arrays)
- Document lists (metadata): 1-indexed (human-readable prompts shown to LLM)
- User-facing references: 1-indexed (intuitive for end users, e.g., [REF:1])
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..connections.postgres import get_database_session, get_database_schema
from .fiscal_context import generate_fiscal_context_statement

logger = logging.getLogger(__name__)

_prompt_cache: Dict[str, Dict[str, Any]] = {}


def get_ordered_data_source_keys(
    available_data_sources: Dict[str, Any],
) -> List[str]:
    """Return data source keys in consistent sorted order.

    Args:
        available_data_sources: Data source configurations keyed by data_source.

    Returns:
        List of data source keys in stable order for index assignment.
    """
    return sorted(available_data_sources.keys())


def load_all_prompts(model: str = "research") -> Tuple[int, List[str]]:
    """Pre-warm the prompt cache by loading all prompts for a model namespace.

    Args:
        model: Model namespace to load (e.g., "research").

    Returns:
        Tuple of (count of prompts loaded, list of "layer/name" identifiers).
    """
    schema = get_database_schema()
    try:
        with get_database_session() as session:
            result = session.execute(
                text(
                    f"""
                    SELECT DISTINCT ON (layer, name)
                        layer, name, system_prompt, user_prompt,
                        tool_definition, description
                    FROM {schema}.prompts
                    WHERE model = :model
                    ORDER BY layer, name, version DESC
                    """
                ),
                {"model": model},
            )
            rows = result.fetchall()

            count = 0
            loaded_prompts = []
            for row in rows:
                layer, name, system_prompt, user_prompt, tool_definition, description = (
                    row
                )
                cache_key = f"{model}/{layer}/{name}"
                _prompt_cache[cache_key] = {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "tool_definition": tool_definition,
                    "description": description,
                }
                loaded_prompts.append(f"{layer}/{name}")
                count += 1

            logger.info(
                "Loaded %d prompts for model '%s' into cache", count, model
            )
            return count, loaded_prompts

    except SQLAlchemyError as exc:
        logger.error("Error loading prompts for model %s: %s", model, exc)
        return 0, []


def get_prompt(
    layer: str,
    name: str,
    model: str = "research",
    inject_fiscal: bool = False,
    inject_data_sources: bool = False,
    available_data_sources: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Any], str]:
    """Retrieve a cached prompt with optional fiscal and data source context injection.

    Args:
        layer: Prompt layer (e.g., "agent", "subagent").
        name: Prompt identifier (e.g., "router", "clarifier").
        model: Model namespace for prompt lookup.
        inject_fiscal: Replace {{FISCAL_CONTEXT}} with current fiscal period.
        inject_data_sources: Replace {{DATA_SOURCE_CONTEXT}} with available sources.
        available_data_sources: Source configs to inject; fetched if None.

    Returns:
        Tuple of (system_prompt, tools_list, user_prompt).

    Raises:
        ValueError: If the requested prompt is not found in cache.
    """
    cache_key = f"{model}/{layer}/{name}"

    if cache_key not in _prompt_cache:
        if not any(k.startswith(f"{model}/") for k in _prompt_cache):
            load_all_prompts(model)

    if cache_key not in _prompt_cache:
        raise ValueError(f"Prompt not found: {cache_key}")

    prompt = _prompt_cache[cache_key].copy()

    system_prompt = prompt.get("system_prompt", "")
    tool_definition = prompt.get("tool_definition")
    user_prompt = prompt.get("user_prompt", "")

    system_prompt = _inject_context(
        system_prompt, inject_fiscal, inject_data_sources, available_data_sources
    )
    user_prompt = _inject_context(
        user_prompt, inject_fiscal, inject_data_sources, available_data_sources
    )

    tools = [tool_definition] if tool_definition else []

    return system_prompt, tools, user_prompt


def _inject_context(
    prompt_text: str,
    inject_fiscal: bool,
    inject_data_sources: bool,
    available_data_sources: Optional[Dict[str, Any]],
) -> str:
    """Replace {{FISCAL_CONTEXT}}, {{DATA_SOURCE_CONTEXT}}, and {{MAX_DATA_SOURCES}}."""
    if inject_fiscal and "{{FISCAL_CONTEXT}}" in prompt_text:
        prompt_text = prompt_text.replace(
            "{{FISCAL_CONTEXT}}", generate_fiscal_context_statement()
        )

    if inject_data_sources and "{{DATA_SOURCE_CONTEXT}}" in prompt_text:
        ds_statement = _format_data_source_context_block(available_data_sources)
        prompt_text = prompt_text.replace("{{DATA_SOURCE_CONTEXT}}", ds_statement)

    if "{{MAX_DATA_SOURCES}}" in prompt_text:
        from .config import config

        prompt_text = prompt_text.replace(
            "{{MAX_DATA_SOURCES}}", str(config.MAX_DATA_SOURCES_PER_QUERY)
        )

    return prompt_text


def _format_data_source_context_block(
    available_data_sources: Optional[Dict[str, Any]],
) -> str:
    """Build XML block describing available data sources with index attributes."""
    if available_data_sources is None:
        try:
            from ..agent.tools.database_metadata import fetch_available_data_sources

            available_data_sources = fetch_available_data_sources()
        except ImportError:
            logger.warning("database_metadata module not available")
            return (
                "<AVAILABLE_DATA_SOURCES>\n"
                "Data source information not available.\n"
                "</AVAILABLE_DATA_SOURCES>"
            )

    if not available_data_sources:
        return (
            "<AVAILABLE_DATA_SOURCES>\n"
            "No data sources available.\n"
            "</AVAILABLE_DATA_SOURCES>"
        )

    ordered_keys = get_ordered_data_source_keys(available_data_sources)
    key_to_index = {key: idx for idx, key in enumerate(ordered_keys)}

    lines: List[str] = ["<AVAILABLE_DATA_SOURCES>"]

    for ds_key in ordered_keys:
        ds_info = available_data_sources[ds_key]
        idx = key_to_index[ds_key]
        name = ds_info.get("display_name", ds_key)
        description = (
            ds_info.get("source_description") or ds_info.get("description", "")
        )
        lines.extend(
            [
                f'<DATA_SOURCE index="{idx}" id="{ds_key}">',
                f"  <NAME>{name}</NAME>",
                f"  <DESCRIPTION>{description}</DESCRIPTION>",
                "</DATA_SOURCE>",
            ]
        )

    lines.append("</AVAILABLE_DATA_SOURCES>")
    return "\n".join(lines)
