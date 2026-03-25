"""
Planner Agent — data source selection via LLM tool calling.

Determines which data sources to query using document metadata similarity
search for context, then LLM-based selection.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from ..connections.llm import execute_llm_call
from ..connections.postgres import get_database_session, get_database_schema
from ..utils.config import config
from ..utils.prompt_loader import get_ordered_data_source_keys, get_prompt

MODEL_CAPABILITY = "large"
MODEL_MAX_TOKENS = 4096
MODEL_TEMPERATURE = 0.0
PLANNER_TOOL_NAME = "select_data_sources"
MAX_CONTEXT_DOCUMENTS = 5
METADATA_SEARCH_TOP_K = 5
EMBEDDING_DIMENSIONS = 3072

logger = logging.getLogger(__name__)


class PlannerError(Exception):
    """Exception raised for planner-related errors."""


# --- Document Metadata Search Functions ---


def _generate_query_embedding_vector(
    query: str, token: Optional[str] = None
) -> Tuple[Optional[List[float]], Optional[Dict[str, Any]]]:
    """Generate an embedding for the query string."""
    logger.info("Generating embedding for query: '%s...'", query[:100])
    usage_details = None

    try:
        if not token:
            raise ValueError("Token required for embedding API call")

        model_config = config.get_model_settings("embedding")
        model_name = model_config["name"]
        prompt_cost = model_config["prompt_token_cost"]
        completion_cost = model_config.get("completion_token_cost", 0.0)

        call_params = {
            "oauth_token": token,
            "prompt_token_cost": prompt_cost,
            "completion_token_cost": completion_cost,
            "model": model_name,
            "input": [query],
            "dimensions": EMBEDDING_DIMENSIONS,
            "database_name": "document_metadata_search",
            "is_embedding": True,
        }

        result = execute_llm_call(**call_params)

        response = None
        if isinstance(result, tuple) and len(result) == 2:
            response, usage_details = result
            if usage_details:
                logger.debug("Embedding usage details: %s", usage_details)
        else:
            response = result
            logger.debug("execute_llm_call did not return usage_details")

        if (
            response
            and hasattr(response, "data")
            and response.data
            and hasattr(response.data[0], "embedding")
            and response.data[0].embedding
        ):
            logger.info("Embedding generated successfully.")
            return response.data[0].embedding, usage_details

        logger.error(
            "No embedding data received from API.",
            extra={"api_response": response},
        )
        return None, usage_details

    except (ValueError, TypeError, KeyError) as exc:
        logger.error(
            "Embedding generation failed (non-retryable): %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return None, usage_details


def _search_document_metadata_by_embedding(
    research_statement: str,
    token: Optional[str] = None,
    available_data_sources: Optional[Dict[str, Any]] = None,
) -> Tuple[
    List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[List[float]]
]:
    """Search metadata embeddings to find relevant documents.

    Searches across all documents in the selected data sources without
    subfolder filters — filter resolution happens after source selection.

    Args:
        research_statement: Research statement to search for.
        token: OAuth token or API key for authentication.
        available_data_sources: Data sources to filter by.

    Returns:
        Tuple of (matching documents, usage details, query embedding).
    """
    logger.info(
        "Searching document metadata for research statement: '%s...'",
        research_statement[:100],
    )
    usage_details = None

    try:
        query_embedding, usage_details = _generate_query_embedding_vector(
            research_statement, token
        )

        if query_embedding is None:
            logger.error(
                "Could not generate embedding for research statement"
            )
            return [], usage_details, None

        embedding_str = json.dumps(query_embedding, separators=(",", ":"))

        if not available_data_sources:
            logger.warning(
                "No available_data_sources provided - cannot search metadata"
            )
            return [], usage_details, query_embedding

        schema = get_database_schema()

        with get_database_session() as session:
            ds_keys = sorted(available_data_sources.keys())
            in_placeholders = ", ".join(
                [f":ds_{i}" for i in range(len(ds_keys))]
            )

            sql = text(
                f"""
                SELECT
                    m.data_source,
                    m.title,
                    m.document_summary,
                    m.filetype,
                    1 - (m.summary_embedding <=> CAST(:embedding AS vector))
                        AS similarity_score
                FROM {schema}.documents m
                WHERE m.summary_embedding IS NOT NULL
                    AND m.data_source IN ({in_placeholders})
                ORDER BY similarity_score DESC
                LIMIT :top_k
            """
            )
            params: Dict[str, Any] = {
                "embedding": embedding_str,
                "top_k": METADATA_SEARCH_TOP_K,
            }
            for i, ds_key in enumerate(ds_keys):
                params[f"ds_{i}"] = ds_key

            result = session.execute(sql, params)
            results_raw = result.mappings().all()

            results = [
                {**dict(row), "rank": i + 1}
                for i, row in enumerate(results_raw)
            ]

            logger.info(
                "Found %d matching documents in %s.documents",
                len(results),
                schema,
            )

            return results, usage_details, query_embedding

    except (ValueError, TypeError, RuntimeError) as exc:
        logger.error(
            "Error searching document metadata: %s", exc, exc_info=True
        )
        return [], usage_details, None


# --- Model Configuration ---


def _get_model_settings() -> Dict[str, Any]:
    """Get model settings from config based on capability tier."""
    model_config = config.get_model_settings(MODEL_CAPABILITY)
    # Ensure completion_token_cost has a default
    model_config.setdefault("completion_token_cost", 0.0)
    return model_config


def _inject_data_source_index_constraints(
    tool_definition: Dict[str, Any],
    available_data_sources: Dict[str, Any],
) -> Dict[str, Any]:
    """Inject data source count constraints into tool definition.

    Raises:
        PlannerError: If tool definition structure is invalid.
    """
    import copy

    tool = copy.deepcopy(tool_definition)
    max_sources = config.MAX_DATA_SOURCES_PER_QUERY
    ds_count = len(available_data_sources)

    try:
        params = tool["function"]["parameters"]["properties"][
            "data_sources"
        ]
        params["items"]["maximum"] = ds_count - 1
        params["maxItems"] = max_sources
    except (KeyError, TypeError) as exc:
        raise PlannerError(
            f"Invalid tool definition structure in database: {exc}"
        ) from exc

    return tool


def _format_document_metadata_context(
    document_metadata_context: Optional[List[Dict[str, Any]]],
    available_data_sources: Optional[Dict[str, Any]] = None,
) -> str:
    """Format document metadata context into a string for the user message."""
    if not document_metadata_context:
        return ""

    ds_to_index = {}
    if available_data_sources:
        ordered_keys = get_ordered_data_source_keys(available_data_sources)
        ds_to_index = {key: idx for idx, key in enumerate(ordered_keys)}

    context_parts = ["<RELEVANT_DOCUMENTS_CONTEXT>"]

    for i, doc in enumerate(
        document_metadata_context[:MAX_CONTEXT_DOCUMENTS], 1
    ):
        data_source = doc.get("data_source", "Unknown")
        ds_index = ds_to_index.get(data_source, "?")
        summary = doc.get("document_summary", "No summary available")
        similarity = doc.get("similarity_score", 0.0)
        context_parts.append(
            f"{i}. DATA_SOURCE: {data_source} (INDEX={ds_index})\n"
            f"   Document: {doc.get('title', 'Unknown')}\n"
            f"   Summary: {summary}\n"
            f"   Similarity: {similarity:.3f}"
        )

    context_parts.append("</RELEVANT_DOCUMENTS_CONTEXT>")
    return "\n\n".join(context_parts)


def _build_planner_user_message(
    user_prompt_template: str,
    research_statement: str,
    document_metadata_context: Optional[List[Dict[str, Any]]],
    available_data_sources: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the user message content for the planner LLM call.

    Raises:
        PlannerError: If user_prompt_template is missing.
    """
    if not user_prompt_template:
        raise PlannerError(
            "user_prompt not found in database for agent/planner. "
            "Please ensure the prompt is configured in the prompts table."
        )

    doc_context = _format_document_metadata_context(
        document_metadata_context, available_data_sources
    )

    user_content = user_prompt_template.replace(
        "{{research_statement}}", research_statement
    )
    user_content = user_content.replace(
        "{{document_metadata_context}}", doc_context
    )

    return user_content


def _parse_planner_tool_response(response: Any) -> Dict[str, Any]:
    """Extract and validate tool call arguments from LLM response.

    Raises:
        PlannerError: If response is invalid or tool call parsing fails.
    """
    if (
        not response
        or not hasattr(response, "choices")
        or not response.choices
    ):
        raise PlannerError("Invalid or empty response received from LLM")

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) if message else None
    if not tool_calls:
        content_returned = (
            message.content if message and message.content else "No content"
        )
        logger.warning(
            "Expected tool call but received content: %s...",
            content_returned[:100],
        )
        raise PlannerError(
            "No tool call received in response, content returned instead."
        )

    tool_call = tool_calls[0]

    if tool_call.function.name != PLANNER_TOOL_NAME:
        raise PlannerError(
            f"Unexpected function call: {tool_call.function.name}"
        )

    arguments = getattr(tool_call.function, "arguments", None)
    if isinstance(arguments, dict):
        return arguments

    if not isinstance(arguments, str):
        raise PlannerError(
            f"Invalid tool arguments type: {type(arguments)}"
        )

    try:
        return json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise PlannerError(
            f"Invalid JSON in tool arguments: {arguments}"
        ) from exc


def _validate_selected_data_source_list(
    arguments: Dict[str, Any],
    available_data_sources: Dict[str, Any],
) -> List[str]:
    """Validate data source indices and map them back to names.

    Raises:
        PlannerError: If indices are invalid types or out of range.
    """
    selected_indices = arguments.get("data_sources", [])

    if not selected_indices:
        logger.warning("LLM returned empty data source selection")
        return []

    ds_keys = get_ordered_data_source_keys(available_data_sources)

    validated = []
    for i, idx in enumerate(selected_indices):
        if not isinstance(idx, int):
            raise PlannerError(
                f"Data source index {i + 1} is not an integer: {idx}"
            )
        if idx < 0 or idx >= len(ds_keys):
            raise PlannerError(
                f"Data source index {i + 1} out of range: {idx} "
                f"(valid: 0-{len(ds_keys) - 1})"
            )
        validated.append(ds_keys[idx])

    return validated


def _load_planner_prompt_components(
    available_data_sources: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Get prompt components for the planner.

    Raises:
        PlannerError: If tool definition is not found.
    """
    system_prompt, tools, user_prompt_template = get_prompt(
        "agent",
        "planner",
        inject_fiscal=True,
        inject_data_sources=True,
        available_data_sources=available_data_sources,
    )

    if not tools:
        raise PlannerError(
            "tool_definition not found in database for agent/planner. "
            "Please ensure the prompt is configured in the prompts table."
        )

    tool_with_enum = _inject_data_source_index_constraints(
        tools[0], available_data_sources
    )

    return system_prompt, [tool_with_enum], user_prompt_template


def _execute_planner_llm_call(
    token: str,
    messages: List[Dict[str, str]],
    tool_definitions: List[Dict[str, Any]],
    model_settings: Dict[str, Any],
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Make the LLM call for data source selection."""
    return execute_llm_call(
        oauth_token=token,
        model=model_settings["name"],
        messages=messages,
        max_tokens=MODEL_MAX_TOKENS,
        temperature=MODEL_TEMPERATURE,
        tools=tool_definitions,
        tool_choice={
            "type": "function",
            "function": {"name": PLANNER_TOOL_NAME},
        },
        stream=False,
        prompt_token_cost=model_settings["prompt_token_cost"],
        completion_token_cost=model_settings["completion_token_cost"],
        reasoning_effort=model_settings.get("reasoning_effort"),
    )


def generate_data_source_selection_plan(
    research_statement: str,
    token: str,
    available_data_sources: Dict[str, Any],
    process_monitor: Optional[Any] = None,
    filters: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Create a plan of selected data sources based on a research statement.

    Performs document metadata similarity search, then uses LLM tool calling
    to select which data sources should be queried.

    Args:
        research_statement: Research statement from the clarifier.
        token: Authentication token for API access.
        available_data_sources: Data source configurations.
        process_monitor: Optional process monitor for substage timing.
        filters: Optional filter constraints (filter_1, filter_2, filter_3).

    Returns:
        Data source selection plan (with ``data_sources`` and
        ``query_embedding``) and usage details.

    Raises:
        PlannerError: If there is an error creating the selection plan.
    """
    usage_details_list: List[Dict[str, Any]] = []

    try:
        if not available_data_sources:
            raise PlannerError(
                "No data sources provided for planner selection."
            )

        # Step 1: Generate embedding and search document metadata
        if process_monitor:
            process_monitor.start_stage("planner_embedding")

        logger.info("Searching document metadata for planner context...")
        metadata_results, embedding_usage, query_embedding = (
            _search_document_metadata_by_embedding(
                research_statement, token, available_data_sources
            )
        )

        if process_monitor:
            process_monitor.end_stage("planner_embedding")
            if embedding_usage:
                process_monitor.add_llm_call_details_to_stage(
                    "planner_embedding", embedding_usage
                )
            process_monitor.add_stage_details(
                "planner_embedding",
                documents_found=(
                    len(metadata_results) if metadata_results else 0
                ),
            )

        if embedding_usage:
            usage_details_list.append(embedding_usage)

        if metadata_results:
            logger.info(
                "Found %d relevant documents in metadata",
                len(metadata_results),
            )
            for doc in metadata_results:
                logger.info(
                    "  -> [%s] %s (similarity: %.4f)",
                    doc.get("data_source", "?"),
                    doc.get("title", "?"),
                    doc.get("similarity_score", 0.0),
                )
        else:
            logger.info("No relevant documents found in metadata")

        # Step 2: Build prompts and call LLM for data source selection
        if process_monitor:
            process_monitor.start_stage("planner_llm_selection")

        system_prompt, tool_definitions, user_prompt_template = (
            _load_planner_prompt_components(available_data_sources)
        )
        model_settings = _get_model_settings()
        user_content = _build_planner_user_message(
            user_prompt_template,
            research_statement,
            metadata_results,
            available_data_sources,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response, llm_usage = _execute_planner_llm_call(
            token, messages, tool_definitions, model_settings
        )

        if process_monitor:
            process_monitor.end_stage("planner_llm_selection")
            if llm_usage:
                process_monitor.add_llm_call_details_to_stage(
                    "planner_llm_selection", llm_usage
                )

        if llm_usage:
            usage_details_list.append(llm_usage)

        arguments = _parse_planner_tool_response(response)
        validated = _validate_selected_data_source_list(
            arguments, available_data_sources
        )

        if process_monitor:
            process_monitor.add_stage_details(
                "planner_llm_selection",
                data_sources_selected=validated,
            )

        ds_keys = get_ordered_data_source_keys(available_data_sources)
        logger.info(
            "Data source index mapping: %s",
            dict(enumerate(ds_keys)),
        )
        logger.info(
            "LLM selected indices: %s",
            arguments.get("data_sources", []),
        )
        logger.info(
            "Data source selection plan created with %d sources: %s",
            len(validated),
            validated,
        )

        return {
            "data_sources": validated,
            "query_embedding": query_embedding,
        }, usage_details_list

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        logger.error(
            "Error creating data source selection plan: %s",
            str(exc),
            exc_info=True,
        )
        raise PlannerError(
            f"Failed to create data source selection plan: {exc}"
        ) from exc
