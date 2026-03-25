"""Helpers for context-safe dense-table description generation."""

import json
from pathlib import Path
from typing import Any, Callable

import openai

from ...utils.config import get_dense_table_description_max_prompt_tokens
from ...utils.prompt_loader import load_prompt
from .content_chunker import count_tokens
from .types import DenseTableDescription

_DENSE_TABLE_ERRORS = (
    json.JSONDecodeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
    openai.OpenAIError,
)
_XLSX_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _build_prompt_messages(
    prompt: dict[str, Any], user_content: str
) -> list[dict[str, str]]:
    """Build prompt messages for an LLM tool-calling request."""
    messages: list[dict[str, str]] = []
    system_prompt = prompt.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": f"{prompt['user_prompt']}\n\n{user_content}",
        }
    )
    return messages


def _estimate_prompt_tokens(prompt: dict[str, Any], user_content: str) -> int:
    """Estimate prompt tokens from rendered message content."""
    messages = _build_prompt_messages(prompt, user_content)
    return sum(count_tokens(message["content"]) for message in messages)


def _build_column_index(eda: Any) -> dict[str, Any]:
    """Build a stable position-keyed column lookup."""
    return {col.position: col for col in eda.columns}


def _build_column_name_index(eda: Any) -> dict[str, Any]:
    """Build a name-keyed column lookup for backward compatibility."""
    return {col.name: col for col in eda.columns}


def _resolve_column_reference(
    column_index: dict[str, Any],
    column_name_index: dict[str, Any],
    key: str,
) -> Any | None:
    """Resolve a column reference by position first, then by name."""
    if key in column_index:
        return column_index[key]
    return column_name_index.get(key)


def _build_description_lookup(
    description: DenseTableDescription,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build position and name lookups for column descriptions."""
    by_position: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for column_description in description.column_descriptions:
        position = str(column_description.get("position", "")).strip()
        name = str(column_description.get("name", "")).strip()
        text = str(column_description.get("description", "")).strip()
        if position:
            by_position[position] = text
        if name:
            by_name[name] = text
    return by_position, by_name


def _lookup_column_description(
    col: Any,
    desc_by_position: dict[str, str],
    desc_by_name: dict[str, str],
) -> str:
    """Look up a column description by position first, then by name."""
    return desc_by_position.get(col.position, desc_by_name.get(col.name, ""))


def _parse_batch_summary_response(
    response: dict[str, Any],
) -> tuple[str, list[str]]:
    """Parse the LLM dense-table merge response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message")

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("LLM response missing tool calls")

    function_data = tool_calls[0].get("function")
    if not isinstance(function_data, dict):
        raise ValueError("LLM response missing function data")

    arguments = function_data.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("LLM response missing arguments")

    parsed = json.loads(arguments)
    description = parsed.get("description")
    if not isinstance(description, str):
        raise ValueError("Missing description string")

    sample_queries = parsed.get("sample_queries")
    if not isinstance(sample_queries, list):
        raise ValueError("Missing sample_queries list")
    return description, [str(query) for query in sample_queries]


def _looks_identifier_name(name: str) -> bool:
    """Check whether a column name suggests an identifier."""
    lowered = name.lower()
    keywords = (
        "id",
        "identifier",
        "reference",
        "ref",
        "account",
        "acct",
        "code",
        "key",
        "number",
    )
    return any(keyword in lowered for keyword in keywords)


def _looks_measure_name(name: str) -> bool:
    """Check whether a column name suggests a numeric measure."""
    lowered = name.lower()
    keywords = (
        "amount",
        "amt",
        "total",
        "balance",
        "revenue",
        "cost",
        "price",
        "value",
        "score",
        "rate",
        "count",
        "quantity",
        "qty",
        "percent",
        "pct",
    )
    return any(keyword in lowered for keyword in keywords)


def _classify_column_role(col: Any, row_count: int) -> str:
    """Classify a column into one dense-table query role."""
    unique_ratio = col.unique_count / row_count if row_count > 0 else 0.0
    if col.dtype in ("date", "boolean"):
        return "filter"
    if _looks_identifier_name(col.name) and (
        col.unique_count >= max(1, row_count - 1) or unique_ratio >= 0.8
    ):
        return "identifier"
    if col.dtype == "numeric":
        return "measure"
    if col.unique_count <= 20 and (row_count <= 20 or unique_ratio <= 0.35):
        return "filter"
    if _looks_measure_name(col.name) and col.dtype == "mixed":
        return "measure"
    return "text_content"


def _build_default_column_description(col: Any) -> str:
    """Build a deterministic description for a column."""
    safe_name = col.name.strip() or col.position
    description = f"{safe_name} contains {col.dtype} values."
    if col.dtype == "numeric":
        num_min = col.stats.get("min")
        num_max = col.stats.get("max")
        if num_min is not None and num_max is not None:
            description = (
                f"{safe_name} contains numeric values ranging from "
                f"{num_min} to {num_max}."
            )
        else:
            description = f"{safe_name} contains numeric values for analysis."
    elif col.dtype == "date":
        date_min = col.stats.get("min")
        date_max = col.stats.get("max")
        if date_min and date_max:
            description = (
                f"{safe_name} contains date values from "
                f"{date_min} to {date_max}."
            )
        else:
            description = (
                f"{safe_name} contains date values for time-based filtering."
            )
    elif col.dtype == "boolean":
        true_count = col.stats.get("true_count")
        false_count = col.stats.get("false_count")
        if true_count is not None and false_count is not None:
            description = (
                f"{safe_name} contains boolean values with "
                f"{true_count} true and {false_count} false entries."
            )
        else:
            description = f"{safe_name} contains boolean values."
    else:
        distribution = col.stats.get("value_distribution")
        if distribution:
            ranked = sorted(
                distribution.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            values = ", ".join(str(value) for value, _ in ranked[:5])
            description = (
                f"{safe_name} contains categorical values such as {values}."
            )
        elif col.sample_values:
            samples = ", ".join(str(value) for value in col.sample_values[:3])
            description = (
                f"{safe_name} contains text values such as {samples}."
            )
    return description


def _build_deterministic_sample_queries(
    page_title: str,
    eda: Any,
    role_lists: dict[str, list[str]],
) -> list[str]:
    """Build fallback sample queries from table structure."""
    column_index = _build_column_index(eda)

    def _name_for(position: str) -> str:
        col = column_index.get(position)
        if col is None:
            return position
        return col.name

    queries: list[str] = []
    measures = role_lists["measure"]
    filters = role_lists["filter"]
    identifiers = role_lists["identifier"]
    text_columns = role_lists["text_content"]

    if measures and filters:
        queries.append(
            "What is the total "
            f"{_name_for(measures[0])} by {_name_for(filters[0])}?"
        )
    if measures:
        queries.append(
            f"What is the average {_name_for(measures[0])} in {page_title}?"
        )
    if filters:
        queries.append(
            "How many rows fall into each "
            f"{_name_for(filters[0])} category?"
        )
    if identifiers and measures:
        queries.append(
            f"Which {_name_for(identifiers[0])} values have the highest "
            f"{_name_for(measures[0])}?"
        )
    if text_columns:
        queries.append(
            f"What details are recorded in {_name_for(text_columns[0])}?"
        )
    if not queries:
        queries.append(f"What records are included in {page_title}?")

    deduped: list[str] = []
    for query in queries:
        if query not in deduped:
            deduped.append(query)
    while len(deduped) < 3:
        deduped.append(f"What does the {page_title} dataset contain?")
    return deduped[:5]


def _build_deterministic_dataset_summary(
    page_title: str,
    eda: Any,
    role_lists: dict[str, list[str]],
) -> str:
    """Build a deterministic dense-table summary."""
    measures = len(role_lists["measure"])
    filters = len(role_lists["filter"])
    identifiers = len(role_lists["identifier"])
    time_scope = ""
    for col in eda.columns:
        if col.dtype != "date":
            continue
        date_min = col.stats.get("min")
        date_max = col.stats.get("max")
        if date_min and date_max:
            time_scope = f" It appears to cover {date_min} to {date_max}."
            break
    return (
        f"{page_title} is a dense table with {eda.row_count} rows and "
        f"{len(eda.columns)} columns from range {eda.used_range or 'unknown'}."
        f"{time_scope} It includes {filters} filter columns, "
        f"{identifiers} identifier columns, and {measures} measure columns "
        f"for retrieval and aggregation."
    )


def _resolve_role_assignments(
    eda: Any, description: DenseTableDescription
) -> dict[str, list[str]]:
    """Resolve and fill role assignments for every column."""
    column_index = _build_column_index(eda)
    column_name_index = _build_column_name_index(eda)
    role_by_position: dict[str, str] = {}
    role_sources = {
        "filter": description.filter_columns,
        "identifier": description.identifier_columns,
        "measure": description.measure_columns,
        "text_content": description.text_content_columns,
    }
    for role_name, values in role_sources.items():
        for key in values:
            col = _resolve_column_reference(
                column_index, column_name_index, key
            )
            if col is not None and col.position not in role_by_position:
                role_by_position[col.position] = role_name

    role_lists: dict[str, list[str]] = {
        "filter": [],
        "identifier": [],
        "measure": [],
        "text_content": [],
    }
    for col in eda.columns:
        role_name = role_by_position.get(
            col.position
        ) or _classify_column_role(col, eda.row_count)
        role_lists[role_name].append(col.position)
    return role_lists


def _normalize_dense_table_description(
    page_title: str,
    eda: Any,
    description: DenseTableDescription,
) -> DenseTableDescription:
    """Ensure a dense-table description covers all columns and roles."""
    desc_by_position, desc_by_name = _build_description_lookup(description)
    role_lists = _resolve_role_assignments(eda, description)

    normalized_column_descriptions = []
    for col in eda.columns:
        text = _lookup_column_description(
            col, desc_by_position, desc_by_name
        ) or _build_default_column_description(col)
        normalized_column_descriptions.append(
            {
                "position": col.position,
                "name": col.name,
                "description": text,
            }
        )

    description_text = description.description.strip()
    if not description_text:
        description_text = _build_deterministic_dataset_summary(
            page_title,
            eda,
            role_lists,
        )
    sample_queries = [
        query.strip()
        for query in description.sample_queries
        if isinstance(query, str) and query.strip()
    ]
    if not sample_queries:
        sample_queries = _build_deterministic_sample_queries(
            page_title,
            eda,
            role_lists,
        )

    return DenseTableDescription(
        description=description_text,
        column_descriptions=normalized_column_descriptions,
        filter_columns=role_lists["filter"],
        identifier_columns=role_lists["identifier"],
        measure_columns=role_lists["measure"],
        text_content_columns=role_lists["text_content"],
        sample_queries=sample_queries[:5],
    )


def _build_subset_eda(eda: Any, columns: list[Any]) -> Any:
    """Build a column-subset TableEDA for batched description."""
    return type(eda)(
        row_count=eda.row_count,
        columns=list(columns),
        header_row=eda.header_row,
        framing_context=eda.framing_context,
        sample_rows=[],
        token_count=eda.token_count,
        used_range=eda.used_range,
        header_mode=eda.header_mode,
        source_region_id=eda.source_region_id,
    )


def _call_dense_description_prompt(
    prompt_name: str,
    user_content: str,
    llm: Any,
    context: str = "",
) -> dict[str, Any]:
    """Call an LLM prompt used by dense-table description paths."""
    prompt = load_prompt(prompt_name, _XLSX_PROMPTS_DIR)
    messages = _build_prompt_messages(prompt, user_content)
    return llm.call(
        messages=messages,
        stage=prompt["stage"],
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
        context=context,
    )


def estimate_dense_description_tokens(
    page_title: str,
    eda: Any,
    build_description_message: Callable[[str, Any], str],
) -> int:
    """Estimate prompt tokens for a one-shot dense-table description."""
    prompt = load_prompt("dense_table_description", _XLSX_PROMPTS_DIR)
    return _estimate_prompt_tokens(
        prompt,
        build_description_message(page_title, eda),
    )


def batch_columns_for_description(
    page_title: str,
    eda: Any,
    max_prompt_tokens: int,
    build_description_message: Callable[[str, Any], str],
) -> list[list[Any]]:
    """Split columns into prompt-safe batches while preserving order."""
    if not eda.columns:
        return []

    batches: list[list[Any]] = []
    current_batch: list[Any] = []
    for col in eda.columns:
        candidate_batch = current_batch + [col]
        candidate_eda = _build_subset_eda(eda, candidate_batch)
        candidate_tokens = estimate_dense_description_tokens(
            page_title,
            candidate_eda,
            build_description_message,
        )
        if candidate_tokens <= max_prompt_tokens:
            current_batch = candidate_batch
            continue

        if not current_batch:
            return []

        batches.append(current_batch)
        current_batch = [col]
        single_eda = _build_subset_eda(eda, current_batch)
        if (
            estimate_dense_description_tokens(
                page_title,
                single_eda,
                build_description_message,
            )
            > max_prompt_tokens
        ):
            return []

    if current_batch:
        batches.append(current_batch)
    return batches


def build_deterministic_dense_description(
    page_title: str,
    eda: Any,
) -> DenseTableDescription:
    """Build a deterministic fallback dense-table description."""
    role_lists: dict[str, list[str]] = {
        "filter": [],
        "identifier": [],
        "measure": [],
        "text_content": [],
    }
    column_descriptions: list[dict[str, str]] = []
    for col in eda.columns:
        column_descriptions.append(
            {
                "position": col.position,
                "name": col.name,
                "description": _build_default_column_description(col),
            }
        )
        role_lists[_classify_column_role(col, eda.row_count)].append(
            col.position
        )

    return DenseTableDescription(
        description=_build_deterministic_dataset_summary(
            page_title,
            eda,
            role_lists,
        ),
        column_descriptions=column_descriptions,
        filter_columns=role_lists["filter"],
        identifier_columns=role_lists["identifier"],
        measure_columns=role_lists["measure"],
        text_content_columns=role_lists["text_content"],
        sample_queries=_build_deterministic_sample_queries(
            page_title,
            eda,
            role_lists,
        ),
    )


def _describe_dense_table_one_shot(
    page_title: str,
    eda: Any,
    llm: Any,
    build_description_message: Callable[[str, Any], str],
    parse_description_response: Callable[
        [dict[str, Any]], DenseTableDescription
    ],
    context: str = "",
) -> tuple[DenseTableDescription, str]:
    """Describe a dense table in a single LLM call."""
    response = _call_dense_description_prompt(
        prompt_name="dense_table_description",
        user_content=build_description_message(page_title, eda),
        llm=llm,
        context=context,
    )
    return (
        _normalize_dense_table_description(
            page_title,
            eda,
            parse_description_response(response),
        ),
        "llm_one_shot",
    )


def _combine_batch_descriptions(
    page_title: str,
    eda: Any,
    batch_results: list[tuple[Any, DenseTableDescription]],
) -> DenseTableDescription:
    """Combine disjoint batch descriptions into one table description."""
    descriptions_by_position: dict[str, dict[str, str]] = {}
    role_by_position: dict[str, str] = {}
    for batch_eda, description in batch_results:
        normalized = _normalize_dense_table_description(
            page_title,
            batch_eda,
            description,
        )
        for column_description in normalized.column_descriptions:
            descriptions_by_position[column_description["position"]] = (
                column_description
            )
        for role_name in (
            "filter_columns",
            "identifier_columns",
            "measure_columns",
            "text_content_columns",
        ):
            role_label = role_name.replace("_columns", "")
            for position in getattr(normalized, role_name):
                role_by_position[position] = role_label

    combined = DenseTableDescription(
        description="",
        column_descriptions=[],
        filter_columns=[],
        identifier_columns=[],
        measure_columns=[],
        text_content_columns=[],
        sample_queries=[],
    )
    for col in eda.columns:
        combined.column_descriptions.append(
            descriptions_by_position.get(
                col.position,
                {
                    "position": col.position,
                    "name": col.name,
                    "description": _build_default_column_description(col),
                },
            )
        )
        role_name = role_by_position.get(
            col.position
        ) or _classify_column_role(col, eda.row_count)
        getattr(combined, f"{role_name}_columns").append(col.position)
    return combined


def _build_batch_merge_message(
    page_title: str,
    eda: Any,
    description: DenseTableDescription,
    batch_results: list[tuple[Any, DenseTableDescription]],
) -> str:
    """Build a compact merge prompt from batched column descriptions."""
    role_by_position: dict[str, str] = {}
    for role_name in (
        "filter_columns",
        "identifier_columns",
        "measure_columns",
        "text_content_columns",
    ):
        role_label = role_name.replace("_columns", "")
        for position in getattr(description, role_name):
            role_by_position[position] = role_label

    lines = [
        "## Sheet context",
        f"- Sheet name: {page_title}",
        f"- Used range: {eda.used_range or 'unknown'}",
        f"- Header mode: {eda.header_mode}",
        f"- Total data rows: {eda.row_count}",
        f"- Total columns: {len(eda.columns)}",
        "",
        "## Batch summaries",
    ]
    for index, (batch_eda, batch_description) in enumerate(batch_results, 1):
        bounds = "none"
        if batch_eda.columns:
            bounds = (
                f"{batch_eda.columns[0].position}-"
                f"{batch_eda.columns[-1].position}"
            )
        batch_text = batch_description.description.strip() or "No summary."
        lines.append(f"- Batch {index} ({bounds}): {batch_text}")

    lines.extend(["", "## Column summaries"])
    for column_description in description.column_descriptions:
        position = column_description["position"]
        role_name = role_by_position.get(position, "text_content")
        lines.append(
            f"- {position} {column_description['name']}: "
            f"role={role_name}; "
            f"description={column_description['description']}"
        )
    return "\n".join(lines)


def _merge_batched_dense_description(
    page_title: str,
    eda: Any,
    description: DenseTableDescription,
    batch_results: list[tuple[Any, DenseTableDescription]],
    llm: Any,
    context: str = "",
) -> DenseTableDescription:
    """Generate dataset-level summary and queries for batched output."""
    merge_message = _build_batch_merge_message(
        page_title,
        eda,
        description,
        batch_results,
    )
    prompt = load_prompt(
        "dense_table_description_merge",
        _XLSX_PROMPTS_DIR,
    )
    if _estimate_prompt_tokens(prompt, merge_message) > (
        get_dense_table_description_max_prompt_tokens()
    ):
        return build_deterministic_dense_description(page_title, eda)

    response = _call_dense_description_prompt(
        prompt_name="dense_table_description_merge",
        user_content=merge_message,
        llm=llm,
        context=f"{context} merge".strip(),
    )
    merged_description, sample_queries = _parse_batch_summary_response(
        response
    )
    description.description = merged_description
    description.sample_queries = sample_queries
    return _normalize_dense_table_description(page_title, eda, description)


def _describe_dense_table_batched(
    page_title: str,
    eda: Any,
    llm: Any,
    build_description_message: Callable[[str, Any], str],
    parse_description_response: Callable[
        [dict[str, Any]], DenseTableDescription
    ],
    context: str = "",
) -> tuple[DenseTableDescription, str]:
    """Describe a dense table through prompt-safe column batches."""
    max_prompt_tokens = get_dense_table_description_max_prompt_tokens()
    column_batches = batch_columns_for_description(
        page_title,
        eda,
        max_prompt_tokens,
        build_description_message,
    )
    if len(column_batches) <= 1:
        return (
            build_deterministic_dense_description(page_title, eda),
            "deterministic_fallback",
        )

    batch_results: list[tuple[Any, DenseTableDescription]] = []
    for batch_columns in column_batches:
        batch_eda = _build_subset_eda(eda, batch_columns)
        batch_description, _mode = _describe_dense_table_one_shot(
            page_title,
            batch_eda,
            llm,
            build_description_message,
            parse_description_response,
            context=f"{context} batch".strip(),
        )
        batch_results.append(
            (
                batch_eda,
                batch_description,
            )
        )

    combined = _combine_batch_descriptions(page_title, eda, batch_results)
    return (
        _merge_batched_dense_description(
            page_title,
            eda,
            combined,
            batch_results,
            llm,
            context=context,
        ),
        "llm_batched",
    )


def describe_dense_table_with_budget(
    page_title: str,
    eda: Any,
    llm: Any,
    build_description_message: Callable[[str, Any], str],
    parse_description_response: Callable[
        [dict[str, Any]], DenseTableDescription
    ],
    context: str = "",
) -> tuple[DenseTableDescription, str]:
    """Describe a dense table with one-shot, batched, and fallback paths."""
    max_prompt_tokens = get_dense_table_description_max_prompt_tokens()
    if (
        estimate_dense_description_tokens(
            page_title,
            eda,
            build_description_message,
        )
        <= max_prompt_tokens
    ):
        return _describe_dense_table_one_shot(
            page_title,
            eda,
            llm,
            build_description_message,
            parse_description_response,
            context=context,
        )

    try:
        return _describe_dense_table_batched(
            page_title,
            eda,
            llm,
            build_description_message,
            parse_description_response,
            context=context,
        )
    except _DENSE_TABLE_ERRORS:
        return (
            build_deterministic_dense_description(page_title, eda),
            "deterministic_fallback",
        )
