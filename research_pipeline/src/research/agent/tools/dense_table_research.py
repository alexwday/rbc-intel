"""
Dense Table Research — sub-retrieval for dense table regions.

Unified pathway for all dense table sizes:

1. If the full table fits within the token budget, load it directly
   into a single research call.
2. If over budget, use the table description + research statement to
   identify optional filters, apply them programmatically, then split
   the (possibly filtered) data into batches.
3. Each batch (including the single-batch case) gets the same research
   prompt with header row + data rows.  Batches run in parallel.
4. Batch results are joined and returned as the dense table finding.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from ...connections.llm import execute_llm_call
from ...connections.postgres import get_database_session, get_database_schema
from ...utils.config import config
from ...utils.prompt_loader import get_prompt

logger = logging.getLogger(__name__)

# Rough conversion: 1 token ≈ 4 characters of JSON
CHARS_PER_TOKEN = 4
DEFAULT_TOKEN_BUDGET = 80_000
# Overhead for prompt template, description, header — reserve tokens
PROMPT_OVERHEAD_TOKENS = 4_000

MODEL_CAPABILITY = "small"


class DenseTableResearchError(Exception):
    """Exception raised for dense table research errors."""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _parse_routing_metadata(routing_json: Any) -> Optional[Dict[str, Any]]:
    """Parse dense_table_routing_json from chunk metadata."""
    if routing_json is None:
        return None
    if isinstance(routing_json, dict):
        return routing_json
    if isinstance(routing_json, str):
        try:
            parsed = json.loads(routing_json)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _parse_raw_content(raw_content_json: Any) -> List[Dict[str, Any]]:
    """Parse raw_content_json into a list of row dicts."""
    if isinstance(raw_content_json, str):
        try:
            raw_content_json = json.loads(raw_content_json)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(raw_content_json, list):
        return raw_content_json
    if isinstance(raw_content_json, dict):
        return raw_content_json.get("rows", [])
    return []


def _extract_header_and_data(
    rows: List[Dict[str, Any]],
) -> Tuple[List[str], List[List[str]]]:
    """Split raw rows into a header list and data rows.

    Row 0 is treated as the header. Each row's cells are flattened
    to a list of string values.
    """
    if not rows:
        return [], []

    def _cell_values(row: Dict[str, Any]) -> List[str]:
        cells = row.get("cells", [])
        return [str(c.get("value", "")) for c in cells]

    header = _cell_values(rows[0])
    data = [_cell_values(r) for r in rows[1:]]
    return header, data


def _format_table_text(
    header: List[str], data_rows: List[List[str]]
) -> str:
    """Format header + data rows as a pipe-delimited table string."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in data_rows:
        # Pad or truncate to header length
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return len(text) // CHARS_PER_TOKEN


def _estimate_table_tokens(
    header: List[str], data_rows: List[List[str]]
) -> int:
    """Estimate token count for a formatted table."""
    table_text = _format_table_text(header, data_rows)
    return _estimate_tokens(table_text)


# ---------------------------------------------------------------------------
# Filter analysis (only for large tables)
# ---------------------------------------------------------------------------


def _analyze_filters(
    research_statement: str,
    chunk_description: str,
    routing: Dict[str, Any],
    header: List[str],
    data_rows: List[List[str]],
    token: str,
) -> Dict[str, str]:
    """Use LLM to identify applicable filters for a large table.

    Returns:
        Dict mapping column name to filter value, or empty dict.
    """
    filter_columns = routing.get("filter_columns", [])
    if not filter_columns:
        return {}

    # Build available filter values from data
    col_indices = {}
    for col_letter in filter_columns:
        # Convert column letter to index (A=0, B=1, etc.)
        idx = ord(col_letter.upper()) - ord("A")
        if idx < len(header):
            col_indices[header[idx]] = idx

    if not col_indices:
        return {}

    # Get distinct values per filter column
    filter_info_parts = []
    for col_name, idx in col_indices.items():
        values = sorted(
            set(
                row[idx]
                for row in data_rows
                if idx < len(row) and row[idx]
            )
        )
        if values:
            filter_info_parts.append(
                f"- {col_name}: {', '.join(values[:20])}"
                + (f" ... ({len(values)} total)" if len(values) > 20 else "")
            )

    if not filter_info_parts:
        return {}

    filter_info = "\n".join(filter_info_parts)

    try:
        system_prompt, tools, user_template = get_prompt(
            "subagent", "dense_table_filter"
        )

        user_content = (
            user_template.replace("{{research_statement}}", research_statement)
            .replace("{{table_description}}", chunk_description[:2000])
            .replace("{{filter_columns}}", filter_info)
        )

        model_settings = config.get_model_settings(MODEL_CAPABILITY)

        response, _ = execute_llm_call(
            oauth_token=token,
            model=model_settings["name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1000,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "select_filters"},
            },
            stream=False,
            prompt_token_cost=model_settings["prompt_token_cost"],
            completion_token_cost=model_settings["completion_token_cost"],
            reasoning_effort=model_settings.get("reasoning_effort"),
        )

        message = response.choices[0].message
        if message.tool_calls:
            args = json.loads(message.tool_calls[0].function.arguments)
            filters = args.get("filters", {})
            if isinstance(filters, dict):
                logger.info(
                    "Dense table filter analysis returned: %s", filters
                )
                return {
                    k: v
                    for k, v in filters.items()
                    if isinstance(v, str) and v
                }

    except Exception as exc:
        logger.warning("Filter analysis failed, proceeding unfiltered: %s", exc)

    return {}


def _apply_filters(
    header: List[str],
    data_rows: List[List[str]],
    filters: Dict[str, str],
) -> List[List[str]]:
    """Apply column filters to data rows."""
    if not filters:
        return data_rows

    col_indices = {}
    for col_name, value in filters.items():
        if col_name in header:
            col_indices[header.index(col_name)] = value.lower()

    if not col_indices:
        return data_rows

    filtered = []
    for row in data_rows:
        match = True
        for idx, value in col_indices.items():
            if idx >= len(row) or row[idx].lower() != value:
                match = False
                break
        if match:
            filtered.append(row)

    logger.info(
        "Dense table filter: %d → %d rows (filters: %s)",
        len(data_rows),
        len(filtered),
        filters,
    )
    return filtered


# ---------------------------------------------------------------------------
# Batch splitting
# ---------------------------------------------------------------------------


def _split_into_batches(
    header: List[str],
    data_rows: List[List[str]],
    token_budget: int,
) -> List[List[List[str]]]:
    """Split data rows into batches that fit within the token budget.

    Each batch will be formatted with the header row, so the budget
    accounts for header + prompt overhead per batch.
    """
    available = token_budget - PROMPT_OVERHEAD_TOKENS
    header_text = "| " + " | ".join(header) + " |"
    header_tokens = _estimate_tokens(header_text) + 10  # separator line

    rows_budget = available - header_tokens
    if rows_budget <= 0:
        return [data_rows]

    # Estimate tokens per row from average row length
    if data_rows:
        sample = data_rows[: min(10, len(data_rows))]
        sample_text = "\n".join(
            "| " + " | ".join(r[: len(header)]) + " |" for r in sample
        )
        avg_tokens_per_row = max(
            1, _estimate_tokens(sample_text) // len(sample)
        )
    else:
        return [[]]

    rows_per_batch = max(1, rows_budget // avg_tokens_per_row)

    batches = []
    for i in range(0, len(data_rows), rows_per_batch):
        batches.append(data_rows[i : i + rows_per_batch])

    return batches


# ---------------------------------------------------------------------------
# Per-batch research
# ---------------------------------------------------------------------------


def _research_batch(
    research_statement: str,
    sheet_name: str,
    description_summary: str,
    header: List[str],
    batch_rows: List[List[str]],
    batch_number: int,
    total_batches: int,
    token: str,
) -> str:
    """Run a single research LLM call on one batch of table data."""
    table_text = _format_table_text(header, batch_rows)

    batch_context = ""
    if total_batches > 1:
        batch_context = (
            f"\nThis is batch {batch_number} of {total_batches}. "
            f"Your findings will be combined with results from the other "
            f"batches to produce the final answer. Extract all relevant "
            f"findings from THIS batch only.\n"
        )

    try:
        system_prompt, _, user_template = get_prompt(
            "subagent", "dense_table_research"
        )

        user_content = (
            user_template.replace("{{research_statement}}", research_statement)
            .replace("{{sheet_name}}", sheet_name)
            .replace("{{description_summary}}", description_summary[:1500])
            .replace("{{batch_context}}", batch_context)
            .replace("{{table_data}}", table_text)
            .replace("{{row_count}}", str(len(batch_rows)))
        )

        model_settings = config.get_model_settings(MODEL_CAPABILITY)

        response, _ = execute_llm_call(
            oauth_token=token,
            model=model_settings["name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=4000,
            stream=False,
            prompt_token_cost=model_settings["prompt_token_cost"],
            completion_token_cost=model_settings["completion_token_cost"],
            reasoning_effort=model_settings.get("reasoning_effort"),
        )

        content = response.choices[0].message.content
        return content or ""

    except Exception as exc:
        logger.error(
            "Dense table batch %d/%d research failed: %s",
            batch_number,
            total_batches,
            exc,
        )
        return f"[Batch {batch_number} research failed: {exc}]"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def research_dense_table(
    research_statement: str,
    document_id: str,
    routing_json: Any,
    chunk_description: str,
    token: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> Optional[str]:
    """Research a dense table using the unified pathway.

    1. If the full table fits within token_budget → single batch.
    2. If over budget → filter analysis, apply filters, split into
       batches that fit within budget.
    3. Research each batch in parallel.
    4. Join results.

    Args:
        research_statement: The user's research query.
        document_id: Parent document identifier.
        routing_json: The dense_table_routing_json from the chunk.
        chunk_description: The chunk's text content (table description).
        token: Authentication token for LLM calls.
        token_budget: Maximum tokens per research call.

    Returns:
        Research findings text, or None if the table cannot be retrieved.
    """
    start_time = time.time()

    # --- Parse routing and fetch raw data ---
    routing = _parse_routing_metadata(routing_json)
    if not routing:
        return None

    region_id = routing.get("selected_region_id") or routing.get("region_id")
    sheet_name_filter = routing.get("sheet_name", "")
    if not region_id:
        return None

    schema = get_database_schema()
    try:
        with get_database_session() as session:
            # Match on region_id AND sheet_name to avoid ambiguity
            # when multiple sheets share the same region_id
            params: Dict[str, Any] = {
                "document_id": document_id,
                "region_id": region_id,
            }
            sheet_clause = ""
            if sheet_name_filter:
                sheet_clause = "AND sheet_name = :sheet_name"
                params["sheet_name"] = sheet_name_filter

            result = session.execute(
                text(
                    f"""
                    SELECT sheet_name, used_range, raw_content_json,
                           routing_metadata_json
                    FROM {schema}.document_dense_tables
                    WHERE document_id = :document_id
                      AND region_id = :region_id
                      {sheet_clause}
                    LIMIT 1
                    """
                ),
                params,
            )
            row = result.mappings().first()
    except Exception as exc:
        logger.error("Failed to fetch dense table: %s", exc)
        return None

    if not row:
        return None

    sheet_name = row["sheet_name"] or ""
    used_range = row["used_range"] or ""
    raw_rows = _parse_raw_content(row["raw_content_json"])

    if not raw_rows:
        return None

    header, data_rows = _extract_header_and_data(raw_rows)
    if not header or not data_rows:
        return None

    logger.info(
        "Dense table research: %s %s (%d data rows, %d columns)",
        sheet_name,
        used_range,
        len(data_rows),
        len(header),
    )

    # --- Step 1: Check if full table fits in budget ---
    full_tokens = _estimate_table_tokens(header, data_rows)
    effective_budget = token_budget - PROMPT_OVERHEAD_TOKENS

    if full_tokens <= effective_budget:
        # Single batch — entire table fits
        logger.info(
            "Dense table fits in budget (%d tokens <= %d) — single batch",
            full_tokens,
            effective_budget,
        )
        result_text = _research_batch(
            research_statement=research_statement,
            sheet_name=sheet_name,
            description_summary=chunk_description,
            header=header,
            batch_rows=data_rows,
            batch_number=1,
            total_batches=1,
            token=token,
        )
        elapsed = time.time() - start_time
        logger.info(
            "Dense table research complete: %s — 1 batch, %.1fs",
            sheet_name,
            elapsed,
        )
        return _format_result(
            result_text, sheet_name, region_id, used_range,
            len(data_rows), 1, "direct",
        )

    # --- Step 2: Over budget — try filtering ---
    logger.info(
        "Dense table over budget (%d tokens > %d) — analyzing filters",
        full_tokens,
        effective_budget,
    )

    filters = _analyze_filters(
        research_statement, chunk_description, routing,
        header, data_rows, token,
    )
    filtered_rows = _apply_filters(header, data_rows, filters)

    # --- Step 3: Split into batches ---
    batches = _split_into_batches(header, filtered_rows, token_budget)
    total_batches = len(batches)

    logger.info(
        "Dense table: %d rows after filtering → %d batch(es)",
        len(filtered_rows),
        total_batches,
    )

    # --- Step 4: Research batches in parallel ---
    batch_results: List[Tuple[int, str]] = []

    if total_batches == 1:
        result_text = _research_batch(
            research_statement=research_statement,
            sheet_name=sheet_name,
            description_summary=chunk_description,
            header=header,
            batch_rows=batches[0],
            batch_number=1,
            total_batches=1,
            token=token,
        )
        batch_results.append((1, result_text))
    else:
        with ThreadPoolExecutor(
            max_workers=min(total_batches, 5)
        ) as executor:
            futures = {}
            for i, batch in enumerate(batches, 1):
                future = executor.submit(
                    _research_batch,
                    research_statement=research_statement,
                    sheet_name=sheet_name,
                    description_summary=chunk_description,
                    header=header,
                    batch_rows=batch,
                    batch_number=i,
                    total_batches=total_batches,
                    token=token,
                )
                futures[future] = i

            for future in as_completed(futures):
                batch_num = futures[future]
                batch_results.append((batch_num, future.result()))

    # Sort by batch number for consistent ordering
    batch_results.sort(key=lambda x: x[0])
    combined_text = "\n\n".join(
        text for _, text in batch_results if text.strip()
    )

    method = "filtered" if filters else "batched"
    elapsed = time.time() - start_time
    logger.info(
        "Dense table research complete: %s — %d batch(es), "
        "%s, %.1fs",
        sheet_name,
        total_batches,
        method,
        elapsed,
    )

    return _format_result(
        combined_text, sheet_name, region_id, used_range,
        len(filtered_rows), total_batches, method,
    )


def _format_result(
    findings_text: str,
    sheet_name: str,
    region_id: str,
    used_range: str,
    rows_analyzed: int,
    batch_count: int,
    method: str,
) -> str:
    """Format dense table research results as XML for chunk injection."""
    return (
        f'  <dense_table_research sheet="{sheet_name}" '
        f'region="{region_id}" used_range="{used_range}" '
        f'rows_analyzed="{rows_analyzed}" batches="{batch_count}" '
        f'method="{method}">\n'
        f"    {findings_text}\n"
        f"  </dense_table_research>"
    )
