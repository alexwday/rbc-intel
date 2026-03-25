"""XLSX-specific dense-table preparation and replacement helpers."""

import json
from typing import Any, List

from ...utils.llm import LLMClient
from ...utils.file_types import PreparedPage
from .dense_table import (
    batch_columns_for_description as _batch_columns_for_description,
    build_deterministic_dense_description,
    describe_dense_table_with_budget,
    estimate_dense_description_tokens as _estimate_dense_description_tokens,
)
from .table_eda import run_table_eda, run_table_eda_from_region
from .types import DenseTableDescription, PreparedDenseTable


def _is_dense_table_candidate(page: dict[str, Any]) -> bool:
    """Check if a page was classified as dense_table_candidate.

    Params: page (dict). Returns: bool.
    """
    metadata = page.get("metadata", {})
    return metadata.get("handling_mode") == "dense_table_candidate"


def _parse_description_response(
    response: dict[str, Any],
) -> DenseTableDescription:
    """Parse the LLM dense table description tool response."""
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

    col_descs = parsed.get("column_descriptions")
    if not isinstance(col_descs, list):
        raise ValueError("Missing column_descriptions list")

    sample_queries = parsed.get("sample_queries")
    if not isinstance(sample_queries, list):
        raise ValueError("Missing sample_queries list")

    def _str_list(key: str) -> List[str]:
        val = parsed.get(key, [])
        if not isinstance(val, list):
            return []
        return [str(v) for v in val]

    return DenseTableDescription(
        description=description,
        column_descriptions=[
            {
                "position": str(cd.get("position", "")),
                "name": str(cd.get("name", "")),
                "description": str(cd.get("description", "")),
            }
            for cd in col_descs
            if isinstance(cd, dict)
        ],
        filter_columns=_str_list("filter_columns"),
        identifier_columns=_str_list("identifier_columns"),
        measure_columns=_str_list("measure_columns"),
        text_content_columns=_str_list("text_content_columns"),
        sample_queries=[str(q) for q in sample_queries],
    )


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


def _format_column_label(col: Any) -> str:
    """Format a column label with a stable position suffix when needed."""
    safe_name = _escape_markdown_table_cell(col.name)
    if safe_name == col.position or safe_name.endswith(f"({col.position})"):
        return safe_name
    return f"{safe_name} ({col.position})"


def _build_description_lookup(
    description: DenseTableDescription,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build position and name lookups for column descriptions."""
    by_position: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for column_description in description.column_descriptions:
        position = str(column_description.get("position", "")).strip()
        name = str(column_description.get("name", "")).strip()
        text = str(column_description.get("description", ""))
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


def _format_column_profile(col: Any) -> str:
    """Format one column's EDA profile for the LLM message."""
    safe_name = _escape_markdown_table_cell(col.name)
    parts = [
        f"- **{safe_name}** (column {col.position}): "
        f"dtype={col.dtype}, "
        f"non_null={col.non_null_count}, "
        f"null={col.null_count}, "
        f"unique={col.unique_count}",
    ]
    dist = col.stats.get("value_distribution")
    if dist:
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        dist_str = ", ".join(f"{v} ({c})" for v, c in ranked)
        parts.append(f"  values: [{dist_str}]")
    else:
        stat_items = {
            k: v for k, v in col.stats.items() if k != "value_distribution"
        }
        if stat_items:
            stats_str = ", ".join(f"{k}={v}" for k, v in stat_items.items())
            parts.append(f"  stats: {stats_str}")
        if col.sample_values:
            samples = ", ".join(repr(s) for s in col.sample_values[:5])
            parts.append(f"  samples: [{samples}]")
    return "\n".join(parts)


def _build_description_message(
    page_title: str,
    eda: Any,
) -> str:
    """Build the user message for dense table description."""
    col_profiles: list[str] = []
    for col in eda.columns:
        profile = _format_column_profile(col)
        col_profiles.append(profile)

    col_names = [_escape_markdown_table_cell(col.name) for col in eda.columns]
    header = "| Row | " + " | ".join(col_names) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(col_names)) + " |"
    preview_lines = [header, sep] + eda.sample_rows[:5]
    preview = "\n".join(preview_lines)

    return (
        f"## Sheet context\n"
        f"- Sheet name: {page_title}\n"
        f"- Used range: {eda.used_range or 'unknown'}\n"
        f"- Header mode: {eda.header_mode}\n"
        f"- Total data rows: {eda.row_count}\n"
        f"- Total columns: {len(eda.columns)}\n"
        f"- Estimated tokens: {eda.token_count}\n\n"
        f"## Column profiles\n" + "\n".join(col_profiles) + "\n\n"
        f"## Framing context\n{eda.framing_context}\n\n"
        f"## Data preview\n{preview}"
    )


def estimate_dense_description_tokens(page_title: str, eda: Any) -> int:
    """Estimate prompt tokens for a one-shot dense table description."""
    return _estimate_dense_description_tokens(
        page_title,
        eda,
        _build_description_message,
    )


def batch_columns_for_description(
    page_title: str,
    eda: Any,
    max_prompt_tokens: int,
) -> list[list[Any]]:
    """Split columns into prompt-safe batches while preserving order."""
    return _batch_columns_for_description(
        page_title,
        eda,
        max_prompt_tokens,
        _build_description_message,
    )


def _build_deterministic_dense_description(
    page_title: str,
    eda: Any,
) -> DenseTableDescription:
    """Build a deterministic fallback dense-table description."""
    return build_deterministic_dense_description(page_title, eda)


def _build_region_framing_context(
    page_title: str,
    page_metadata: dict[str, Any],
    dense_region: dict[str, Any] | None = None,
) -> str:
    """Build framing context for a region-sourced dense table."""
    selected_region = dense_region if isinstance(dense_region, dict) else None
    used_range = ""
    if selected_region is not None:
        used_range = str(selected_region.get("used_range", "")).strip()
    if not used_range:
        used_range = str(
            page_metadata.get("dense_table_used_range", "")
        ).strip()
    framing_summary = str(page_metadata.get("framing_summary", "")).strip()
    lines = [f"# Sheet: {page_title}"]
    if used_range:
        lines.append(f"- Dense table used range: {used_range}")
    if framing_summary:
        lines.extend(["", framing_summary])
    return "\n".join(lines)


def _build_dense_table_eda(
    page_title: str,
    content: str,
    page_metadata: dict[str, Any] | None,
    dense_region: dict[str, Any] | None = None,
) -> Any:
    """Build dense-table EDA from region metadata when available."""
    if not isinstance(page_metadata, dict):
        return run_table_eda(content)

    selected_region = dense_region if isinstance(dense_region, dict) else None
    if selected_region is None:
        selected_region = page_metadata.get("dense_table_region")
    if not isinstance(selected_region, dict):
        return run_table_eda(content)

    framing_context = _build_region_framing_context(
        page_title,
        page_metadata,
        selected_region,
    )
    return run_table_eda_from_region(
        region=selected_region,
        framing_context=framing_context,
        token_source=content,
    )


def _dense_table_log_context(
    file_label: str,
    page_title: str,
    dense_region: dict[str, Any] | None = None,
) -> str:
    """Build a concise file/sheet/region label for dense-table logs."""
    region_id = ""
    if isinstance(dense_region, dict):
        region_id = str(dense_region.get("region_id", "")).strip()
    context = f"{file_label} sheet '{page_title}'"
    if region_id:
        return f"{context} region {region_id}"
    return context


def _describe_dense_table(
    page_title: str,
    content: str,
    llm: LLMClient,
    page_metadata: dict[str, Any] | None = None,
    dense_region: dict[str, Any] | None = None,
    context: str = "",
) -> tuple[Any, DenseTableDescription, str]:
    """Run EDA and LLM description for a dense table page.

    Params:
        page_title: Sheet name / page title
        content: Raw markdown content from extraction
        llm: LLMClient instance

    Returns:
        tuple of (TableEDA, DenseTableDescription, generation_mode)
    """
    eda = _build_dense_table_eda(
        page_title,
        content,
        page_metadata,
        dense_region,
    )
    description, generation_mode = describe_dense_table_with_budget(
        page_title=page_title,
        eda=eda,
        llm=llm,
        build_description_message=_build_description_message,
        parse_description_response=_parse_description_response,
        context=context,
    )
    return eda, description, generation_mode


def _build_column_roles_section(
    eda: Any, description: DenseTableDescription
) -> str:
    """Build column roles with EDA-sourced values."""
    column_index = _build_column_index(eda)
    column_name_index = _build_column_name_index(eda)
    sections: list[str] = []

    if description.filter_columns:
        lines = ["### Filter Columns"]
        for key in description.filter_columns:
            col = _resolve_column_reference(
                column_index, column_name_index, key
            )
            if col:
                lines.append(_format_filter_line(col))
        sections.append("\n".join(lines))

    if description.identifier_columns:
        items = ", ".join(
            (
                _format_column_label(
                    _resolve_column_reference(
                        column_index,
                        column_name_index,
                        key,
                    )
                )
                if _resolve_column_reference(
                    column_index,
                    column_name_index,
                    key,
                )
                else _escape_markdown_table_cell(key)
            )
            for key in description.identifier_columns
        )
        sections.append(f"### Identifiers\n- {items}")

    if description.measure_columns:
        lines = ["### Measures"]
        for key in description.measure_columns:
            col = _resolve_column_reference(
                column_index, column_name_index, key
            )
            if not col:
                continue
            safe_name = _format_column_label(col)
            num_min = col.stats.get("min")
            num_max = col.stats.get("max")
            if num_min is not None and num_max is not None:
                lines.append(f"- {safe_name}: {num_min} to {num_max}")
            else:
                lines.append(f"- {safe_name}")
        sections.append("\n".join(lines))

    if description.text_content_columns:
        items = ", ".join(
            (
                _format_column_label(
                    _resolve_column_reference(
                        column_index,
                        column_name_index,
                        key,
                    )
                )
                if _resolve_column_reference(
                    column_index,
                    column_name_index,
                    key,
                )
                else _escape_markdown_table_cell(key)
            )
            for key in description.text_content_columns
        )
        sections.append(f"### Text Content\n- {items}")

    return "\n\n".join(sections) if sections else ""


def _format_filter_line(col: Any) -> str:
    """Format a filter column's values from EDA stats."""
    name = _format_column_label(col)
    if col.dtype == "date":
        date_min = col.stats.get("min", "")
        date_max = col.stats.get("max", "")
        if date_min and date_max:
            return f"- {name}: {date_min} to {date_max}"
    dist = col.stats.get("value_distribution")
    if dist:
        vals = sorted(dist.keys(), key=dist.get, reverse=True)
        safe_values = [
            _escape_markdown_table_cell(str(value)) for value in vals
        ]
        return f"- {name} ({len(vals)} values): " + ", ".join(safe_values)
    if col.sample_values:
        safe_samples = ", ".join(
            _escape_markdown_table_cell(str(value))
            for value in col.sample_values
        )
        return f"- {name} ({col.unique_count} unique): " f"{safe_samples} ..."
    return f"- {name}: {col.unique_count} unique"


def _build_replacement_content(
    page_title: str,
    eda: Any,
    description: DenseTableDescription,
    routing_metadata: dict[str, Any],
) -> str:
    """Build markdown replacement content for a dense table.

    Params:
        page_title: Sheet name / page title
        eda: TableEDA from analysis
        description: LLM-generated description

    Returns:
        str — markdown replacement content
    """
    col_table = _build_column_table(eda, description)
    queries = "\n".join(f"- {q}" for q in description.sample_queries)
    roles = _build_column_roles_section(eda, description)
    preview = _build_data_preview(eda)
    filter_columns = ", ".join(
        _escape_markdown_table_cell(value)
        for value in routing_metadata.get("filter_columns", [])
    )
    identifier_columns = ", ".join(
        _escape_markdown_table_cell(value)
        for value in routing_metadata.get("identifier_columns", [])
    )
    measure_columns = ", ".join(
        _escape_markdown_table_cell(value)
        for value in routing_metadata.get("measure_columns", [])
    )
    text_content_columns = ", ".join(
        _escape_markdown_table_cell(value)
        for value in routing_metadata.get("text_content_columns", [])
    )
    route_lines = [
        f"- Sheet name: {routing_metadata.get('sheet_name', page_title)}",
        "- Selected region ID: "
        + str(routing_metadata.get("selected_region_id", "")),
        f"- Used range: {routing_metadata.get('used_range', eda.used_range)}",
        "- Available dense regions: "
        + ", ".join(routing_metadata.get("dense_table_region_ids", [])),
        f"- Filter columns: {filter_columns}",
        f"- Identifier columns: {identifier_columns}",
        f"- Measure columns: {measure_columns}",
        f"- Text content columns: {text_content_columns}",
    ]
    route_section = "\n".join(route_lines)

    return (
        f"# Dense Table: {page_title}\n\n"
        f"## Description\n"
        f"{description.description}\n\n"
        "## Subretrieval Routing\n"
        "Use this chunk to route the user request to the preserved dense "
        "table data rather than answering from the summary alone.\n"
        f"{route_section}\n\n"
        f"## Columns ({len(eda.columns)} columns, "
        f"{eda.row_count} rows)\n"
        f"{col_table}\n\n"
        f"## Column Roles\n{roles}\n\n"
        f"## Sample Queries\n{queries}\n\n"
        f"## Data Preview\n{preview}"
    )


def _build_dense_table_routing_metadata(
    page_title: str,
    eda: Any,
    description: DenseTableDescription,
    page_metadata: dict[str, Any] | None,
    dense_region: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic routing metadata for dense-table chunks."""
    metadata = page_metadata if isinstance(page_metadata, dict) else {}
    selected_region = dense_region if isinstance(dense_region, dict) else {}
    if not selected_region:
        selected_region = metadata.get("dense_table_region", {})
    region_id = str(selected_region.get("region_id", "")).strip()
    used_range = (
        str(selected_region.get("used_range", "")).strip() or eda.used_range
    )
    dense_regions = metadata.get("dense_table_regions", [])
    dense_region_ids = [
        str(region.get("region_id", "")).strip()
        for region in dense_regions
        if isinstance(region, dict)
        and str(region.get("region_id", "")).strip()
    ]
    if region_id and region_id not in dense_region_ids:
        dense_region_ids.insert(0, region_id)
    return {
        "page_title": page_title,
        "sheet_name": str(metadata.get("sheet_name", page_title)).strip(),
        "selected_region_id": region_id or eda.source_region_id,
        "used_range": used_range,
        "source_region_id": eda.source_region_id or region_id,
        "dense_table_region_ids": dense_region_ids,
        "filter_columns": list(description.filter_columns),
        "identifier_columns": list(description.identifier_columns),
        "measure_columns": list(description.measure_columns),
        "text_content_columns": list(description.text_content_columns),
        "sample_queries": list(description.sample_queries),
    }


def _get_dense_table_regions(
    page_metadata: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return all dense-table regions available for a page."""
    if not isinstance(page_metadata, dict):
        return []

    dense_regions = page_metadata.get("dense_table_regions", [])
    if isinstance(dense_regions, list) and dense_regions:
        return [region for region in dense_regions if isinstance(region, dict)]

    dense_region = page_metadata.get("dense_table_region")
    if isinstance(dense_region, dict):
        return [dense_region]
    return []


def _extract_dense_table_raw_rows(
    dense_region: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Extract structured raw rows from dense-region metadata."""
    if not isinstance(dense_region, dict):
        return []

    rows = dense_region.get("rows", [])
    if not isinstance(rows, list):
        return []

    serialized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_number = row.get("row_number")
        if not isinstance(row_number, int):
            continue
        cells = row.get("cells", [])
        if not isinstance(cells, list):
            continue
        serialized_cells = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            column_number = cell.get("column_number")
            if not isinstance(column_number, int):
                continue
            serialized_cells.append(
                {
                    "column_number": column_number,
                    "value": str(cell.get("value", "")),
                }
            )
        serialized_rows.append(
            {
                "row_number": row_number,
                "cells": serialized_cells,
            }
        )
    return serialized_rows


def _build_region_markers(region_id: str) -> tuple[str, str]:
    """Build the serialized XLSX region markers for a sheet region."""
    clean_region_id = region_id.strip()
    return (
        f"<!-- region:{clean_region_id} start -->",
        f"<!-- region:{clean_region_id} end -->",
    )


def _find_region_bounds(
    content: str,
    region_id: str,
) -> tuple[int, int, int, int] | None:
    """Find a region marker block and the inner markdown it wraps."""
    if not region_id.strip():
        return None

    start_marker, end_marker = _build_region_markers(region_id)
    block_start = content.find(start_marker)
    if block_start < 0:
        return None

    block_end = content.find(end_marker, block_start + len(start_marker))
    if block_end < 0:
        return None

    body_start = block_start + len(start_marker)
    if body_start < len(content) and content[body_start] == "\n":
        body_start += 1

    body_end = block_end
    while body_end > body_start and content[body_end - 1] == "\n":
        body_end -= 1

    block_end += len(end_marker)
    if block_end < len(content) and content[block_end] == "\n":
        block_end += 1

    return block_start, block_end, body_start, body_end


def _strip_region_markers(content: str) -> str:
    """Remove XLSX region markers from serialized sheet content."""
    lines = [
        line
        for line in content.splitlines()
        if not (
            line.strip().startswith("<!-- region:")
            and line.strip().endswith("-->")
        )
    ]
    cleaned = "\n".join(lines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def _build_dense_table_fallback_content(
    dense_tables: list[PreparedDenseTable],
) -> str:
    """Fall back to standalone dense-table descriptions when splicing fails."""
    return "\n\n".join(
        dense_table.replacement_content.strip()
        for dense_table in dense_tables
        if dense_table.replacement_content.strip()
    )


def _splice_dense_table_replacements(
    content: str,
    dense_tables: list[PreparedDenseTable],
) -> str:
    """Replace serialized region blocks with dense-table descriptions."""
    spliced_content = content
    markers_found = True

    for dense_table in dense_tables:
        bounds = _find_region_bounds(spliced_content, dense_table.region_id)
        if bounds is None:
            markers_found = False
            continue

        block_start, block_end, _, _ = bounds
        spliced_content = (
            spliced_content[:block_start]
            + dense_table.replacement_content.strip()
            + spliced_content[block_end:]
        )

    if not markers_found:
        return _build_dense_table_fallback_content(dense_tables)
    return _strip_region_markers(spliced_content)


def _prepare_dense_table_region(
    page_title: str,
    content: str,
    llm: LLMClient,
    page_metadata: dict[str, Any] | None,
    dense_region: dict[str, Any] | None,
    file_label: str = "",
) -> PreparedDenseTable:
    """Prepare one dense-table region for replacement and subretrieval."""
    eda, description, generation_mode = _describe_dense_table(
        page_title,
        content,
        llm,
        page_metadata,
        dense_region,
        context=_dense_table_log_context(
            file_label,
            page_title,
            dense_region,
        ),
    )
    routing_metadata = _build_dense_table_routing_metadata(
        page_title,
        eda,
        description,
        page_metadata,
        dense_region,
    )
    replacement_content = _build_replacement_content(
        page_title,
        eda,
        description,
        routing_metadata,
    )

    return PreparedDenseTable(
        region_id=str(routing_metadata.get("selected_region_id", "")).strip(),
        used_range=str(routing_metadata.get("used_range", "")).strip(),
        routing_metadata=routing_metadata,
        raw_content=_extract_dense_table_raw_rows(dense_region),
        replacement_content=replacement_content,
        dense_table_eda=eda,
        dense_table_description=description,
        description_generation_mode=generation_mode,
    )


def _escape_markdown_table_cell(text: str) -> str:
    """Escape markdown table cell text. Params: text. Returns: str."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("|", "\\|").replace("\n", " ")


def _escape_table_cell(text: str) -> str:
    """Backward-compatible wrapper for markdown table cell escaping."""
    return _escape_markdown_table_cell(text)


def _build_column_table(eda: Any, description: DenseTableDescription) -> str:
    """Build the column description pipe table."""
    desc_by_position, desc_by_name = _build_description_lookup(description)
    rows = []
    for col in eda.columns:
        col_desc = _lookup_column_description(
            col,
            desc_by_position,
            desc_by_name,
        )
        rows.append(
            f"| {_escape_markdown_table_cell(col.name)} | {col.dtype}"
            " | "
            f"{_escape_markdown_table_cell(col_desc)} |"
        )
    return "\n".join(
        ["| Column | Type | Description |", "| --- | --- | --- |"] + rows
    )


def _build_data_preview(eda: Any) -> str:
    """Build the data preview pipe table from EDA samples."""
    col_names = [_escape_markdown_table_cell(col.name) for col in eda.columns]
    header = "| Row | " + " | ".join(col_names) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(col_names)) + " |"
    return "\n".join([header, sep] + eda.sample_rows[:5])


def prepare_xlsx_page(
    page: dict[str, Any],
    llm: LLMClient,
    file_label: str = "",
) -> PreparedPage:
    """Process a single page through content preparation.

    Params:
        page: Page dict from extraction result
        llm: LLMClient instance

    Returns:
        PreparedPage with full content and optional dense table data
    """
    page_number = page["page_number"]
    page_title = page["page_title"]
    content = page["content"]
    prepared_content = content
    original_content = ""
    eda = None
    description = None
    description_generation_mode = ""
    method = "passthrough"
    dense_tables: list[PreparedDenseTable] = []

    if _is_dense_table_candidate(page):
        original_content = content
        dense_regions = _get_dense_table_regions(page.get("metadata", {}))
        if dense_regions:
            for dense_region in dense_regions:
                dense_tables.append(
                    _prepare_dense_table_region(
                        page_title=page_title,
                        content=content,
                        llm=llm,
                        page_metadata=page.get("metadata", {}),
                        dense_region=dense_region,
                        file_label=file_label,
                    )
                )
        else:
            dense_tables.append(
                _prepare_dense_table_region(
                    page_title=page_title,
                    content=content,
                    llm=llm,
                    page_metadata=page.get("metadata", {}),
                    dense_region=None,
                    file_label=file_label,
                )
            )

        eda = dense_tables[0].dense_table_eda
        description = dense_tables[0].dense_table_description
        description_generation_mode = dense_tables[
            0
        ].description_generation_mode
        prepared_content = _splice_dense_table_replacements(
            content,
            dense_tables,
        )
        method = "dense_table_replaced"
    else:
        if page.get("method") == "xlsx_sheet_classification":
            prepared_content = _strip_region_markers(content)

    return PreparedPage(
        page_number=page_number,
        page_title=page_title,
        content=prepared_content,
        method=method,
        metadata=page.get("metadata", {}),
        original_content=original_content,
        dense_tables=dense_tables,
        dense_table_eda=eda,
        dense_table_description=description,
        description_generation_mode=description_generation_mode,
    )
