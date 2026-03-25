"""
Dense Table Subagent — sub-retrieval for dense table regions.

When the file research subagent encounters a chunk with
``is_dense_table_description=True``, this module fetches the raw table data
from ``document_dense_tables`` and formats it for injection into the LLM
research context.

The dense table description (already in the chunk content) summarises the
table columns and provides routing metadata.  The raw data provides the
actual rows so the LLM can extract specific values.

Flow:
    1. file_research_subagent detects ``is_dense_table_description`` on a chunk
    2. Calls ``fetch_dense_table_context`` with document_id + routing JSON
    3. This module queries ``document_dense_tables`` for the matching region
    4. Returns formatted XML that is appended after the chunk content
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from ...connections.postgres import get_database_session, get_database_schema

logger = logging.getLogger(__name__)


class DenseTableSubagentError(Exception):
    """Exception raised for dense table retrieval errors."""


def _parse_routing_metadata(
    routing_json: Any,
) -> Optional[Dict[str, Any]]:
    """Parse dense_table_routing_json from chunk metadata.

    Args:
        routing_json: JSON string or dict from chunk metadata.

    Returns:
        Parsed dict, or None if parsing fails.
    """
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
            logger.warning("Failed to parse dense_table_routing_json")
    return None


def _fetch_dense_table_row(
    document_id: str,
    region_id: str,
) -> Optional[Dict[str, Any]]:
    """Query document_dense_tables for a specific region.

    Args:
        document_id: Parent document identifier.
        region_id: Dense table region identifier.

    Returns:
        Row dict with raw_content_json, sheet_name, used_range, etc.,
        or None if not found.
    """
    schema = get_database_schema()

    try:
        with get_database_session() as session:
            result = session.execute(
                text(
                    f"""
                    SELECT
                        dense_table_id,
                        region_id,
                        sheet_name,
                        used_range,
                        page_title,
                        replacement_content,
                        routing_metadata_json,
                        raw_content_json
                    FROM {schema}.document_dense_tables
                    WHERE document_id = :document_id
                      AND region_id = :region_id
                    LIMIT 1
                    """
                ),
                {"document_id": document_id, "region_id": region_id},
            )
            row = result.mappings().first()

            if not row:
                logger.debug(
                    "No dense table found for document_id=%s, region_id=%s",
                    document_id,
                    region_id,
                )
                return None

            return dict(row)

    except Exception as exc:
        logger.error(
            "Error fetching dense table for document_id=%s, region_id=%s: %s",
            document_id,
            region_id,
            exc,
        )
        return None


def _format_raw_content_xml(
    raw_content_json: Any,
    sheet_name: str,
    region_id: str,
    used_range: str,
) -> str:
    """Format raw dense table content as XML for LLM context.

    Args:
        raw_content_json: JSONB payload containing serialized table rows.
        sheet_name: Workbook sheet name.
        region_id: Region identifier.
        used_range: Cell range string (e.g. 'A1:J101').

    Returns:
        XML string suitable for injection after chunk content.
    """
    if raw_content_json is None:
        return ""

    # Parse if needed
    if isinstance(raw_content_json, str):
        try:
            raw_content = json.loads(raw_content_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse raw_content_json as JSON")
            return ""
    else:
        raw_content = raw_content_json

    # Build the XML block
    lines: List[str] = []

    # Count rows for the attribute
    row_count = 0
    col_count = 0
    if isinstance(raw_content, list):
        row_count = len(raw_content)
        if raw_content and isinstance(raw_content[0], dict):
            col_count = len(raw_content[0].get("cells", []))
    elif isinstance(raw_content, dict):
        rows = raw_content.get("rows", [])
        row_count = len(rows)
        if rows and isinstance(rows[0], dict):
            col_count = len(rows[0].get("cells", []))

    lines.append(
        f'  <dense_table_data sheet="{sheet_name}" '
        f'region="{region_id}" used_range="{used_range}" '
        f'rows="{row_count}" columns="{col_count}">'
    )

    # Serialize the raw content as indented JSON for the LLM
    try:
        content_str = json.dumps(raw_content, indent=2, default=str)
        for content_line in content_str.splitlines():
            lines.append(f"    {content_line}")
    except (TypeError, ValueError):
        lines.append("    [raw content could not be serialized]")

    lines.append("  </dense_table_data>")

    return "\n".join(lines)


def fetch_dense_table_context(
    document_id: str,
    routing_json: Any,
) -> Optional[str]:
    """Fetch and format dense table data for a chunk.

    Called from file_research_subagent when a chunk has
    ``is_dense_table_description=True``.

    Args:
        document_id: Parent document identifier.
        routing_json: The ``dense_table_routing_json`` from the chunk.

    Returns:
        Formatted XML string to append after chunk content, or None if
        the dense table cannot be retrieved or routing is missing.
    """
    routing = _parse_routing_metadata(routing_json)
    if not routing:
        logger.debug(
            "No routing metadata for dense table chunk in document %s",
            document_id,
        )
        return None

    region_id = routing.get("selected_region_id") or routing.get("region_id")
    if not region_id:
        logger.debug(
            "No selected_region_id or region_id in routing metadata "
            "for document %s",
            document_id,
        )
        return None

    row = _fetch_dense_table_row(document_id, region_id)
    if not row:
        return None

    sheet_name = row.get("sheet_name", "")
    used_range = row.get("used_range", "")
    raw_content_json = row.get("raw_content_json")

    if not raw_content_json:
        logger.debug(
            "Dense table row found but raw_content_json is empty "
            "for document_id=%s, region_id=%s",
            document_id,
            region_id,
        )
        return None

    return _format_raw_content_xml(
        raw_content_json, sheet_name, region_id, used_range
    )


def get_dense_table_regions_for_document(
    document_id: str,
) -> List[Dict[str, Any]]:
    """Fetch all dense table regions for a document.

    Useful for pre-loading all dense table data before chunk formatting,
    avoiding N+1 queries when a document has multiple dense table chunks.

    Args:
        document_id: Parent document identifier.

    Returns:
        List of dense table row dicts.
    """
    schema = get_database_schema()

    try:
        with get_database_session() as session:
            result = session.execute(
                text(
                    f"""
                    SELECT
                        dense_table_id,
                        region_id,
                        sheet_name,
                        used_range,
                        page_title,
                        routing_metadata_json,
                        raw_content_json
                    FROM {schema}.document_dense_tables
                    WHERE document_id = :document_id
                    ORDER BY region_id
                    """
                ),
                {"document_id": document_id},
            )
            return [dict(row) for row in result.mappings().all()]

    except Exception as exc:
        logger.error(
            "Error fetching dense tables for document_id=%s: %s",
            document_id,
            exc,
        )
        return []
