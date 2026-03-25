"""Tests for content preparation stage orchestrator."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ingestion.processors.xlsx.content_preparation import (
    _build_column_roles_section,
    _build_description_message,
    _build_dense_table_routing_metadata,
    _build_region_framing_context,
    _build_replacement_content,
    _describe_dense_table,
    _escape_table_cell,
    _format_column_profile,
    _format_filter_line,
    _get_dense_table_regions,
    _is_dense_table_candidate,
    _parse_description_response,
)
from ingestion.stages.content_preparation import (
    _extract_filename,
    _load_extraction_results,
    _prepare_file,
    _prepare_page,
    _write_result,
    run_content_preparation,
)
from ingestion.utils.file_types import ContentPreparationResult, PreparedPage
from ingestion.processors.xlsx.types import (
    ColumnProfile,
    DenseTableDescription,
    PreparedDenseTable,
    TableEDA,
)

# ── helpers ──────────────────────────────────────────────────


def _make_page(**overrides):
    """Build a minimal page dict for testing."""
    defaults = {
        "page_number": 1,
        "page_title": "Test Page",
        "content": "Some content here.",
        "method": "full_dpi",
        "error": "",
        "metadata": {},
    }
    defaults.update(overrides)
    return defaults


def _make_extraction(**overrides):
    """Build a minimal extraction dict for testing."""
    defaults = {
        "file_path": "/data/src/test.pdf",
        "filetype": "pdf",
        "pages": [_make_page()],
        "total_pages": 1,
        "pages_succeeded": 1,
        "pages_failed": 0,
    }
    defaults.update(overrides)
    return defaults


def _make_dense_page(**overrides):
    """Build a page classified as dense_table_candidate."""
    page = _make_page(
        page_title="Transactions",
        content=(
            "# Sheet: Transactions\n"
            "- Sheet type: worksheet\n\n"
            "<!-- region:region_1 start -->\n"
            "| Row | A | B |\n"
            "| --- | --- | --- |\n"
            "| 1 | Portfolio | Summary |\n"
            "<!-- region:region_1 end -->\n\n"
            "<!-- region:region_2 start -->\n"
            "| Row | A | B |\n"
            "| --- | --- | --- |\n"
            "| 3 | 2024-01-01 | 5000 |\n"
            "| 4 | 2024-02-01 | 3000 |\n"
            "| 5 | 2024-03-01 | 7500 |\n"
            "<!-- region:region_2 end -->"
        ),
        method="xlsx_sheet_classification",
        metadata={
            "handling_mode": "dense_table_candidate",
            "sheet_name": "Transactions",
        },
    )
    page.update(overrides)
    return page


def _make_dense_region_metadata() -> dict[str, object]:
    """Build dense-table region metadata for XLSX pages."""
    return {
        "handling_mode": "dense_table_candidate",
        "sheet_name": "Transactions",
        "dense_table_used_range": "A3:B5",
        "framing_summary": (
            "- region_1: framing, range=A1:B1, rows=1, columns=2, "
            "sample=Portfolio Summary"
        ),
        "dense_table_region": {
            "region_id": "region_2",
            "used_range": "A3:B5",
            "row_numbers": [3, 4, 5],
            "column_numbers": [1, 2],
            "rows": [
                {
                    "row_number": 3,
                    "cells": [
                        {"column_number": 1, "value": "2024-01-01"},
                        {"column_number": 2, "value": "5000"},
                    ],
                },
                {
                    "row_number": 4,
                    "cells": [
                        {"column_number": 1, "value": "2024-02-01"},
                        {"column_number": 2, "value": "3000"},
                    ],
                },
                {
                    "row_number": 5,
                    "cells": [
                        {"column_number": 1, "value": "2024-03-01"},
                        {"column_number": 2, "value": "7500"},
                    ],
                },
            ],
        },
    }


def _make_second_dense_region() -> dict[str, object]:
    """Build a second dense-table region for multi-region tests."""
    return {
        "region_id": "region_3",
        "used_range": "D3:E5",
        "row_numbers": [3, 4, 5],
        "column_numbers": [4, 5],
        "rows": [
            {
                "row_number": 3,
                "cells": [
                    {"column_number": 4, "value": "East"},
                    {"column_number": 5, "value": "3000"},
                ],
            },
            {
                "row_number": 4,
                "cells": [
                    {"column_number": 4, "value": "West"},
                    {"column_number": 5, "value": "2500"},
                ],
            },
            {
                "row_number": 5,
                "cells": [
                    {"column_number": 4, "value": "North"},
                    {"column_number": 5, "value": "4100"},
                ],
            },
        ],
    }


def _make_llm_description_response():
    """Build a mock LLM dense table description response."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": ("describe_dense_table"),
                                "arguments": json.dumps(
                                    {
                                        "description": "A log.",
                                        "column_descriptions": [
                                            {
                                                "name": "Date",
                                                "description": "Txn date",
                                            },
                                            {
                                                "name": "Amount",
                                                "description": "Dollars",
                                            },
                                        ],
                                        "filter_columns": ["Date"],
                                        "identifier_columns": [],
                                        "measure_columns": ["Amount"],
                                        "text_content_columns": [],
                                        "sample_queries": [
                                            "What is the total?"
                                        ],
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }


def _make_routing_metadata(**overrides):
    """Build deterministic dense-table routing metadata for tests."""
    metadata = {
        "sheet_name": "Transactions",
        "selected_region_id": "region_2",
        "used_range": "A3:B5",
        "source_region_id": "region_2",
        "dense_table_region_ids": ["region_2", "region_3"],
        "filter_columns": ["Date"],
        "identifier_columns": [],
        "measure_columns": ["Amount"],
        "text_content_columns": [],
        "sample_queries": ["What is the total?"],
    }
    metadata.update(overrides)
    return metadata


# ── _escape_table_cell ──────────────────────────────────────


def test_escape_table_cell_pipe():
    """Escapes pipe characters in table cell text."""
    assert _escape_table_cell("A | B") == "A \\| B"


def test_escape_table_cell_newline():
    """Replaces newlines with spaces."""
    assert _escape_table_cell("Product\nName") == "Product Name"


def test_escape_table_cell_both():
    """Handles pipes and newlines together."""
    assert _escape_table_cell("A|B\nC") == "A\\|B C"


def test_escape_table_cell_no_specials():
    """Leaves clean text unchanged."""
    assert _escape_table_cell("no specials") == "no specials"


def test_escape_table_cell_empty():
    """Returns empty string for empty input."""
    assert _escape_table_cell("") == ""


# ── _is_dense_table_candidate ───────────────────────────────


def test_is_dense_table_candidate_true():
    """Detects dense_table_candidate metadata."""
    page = _make_page(metadata={"handling_mode": "dense_table_candidate"})
    assert _is_dense_table_candidate(page)


def test_is_dense_table_candidate_false():
    """Rejects page_like pages."""
    page = _make_page(metadata={"handling_mode": "page_like"})
    assert not _is_dense_table_candidate(page)


def test_is_dense_table_candidate_no_metadata():
    """Rejects pages with no metadata."""
    page = _make_page()
    assert not _is_dense_table_candidate(page)


# ── _extract_filename ────────────────────────────────────────


def test_extract_filename():
    """Extracts basename from file path."""
    ext = {"file_path": "/data/src/report.pdf"}
    assert _extract_filename(ext) == "report.pdf"


def test_extract_filename_missing():
    """Falls back to 'unknown' when missing."""
    assert _extract_filename({}) == "unknown"


# ── _parse_description_response ──────────────────────────────


def test_parse_description_response_valid():
    """Parses valid LLM description response."""
    response = _make_llm_description_response()
    desc = _parse_description_response(response)
    assert desc.description == "A log."
    assert len(desc.column_descriptions) == 2
    assert desc.filter_columns == ["Date"]
    assert desc.measure_columns == ["Amount"]
    assert len(desc.sample_queries) == 1


def test_parse_description_response_missing_choices():
    """Raises on empty response."""
    with pytest.raises(ValueError, match="choices"):
        _parse_description_response({})


def test_parse_description_response_missing_message():
    """Raises on missing message."""
    with pytest.raises(ValueError, match="message"):
        _parse_description_response({"choices": [{}]})


def test_parse_description_response_missing_tool_calls():
    """Raises on missing tool calls."""
    with pytest.raises(ValueError, match="tool calls"):
        _parse_description_response({"choices": [{"message": {}}]})


def test_parse_description_response_missing_function():
    """Raises on missing function data."""
    with pytest.raises(ValueError, match="function data"):
        _parse_description_response(
            {"choices": [{"message": {"tool_calls": [{"id": "x"}]}}]}
        )


def test_parse_description_response_missing_arguments():
    """Raises on missing arguments."""
    with pytest.raises(ValueError, match="arguments"):
        _parse_description_response(
            {
                "choices": [
                    {"message": {"tool_calls": [{"function": {"name": "x"}}]}}
                ]
            }
        )


def test_parse_description_response_missing_fields():
    """Raises on missing required fields."""
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps({"description": "ok"})
                            }
                        }
                    ]
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="column_descriptions"):
        _parse_description_response(response)


def _make_partial_desc_response(**fields):
    """Build a description response with given fields."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"arguments": json.dumps(fields)}}
                    ]
                }
            }
        ]
    }


def test_parse_description_missing_description():
    """Raises on missing description string."""
    with pytest.raises(ValueError, match="description string"):
        _parse_description_response(
            _make_partial_desc_response(column_descriptions=[])
        )


def test_parse_description_missing_sample_queries():
    """Raises on missing sample_queries list."""
    with pytest.raises(ValueError, match="sample_queries"):
        _parse_description_response(
            _make_partial_desc_response(
                description="ok",
                column_descriptions=[],
            )
        )


def test_parse_description_non_list_filter_columns():
    """Non-list filter_columns defaults to empty list."""
    resp = _make_partial_desc_response(
        description="ok",
        column_descriptions=[],
        filter_columns="bad",
        sample_queries=["q"],
    )
    desc = _parse_description_response(resp)
    assert not desc.filter_columns


# ── _format_column_profile ───────────────────────────────────


def test_format_column_profile_with_distribution():
    """Includes value distribution for low-cardinality."""
    col = ColumnProfile(
        name="Status",
        position="A",
        dtype="text",
        stats={
            "value_distribution": {
                "Open": 10,
                "Closed": 90,
            }
        },
        sample_values=["Open", "Closed"],
        non_null_count=100,
        null_count=0,
        unique_count=2,
    )
    result = _format_column_profile(col)
    assert "Closed (90)" in result
    assert "Open (10)" in result
    assert "values:" in result


def test_format_column_profile_without_distribution():
    """Shows stats and samples for high-cardinality."""
    col = ColumnProfile(
        name="Amount",
        position="B",
        dtype="numeric",
        stats={"min": 1.0, "max": 100.0, "mean": 50.0},
        sample_values=["1.0", "50.0", "100.0"],
        non_null_count=100,
        null_count=0,
        unique_count=80,
    )
    result = _format_column_profile(col)
    assert "stats:" in result
    assert "samples:" in result


# ── _format_filter_line ──────────────────────────────────────


def test_format_filter_line_date():
    """Formats date column as range."""
    col = ColumnProfile(
        "Date",
        "A",
        "date",
        {"min": "2024-01-01", "max": "2024-12-31"},
        [],
        100,
        0,
        90,
    )
    line = _format_filter_line(col)
    assert "2024-01-01 to 2024-12-31" in line


def test_format_filter_line_categorical():
    """Formats categorical column with all values."""
    col = ColumnProfile(
        "Status",
        "B",
        "text",
        {"value_distribution": {"Open": 5, "Closed": 10}},
        ["Open", "Closed"],
        15,
        0,
        2,
    )
    line = _format_filter_line(col)
    assert "Closed" in line
    assert "Open" in line
    assert "2 values" in line


def test_format_filter_line_high_cardinality():
    """Formats high-cardinality with samples."""
    col = ColumnProfile(
        "Account",
        "C",
        "text",
        {},
        ["ACC-1", "ACC-2"],
        100,
        0,
        100,
    )
    line = _format_filter_line(col)
    assert "100 unique" in line
    assert "ACC-1" in line


def test_format_filter_line_no_info():
    """Formats column with no stats or samples."""
    col = ColumnProfile("Empty", "D", "text", {}, [], 0, 100, 0)
    line = _format_filter_line(col)
    assert "0 unique" in line


# ── _build_column_roles_section ──────────────────────────────


def test_build_column_roles_all_sections():
    """Builds all four role sections."""
    eda = TableEDA(
        row_count=10,
        columns=[
            ColumnProfile(
                "Status",
                "A",
                "text",
                {"value_distribution": {"Open": 5}},
                ["Open"],
                10,
                0,
                1,
            ),
            ColumnProfile(
                "ID",
                "B",
                "text",
                {},
                ["ID-1"],
                10,
                0,
                10,
            ),
            ColumnProfile(
                "Amount",
                "C",
                "numeric",
                {"min": 1.0, "max": 99.0},
                [],
                10,
                0,
                10,
            ),
            ColumnProfile(
                "Notes",
                "D",
                "text",
                {},
                ["note"],
                10,
                0,
                10,
            ),
        ],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=100,
    )
    desc = DenseTableDescription(
        description="test",
        column_descriptions=[],
        filter_columns=["Status"],
        identifier_columns=["ID"],
        measure_columns=["Amount"],
        text_content_columns=["Notes"],
        sample_queries=[],
    )
    result = _build_column_roles_section(eda, desc)
    assert "### Filter Columns" in result
    assert "Open" in result
    assert "### Identifiers" in result
    assert "ID" in result
    assert "### Measures" in result
    assert "1.0 to 99.0" in result
    assert "### Text Content" in result
    assert "Notes" in result


def test_build_column_roles_missing_column():
    """Gracefully handles LLM naming a column not in EDA."""
    eda = TableEDA(
        row_count=5,
        columns=[ColumnProfile("A", "A", "text", {}, [], 5, 0, 5)],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=100,
    )
    desc = DenseTableDescription(
        description="test",
        column_descriptions=[],
        filter_columns=["Nonexistent"],
        identifier_columns=[],
        measure_columns=["AlsoMissing"],
        text_content_columns=[],
        sample_queries=[],
    )
    result = _build_column_roles_section(eda, desc)
    assert "### Filter Columns" in result
    assert "### Measures" in result


def test_build_column_roles_measure_no_stats():
    """Measure column without min/max still listed."""
    eda = TableEDA(
        row_count=5,
        columns=[
            ColumnProfile(
                "Score",
                "A",
                "numeric",
                {},
                [],
                5,
                0,
                5,
            )
        ],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=100,
    )
    desc = DenseTableDescription(
        description="test",
        column_descriptions=[],
        filter_columns=[],
        identifier_columns=[],
        measure_columns=["Score"],
        text_content_columns=[],
        sample_queries=[],
    )
    result = _build_column_roles_section(eda, desc)
    assert "Score" in result


# ── _build_description_message ───────────────────────────────


def test_build_description_message():
    """Builds a complete description message with EDA stats."""
    eda = TableEDA(
        row_count=10,
        columns=[
            ColumnProfile(
                name="Status",
                position="A",
                dtype="text",
                stats={"value_distribution": {"Open": 5, "Closed": 5}},
                sample_values=["Open", "Closed"],
                non_null_count=10,
                null_count=0,
                unique_count=2,
            )
        ],
        header_row=1,
        framing_context="# Sheet: Test",
        sample_rows=["| 2 | Open |"],
        token_count=500,
        used_range="A1:A10",
        header_mode="header_row",
        source_region_id="region_1",
    )
    msg = _build_description_message("Test Sheet", eda)
    assert "Sheet name: Test Sheet" in msg
    assert "Used range: A1:A10" in msg
    assert "Header mode: header_row" in msg
    assert "Total data rows: 10" in msg
    assert "Status" in msg
    assert "Column profiles" in msg
    assert "Open (5)" in msg
    assert "Closed (5)" in msg


def test_build_dense_table_routing_metadata_uses_region_metadata():
    """Routing metadata preserves deterministic links back to sheet regions."""
    eda = TableEDA(
        row_count=3,
        columns=[
            ColumnProfile("Date", "A", "date", {}, [], 3, 0, 3),
            ColumnProfile("Amount", "B", "numeric", {}, [], 3, 0, 3),
        ],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=100,
        used_range="A3:B5",
        source_region_id="region_2",
    )
    desc = DenseTableDescription(
        description="Dataset.",
        column_descriptions=[],
        filter_columns=["Date"],
        identifier_columns=[],
        measure_columns=["Amount"],
        text_content_columns=[],
        sample_queries=["What is the total?"],
    )

    routing_metadata = _build_dense_table_routing_metadata(
        "Transactions",
        eda,
        desc,
        {
            **_make_dense_region_metadata(),
            "dense_table_regions": [
                {"region_id": "region_2"},
                {"region_id": "region_3"},
            ],
        },
    )

    assert routing_metadata["sheet_name"] == "Transactions"
    assert routing_metadata["selected_region_id"] == "region_2"
    assert routing_metadata["used_range"] == "A3:B5"
    assert routing_metadata["source_region_id"] == "region_2"
    assert routing_metadata["dense_table_region_ids"] == [
        "region_2",
        "region_3",
    ]
    assert routing_metadata["filter_columns"] == ["Date"]
    assert routing_metadata["measure_columns"] == ["Amount"]
    assert routing_metadata["sample_queries"] == ["What is the total?"]


def test_build_dense_table_routing_metadata_inserts_selected_region_id():
    """Selected dense regions are added to the routing region-id list."""
    eda = TableEDA(
        row_count=3,
        columns=[ColumnProfile("Amount", "B", "numeric", {}, [], 3, 0, 3)],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=100,
        used_range="A3:B5",
        source_region_id="region_2",
    )
    desc = DenseTableDescription(
        description="Dataset.",
        column_descriptions=[],
        filter_columns=[],
        identifier_columns=[],
        measure_columns=["Amount"],
        text_content_columns=[],
        sample_queries=[],
    )

    routing_metadata = _build_dense_table_routing_metadata(
        "Transactions",
        eda,
        desc,
        {
            "sheet_name": "Transactions",
            "dense_table_region": {
                "region_id": "region_2",
                "used_range": "A3:B5",
            },
            "dense_table_regions": [{"region_id": "region_3"}],
        },
    )

    assert routing_metadata["dense_table_region_ids"] == [
        "region_2",
        "region_3",
    ]


def test_build_region_framing_context_uses_fallback_used_range():
    """Region framing falls back when the dense region omits its range."""
    context = _build_region_framing_context(
        "Transactions",
        {
            "dense_table_used_range": "A3:B5",
            "framing_summary": "Context",
        },
        {"used_range": ""},
    )

    assert "Dense table used range: A3:B5" in context
    assert "Context" in context


def test_get_dense_table_regions_handles_fallback_shapes():
    """Dense region helpers handle non-dicts and single-region metadata."""
    assert _get_dense_table_regions(None) == []
    assert _get_dense_table_regions(
        {"dense_table_region": {"region_id": "r1"}}
    ) == [{"region_id": "r1"}]


# ── _build_replacement_content ───────────────────────────────


def test_build_replacement_content():
    """Builds markdown replacement with all sections."""
    eda = TableEDA(
        row_count=100,
        columns=[
            ColumnProfile(
                name="Date",
                position="A",
                dtype="date",
                stats={},
                sample_values=[],
                non_null_count=100,
                null_count=0,
                unique_count=50,
            ),
            ColumnProfile(
                name="Amount",
                position="B",
                dtype="numeric",
                stats={},
                sample_values=[],
                non_null_count=100,
                null_count=0,
                unique_count=80,
            ),
        ],
        header_row=1,
        framing_context="",
        sample_rows=[
            "| 2 | 2024-01-01 | 5000 |",
            "| 3 | 2024-02-01 | 3000 |",
        ],
        token_count=5000,
    )
    desc = DenseTableDescription(
        description="A financial dataset.",
        column_descriptions=[
            {"name": "Date", "description": "The date"},
            {
                "name": "Amount",
                "description": "Dollar value",
            },
        ],
        filter_columns=["Date"],
        identifier_columns=[],
        measure_columns=[],
        text_content_columns=[],
        sample_queries=["What is the total?"],
    )
    content = _build_replacement_content(
        "Transactions",
        eda,
        desc,
        _make_routing_metadata(),
    )
    assert "# Dense Table: Transactions" in content
    assert "A financial dataset." in content
    assert "2 columns, 100 rows" in content
    assert "| Date | date | The date |" in content
    assert "## Subretrieval Routing" in content
    assert "Selected region ID: region_2" in content
    assert "## Column Roles" in content
    assert "## Sample Queries" in content
    assert "## Data Preview" in content


def test_build_replacement_content_escapes_pipes():
    """Column names with pipes are escaped in replacement content."""
    eda = TableEDA(
        row_count=5,
        columns=[
            ColumnProfile(
                name="Revenue | Net",
                position="A",
                dtype="numeric",
                stats={"min": 100, "max": 500, "mean": 300.0},
                sample_values=["100", "500"],
                non_null_count=5,
                null_count=0,
                unique_count=5,
            ),
        ],
        header_row=1,
        framing_context="",
        sample_rows=["| 2 | 100 |"],
        token_count=100,
    )
    desc = DenseTableDescription(
        description="Test data.",
        column_descriptions=[
            {
                "name": "Revenue | Net",
                "description": "Net revenue | after tax",
            },
        ],
        filter_columns=[],
        identifier_columns=[],
        measure_columns=["Revenue | Net"],
        text_content_columns=[],
        sample_queries=["Total?"],
    )
    content = _build_replacement_content(
        "Sheet",
        eda,
        desc,
        _make_routing_metadata(
            sheet_name="Sheet",
            measure_columns=["Revenue | Net"],
            sample_queries=["Total?"],
        ),
    )
    assert "Revenue \\| Net" in content
    assert "after tax" in content
    assert "| Revenue | Net |" not in content


def test_build_replacement_content_escapes_newlines():
    """Column names with newlines are flattened in replacement content."""
    eda = TableEDA(
        row_count=5,
        columns=[
            ColumnProfile(
                name="Product\nName",
                position="A",
                dtype="text",
                stats={"min_length": 3, "max_length": 10, "avg_length": 6.0},
                sample_values=["Widget"],
                non_null_count=5,
                null_count=0,
                unique_count=5,
            ),
        ],
        header_row=1,
        framing_context="",
        sample_rows=["| 2 | Widget |"],
        token_count=100,
    )
    desc = DenseTableDescription(
        description="Product catalog.",
        column_descriptions=[
            {
                "name": "Product\nName",
                "description": "Full product\nname",
            },
        ],
        filter_columns=[],
        identifier_columns=[],
        measure_columns=[],
        text_content_columns=["Product\nName"],
        sample_queries=["List products?"],
    )
    content = _build_replacement_content(
        "Sheet",
        eda,
        desc,
        _make_routing_metadata(
            sheet_name="Sheet",
            text_content_columns=["Product\nName"],
            sample_queries=["List products?"],
        ),
    )
    assert "Product Name" in content
    assert "Product\nName" not in content
    assert "Full product name" in content


# ── _describe_dense_table ────────────────────────────────────


@patch("ingestion.processors.xlsx.dense_table.load_prompt")
def test_describe_dense_table(mock_load_prompt):
    """Runs EDA and LLM description end-to-end."""
    mock_load_prompt.return_value = {
        "stage": "dense_table_description",
        "user_prompt": "Describe this table.",
        "system_prompt": "You are an analyst.",
        "tools": [{"type": "function", "function": {}}],
        "tool_choice": "required",
    }
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_llm_description_response()

    content = (
        "# Sheet: Test\n\n"
        "| Row | A | B |\n"
        "| --- | --- | --- |\n"
        "| 1 | Date | Amount |\n"
        "| 2 | 2024-01-01 | 100 |"
    )
    eda, desc, mode = _describe_dense_table("Test", content, mock_llm)
    assert eda.row_count == 1
    assert desc.description == "A log."
    assert mode == "llm_one_shot"
    mock_llm.call.assert_called_once()


@patch("ingestion.processors.xlsx.dense_table.load_prompt")
def test_describe_dense_table_uses_region_metadata(mock_load_prompt):
    """Dense table description prefers region metadata over sheet markdown."""
    mock_load_prompt.return_value = {
        "stage": "dense_table_description",
        "user_prompt": "Describe this table.",
        "system_prompt": "You are an analyst.",
        "tools": [{"type": "function", "function": {}}],
        "tool_choice": "required",
    }
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_llm_description_response()

    content = (
        "# Sheet: Transactions\n"
        "- Used range: A1:B5\n\n"
        "| Row | A | B |\n"
        "| --- | --- | --- |\n"
        "| 1 | Title | Value |\n"
        "| 3 | 2024-01-01 | 5000 |\n"
        "| 4 | 2024-02-01 | 3000 |"
    )
    eda, desc, mode = _describe_dense_table(
        "Transactions",
        content,
        mock_llm,
        _make_dense_region_metadata(),
    )

    prompt_text = mock_llm.call.call_args.kwargs["messages"][1]["content"]
    assert eda.used_range == "A3:B5"
    assert eda.source_region_id == "region_2"
    assert eda.header_mode == "headerless"
    assert eda.row_count == 3
    assert desc.description == "A log."
    assert mode == "llm_one_shot"
    assert "Used range: A3:B5" in prompt_text
    assert "Header mode: headerless" in prompt_text
    assert "Dense table used range: A3:B5" in prompt_text


@patch("ingestion.processors.xlsx.dense_table.load_prompt")
def test_describe_dense_table_falls_back_without_dense_region(
    mock_load_prompt,
):
    """Dict metadata without a dense region falls back to markdown parsing."""
    mock_load_prompt.return_value = {
        "stage": "dense_table_description",
        "user_prompt": "Describe this table.",
        "system_prompt": "You are an analyst.",
        "tools": [{"type": "function", "function": {}}],
        "tool_choice": "required",
    }
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_llm_description_response()

    content = (
        "# Sheet: Test\n"
        "- Used range: A1:B2\n\n"
        "| Row | A | B |\n"
        "| --- | --- | --- |\n"
        "| 1 | Date | Amount |\n"
        "| 2 | 2024-01-01 | 100 |"
    )
    eda, _, mode = _describe_dense_table(
        "Test",
        content,
        mock_llm,
        {"handling_mode": "dense_table_candidate"},
    )

    assert eda.used_range == "A1:B2"
    assert eda.header_mode == "header_row"
    assert mode == "llm_one_shot"


# ── _load_extraction_results ─────────────────────────────────


def test_load_extraction_results(tmp_path, monkeypatch):
    """Loads all JSON files from extraction directory."""
    ext_dir = tmp_path / "processing" / "extraction"
    ext_dir.mkdir(parents=True)
    ext = _make_extraction()
    (ext_dir / "test_abc123.json").write_text(json.dumps(ext))

    monkeypatch.setattr(
        "ingestion.stages.content_preparation.EXTRACTION_DIR",
        ext_dir,
    )
    results = _load_extraction_results()
    assert len(results) == 1
    assert results[0]["file_path"] == "/data/src/test.pdf"


def test_load_extraction_results_no_dir(tmp_path, monkeypatch):
    """Returns empty list when extraction dir missing."""
    monkeypatch.setattr(
        "ingestion.stages.content_preparation.EXTRACTION_DIR",
        tmp_path / "nonexistent",
    )
    results = _load_extraction_results()
    assert not results


def test_load_extraction_results_malformed(tmp_path, monkeypatch):
    """Skips malformed JSON files."""
    ext_dir = tmp_path / "processing" / "extraction"
    ext_dir.mkdir(parents=True)
    (ext_dir / "bad.json").write_text("not json")
    (ext_dir / "good.json").write_text(json.dumps(_make_extraction()))

    monkeypatch.setattr(
        "ingestion.stages.content_preparation.EXTRACTION_DIR",
        ext_dir,
    )
    results = _load_extraction_results()
    assert len(results) == 1


# ── _prepare_page ────────────────────────────────────────────


def test_prepare_page_passthrough():
    """Non-XLSX pages pass through unchanged."""
    result = _prepare_page(_make_page(), "pdf", MagicMock())
    assert result.method == "passthrough"
    assert result.content == "Some content here."
    assert result.original_content == ""
    assert result.dense_table_eda is None


def test_prepare_page_page_like_xlsx_strips_region_markers():
    """Page-like XLSX content drops internal region markers."""
    page = _make_page(
        method="xlsx_sheet_classification",
        content=(
            "# Sheet: Test\n\n"
            "<!-- region:region_1 start -->\n"
            "| Row | A |\n"
            "| --- | --- |\n"
            "| 1 | Value |\n"
            "<!-- region:region_1 end -->"
        ),
        metadata={"handling_mode": "page_like"},
    )

    result = _prepare_page(page, "xlsx", MagicMock())

    assert result.method == "passthrough"
    assert "<!-- region:" not in result.content
    assert "| 1 | Value |" in result.content


@patch("ingestion.processors.xlsx.content_preparation._describe_dense_table")
def test_prepare_page_dense_table_replaces_region_content(mock_describe):
    """Dense table pages splice replacements into the original sheet."""
    eda = TableEDA(
        row_count=3,
        columns=[ColumnProfile("Date", "A", "date", {}, [], 3, 0, 3)],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=500,
        used_range="A3:B5",
        source_region_id="region_2",
    )
    desc = DenseTableDescription(
        description="A dataset.",
        column_descriptions=[{"name": "Date", "description": "The date"}],
        filter_columns=["Date"],
        identifier_columns=[],
        measure_columns=[],
        text_content_columns=[],
        sample_queries=["When?"],
    )
    mock_describe.return_value = (eda, desc, "llm_one_shot")

    page = _make_dense_page(metadata=_make_dense_region_metadata())
    result = _prepare_page(page, "xlsx", MagicMock())

    assert result.method == "dense_table_replaced"
    assert result.original_content == page["content"]
    assert result.dense_table_eda is not None
    assert result.dense_table_description is not None
    assert result.description_generation_mode == "llm_one_shot"
    assert len(result.dense_tables) == 1
    assert result.dense_tables[0].routing_metadata["sheet_name"] == (
        "Transactions"
    )
    assert "# Dense Table: Transactions" in result.content
    assert "| 1 | Portfolio | Summary |" in result.content
    assert "<!-- region:" not in result.content
    assert result.dense_tables[0].raw_content == (
        _make_dense_region_metadata()["dense_table_region"]["rows"]
    )


@patch(
    "ingestion.processors.xlsx.content_preparation."
    "_prepare_dense_table_region"
)
def test_prepare_page_multiple_dense_tables(mock_prepare_region):
    """Multiple dense regions are spliced independently into one page."""
    dense_page = _make_dense_page(
        content=(
            "# Sheet: Transactions\n"
            "- Sheet type: worksheet\n\n"
            "<!-- region:region_1 start -->\n"
            "| Row | A | B |\n"
            "| --- | --- | --- |\n"
            "| 1 | Portfolio | Summary |\n"
            "<!-- region:region_1 end -->\n\n"
            "<!-- region:region_2 start -->\n"
            "| Row | A | B |\n"
            "| --- | --- | --- |\n"
            "| 3 | 2024-01-01 | 5000 |\n"
            "| 4 | 2024-02-01 | 3000 |\n"
            "| 5 | 2024-03-01 | 7500 |\n"
            "<!-- region:region_2 end -->\n\n"
            "<!-- region:region_3 start -->\n"
            "| Row | D | E |\n"
            "| --- | --- | --- |\n"
            "| 3 | East | 3000 |\n"
            "| 4 | West | 2500 |\n"
            "| 5 | North | 4100 |\n"
            "<!-- region:region_3 end -->"
        ),
        metadata={
            **_make_dense_region_metadata(),
            "dense_table_regions": [
                _make_dense_region_metadata()["dense_table_region"],
                _make_second_dense_region(),
            ],
        },
    )
    first_eda = TableEDA(
        row_count=3,
        columns=[ColumnProfile("Date", "A", "date", {}, [], 3, 0, 3)],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=100,
        used_range="A3:B5",
        source_region_id="region_2",
    )
    second_eda = TableEDA(
        row_count=3,
        columns=[ColumnProfile("Region", "D", "text", {}, [], 3, 0, 3)],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=100,
        used_range="D3:E5",
        source_region_id="region_3",
    )
    desc = DenseTableDescription(
        description="Dataset.",
        column_descriptions=[],
        filter_columns=[],
        identifier_columns=[],
        measure_columns=[],
        text_content_columns=[],
        sample_queries=["Question?"],
    )
    mock_prepare_region.side_effect = [
        PreparedDenseTable(
            region_id="region_2",
            used_range="A3:B5",
            routing_metadata=_make_routing_metadata(
                selected_region_id="region_2",
                source_region_id="region_2",
                used_range="A3:B5",
            ),
            raw_content=_make_dense_region_metadata()["dense_table_region"][
                "rows"
            ],
            replacement_content="# Dense A",
            dense_table_eda=first_eda,
            dense_table_description=desc,
            description_generation_mode="llm_one_shot",
        ),
        PreparedDenseTable(
            region_id="region_3",
            used_range="D3:E5",
            routing_metadata=_make_routing_metadata(
                selected_region_id="region_3",
                source_region_id="region_3",
                used_range="D3:E5",
            ),
            raw_content=_make_second_dense_region()["rows"],
            replacement_content="# Dense B",
            dense_table_eda=second_eda,
            dense_table_description=desc,
            description_generation_mode="llm_batched",
        ),
    ]

    result = _prepare_page(dense_page, "xlsx", MagicMock())

    assert result.method == "dense_table_replaced"
    assert len(result.dense_tables) == 2
    assert result.dense_table_eda == first_eda
    assert result.description_generation_mode == "llm_one_shot"
    assert "# Dense A" in result.content
    assert "# Dense B" in result.content
    assert "<!-- region:" not in result.content
    assert "Portfolio | Summary" in result.content
    assert result.dense_tables[0].raw_content == (
        _make_dense_region_metadata()["dense_table_region"]["rows"]
    )
    assert result.dense_tables[1].raw_content == (
        _make_second_dense_region()["rows"]
    )


@patch("ingestion.processors.xlsx.content_preparation._describe_dense_table")
def test_prepare_page_dense_table_missing_markers_falls_back(
    mock_describe,
):
    """Missing region markers fall back to standalone replacements."""
    eda = TableEDA(
        row_count=3,
        columns=[ColumnProfile("Date", "A", "date", {}, [], 3, 0, 3)],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=500,
        used_range="A3:B5",
        source_region_id="region_2",
    )
    desc = DenseTableDescription(
        description="A dataset.",
        column_descriptions=[{"name": "Date", "description": "The date"}],
        filter_columns=["Date"],
        identifier_columns=[],
        measure_columns=[],
        text_content_columns=[],
        sample_queries=["When?"],
    )
    mock_describe.return_value = (eda, desc, "llm_one_shot")

    page = _make_dense_page(
        content=(
            "# Sheet: Transactions\n"
            "- Sheet type: worksheet\n\n"
            "| Row | A | B |\n"
            "| --- | --- | --- |\n"
            "| 3 | 2024-01-01 | 5000 |"
        ),
        metadata=_make_dense_region_metadata(),
    )
    result = _prepare_page(page, "xlsx", MagicMock())

    assert result.method == "dense_table_replaced"
    assert result.content.startswith("# Dense Table: Transactions")
    assert result.dense_tables[0].raw_content == (
        _make_dense_region_metadata()["dense_table_region"]["rows"]
    )


# ── _prepare_file ────────────────────────────────────────────


@patch("ingestion.stages.content_preparation._prepare_page")
def test_prepare_file_basic(mock_prepare):
    """Processes all pages and counts dense-table splices."""
    mock_prepare.return_value = PreparedPage(
        page_number=1,
        page_title="Test",
        content="content",
        method="passthrough",
    )
    result = _prepare_file(_make_extraction(), MagicMock())
    assert result.file_path == "/data/src/test.pdf"
    assert len(result.pages) == 1
    assert result.dense_tables_spliced == 0


@patch("ingestion.stages.content_preparation._prepare_page")
def test_prepare_file_counts_dense_tables(mock_prepare):
    """Dense table counts track sheet regions, not page count."""
    first_dense_rows = _make_dense_region_metadata()["dense_table_region"][
        "rows"
    ]
    second_dense_rows = _make_second_dense_region()["rows"]
    dense_page = PreparedPage(
        page_number=1,
        page_title="Dense",
        content="# Dense Table: Transactions",
        method="dense_table_replaced",
        dense_tables=[
            PreparedDenseTable(
                region_id="region_2",
                used_range="A3:B5",
                routing_metadata=_make_routing_metadata(
                    selected_region_id="region_2"
                ),
                raw_content=first_dense_rows,
                replacement_content="# Dense A",
            ),
            PreparedDenseTable(
                region_id="region_3",
                used_range="D3:E5",
                routing_metadata=_make_routing_metadata(
                    selected_region_id="region_3",
                    used_range="D3:E5",
                    source_region_id="region_3",
                ),
                raw_content=second_dense_rows,
                replacement_content="# Dense B",
            ),
        ],
    )
    mock_prepare.return_value = dense_page

    result = _prepare_file(_make_extraction(), MagicMock())

    assert result.dense_tables_spliced == 2


# ── _write_result ────────────────────────────────────────────


def test_write_result(tmp_path, monkeypatch):
    """Writes preparation result to JSON."""
    prep_dir = tmp_path / "processing" / "content_preparation"
    monkeypatch.setattr(
        "ingestion.stages.content_preparation.CONTENT_PREP_DIR",
        prep_dir,
    )
    result = ContentPreparationResult(
        file_path="/data/src/report.pdf",
        filetype="pdf",
        pages=[
            PreparedPage(
                page_number=1,
                page_title="T",
                content="C",
                method="passthrough",
            )
        ],
        dense_tables_spliced=0,
    )
    _write_result(result)
    files = list(prep_dir.glob("*.json"))
    assert len(files) == 1
    assert files[0].name.startswith("report_")
    data = json.loads(files[0].read_text())
    assert data["file_path"] == "/data/src/report.pdf"


def test_write_result_path_hash(tmp_path, monkeypatch):
    """Different paths produce different filenames."""
    prep_dir = tmp_path / "processing" / "content_preparation"
    monkeypatch.setattr(
        "ingestion.stages.content_preparation.CONTENT_PREP_DIR",
        prep_dir,
    )
    for path in ("/dir_a/report.pdf", "/dir_b/report.pdf"):
        _write_result(
            ContentPreparationResult(
                file_path=path,
                filetype="pdf",
                pages=[],
                dense_tables_spliced=0,
            )
        )
    files = list(prep_dir.glob("*.json"))
    assert len(files) == 2


# ── run_content_preparation ──────────────────────────────────


@patch("ingestion.stages.content_preparation._write_result")
@patch("ingestion.stages.content_preparation._prepare_file")
@patch("ingestion.stages.content_preparation._load_extraction_results")
def test_run_content_preparation_basic(mock_load, mock_prepare, mock_write):
    """Processes and writes results for all files."""
    mock_load.return_value = [
        _make_extraction(file_path="/data/a.pdf"),
        _make_extraction(file_path="/data/b.pdf"),
    ]
    mock_prepare.return_value = ContentPreparationResult(
        file_path="/data/a.pdf",
        filetype="pdf",
        pages=[],
        dense_tables_spliced=0,
    )
    run_content_preparation(MagicMock())
    assert mock_prepare.call_count == 2
    assert mock_write.call_count == 2


@patch("ingestion.stages.content_preparation._load_extraction_results")
def test_run_content_preparation_empty(mock_load):
    """No extraction results exits early."""
    mock_load.return_value = []
    run_content_preparation(MagicMock())


@patch("ingestion.stages.content_preparation._write_result")
@patch("ingestion.stages.content_preparation._prepare_file")
@patch("ingestion.stages.content_preparation._load_extraction_results")
def test_run_content_preparation_failure(mock_load, mock_prepare, mock_write):
    """Per-file failure does not crash the stage."""
    mock_load.return_value = [
        _make_extraction(file_path="/data/a.pdf"),
        _make_extraction(file_path="/data/b.pdf"),
    ]
    mock_prepare.side_effect = [
        RuntimeError("fail"),
        ContentPreparationResult(
            file_path="/data/b.pdf",
            filetype="pdf",
            pages=[],
            dense_tables_spliced=0,
        ),
    ]
    run_content_preparation(MagicMock())
    assert mock_write.call_count == 1
