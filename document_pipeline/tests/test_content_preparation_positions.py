"""Position-based dense table description tests."""

import json

from ingestion.processors.xlsx.content_preparation import (
    _build_column_roles_section,
    _build_column_table,
    _parse_description_response,
)
from ingestion.processors.xlsx.types import (
    ColumnProfile,
    DenseTableDescription,
    TableEDA,
)


def _make_position_response() -> dict:
    """Build a position-keyed dense table description response."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(
                                    {
                                        "description": "A dataset.",
                                        "column_descriptions": [
                                            {
                                                "position": "A",
                                                "name": "Status",
                                                "description": "Record status",
                                            },
                                            {
                                                "position": "C",
                                                "name": "Amount (C)",
                                                "description": "Expected amt",
                                            },
                                        ],
                                        "filter_columns": ["A"],
                                        "identifier_columns": ["B"],
                                        "measure_columns": ["C"],
                                        "text_content_columns": [],
                                        "sample_queries": [
                                            "What is the total?"
                                        ],
                                    }
                                )
                            }
                        }
                    ]
                }
            }
        ]
    }


def _make_position_eda() -> TableEDA:
    """Build a dense-table EDA object with position-keyed columns."""
    return TableEDA(
        row_count=5,
        columns=[
            ColumnProfile(
                "Status",
                "A",
                "text",
                {"value_distribution": {"Open": 2, "Closed": 3}},
                ["Open", "Closed"],
                5,
                0,
                2,
            ),
            ColumnProfile(
                "Amount (B)",
                "B",
                "numeric",
                {"min": 100.0, "max": 200.0, "mean": 150.0},
                ["100", "200"],
                5,
                0,
                5,
            ),
            ColumnProfile(
                "Amount (C)",
                "C",
                "numeric",
                {"min": 300.0, "max": 400.0, "mean": 350.0},
                ["300", "400"],
                5,
                0,
                5,
            ),
        ],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=100,
        used_range="A1:C6",
        header_mode="header_row",
        source_region_id="region_2",
    )


def test_parse_description_response_preserves_position_keys():
    """Parser keeps position-keyed column descriptions and roles."""
    description = _parse_description_response(_make_position_response())

    assert description.column_descriptions == [
        {
            "position": "A",
            "name": "Status",
            "description": "Record status",
        },
        {
            "position": "C",
            "name": "Amount (C)",
            "description": "Expected amt",
        },
    ]
    assert description.filter_columns == ["A"]
    assert description.identifier_columns == ["B"]
    assert description.measure_columns == ["C"]


def test_build_column_roles_section_uses_position_index():
    """Role lookups resolve columns by position, not only by name."""
    description = DenseTableDescription(
        description="A dataset.",
        column_descriptions=[],
        filter_columns=["A"],
        identifier_columns=["B"],
        measure_columns=["C"],
        text_content_columns=[],
        sample_queries=[],
    )

    result = _build_column_roles_section(_make_position_eda(), description)

    assert "### Filter Columns" in result
    assert "Status (A)" in result
    assert "### Identifiers" in result
    assert "Amount (B)" in result
    assert "### Measures" in result
    assert "300.0 to 400.0" in result
    assert "100.0 to 200.0" not in result


def test_build_column_table_uses_position_descriptions_without_collision():
    """Position-keyed descriptions map cleanly onto duplicate-like columns."""
    description = DenseTableDescription(
        description="A dataset.",
        column_descriptions=[
            {
                "position": "B",
                "name": "Amount (B)",
                "description": "Booked amount",
            },
            {
                "position": "C",
                "name": "Amount (C)",
                "description": "Expected amt",
            },
        ],
        filter_columns=[],
        identifier_columns=[],
        measure_columns=[],
        text_content_columns=[],
        sample_queries=[],
    )

    table = _build_column_table(_make_position_eda(), description)

    assert "| Amount (B) | numeric | Booked amount |" in table
    assert "| Amount (C) | numeric | Expected amt |" in table
