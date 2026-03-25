"""Tests for dense-table description generation mode persistence."""

from unittest.mock import MagicMock, patch

from ingestion.processors.xlsx.content_preparation import prepare_xlsx_page
from ingestion.processors.xlsx.types import (
    ColumnProfile,
    DenseTableDescription,
    TableEDA,
)


def _make_dense_page() -> dict:
    """Build a dense table candidate page fixture."""
    return {
        "page_number": 1,
        "page_title": "Transactions",
        "content": "raw dense table",
        "metadata": {"handling_mode": "dense_table_candidate"},
    }


def _make_dense_description() -> DenseTableDescription:
    """Build a dense table description fixture."""
    return DenseTableDescription(
        description="Dense dataset.",
        column_descriptions=[
            {
                "position": "A",
                "name": "Date",
                "description": "Transaction date",
            }
        ],
        filter_columns=["A"],
        identifier_columns=[],
        measure_columns=[],
        text_content_columns=[],
        sample_queries=["When did transactions occur?"],
    )


def _make_dense_eda() -> TableEDA:
    """Build a dense table EDA fixture."""
    return TableEDA(
        row_count=2,
        columns=[ColumnProfile("Date", "A", "date", {}, [], 2, 0, 2)],
        header_row=1,
        framing_context="",
        sample_rows=[],
        token_count=100,
    )


@patch("ingestion.processors.xlsx.content_preparation._describe_dense_table")
def test_prepare_page_persists_description_generation_mode(
    mock_describe,
):
    """Prepared dense pages persist the description generation mode."""
    mock_describe.return_value = (
        _make_dense_eda(),
        _make_dense_description(),
        "deterministic_fallback",
    )

    result = prepare_xlsx_page(_make_dense_page(), MagicMock())

    assert result.description_generation_mode == "deterministic_fallback"


def test_prepare_page_passthrough_leaves_generation_mode_empty():
    """Non-dense pages do not set a dense-table generation mode."""
    result = prepare_xlsx_page(
        {
            "page_number": 1,
            "page_title": "Notes",
            "content": "content",
            "metadata": {"handling_mode": "page_like"},
        },
        MagicMock(),
    )

    assert result.description_generation_mode == ""
