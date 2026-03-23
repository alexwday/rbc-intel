"""Focused markdown escaping tests for dense-table content preparation."""

from ingestion.stages.content_preparation import (
    _build_column_table,
    _build_data_preview,
    _build_description_message,
    _escape_markdown_table_cell,
    _format_filter_line,
)
from ingestion.utils.content_types import (
    ColumnProfile,
    DenseTableDescription,
    TableEDA,
)


def _make_escaping_eda() -> TableEDA:
    """Build an EDA fixture with multiline and pipe-containing headers."""
    return TableEDA(
        row_count=2,
        columns=[
            ColumnProfile(
                name="Product\nName",
                position="A",
                dtype="text",
                stats={"value_distribution": {"Widget": 2}},
                sample_values=["Widget", "Widget XL"],
                non_null_count=2,
                null_count=0,
                unique_count=2,
            ),
            ColumnProfile(
                name="Risk | Band",
                position="B",
                dtype="text",
                stats={"value_distribution": {"High": 1, "Low": 1}},
                sample_values=["High", "Low"],
                non_null_count=2,
                null_count=0,
                unique_count=2,
            ),
        ],
        header_row=1,
        framing_context="# Sheet: Escaping",
        sample_rows=["| 2 | Widget<br>XL | High\\|Low |"],
        token_count=100,
        used_range="A1:B3",
        header_mode="header_row",
        source_region_id="region_1",
    )


def test_escape_markdown_table_cell_escapes_pipe_and_newline():
    """Markdown cell escaping normalizes pipes and line breaks."""
    assert _escape_markdown_table_cell("A|B\nC\r\nD") == "A\\|B C D"


def test_build_description_message_escapes_multiline_headers():
    """Prompt message uses escaped header text in profiles and preview."""
    message = _build_description_message("Escaping", _make_escaping_eda())

    assert "Product Name" in message
    assert "Risk \\| Band" in message
    assert "Product\nName" not in message
    assert "| Row | Product Name | Risk \\| Band |" in message


def test_build_column_table_escapes_multiline_headers_and_descriptions():
    """Replacement column table escapes both headers and descriptions."""
    description = DenseTableDescription(
        description="Escaping dataset.",
        column_descriptions=[
            {
                "position": "A",
                "name": "Product\nName",
                "description": "Full product\nname",
            },
            {
                "position": "B",
                "name": "Risk | Band",
                "description": "Risk | grouping",
            },
        ],
        filter_columns=["B"],
        identifier_columns=[],
        measure_columns=[],
        text_content_columns=["A"],
        sample_queries=["What products are in each risk band?"],
    )

    table = _build_column_table(_make_escaping_eda(), description)

    assert "| Product Name | text | Full product name |" in table
    assert "| Risk \\| Band | text | Risk \\| grouping |" in table
    assert "Product\nName" not in table


def test_build_data_preview_escapes_multiline_headers():
    """Data preview table escapes header cells before joining sample rows."""
    preview = _build_data_preview(_make_escaping_eda())

    assert "| Row | Product Name | Risk \\| Band |" in preview
    assert "Widget<br>XL" in preview
    assert "High\\|Low" in preview
    assert "Product\nName" not in preview


def test_format_filter_line_escapes_multiline_values():
    """Filter-role summaries normalize multiline categorical values."""
    col = ColumnProfile(
        name="Status",
        position="C",
        dtype="text",
        stats={"value_distribution": {"Open\nPending": 2, "Closed|Won": 1}},
        sample_values=["Open\nPending", "Closed|Won"],
        non_null_count=3,
        null_count=0,
        unique_count=2,
    )

    line = _format_filter_line(col)

    assert "Open Pending" in line
    assert "Closed\\|Won" in line
    assert "Open\nPending" not in line


def test_format_filter_line_escapes_multiline_samples():
    """High-cardinality filter summaries escape sample values too."""
    col = ColumnProfile(
        name="Account",
        position="D",
        dtype="text",
        stats={},
        sample_values=["ACC\n001", "ACC|002"],
        non_null_count=2,
        null_count=0,
        unique_count=2,
    )

    line = _format_filter_line(col)

    assert "ACC 001" in line
    assert "ACC\\|002" in line
    assert "ACC\n001" not in line
