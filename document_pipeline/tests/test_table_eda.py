"""Tests for table EDA analysis."""

import pytest

from ingestion.utils.table_eda import (
    _classify_column_dtype,
    _compute_boolean_stats,
    _compute_date_stats,
    _compute_numeric_stats,
    _compute_text_stats,
    _dominant_non_null_type,
    _detect_type,
    _disambiguate_names,
    _estimate_tokens,
    _extract_sample_rows,
    _extract_used_range,
    _is_pipe_row,
    _is_label_like,
    _is_separator_row,
    _looks_like_data_row,
    _parse_header_row_num,
    _parse_numeric,
    _pick_samples,
    _profile_column,
    _score_header_cell,
    _split_pipe_row,
    _unescape_cell,
    _value_shape,
    build_column_names,
    detect_header_mode,
    run_table_eda,
    run_table_eda_from_region,
)

# ── _estimate_tokens ────────────────────────────────────────


def test_estimate_tokens_basic():
    """Returns positive integer for normal text."""
    assert _estimate_tokens("hello world") >= 2


def test_estimate_tokens_empty():
    """Returns 1 for empty string (minimum)."""
    assert _estimate_tokens("") == 1


# ── _unescape_cell ──────────────────────────────────────────


def test_unescape_pipe():
    """Reverses escaped pipe delimiter."""
    assert _unescape_cell("a\\|b") == "a|b"


def test_unescape_newline():
    """Reverses escaped newline."""
    assert _unescape_cell("line1<br>line2") == "line1\nline2"


# ── _split_pipe_row ─────────────────────────────────────────


def test_split_pipe_row_basic():
    """Splits a normal pipe table row."""
    cells = _split_pipe_row("| Row | A | B |")
    assert cells == ["Row", "A", "B"]


def test_split_pipe_row_escaped_pipe():
    """Escaped pipes stay within the cell."""
    cells = _split_pipe_row("| a\\|b | c |")
    assert cells == ["a\\|b", "c"]


def test_split_pipe_row_empty_cells():
    """Empty cells are preserved."""
    cells = _split_pipe_row("| | data | |")
    assert cells == ["", "data", ""]


# ── _is_separator_row / _is_pipe_row ────────────────────────


def test_is_separator_row_dashes():
    """Recognizes standard separator cells."""
    assert _is_separator_row(["---", "---", "---"])


def test_is_separator_row_aligned():
    """Recognizes colon-aligned separators."""
    assert _is_separator_row([":---", "---:", ":---:"])


def test_is_separator_row_not_separator():
    """Rejects data cells."""
    assert not _is_separator_row(["data", "---"])


def test_is_separator_row_empty():
    """Empty cell list is not a separator."""
    assert not _is_separator_row([])


def test_is_pipe_row_valid():
    """Recognizes pipe table rows."""
    assert _is_pipe_row("| a | b |")


def test_is_pipe_row_invalid():
    """Rejects non-pipe lines."""
    assert not _is_pipe_row("just text")


# ── _parse_numeric ───────────────────────────────────────────


def test_parse_numeric_integer():
    """Parses plain integer."""
    assert _parse_numeric("5000") == 5000.0


def test_parse_numeric_float():
    """Parses decimal number."""
    assert _parse_numeric("3.14") == 3.14


def test_parse_numeric_negative():
    """Parses negative number."""
    assert _parse_numeric("-42") == -42.0


def test_parse_numeric_paren_negative():
    """Parses accounting-style negatives."""
    assert _parse_numeric("(1,234.56)") == -1234.56


def test_parse_numeric_dollar():
    """Strips dollar sign."""
    assert _parse_numeric("$5,000") == 5000.0


def test_parse_numeric_percent():
    """Strips percent sign."""
    assert _parse_numeric("75%") == 75.0


def test_parse_numeric_empty():
    """Returns None for empty string."""
    assert _parse_numeric("") is None


def test_parse_numeric_text():
    """Returns None for non-numeric text."""
    assert _parse_numeric("hello") is None


def test_parse_numeric_dash_only():
    """Returns None for bare dash."""
    assert _parse_numeric("-") is None


# ── _detect_type ─────────────────────────────────────────────


def test_detect_type_null():
    """Empty values are null."""
    assert _detect_type("") == "null"
    assert _detect_type("   ") == "null"


def test_detect_type_boolean():
    """TRUE/FALSE are boolean."""
    assert _detect_type("TRUE") == "boolean"
    assert _detect_type("FALSE") == "boolean"
    assert _detect_type("true") == "boolean"


def test_detect_type_date():
    """Date patterns detected."""
    assert _detect_type("2024-01-15") == "date"
    assert _detect_type("2024-01-15 12:30:00") == "date"


def test_detect_type_numeric():
    """Numbers detected."""
    assert _detect_type("42") == "numeric"
    assert _detect_type("$1,000") == "numeric"


def test_detect_type_text():
    """Non-matching values are text."""
    assert _detect_type("hello") == "text"


# ── _classify_column_dtype ───────────────────────────────────


def test_classify_single_type():
    """Single non-null type returned directly."""
    assert _classify_column_dtype({"numeric": 10, "null": 2}) == "numeric"


def test_classify_mixed_types():
    """Multiple non-null types classified as mixed."""
    assert _classify_column_dtype({"numeric": 5, "text": 3}) == "mixed"


def test_classify_all_null():
    """All-null column defaults to text."""
    assert _classify_column_dtype({"null": 10}) == "text"


def test_classify_empty():
    """Empty counts default to text."""
    assert _classify_column_dtype({}) == "text"


# ── stats functions ──────────────────────────────────────────


def test_compute_numeric_stats():
    """Computes min, max, mean."""
    stats = _compute_numeric_stats([1.0, 2.0, 3.0])
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["mean"] == 2.0


def test_compute_numeric_stats_empty():
    """Empty list returns empty dict."""
    assert not _compute_numeric_stats([])


def test_compute_date_stats():
    """Computes min and max dates."""
    stats = _compute_date_stats(["2024-01-15", "2024-03-01", "2024-02-10"])
    assert stats["min"] == "2024-01-15"
    assert stats["max"] == "2024-03-01"


def test_compute_date_stats_empty():
    """Empty list returns empty dict."""
    assert not _compute_date_stats([])


def test_compute_text_stats_basic():
    """Computes length statistics."""
    stats = _compute_text_stats(["abc", "de", "fghij"], unique_count=3)
    assert stats["min_length"] == 2
    assert stats["max_length"] == 5
    assert "value_distribution" in stats


def test_compute_text_stats_high_cardinality():
    """No distribution for high cardinality."""
    stats = _compute_text_stats(
        [f"val{i}" for i in range(25)], unique_count=25
    )
    assert "value_distribution" not in stats


def test_compute_text_stats_empty():
    """Empty values returns empty dict."""
    assert not _compute_text_stats([], unique_count=0)


def test_compute_boolean_stats():
    """Counts true and false values."""
    stats = _compute_boolean_stats(["TRUE", "FALSE", "TRUE"])
    assert stats["true_count"] == 2
    assert stats["false_count"] == 1


# ── _pick_samples ────────────────────────────────────────────


def test_pick_samples_basic():
    """Picks distinct non-empty values."""
    samples = _pick_samples(["a", "b", "a", "c"])
    assert samples == ["a", "b", "c"]


def test_pick_samples_limit():
    """Respects count limit."""
    samples = _pick_samples([str(i) for i in range(20)], count=3)
    assert len(samples) == 3


def test_pick_samples_skips_empty():
    """Skips empty and whitespace values."""
    samples = _pick_samples(["", " ", "a", "b"])
    assert samples == ["a", "b"]


# ── _profile_column ──────────────────────────────────────────


def test_profile_column_numeric():
    """Profiles a numeric column."""
    col = _profile_column("A", "Amount", ["100", "200", "300"])
    assert col.dtype == "numeric"
    assert col.non_null_count == 3
    assert col.null_count == 0
    assert col.stats["min"] == 100.0


def test_profile_column_text():
    """Profiles a text column."""
    col = _profile_column("B", "Name", ["Alice", "Bob", ""])
    assert col.dtype == "text"
    assert col.non_null_count == 2
    assert col.null_count == 1


def test_profile_column_empty():
    """Profiles column with no values."""
    col = _profile_column("C", "Empty", [])
    assert col.dtype == "text"
    assert col.non_null_count == 0


# ── _extract_sample_rows ────────────────────────────────────


def test_extract_sample_rows_short():
    """Short list returned as-is."""
    rows = ["r1", "r2", "r3"]
    assert _extract_sample_rows(rows) == rows


def test_extract_sample_rows_long():
    """Long list returns first 5 + last 3."""
    rows = [f"r{i}" for i in range(20)]
    result = _extract_sample_rows(rows)
    assert len(result) == 8
    assert result[:5] == rows[:5]
    assert result[5:] == rows[-3:]


def test_extract_used_range():
    """Used range metadata is extracted when present and empty otherwise."""
    content = "# Sheet: Data\n- Used range: B2:D9\n| Row | B |"
    assert _extract_used_range(content) == "B2:D9"
    assert _extract_used_range("# Sheet: Data") == ""


def test_value_shape_collapse():
    """Value shape collapses letters and numbers into a simple signature."""
    assert _value_shape("ACC-1001") == "A-0"


def test_value_shape_preserves_whitespace_shape():
    """Whitespace transitions are reflected in the collapsed shape."""
    assert _value_shape("A 100") == "A 0"


def test_dominant_non_null_type():
    """Dominant non-null type prefers the most common detected type."""
    assert _dominant_non_null_type(["", "100", "200", "hello"]) == "numeric"


def test_dominant_non_null_type_all_null():
    """All-null values return the null sentinel."""
    assert _dominant_non_null_type(["", " "]) == "null"


def test_is_label_like():
    """Header-like labels are recognized and blank labels are rejected."""
    assert _is_label_like("Loan Status")
    assert _is_label_like("ENTITY_ID")
    assert not _is_label_like("East")
    assert not _is_label_like("")


def test_score_header_cell_empty_inputs():
    """Empty or missing values do not contribute header scores."""
    assert _score_header_cell("", ["100"]) == (0, 0)
    assert _score_header_cell("Amount", []) == (0, 0)


def test_score_header_cell_detects_text_header_over_numeric_column():
    """Header-style text scores against numeric data values."""
    assert _score_header_cell("Amount", ["100", "200"]) == (0, 2)


def test_score_header_cell_detects_matching_text_shapes_as_data():
    """Matching text shapes score toward headerless data detection."""
    assert _score_header_cell("A-1", ["A-2", "A-3"]) == (2, 0)


def test_score_header_cell_falls_back_to_generic_header_penalty():
    """Mismatched non-text types still count as header-like."""
    assert _score_header_cell("100", ["Open", "Closed"]) == (0, 1)


# ── run_table_eda ────────────────────────────────────────────


_SAMPLE_CONTENT = """\
# Sheet: Transactions
- Sheet type: worksheet
- Used range: A1:D5
- Populated grid: rows=5, columns=4

| Row | A | B | C | D |
| --- | --- | --- | --- | --- |
| 1 | Date | Amount | Category | Active |
| 2 | 2024-01-15 | 5000 | Loan | TRUE |
| 3 | 2024-02-20 | 3000 | Deposit | FALSE |
| 4 | 2024-03-10 | 7500 | Loan | TRUE |

## Visual Elements
- Charts: 0
- Images: 0"""


def test_run_table_eda_basic():
    """Profiles a standard XLSX markdown table."""
    eda = run_table_eda(_SAMPLE_CONTENT)
    assert eda.row_count == 3
    assert len(eda.columns) == 4
    assert eda.header_row == 1
    assert eda.columns[0].name == "Date"
    assert eda.columns[0].dtype == "date"
    assert eda.columns[1].name == "Amount"
    assert eda.columns[1].dtype == "numeric"
    assert eda.columns[2].name == "Category"
    assert eda.columns[2].dtype == "text"
    assert eda.columns[3].name == "Active"
    assert eda.columns[3].dtype == "boolean"


def test_run_table_eda_framing_context():
    """Extracts framing context from non-table content."""
    eda = run_table_eda(_SAMPLE_CONTENT)
    assert "Sheet: Transactions" in eda.framing_context
    assert "Visual Elements" in eda.framing_context
    assert "| Row |" not in eda.framing_context


def test_run_table_eda_sample_rows():
    """Sample rows are pipe table data strings."""
    eda = run_table_eda(_SAMPLE_CONTENT)
    assert len(eda.sample_rows) == 3
    assert "2024-01-15" in eda.sample_rows[0]


def test_run_table_eda_token_count():
    """Token count is positive."""
    eda = run_table_eda(_SAMPLE_CONTENT)
    assert eda.token_count > 0


def test_run_table_eda_no_pipe_table():
    """Content without pipe table returns empty EDA."""
    content = "# Just a header\n- Some bullet"
    eda = run_table_eda(content)
    assert eda.row_count == 0
    assert not eda.columns
    assert "Just a header" in eda.framing_context


def test_run_table_eda_header_only():
    """Table with header and separator but no data rows."""
    content = "| Row | A |\n| --- | --- |"
    eda = run_table_eda(content)
    assert eda.row_count == 0
    assert not eda.columns


def test_run_table_eda_header_and_names_only():
    """Table with header, separator, and column names but no data."""
    content = (
        "| Row | A | B |\n" + "| --- | --- | --- |\n" + "| 1 | Name | Value |"
    )
    eda = run_table_eda(content)
    assert eda.row_count == 0
    assert len(eda.columns) == 2
    assert eda.columns[0].name == "Name"
    assert eda.columns[1].name == "Value"


def test_run_table_eda_unsupported_format():
    """Raises ValueError for unsupported source format."""
    with pytest.raises(ValueError, match="Unsupported source_format"):
        run_table_eda("content", source_format="csv")


def test_run_table_eda_mixed_types():
    """Column with mixed types detected correctly."""
    content = (
        "| Row | A |\n"
        "| --- | --- |\n"
        "| 1 | Header |\n"
        "| 2 | 100 |\n"
        "| 3 | text |\n"
        "| 4 | 200 |"
    )
    eda = run_table_eda(content)
    assert eda.row_count == 3
    assert eda.columns[0].dtype == "mixed"


def test_run_table_eda_escaped_cells():
    """Handles escaped pipe characters in cells."""
    content = (
        "| Row | A |\n"
        + "| --- | --- |\n"
        + "| 1 | Name |\n"
        + "| 2 | a\\|b |"
    )
    eda = run_table_eda(content)
    assert eda.row_count == 1
    assert eda.columns[0].sample_values == ["a|b"]


def test_parse_header_row_num_no_row_col():
    """Returns 0 when table has no Row column."""
    assert _parse_header_row_num(["Name", "Value"], False) == 0


def test_parse_header_row_num_empty():
    """Returns 0 for empty first_cells."""
    assert _parse_header_row_num([], True) == 0


def test_run_table_eda_non_numeric_row_number():
    """Handles non-numeric first cell in header row."""
    content = (
        "| Row | A |\n"
        + "| --- | --- |\n"
        + "| abc | Name |\n"
        + "| 2 | value |"
    )
    eda = run_table_eda(content)
    assert eda.header_row == 0


def test_run_table_eda_fewer_names_than_positions():
    """Pads missing column names and disambiguates duplicates."""
    content = (
        "| Row | A | B | C |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | Name |\n"
        "| 2 | data | d2 | d3 |"
    )
    eda = run_table_eda(content)
    assert len(eda.columns) == 3
    assert eda.columns[1].name == " (B)"
    assert eda.columns[2].name == " (C)"


# ── _looks_like_data_row ────────────────────────────────────


def test_looks_like_data_row_all_text():
    """All-text row is a header, not data."""
    assert not _looks_like_data_row(["Entity_ID", "Name", "Status"])


def test_looks_like_data_row_has_numeric():
    """Row with majority numeric values is data."""
    assert _looks_like_data_row(["2024-01-01", "100", "200"])


def test_looks_like_data_row_has_date():
    """Row with a date value is data when majority non-text."""
    assert _looks_like_data_row(["2024-01-01", "100"])


def test_looks_like_data_row_minority_numeric():
    """Single numeric among text values is still a header."""
    assert not _looks_like_data_row(["2024", "Revenue", "Cost"])


def test_looks_like_data_row_all_null():
    """All-null row is ambiguous, defaults to header."""
    assert not _looks_like_data_row(["", "", ""])


def test_looks_like_data_row_empty():
    """Empty list is not data."""
    assert not _looks_like_data_row([])


def test_run_table_eda_headerless():
    """Headerless sheet uses column letters and keeps all rows as data."""
    content = (
        "| Row | A | B | C |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | 2024-01-01 | 100 | Active |\n"
        "| 2 | 2024-02-01 | 200 | Closed |\n"
        "| 3 | 2024-03-01 | 300 | Active |"
    )
    eda = run_table_eda(content)
    names = [col.name for col in eda.columns]
    assert names == ["A", "B", "C"]
    assert eda.row_count == 3
    assert eda.header_row == 0
    assert eda.columns[1].dtype == "numeric"
    assert eda.columns[1].stats["min"] == 100.0


def test_run_table_eda_with_header():
    """Normal header row is preserved as column names."""
    content = (
        "| Row | A | B |\n"
        "| --- | --- | --- |\n"
        "| 1 | Entity_ID | Amount |\n"
        "| 2 | E001 | 500 |\n"
        "| 3 | E002 | 600 |"
    )
    eda = run_table_eda(content)
    names = [col.name for col in eda.columns]
    assert names == ["Entity_ID", "Amount"]
    assert eda.row_count == 2
    assert eda.header_row == 1
    assert eda.used_range == ""
    assert eda.header_mode == "header_row"


def test_detect_header_mode_identifies_real_header_row():
    """Header detection keeps a true label row as the header."""
    rows = [
        {1: "Entity_ID", 2: "Amount"},
        {1: "E001", 2: "500"},
        {1: "E002", 2: "600"},
    ]

    assert detect_header_mode(rows, [1, 2]) == "header_row"


def test_detect_header_mode_falls_back_for_headerless_region():
    """Header detection keeps headerless first rows as data."""
    rows = [
        {1: "2024-01-01", 2: "100", 3: "Open"},
        {1: "2024-02-01", 2: "200", 3: "Closed"},
    ]

    assert detect_header_mode(rows, [1, 2, 3]) == "headerless"


def test_detect_header_mode_uses_matching_shapes_for_text_data():
    """All-text records stay headerless when shapes match later rows."""
    rows = [
        {1: "A-1", 2: "East", 3: "Open"},
        {1: "A-2", 2: "West", 3: "Closed"},
        {1: "A-3", 2: "North", 3: "Open"},
    ]

    assert detect_header_mode(rows, [1, 2, 3]) == "headerless"


def test_detect_header_mode_single_row_defaults_to_header():
    """Single-row regions default to header mode."""
    assert detect_header_mode([{1: "Date", 2: "Amount"}], [1, 2]) == (
        "header_row"
    )


def test_build_column_names_generates_stable_names_for_headerless_region():
    """Headerless regions use Excel column letters as stable names."""
    header_row, col_names, value_rows = build_column_names(
        rows=[{1: "2024-01-01", 2: "100"}],
        column_numbers=[1, 2],
        header_mode="headerless",
        row_numbers=[5],
    )

    assert header_row == 0
    assert col_names == ["A", "B"]
    assert value_rows == [{1: "2024-01-01", 2: "100"}]


def test_build_column_names_uses_first_row_for_headers():
    """Header-row regions consume the first row as column names."""
    header_row, col_names, value_rows = build_column_names(
        rows=[{1: "Date", 2: "Amount"}, {1: "2024-01-01", 2: "100"}],
        column_numbers=[1, 2],
        header_mode="header_row",
        row_numbers=[3, 4],
    )

    assert header_row == 3
    assert col_names == ["Date", "Amount"]
    assert value_rows == [{1: "2024-01-01", 2: "100"}]


def test_build_column_names_empty_rows():
    """Empty region rows fall back to column letters and no data rows."""
    header_row, col_names, value_rows = build_column_names(
        rows=[],
        column_numbers=[2, 3],
        header_mode="header_row",
        row_numbers=[7, 8],
    )

    assert header_row == 0
    assert col_names == ["B", "C"]
    assert value_rows == []


def test_build_column_names_pads_missing_header_names():
    """Sparse header rows keep positional placeholders for missing names."""
    header_row, col_names, value_rows = build_column_names(
        rows=[{1: "OnlyOneName"}, {1: "v1", 2: "v2"}],
        column_numbers=[1, 2],
        header_mode="header_row",
        row_numbers=[1, 2],
    )

    assert header_row == 1
    assert col_names == ["OnlyOneName", ""]
    assert value_rows == [{1: "v1", 2: "v2"}]


def test_run_table_eda_from_region_preserves_first_data_row_when_headerless():
    """Headerless region parsing keeps row 1 as data and uses stable names."""
    region = {
        "region_id": "region_2",
        "used_range": "A3:C5",
        "row_numbers": [3, 4, 5],
        "column_numbers": [1, 2, 3],
        "rows": [
            {
                "row_number": 3,
                "cells": [
                    {"column_number": 1, "value": "2024-01-01"},
                    {"column_number": 2, "value": "100"},
                    {"column_number": 3, "value": "Open"},
                ],
            },
            {
                "row_number": 4,
                "cells": [
                    {"column_number": 1, "value": "2024-02-01"},
                    {"column_number": 2, "value": "200"},
                    {"column_number": 3, "value": "Closed"},
                ],
            },
            {
                "row_number": 5,
                "cells": [
                    {"column_number": 1, "value": "2024-03-01"},
                    {"column_number": 2, "value": "300"},
                    {"column_number": 3, "value": "Open"},
                ],
            },
        ],
    }

    eda = run_table_eda_from_region(
        region,
        framing_context="# Sheet: Transactions",
        token_source="full sheet content",
    )

    assert eda.used_range == "A3:C5"
    assert eda.source_region_id == "region_2"
    assert eda.header_mode == "headerless"
    assert eda.header_row == 0
    assert [col.name for col in eda.columns] == ["A", "B", "C"]
    assert eda.row_count == 3
    assert "2024-01-01" in eda.sample_rows[0]


def test_run_table_eda_from_region_preserves_used_range():
    """Region parsing retains source bounds and framing context."""
    region = {
        "region_id": "region_3",
        "used_range": "B7:C9",
        "row_numbers": [7, 8, 9],
        "column_numbers": [2, 3],
        "rows": [
            {
                "row_number": 7,
                "cells": [
                    {"column_number": 2, "value": "Date"},
                    {"column_number": 3, "value": "Amount"},
                ],
            },
            {
                "row_number": 8,
                "cells": [
                    {"column_number": 2, "value": "2024-01-01"},
                    {"column_number": 3, "value": "100"},
                ],
            },
            {
                "row_number": 9,
                "cells": [
                    {"column_number": 2, "value": "2024-01-02"},
                    {"column_number": 3, "value": "200"},
                ],
            },
        ],
    }

    eda = run_table_eda_from_region(region, framing_context="summary")

    assert eda.used_range == "B7:C9"
    assert eda.header_mode == "header_row"
    assert eda.header_row == 7
    assert eda.framing_context == "summary"
    assert [col.position for col in eda.columns] == ["B", "C"]


def test_run_table_eda_from_region_disambiguates_duplicate_headers():
    """Duplicate region headers are suffixed by position."""
    region = {
        "region_id": "region_4",
        "used_range": "A1:C3",
        "row_numbers": [1, 2, 3],
        "column_numbers": [1, 2, 3],
        "rows": [
            {
                "row_number": 1,
                "cells": [
                    {"column_number": 1, "value": "Amount"},
                    {"column_number": 2, "value": "Status"},
                    {"column_number": 3, "value": "Amount"},
                ],
            },
            {
                "row_number": 2,
                "cells": [
                    {"column_number": 1, "value": "100"},
                    {"column_number": 2, "value": "Open"},
                    {"column_number": 3, "value": "200"},
                ],
            },
            {
                "row_number": 3,
                "cells": [
                    {"column_number": 1, "value": "300"},
                    {"column_number": 2, "value": "Closed"},
                    {"column_number": 3, "value": "400"},
                ],
            },
        ],
    }

    eda = run_table_eda_from_region(region)

    assert [col.name for col in eda.columns] == [
        "Amount (A)",
        "Status",
        "Amount (C)",
    ]


def test_run_table_eda_from_region_empty_region():
    """Empty serialized regions produce an empty EDA result."""
    eda = run_table_eda_from_region({"region_id": "region_5"})
    assert eda.row_count == 0
    assert eda.used_range == ""
    assert eda.header_mode == "headerless"


def test_run_table_eda_from_region_derives_columns_from_cells():
    """Column numbers can be inferred from row cell metadata."""
    region = {
        "region_id": "region_7",
        "row_numbers": [10],
        "rows": [
            "skip-me",
            {"row_number": 9, "cells": "bad-cells"},
            {
                "row_number": 10,
                "cells": [
                    {"column_number": 2, "value": "Date"},
                    {"column_number": 3, "value": "Amount"},
                    "bad-cell",
                ],
            },
            {
                "row_number": 11,
                "cells": [
                    {"column_number": 2, "value": "2024-01-01"},
                    {"column_number": 3, "value": "100"},
                    {"column_number": "bad", "value": "ignored"},
                ],
            },
            {
                "row_number": "bad",
                "cells": [{"column_number": 2, "value": "ignored"}],
            },
        ],
    }

    eda = run_table_eda_from_region(region)

    assert [col.position for col in eda.columns] == ["B", "C"]
    assert eda.header_row == 10
    assert eda.row_count == 1


def test_run_table_eda_from_region_sample_rows_truncate():
    """Region sample rows keep the first five and last three rows."""
    region_rows = [
        {
            "row_number": 1,
            "cells": [
                {"column_number": 1, "value": "Account"},
                {"column_number": 2, "value": "Amount"},
            ],
        }
    ]
    for row_number in range(2, 12):
        region_rows.append(
            {
                "row_number": row_number,
                "cells": [
                    {"column_number": 1, "value": f"ID-{row_number}"},
                    {"column_number": 2, "value": str(row_number * 10)},
                ],
            }
        )
    region = {
        "region_id": "region_8",
        "used_range": "A1:B11",
        "row_numbers": list(range(1, 12)),
        "column_numbers": [1, 2],
        "rows": region_rows,
    }

    eda = run_table_eda_from_region(region)

    assert len(eda.sample_rows) == 8
    assert "ID-2" in eda.sample_rows[0]
    assert "ID-11" in eda.sample_rows[-1]


# ── _disambiguate_names ─────────────────────────────────────


def test_disambiguate_names_no_duplicates():
    """Returns names unchanged when all unique."""
    result = _disambiguate_names(["ID", "Name", "Amount"], ["A", "B", "C"])
    assert result == ["ID", "Name", "Amount"]


def test_disambiguate_names_with_duplicates():
    """Appends position to duplicate names only."""
    result = _disambiguate_names(["ID", "Amount", "Amount"], ["A", "B", "C"])
    assert result == ["ID", "Amount (B)", "Amount (C)"]


def test_disambiguate_names_all_same():
    """All columns with same name get positions."""
    result = _disambiguate_names(["Val", "Val", "Val"], ["A", "B", "C"])
    assert result == ["Val (A)", "Val (B)", "Val (C)"]


def test_disambiguate_names_empty():
    """Empty input returns empty list."""
    assert not _disambiguate_names([], [])


def test_disambiguate_names_empty_name_duplicates():
    """Empty-string duplicates get position suffix."""
    result = _disambiguate_names(["", "", "X"], ["A", "B", "C"])
    assert result == [" (A)", " (B)", "X"]


def test_run_table_eda_duplicate_columns():
    """Duplicate column names are disambiguated with position."""
    content = (
        "| Row | A | B | C |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | Amount | Status | Amount |\n"
        "| 2 | 100 | Open | 200 |\n"
        "| 3 | 300 | Closed | 400 |"
    )
    eda = run_table_eda(content)
    names = [col.name for col in eda.columns]
    assert names == ["Amount (A)", "Status", "Amount (C)"]
    assert eda.columns[0].stats["min"] == 100.0
    assert eda.columns[2].stats["min"] == 200.0


def test_run_table_eda_row_fewer_cells_than_columns():
    """Rows with fewer cells fill missing columns with empty."""
    content = (
        "| Row | A | B | C |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | H1 | H2 | H3 |\n"
        "| 2 | v1 |"
    )
    eda = run_table_eda(content)
    assert eda.columns[1].null_count == 1
    assert eda.columns[2].null_count == 1
