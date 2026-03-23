"""Focused tests for single-row and empty region header detection."""

from ingestion.utils.table_eda import (
    detect_header_mode,
    run_table_eda_from_region,
)


def test_detect_header_mode_single_data_row_can_be_headerless():
    """Single data-like rows stay available for region-based parsing."""
    assert (
        detect_header_mode(
            [{1: "2024-01-01", 2: "100"}],
            [1, 2],
        )
        == "headerless"
    )


def test_detect_header_mode_empty_rows_default_to_header():
    """Empty region inputs keep the conservative header default."""
    assert detect_header_mode([], [1, 2]) == "header_row"


def test_run_table_eda_from_region_single_data_row_preserves_values():
    """Single-row regions keep obvious data rows instead of dropping them."""
    region = {
        "region_id": "region_6",
        "used_range": "A3:B3",
        "row_numbers": [3],
        "column_numbers": [1, 2],
        "rows": [
            {
                "row_number": 3,
                "cells": [
                    {"column_number": 1, "value": "2024-01-01"},
                    {"column_number": 2, "value": "100"},
                ],
            }
        ],
    }

    eda = run_table_eda_from_region(region)

    assert eda.header_mode == "headerless"
    assert eda.header_row == 0
    assert eda.row_count == 1
    assert [col.name for col in eda.columns] == ["A", "B"]
    assert eda.sample_rows == ["| 3 | 2024-01-01 | 100 |"]
