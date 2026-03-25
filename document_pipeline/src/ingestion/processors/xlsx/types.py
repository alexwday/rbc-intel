"""XLSX-specific dataclasses for table analysis and dense-table handling."""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class ColumnProfile:
    """Per-column EDA output from table analysis.

    Params:
        name: Column header text
        position: Column letter (e.g. "A", "B")
        dtype: Detected type (numeric, text, date, boolean, mixed)
        stats: Type-specific statistics
        sample_values: Up to 5 representative distinct values
        non_null_count: Number of non-empty cells
        null_count: Number of empty cells
        unique_count: Number of distinct values

    Example:
        >>> col = ColumnProfile(
        ...     name="Revenue", position="B", dtype="numeric",
        ...     stats={"min": 100, "max": 5000, "mean": 2500.0},
        ...     sample_values=["100", "2500", "5000"],
        ...     non_null_count=50, null_count=2, unique_count=45,
        ... )
    """

    name: str
    position: str
    dtype: str
    stats: dict[str, Any]
    sample_values: List[str]
    non_null_count: int
    null_count: int
    unique_count: int


@dataclass
class TableEDA:
    """Aggregate EDA output for a dense table.

    Params:
        row_count: Total data rows (excluding header)
        columns: Per-column profiles
        header_row: Excel row number containing column headers
        framing_context: Non-table content (metadata, visuals)
        sample_rows: Representative rows (first 5 + last 3)
        token_count: Token count of original content
        used_range: Exact worksheet range for the parsed table
        header_mode: "header_row" or "headerless"
        source_region_id: Worksheet region ID when sourced from layout metadata

    Example:
        >>> eda = TableEDA(
        ...     row_count=100, columns=[], header_row=1,
        ...     framing_context="# Sheet: Data Export",
        ...     sample_rows=[], token_count=5000,
        ... )
    """

    row_count: int
    columns: List[ColumnProfile]
    header_row: int
    framing_context: str
    sample_rows: List[str]
    token_count: int
    used_range: str = ""
    header_mode: str = "header_row"
    source_region_id: str = ""


@dataclass
class DenseTableDescription:
    """LLM-generated description of a dense table.

    Params:
        description: 2-4 sentence summary of the dataset
        column_descriptions: Per-column position/name/description
        filter_columns: Column positions for filtering
        identifier_columns: Column positions for record identifiers
        measure_columns: Column positions for aggregation
        text_content_columns: Column positions for descriptive text
        sample_queries: Natural language questions the data answers

    Example:
        >>> desc = DenseTableDescription(
        ...     description="Monthly transaction log...",
        ...     column_descriptions=[
        ...         {
        ...             "position": "A",
        ...             "name": "Date",
        ...             "description": "Txn date",
        ...         }
        ...     ],
        ...     filter_columns=["B", "C"],
        ...     identifier_columns=["A"],
        ...     measure_columns=["D"],
        ...     text_content_columns=[],
        ...     sample_queries=["What are total deposits?"],
        ... )
    """

    description: str
    column_descriptions: List[dict[str, str]]
    filter_columns: List[str]
    identifier_columns: List[str]
    measure_columns: List[str]
    text_content_columns: List[str]
    sample_queries: List[str]


@dataclass
class PreparedDenseTable:
    """Prepared dense-table payload for one sheet region.

    Params:
        region_id: Source worksheet region ID when available
        used_range: Excel range backing this dense table
        routing_metadata: Deterministic metadata for second-stage retrieval
        raw_content: Structured raw rows replaced on the sheet
        replacement_content: Full replacement markdown inserted on the sheet
        dense_table_eda: TableEDA built for this region
        dense_table_description: LLM or deterministic region description
        description_generation_mode: How the description was produced

    Example:
        >>> dense = PreparedDenseTable(
        ...     region_id="region_2", used_range="A1:B10",
        ...     routing_metadata={},
        ...     raw_content=[{"row_number": 3, "cells": []}],
        ...     replacement_content="# Dense Table",
        ... )
    """

    region_id: str
    used_range: str
    routing_metadata: dict[str, Any]
    raw_content: List[dict[str, Any]]
    replacement_content: str
    dense_table_eda: TableEDA | None = None
    dense_table_description: DenseTableDescription | None = None
    description_generation_mode: str = ""
