"""Dataclasses for the content preparation stage."""

from dataclasses import dataclass, field
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
class ContentChunk:
    """One chunk of prepared page content.

    Params:
        chunk_index: 0-based position within the page
        content: Markdown content of this chunk
        token_count: Token count of this chunk
        is_dense_table_description: Whether this is a generated
            dense table description (vs original content)
        routing_metadata: Deterministic retrieval metadata for
            second-stage routing when the chunk represents a
            dense-table handoff

    Example:
        >>> chunk = ContentChunk(
        ...     chunk_index=0, content="# Revenue...",
        ...     token_count=500,
        ... )
    """

    chunk_index: int
    content: str
    token_count: int
    is_dense_table_description: bool = False
    routing_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedDenseTable:
    """Prepared dense-table payload for one sheet region.

    Params:
        region_id: Source worksheet region ID when available
        used_range: Excel range backing this dense table
        routing_metadata: Deterministic metadata for second-stage retrieval
        chunks: Dense-table replacement chunks emitted for similarity search
        replacement_content: Full replacement markdown prior to chunking
        dense_table_eda: TableEDA built for this region
        dense_table_description: LLM or deterministic region description
        description_generation_mode: How the description was produced

    Example:
        >>> dense = PreparedDenseTable(
        ...     region_id="region_2", used_range="A1:B10",
        ...     routing_metadata={}, chunks=[],
        ...     replacement_content="# Dense Table",
        ... )
    """

    region_id: str
    used_range: str
    routing_metadata: dict[str, Any]
    chunks: List[ContentChunk]
    replacement_content: str
    dense_table_eda: "TableEDA | None" = None
    dense_table_description: "DenseTableDescription | None" = None
    description_generation_mode: str = ""


@dataclass
class PreparedPage:
    """Content preparation output for a single page.

    Params:
        page_number: 1-indexed page number from extraction
        page_title: Page title from extraction
        chunks: Content split into embeddable chunks
        method: Preparation method applied
        original_content: Preserved raw content for dense tables
        dense_tables: Prepared dense-table payloads for all detected
            dense regions on this page
        dense_table_eda: EDA output if page was a dense table
        dense_table_description: LLM description if dense table
        description_generation_mode: How the dense table description
            was produced ("llm_one_shot", "llm_batched",
            "deterministic_fallback")

    Example:
        >>> page = PreparedPage(
        ...     page_number=1, page_title="Data Export",
        ...     chunks=[], method="passthrough",
        ... )
    """

    page_number: int
    page_title: str
    chunks: List[ContentChunk]
    method: str
    original_content: str = ""
    dense_tables: List["PreparedDenseTable"] = field(default_factory=list)
    dense_table_eda: "TableEDA | None" = None
    dense_table_description: "DenseTableDescription | None" = None
    description_generation_mode: str = ""


@dataclass
class ContentPreparationResult:
    """Content preparation result for an entire file.

    Params:
        file_path: Absolute path to the source file
        filetype: Lowercase extension without dot
        pages: Per-page preparation results
        dense_tables_processed: Dense table regions processed across all pages
        pages_chunked: Pages requiring LLM chunking

    Example:
        >>> result = ContentPreparationResult(
        ...     file_path="/data/report.xlsx", filetype="xlsx",
        ...     pages=[], dense_tables_processed=1,
        ...     pages_chunked=0,
        ... )
    """

    file_path: str
    filetype: str
    pages: List[PreparedPage]
    dense_tables_processed: int
    pages_chunked: int
