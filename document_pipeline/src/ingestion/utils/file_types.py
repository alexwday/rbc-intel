"""Shared dataclasses and file utilities for the pipeline."""

import hashlib
from dataclasses import dataclass, field
from typing import Any, List

from .config import get_accepted_filetypes


@dataclass
class FileRecord:
    """A discovered file with its metadata and path components.

    Params:
        data_source: Top-level folder name (first subfolder under base)
        filter_1: Second-level subfolder or empty string
        filter_2: Third-level subfolder or empty string
        filter_3: Fourth-level subfolder or empty string
        filename: Basename of the file
        filetype: Lowercase file extension without dot
        file_size: Size in bytes
        date_last_modified: Raw mtime from os.stat
        file_hash: SHA-256 hex digest, empty until computed
        file_path: Full absolute path to the file
        supported: Whether the filetype is accepted

    Example:
        >>> r = FileRecord(
        ...     data_source="policy", filter_1="2026",
        ...     filter_2="", filter_3="", filename="doc.pdf",
        ...     filetype="pdf", file_size=1024,
        ...     date_last_modified=1700000000.0,
        ...     file_hash="", file_path="/data/policy/2026/doc.pdf",
        ... )
        >>> r.supported
        True
    """

    data_source: str
    filter_1: str
    filter_2: str
    filter_3: str
    filename: str
    filetype: str
    file_size: int
    date_last_modified: float
    file_hash: str
    file_path: str
    supported: bool = field(init=False)

    def __post_init__(self) -> None:
        """Compute supported flag from filetype."""
        self.supported = self.filetype in get_accepted_filetypes()


@dataclass
class DiscoveryDiff:
    """Result of comparing filesystem against the catalog.

    Params:
        new: Files on disk but not in the catalog
        modified: Files on disk whose size or hash changed
        deleted: Files in catalog but no longer on disk

    Example:
        >>> diff = DiscoveryDiff(new=[], modified=[], deleted=[])
        >>> len(diff.new)
        0
    """

    new: List[FileRecord]
    modified: List[FileRecord]
    deleted: List[FileRecord]


@dataclass
class PageResult:
    """Extraction result for a single page.

    Params:
        page_number: 1-indexed page number
        page_title: Title inferred from page content
        content: Extracted markdown content
        method: Extraction method used by the processor
            (e.g. "full_dpi", "split_halves",
            "xlsx_sheet_classification", or "failed")
        error: Error message if extraction failed, empty on success
        metadata: Optional processor-specific metadata
            for downstream stages

    Example:
        >>> page = PageResult(
        ...     page_number=1, page_title="Q3 Results",
        ...     content="### Text\\n...", method="full_dpi",
        ... )
        >>> page.error
        ''
    """

    page_number: int
    page_title: str
    content: str
    method: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Extraction result for an entire file.

    Params:
        file_path: Absolute path to the source file
        filetype: Lowercase extension without dot
        pages: Per-page extraction results
        total_pages: Total number of pages in the file
        pages_succeeded: Pages with usable content
        pages_failed: Pages that failed extraction

    Example:
        >>> result = ExtractionResult(
        ...     file_path="/data/doc.pdf", filetype="pdf",
        ...     pages=[], total_pages=0,
        ...     pages_succeeded=0, pages_failed=0,
        ... )
    """

    file_path: str
    filetype: str
    pages: List[PageResult]
    total_pages: int
    pages_succeeded: int
    pages_failed: int


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
class PreparedPage:
    """Content preparation output for a single page.

    Params:
        page_number: 1-indexed page number from extraction
        page_title: Page title from extraction
        content: Full prepared page content
        method: Preparation method applied
        metadata: Extraction metadata passed through for
            downstream enrichment
        original_content: Preserved raw content for dense tables
        dense_tables: Prepared dense-table payloads for all detected
            dense regions on this page
        dense_table_eda: Serialized EDA output if page was a dense table
        dense_table_description: Serialized dense table description
        description_generation_mode: How the dense table description
            was produced ("llm_one_shot", "llm_batched",
            "deterministic_fallback")

    Example:
        >>> page = PreparedPage(
        ...     page_number=1, page_title="Data Export",
        ...     content="# Sheet: Data Export", method="passthrough",
        ... )
    """

    page_number: int
    page_title: str
    content: str
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    original_content: str = ""
    dense_tables: List[Any] = field(default_factory=list)
    dense_table_eda: Any | None = None
    dense_table_description: Any | None = None
    description_generation_mode: str = ""


@dataclass
class ContentPreparationResult:
    """Content preparation result for an entire file.

    Params:
        file_path: Absolute path to the source file
        filetype: Lowercase extension without dot
        pages: Per-page preparation results
        dense_tables_spliced: Dense table regions replaced across all pages

    Example:
        >>> result = ContentPreparationResult(
        ...     file_path="/data/report.xlsx", filetype="xlsx",
        ...     pages=[], dense_tables_spliced=1,
        ... )
    """

    file_path: str
    filetype: str
    pages: List[PreparedPage]
    dense_tables_spliced: int


@dataclass
class EnrichedPage(PreparedPage):
    """Enrichment output for a single page.

    Params:
        page_number: 1-indexed page number from preparation
        page_title: Page title from extraction
        content: Prepared page content sent to enrichment
        method: Content-preparation method applied
        metadata: Extraction and preparation metadata
        original_content: Raw content preserved from Stage 3
            for dense-table pages
        dense_tables: Dense-table payloads preserved from Stage 3
        dense_table_eda: EDA output preserved from Stage 3
        dense_table_description: LLM description preserved
            from Stage 3
        description_generation_mode: How the dense table
            description was produced
        summary: Short page summary
        usage_description: Retrieval-focused usage guidance
        keywords: Descriptive search keywords
        classifications: High-level content labels
        entities: Named entities extracted from the page
        section_hierarchy: Flat heading list with levels

    Example:
        >>> page = EnrichedPage(
        ...     page_number=1,
        ...     page_title="Overview",
        ...     content="# Overview",
        ...     method="passthrough",
        ... )
    """

    summary: str = ""
    usage_description: str = ""
    keywords: List[str] = field(default_factory=list)
    classifications: List[str] = field(default_factory=list)
    entities: List[dict[str, str]] = field(default_factory=list)
    section_hierarchy: List[dict[str, Any]] = field(default_factory=list)


@dataclass
class EnrichmentResult:
    """Enrichment result for an entire file.

    Params:
        file_path: Absolute path to the source file
        filetype: Lowercase extension without dot
        pages: Per-page enrichment outputs
        pages_enriched: Number of enriched pages
        pages_failed: Number of pages that failed enrichment
        dense_tables_spliced: Dense table regions spliced
            (preserved from Stage 3)

    Example:
        >>> result = EnrichmentResult(
        ...     file_path="/data/report.pdf",
        ...     filetype="pdf",
        ...     pages=[],
        ...     pages_enriched=0,
        ...     pages_failed=0,
        ... )
    """

    file_path: str
    filetype: str
    pages: List[EnrichedPage]
    pages_enriched: int
    pages_failed: int
    dense_tables_spliced: int = 0


@dataclass
class DocumentSubsection:
    """A subsection within a primary section.

    Params:
        subsection_number: 1-indexed sequence within the parent section
        title: Display title for the subsection
        page_start: First page covered by the subsection
        page_end: Last page covered by the subsection
        page_count: Total pages in the subsection
        summary: Structured subsection summary metadata

    Example:
        >>> subsection = DocumentSubsection(
        ...     subsection_number=1,
        ...     title="Executive Summary",
        ...     page_start=1,
        ...     page_end=2,
        ...     page_count=2,
        ... )
        >>> subsection.page_count
        2
    """

    subsection_number: int
    title: str
    page_start: int
    page_end: int
    page_count: int
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSection:
    """A primary section in the document hierarchy.

    Params:
        section_number: 1-indexed sequence within the document
        title: Display title for the section
        page_start: First page covered by the section
        page_end: Last page covered by the section
        page_count: Total pages in the section
        summary: Structured section summary metadata
        subsections: Nested subsection ranges

    Example:
        >>> section = DocumentSection(
        ...     section_number=1,
        ...     title="Overview",
        ...     page_start=1,
        ...     page_end=3,
        ...     page_count=3,
        ... )
        >>> section.title
        'Overview'
    """

    section_number: int
    title: str
    page_start: int
    page_end: int
    page_count: int
    summary: dict[str, Any] = field(default_factory=dict)
    subsections: List[DocumentSubsection] = field(default_factory=list)


@dataclass
class DocumentChunk:
    """One chunk of finalized content with hierarchy context.

    Params:
        chunk_number: 0-indexed chunk position within the document
        page_number: Source page number for this chunk
        content: Chunk text content
        primary_section_number: Parent primary section number
        primary_section_name: Parent primary section title
        subsection_number: Parent subsection number or 0
        subsection_name: Parent subsection title or section title
        hierarchy_path: Breadcrumb string for retrieval context
        primary_section_page_count: Pages in the parent section
        subsection_page_count: Pages in the parent subsection
        embedding_prefix: Prefix prepended before embedding
        embedding: Vector embedding for the chunk
        is_dense_table_description: Whether the chunk is a dense-table handoff
        dense_table_routing: Deterministic routing payload for dense tables
        metadata: Source page metadata preserved from earlier stages

    Example:
        >>> chunk = DocumentChunk(
        ...     chunk_number=0,
        ...     page_number=1,
        ...     content="Overview content",
        ...     primary_section_number=1,
        ...     primary_section_name="Overview",
        ...     subsection_number=0,
        ...     subsection_name="Overview",
        ...     hierarchy_path="Overview",
        ...     primary_section_page_count=1,
        ...     subsection_page_count=1,
        ... )
        >>> chunk.page_number
        1
    """

    chunk_number: int
    page_number: int
    content: str
    primary_section_number: int
    primary_section_name: str
    subsection_number: int
    subsection_name: str
    hierarchy_path: str
    primary_section_page_count: int
    subsection_page_count: int
    embedding_prefix: str = ""
    embedding: List[float] = field(default_factory=list)
    is_dense_table_description: bool = False
    dense_table_routing: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalizedDocumentBase:
    """Base finalized-document fields shared across all outputs."""

    file_path: str
    filetype: str
    file_name: str
    document_summary: str
    document_description: str
    document_usage: str
    document_metadata: dict[str, Any] = field(default_factory=dict)
    structure_type: str = "semantic"
    structure_confidence: str = "low"
    degradation_signals: List[str] = field(default_factory=list)
    summary_embedding: List[float] = field(default_factory=list)


@dataclass
class FinalizedDocument(FinalizedDocumentBase):
    """Complete finalized document ready for downstream storage.

    Params:
        file_path: Absolute path to the source file
        filetype: Lowercase extension without dot
        file_name: Basename of the source file
        document_metadata: Structured metadata extracted from the file
        document_summary: Multi-sentence document summary
        document_description: Short catalog description
        document_usage: Retrieval-oriented usage guidance
        structure_type: High-level document organization label
        structure_confidence: Confidence label for structure_type
        degradation_signals: Quality signals recorded during finalization
        summary_embedding: Embedding for the document summary
        sections: Primary document sections with subsections
        chunks: One finalized chunk per page
        dense_tables: Dense-table payloads preserved for future storage
        sheet_summaries: XLSX sheet-level summaries for discovery
        keyword_embeddings: Per-keyword contextualized embeddings
        extracted_metrics: XLSX metric payloads with embeddings
        sheet_context_chains: XLSX sheet dependency chains
        extraction_metadata: Earlier-stage metadata preserved verbatim

    Example:
        >>> finalized = FinalizedDocument(
        ...     file_path="/data/report.pdf",
        ...     filetype="pdf",
        ...     file_name="report.pdf",
        ...     document_summary="Summary",
        ...     document_description="Description",
        ...     document_usage="Usage",
        ... )
        >>> finalized.chunk_count
        0
    """

    sections: List[DocumentSection] = field(default_factory=list)
    chunks: List[DocumentChunk] = field(default_factory=list)
    dense_tables: List[dict[str, Any]] = field(default_factory=list)
    sheet_summaries: List[dict[str, Any]] = field(default_factory=list)
    keyword_embeddings: List[dict[str, Any]] = field(default_factory=list)
    extracted_metrics: List[dict[str, Any]] = field(default_factory=list)
    sheet_context_chains: List[dict[str, Any]] = field(default_factory=list)
    extraction_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        """Return the number of source pages represented in chunks."""
        return len(self.chunks)

    @property
    def primary_section_count(self) -> int:
        """Return the number of primary sections."""
        return len(self.sections)

    @property
    def subsection_count(self) -> int:
        """Return the number of subsections across all sections."""
        return sum(len(section.subsections) for section in self.sections)

    @property
    def chunk_count(self) -> int:
        """Return the total number of chunks."""
        return len(self.chunks)

    @property
    def dense_table_count(self) -> int:
        """Return the number of preserved dense-table payloads."""
        return len(self.dense_tables)


def compute_file_hash(path: str) -> str:
    """Compute SHA-256 hex digest of a file using 8KB chunks.

    Params:
        path: Absolute path to the file

    Returns:
        str — hex digest of the file contents

    Example:
        >>> compute_file_hash("/tmp/test.txt")
        'e3b0c44298fc1c149afb...'
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()
