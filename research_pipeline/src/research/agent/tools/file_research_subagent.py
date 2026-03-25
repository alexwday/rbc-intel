"""
File Research Subagent for Universal Cascading Retrieval Architecture.

Stage 2 of the cascading retrieval process:
1. Receive document IDs from Stage 1 (metadata subagent)
2. Fetch chunks using similarity search (query_embedding required)
3. Process documents in parallel - each document synthesized by LLM
4. Return page-based research findings with citations

Chunk Retrieval:
Hierarchical retrieval prioritizes document structure. For small documents,
all chunks are returned in page order. For larger files, similarity search
selects seed chunks which are then expanded by primary section, subsection,
or neighbor ranges for coherent context.

This subagent is UNIVERSAL - works for ALL databases (internal and external)
by querying the unified document tables.

Functions:
    fetch_chunks_with_hierarchical_expansion: Structured retrieval with expansion
    load_file_research_prompt_config: Load prompt configuration from PostgreSQL
    format_document_chunks_for_llm: Format document chunks for LLM prompt
    synthesize_document_research: LLM synthesis for a single document
    execute_file_research_sync: Main entry point for Stage 2 file research

Classes:
    ChunkData: TypedDict for chunk data from database
    DocumentChunks: TypedDict for document with chunks
    PageResearch: TypedDict for page-level research findings
    DocumentResearch: TypedDict for document research output
    FileResearchResult: TypedDict for full subagent result
    FileResearchError: Exception for file research errors
"""

import json
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict, Literal, NotRequired

from sqlalchemy import text

from ...utils.config import config
from ...utils.prompt_loader import get_prompt
from ...connections.postgres import get_database_session, get_database_schema
from ...connections.llm import execute_llm_call
from .database_metadata import DatabaseMetadataCache
from .dense_table_research import research_dense_table
from .research_types import Finding, FindingsList

logger = logging.getLogger(__name__)


class FileResearchError(Exception):
    """Exception raised for file research errors."""


MODEL_CAPABILITY = "small"
MODEL_MAX_TOKENS = 16384
MODEL_TEMPERATURE = 0.2

SynthesisContext = Dict[str, Any]
ResearchContext = Dict[str, Any]


class ChunkData(TypedDict):
    """Chunk data from document_chunks."""

    chunk_id: str
    chunk_number: int
    content: str
    primary_section_name: Optional[str]
    subsection_name: Optional[str]
    hierarchy_path: Optional[str]
    page_number: Optional[int]
    page_reference: Optional[str]
    primary_section_number: NotRequired[Optional[int]]
    subsection_number: NotRequired[Optional[int]]
    primary_section_page_count: NotRequired[Optional[int]]
    subsection_page_count: NotRequired[Optional[int]]
    is_dense_table_description: NotRequired[bool]
    dense_table_routing_json: NotRequired[Optional[str]]
    embedding_prefix: NotRequired[Optional[str]]


ExpansionLevel = Literal[
    "document_full",
    "primary_section",
    "subsection",
    "neighbor",
]


class ChunkGroup(TypedDict):
    """Grouped chunks with expansion metadata for formatting."""

    primary_section_number: Optional[int]
    primary_section_name: Optional[str]
    hierarchy_path: Optional[str]
    subsection_number: Optional[int]
    subsection_name: Optional[str]
    expansion_level: ExpansionLevel
    chunks: List[ChunkData]


class ExpansionMetrics(TypedDict):
    """Metrics tracking hierarchical expansion effectiveness."""

    # Document-level info
    document_page_count: Optional[int]
    retrieval_method: Literal["full_context", "similarity_expansion"]

    # Seed chunk stats (from similarity search)
    seed_chunks_count: int
    seed_primary_sections_count: int  # How many different primary sections in seed

    # Expansion stats
    groups_total: int
    groups_document_full: int
    groups_primary_section: int
    groups_subsection: int
    groups_neighbor: int

    # Gap filling stats
    gap_ranges_filled: int  # Number of gaps that were filled
    gap_pages_filled: int  # Total pages added via gap filling

    # Final context stats
    final_chunks_count: int
    final_pages_count: int
    final_primary_sections_count: int

    # Expansion ratio (final chunks / seed chunks)
    expansion_ratio: float


class DocumentChunks(TypedDict):
    """Document with all its chunks for research."""

    document_id: str
    document_name: str
    file_name: Optional[str]
    chunks: List[ChunkData]
    chunk_groups: NotRequired[List[ChunkGroup]]
    expansion_metrics: NotRequired[ExpansionMetrics]


class PageResearch(TypedDict):
    """Research finding for a specific page (mapped from extracted facts)."""

    page_number: int
    research_content: str
    file_link: str
    file_name: str


class DocumentResearch(TypedDict):
    """Research output for a single document."""

    document_name: str
    file_link: str
    status_summary: str
    page_research: List[PageResearch]


class FileResearchResult(TypedDict):
    """Result structure from file research subagent."""

    findings: FindingsList  # Unified finding format for summarizer
    status_summary: str
    data_source: str


def _map_chunk_row(row: Any) -> ChunkData:
    """Map a database row to ChunkData."""

    return {
        "chunk_id": str(row["chunk_id"]),
        "chunk_number": row["chunk_number"],
        "content": row["content"],
        "primary_section_name": row.get("primary_section_name"),
        "subsection_name": row.get("subsection_name"),
        "hierarchy_path": row.get("hierarchy_path"),
        "page_number": row.get("page_number"),
        "page_reference": row.get("hierarchy_path"),
        "primary_section_number": row.get("primary_section_number"),
        "subsection_number": row.get("subsection_number"),
        "primary_section_page_count": row.get("primary_section_page_count"),
        "subsection_page_count": row.get("subsection_page_count"),
        "is_dense_table_description": row.get("is_dense_table_description", False),
        "dense_table_routing_json": row.get("dense_table_routing_json"),
        "embedding_prefix": row.get("embedding_prefix"),
    }


def _sort_chunks(chunks: List[ChunkData]) -> List[ChunkData]:
    """Sort chunks by page_number then chunk_number."""

    return sorted(
        chunks,
        key=lambda r: ((r.get("page_number") or 0), r.get("chunk_number", 0)),
    )


def _primary_section_sort_key(section_number: Optional[int]) -> Tuple[int, int]:
    """Sort key that places numbered sections before unknown ones."""

    if section_number is None:
        return (1, 0)
    return (0, section_number)


def _chunk_group_sort_key(group: ChunkGroup) -> Tuple[int, int, int, str]:
    """Sort chunk groups by primary and subsection ordering."""

    primary_sort = _primary_section_sort_key(group.get("primary_section_number"))
    subsection_number = group.get("subsection_number")
    subsection_sort = subsection_number if subsection_number is not None else -1
    return (
        primary_sort[0],
        primary_sort[1],
        subsection_sort,
        group.get("expansion_level", "neighbor"),
    )


def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping numeric ranges."""

    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged: List[Tuple[int, int]] = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def _dedup_chunks(chunks: List[ChunkData]) -> List[ChunkData]:
    """Remove duplicate chunks while preserving sort order."""

    seen_ids: Set[str] = set()
    deduped: List[ChunkData] = []

    for chunk in _sort_chunks(chunks):
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            deduped.append(chunk)

    return deduped


def _fetch_document_metadata(
    session: Any, doc_id: str, data_source: str
) -> Dict[str, Any]:
    """Fetch metadata for a single document."""

    schema = get_database_schema()

    doc_result = session.execute(
        text(
            f"""
            SELECT document_id, title, file_name, page_count
            FROM {schema}.documents
            WHERE document_id = :doc_id AND data_source = :data_source
        """
        ),
        {"doc_id": doc_id, "data_source": data_source},
    )
    doc_row = doc_result.mappings().first()

    if not doc_row:
        raise FileResearchError(f"Document {doc_id} not found in {data_source}")

    return {
        "document_id": str(doc_row["document_id"]),
        "title": doc_row["title"],
        "file_name": doc_row.get("file_name"),
        "page_count": doc_row.get("page_count"),
    }


def _fetch_all_chunks_for_document(
    session: Any, doc_id: str
) -> List[ChunkData]:
    """Fetch every chunk for a document ordered by page and chunk number."""

    schema = get_database_schema()

    chunk_result = session.execute(
        text(
            f"""
            SELECT
                chunk_id,
                chunk_number,
                content,
                primary_section_name,
                subsection_name,
                hierarchy_path,
                page_number,
                primary_section_number,
                subsection_number,
                primary_section_page_count,
                subsection_page_count,
                is_dense_table_description,
                dense_table_routing_json,
                embedding_prefix
            FROM {schema}.document_chunks
            WHERE document_id = :doc_id
            ORDER BY page_number, chunk_number
        """
        ),
        {"doc_id": doc_id},
    )

    chunk_rows = chunk_result.mappings().all()
    return _sort_chunks([_map_chunk_row(row) for row in chunk_rows])


def _fetch_chunks_by_page_range(
    session: Any,
    doc_id: str,
    start_page: int,
    end_page: int,
) -> List[ChunkData]:
    """Fetch chunks within a page range (inclusive).

    Args:
        session: Database session.
        doc_id: Document UUID.
        start_page: First page number (inclusive).
        end_page: Last page number (inclusive).

    Returns:
        List[ChunkData]: Chunks from the specified page range.
    """
    schema = get_database_schema()

    chunk_result = session.execute(
        text(
            f"""
            SELECT
                chunk_id,
                chunk_number,
                content,
                primary_section_name,
                subsection_name,
                hierarchy_path,
                page_number,
                primary_section_number,
                subsection_number,
                primary_section_page_count,
                subsection_page_count,
                is_dense_table_description,
                dense_table_routing_json,
                embedding_prefix
            FROM {schema}.document_chunks
            WHERE document_id = :doc_id
              AND page_number >= :start_page
              AND page_number <= :end_page
            ORDER BY page_number, chunk_number
        """
        ),
        {
            "doc_id": doc_id,
            "start_page": start_page,
            "end_page": end_page,
        },
    )

    chunk_rows = chunk_result.mappings().all()
    return _sort_chunks([_map_chunk_row(row) for row in chunk_rows])


def _fetch_similarity_chunks_for_document(
    session: Any,
    doc_id: str,
    embedding_str: str,
    limit: int,
) -> List[ChunkData]:
    """Fetch top chunks for a document using similarity search."""

    schema = get_database_schema()

    chunk_result = session.execute(
        text(
            f"""
            SELECT
                chunk_id,
                chunk_number,
                content,
                primary_section_name,
                subsection_name,
                hierarchy_path,
                page_number,
                primary_section_number,
                subsection_number,
                primary_section_page_count,
                subsection_page_count,
                is_dense_table_description,
                dense_table_routing_json,
                embedding_prefix
            FROM {schema}.document_chunks
            WHERE document_id = :doc_id
              AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """
        ),
        {
            "doc_id": doc_id,
            "embedding": embedding_str,
            "limit": limit,
        },
    )

    chunk_rows = chunk_result.mappings().all()
    return _sort_chunks([_map_chunk_row(row) for row in chunk_rows])


def _build_primary_section_clause(
    primary_section_number: Optional[int],
) -> Tuple[str, Dict[str, Any]]:
    """Build SQL clause and params for primary section filtering."""

    if primary_section_number is None:
        return "primary_section_number IS NULL", {}
    return "primary_section_number = :primary_section_number", {
        "primary_section_number": primary_section_number
    }


def _build_subsection_clause(
    subsection_number: Optional[int],
) -> Tuple[str, Dict[str, Any]]:
    """Build SQL clause and params for subsection filtering."""

    if subsection_number is None:
        return "subsection_number IS NULL", {}
    return "subsection_number = :subsection_number", {
        "subsection_number": subsection_number
    }


def _fetch_primary_section_chunks(
    session: Any,
    document_id: str,
    primary_section_number: Optional[int],
) -> List[ChunkData]:
    """Fetch all chunks for a primary section."""

    schema = get_database_schema()

    primary_clause, primary_params = _build_primary_section_clause(
        primary_section_number
    )

    query = f"""
        SELECT
            chunk_id,
            chunk_number,
            content,
            primary_section_name,
            subsection_name,
            hierarchy_path,
            page_number,
            primary_section_number,
            subsection_number,
            primary_section_page_count,
            subsection_page_count,
            is_dense_table_description,
            dense_table_routing_json,
            embedding_prefix
        FROM {schema}.document_chunks
        WHERE document_id = :doc_id
          AND {primary_clause}
        ORDER BY page_number, chunk_number
    """

    params: Dict[str, Any] = {"doc_id": document_id}
    params.update(primary_params)

    chunk_rows = session.execute(text(query), params).mappings().all()
    return _sort_chunks([_map_chunk_row(row) for row in chunk_rows])


def _fetch_subsection_chunks(
    session: Any,
    document_id: str,
    primary_section_number: Optional[int],
    subsection_number: Optional[int],
) -> List[ChunkData]:
    """Fetch all chunks for a subsection within a primary section."""

    schema = get_database_schema()

    primary_clause, primary_params = _build_primary_section_clause(
        primary_section_number
    )
    subsection_clause, subsection_params = _build_subsection_clause(subsection_number)

    query = f"""
        SELECT
            chunk_id,
            chunk_number,
            content,
            primary_section_name,
            subsection_name,
            hierarchy_path,
            page_number,
            primary_section_number,
            subsection_number,
            primary_section_page_count,
            subsection_page_count,
            is_dense_table_description,
            dense_table_routing_json,
            embedding_prefix
        FROM {schema}.document_chunks
        WHERE document_id = :doc_id
          AND {primary_clause}
          AND {subsection_clause}
        ORDER BY page_number, chunk_number
    """

    params: Dict[str, Any] = {"doc_id": document_id}
    params.update(primary_params)
    params.update(subsection_params)

    chunk_rows = session.execute(text(query), params).mappings().all()
    return _sort_chunks([_map_chunk_row(row) for row in chunk_rows])


def _fetch_neighbor_chunks(
    session: Any,
    document_id: str,
    primary_section_number: Optional[int],
    start_chunk: int,
    end_chunk: int,
) -> List[ChunkData]:
    """Fetch neighbor chunks around a chunk range within a primary section."""

    schema = get_database_schema()

    primary_clause, primary_params = _build_primary_section_clause(
        primary_section_number
    )

    query = f"""
        SELECT
            chunk_id,
            chunk_number,
            content,
            primary_section_name,
            subsection_name,
            hierarchy_path,
            page_number,
            primary_section_number,
            subsection_number,
            primary_section_page_count,
            subsection_page_count,
            is_dense_table_description,
            dense_table_routing_json,
            embedding_prefix
        FROM {schema}.document_chunks
        WHERE document_id = :doc_id
          AND {primary_clause}
          AND chunk_number BETWEEN :start_chunk AND :end_chunk
        ORDER BY page_number, chunk_number
    """

    params: Dict[str, Any] = {
        "doc_id": document_id,
        "start_chunk": start_chunk,
        "end_chunk": end_chunk,
    }
    params.update(primary_params)

    chunk_rows = session.execute(text(query), params).mappings().all()
    return _sort_chunks([_map_chunk_row(row) for row in chunk_rows])


def _group_full_document_chunks(chunks: List[ChunkData]) -> List[ChunkGroup]:
    """Group chunks for full-document retrieval with subsection hierarchy."""

    primary_sections: Dict[Optional[int], Dict[str, Any]] = {}

    for chunk in _sort_chunks(chunks):
        primary_number = chunk.get("primary_section_number")
        primary_entry = primary_sections.setdefault(
            primary_number,
            {
                "primary_section_name": chunk.get("primary_section_name"),
                "hierarchy_path": chunk.get("hierarchy_path"),
                "primary_chunks": [],
                "subsections": {},
            },
        )

        if not primary_entry["primary_section_name"] and chunk.get(
            "primary_section_name"
        ):
            primary_entry["primary_section_name"] = chunk.get("primary_section_name")
        if not primary_entry["hierarchy_path"] and chunk.get("hierarchy_path"):
            primary_entry["hierarchy_path"] = chunk.get("hierarchy_path")

        subsection_number = chunk.get("subsection_number")
        if subsection_number is None:
            primary_entry["primary_chunks"].append(chunk)
            continue

        subsections: Dict[Optional[int], Dict[str, Any]] = primary_entry["subsections"]
        subsection_entry = subsections.setdefault(
            subsection_number,
            {
                "subsection_name": chunk.get("subsection_name"),
                "hierarchy_path": chunk.get("hierarchy_path"),
                "chunks": [],
            },
        )
        if not subsection_entry["subsection_name"] and chunk.get("subsection_name"):
            subsection_entry["subsection_name"] = chunk.get("subsection_name")
        if not subsection_entry["hierarchy_path"] and chunk.get("hierarchy_path"):
            subsection_entry["hierarchy_path"] = chunk.get("hierarchy_path")
        subsection_entry["chunks"].append(chunk)

    chunk_groups: List[ChunkGroup] = []

    for primary_number in sorted(primary_sections.keys(), key=_primary_section_sort_key):
        primary_entry = primary_sections[primary_number]

        if primary_entry["primary_chunks"]:
            chunk_groups.append(
                {
                    "primary_section_number": primary_number,
                    "primary_section_name": primary_entry["primary_section_name"],
                    "hierarchy_path": primary_entry["hierarchy_path"],
                    "subsection_number": None,
                    "subsection_name": None,
                    "expansion_level": "document_full",
                    "chunks": _dedup_chunks(primary_entry["primary_chunks"]),
                }
            )

        subsections: Dict[Optional[int], Dict[str, Any]] = primary_entry["subsections"]
        for subsection_number in sorted(
            subsections.keys(),
            key=lambda s: (1, 0) if s is None else (0, s),
        ):
            subsection_entry = subsections[subsection_number]
            chunk_groups.append(
                {
                    "primary_section_number": primary_number,
                    "primary_section_name": primary_entry["primary_section_name"],
                    "hierarchy_path": subsection_entry.get("hierarchy_path"),
                    "subsection_number": subsection_number,
                    "subsection_name": subsection_entry.get("subsection_name"),
                    "expansion_level": "subsection",
                    "chunks": _dedup_chunks(subsection_entry.get("chunks", [])),
                }
            )

    return sorted(chunk_groups, key=_chunk_group_sort_key)


def _first_non_empty(chunks: List[ChunkData], key: str) -> Optional[str]:
    """Return the first non-empty value for a key from chunk list."""

    for chunk in chunks:
        value = chunk.get(key)
        if value:
            return value
    return None


def _expand_chunks_for_document(
    session: Any,
    document_id: str,
    seed_chunks: List[ChunkData],
    research_config: Dict[str, Any],
) -> List[ChunkGroup]:
    """Apply hierarchical expansion to retrieved chunks."""

    max_primary_pages = research_config["max_primary_section_page_count"]
    max_subsection_pages = research_config["max_subsection_page_count"]
    max_neighbour_chunks = research_config["max_neighbour_chunks"]

    chunk_groups: List[ChunkGroup] = []
    neighbor_candidates: Dict[Optional[int], List[ChunkData]] = {}
    primary_sections: Dict[Optional[int], List[ChunkData]] = {}

    for chunk in seed_chunks:
        primary_sections.setdefault(chunk.get("primary_section_number"), []).append(
            chunk
        )

    for primary_number, primary_chunks in primary_sections.items():
        primary_page_count = primary_chunks[0].get("primary_section_page_count")

        if primary_page_count is not None and primary_page_count <= max_primary_pages:
            section_chunks = _fetch_primary_section_chunks(
                session, document_id, primary_number
            )
            if not section_chunks:
                logger.warning(
                    "Primary section expansion returned no chunks for document %s, "
                    "section_number=%s; falling back to seed chunks",
                    document_id,
                    primary_number,
                )
                section_chunks = primary_chunks

            chunk_groups.append(
                {
                    "primary_section_number": primary_number,
                    "primary_section_name": _first_non_empty(
                        section_chunks or primary_chunks, "primary_section_name"
                    ),
                    "hierarchy_path": _first_non_empty(
                        section_chunks or primary_chunks, "hierarchy_path"
                    ),
                    "subsection_number": None,
                    "subsection_name": None,
                    "expansion_level": "primary_section",
                    "chunks": _dedup_chunks(section_chunks),
                }
            )
            continue

        subsection_groups: Dict[Optional[int], List[ChunkData]] = {}
        for chunk in primary_chunks:
            subsection_groups.setdefault(chunk.get("subsection_number"), []).append(
                chunk
            )

        for subsection_number, subsection_chunks in subsection_groups.items():
            subsection_page_count = subsection_chunks[0].get("subsection_page_count")

            if (
                subsection_page_count is not None
                and subsection_page_count <= max_subsection_pages
            ):
                expanded_subsection = _fetch_subsection_chunks(
                    session,
                    document_id,
                    primary_number,
                    subsection_number,
                )
                if not expanded_subsection:
                    logger.warning(
                        "Subsection expansion returned no chunks for document %s, "
                        "section_number=%s, subsection_number=%s; "
                        "falling back to seed chunks",
                        document_id,
                        primary_number,
                        subsection_number,
                    )
                    expanded_subsection = subsection_chunks

                chunk_groups.append(
                    {
                        "primary_section_number": primary_number,
                        "primary_section_name": _first_non_empty(
                            expanded_subsection or subsection_chunks,
                            "primary_section_name",
                        ),
                        "hierarchy_path": _first_non_empty(
                            expanded_subsection or subsection_chunks, "hierarchy_path"
                        ),
                        "subsection_number": subsection_number,
                        "subsection_name": _first_non_empty(
                            expanded_subsection or subsection_chunks, "subsection_name"
                        ),
                        "expansion_level": "subsection",
                        "chunks": _dedup_chunks(expanded_subsection),
                    }
                )
            else:
                neighbor_candidates.setdefault(primary_number, []).extend(
                    subsection_chunks
                )

    for primary_number, candidate_chunks in neighbor_candidates.items():
        if not candidate_chunks:
            continue

        ranges: List[Tuple[int, int]] = []
        for chunk in candidate_chunks:
            chunk_num = chunk.get("chunk_number")
            if chunk_num is None:
                continue
            start_chunk = max(chunk_num - max_neighbour_chunks, 1)
            end_chunk = chunk_num + max_neighbour_chunks
            ranges.append((start_chunk, end_chunk))

        neighbor_chunks: List[ChunkData] = []
        for start_chunk, end_chunk in _merge_ranges(ranges):
            neighbor_chunks.extend(
                _fetch_neighbor_chunks(
                    session,
                    document_id,
                    primary_number,
                    start_chunk,
                    end_chunk,
                )
            )

        if not neighbor_chunks:
            logger.warning(
                "Neighbor expansion returned no chunks for document %s, "
                "section_number=%s; falling back to candidate chunks",
                document_id,
                primary_number,
            )
            neighbor_chunks = candidate_chunks

        chunk_groups.append(
            {
                "primary_section_number": primary_number,
                "primary_section_name": _first_non_empty(
                    neighbor_chunks or candidate_chunks, "primary_section_name"
                ),
                "hierarchy_path": _first_non_empty(
                    neighbor_chunks or candidate_chunks, "hierarchy_path"
                ),
                "subsection_number": None,
                "subsection_name": None,
                "expansion_level": "neighbor",
                "chunks": _dedup_chunks(neighbor_chunks),
            }
        )

    return sorted(chunk_groups, key=_chunk_group_sort_key)


def _fill_page_gaps(
    session: Any,
    doc_id: str,
    existing_chunks: List[ChunkData],
    max_gap_pages: int,
) -> Tuple[List[ChunkData], int, int]:
    """Fill gaps between retrieved page ranges.

    If pages [4, 5, 9, 10] are retrieved and max_gap_pages=3, the gap
    between 5 and 9 is 3 pages (6, 7, 8), so we fill it to get
    [4, 5, 6, 7, 8, 9, 10].

    Args:
        session: Database session.
        doc_id: Document UUID.
        existing_chunks: Already retrieved chunks.
        max_gap_pages: Maximum gap size (in pages) to fill.

    Returns:
        Tuple of (gap_chunks, gap_ranges_filled, gap_pages_filled):
        - gap_chunks: List of chunks that fill the gaps (not including existing)
        - gap_ranges_filled: Number of gaps that were filled
        - gap_pages_filled: Total pages added via gap filling
    """
    if not existing_chunks or max_gap_pages <= 0:
        return [], 0, 0

    # Get unique page numbers, sorted
    existing_pages = sorted(
        set(c.get("page_number") for c in existing_chunks if c.get("page_number"))
    )

    if len(existing_pages) < 2:
        return [], 0, 0

    # Find gaps to fill
    gap_chunks: List[ChunkData] = []
    gap_ranges_filled = 0
    gap_pages_filled = 0
    existing_chunk_ids = set(c.get("chunk_id") for c in existing_chunks)

    for i in range(len(existing_pages) - 1):
        current_page = existing_pages[i]
        next_page = existing_pages[i + 1]
        gap_size = next_page - current_page - 1

        # If gap is within limit, fill it
        if 0 < gap_size <= max_gap_pages:
            gap_start = current_page + 1
            gap_end = next_page - 1

            fetched_chunks = _fetch_chunks_by_page_range(
                session, doc_id, gap_start, gap_end
            )

            # Only add chunks we don't already have
            new_chunks = [
                c for c in fetched_chunks if c.get("chunk_id") not in existing_chunk_ids
            ]

            if new_chunks:
                gap_chunks.extend(new_chunks)
                existing_chunk_ids.update(c.get("chunk_id") for c in new_chunks)
                gap_ranges_filled += 1
                gap_pages_filled += gap_size

                logger.debug(
                    "Filled gap pages %d-%d (%d pages, %d chunks) for document %s",
                    gap_start,
                    gap_end,
                    gap_size,
                    len(new_chunks),
                    doc_id,
                )

    if gap_chunks:
        logger.info(
            "Gap filling for %s: filled %d gaps, added %d pages (%d chunks)",
            doc_id,
            gap_ranges_filled,
            gap_pages_filled,
            len(gap_chunks),
        )

    return gap_chunks, gap_ranges_filled, gap_pages_filled


def _compute_expansion_metrics(
    chunk_groups: List[ChunkGroup],
    page_count: Optional[int],
    retrieval_method: Literal["full_context", "similarity_expansion"],
    seed_chunks_count: int = 0,
    seed_primary_sections: Optional[Set[Optional[int]]] = None,
    gap_ranges_filled: int = 0,
    gap_pages_filled: int = 0,
) -> ExpansionMetrics:
    """Compute expansion metrics from chunk groups.

    Args:
        chunk_groups: List of chunk groups with expansion levels.
        page_count: Document page count.
        retrieval_method: How chunks were retrieved.
        seed_chunks_count: Number of seed chunks from similarity search.
        seed_primary_sections: Set of primary section numbers in seed chunks.
        gap_ranges_filled: Number of gaps that were filled.
        gap_pages_filled: Total pages added via gap filling.

    Returns:
        ExpansionMetrics: Computed metrics for tracking.
    """
    # Count groups by expansion level
    groups_document_full = 0
    groups_primary_section = 0
    groups_subsection = 0
    groups_neighbor = 0

    all_chunks: List[ChunkData] = []
    all_pages: Set[int] = set()
    all_primary_sections: Set[Optional[int]] = set()

    for group in chunk_groups:
        level = group.get("expansion_level", "neighbor")
        if level == "document_full":
            groups_document_full += 1
        elif level == "primary_section":
            groups_primary_section += 1
        elif level == "subsection":
            groups_subsection += 1
        else:
            groups_neighbor += 1

        for chunk in group.get("chunks", []):
            all_chunks.append(chunk)
            if chunk.get("page_number") is not None:
                all_pages.add(chunk["page_number"])
            all_primary_sections.add(chunk.get("primary_section_number"))

    final_chunks_count = len(_dedup_chunks(all_chunks))
    seed_count = seed_chunks_count if seed_chunks_count > 0 else final_chunks_count

    return {
        "document_page_count": page_count,
        "retrieval_method": retrieval_method,
        "seed_chunks_count": seed_chunks_count,
        "seed_primary_sections_count": len(seed_primary_sections) if seed_primary_sections else 0,
        "groups_total": len(chunk_groups),
        "groups_document_full": groups_document_full,
        "groups_primary_section": groups_primary_section,
        "groups_subsection": groups_subsection,
        "groups_neighbor": groups_neighbor,
        "gap_ranges_filled": gap_ranges_filled,
        "gap_pages_filled": gap_pages_filled,
        "final_chunks_count": final_chunks_count,
        "final_pages_count": len(all_pages),
        "final_primary_sections_count": len(all_primary_sections),
        "expansion_ratio": round(final_chunks_count / seed_count, 2) if seed_count > 0 else 1.0,
    }


def _log_expansion_metrics(
    document_name: str,
    metrics: ExpansionMetrics,
) -> None:
    """Log expansion metrics for observability.

    Args:
        document_name: Name of the document.
        metrics: Computed expansion metrics.
    """
    gap_info = ""
    if metrics["gap_pages_filled"] > 0:
        gap_info = f", gap_fill={metrics['gap_ranges_filled']} ranges/{metrics['gap_pages_filled']} pages"

    logger.info(
        "Expansion metrics for '%s': method=%s, seed=%d chunks from %d sections, "
        "groups=%d (full=%d, primary=%d, subsec=%d, neighbor=%d)%s, "
        "final=%d chunks/%d pages, expansion_ratio=%.2f",
        document_name,
        metrics["retrieval_method"],
        metrics["seed_chunks_count"],
        metrics["seed_primary_sections_count"],
        metrics["groups_total"],
        metrics["groups_document_full"],
        metrics["groups_primary_section"],
        metrics["groups_subsection"],
        metrics["groups_neighbor"],
        gap_info,
        metrics["final_chunks_count"],
        metrics["final_pages_count"],
        metrics["expansion_ratio"],
    )


def fetch_chunks_with_hierarchical_expansion(
    document_ids: List[str],
    data_source: str,
    research_config: Dict[str, Any],
    query_embedding: Optional[List[float]] = None,
) -> List[DocumentChunks]:
    """
    Fetch chunks using hierarchical expansion and document structure.

    Steps:
        1. If page_count is below threshold, fetch full document content.
        2. Otherwise run similarity search to get top chunks.
        3. Expand by primary section, subsection, or neighbor ranges.
        4. Return grouped chunks for XML formatting.

    Args:
        document_ids: Document UUIDs to fetch.
        data_source: Database source.
        research_config: Registry config containing expansion thresholds.
        query_embedding: Query embedding for similarity search (required for
            large documents).

    Returns:
        List[DocumentChunks]: Documents with grouped chunks.

    Raises:
        FileResearchError: If configuration is missing or retrieval fails.
    """

    required_config = [
        "max_pages_for_full_context",
        "max_chunks_per_file",
        "max_primary_section_page_count",
        "max_subsection_page_count",
        "max_neighbour_chunks",
        "max_gap_fill_pages",
    ]
    missing_config = [field for field in required_config if field not in research_config]
    if missing_config:
        raise FileResearchError(
            f"Missing research_config fields for {data_source}: {missing_config}"
        )

    embedding_str = (
        "[" + ",".join(str(x) for x in query_embedding) + "]"
        if query_embedding
        else None
    )

    logger.info(
        "Fetching chunks with hierarchical expansion for %d documents from %s",
        len(document_ids),
        data_source,
    )

    documents: List[DocumentChunks] = []

    try:
        with get_database_session() as session:
            for doc_id in document_ids:
                metadata = _fetch_document_metadata(session, doc_id, data_source)
                page_count = metadata.get("page_count")

                if (
                    page_count is not None
                    and page_count <= research_config["max_pages_for_full_context"]
                ):
                    logger.info(
                        "Document %s has %s pages <= threshold %s. Fetching full context.",
                        metadata["title"],
                        page_count,
                        research_config["max_pages_for_full_context"],
                    )
                    all_chunks = _fetch_all_chunks_for_document(
                        session, doc_id
                    )
                    chunk_groups = _group_full_document_chunks(all_chunks)

                    # Compute metrics for full context retrieval
                    metrics = _compute_expansion_metrics(
                        chunk_groups=chunk_groups,
                        page_count=page_count,
                        retrieval_method="full_context",
                        seed_chunks_count=len(all_chunks),
                        seed_primary_sections=set(
                            c.get("primary_section_number") for c in all_chunks
                        ),
                    )
                    _log_expansion_metrics(metadata["title"], metrics)

                    documents.append(
                        {
                            "document_id": metadata["document_id"],
                            "document_name": metadata["title"],
                            "file_name": metadata.get("file_name"),
                            "chunks": all_chunks,
                            "chunk_groups": chunk_groups,
                            "expansion_metrics": metrics,
                        }
                    )
                    continue

                if not embedding_str:
                    raise FileResearchError(
                        "query_embedding is required for similarity-based retrieval"
                    )

                seed_chunks = _fetch_similarity_chunks_for_document(
                    session,
                    doc_id,
                    embedding_str,
                    research_config["max_chunks_per_file"],
                )

                if not seed_chunks:
                    raise FileResearchError(
                        f"No chunks found for document {metadata['title']}"
                    )

                expanded_groups = _expand_chunks_for_document(
                    session, doc_id, seed_chunks, research_config
                )
                if not expanded_groups:
                    logger.warning(
                        "Hierarchical expansion returned no groups for document '%s' "
                        "in %s; falling back to seed chunks as single group",
                        metadata["title"],
                        data_source,
                    )
                    expanded_groups = [
                        {
                            "primary_section_number": seed_chunks[0].get(
                                "primary_section_number"
                            ),
                            "primary_section_name": _first_non_empty(
                                seed_chunks, "primary_section_name"
                            ),
                            "hierarchy_path": _first_non_empty(
                                seed_chunks, "hierarchy_path"
                            ),
                            "subsection_number": seed_chunks[0].get(
                                "subsection_number"
                            ),
                            "subsection_name": _first_non_empty(
                                seed_chunks, "subsection_name"
                            ),
                            "expansion_level": "neighbor",
                            "chunks": _dedup_chunks(seed_chunks),
                        }
                    ]

                aggregated_chunks = _dedup_chunks(
                    [chunk for group in expanded_groups for chunk in group["chunks"]]
                )

                # Gap filling: fill small gaps between retrieved page ranges
                gap_ranges_filled = 0
                gap_pages_filled = 0
                max_gap_pages = research_config.get("max_gap_fill_pages", 0)

                if max_gap_pages > 0 and aggregated_chunks:
                    gap_chunks, gap_ranges_filled, gap_pages_filled = _fill_page_gaps(
                        session,
                        doc_id,
                        aggregated_chunks,
                        max_gap_pages,
                    )
                    if gap_chunks:
                        # Add gap chunks to aggregated set
                        aggregated_chunks = _dedup_chunks(
                            aggregated_chunks + gap_chunks
                        )

                # Compute metrics for similarity-based expansion
                seed_primary_sections = set(
                    c.get("primary_section_number") for c in seed_chunks
                )
                metrics = _compute_expansion_metrics(
                    chunk_groups=expanded_groups,
                    page_count=page_count,
                    retrieval_method="similarity_expansion",
                    seed_chunks_count=len(seed_chunks),
                    seed_primary_sections=seed_primary_sections,
                    gap_ranges_filled=gap_ranges_filled,
                    gap_pages_filled=gap_pages_filled,
                )
                _log_expansion_metrics(metadata["title"], metrics)

                documents.append(
                    {
                        "document_id": metadata["document_id"],
                        "document_name": metadata["title"],
                        "file_name": metadata.get("file_name"),
                        "chunks": aggregated_chunks,
                        "chunk_groups": expanded_groups,
                        "expansion_metrics": metrics,
                    }
                )

            logger.info(
                "Retrieved expanded chunks for %d documents from %s",
                len(documents),
                data_source,
            )
    except FileResearchError:
        raise
    except Exception as exc:
        raise FileResearchError(
            f"Error fetching hierarchical chunks for {data_source}: {exc}"
        ) from exc

    return documents


@lru_cache(maxsize=1)
def load_file_research_prompt_config() -> Dict[str, Any]:
    """
    Load file research configuration from PostgreSQL.

    Returns:
        Dict[str, Any]: Configuration with system prompt, tools, and user_prompt.

    Raises:
        ValueError: Raised when prompt is missing in the database.
    """
    system_prompt, tools, user_prompt = get_prompt(
        "subagent", "file_research", inject_fiscal=True
    )

    if not user_prompt:
        raise ValueError(
            "user_prompt not found in database for subagent/file_research. "
            "Please ensure the prompt is configured in the prompts table."
        )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "tools": tools,
    }


def format_document_chunks_for_llm(
    document: DocumentChunks,
    enable_dense_table_retrieval: bool = True,
) -> str:
    """Format a document's chunks for LLM research synthesis.

    When a chunk is a dense table description and retrieval is enabled,
    the raw table data is fetched and injected into the XML context.

    Args:
        document: Document with all its chunks.
        enable_dense_table_retrieval: Whether to fetch raw dense table data.

    Returns:
        Formatted string for LLM prompt.
    """
    if not document["chunks"]:
        return "No content available.\n"

    chunk_groups = document.get("chunk_groups") or _group_full_document_chunks(
        document["chunks"]
    )
    if not chunk_groups:
        return "No content available.\n"

    return _format_chunk_groups_xml(
        document, chunk_groups, enable_dense_table_retrieval
    )


def _format_chunk_xml(
    chunk: ChunkData,
    indent: str = "  ",
    document_id: Optional[str] = None,
    enable_dense_table_retrieval: bool = True,
) -> List[str]:
    """Render a single chunk as XML with content markers.

    When the chunk is a dense table description and dense table retrieval is
    enabled, the raw table data is fetched and appended after the content.
    """
    page_number = chunk.get("page_number")
    page_attr = page_number if page_number is not None else "unknown"
    content_lines = (chunk.get("content") or "").splitlines() or [""]
    lines = [
        f'{indent}<chunk page="{page_attr}">',
        f"{indent}  <content_start/>",
    ]
    for line in content_lines:
        lines.append(f"{indent}  {line}")
    lines.append(f"{indent}  <content_end/>")

    # Dense table research: inject pre-computed findings
    dense_result = chunk.get("dense_table_research_result")
    if dense_result:
        for dense_line in dense_result.splitlines():
            lines.append(f"{indent}{dense_line}")

    lines.append(f"{indent}</chunk>")
    return lines


def _format_chunk_groups_xml(
    document: DocumentChunks,
    chunk_groups: List[ChunkGroup],
    enable_dense_table_retrieval: bool = True,
) -> str:
    """Format chunk groups into the hierarchical XML structure."""

    file_name = document.get("file_name") or document["document_name"]
    document_id = document.get("document_id")
    primary_sections: Dict[Optional[int], Dict[str, Any]] = {}

    for group in chunk_groups:
        primary_number = group.get("primary_section_number")
        primary_entry = primary_sections.setdefault(
            primary_number,
            {
                "primary_section_name": group.get("primary_section_name"),
                "hierarchy_path": group.get("hierarchy_path"),
                "chunks": [],
                "subsections": {},
            },
        )

        if not primary_entry["primary_section_name"] and group.get(
            "primary_section_name"
        ):
            primary_entry["primary_section_name"] = group.get("primary_section_name")
        if not primary_entry["hierarchy_path"] and group.get("hierarchy_path"):
            primary_entry["hierarchy_path"] = group.get("hierarchy_path")

        if group.get("expansion_level") == "subsection":
            subsection_number = group.get("subsection_number")
            subsections: Dict[Optional[int], Dict[str, Any]] = primary_entry[
                "subsections"
            ]
            subsection_entry = subsections.setdefault(
                subsection_number,
                {
                    "subsection_name": group.get("subsection_name"),
                    "hierarchy_path": group.get("hierarchy_path"),
                    "chunks": [],
                },
            )
            if not subsection_entry["subsection_name"] and group.get(
                "subsection_name"
            ):
                subsection_entry["subsection_name"] = group.get("subsection_name")
            if not subsection_entry.get("hierarchy_path") and group.get(
                "hierarchy_path"
            ):
                subsection_entry["hierarchy_path"] = group.get("hierarchy_path")
            subsection_entry["chunks"].extend(group.get("chunks", []))
        else:
            primary_entry["chunks"].extend(group.get("chunks", []))

    lines = [f'<document id="{document["document_id"]}" filename="{file_name}">']

    for primary_number in sorted(
        primary_sections.keys(), key=_primary_section_sort_key
    ):
        primary_entry = primary_sections[primary_number]
        attrs: List[str] = []
        if primary_entry.get("hierarchy_path"):
            attrs.append(f'hierarchy_path="{primary_entry["hierarchy_path"]}"')
        if primary_entry.get("primary_section_name"):
            attrs.append(f'name="{primary_entry["primary_section_name"]}"')
        attr_str = f" {' '.join(attrs)}" if attrs else ""

        lines.append(f"  <primary_section{attr_str}>")

        subsection_entries: Dict[Optional[int], Dict[str, Any]] = primary_entry.get(
            "subsections", {}
        )
        for subsection_number in sorted(
            subsection_entries.keys(),
            key=lambda s: (1, 0) if s is None else (0, s),
        ):
            subsection_entry = subsection_entries[subsection_number]
            subsection_name = subsection_entry.get("subsection_name") or (
                f"Subsection {subsection_number}"
                if subsection_number is not None
                else "Subsection"
            )
            subsection_attrs: List[str] = []
            if subsection_entry.get("hierarchy_path"):
                subsection_attrs.append(
                    f'hierarchy_path="{subsection_entry["hierarchy_path"]}"'
                )
            if subsection_name:
                subsection_attrs.append(f'name="{subsection_name}"')
            subsection_attr_str = (
                f" {' '.join(subsection_attrs)}" if subsection_attrs else ""
            )

            lines.append(f"    <subsection{subsection_attr_str}>")
            for chunk in _dedup_chunks(subsection_entry.get("chunks", [])):
                lines.extend(_format_chunk_xml(
                    chunk,
                    indent="      ",
                    document_id=document_id,
                    enable_dense_table_retrieval=enable_dense_table_retrieval,
                ))
            lines.append("    </subsection>")

        for chunk in _dedup_chunks(primary_entry.get("chunks", [])):
            lines.extend(_format_chunk_xml(
                chunk,
                indent="    ",
                document_id=document_id,
                enable_dense_table_retrieval=enable_dense_table_retrieval,
            ))

        lines.append("  </primary_section>")

    lines.append("</document>")
    return "\n".join(lines)


def _build_llm_messages(
    system_prompt: str,
    user_prompt: str,
) -> List[Dict[str, str]]:
    """
    Build message list for LLM call.

    Args:
        system_prompt (str): The system prompt content.
        user_prompt (str): The user prompt content from database.

    Returns:
        List[Dict[str, str]]: Messages for the LLM call.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_tool_response(
    result: Any,
    file_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Parse LLM tool call response.

    Args:
        result (Any): Response from execute_llm_call (may be tuple or response object).
        file_name (str): File name for page research entries.

    Returns:
        Optional[Dict[str, Any]]: Parsed arguments if a tool call is present.
    """
    if isinstance(result, tuple) and len(result) == 2:
        response, _ = result
    else:
        response = result

    if not (
        response
        and hasattr(response, "choices")
        and response.choices
        and response.choices[0].message
        and response.choices[0].message.tool_calls
    ):
        return None

    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call.function.name != "extract_page_research":
        return None

    try:
        arguments = json.loads(tool_call.function.arguments)
        logger.debug(
            "RESEARCH_OUTPUT [file_research_%s]: Raw LLM tool arguments:\n%s",
            file_name,
            json.dumps(arguments, indent=2),
        )
    except json.JSONDecodeError:
        logger.error("Invalid tool arguments for %s", file_name)
        return None

    page_research: List[PageResearch] = [
        {
            "page_number": page_item.get("page_number"),
            # LLM returns finding; map to research_content for internal consistency.
            "research_content": (
                page_item.get("finding")
                or page_item.get("extracted_fact")
                or page_item.get("research_content")
                or ""
            ),
            "file_link": file_name,
            "file_name": file_name,
        }
        for page_item in arguments.get("page_research", [])
    ]

    logger.debug(
        "RESEARCH_OUTPUT [file_research_%s]: Parsed page findings:\n%s",
        file_name,
        json.dumps(
            [
                {"page": pr["page_number"], "finding": pr["research_content"]}
                for pr in page_research
            ],
            indent=2,
        ),
    )

    return {
        "page_research": page_research,
        "status_summary": arguments.get("status_summary"),
    }


def _track_llm_usage(
    result: Any,
    context: SynthesisContext,
) -> None:
    """
    Track LLM usage if process monitor available.

    Args:
        result (Any): The result from execute_llm_call.
        context (SynthesisContext): Contains process_monitor and stage_name.
    """
    process_monitor = context.get("process_monitor")
    stage_name = context.get("stage_name")

    if isinstance(result, tuple) and len(result) == 2:
        _, usage_details = result
        if usage_details and process_monitor and stage_name:
            process_monitor.add_llm_call_details_to_stage(stage_name, usage_details)


def synthesize_document_research(
    research_statement: str,
    document: DocumentChunks,
    synthesis_context: Optional[SynthesisContext] = None,
) -> DocumentResearch:
    """
    Synthesize research findings for a single document using LLM.

    Args:
        research_statement (str): The research query.
        document (DocumentChunks): Document with chunks to analyze.
        synthesis_context (SynthesisContext | None): Optional context containing token,
            data_source, process_monitor, and stage_name.

    Returns:
        DocumentResearch: Page-level findings for the document.

    Raises:
        FileResearchError: Raised when synthesis fails or response is invalid.
    """
    ctx = synthesis_context or {}
    doc_name = document["document_name"]
    file_name = document.get("file_name") or doc_name
    logger.info(
        "Synthesizing research for %s from %s",
        doc_name,
        ctx.get("data_source", "unknown"),
    )

    if not document["chunks"]:
        raise FileResearchError(f"No content available for document {doc_name}")

    try:
        prompt_config = load_file_research_prompt_config()
        enable_dense = ctx.get("enable_dense_table_retrieval", True)
        token = ctx.get("token")

        # Pre-process: run dense table research on applicable chunks
        if enable_dense and token:
            doc_id = document.get("document_id")
            for chunk in document.get("chunks", []):
                if chunk.get("is_dense_table_description") and doc_id:
                    logger.info(
                        "Dense table research: %s page %s — starting",
                        doc_name,
                        chunk.get("page_number"),
                    )
                    dt_result = research_dense_table(
                        research_statement=research_statement,
                        document_id=doc_id,
                        routing_json=chunk.get("dense_table_routing_json"),
                        chunk_description=chunk.get("content", ""),
                        token=token,
                    )
                    if dt_result:
                        chunk["dense_table_research_result"] = dt_result
                        logger.info(
                            "Dense table research: %s page %s — "
                            "%d chars of findings",
                            doc_name,
                            chunk.get("page_number"),
                            len(dt_result),
                        )

        document_content = format_document_chunks_for_llm(
            document, enable_dense_table_retrieval=enable_dense
        )

        logger.debug(
            "RESEARCH_INPUT [file_research_%s]: Document content sent to LLM:\n%s",
            doc_name,
            document_content[:15000] if len(document_content) > 15000 else document_content,
        )

        system_prompt = (
            prompt_config.get("system_prompt", "")
            .replace("{{research_statement}}", research_statement)
            .replace("{{document_content}}", document_content)
            .replace("{{document_name}}", doc_name)
        )

        user_prompt = (
            prompt_config.get("user_prompt", "")
            .replace("{{research_statement}}", research_statement)
            .replace("{{document_content}}", document_content)
            .replace("{{document_name}}", doc_name)
        )

        token = ctx.get("token")
        if not token:
            raise FileResearchError(
                "OAuth token required for LLM call in file research synthesis"
            )

        model_config = config.get_model_settings(MODEL_CAPABILITY)

        result = execute_llm_call(
            oauth_token=token,
            model=model_config["name"],
            messages=_build_llm_messages(system_prompt, user_prompt),
            max_tokens=MODEL_MAX_TOKENS,
            temperature=MODEL_TEMPERATURE,
            tools=prompt_config.get("tools", []),
            tool_choice={
                "type": "function",
                "function": {"name": "extract_page_research"},
            },
            stream=False,
            prompt_token_cost=model_config["prompt_token_cost"],
            completion_token_cost=model_config["completion_token_cost"],
            reasoning_effort=model_config.get("reasoning_effort"),
        )

        _track_llm_usage(result, ctx)

        parsed = _parse_tool_response(result, file_name)
        if not parsed:
            raise FileResearchError(
                f"No valid tool response from LLM for document {doc_name}"
            )

        return {
            "document_name": doc_name,
            "file_link": file_name,
            "status_summary": parsed.get("status_summary") or f"Analyzed {doc_name}",
            "page_research": parsed.get("page_research", []),
        }

    except FileResearchError:
        raise
    except Exception as exc:
        raise FileResearchError(
            f"Error synthesizing research for {doc_name}: {exc}"
        ) from exc


def _build_findings_from_research(
    document_results: List[DocumentResearch],
    document_id_lookup: Dict[str, str],
    data_source: str,
) -> FindingsList:
    """
    Build unified findings list from document research results.

    Args:
        document_results: Per-document research results.
        document_id_lookup: Mapping of document_name to document_id.
        data_source: Database identifier.

    Returns:
        List of Finding objects in unified format.
    """
    findings: FindingsList = []

    for result in document_results:
        doc_name = result["document_name"]
        page_research = result.get("page_research", [])
        doc_id = document_id_lookup.get(doc_name, "")

        if page_research and not result["status_summary"].startswith("Error"):
            for page_item in sorted(
                page_research, key=lambda x: x.get("page_number", 0)
            ):
                page_number = page_item.get("page_number", 0)

                file_link = page_item.get("file_link", "") or result.get(
                    "file_link", ""
                )
                file_name = page_item.get("file_name", "") or result.get(
                    "file_link", ""
                )
                content = (
                    page_item.get("research_content")
                    or page_item.get("extracted_fact")
                    or ""
                )

                if content:  # Only add findings with actual content
                    findings.append({
                        "document_id": doc_id,
                        "document_name": doc_name,
                        "file_name": file_name,
                        "file_link": file_link,
                        "page": page_number,
                        "finding": content.strip(),
                        "source": "file_research",
                        "data_source": data_source,
                    })

    return findings


def _build_status_summary(
    findings: FindingsList,
    data_source: str,
) -> str:
    """
    Build status summary string from findings.

    Args:
        findings: List of Finding objects.
        data_source: Database source identifier.

    Returns:
        str: Human-readable status summary.
    """
    if not findings:
        return f"No relevant information found in {data_source} documents"

    unique_docs = set(f["document_name"] for f in findings)
    unique_pages = set((f["document_name"], f["page"]) for f in findings)
    return (
        f"Found {len(findings)} finding(s) in {len(unique_docs)} "
        f"document(s) across {len(unique_pages)} page(s)"
    )


def _process_documents_parallel(
    research_statement: str,
    documents: List[DocumentChunks],
    data_source: str,
    ctx: ResearchContext,
    research_config: Dict[str, Any],
) -> List[DocumentResearch]:
    """
    Process documents in parallel using ThreadPoolExecutor.

    Args:
        research_statement (str): The research query.
        documents (List[DocumentChunks]): Documents with chunks.
        data_source (str): Database source identifier.
        ctx (ResearchContext): Research context with token, process_monitor, stage_name.
        research_config (Dict[str, Any]): Research configuration from registry.

    Returns:
        List[DocumentResearch]: Research results for each document.
    """
    synthesis_ctx: SynthesisContext = {
        "token": ctx.get("token"),
        "data_source": data_source,
        "process_monitor": ctx.get("process_monitor"),
        "stage_name": ctx.get("stage_name"),
        "enable_dense_table_retrieval": research_config.get(
            "enable_dense_table_retrieval", True
        ),
    }

    max_parallel = research_config["max_parallel_files"]
    results: List[DocumentResearch] = []

    with ThreadPoolExecutor(max_workers=min(len(documents), max_parallel)) as executor:
        future_to_doc = {
            executor.submit(
                synthesize_document_research,
                research_statement,
                doc,
                synthesis_ctx,
            ): doc
            for doc in documents
        }

        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                results.append(future.result())
                logger.info("Completed research for: %s", doc["document_name"])
            except Exception as exc:
                raise FileResearchError(
                    f"Failed to process document {doc.get('document_name')}: {exc}"
                ) from exc

    return results


def execute_file_research_sync(
    research_statement: str,
    document_ids: List[str],
    data_source: str,
    research_context: Optional[ResearchContext] = None,
) -> FileResearchResult:
    """
    Stage 2: Deep research on selected documents.

    This is the main entry point for the File Research Subagent.

    Uses hierarchical chunk retrieval with similarity search for larger files,
    and full-context retrieval for small documents. Retrieved chunks are
    expanded by primary section, subsection, or neighbor ranges to provide
    coherent context for the LLM.

    Args:
        research_statement (str): The research query/statement.
        document_ids (List[str]): Document UUIDs to research (from Stage 1).
        data_source (str): Database source (e.g., 'internal_capm', 'external_ey').
        research_context (ResearchContext | None): Context containing:
            - token: OAuth token for API calls
            - process_monitor: For tracking
            - stage_name: For tracking
            - query_embedding: Embedding for similarity-based retrieval of
                large documents

    Returns:
        FileResearchResult: Documents, status_summary, reference_index, and data_source.

    Raises:
        FileResearchError: Raised when inputs are invalid or processing fails.
    """
    ctx = research_context or {}
    query_embedding = ctx.get("query_embedding")

    if not document_ids:
        raise FileResearchError("No document_ids provided for file research")

    logger.info(
        "Starting file research for %d documents in %s",
        len(document_ids),
        data_source,
    )

    research_config = DatabaseMetadataCache().get_research_config(data_source)

    documents = fetch_chunks_with_hierarchical_expansion(
        document_ids=document_ids,
        data_source=data_source,
        research_config=research_config,
        query_embedding=query_embedding,
    )

    if not documents:
        raise FileResearchError(
            f"No content found for any of the {len(document_ids)} "
            f"selected documents in {data_source}"
        )

    document_results = _process_documents_parallel(
        research_statement, documents, data_source, ctx, research_config
    )

    # Build document_id lookup from fetched documents
    document_id_lookup: Dict[str, str] = {
        doc["document_name"]: doc["document_id"] for doc in documents
    }

    # Extract retrieval paths per document for logging
    retrieval_paths = {
        doc["document_name"]: doc.get("expansion_metrics", {}).get(
            "retrieval_method", "unknown"
        )
        for doc in documents
    }

    # Convert to unified findings format
    findings = _build_findings_from_research(
        document_results, document_id_lookup, data_source
    )
    status_summary = _build_status_summary(findings, data_source)

    logger.info("File research complete for %s: %s", data_source, status_summary)

    return {
        "findings": findings,
        "status_summary": status_summary,
        "data_source": data_source,
        "retrieval_paths": retrieval_paths,
    }
