"""Unified types for the research retrieval system.

Defines the core data structures used across metadata subagent, file research
subagent, database router, and summarizer.

The research flow produces Finding objects which are consolidated and assigned
reference IDs to become IndexedFinding objects for the summarizer.
"""

from typing import Dict, List, Literal, Optional, TypedDict


class Finding(TypedDict):
    """A single research finding from any source.

    Each finding represents one piece of information from one page of one
    document.
    """

    document_id: str
    document_name: str
    file_name: str
    file_link: str
    page: Optional[int]
    finding: str
    source: Literal["metadata", "file_research"]
    data_source: str


class IndexedFinding(Finding):
    """Finding with assigned reference number for citation.

    Extends Finding with a ref_id that the summarizer uses for citations.
    The ref_id is assigned during consolidation across all data sources.
    """

    ref_id: str


# Type aliases for clarity
FindingsList = List[Finding]
IndexedFindingsList = List[IndexedFinding]
FindingsByDataSource = Dict[str, FindingsList]
