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
