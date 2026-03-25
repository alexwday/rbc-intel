"""Stage 1: Filesystem scanning and catalog diff."""

import json
import os
from dataclasses import asdict
from typing import List

from ..utils.postgres import fetch_catalog_records
from ..utils.config import get_data_source_path
from ..utils.file_types import (
    DiscoveryDiff,
    FileRecord,
    compute_file_hash,
)
from ..utils.logging_setup import get_stage_logger
from .startup import PROCESSING_DIR

STAGE = "1-DISCOVERY"


def _parse_path_parts(rel_path: str) -> tuple:
    """Split a relative path into data_source and filters.

    Params:
        rel_path: Path relative to base, e.g. "src/2026/Q1/RBC"

    Returns:
        tuple of (data_source, filter_1, filter_2, filter_3)
    """
    parts = rel_path.split(os.sep)
    data_source = parts[0]
    filter_1 = parts[1] if len(parts) > 1 else ""
    filter_2 = parts[2] if len(parts) > 2 else ""
    filter_3 = os.sep.join(parts[3:]) if len(parts) > 3 else ""
    return data_source, filter_1, filter_2, filter_3


def _build_file_record(
    dirpath: str, fname: str, path_parts: tuple
) -> FileRecord:
    """Build a FileRecord from directory context and filename.

    Params:
        dirpath: Directory containing the file
        fname: Filename
        path_parts: Tuple of (data_source, f1, f2, f3)

    Returns:
        FileRecord
    """
    full_path = os.path.join(dirpath, fname)
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    stat = os.stat(full_path)
    return FileRecord(
        data_source=path_parts[0],
        filter_1=path_parts[1],
        filter_2=path_parts[2],
        filter_3=path_parts[3],
        filename=fname,
        filetype=ext,
        file_size=stat.st_size,
        date_last_modified=stat.st_mtime,
        file_hash="",
        file_path=full_path,
    )


def scan_filesystem(base_path: str) -> List[FileRecord]:
    """Walk a data-source tree and build FileRecords.

    First subfolder under base_path is the data_source.
    The next 1-3 levels become filter_1, filter_2, filter_3.
    Deeper nesting is flattened into filter_3.
    Hidden files and directories are skipped.

    Params:
        base_path: Absolute path to the data sources root

    Returns:
        list[FileRecord] — one per file found

    Example:
        >>> records = scan_filesystem("/data/sources")
        >>> records[0].data_source
        "policy_docs"
    """
    logger = get_stage_logger(__name__, STAGE)
    records: List[FileRecord] = []

    if not os.path.isdir(base_path):
        logger.error("Base path does not exist: %s", base_path)
        return records

    for dirpath, dirnames, filenames in os.walk(base_path):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        rel = os.path.relpath(dirpath, base_path)
        if rel == ".":
            continue

        path_parts = _parse_path_parts(rel)

        for fname in filenames:
            if fname.startswith("."):
                continue
            records.append(_build_file_record(dirpath, fname, path_parts))

    logger.info(
        "Scanned %d files across %s",
        len(records),
        base_path,
    )
    return records


def compute_diff(
    discovered: List[FileRecord],
    cataloged: List[FileRecord],
) -> DiscoveryDiff:
    """Compare discovered files against the catalog.

    Keys on file_path. New = on disk only. Deleted = in DB only.
    Modified = path matches but size differs, or size matches
    but date differs and hash differs (lazy hash). Unchanged =
    path + size + date match.

    Params:
        discovered: Files found on the filesystem
        cataloged: Files from the database catalog

    Returns:
        DiscoveryDiff with new, modified, deleted lists

    Example:
        >>> diff = compute_diff(disk_files, db_files)
        >>> len(diff.new)
        5
    """
    catalog_map = {r.file_path: r for r in cataloged}
    disk_map = {r.file_path: r for r in discovered}

    new: List[FileRecord] = []
    modified: List[FileRecord] = []
    deleted: List[FileRecord] = []

    for path, disk_rec in disk_map.items():
        if path not in catalog_map:
            new.append(disk_rec)
            continue

        cat_rec = catalog_map[path]
        if disk_rec.file_size != cat_rec.file_size:
            modified.append(disk_rec)
        elif disk_rec.date_last_modified != cat_rec.date_last_modified:
            disk_hash = compute_file_hash(path)
            cat_hash = cat_rec.file_hash
            if disk_hash != cat_hash:
                disk_rec.file_hash = disk_hash
                modified.append(disk_rec)

    for path, cat_rec in catalog_map.items():
        if path not in disk_map:
            deleted.append(cat_rec)

    return DiscoveryDiff(new=new, modified=modified, deleted=deleted)


def run_discovery(conn) -> DiscoveryDiff:
    """Orchestrate filesystem scan, catalog fetch, and diff.

    Reads DATA_SOURCE_PATH from config, scans the tree, fetches
    the existing catalog from the database, computes the diff,
    and logs a summary.

    Params:
        conn: psycopg2 database connection

    Returns:
        DiscoveryDiff with new, modified, deleted lists

    Example:
        >>> diff = run_discovery(conn)
        >>> print(f"{len(diff.new)} new files")
    """
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Starting file discovery")

    base_path = get_data_source_path()
    discovered = scan_filesystem(base_path)
    cataloged = fetch_catalog_records(conn)
    diff = compute_diff(discovered, cataloged)

    output = {
        "new": [asdict(r) for r in diff.new],
        "modified": [asdict(r) for r in diff.modified],
        "deleted": [asdict(r) for r in diff.deleted],
    }
    output_path = PROCESSING_DIR / "discovery.json"
    output_path.write_text(json.dumps(output, indent=2))

    logger.info(
        "Discovery complete — new: %d, modified: %d, "
        "deleted: %d, unchanged: %d",
        len(diff.new),
        len(diff.modified),
        len(diff.deleted),
        len(discovered) - len(diff.new) - len(diff.modified),
    )
    return diff
