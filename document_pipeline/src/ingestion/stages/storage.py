"""Stage 6: Persist finalized outputs into canonical CSV masters."""

import csv
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from ..utils.config import (
    get_data_source_path,
    get_storage_master_dir,
    get_storage_push_to_postgres,
)
from ..utils.file_types import FileRecord, compute_file_hash
from ..utils.logging_setup import get_stage_logger
from ..utils.postgres import (
    STORAGE_TABLE_COLUMNS,
    STORAGE_TABLE_ORDER,
    refresh_storage_tables,
)
from .startup import ARCHIVE_DIR, PROCESSING_DIR

STAGE = "6-STORAGE"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FINALIZATION_DIR = PROCESSING_DIR / "finalization"
STORAGE_DIR = PROCESSING_DIR / "storage"
NULL_MARKER = r"\N"
StorageRow = dict[str, str]
StorageRows = dict[str, list[StorageRow]]


def _storage_master_dir() -> Path:
    """Resolve the configured master CSV directory. Returns: Path."""
    master_dir = Path(get_storage_master_dir()).expanduser()
    if master_dir.is_absolute():
        return master_dir
    return PROJECT_ROOT / master_dir


def _configure_csv_field_limit() -> None:
    """Raise the CSV field limit so large embeddings/content can be read."""
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            return
        except OverflowError:
            field_limit //= 10


def _stable_id(*parts: Any) -> str:
    """Build a deterministic SHA-256 identifier from stringable parts."""
    payload = "||".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector_cell(vector: Any) -> str:
    """Serialize an embedding list into pgvector text format."""
    if not isinstance(vector, list) or not vector:
        return NULL_MARKER
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _clean_text(value: str) -> str:
    """Remove characters that cannot be stored in CSV/Postgres text."""
    return value.replace("\x00", "")


def _json_cell(value: Any) -> str:
    """Serialize a JSON-compatible value for CSV storage."""
    return json.dumps(value, ensure_ascii=True)


def _cell_value(value: Any) -> str:
    """Serialize one scalar value for CSV output."""
    if value is None:
        return NULL_MARKER
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return _json_cell(value)
    return _clean_text(str(value))


def _text_value(value: Any) -> str:
    """Normalize a metadata field into a display string."""
    if isinstance(value, list):
        return "; ".join(
            _clean_text(str(item)).strip()
            for item in value
            if _clean_text(str(item)).strip()
        )
    if isinstance(value, dict):
        return _json_cell(value)
    return _clean_text(str(value or "")).strip()


def _summary_text(summary: dict[str, Any], key: str) -> str:
    """Read a string summary field with an empty-string default."""
    return _clean_text(str(summary.get(key, "") or "")).strip()


def _summary_list(summary: dict[str, Any], key: str) -> list[Any]:
    """Read a list summary field with an empty-list default."""
    value = summary.get(key, [])
    return value if isinstance(value, list) else []


def _summary_dict(summary: dict[str, Any], key: str) -> dict[str, Any]:
    """Read a dict summary field with an empty-dict default."""
    value = summary.get(key, {})
    return value if isinstance(value, dict) else {}


def _scan_current_filesystem(base_path: str) -> list[FileRecord]:
    """Walk the current input base and return FileRecords."""
    records: list[FileRecord] = []
    for dirpath, dirnames, filenames in os.walk(base_path):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        rel_path = os.path.relpath(dirpath, base_path)
        if rel_path == ".":
            continue
        parts = rel_path.split(os.sep)
        data_source = parts[0]
        filter_1 = parts[1] if len(parts) > 1 else ""
        filter_2 = parts[2] if len(parts) > 2 else ""
        filter_3 = os.sep.join(parts[3:]) if len(parts) > 3 else ""
        for filename in filenames:
            if filename.startswith("."):
                continue
            full_path = os.path.join(dirpath, filename)
            stat = os.stat(full_path)
            filetype = (
                filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            )
            records.append(
                FileRecord(
                    data_source=data_source,
                    filter_1=filter_1,
                    filter_2=filter_2,
                    filter_3=filter_3,
                    filename=filename,
                    filetype=filetype,
                    file_size=stat.st_size,
                    date_last_modified=stat.st_mtime,
                    file_hash="",
                    file_path=full_path,
                )
            )
    return sorted(records, key=lambda record: record.file_path)


def _same_catalog_snapshot(record: FileRecord, row: dict[str, str]) -> bool:
    """Check whether a stored catalog row matches the current file snapshot."""
    try:
        row_size = int(row.get("file_size", ""))
        row_modified = float(row.get("date_last_modified", ""))
    except ValueError:
        return False
    return (
        row_size == record.file_size
        and abs(row_modified - record.date_last_modified) < 1e-9
    )


def _catalog_row(
    record: FileRecord, previous_row: dict[str, str]
) -> dict[str, str]:
    """Build one catalog CSV row, reusing prior hashes when possible."""
    file_hash = ""
    if previous_row and _same_catalog_snapshot(record, previous_row):
        file_hash = previous_row.get("file_hash", "")
    if not file_hash:
        file_hash = compute_file_hash(record.file_path)
    return {
        "data_source": record.data_source,
        "filter_1": record.filter_1,
        "filter_2": record.filter_2,
        "filter_3": record.filter_3,
        "filename": record.filename,
        "filetype": record.filetype,
        "file_size": str(record.file_size),
        "date_last_modified": str(record.date_last_modified),
        "file_hash": file_hash,
        "file_path": record.file_path,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows from disk. Params: path. Returns: list[dict[str, str]]."""
    if not path.is_file():
        return []
    _configure_csv_field_limit()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_existing_master_rows(
    master_dir: Path,
) -> dict[str, list[dict[str, str]]]:
    """Load prior master CSV rows for every storage table."""
    rows: dict[str, list[dict[str, str]]] = {}
    for table_name in STORAGE_TABLE_ORDER:
        csv_path = master_dir / f"{table_name}.csv"
        rows[table_name] = _read_csv_rows(csv_path)
    return rows


def _load_finalization_results() -> dict[str, dict[str, Any]]:
    """Load finalized documents from the current processing directory."""
    logger = get_stage_logger(__name__, STAGE)
    results: dict[str, dict[str, Any]] = {}
    if not FINALIZATION_DIR.is_dir():
        return results
    for json_path in sorted(FINALIZATION_DIR.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Skipping malformed finalization file %s: %s",
                json_path.name,
                exc,
            )
            continue
        file_path = str(payload.get("file_path", "")).strip()
        if file_path:
            results[file_path] = payload
    return results


def _load_archived_finalizations(
    missing_paths: set[str],
) -> dict[str, dict[str, Any]]:
    """Load fallback finalized documents from prior archived runs."""
    logger = get_stage_logger(__name__, STAGE)
    recovered: dict[str, dict[str, Any]] = {}
    if not missing_paths or not ARCHIVE_DIR.is_dir():
        return recovered
    archive_paths = sorted(ARCHIVE_DIR.glob("run_*.zip"), reverse=True)
    for archive_path in archive_paths:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    if not member.startswith("finalization/"):
                        continue
                    if not member.endswith(".json"):
                        continue
                    payload = json.loads(archive.read(member))
                    file_path = str(payload.get("file_path", "")).strip()
                    if (
                        file_path
                        and file_path in missing_paths
                        and file_path not in recovered
                    ):
                        recovered[file_path] = payload
                missing_paths = missing_paths - set(recovered)
                if not missing_paths:
                    return recovered
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            logger.warning(
                "Skipping unreadable archive %s: %s",
                archive_path.name,
                exc,
            )
    return recovered


def _document_row(
    finalized: dict[str, Any],
    catalog_row: StorageRow,
    file_path: str,
    document_id: str,
    metadata: dict[str, Any],
) -> StorageRow:
    """Build the storage row for the documents table."""
    return {
        "document_id": document_id,
        "file_path": file_path,
        "file_name": str(finalized.get("file_name", "")).strip(),
        "filetype": str(finalized.get("filetype", "")).strip(),
        "data_source": catalog_row["data_source"],
        "filter_1": catalog_row["filter_1"],
        "filter_2": catalog_row["filter_2"],
        "filter_3": catalog_row["filter_3"],
        "file_size": catalog_row["file_size"],
        "date_last_modified": catalog_row["date_last_modified"],
        "file_hash": catalog_row["file_hash"],
        "title": _text_value(metadata.get("title", "")),
        "publication_date": _text_value(metadata.get("publication_date", "")),
        "authors_text": _text_value(metadata.get("authors", "")),
        "document_type": _text_value(metadata.get("document_type", "")),
        "abstract": _text_value(metadata.get("abstract", "")),
        "metadata_json": _json_cell(metadata),
        "document_summary": str(finalized.get("document_summary", "")).strip(),
        "document_description": str(
            finalized.get("document_description", "")
        ).strip(),
        "document_usage": str(finalized.get("document_usage", "")).strip(),
        "structure_type": str(finalized.get("structure_type", "")).strip(),
        "structure_confidence": str(
            finalized.get("structure_confidence", "")
        ).strip(),
        "degradation_signals_json": _json_cell(
            finalized.get("degradation_signals", [])
        ),
        "summary_embedding": _vector_cell(
            finalized.get("summary_embedding", [])
        ),
        "page_count": str(int(finalized.get("page_count", 0))),
        "primary_section_count": str(
            int(finalized.get("primary_section_count", 0))
        ),
        "subsection_count": str(int(finalized.get("subsection_count", 0))),
        "chunk_count": str(int(finalized.get("chunk_count", 0))),
        "dense_table_count": str(int(finalized.get("dense_table_count", 0))),
        "extraction_metadata_json": _json_cell(
            finalized.get("extraction_metadata", {})
        ),
    }


def _section_rows(
    sections: Any,
    file_path: str,
    document_id: str,
) -> tuple[list[StorageRow], list[StorageRow]]:
    """Build section and subsection storage rows."""
    section_rows: list[StorageRow] = []
    subsection_rows: list[StorageRow] = []
    if not isinstance(sections, list):
        return section_rows, subsection_rows

    for section in sections:
        if not isinstance(section, dict):
            continue
        summary = section.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        section_number = int(section.get("section_number", 0))
        section_id = _stable_id("section", file_path, section_number)
        section_rows.append(
            {
                "section_id": section_id,
                "document_id": document_id,
                "file_path": file_path,
                "section_number": str(section_number),
                "title": str(section.get("title", "")).strip(),
                "page_start": str(int(section.get("page_start", 0))),
                "page_end": str(int(section.get("page_end", 0))),
                "page_count": str(int(section.get("page_count", 0))),
                "overview": _summary_text(summary, "overview"),
                "key_topics_json": _json_cell(
                    _summary_list(summary, "key_topics")
                ),
                "key_metrics_json": _json_cell(
                    _summary_dict(summary, "key_metrics")
                ),
                "key_findings_json": _json_cell(
                    _summary_list(summary, "key_findings")
                ),
                "notable_facts_json": _json_cell(
                    _summary_list(summary, "notable_facts")
                ),
                "is_fallback": _cell_value(
                    bool(summary.get("is_fallback", False))
                ),
                "summary_json": _json_cell(summary),
            }
        )
        subsection_rows.extend(
            _subsection_rows(
                section.get("subsections", []),
                file_path=file_path,
                document_id=document_id,
                section_id=section_id,
                section_number=section_number,
            )
        )
    return section_rows, subsection_rows


def _subsection_rows(
    subsections: Any,
    file_path: str,
    document_id: str,
    section_id: str,
    section_number: int,
) -> list[StorageRow]:
    """Build subsection storage rows."""
    rows: list[StorageRow] = []
    if not isinstance(subsections, list):
        return rows
    for subsection in subsections:
        if not isinstance(subsection, dict):
            continue
        summary = subsection.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        subsection_number = int(subsection.get("subsection_number", 0))
        rows.append(
            {
                "subsection_id": _stable_id(
                    "subsection",
                    file_path,
                    section_number,
                    subsection_number,
                ),
                "section_id": section_id,
                "document_id": document_id,
                "file_path": file_path,
                "section_number": str(section_number),
                "subsection_number": str(subsection_number),
                "title": str(subsection.get("title", "")).strip(),
                "page_start": str(int(subsection.get("page_start", 0))),
                "page_end": str(int(subsection.get("page_end", 0))),
                "page_count": str(int(subsection.get("page_count", 0))),
                "overview": _summary_text(summary, "overview"),
                "key_topics_json": _json_cell(
                    _summary_list(summary, "key_topics")
                ),
                "key_metrics_json": _json_cell(
                    _summary_dict(summary, "key_metrics")
                ),
                "key_findings_json": _json_cell(
                    _summary_list(summary, "key_findings")
                ),
                "notable_facts_json": _json_cell(
                    _summary_list(summary, "notable_facts")
                ),
                "is_fallback": _cell_value(
                    bool(summary.get("is_fallback", False))
                ),
                "summary_json": _json_cell(summary),
            }
        )
    return rows


def _chunk_rows(
    chunks: Any,
    file_path: str,
    document_id: str,
) -> list[StorageRow]:
    """Build chunk storage rows."""
    rows: list[StorageRow] = []
    if not isinstance(chunks, list):
        return rows
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_number = int(chunk.get("chunk_number", 0))
        rows.append(
            {
                "chunk_id": _stable_id("chunk", file_path, chunk_number),
                "document_id": document_id,
                "file_path": file_path,
                "chunk_number": str(chunk_number),
                "page_number": str(int(chunk.get("page_number", 0))),
                "content": str(chunk.get("content", "")),
                "primary_section_number": str(
                    int(chunk.get("primary_section_number", 0))
                ),
                "primary_section_name": str(
                    chunk.get("primary_section_name", "")
                ).strip(),
                "subsection_number": str(
                    int(chunk.get("subsection_number", 0))
                ),
                "subsection_name": str(
                    chunk.get("subsection_name", "")
                ).strip(),
                "hierarchy_path": str(chunk.get("hierarchy_path", "")).strip(),
                "primary_section_page_count": str(
                    int(chunk.get("primary_section_page_count", 0))
                ),
                "subsection_page_count": str(
                    int(chunk.get("subsection_page_count", 0))
                ),
                "embedding_prefix": str(
                    chunk.get("embedding_prefix", "")
                ).strip(),
                "embedding": _vector_cell(chunk.get("embedding", [])),
                "is_dense_table_description": _cell_value(
                    bool(chunk.get("is_dense_table_description", False))
                ),
                "dense_table_routing_json": _json_cell(
                    chunk.get("dense_table_routing", {})
                ),
                "metadata_json": _json_cell(chunk.get("metadata", {})),
            }
        )
    return rows


def _dense_table_rows(
    dense_tables: Any,
    file_path: str,
    document_id: str,
) -> list[StorageRow]:
    """Build dense-table storage rows."""
    rows: list[StorageRow] = []
    if not isinstance(dense_tables, list):
        return rows
    for dense_table in dense_tables:
        if not isinstance(dense_table, dict):
            continue
        routing_metadata = dense_table.get("routing_metadata", {})
        if not isinstance(routing_metadata, dict):
            routing_metadata = {}
        rows.append(
            {
                "dense_table_id": _stable_id(
                    "dense_table",
                    file_path,
                    dense_table.get("region_id", ""),
                    dense_table.get("used_range", ""),
                ),
                "document_id": document_id,
                "file_path": file_path,
                "region_id": str(dense_table.get("region_id", "")).strip(),
                "used_range": str(dense_table.get("used_range", "")).strip(),
                "sheet_name": str(
                    routing_metadata.get("sheet_name", "")
                ).strip(),
                "page_title": str(
                    routing_metadata.get("page_title", "")
                ).strip(),
                "description_generation_mode": str(
                    dense_table.get("description_generation_mode", "")
                ).strip(),
                "replacement_content": str(
                    dense_table.get("replacement_content", "")
                ),
                "routing_metadata_json": _json_cell(routing_metadata),
                "dense_table_description_json": _json_cell(
                    dense_table.get("dense_table_description", {})
                ),
                "dense_table_eda_json": _json_cell(
                    dense_table.get("dense_table_eda", {})
                ),
                "raw_content_json": _json_cell(
                    dense_table.get("raw_content", [])
                ),
            }
        )
    return rows


def _keyword_rows(
    keywords: Any,
    file_path: str,
    document_id: str,
) -> list[StorageRow]:
    """Build keyword storage rows."""
    rows: list[StorageRow] = []
    if not isinstance(keywords, list):
        return rows
    for keyword_row in keywords:
        if not isinstance(keyword_row, dict):
            continue
        keyword = str(keyword_row.get("keyword", "")).strip()
        page_number = int(keyword_row.get("page_number", 0))
        rows.append(
            {
                "keyword_id": _stable_id(
                    "keyword",
                    file_path,
                    keyword,
                    page_number,
                ),
                "document_id": document_id,
                "file_path": file_path,
                "keyword": keyword,
                "page_number": str(page_number),
                "page_title": str(keyword_row.get("page_title", "")).strip(),
                "section": str(keyword_row.get("section", "")).strip(),
                "embedding": _vector_cell(keyword_row.get("embedding", [])),
            }
        )
    return rows


def _metric_rows(
    metrics: Any,
    file_path: str,
    document_id: str,
) -> list[StorageRow]:
    """Build workbook metric storage rows."""
    rows: list[StorageRow] = []
    if not isinstance(metrics, list):
        return rows
    for metric_row in metrics:
        if not isinstance(metric_row, dict):
            continue
        metric_name = str(metric_row.get("metric_name", "")).strip()
        page_number = int(metric_row.get("page_number", 0))
        sheet_name = str(metric_row.get("sheet_name", "")).strip()
        region_id = str(metric_row.get("region_id", "")).strip()
        rows.append(
            {
                "metric_id": _stable_id(
                    "metric",
                    file_path,
                    metric_name,
                    sheet_name,
                    page_number,
                    region_id,
                ),
                "document_id": document_id,
                "file_path": file_path,
                "metric_name": metric_name,
                "page_number": str(page_number),
                "sheet_name": sheet_name,
                "region_id": region_id,
                "used_range": str(metric_row.get("used_range", "")).strip(),
                "platform": str(metric_row.get("platform", "")).strip(),
                "sub_platform": str(
                    metric_row.get("sub_platform", "")
                ).strip(),
                "periods_available_json": _json_cell(
                    metric_row.get("periods_available", [])
                ),
                "embedding": _vector_cell(metric_row.get("embedding", [])),
            }
        )
    return rows


def _sheet_summary_rows(
    sheet_summaries: Any,
    file_path: str,
    document_id: str,
) -> list[StorageRow]:
    """Build sheet-summary storage rows."""
    rows: list[StorageRow] = []
    if not isinstance(sheet_summaries, list):
        return rows
    for sheet_summary in sheet_summaries:
        if not isinstance(sheet_summary, dict):
            continue
        sheet_name = str(sheet_summary.get("sheet_name", "")).strip()
        rows.append(
            {
                "sheet_summary_id": _stable_id(
                    "sheet_summary",
                    file_path,
                    sheet_name,
                ),
                "document_id": document_id,
                "file_path": file_path,
                "sheet_name": sheet_name,
                "handling_mode": str(
                    sheet_summary.get("handling_mode", "")
                ).strip(),
                "summary": str(sheet_summary.get("summary", "")).strip(),
                "usage": str(sheet_summary.get("usage", "")).strip(),
            }
        )
    return rows


def _context_chain_rows(
    context_chains: Any,
    file_path: str,
    document_id: str,
) -> list[StorageRow]:
    """Build sheet-context-chain storage rows."""
    rows: list[StorageRow] = []
    if not isinstance(context_chains, list):
        return rows
    for chain in context_chains:
        if not isinstance(chain, dict):
            continue
        sheet_index = int(chain.get("sheet_index", 0))
        rows.append(
            {
                "chain_id": _stable_id(
                    "context_chain",
                    file_path,
                    sheet_index,
                    chain.get("sheet_name", ""),
                ),
                "document_id": document_id,
                "file_path": file_path,
                "sheet_index": str(sheet_index),
                "sheet_name": str(chain.get("sheet_name", "")).strip(),
                "context_sheet_indices_json": _json_cell(
                    chain.get("context_sheet_indices", [])
                ),
            }
        )
    return rows


def _document_table_rows(
    finalized: dict[str, Any],
    catalog_row: StorageRow,
) -> StorageRows:
    """Flatten one finalized document into storage table rows."""
    file_path = str(finalized.get("file_path", "")).strip()
    document_id = _stable_id("document", file_path)
    metadata = finalized.get("document_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    section_rows, subsection_rows = _section_rows(
        finalized.get("sections", []),
        file_path=file_path,
        document_id=document_id,
    )

    return {
        "documents": [
            _document_row(
                finalized=finalized,
                catalog_row=catalog_row,
                file_path=file_path,
                document_id=document_id,
                metadata=metadata,
            )
        ],
        "document_sections": section_rows,
        "document_subsections": subsection_rows,
        "document_chunks": _chunk_rows(
            finalized.get("chunks", []),
            file_path=file_path,
            document_id=document_id,
        ),
        "document_dense_tables": _dense_table_rows(
            finalized.get("dense_tables", []),
            file_path=file_path,
            document_id=document_id,
        ),
        "document_keywords": _keyword_rows(
            finalized.get("keyword_embeddings", []),
            file_path=file_path,
            document_id=document_id,
        ),
        "document_metrics": _metric_rows(
            finalized.get("extracted_metrics", []),
            file_path=file_path,
            document_id=document_id,
        ),
        "document_sheet_summaries": _sheet_summary_rows(
            finalized.get("sheet_summaries", []),
            file_path=file_path,
            document_id=document_id,
        ),
        "document_sheet_context_chains": _context_chain_rows(
            finalized.get("sheet_context_chains", []),
            file_path=file_path,
            document_id=document_id,
        ),
    }


def _merge_rows(
    existing_rows: list[dict[str, str]],
    replacement_rows: list[dict[str, str]],
    current_paths: set[str],
    replacement_paths: set[str],
) -> list[dict[str, str]]:
    """Merge current replacements into existing master rows."""
    kept_rows = [
        row
        for row in existing_rows
        if row.get("file_path", "") in current_paths
        and row.get("file_path", "") not in replacement_paths
    ]
    merged = kept_rows + replacement_rows
    fieldnames = list(merged[0].keys()) if merged else []
    if not fieldnames:
        return merged
    return sorted(
        merged,
        key=lambda row: tuple(row.get(name, "") for name in fieldnames),
    )


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    """Write CSV rows atomically with a stable header order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _cell_value(row.get(field, ""))
                    for field in fieldnames
                }
            )
    temp_path.replace(path)


def _write_storage_outputs(
    table_rows: StorageRows,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    """Write run-local and master CSV files. Returns: dict[str, Path]."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    master_dir = _storage_master_dir()
    csv_paths: dict[str, Path] = {}
    for table_name in STORAGE_TABLE_ORDER:
        rows = table_rows[table_name]
        fieldnames = STORAGE_TABLE_COLUMNS[table_name]
        processing_path = STORAGE_DIR / f"{table_name}.csv"
        master_path = master_dir / f"{table_name}.csv"
        _write_csv(processing_path, fieldnames, rows)
        _write_csv(master_path, fieldnames, rows)
        csv_paths[table_name] = master_path
    manifest_path = STORAGE_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return csv_paths


def _current_catalog_snapshot(
    existing_master_rows: StorageRows,
) -> tuple[list[StorageRow], dict[str, StorageRow], set[str]]:
    """Build the current catalog rows and lookups from the live source tree."""
    previous_catalog_rows = {
        row["file_path"]: row
        for row in existing_master_rows["document_catalog"]
        if row.get("file_path")
    }
    current_records = _scan_current_filesystem(get_data_source_path())
    current_catalog_rows = [
        _catalog_row(record, previous_catalog_rows.get(record.file_path, {}))
        for record in current_records
    ]
    current_catalog_lookup = {
        row["file_path"]: row for row in current_catalog_rows
    }
    current_supported_paths = {
        record.file_path for record in current_records if record.supported
    }
    return (
        current_catalog_rows,
        current_catalog_lookup,
        current_supported_paths,
    )


def _resolve_replacement_finalizations(
    existing_master_rows: StorageRows,
    current_supported_paths: set[str],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Resolve finalized documents from this run plus archived fallbacks."""
    current_finalizations = _load_finalization_results()
    existing_document_paths = {
        row["file_path"]
        for row in existing_master_rows["documents"]
        if row.get("file_path")
    }
    archived_needed = (
        current_supported_paths
        - set(current_finalizations)
        - existing_document_paths
    )
    archived_finalizations = _load_archived_finalizations(set(archived_needed))
    replacement_finalizations = dict(archived_finalizations)
    replacement_finalizations.update(current_finalizations)
    missing_paths = (
        current_supported_paths
        - set(replacement_finalizations)
        - existing_document_paths
    )
    if missing_paths:
        missing_list = ", ".join(sorted(missing_paths))
        raise RuntimeError(
            "Storage cannot build a complete current snapshot for: "
            f"{missing_list}"
        )
    return (
        current_finalizations,
        archived_finalizations,
        replacement_finalizations,
    )


def _replacement_rows(
    replacement_finalizations: dict[str, dict[str, Any]],
    current_catalog_lookup: dict[str, StorageRow],
) -> StorageRows:
    """Build storage rows for every finalized document being replaced."""
    rows: StorageRows = {
        table_name: []
        for table_name in STORAGE_TABLE_ORDER
        if table_name != "document_catalog"
    }
    for file_path in sorted(replacement_finalizations):
        catalog_row = current_catalog_lookup.get(file_path)
        if catalog_row is None:
            continue
        document_rows = _document_table_rows(
            replacement_finalizations[file_path],
            catalog_row,
        )
        for table_name, table_rows in document_rows.items():
            rows[table_name].extend(table_rows)
    return rows


def _merge_storage_rows(
    existing_master_rows: StorageRows,
    current_catalog_rows: list[StorageRow],
    current_supported_paths: set[str],
    replacement_finalizations: dict[str, dict[str, Any]],
    replacement_rows: StorageRows,
) -> StorageRows:
    """Merge current replacements into the persistent master rowsets."""
    table_rows: StorageRows = {"document_catalog": current_catalog_rows}
    replacement_paths = set(replacement_finalizations)
    for table_name in STORAGE_TABLE_ORDER:
        if table_name == "document_catalog":
            continue
        table_rows[table_name] = _merge_rows(
            existing_rows=existing_master_rows[table_name],
            replacement_rows=replacement_rows[table_name],
            current_paths=current_supported_paths,
            replacement_paths=replacement_paths,
        )
    return table_rows


def _storage_manifest(
    master_dir: Path,
    table_rows: StorageRows,
    current_catalog_rows: list[StorageRow],
    current_supported_paths: set[str],
    current_finalizations: dict[str, dict[str, Any]],
    archived_finalizations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the storage manifest payload."""
    return {
        "current_catalog_file_count": len(current_catalog_rows),
        "current_supported_file_count": len(current_supported_paths),
        "current_finalization_file_count": len(current_finalizations),
        "archive_bootstrap_file_count": len(archived_finalizations),
        "postgres_sync_enabled": get_storage_push_to_postgres(),
        "master_dir": str(master_dir),
        "row_counts": {
            table_name: len(rows) for table_name, rows in table_rows.items()
        },
    }


def run_storage(conn) -> None:
    """Build storage CSV masters and optionally sync them to PostgreSQL.

    Params:
        conn: psycopg2 database connection

    Returns:
        None

    Example:
        >>> run_storage(conn)
    """
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Starting storage materialization")

    master_dir = _storage_master_dir()
    existing_master_rows = _load_existing_master_rows(master_dir)
    (
        current_catalog_rows,
        current_catalog_lookup,
        current_supported_paths,
    ) = _current_catalog_snapshot(existing_master_rows)
    (
        current_finalizations,
        archived_finalizations,
        replacement_finalizations,
    ) = _resolve_replacement_finalizations(
        existing_master_rows,
        current_supported_paths,
    )
    replacement_rows = _replacement_rows(
        replacement_finalizations,
        current_catalog_lookup,
    )
    table_rows = _merge_storage_rows(
        existing_master_rows=existing_master_rows,
        current_catalog_rows=current_catalog_rows,
        current_supported_paths=current_supported_paths,
        replacement_finalizations=replacement_finalizations,
        replacement_rows=replacement_rows,
    )
    manifest = _storage_manifest(
        master_dir=master_dir,
        table_rows=table_rows,
        current_catalog_rows=current_catalog_rows,
        current_supported_paths=current_supported_paths,
        current_finalizations=current_finalizations,
        archived_finalizations=archived_finalizations,
    )
    csv_paths = _write_storage_outputs(table_rows, manifest)

    if get_storage_push_to_postgres():
        refresh_storage_tables(conn, csv_paths)
        logger.info(
            "Storage complete — %d tables written and PostgreSQL synced",
            len(STORAGE_TABLE_ORDER),
        )
        return

    logger.info(
        "Storage complete — %d tables written, PostgreSQL sync skipped",
        len(STORAGE_TABLE_ORDER),
    )
