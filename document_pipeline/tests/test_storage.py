"""Tests for stages.storage."""

import csv
import json
import os
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from ingestion.stages.storage import (
    _read_csv_rows,
    _document_table_rows,
    _write_csv,
    run_storage,
)
from ingestion.utils.postgres import STORAGE_TABLE_COLUMNS, STORAGE_TABLE_ORDER


def _sample_finalization(file_path: str, filetype: str = "pdf") -> dict:
    """Build a minimal finalized document payload for storage tests."""
    return {
        "file_path": file_path,
        "file_name": file_path.rsplit("/", 1)[-1],
        "filetype": filetype,
        "document_summary": f"Summary for {file_path}",
        "document_description": f"Description for {file_path}",
        "document_usage": f"Usage for {file_path}",
        "document_metadata": {
            "title": f"Title for {file_path}",
            "publication_date": "2026-03-24",
            "authors": ["Northbridge"],
            "document_type": "Risk pack",
            "abstract": "Abstract",
        },
        "structure_type": "sections",
        "structure_confidence": "high",
        "degradation_signals": [],
        "summary_embedding": [0.1, 0.2],
        "page_count": 1,
        "primary_section_count": 1,
        "subsection_count": 1,
        "chunk_count": 1,
        "dense_table_count": 0,
        "extraction_metadata": {"pages_enriched": 1},
        "sections": [
            {
                "section_number": 1,
                "title": "Overview",
                "page_start": 1,
                "page_end": 1,
                "page_count": 1,
                "summary": {
                    "overview": "Overview",
                    "key_topics": ["Risk"],
                    "key_metrics": {"CET1": "13.4%"},
                    "key_findings": ["Buffers remain above triggers."],
                    "notable_facts": ["One-page sample"],
                    "is_fallback": False,
                },
                "subsections": [
                    {
                        "subsection_number": 1,
                        "title": "Overview detail",
                        "page_start": 1,
                        "page_end": 1,
                        "page_count": 1,
                        "summary": {
                            "overview": "Subsection overview",
                            "key_topics": ["Detail"],
                            "key_metrics": {},
                            "key_findings": [],
                            "notable_facts": [],
                            "is_fallback": False,
                        },
                    }
                ],
            }
        ],
        "chunks": [
            {
                "chunk_number": 0,
                "page_number": 1,
                "content": "Chunk content",
                "primary_section_number": 1,
                "primary_section_name": "Overview",
                "subsection_number": 1,
                "subsection_name": "Overview detail",
                "hierarchy_path": "Overview > Overview detail",
                "primary_section_page_count": 1,
                "subsection_page_count": 1,
                "embedding_prefix": "Overview: ",
                "embedding": [0.3, 0.4],
                "is_dense_table_description": False,
                "dense_table_routing": {},
                "metadata": {"page_title": "Overview"},
            }
        ],
        "dense_tables": [],
        "sheet_summaries": [],
        "keyword_embeddings": [
            {
                "keyword": "capital",
                "page_number": 1,
                "page_title": "Overview",
                "section": "Overview",
                "embedding": [0.5, 0.6],
            }
        ],
        "extracted_metrics": [],
        "sheet_context_chains": [],
    }


def _write_finalization_json(directory, payload: dict) -> None:
    """Write one finalization JSON file. Params: directory, payload."""
    directory.mkdir(parents=True, exist_ok=True)
    file_name = payload["file_name"].rsplit(".", 1)[0] + "_sample.json"
    (directory / file_name).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _catalog_row(
    file_path: str, filename: str, filetype: str
) -> dict[str, str]:
    """Build one catalog row for a storage test file."""
    stat = path_stat(file_path)
    return {
        "data_source": "source_a",
        "filter_1": "2026",
        "filter_2": "",
        "filter_3": "",
        "filename": filename,
        "filetype": filetype,
        "file_size": str(stat.st_size),
        "date_last_modified": str(stat.st_mtime),
        "file_hash": f"hash-{filename}",
        "file_path": file_path,
    }


def path_stat(file_path: str):
    """Return the file stat object.

    Params: file_path. Returns: os.stat_result.
    """
    return os.stat(file_path)


def _write_master_snapshot(
    master_dir, finalized: dict, catalog_row: dict
) -> None:
    """Write one prior-master snapshot for a single finalized document."""
    table_rows = {"document_catalog": [catalog_row]}
    table_rows.update(_document_table_rows(finalized, catalog_row))
    for table_name in STORAGE_TABLE_ORDER:
        rows = table_rows.get(table_name, [])
        _write_csv(
            master_dir / f"{table_name}.csv",
            STORAGE_TABLE_COLUMNS[table_name],
            rows,
        )


def test_run_storage_builds_incremental_master_csvs(tmp_path, monkeypatch):
    """Current finalizations replace changed files and keep unchanged rows."""
    base_dir = tmp_path / "sources"
    source_dir = base_dir / "source_a" / "2026"
    source_dir.mkdir(parents=True)
    current_file = source_dir / "current.pdf"
    current_file.write_text("current", encoding="utf-8")
    unchanged_file = source_dir / "unchanged.pdf"
    unchanged_file.write_text("unchanged", encoding="utf-8")
    stale_file = source_dir / "stale.pdf"
    stale_file.write_text("stale", encoding="utf-8")

    processing_dir = tmp_path / "processing"
    finalization_dir = processing_dir / "finalization"
    storage_dir = processing_dir / "storage"
    master_dir = tmp_path / "masters"

    current_payload = _sample_finalization(str(current_file))
    _write_finalization_json(finalization_dir, current_payload)

    unchanged_payload = _sample_finalization(str(unchanged_file))
    _write_master_snapshot(
        master_dir,
        unchanged_payload,
        _catalog_row(str(unchanged_file), "unchanged.pdf", "pdf"),
    )
    stale_payload = _sample_finalization(str(stale_file))
    stale_rows = {
        "document_catalog": [_catalog_row(str(stale_file), "stale.pdf", "pdf")]
    }
    stale_rows.update(
        _document_table_rows(
            stale_payload,
            _catalog_row(str(stale_file), "stale.pdf", "pdf"),
        )
    )
    for table_name in STORAGE_TABLE_ORDER:
        rows = _read_table_rows(master_dir, table_name)
        rows.extend(stale_rows.get(table_name, []))
        _write_csv(
            master_dir / f"{table_name}.csv",
            STORAGE_TABLE_COLUMNS[table_name],
            rows,
        )
    stale_file.unlink()

    monkeypatch.setenv("DATA_SOURCE_PATH", str(base_dir))
    monkeypatch.setenv("STORAGE_MASTER_DIR", str(master_dir))
    monkeypatch.setenv("STORAGE_PUSH_TO_POSTGRES", "false")
    monkeypatch.setattr(
        "ingestion.stages.storage.FINALIZATION_DIR", finalization_dir
    )
    monkeypatch.setattr("ingestion.stages.storage.STORAGE_DIR", storage_dir)
    monkeypatch.setattr(
        "ingestion.stages.storage.ARCHIVE_DIR", tmp_path / "archive"
    )

    run_storage(MagicMock())

    documents_csv = (master_dir / "documents.csv").read_text(encoding="utf-8")
    assert str(current_file) in documents_csv
    assert str(unchanged_file) in documents_csv
    assert str(stale_file) not in documents_csv

    catalog_csv = (master_dir / "document_catalog.csv").read_text(
        encoding="utf-8"
    )
    assert str(current_file) in catalog_csv
    assert str(unchanged_file) in catalog_csv
    assert str(stale_file) not in catalog_csv

    manifest = json.loads((storage_dir / "manifest.json").read_text())
    assert manifest["current_finalization_file_count"] == 1
    assert manifest["archive_bootstrap_file_count"] == 0


def test_run_storage_bootstraps_from_archived_finalizations(
    tmp_path, monkeypatch
):
    """Archived finalization outputs can seed masters on a first run."""
    base_dir = tmp_path / "sources"
    source_dir = base_dir / "source_a" / "2026"
    source_dir.mkdir(parents=True)
    archived_file = source_dir / "archived.pdf"
    archived_file.write_text("archived", encoding="utf-8")

    processing_dir = tmp_path / "processing"
    storage_dir = processing_dir / "storage"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    archive_path = archive_dir / "run_20260324_140000.zip"
    payload = _sample_finalization(str(archived_file))
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "finalization/archived_sample.json",
            json.dumps(payload, indent=2),
        )

    master_dir = tmp_path / "masters"
    monkeypatch.setenv("DATA_SOURCE_PATH", str(base_dir))
    monkeypatch.setenv("STORAGE_MASTER_DIR", str(master_dir))
    monkeypatch.setenv("STORAGE_PUSH_TO_POSTGRES", "false")
    monkeypatch.setattr(
        "ingestion.stages.storage.FINALIZATION_DIR",
        processing_dir / "finalization",
    )
    monkeypatch.setattr("ingestion.stages.storage.STORAGE_DIR", storage_dir)
    monkeypatch.setattr("ingestion.stages.storage.ARCHIVE_DIR", archive_dir)

    run_storage(MagicMock())

    documents_csv = (master_dir / "documents.csv").read_text(encoding="utf-8")
    assert str(archived_file) in documents_csv
    manifest = json.loads((storage_dir / "manifest.json").read_text())
    assert manifest["archive_bootstrap_file_count"] == 1


def test_run_storage_raises_when_snapshot_cannot_be_completed(
    tmp_path, monkeypatch
):
    """Storage fails when there is no way to materialize a supported file."""
    base_dir = tmp_path / "sources"
    source_dir = base_dir / "source_a" / "2026"
    source_dir.mkdir(parents=True)
    missing_file = source_dir / "missing.pdf"
    missing_file.write_text("missing", encoding="utf-8")

    processing_dir = tmp_path / "processing"
    monkeypatch.setenv("DATA_SOURCE_PATH", str(base_dir))
    monkeypatch.setenv("STORAGE_MASTER_DIR", str(tmp_path / "masters"))
    monkeypatch.setenv("STORAGE_PUSH_TO_POSTGRES", "false")
    monkeypatch.setattr(
        "ingestion.stages.storage.FINALIZATION_DIR",
        processing_dir / "finalization",
    )
    monkeypatch.setattr(
        "ingestion.stages.storage.STORAGE_DIR",
        processing_dir / "storage",
    )
    monkeypatch.setattr(
        "ingestion.stages.storage.ARCHIVE_DIR", tmp_path / "archive"
    )

    with pytest.raises(RuntimeError, match=str(missing_file)):
        run_storage(MagicMock())


def test_run_storage_syncs_postgres_when_enabled(tmp_path, monkeypatch):
    """Storage calls the Postgres refresh helper when sync is enabled."""
    base_dir = tmp_path / "sources"
    source_dir = base_dir / "source_a" / "2026"
    source_dir.mkdir(parents=True)
    current_file = source_dir / "current.pdf"
    current_file.write_text("current", encoding="utf-8")

    processing_dir = tmp_path / "processing"
    finalization_dir = processing_dir / "finalization"
    storage_dir = processing_dir / "storage"
    _write_finalization_json(
        finalization_dir, _sample_finalization(str(current_file))
    )

    monkeypatch.setenv("DATA_SOURCE_PATH", str(base_dir))
    monkeypatch.setenv("STORAGE_MASTER_DIR", str(tmp_path / "masters"))
    monkeypatch.setenv("STORAGE_PUSH_TO_POSTGRES", "true")
    monkeypatch.setattr(
        "ingestion.stages.storage.FINALIZATION_DIR", finalization_dir
    )
    monkeypatch.setattr("ingestion.stages.storage.STORAGE_DIR", storage_dir)
    monkeypatch.setattr(
        "ingestion.stages.storage.ARCHIVE_DIR", tmp_path / "archive"
    )

    conn = MagicMock()
    with patch(
        "ingestion.stages.storage.refresh_storage_tables"
    ) as mock_refresh:
        run_storage(conn)

    mock_refresh.assert_called_once()
    assert mock_refresh.call_args.args[0] is conn
    csv_paths = mock_refresh.call_args.args[1]
    assert set(csv_paths) == set(STORAGE_TABLE_ORDER)


def test_read_csv_rows_handles_large_fields(tmp_path):
    """Storage CSV readers support rows larger than the stdlib default."""
    csv_path = tmp_path / "large.csv"
    large_value = "x" * 200000
    _write_csv(csv_path, ["payload"], [{"payload": large_value}])

    rows = _read_csv_rows(csv_path)

    assert rows == [{"payload": large_value}]


def test_write_csv_strips_null_bytes(tmp_path):
    """Storage CSV output removes NUL bytes that Postgres rejects."""
    csv_path = tmp_path / "nulls.csv"
    _write_csv(csv_path, ["payload"], [{"payload": "abc\x00def"}])

    rows = _read_csv_rows(csv_path)

    assert rows == [{"payload": "abcdef"}]


def _read_table_rows(master_dir, table_name: str) -> list[dict[str, str]]:
    """Read an existing test master CSV. Params: master_dir, table_name."""
    csv_path = master_dir / f"{table_name}.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]
