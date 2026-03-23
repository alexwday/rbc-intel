"""Tests for extraction stage orchestrator."""

import json
from unittest.mock import MagicMock, patch

import pytest

from helpers import make_extraction_result, make_file_record
from ingestion.stages.extraction import (
    _load_discovery,
    _process_file,
    _write_result,
    run_extraction,
)

_make_extraction_result = make_extraction_result


# ── _load_discovery ──────────────────────────────────────────────


def test_load_discovery_combines_new_and_modified(tmp_path, monkeypatch):
    """Loads both new and modified files from discovery.json."""
    discovery = {
        "new": [
            {
                "data_source": "src",
                "filter_1": "",
                "filter_2": "",
                "filter_3": "",
                "filename": "new.pdf",
                "filetype": "pdf",
                "file_size": 1024,
                "date_last_modified": 1700000000.0,
                "file_hash": "",
                "file_path": "/data/src/new.pdf",
                "supported": True,
            }
        ],
        "modified": [
            {
                "data_source": "src",
                "filter_1": "",
                "filter_2": "",
                "filter_3": "",
                "filename": "mod.pdf",
                "filetype": "pdf",
                "file_size": 2048,
                "date_last_modified": 1700000000.0,
                "file_hash": "",
                "file_path": "/data/src/mod.pdf",
                "supported": True,
            }
        ],
        "deleted": [],
    }
    proc_dir = tmp_path / "processing"
    proc_dir.mkdir()
    (proc_dir / "discovery.json").write_text(json.dumps(discovery))

    monkeypatch.setattr("ingestion.stages.extraction.PROCESSING_DIR", proc_dir)
    records = _load_discovery()
    assert len(records) == 2
    filenames = {r.filename for r in records}
    assert filenames == {"new.pdf", "mod.pdf"}


def test_load_discovery_filters_unsupported(tmp_path, monkeypatch):
    """Unsupported filetypes are excluded."""
    discovery = {
        "new": [
            {
                "data_source": "src",
                "filter_1": "",
                "filter_2": "",
                "filter_3": "",
                "filename": "readme.txt",
                "filetype": "txt",
                "file_size": 100,
                "date_last_modified": 1700000000.0,
                "file_hash": "",
                "file_path": "/data/src/readme.txt",
                "supported": True,
            }
        ],
        "modified": [],
        "deleted": [],
    }
    proc_dir = tmp_path / "processing"
    proc_dir.mkdir()
    (proc_dir / "discovery.json").write_text(json.dumps(discovery))

    monkeypatch.setattr("ingestion.stages.extraction.PROCESSING_DIR", proc_dir)
    records = _load_discovery()
    assert len(records) == 0


def test_load_discovery_empty(tmp_path, monkeypatch):
    """Empty discovery returns empty list."""
    discovery = {"new": [], "modified": [], "deleted": []}
    proc_dir = tmp_path / "processing"
    proc_dir.mkdir()
    (proc_dir / "discovery.json").write_text(json.dumps(discovery))

    monkeypatch.setattr("ingestion.stages.extraction.PROCESSING_DIR", proc_dir)
    records = _load_discovery()
    assert not records


def test_load_discovery_skips_malformed_records(tmp_path, monkeypatch):
    """Malformed discovery records are skipped instead of crashing."""
    discovery = {
        "new": [
            {
                "data_source": "src",
                "filter_1": "",
                "filter_2": "",
                "filter_3": "",
                "filename": "broken.pdf",
                "filetype": "pdf",
                "file_size": 100,
                "date_last_modified": 1700000000.0,
                "file_hash": "",
            },
            {
                "data_source": "src",
                "filter_1": "",
                "filter_2": "",
                "filter_3": "",
                "filename": "ok.pdf",
                "filetype": "pdf",
                "file_size": 100,
                "date_last_modified": 1700000000.0,
                "file_hash": "",
                "file_path": "/data/src/ok.pdf",
            },
        ],
        "modified": [],
        "deleted": [],
    }
    proc_dir = tmp_path / "processing"
    proc_dir.mkdir()
    (proc_dir / "discovery.json").write_text(json.dumps(discovery))

    monkeypatch.setattr("ingestion.stages.extraction.PROCESSING_DIR", proc_dir)
    records = _load_discovery()
    assert len(records) == 1
    assert records[0].filename == "ok.pdf"


def test_load_discovery_skips_non_list_category(tmp_path, monkeypatch):
    """Non-list discovery categories are ignored."""
    discovery = {"new": {}, "modified": [], "deleted": []}
    proc_dir = tmp_path / "processing"
    proc_dir.mkdir()
    (proc_dir / "discovery.json").write_text(json.dumps(discovery))

    monkeypatch.setattr("ingestion.stages.extraction.PROCESSING_DIR", proc_dir)
    records = _load_discovery()
    assert not records


def test_load_discovery_skips_non_object_items(tmp_path, monkeypatch):
    """Non-object discovery records are skipped."""
    discovery = {
        "new": ["bad-record"],
        "modified": [],
        "deleted": [],
    }
    proc_dir = tmp_path / "processing"
    proc_dir.mkdir()
    (proc_dir / "discovery.json").write_text(json.dumps(discovery))

    monkeypatch.setattr("ingestion.stages.extraction.PROCESSING_DIR", proc_dir)
    records = _load_discovery()
    assert not records


# ── _process_file ────────────────────────────────────────────────


@patch("ingestion.stages.extraction.process_pdf")
def test_process_file_routes_pdf(mock_process):
    """PDF files are routed to process_pdf."""
    mock_process.return_value = _make_extraction_result()
    record = make_file_record(filetype="pdf")
    mock_llm = MagicMock()

    result = _process_file(record, mock_llm)
    mock_process.assert_called_once_with(record.file_path, mock_llm)
    assert result.filetype == "pdf"


@patch("ingestion.stages.extraction.process_docx")
def test_process_file_routes_docx(mock_process):
    """DOCX files are routed to process_docx."""
    mock_process.return_value = _make_extraction_result(filetype="docx")
    record = make_file_record(filetype="docx")
    mock_llm = MagicMock()

    result = _process_file(record, mock_llm)
    mock_process.assert_called_once_with(record.file_path, mock_llm)
    assert result.filetype == "docx"


@patch("ingestion.stages.extraction.process_pptx")
def test_process_file_routes_pptx(mock_process):
    """PPTX files are routed to process_pptx."""
    mock_process.return_value = _make_extraction_result(filetype="pptx")
    record = make_file_record(filetype="pptx")
    mock_llm = MagicMock()

    result = _process_file(record, mock_llm)
    mock_process.assert_called_once_with(record.file_path, mock_llm)
    assert result.filetype == "pptx"


@patch("ingestion.stages.extraction.process_xlsx")
def test_process_file_routes_xlsx(mock_process):
    """XLSX files are routed to process_xlsx."""
    mock_process.return_value = _make_extraction_result(filetype="xlsx")
    record = make_file_record(filetype="xlsx", filename="book.xlsx")
    mock_llm = MagicMock()

    result = _process_file(record, mock_llm)
    mock_process.assert_called_once_with(record.file_path, mock_llm)
    assert result.filetype == "xlsx"


def test_process_file_unsupported_type():
    """Unsupported filetype raises ValueError."""
    record = make_file_record(filetype="xyz")
    mock_llm = MagicMock()

    with pytest.raises(ValueError, match="Unsupported filetype: xyz"):
        _process_file(record, mock_llm)


# ── _write_result ────────────────────────────────────────────────


def test_write_result_creates_json(tmp_path, monkeypatch):
    """Writes extraction result to JSON file."""
    extraction_dir = tmp_path / "processing" / "extraction"
    monkeypatch.setattr(
        "ingestion.stages.extraction.EXTRACTION_DIR", extraction_dir
    )

    result = _make_extraction_result(file_path="/data/src/report.pdf")
    _write_result(result)

    output_files = list(extraction_dir.glob("*.json"))
    assert len(output_files) == 1
    assert output_files[0].name.startswith("report_")

    data = json.loads(output_files[0].read_text())
    assert data["file_path"] == "/data/src/report.pdf"
    assert data["total_pages"] == 1


def test_write_result_uses_path_hash(tmp_path, monkeypatch):
    """Different paths produce different filenames."""
    extraction_dir = tmp_path / "processing" / "extraction"
    monkeypatch.setattr(
        "ingestion.stages.extraction.EXTRACTION_DIR", extraction_dir
    )

    _write_result(_make_extraction_result(file_path="/dir_a/report.pdf"))
    _write_result(_make_extraction_result(file_path="/dir_b/report.pdf"))

    output_files = list(extraction_dir.glob("*.json"))
    assert len(output_files) == 2
    names = {f.name for f in output_files}
    assert len(names) == 2


# ── run_extraction ───────────────────────────────────────────────


@patch("ingestion.stages.extraction._write_result")
@patch("ingestion.stages.extraction._process_file")
@patch("ingestion.stages.extraction._load_discovery")
def test_run_extraction_processes_all_files(
    mock_load, mock_process, mock_write
):
    """Processes and writes results for all discovered files."""
    records = [
        make_file_record(filename="a.pdf", file_path="/data/a.pdf"),
        make_file_record(filename="b.pdf", file_path="/data/b.pdf"),
    ]
    mock_load.return_value = records
    mock_process.return_value = _make_extraction_result()
    mock_llm = MagicMock()

    run_extraction(mock_llm)
    assert mock_process.call_count == 2
    assert mock_write.call_count == 2


@patch("ingestion.stages.extraction.get_stage_logger")
@patch("ingestion.stages.extraction._write_result")
@patch("ingestion.stages.extraction._process_file")
@patch("ingestion.stages.extraction._load_discovery")
def test_run_extraction_counts_unusable_files_separately(
    mock_load, mock_process, mock_write, mock_get_logger
):
    """Files with zero usable pages are not counted as successes."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_load.return_value = [
        make_file_record(filename="empty.pdf", file_path="/data/empty.pdf")
    ]
    mock_process.return_value = _make_extraction_result(
        file_path="/data/empty.pdf",
        pages=[],
        total_pages=0,
        pages_succeeded=0,
        pages_failed=0,
    )
    mock_llm = MagicMock()

    run_extraction(mock_llm)

    assert mock_write.call_count == 1
    mock_logger.debug.assert_any_call(
        "Extracted %s but produced no usable pages",
        "empty.pdf",
    )
    mock_logger.info.assert_any_call(
        "Extraction complete — files: %d succeeded, %d unusable, %d failed",
        0,
        1,
        0,
    )


@patch("ingestion.stages.extraction._write_result")
@patch("ingestion.stages.extraction._process_file")
@patch("ingestion.stages.extraction._load_discovery")
def test_run_extraction_continues_on_failure(
    mock_load, mock_process, mock_write
):
    """Per-file failure does not crash the stage."""
    records = [
        make_file_record(filename="a.pdf", file_path="/data/a.pdf"),
        make_file_record(filename="b.pdf", file_path="/data/b.pdf"),
    ]
    mock_load.return_value = records
    mock_process.side_effect = [
        RuntimeError("fail"),
        _make_extraction_result(),
    ]
    mock_llm = MagicMock()

    run_extraction(mock_llm)
    assert mock_write.call_count == 1


@patch("ingestion.stages.extraction._load_discovery")
def test_run_extraction_empty_discovery(mock_load):
    """No files to process exits early."""
    mock_load.return_value = []
    mock_llm = MagicMock()

    run_extraction(mock_llm)
