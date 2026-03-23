"""Tests for ingestion.stages.discovery."""

import json
import logging
import os
from unittest.mock import MagicMock

from helpers import make_file_record
from ingestion.stages.discovery import (
    compute_diff,
    run_discovery,
    scan_filesystem,
)
from ingestion.utils.file_types import compute_file_hash

# --- scan_filesystem ---


def test_scan_empty_directory(tmp_path):
    """Empty base path returns no records."""
    records = scan_filesystem(str(tmp_path))
    assert not records


def test_scan_nonexistent_path(tmp_path):
    """Missing base path returns empty list and logs error."""
    missing = str(tmp_path / "no_such_dir")
    records = scan_filesystem(missing)
    assert not records


def test_scan_data_source_only(tmp_path):
    """Single file at data_source level."""
    ds = tmp_path / "policy"
    ds.mkdir()
    (ds / "doc.pdf").write_bytes(b"pdf")
    records = scan_filesystem(str(tmp_path))
    assert len(records) == 1
    assert records[0].data_source == "policy"
    assert records[0].filter_1 == ""
    assert records[0].filename == "doc.pdf"
    assert records[0].filetype == "pdf"


def test_scan_one_filter(tmp_path):
    """File at data_source/filter_1 level."""
    path = tmp_path / "policy" / "2026"
    path.mkdir(parents=True)
    (path / "report.docx").write_bytes(b"docx")
    records = scan_filesystem(str(tmp_path))
    assert len(records) == 1
    assert records[0].data_source == "policy"
    assert records[0].filter_1 == "2026"
    assert records[0].filter_2 == ""


def test_scan_two_filters(tmp_path):
    """File at data_source/filter_1/filter_2 level."""
    path = tmp_path / "fin" / "2026" / "Q1"
    path.mkdir(parents=True)
    (path / "data.xlsx").write_bytes(b"xlsx")
    records = scan_filesystem(str(tmp_path))
    assert len(records) == 1
    assert records[0].filter_1 == "2026"
    assert records[0].filter_2 == "Q1"
    assert records[0].filter_3 == ""


def test_scan_three_filters(tmp_path):
    """File at data_source/f1/f2/f3 level."""
    path = tmp_path / "fin" / "2026" / "Q1" / "RBC"
    path.mkdir(parents=True)
    (path / "supp.csv").write_bytes(b"csv")
    records = scan_filesystem(str(tmp_path))
    assert len(records) == 1
    assert records[0].filter_3 == "RBC"


def test_scan_deep_nesting_flattens(tmp_path):
    """Levels beyond filter_3 are joined into filter_3."""
    path = tmp_path / "src" / "a" / "b" / "c" / "d"
    path.mkdir(parents=True)
    (path / "deep.md").write_bytes(b"md")
    records = scan_filesystem(str(tmp_path))
    assert len(records) == 1
    assert records[0].filter_1 == "a"
    assert records[0].filter_2 == "b"
    assert records[0].filter_3 == os.path.join("c", "d")


def test_scan_skips_hidden_files(tmp_path):
    """Files starting with a dot are ignored."""
    ds = tmp_path / "policy"
    ds.mkdir()
    (ds / ".DS_Store").write_bytes(b"hidden")
    (ds / "visible.pdf").write_bytes(b"pdf")
    records = scan_filesystem(str(tmp_path))
    assert len(records) == 1
    assert records[0].filename == "visible.pdf"


def test_scan_skips_hidden_dirs(tmp_path):
    """Directories starting with a dot are skipped."""
    hidden = tmp_path / "policy" / ".cache"
    hidden.mkdir(parents=True)
    (hidden / "cached.pdf").write_bytes(b"pdf")
    visible = tmp_path / "policy"
    (visible / "real.pdf").write_bytes(b"pdf")
    records = scan_filesystem(str(tmp_path))
    assert len(records) == 1
    assert records[0].filename == "real.pdf"


def test_scan_file_without_extension(tmp_path):
    """File with no dot gets empty extension."""
    ds = tmp_path / "src"
    ds.mkdir()
    (ds / "Makefile").write_bytes(b"make")
    records = scan_filesystem(str(tmp_path))
    assert len(records) == 1
    assert records[0].filetype == ""
    assert records[0].supported is False


def test_scan_captures_file_size(tmp_path):
    """File size matches written bytes."""
    ds = tmp_path / "src"
    ds.mkdir()
    content = b"x" * 2048
    (ds / "sized.pdf").write_bytes(content)
    records = scan_filesystem(str(tmp_path))
    assert records[0].file_size == 2048


def test_scan_captures_mtime(tmp_path):
    """Modification time is populated."""
    ds = tmp_path / "src"
    ds.mkdir()
    (ds / "timed.pdf").write_bytes(b"pdf")
    records = scan_filesystem(str(tmp_path))
    assert records[0].date_last_modified > 0


def test_scan_multiple_data_sources(tmp_path):
    """Multiple data source folders are scanned."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "a.pdf").write_bytes(b"a")
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "b.pdf").write_bytes(b"b")
    records = scan_filesystem(str(tmp_path))
    sources = {r.data_source for r in records}
    assert sources == {"alpha", "beta"}


def test_scan_files_at_base_root_ignored(tmp_path):
    """Files directly in base_path (no data_source) are ignored."""
    (tmp_path / "root.pdf").write_bytes(b"pdf")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "in_ds.pdf").write_bytes(b"pdf")
    records = scan_filesystem(str(tmp_path))
    assert len(records) == 1
    assert records[0].data_source == "src"


# --- compute_diff ---


def test_diff_all_new():
    """All files are new when catalog is empty."""
    disk = [
        make_file_record(file_path="/a"),
        make_file_record(file_path="/b"),
    ]
    diff = compute_diff(disk, [])
    assert len(diff.new) == 2
    assert not diff.modified
    assert not diff.deleted


def test_diff_all_deleted():
    """All files deleted when disk is empty."""
    catalog = [
        make_file_record(file_path="/a"),
        make_file_record(file_path="/b"),
    ]
    diff = compute_diff([], catalog)
    assert not diff.new
    assert not diff.modified
    assert len(diff.deleted) == 2


def test_diff_unchanged_same_size_and_date():
    """Matching size and date means unchanged."""
    rec = make_file_record(file_size=100, date_last_modified=1000.0)
    diff = compute_diff([rec], [rec])
    assert not diff.new
    assert not diff.modified
    assert not diff.deleted


def test_diff_modified_size_changed():
    """Different file size means modified."""
    disk = make_file_record(file_path="/a", file_size=200)
    cat = make_file_record(file_path="/a", file_size=100)
    diff = compute_diff([disk], [cat])
    assert len(diff.modified) == 1
    assert diff.modified[0].file_path == "/a"


def test_diff_modified_date_changed_hash_differs(tmp_path):
    """Different date + different hash means modified."""
    f = tmp_path / "file.pdf"
    f.write_bytes(b"new content")
    disk = make_file_record(
        file_path=str(f),
        file_size=100,
        date_last_modified=2000.0,
    )
    cat = make_file_record(
        file_path=str(f),
        file_size=100,
        date_last_modified=1000.0,
        file_hash="oldhash",
    )
    diff = compute_diff([disk], [cat])
    assert len(diff.modified) == 1


def test_diff_unchanged_date_changed_hash_same(tmp_path):
    """Different date but same hash means unchanged."""
    f = tmp_path / "file.pdf"
    f.write_bytes(b"same")
    real_hash = compute_file_hash(str(f))
    disk = make_file_record(
        file_path=str(f),
        file_size=100,
        date_last_modified=2000.0,
    )
    cat = make_file_record(
        file_path=str(f),
        file_size=100,
        date_last_modified=1000.0,
        file_hash=real_hash,
    )
    diff = compute_diff([disk], [cat])
    assert not diff.modified


def test_diff_mixed():
    """Mix of new, modified, deleted, unchanged."""
    new_rec = make_file_record(file_path="/new")
    unchanged = make_file_record(
        file_path="/same",
        file_size=100,
        date_last_modified=1000.0,
    )
    mod_disk = make_file_record(file_path="/mod", file_size=200)
    mod_cat = make_file_record(file_path="/mod", file_size=100)
    deleted = make_file_record(file_path="/gone")

    disk = [new_rec, unchanged, mod_disk]
    catalog = [unchanged, mod_cat, deleted]

    diff = compute_diff(disk, catalog)
    assert len(diff.new) == 1
    assert len(diff.modified) == 1
    assert len(diff.deleted) == 1


# --- run_discovery ---


def test_run_discovery_orchestrates(tmp_path, monkeypatch):
    """run_discovery calls scan, fetch, diff and logs summary."""
    ds = tmp_path / "source" / "docs"
    ds.mkdir(parents=True)
    (ds / "new.pdf").write_bytes(b"pdf")
    proc = tmp_path / "processing"
    proc.mkdir()

    monkeypatch.setattr(
        "ingestion.stages.discovery.get_data_source_path",
        lambda: str(tmp_path / "source"),
    )
    monkeypatch.setattr(
        "ingestion.stages.discovery.fetch_catalog_records",
        lambda conn: [],
    )
    monkeypatch.setattr("ingestion.stages.discovery.PROCESSING_DIR", proc)

    diff = run_discovery(MagicMock())
    assert len(diff.new) == 1
    assert diff.new[0].filename == "new.pdf"


def test_run_discovery_writes_json(tmp_path, monkeypatch):
    """run_discovery writes discovery.json to processing dir."""
    ds = tmp_path / "source" / "docs"
    ds.mkdir(parents=True)
    (ds / "new.pdf").write_bytes(b"pdf")
    proc = tmp_path / "processing"
    proc.mkdir()

    monkeypatch.setattr(
        "ingestion.stages.discovery.get_data_source_path",
        lambda: str(tmp_path / "source"),
    )
    monkeypatch.setattr(
        "ingestion.stages.discovery.fetch_catalog_records",
        lambda conn: [],
    )
    monkeypatch.setattr("ingestion.stages.discovery.PROCESSING_DIR", proc)

    run_discovery(MagicMock())

    output_file = proc / "discovery.json"
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert len(data["new"]) == 1
    assert data["new"][0]["filename"] == "new.pdf"
    assert data["modified"] == []
    assert data["deleted"] == []


def test_run_discovery_logs_summary(tmp_path, monkeypatch, caplog):
    """run_discovery logs a summary line."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.pdf").write_bytes(b"a")
    proc = tmp_path / "processing"
    proc.mkdir()

    monkeypatch.setattr(
        "ingestion.stages.discovery.get_data_source_path",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(
        "ingestion.stages.discovery.fetch_catalog_records",
        lambda conn: [],
    )
    monkeypatch.setattr("ingestion.stages.discovery.PROCESSING_DIR", proc)

    with caplog.at_level(logging.INFO):
        run_discovery(MagicMock())

    assert any("Discovery complete" in m for m in caplog.messages)
