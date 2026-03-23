"""Tests for file_types dataclasses and hash utility."""

import os

from helpers import make_file_record
from ingestion.utils.file_types import (
    DiscoveryDiff,
    ExtractionResult,
    PageResult,
    compute_file_hash,
)

EMPTY_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb924" + "27ae41e4649b934ca495991b7852b855"
)


def test_file_record_supported_pdf():
    """PDF files are supported."""
    record = make_file_record(filetype="pdf")
    assert record.supported is True


def test_file_record_supported_docx():
    """DOCX files are supported."""
    record = make_file_record(filetype="docx")
    assert record.supported is True


def test_file_record_supported_pptx():
    """PPTX files are supported."""
    record = make_file_record(filetype="pptx")
    assert record.supported is True


def test_file_record_supported_xlsx():
    """XLSX files are supported."""
    record = make_file_record(filetype="xlsx")
    assert record.supported is True


def test_file_record_supported_csv():
    """CSV files are supported."""
    record = make_file_record(filetype="csv")
    assert record.supported is True


def test_file_record_supported_md():
    """Markdown files are supported."""
    record = make_file_record(filetype="md")
    assert record.supported is True


def test_file_record_unsupported_extension():
    """Unknown extensions are not supported."""
    record = make_file_record(filetype="txt")
    assert record.supported is False


def test_file_record_empty_extension():
    """Empty extension is not supported."""
    record = make_file_record(filetype="")
    assert record.supported is False


def test_file_record_filter_defaults():
    """Filters default to empty strings."""
    record = make_file_record()
    assert record.filter_1 == ""
    assert record.filter_2 == ""
    assert record.filter_3 == ""


def test_file_record_with_filters():
    """Filters can be populated."""
    record = make_file_record(filter_1="2026", filter_2="Q1", filter_3="RBC")
    assert record.filter_1 == "2026"
    assert record.filter_2 == "Q1"
    assert record.filter_3 == "RBC"


def test_discovery_diff_empty():
    """Empty diff has three empty lists."""
    diff = DiscoveryDiff(new=[], modified=[], deleted=[])
    assert not diff.new
    assert not diff.modified
    assert not diff.deleted


def test_discovery_diff_with_records():
    """Diff holds records in correct lists."""
    rec = make_file_record()
    diff = DiscoveryDiff(new=[rec], modified=[], deleted=[])
    assert len(diff.new) == 1
    assert diff.new[0] is rec


def test_compute_file_hash_empty_file(tmp_path):
    """Empty file has the SHA-256 of empty bytes."""
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")
    result = compute_file_hash(str(f))
    assert len(result) == 64
    assert result == EMPTY_SHA256


def test_compute_file_hash_known_content(tmp_path):
    """Hash matches expected SHA-256 for known bytes."""
    f = tmp_path / "hello.txt"
    f.write_bytes(b"hello")
    result = compute_file_hash(str(f))
    assert len(result) == 64


def test_compute_file_hash_large_file(tmp_path):
    """Files larger than 8KB chunk size hash correctly."""
    f = tmp_path / "big.bin"
    data = os.urandom(32768)
    f.write_bytes(data)
    result = compute_file_hash(str(f))
    assert len(result) == 64


def test_compute_file_hash_deterministic(tmp_path):
    """Same content always produces the same hash."""
    f = tmp_path / "repeat.txt"
    f.write_bytes(b"deterministic")
    first = compute_file_hash(str(f))
    second = compute_file_hash(str(f))
    assert first == second


def test_page_result_fields():
    """PageResult stores page extraction metadata."""
    page = PageResult(
        page_number=1,
        page_title="Q3 Results",
        content="### Text\nRevenue grew 5%",
        method="full_dpi",
    )
    assert page.page_number == 1
    assert page.page_title == "Q3 Results"
    assert "Revenue" in page.content
    assert page.method == "full_dpi"
    assert page.error == ""
    assert not page.metadata


def test_page_result_with_error():
    """PageResult stores error details for failed pages."""
    page = PageResult(
        page_number=3,
        page_title="",
        content="",
        method="failed",
        error="Vision call timed out",
    )
    assert page.method == "failed"
    assert page.error == "Vision call timed out"
    assert page.content == ""
    assert not page.metadata


def test_page_result_all_methods():
    """PageResult supports all extraction methods."""
    methods = (
        "full_dpi",
        "high_detail",
        "half_dpi",
        "split_halves",
        "xlsx_sheet_classification",
        "failed",
    )
    for method in methods:
        page = PageResult(
            page_number=1,
            page_title="Test",
            content="content",
            method=method,
        )
        assert page.method == method


def test_extraction_result_fields():
    """ExtractionResult aggregates page-level results."""
    pages = [
        PageResult(
            page_number=1,
            page_title="Page 1",
            content="content",
            method="full_dpi",
        ),
        PageResult(
            page_number=2,
            page_title="Page 2",
            content="more content",
            method="half_dpi",
        ),
    ]
    result = ExtractionResult(
        file_path="/data/doc.pdf",
        filetype="pdf",
        pages=pages,
        total_pages=2,
        pages_succeeded=2,
        pages_failed=0,
    )
    assert result.file_path == "/data/doc.pdf"
    assert result.filetype == "pdf"
    assert len(result.pages) == 2
    assert result.total_pages == 2
    assert result.pages_succeeded == 2
    assert result.pages_failed == 0


def test_extraction_result_empty():
    """ExtractionResult works with zero pages."""
    result = ExtractionResult(
        file_path="/data/empty.pdf",
        filetype="pdf",
        pages=[],
        total_pages=0,
        pages_succeeded=0,
        pages_failed=0,
    )
    assert result.total_pages == 0
    assert not result.pages
