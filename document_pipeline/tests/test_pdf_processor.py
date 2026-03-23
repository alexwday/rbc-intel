"""Tests for PDF processor."""

from unittest.mock import MagicMock, patch

import pytest

from helpers import make_extraction_prompt, make_rendered_pdf
from ingestion.processors.pdf import _build_context_prompt, process_pdf
from ingestion.utils.file_types import PageResult


def _make_prompt():
    """Build a minimal prompt dict for testing."""
    return make_extraction_prompt()


def test_build_context_prompt_truncates_long_content():
    """Long prior-page content is truncated before prompt injection."""
    prompt = _make_prompt()
    long_content = ("x" * 500) + "\nsecond\n" + ("y" * 500)

    result = _build_context_prompt(prompt, long_content)

    assert "CONTEXT" in result["user_prompt"]
    assert "second" in result["user_prompt"]


@patch(
    "ingestion.processors.pdf._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "table_continuation_detected": False,
        "contains_page_furniture": False,
    },
)
@patch("ingestion.processors.pdf.load_prompt", return_value=_make_prompt())
@patch("ingestion.processors.pdf.get_vision_dpi_scale", return_value=2.0)
@patch("ingestion.processors.pdf.render_page")
@patch("ingestion.processors.pdf.open_rendered_pdf")
@patch("ingestion.processors.pdf.process_page")
def test_process_pdf_success(mock_page, mock_open, mock_render, _dpi, _prompt, _classify):
    """Processes all pages and returns ExtractionResult."""
    mock_open.return_value = make_rendered_pdf(2)
    mock_render.side_effect = [b"page1", b"page2"]
    mock_page.side_effect = [
        PageResult(1, "Title 1", "Content 1", "full_dpi"),
        PageResult(2, "Title 2", "Content 2", "full_dpi"),
    ]

    result = process_pdf("/data/test.pdf", MagicMock())
    assert result.file_path == "/data/test.pdf"
    assert result.filetype == "pdf"
    assert result.total_pages == 2
    assert result.pages_succeeded == 2
    assert result.pages_failed == 0
    assert result.pages[0].page_title == "Title 1"


@patch(
    "ingestion.processors.pdf._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "table_continuation_detected": False,
        "contains_page_furniture": False,
    },
)
@patch("ingestion.processors.pdf.load_prompt", return_value=_make_prompt())
@patch("ingestion.processors.pdf.get_vision_dpi_scale", return_value=2.0)
@patch("ingestion.processors.pdf.open_rendered_pdf")
def test_process_pdf_empty_document(mock_open, _dpi, _prompt, _classify):
    """Empty PDF returns zero pages."""
    mock_open.return_value = make_rendered_pdf(0)

    result = process_pdf("/data/empty.pdf", MagicMock())
    assert result.total_pages == 0
    assert result.pages_succeeded == 0


@patch(
    "ingestion.processors.pdf._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "table_continuation_detected": False,
        "contains_page_furniture": False,
    },
)
@patch("ingestion.processors.pdf.load_prompt", return_value=_make_prompt())
@patch("ingestion.processors.pdf.get_vision_dpi_scale", return_value=2.0)
@patch("ingestion.processors.pdf.render_page")
@patch("ingestion.processors.pdf.open_rendered_pdf")
def test_process_pdf_render_failure_becomes_failed_page(
    mock_open, mock_render, _dpi, _prompt, _classify
):
    """Render failure aborts the PDF so the file can be retried cleanly."""
    mock_open.return_value = make_rendered_pdf(1)
    mock_render.side_effect = RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        process_pdf("/data/test.pdf", MagicMock())


@patch(
    "ingestion.processors.pdf._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "table_continuation_detected": False,
        "contains_page_furniture": False,
    },
)
@patch("ingestion.processors.pdf.load_prompt", return_value=_make_prompt())
@patch("ingestion.processors.pdf.get_vision_dpi_scale", return_value=2.0)
@patch("ingestion.processors.pdf.render_page")
@patch("ingestion.processors.pdf.open_rendered_pdf")
@patch("ingestion.processors.pdf.process_page")
def test_process_pdf_partial_failure(
    mock_page, mock_open, mock_render, _dpi, _prompt, _classify
):
    """One unrecovered page failure aborts the whole PDF."""
    mock_open.return_value = make_rendered_pdf(3)
    mock_render.side_effect = [b"p1", b"p2", b"p3"]
    mock_page.side_effect = [
        PageResult(1, "Title 1", "Content 1", "full_dpi"),
        RuntimeError("vision timeout"),
        PageResult(3, "Title 3", "Content 3", "full_dpi"),
    ]

    with pytest.raises(RuntimeError, match="vision timeout"):
        process_pdf("/data/test.pdf", MagicMock())


@patch(
    "ingestion.processors.pdf._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "table_continuation_detected": False,
        "contains_page_furniture": False,
    },
)
@patch("ingestion.processors.pdf.load_prompt", return_value=_make_prompt())
@patch("ingestion.processors.pdf.get_vision_dpi_scale", return_value=2.0)
@patch("ingestion.processors.pdf.render_page")
@patch("ingestion.processors.pdf.open_rendered_pdf")
@patch("ingestion.processors.pdf.process_page")
def test_process_pdf_all_pages_fail(
    mock_page, mock_open, mock_render, _dpi, _prompt, _classify
):
    """A PDF with unrecovered failures raises instead of partial success."""
    mock_open.return_value = make_rendered_pdf(2)
    mock_render.side_effect = [b"p1", b"p2"]
    mock_page.side_effect = RuntimeError("always fails")

    with pytest.raises(RuntimeError, match="always fails"):
        process_pdf("/data/test.pdf", MagicMock())
