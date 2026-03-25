"""Tests for DOCX processor."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from helpers import make_extraction_prompt, make_rendered_pdf
from ingestion.processors.docx.processor import (
    _build_context_prompt,
    _convert_to_pdf,
    _find_soffice,
    extract_document_outline,
    process_docx,
)
from ingestion.utils.file_types import PageResult


def _make_prompt():
    """Build a minimal prompt dict for testing."""
    return make_extraction_prompt()


# ── _find_soffice ──────────────────────────────────────────────


@patch(
    "ingestion.processors.docx.processor.shutil.which",
    return_value="/usr/bin/soffice",
)
def test_find_soffice_on_path(_mock_which):
    """Finds soffice when it is on PATH."""
    assert _find_soffice() == "soffice"


@patch("ingestion.processors.docx.processor.Path.is_file", return_value=True)
@patch("ingestion.processors.docx.processor.shutil.which", return_value=None)
def test_find_soffice_absolute_path(_mock_which, _mock_is_file):
    """Falls back to absolute path when not on PATH."""
    assert "soffice" in _find_soffice()


@patch("ingestion.processors.docx.processor.Path.is_file", return_value=False)
@patch("ingestion.processors.docx.processor.shutil.which", return_value=None)
def test_find_soffice_not_found(_mock_which, _mock_is_file):
    """Raises RuntimeError when LibreOffice is not installed."""
    with pytest.raises(RuntimeError, match="LibreOffice not found"):
        _find_soffice()


# ── _convert_to_pdf ────────────────────────────────────────────


@patch(
    "ingestion.processors.docx.processor._find_soffice", return_value="soffice"
)
@patch("ingestion.processors.docx.processor.subprocess.run")
def test_convert_to_pdf_success(mock_run, _soffice, tmp_path):
    """Converts DOCX to PDF and returns the output path."""
    docx_path = tmp_path / "report.docx"
    docx_path.write_text("dummy")
    pdf_output = tmp_path / "report.pdf"
    pdf_output.write_text("dummy-pdf")
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    result = _convert_to_pdf(docx_path, tmp_path)
    assert result == pdf_output
    cmd = mock_run.call_args[0][0]
    assert "--headless" in cmd
    assert any(p.startswith("-env:UserInstallation=") for p in cmd)


@patch(
    "ingestion.processors.docx.processor._find_soffice", return_value="soffice"
)
@patch("ingestion.processors.docx.processor.subprocess.run")
def test_convert_to_pdf_nonzero_exit(mock_run, _soffice, tmp_path):
    """Raises RuntimeError on non-zero exit code."""
    docx_path = tmp_path / "report.docx"
    docx_path.write_text("dummy")
    mock_run.return_value = MagicMock(
        returncode=1, stdout="detail", stderr="error"
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        _convert_to_pdf(docx_path, tmp_path)


@patch(
    "ingestion.processors.docx.processor._find_soffice", return_value="soffice"
)
@patch("ingestion.processors.docx.processor.subprocess.run")
def test_convert_to_pdf_no_output(mock_run, _soffice, tmp_path):
    """Raises RuntimeError when conversion produces no file."""
    docx_path = tmp_path / "report.docx"
    docx_path.write_text("dummy")
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="produced no output"):
        _convert_to_pdf(docx_path, tmp_path)


@patch(
    "ingestion.processors.docx.processor._find_soffice", return_value="soffice"
)
@patch("ingestion.processors.docx.processor.subprocess.run")
def test_convert_to_pdf_timeout(mock_run, _soffice, tmp_path):
    """Raises RuntimeError when conversion times out."""
    docx_path = tmp_path / "report.docx"
    docx_path.write_text("dummy")
    mock_run.side_effect = subprocess.TimeoutExpired(
        cmd="soffice", timeout=120
    )

    with pytest.raises(RuntimeError, match="timed out"):
        _convert_to_pdf(docx_path, tmp_path)


# ── _build_context_prompt ──────────────────────────────────────


def test_build_context_prompt_prepends_context():
    """Augments user_prompt with previous-page content."""
    prompt = _make_prompt()
    result = _build_context_prompt(prompt, "### Tables\n| A | B |")

    assert "previous page" in result["user_prompt"]
    assert "Do not copy or restate" in result["user_prompt"]
    assert "### Tables" in result["user_prompt"]
    assert result["user_prompt"].endswith(prompt["user_prompt"])


def test_build_context_prompt_truncates_long_content():
    """Truncates content longer than 800 chars at newline."""
    prompt = _make_prompt()
    long_content = ("x" * 500) + "\nsecond\n" + ("y" * 500)
    result = _build_context_prompt(prompt, long_content)

    assert "second" in result["user_prompt"]


def test_build_context_prompt_truncates_without_newline():
    """Truncates content with no newline in the tail region."""
    prompt = _make_prompt()
    result = _build_context_prompt(prompt, "x" * 1200)
    assert "previous page" in result["user_prompt"]


def test_build_context_prompt_does_not_mutate_original():
    """Returns a new dict; original prompt is unchanged."""
    prompt = _make_prompt()
    original = prompt["user_prompt"]
    _build_context_prompt(prompt, "context")
    assert prompt["user_prompt"] == original


# ── extract_document_outline ───────────────────────────────────


def _make_paragraph(style_name, text, hyperlinks=None):
    """Build a mock paragraph with style, text, and hyperlinks."""
    para = MagicMock()
    para.style.name = style_name
    para.text = text
    if hyperlinks is not None:
        para.hyperlinks = hyperlinks
    else:
        para.hyperlinks = []
    return para


@patch("ingestion.processors.docx.processor.Document")
def test_extract_outline_with_headings(mock_doc_cls):
    """Returns formatted outline when headings are present."""
    doc = MagicMock()
    doc.paragraphs = [
        _make_paragraph("Heading 1", "Executive Summary"),
        _make_paragraph("Normal", "Some body text."),
        _make_paragraph("Heading 2", "Credit Exposure"),
        _make_paragraph("Heading 1", "Portfolio Narrative"),
    ]
    mock_doc_cls.return_value = doc

    result = extract_document_outline("/data/report.docx")
    assert "## Document Outline" in result
    assert "- H1: Executive Summary" in result
    assert "- H2: Credit Exposure" in result
    assert "- H1: Portfolio Narrative" in result
    assert "## Hyperlinks" not in result


@patch("ingestion.processors.docx.processor.Document")
def test_extract_outline_with_hyperlinks(mock_doc_cls):
    """Includes hyperlinks section when links are found."""
    link = MagicMock()
    link.url = "https://example.com"
    doc = MagicMock()
    doc.paragraphs = [
        _make_paragraph("Heading 1", "Summary", hyperlinks=[link]),
    ]
    mock_doc_cls.return_value = doc

    result = extract_document_outline("/data/report.docx")
    assert "## Hyperlinks" in result
    assert "https://example.com" in result
    assert '(in "Summary")' in result


@patch("ingestion.processors.docx.processor.Document")
def test_extract_outline_no_headings(mock_doc_cls):
    """Returns empty string when no headings exist."""
    doc = MagicMock()
    doc.paragraphs = [
        _make_paragraph("Normal", "Just body text."),
    ]
    mock_doc_cls.return_value = doc

    assert extract_document_outline("/data/report.docx") == ""


@patch("ingestion.processors.docx.processor.Document")
def test_extract_outline_empty_heading_text(mock_doc_cls):
    """Skips headings with empty text."""
    doc = MagicMock()
    doc.paragraphs = [
        _make_paragraph("Heading 1", ""),
        _make_paragraph("Heading 2", "Real Heading"),
    ]
    mock_doc_cls.return_value = doc

    result = extract_document_outline("/data/report.docx")
    assert "- H2: Real Heading" in result
    assert "H1" not in result


@patch(
    "ingestion.processors.docx.processor.Document",
    side_effect=OSError("file not found"),
)
def test_extract_outline_file_error(_mock_doc_cls):
    """Returns empty string on file read error."""
    assert extract_document_outline("/bad/path.docx") == ""


@patch("ingestion.processors.docx.processor.Document")
def test_extract_outline_hyperlinks_no_attribute(mock_doc_cls):
    """Falls back gracefully when hyperlinks attribute is missing."""
    para = MagicMock()
    para.style.name = "Heading 1"
    para.text = "Title"
    del para.hyperlinks
    doc = MagicMock()
    doc.paragraphs = [para]
    mock_doc_cls.return_value = doc

    result = extract_document_outline("/data/report.docx")
    assert "- H1: Title" in result


@patch("ingestion.processors.docx.processor.Document")
def test_extract_outline_hyperlink_without_section(mock_doc_cls):
    """Hyperlinks before any heading have no section label."""
    link = MagicMock()
    link.url = "https://orphan.com"
    doc = MagicMock()
    doc.paragraphs = [
        _make_paragraph("Normal", "intro", hyperlinks=[link]),
        _make_paragraph("Heading 1", "First Section"),
    ]
    mock_doc_cls.return_value = doc

    result = extract_document_outline("/data/report.docx")
    assert "- https://orphan.com" in result
    assert '(in "' not in result.split("https://orphan.com")[1].split("\n")[0]


# ── process_docx ───────────────────────────────────────────────


@patch(
    "ingestion.processors.docx.processor.extract_document_outline",
    return_value="",
)
@patch(
    "ingestion.processors.docx.processor._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "section_continuation_detected": False,
        "table_continuation_detected": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "contains_page_furniture": False,
    },
)
@patch(
    "ingestion.processors.docx.processor.load_prompt",
    return_value=_make_prompt(),
)
@patch(
    "ingestion.processors.docx.processor.get_vision_dpi_scale",
    return_value=2.0,
)
@patch("ingestion.processors.docx.processor.render_page")
@patch("ingestion.processors.docx.processor.open_rendered_pdf")
@patch("ingestion.processors.docx.processor._convert_to_pdf")
@patch("ingestion.processors.docx.processor.process_page")
def test_process_docx_success(
    mock_page,
    mock_convert,
    mock_open,
    mock_render,
    _dpi,
    _prompt,
    _classify,
    _outline,
    tmp_path,
):
    """Converts DOCX, processes pages, returns ExtractionResult."""
    mock_convert.return_value = tmp_path / "doc.pdf"
    mock_open.return_value = make_rendered_pdf(3)
    mock_render.side_effect = [b"p1", b"p2", b"p3"]
    mock_page.side_effect = [
        PageResult(1, "T1", "C1", "full_dpi"),
        PageResult(2, "T2", "C2", "full_dpi"),
        PageResult(3, "T3", "C3", "full_dpi"),
    ]

    result = process_docx("/data/report.docx", MagicMock())
    assert result.filetype == "docx"
    assert result.total_pages == 3
    assert result.pages_succeeded == 3
    assert result.pages_failed == 0


@patch(
    "ingestion.processors.docx.processor.extract_document_outline",
    return_value="",
)
@patch(
    "ingestion.processors.docx.processor._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "section_continuation_detected": False,
        "table_continuation_detected": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "contains_page_furniture": False,
    },
)
@patch(
    "ingestion.processors.docx.processor.load_prompt",
    return_value=_make_prompt(),
)
@patch("ingestion.processors.docx.processor._convert_to_pdf")
def test_process_docx_conversion_failure(
    mock_convert, _prompt, _classify, _outline
):
    """Propagates RuntimeError when conversion fails."""
    mock_convert.side_effect = RuntimeError("LibreOffice failed")

    with pytest.raises(RuntimeError, match="LibreOffice failed"):
        process_docx("/data/report.docx", MagicMock())


@patch(
    "ingestion.processors.docx.processor.extract_document_outline",
    return_value="",
)
@patch(
    "ingestion.processors.docx.processor._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "section_continuation_detected": False,
        "table_continuation_detected": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "contains_page_furniture": False,
    },
)
@patch(
    "ingestion.processors.docx.processor.load_prompt",
    return_value=_make_prompt(),
)
@patch(
    "ingestion.processors.docx.processor.get_vision_dpi_scale",
    return_value=2.0,
)
@patch("ingestion.processors.docx.processor.open_rendered_pdf")
@patch("ingestion.processors.docx.processor._convert_to_pdf")
def test_process_docx_empty_document(
    mock_convert, mock_open, _dpi, _prompt, _classify, _outline, tmp_path
):
    """Empty DOCX returns zero pages."""
    mock_convert.return_value = tmp_path / "doc.pdf"
    mock_open.return_value = make_rendered_pdf(0)

    result = process_docx("/data/empty.docx", MagicMock())
    assert result.total_pages == 0
    assert result.pages_succeeded == 0


@patch(
    "ingestion.processors.docx.processor.extract_document_outline",
    return_value="",
)
@patch(
    "ingestion.processors.docx.processor._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "section_continuation_detected": False,
        "table_continuation_detected": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "contains_page_furniture": False,
    },
)
@patch(
    "ingestion.processors.docx.processor.load_prompt",
    return_value=_make_prompt(),
)
@patch(
    "ingestion.processors.docx.processor.get_vision_dpi_scale",
    return_value=2.0,
)
@patch("ingestion.processors.docx.processor.render_page")
@patch("ingestion.processors.docx.processor.open_rendered_pdf")
@patch("ingestion.processors.docx.processor._convert_to_pdf")
@patch("ingestion.processors.docx.processor.process_page")
def test_process_docx_passes_context(
    mock_page,
    mock_convert,
    mock_open,
    mock_render,
    _dpi,
    _prompt,
    _classify,
    _outline,
    tmp_path,
):
    """Subsequent pages receive previous-page content as context."""
    mock_convert.return_value = tmp_path / "doc.pdf"
    mock_open.return_value = make_rendered_pdf(2)
    mock_render.side_effect = [b"p1", b"p2"]
    mock_page.side_effect = [
        PageResult(1, "T1", "Page 1 table", "full_dpi"),
        PageResult(2, "T2", "C2", "full_dpi"),
    ]

    process_docx("/data/report.docx", MagicMock())

    page2_prompt = mock_page.call_args_list[1][0][4]
    assert "previous page" in page2_prompt["user_prompt"]
    assert "Page 1 table" in page2_prompt["user_prompt"]


@patch(
    "ingestion.processors.docx.processor.extract_document_outline",
    return_value="",
)
@patch(
    "ingestion.processors.docx.processor._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "section_continuation_detected": False,
        "table_continuation_detected": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "contains_page_furniture": False,
    },
)
@patch(
    "ingestion.processors.docx.processor.load_prompt",
    return_value=_make_prompt(),
)
@patch(
    "ingestion.processors.docx.processor.get_vision_dpi_scale",
    return_value=2.0,
)
@patch("ingestion.processors.docx.processor.render_page")
@patch("ingestion.processors.docx.processor.open_rendered_pdf")
@patch("ingestion.processors.docx.processor._convert_to_pdf")
@patch("ingestion.processors.docx.processor.process_page")
def test_process_docx_partial_failure(
    mock_page,
    mock_convert,
    mock_open,
    mock_render,
    _dpi,
    _prompt,
    _classify,
    _outline,
    tmp_path,
):
    """One unrecovered page failure aborts the whole DOCX."""
    mock_convert.return_value = tmp_path / "doc.pdf"
    mock_open.return_value = make_rendered_pdf(3)
    mock_render.side_effect = [b"p1", b"p2", b"p3"]
    mock_page.side_effect = [
        PageResult(1, "T1", "C1", "full_dpi"),
        RuntimeError("vision timeout"),
        PageResult(3, "T3", "C3", "full_dpi"),
    ]

    with pytest.raises(RuntimeError, match="vision timeout"):
        process_docx("/data/report.docx", MagicMock())


@patch(
    "ingestion.processors.docx.processor.extract_document_outline",
    return_value="## Document Outline\n- H1: Summary",
)
@patch(
    "ingestion.processors.docx.processor._classify_continuation_with_retry",
    return_value={
        "continued_from_previous_page": False,
        "section_continuation_detected": False,
        "table_continuation_detected": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "contains_page_furniture": False,
    },
)
@patch(
    "ingestion.processors.docx.processor.load_prompt",
    return_value=_make_prompt(),
)
@patch(
    "ingestion.processors.docx.processor.get_vision_dpi_scale",
    return_value=2.0,
)
@patch("ingestion.processors.docx.processor.render_page")
@patch("ingestion.processors.docx.processor.open_rendered_pdf")
@patch("ingestion.processors.docx.processor._convert_to_pdf")
@patch("ingestion.processors.docx.processor.process_page")
def test_process_docx_prepends_outline(
    mock_page,
    mock_convert,
    mock_open,
    mock_render,
    _dpi,
    _prompt,
    _classify,
    _outline,
    tmp_path,
):
    """Document outline is prepended to extracted_text for each page."""
    mock_convert.return_value = tmp_path / "doc.pdf"
    mock_open.return_value = make_rendered_pdf(1)
    mock_render.side_effect = [b"p1"]
    mock_page.side_effect = [
        PageResult(1, "T1", "C1", "full_dpi"),
    ]

    process_docx("/data/report.docx", MagicMock())

    call_kwargs = mock_page.call_args_list[0][1]
    supplementary = call_kwargs.get("extracted_text", "")
    assert "## Document Outline" in supplementary
    assert "- H1: Summary" in supplementary
