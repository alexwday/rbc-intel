"""Tests for PPTX processor."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from helpers import make_extraction_prompt, make_rendered_pdf
from ingestion.processors.pptx import (
    _convert_to_pdf,
    _find_soffice,
    process_pptx,
)
from ingestion.utils.file_types import PageResult


def _make_prompt():
    """Build a minimal prompt dict for testing."""
    return make_extraction_prompt()


def _stub_classification(**overrides):
    """Return a default slide classification dict."""
    result = {
        "slide_type_guess": "content_slide",
        "contains_chart": False,
        "chart_count_estimate": 0,
        "contains_dashboard": False,
        "contains_comparison_layout": False,
        "has_dense_visual_content": False,
    }
    result.update(overrides)
    return result


# ── _find_soffice ──────────────────────────────────────────────


@patch(
    "ingestion.processors.pptx.shutil.which",
    return_value="/usr/bin/soffice",
)
def test_find_soffice_on_path(_mock_which):
    """Finds soffice when it is on PATH."""
    assert _find_soffice() == "soffice"


@patch("ingestion.processors.pptx.Path.is_file", return_value=True)
@patch("ingestion.processors.pptx.shutil.which", return_value=None)
def test_find_soffice_absolute_path(_mock_which, _mock_is_file):
    """Falls back to absolute path when not on PATH."""
    assert "soffice" in _find_soffice()


@patch("ingestion.processors.pptx.Path.is_file", return_value=False)
@patch("ingestion.processors.pptx.shutil.which", return_value=None)
def test_find_soffice_not_found(_mock_which, _mock_is_file):
    """Raises RuntimeError when LibreOffice is not installed."""
    with pytest.raises(RuntimeError, match="LibreOffice not found"):
        _find_soffice()


# ── _convert_to_pdf ────────────────────────────────────────────


@patch(
    "ingestion.processors.pptx._find_soffice",
    return_value="soffice",
)
@patch("ingestion.processors.pptx.subprocess.run")
def test_convert_to_pdf_success(mock_run, _soffice, tmp_path):
    """Converts PPTX to PDF and returns the output path."""
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_text("dummy")
    pdf_output = tmp_path / "deck.pdf"
    pdf_output.write_text("dummy-pdf")
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    result = _convert_to_pdf(pptx_path, tmp_path)
    assert result == pdf_output
    cmd = mock_run.call_args[0][0]
    assert "--headless" in cmd
    assert any(p.startswith("-env:UserInstallation=") for p in cmd)


@patch(
    "ingestion.processors.pptx._find_soffice",
    return_value="soffice",
)
@patch("ingestion.processors.pptx.subprocess.run")
def test_convert_to_pdf_nonzero_exit(mock_run, _soffice, tmp_path):
    """Raises RuntimeError on non-zero exit code."""
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_text("dummy")
    mock_run.return_value = MagicMock(
        returncode=1, stdout="detail", stderr="error"
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        _convert_to_pdf(pptx_path, tmp_path)


@patch(
    "ingestion.processors.pptx._find_soffice",
    return_value="soffice",
)
@patch("ingestion.processors.pptx.subprocess.run")
def test_convert_to_pdf_no_output(mock_run, _soffice, tmp_path):
    """Raises RuntimeError when conversion produces no file."""
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_text("dummy")
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="produced no output"):
        _convert_to_pdf(pptx_path, tmp_path)


@patch(
    "ingestion.processors.pptx._find_soffice",
    return_value="soffice",
)
@patch("ingestion.processors.pptx.subprocess.run")
def test_convert_to_pdf_timeout(mock_run, _soffice, tmp_path):
    """Raises RuntimeError when conversion times out."""
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_text("dummy")
    mock_run.side_effect = subprocess.TimeoutExpired(
        cmd="soffice", timeout=120
    )

    with pytest.raises(RuntimeError, match="timed out"):
        _convert_to_pdf(pptx_path, tmp_path)


# ── process_pptx ───────────────────────────────────────────────


@patch(
    "ingestion.processors.pptx.load_prompt",
    return_value=_make_prompt(),
)
@patch(
    "ingestion.processors.pptx.get_vision_dpi_scale",
    return_value=2.0,
)
@patch(
    "ingestion.processors.pptx._classify_slide_with_retry",
    return_value=_stub_classification(),
)
@patch("ingestion.processors.pptx.render_page")
@patch("ingestion.processors.pptx.open_rendered_pdf")
@patch("ingestion.processors.pptx._convert_to_pdf")
@patch("ingestion.processors.pptx.process_page")
def test_process_pptx_success(
    mock_page,
    mock_convert,
    mock_open,
    mock_render,
    _classify,
    _dpi,
    _prompt,
    tmp_path,
):
    """Converts PPTX, processes slides, returns ExtractionResult."""
    mock_convert.return_value = tmp_path / "deck.pdf"
    mock_open.return_value = make_rendered_pdf(3)
    mock_render.side_effect = [b"s1", b"s2", b"s3"]
    mock_page.side_effect = [
        PageResult(1, "Title", "C1", "full_dpi"),
        PageResult(2, "Agenda", "C2", "full_dpi"),
        PageResult(3, "Summary", "C3", "full_dpi"),
    ]

    result = process_pptx("/data/deck.pptx", MagicMock())
    assert result.filetype == "pptx"
    assert result.total_pages == 3
    assert result.pages_succeeded == 3
    assert result.pages_failed == 0


@patch(
    "ingestion.processors.pptx.load_prompt",
    return_value=_make_prompt(),
)
@patch(
    "ingestion.processors.pptx._classify_slide_with_retry",
    return_value=_stub_classification(),
)
@patch("ingestion.processors.pptx._convert_to_pdf")
def test_process_pptx_conversion_failure(mock_convert, _classify, _prompt):
    """Propagates RuntimeError when conversion fails."""
    mock_convert.side_effect = RuntimeError("LibreOffice failed")

    with pytest.raises(RuntimeError, match="LibreOffice failed"):
        process_pptx("/data/deck.pptx", MagicMock())


@patch(
    "ingestion.processors.pptx.load_prompt",
    return_value=_make_prompt(),
)
@patch(
    "ingestion.processors.pptx.get_vision_dpi_scale",
    return_value=2.0,
)
@patch(
    "ingestion.processors.pptx._classify_slide_with_retry",
    return_value=_stub_classification(),
)
@patch("ingestion.processors.pptx.open_rendered_pdf")
@patch("ingestion.processors.pptx._convert_to_pdf")
def test_process_pptx_empty_presentation(
    mock_convert, mock_open, _classify, _dpi, _prompt, tmp_path
):
    """Empty PPTX returns zero pages."""
    mock_convert.return_value = tmp_path / "deck.pdf"
    mock_open.return_value = make_rendered_pdf(0)

    result = process_pptx("/data/empty.pptx", MagicMock())
    assert result.total_pages == 0
    assert result.pages_succeeded == 0


@patch(
    "ingestion.processors.pptx.load_prompt",
    return_value=_make_prompt(),
)
@patch(
    "ingestion.processors.pptx.get_vision_dpi_scale",
    return_value=2.0,
)
@patch(
    "ingestion.processors.pptx._classify_slide_with_retry",
    return_value=_stub_classification(),
)
@patch("ingestion.processors.pptx.render_page")
@patch("ingestion.processors.pptx.open_rendered_pdf")
@patch("ingestion.processors.pptx._convert_to_pdf")
@patch("ingestion.processors.pptx.process_page")
def test_process_pptx_no_context_between_slides(
    mock_page,
    mock_convert,
    mock_open,
    mock_render,
    _classify,
    _dpi,
    _prompt,
    tmp_path,
):
    """Each slide gets the original prompt, no context passing."""
    mock_convert.return_value = tmp_path / "deck.pdf"
    mock_open.return_value = make_rendered_pdf(2)
    mock_render.side_effect = [b"s1", b"s2"]
    mock_page.side_effect = [
        PageResult(1, "S1", "Slide 1 content", "full_dpi"),
        PageResult(2, "S2", "C2", "full_dpi"),
    ]

    process_pptx("/data/deck.pptx", MagicMock())

    slide1_prompt = mock_page.call_args_list[0][0][4]
    slide2_prompt = mock_page.call_args_list[1][0][4]
    assert slide1_prompt == slide2_prompt
    assert "previous page" not in slide2_prompt["user_prompt"]


@patch(
    "ingestion.processors.pptx.load_prompt",
    return_value=_make_prompt(),
)
@patch(
    "ingestion.processors.pptx.get_vision_dpi_scale",
    return_value=2.0,
)
@patch(
    "ingestion.processors.pptx._classify_slide_with_retry",
    return_value=_stub_classification(),
)
@patch("ingestion.processors.pptx.render_page")
@patch("ingestion.processors.pptx.open_rendered_pdf")
@patch("ingestion.processors.pptx._convert_to_pdf")
@patch("ingestion.processors.pptx.process_page")
def test_process_pptx_partial_failure(
    mock_page,
    mock_convert,
    mock_open,
    mock_render,
    _classify,
    _dpi,
    _prompt,
    tmp_path,
):
    """One unrecovered slide failure aborts the whole PPTX."""
    mock_convert.return_value = tmp_path / "deck.pdf"
    mock_open.return_value = make_rendered_pdf(3)
    mock_render.side_effect = [b"s1", b"s2", b"s3"]
    mock_page.side_effect = [
        PageResult(1, "T1", "C1", "full_dpi"),
        RuntimeError("vision timeout"),
        PageResult(3, "T3", "C3", "full_dpi"),
    ]

    with pytest.raises(RuntimeError, match="vision timeout"):
        process_pptx("/data/deck.pptx", MagicMock())
