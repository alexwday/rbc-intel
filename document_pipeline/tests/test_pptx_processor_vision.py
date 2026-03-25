"""Tests for PPTX processor vision pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from helpers import make_extraction_prompt
from ingestion.processors.pptx.processor import (
    RenderedPdf,
    call_vision,
    extract_page_text,
    open_rendered_pdf,
    parse_vision_response,
    process_page,
    render_page,
    shrink_image,
    split_image,
)


def _make_timeout_error():
    """Build an openai.APITimeoutError for testing."""
    return openai.APITimeoutError(request=httpx.Request("POST", "http://x"))


def _make_bad_request_error():
    """Build an openai.BadRequestError for testing."""
    request = httpx.Request("POST", "http://x")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError("bad request", response=response, body={})


def _make_vision_response(page_title="Test Title", content="### Text\nHello"):
    """Build a mock LLM response with a tool call."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "extract_page_content",
                                "arguments": json.dumps(
                                    {
                                        "page_title": page_title,
                                        "content": content,
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }


def _make_prompt():
    """Build a minimal prompt dict for testing."""
    return make_extraction_prompt()


# -- open_rendered_pdf -----------------------------------------------


@patch("ingestion.processors.pptx.processor.fitz")
def test_open_rendered_pdf_success(mock_fitz):
    """Opens a PDF and exposes total pages for streaming."""
    mock_doc = MagicMock()
    mock_doc.page_count = 2
    mock_fitz.open.return_value = mock_doc
    mock_fitz.Matrix.return_value = "matrix"

    with open_rendered_pdf(Path("test.pdf"), 2.0) as rendered:
        assert rendered.total_pages == 2
        assert rendered.matrix == "matrix"
        assert rendered.pdf_path == Path("test.pdf")

    mock_doc.close.assert_called_once()
    mock_fitz.TOOLS.mupdf_display_errors.assert_any_call(False)


@patch("ingestion.processors.pptx.processor.fitz")
def test_open_rendered_pdf_open_fails(mock_fitz):
    """Raises when PDF cannot be opened."""
    mock_fitz.open.side_effect = RuntimeError("corrupt")
    with pytest.raises(RuntimeError, match="Failed to open PDF"):
        with open_rendered_pdf(Path("bad.pdf"), 2.0):
            pass
    mock_fitz.TOOLS.mupdf_display_errors.assert_any_call(True)


# -- render_page -----------------------------------------------------


@patch("ingestion.processors.pptx.processor.fitz")
def test_render_page_success(_mock_fitz):
    """Renders a single page to PNG bytes."""
    mock_page = MagicMock()
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"png-data"
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc = MagicMock()
    mock_doc.load_page.return_value = mock_page

    rendered = RenderedPdf(Path("test.pdf"), mock_doc, "matrix", 2)

    result = render_page(rendered, 1)
    assert result == b"png-data"
    mock_doc.load_page.assert_called_once_with(0)


@patch("ingestion.processors.pptx.processor.fitz")
def test_render_page_page_fails(_mock_fitz):
    """Raises when a page fails to render."""
    bad_page = MagicMock()
    bad_page.get_pixmap.side_effect = RuntimeError("render err")
    mock_doc = MagicMock()
    mock_doc.load_page.return_value = bad_page

    rendered = RenderedPdf(Path("test.pdf"), mock_doc, "matrix", 2)

    with pytest.raises(RuntimeError, match="Failed to render page 2"):
        render_page(rendered, 2)


def test_render_page_out_of_range():
    """Out-of-range page requests raise ValueError."""
    rendered = RenderedPdf(Path("test.pdf"), MagicMock(), "matrix", 1)

    with pytest.raises(ValueError, match="out of range"):
        render_page(rendered, 2)


# -- extract_page_text -----------------------------------------------


def test_extract_page_text_success():
    """Extracts text from a valid page."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = "  Hello world  "
    mock_doc = MagicMock()
    mock_doc.load_page.return_value = mock_page
    rendered = RenderedPdf(Path("test.pdf"), mock_doc, "matrix", 3)

    result = extract_page_text(rendered, 2)
    assert result == "Hello world"
    mock_doc.load_page.assert_called_once_with(1)


def test_extract_page_text_out_of_range():
    """Returns empty string for out-of-range page number."""
    rendered = RenderedPdf(Path("test.pdf"), MagicMock(), "matrix", 2)
    assert extract_page_text(rendered, 0) == ""
    assert extract_page_text(rendered, 3) == ""


def test_extract_page_text_fitz_error():
    """Returns empty string when PyMuPDF raises an error."""
    mock_doc = MagicMock()
    mock_doc.load_page.side_effect = RuntimeError("corrupt page")
    rendered = RenderedPdf(Path("test.pdf"), mock_doc, "matrix", 2)

    assert extract_page_text(rendered, 1) == ""


# -- shrink_image ----------------------------------------------------


@patch("ingestion.processors.pptx.processor.fitz")
def test_shrink_image(mock_fitz):
    """Shrinks image to half resolution."""
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"small-png"
    mock_fitz.Pixmap.return_value = mock_pix

    result = shrink_image(b"big-png")
    mock_pix.shrink.assert_called_once_with(1)
    assert result == b"small-png"


# -- split_image -----------------------------------------------------


@patch("ingestion.processors.pptx.processor.fitz")
def test_split_image_portrait(mock_fitz):
    """Portrait images split into top and bottom halves."""
    mock_src = MagicMock()
    mock_src.height = 100
    mock_src.width = 80

    mock_top = MagicMock()
    mock_top.tobytes.return_value = b"top-png"
    mock_bot = MagicMock()
    mock_bot.tobytes.return_value = b"bot-png"

    mock_fitz.Pixmap.side_effect = [
        mock_src,
        mock_top,
        mock_bot,
    ]
    mock_fitz.IRect.side_effect = ["top-rect", "bot-rect"]

    top, bot, orientation = split_image(b"full-png")
    assert top == b"top-png"
    assert bot == b"bot-png"
    assert orientation == "vertical"
    mock_fitz.IRect.assert_any_call(0, 0, 80, 50)
    mock_fitz.IRect.assert_any_call(0, 50, 80, 100)


@patch("ingestion.processors.pptx.processor.fitz")
def test_split_image_landscape(mock_fitz):
    """Landscape images split into left and right halves."""
    mock_src = MagicMock()
    mock_src.height = 100
    mock_src.width = 200

    mock_left = MagicMock()
    mock_left.tobytes.return_value = b"left-png"
    mock_right = MagicMock()
    mock_right.tobytes.return_value = b"right-png"

    mock_fitz.Pixmap.side_effect = [
        mock_src,
        mock_left,
        mock_right,
    ]
    mock_fitz.IRect.side_effect = ["left-rect", "right-rect"]

    left, right, orientation = split_image(b"full-png")
    assert left == b"left-png"
    assert right == b"right-png"
    assert orientation == "horizontal"
    mock_fitz.IRect.assert_any_call(0, 0, 100, 100)
    mock_fitz.IRect.assert_any_call(100, 0, 200, 100)


@patch("ingestion.processors.pptx.processor.fitz")
def test_split_image_too_short(mock_fitz):
    """Very short portrait images raise ValueError."""
    mock_src = MagicMock()
    mock_src.height = 10
    mock_src.width = 10
    mock_fitz.Pixmap.return_value = mock_src

    with pytest.raises(ValueError, match="too short"):
        split_image(b"tiny-png")


@patch("ingestion.processors.pptx.processor.fitz")
def test_split_image_too_narrow_landscape(mock_fitz):
    """Very narrow landscape images raise ValueError."""
    mock_src = MagicMock()
    mock_src.height = 10
    mock_src.width = 20
    mock_fitz.Pixmap.return_value = mock_src

    with pytest.raises(ValueError, match="too narrow"):
        split_image(b"tiny-png")


# -- parse_vision_response -------------------------------------------


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "missing choices"),
        (
            {"choices": [{"message": None}]},
            "missing message payload",
        ),
        (
            {"choices": [{"message": {"tool_calls": [{"function": None}]}}]},
            "missing function payload",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": None,
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "missing function arguments",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": json.dumps([]),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "must decode to an object",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": json.dumps(
                                            {
                                                "page_title": "T",
                                                "content": 1,
                                            }
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "missing string page_title/content",
        ),
    ],
)
def test_parse_vision_response_validates_shape(response, message):
    """Invalid tool payload shapes raise targeted ValueErrors."""
    with pytest.raises(ValueError, match=message):
        parse_vision_response(response)


def test_parse_vision_response_empty_tool_calls():
    """Empty tool_calls list raises ValueError."""
    response = {"choices": [{"message": {"tool_calls": []}}]}
    with pytest.raises(ValueError, match="missing tool calls"):
        parse_vision_response(response)


# -- call_vision -----------------------------------------------------


def test_call_vision_success():
    """Parses tool call response into title and content."""
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_vision_response("Title", "Body")
    prompt = _make_prompt()

    title, content = call_vision(mock_llm, b"img", prompt, "required")
    assert title == "Title"
    assert content == "Body"
    mock_llm.call.assert_called_once()


def test_call_vision_defaults_to_auto_detail():
    """Default detail level is 'auto'."""
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_vision_response()
    prompt = _make_prompt()

    call_vision(mock_llm, b"img", prompt, "required")
    messages = mock_llm.call.call_args[1]["messages"]
    image_block = messages[1]["content"][0]
    assert image_block["image_url"]["detail"] == "auto"


def test_call_vision_passes_explicit_detail():
    """Explicit detail parameter overrides the default."""
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_vision_response()
    prompt = _make_prompt()

    call_vision(mock_llm, b"img", prompt, "required", detail="high")
    messages = mock_llm.call.call_args[1]["messages"]
    image_block = messages[1]["content"][0]
    assert image_block["image_url"]["detail"] == "high"


@patch("ingestion.processors.pptx.processor.time.sleep")
def test_call_vision_retries_on_transient_error(mock_sleep):
    """Retries on retryable errors with exponential backoff."""
    mock_llm = MagicMock()
    mock_llm.call.side_effect = [
        _make_timeout_error(),
        _make_vision_response(),
    ]
    prompt = _make_prompt()

    title, _ = call_vision(mock_llm, b"img", prompt, "required")
    assert title == "Test Title"
    assert mock_llm.call.call_count == 2
    mock_sleep.assert_called_once_with(2.0)


@patch("ingestion.processors.pptx.processor.time.sleep")
def test_call_vision_exhausts_retries(mock_sleep):
    """Raises after max retries exhausted."""
    mock_llm = MagicMock()
    mock_llm.call.side_effect = _make_timeout_error()
    prompt = _make_prompt()

    with pytest.raises(openai.APITimeoutError):
        call_vision(mock_llm, b"img", prompt, "required")
    assert mock_llm.call.call_count == 3
    assert mock_sleep.call_count == 2


def test_call_vision_includes_extracted_text():
    """Extracted text is appended to user prompt when provided."""
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_vision_response()
    prompt = _make_prompt()

    call_vision(
        mock_llm,
        b"img",
        prompt,
        "required",
        extracted_text="Some slide text",
    )
    messages = mock_llm.call.call_args[1]["messages"]
    user_text = messages[1]["content"][1]["text"]
    assert "Programmatic Text Extraction" in user_text
    assert "Some slide text" in user_text


def test_call_vision_no_extracted_text_uses_plain_prompt():
    """Empty extracted_text leaves user prompt unchanged."""
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_vision_response()
    prompt = _make_prompt()

    call_vision(mock_llm, b"img", prompt, "required", extracted_text="")
    messages = mock_llm.call.call_args[1]["messages"]
    user_text = messages[1]["content"][1]["text"]
    assert user_text == prompt["user_prompt"]
    assert "Programmatic Text Extraction" not in user_text


def test_call_vision_non_retryable_raises():
    """Non-retryable errors propagate immediately."""
    mock_llm = MagicMock()
    mock_llm.call.side_effect = ValueError("bad input")
    prompt = _make_prompt()

    with pytest.raises(ValueError, match="bad input"):
        call_vision(mock_llm, b"img", prompt, "required")
    assert mock_llm.call.call_count == 1


@patch(
    "ingestion.processors.pptx.processor.get_pptx_vision_max_retries",
    return_value=0,
)
def test_call_vision_zero_retries_raises_runtime_error(
    _mock_config,
):
    """Zero-retry configuration fails with a clear error."""
    mock_llm = MagicMock()
    prompt = _make_prompt()

    with pytest.raises(RuntimeError, match="without a response"):
        call_vision(mock_llm, b"img", prompt, "required")
    mock_llm.call.assert_not_called()


# -- process_page ----------------------------------------------------


def test_process_page_full_dpi():
    """Successful extraction at full DPI."""
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_vision_response("Title", "Content")
    prompt = _make_prompt()

    result = process_page(mock_llm, b"img", 1, 5, prompt)
    assert result.method == "full_dpi"
    assert result.page_title == "Title"
    assert result.content == "Content"


def test_process_page_high_detail():
    """Falls back to high detail when full DPI fails."""
    mock_llm = MagicMock()
    mock_llm.call.side_effect = [
        ValueError("full failed"),
        _make_vision_response("Orig", "OrigContent"),
    ]
    prompt = _make_prompt()

    result = process_page(mock_llm, b"img", 1, 5, prompt)
    assert result.method == "high_detail"
    assert result.page_title == "Orig"


@patch(
    "ingestion.processors.pptx.processor.shrink_image",
    return_value=b"small",
)
def test_process_page_half_dpi(_mock_shrink):
    """Falls back to half DPI when full and high detail fail."""
    mock_llm = MagicMock()
    mock_llm.call.side_effect = [
        ValueError("full failed"),
        ValueError("original failed"),
        _make_vision_response("Half", "HalfContent"),
    ]
    prompt = _make_prompt()

    result = process_page(mock_llm, b"img", 1, 5, prompt)
    assert result.method == "half_dpi"
    assert result.page_title == "Half"


@patch(
    "ingestion.processors.pptx.processor.split_image",
    return_value=(b"top", b"bot", "vertical"),
)
@patch(
    "ingestion.processors.pptx.processor.shrink_image",
    return_value=b"small",
)
def test_process_page_split_halves(_mock_shrink, _mock_split):
    """Falls back to split halves when all prior steps fail."""
    mock_llm = MagicMock()
    mock_llm.call.side_effect = [
        ValueError("full failed"),
        ValueError("original failed"),
        ValueError("half failed"),
        _make_vision_response("Top", "TopContent"),
        _make_vision_response("Bot", "BotContent"),
    ]
    prompt = _make_prompt()

    result = process_page(mock_llm, b"img", 1, 5, prompt)
    assert result.method == "split_halves"
    assert result.page_title == "Top"
    assert "TopContent" in result.content
    assert "BotContent" in result.content


@patch(
    "ingestion.processors.pptx.processor.split_image",
    return_value=(b"top", b"bot", "vertical"),
)
@patch(
    "ingestion.processors.pptx.processor.shrink_image",
    return_value=b"small",
)
def test_process_page_all_attempts_fail(_mock_shrink, _mock_split):
    """Raises when all fallback attempts fail."""
    mock_llm = MagicMock()
    mock_llm.call.side_effect = ValueError("always fails")
    prompt = _make_prompt()

    with pytest.raises(RuntimeError, match="failed all extraction attempts"):
        process_page(mock_llm, b"img", 1, 5, prompt)


def test_process_page_stamps_page_number():
    """Sets the correct 1-indexed page number on the result."""
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_vision_response()
    prompt = _make_prompt()

    result = process_page(mock_llm, b"img", 3, 10, prompt)
    assert result.page_number == 3


def test_process_page_high_detail_passes_detail_param():
    """High-detail fallback sends detail='high' to the LLM."""
    mock_llm = MagicMock()
    mock_llm.call.side_effect = [
        ValueError("full failed"),
        _make_vision_response("T", "C"),
    ]
    prompt = _make_prompt()

    result = process_page(mock_llm, b"img", 1, 1, prompt)
    assert result.method == "high_detail"
    messages = mock_llm.call.call_args[1]["messages"]
    image_block = messages[1]["content"][0]
    assert image_block["image_url"]["detail"] == "high"


@patch(
    "ingestion.processors.pptx.processor.shrink_image",
    return_value=b"small",
)
def test_process_page_falls_back_on_bad_request(_mock_shrink):
    """BadRequestError still triggers the fallback chain."""
    mock_llm = MagicMock()
    mock_llm.call.side_effect = [
        _make_bad_request_error(),
        _make_vision_response("Orig", "OrigContent"),
    ]
    prompt = _make_prompt()

    result = process_page(mock_llm, b"img", 1, 5, prompt)
    assert result.method == "high_detail"
    assert result.page_title == "Orig"


@patch(
    "ingestion.processors.pptx.processor.shrink_image",
    return_value=b"small",
)
def test_process_page_falls_back_on_malformed_tool_response(
    _mock_shrink,
):
    """Malformed tool payloads fall back to the next strategy."""
    mock_llm = MagicMock()
    mock_llm.call.side_effect = [
        {"choices": [{"message": {"tool_calls": []}}]},
        _make_vision_response("Orig", "OrigContent"),
    ]
    prompt = _make_prompt()

    result = process_page(mock_llm, b"img", 1, 5, prompt)
    assert result.method == "high_detail"
    assert result.page_title == "Orig"
