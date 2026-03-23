"""Tests for DOCX processor continuation classification helpers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ingestion.processors.docx import (
    _CONTEXT_MAX_CHARS,
    _CONTEXT_MIN_CHARS,
    _classify_continuation_with_retry,
    _classify_page_continuation,
    _truncate_head_tail,
    _truncate_tail,
)


def _make_classification_response(**overrides):
    """Build a mock LLM classification tool-call response.

    Params: overrides (keyword args). Returns: dict.
    """
    defaults = {
        "continued_from_previous_page": False,
        "table_continuation_detected": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
    }
    defaults.update(overrides)
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": ("classify_page_continuation"),
                                "arguments": json.dumps(defaults),
                            }
                        }
                    ]
                }
            }
        ],
    }


def _make_classification_prompt():
    """Build a minimal classification prompt dict.

    Params: none. Returns: dict.
    """
    return {
        "stage": "page_classification",
        "system_prompt": "You classify document page transitions.",
        "user_prompt": "Classify the page continuation.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "classify_page_continuation",
                    "parameters": {},
                },
            }
        ],
        "tool_choice": "required",
    }


# ── _truncate_tail ────────────────────────────────────────────


def test_truncate_tail_short_content():
    """Short content passes through unchanged."""
    content = "a" * _CONTEXT_MAX_CHARS
    assert _truncate_tail(content) == content


def test_truncate_tail_long_content_newline_boundary():
    """Long content truncated at newline when enough remains."""
    padding = "x" * (_CONTEXT_MAX_CHARS + 200)
    tail_after_nl = "y" * _CONTEXT_MIN_CHARS
    content = padding + "\n" + tail_after_nl

    result = _truncate_tail(content)
    assert result == tail_after_nl


def test_truncate_tail_long_content_no_suitable_newline():
    """Long content without suitable newline uses raw tail."""
    content = "z" * (_CONTEXT_MAX_CHARS + 500)
    result = _truncate_tail(content)
    assert len(result) == _CONTEXT_MAX_CHARS
    assert result == content[-_CONTEXT_MAX_CHARS:]


# ── _truncate_head_tail ───────────────────────────────────────


def test_truncate_head_tail_short_content():
    """Short content passes through unchanged."""
    content = "a" * (_CONTEXT_MAX_CHARS * 2)
    assert _truncate_head_tail(content) == content


def test_truncate_head_tail_long_content():
    """Long content gets head + omission marker + tail."""
    content = "H" * _CONTEXT_MAX_CHARS
    content += "M" * 500
    content += "T" * _CONTEXT_MAX_CHARS

    result = _truncate_head_tail(content)

    expected_head = content[:_CONTEXT_MAX_CHARS]
    expected_tail = content[-_CONTEXT_MAX_CHARS:]
    marker = "... (middle content omitted) ..."
    assert result.startswith(expected_head)
    assert result.endswith(expected_tail)
    assert marker in result


# ── _classify_page_continuation ───────────────────────────────


def test_classify_page_continuation_valid_response():
    """Valid LLM response parsed with correct derived fields."""
    llm = MagicMock()
    llm.call.return_value = _make_classification_response(
        continued_from_previous_page=True,
        table_continuation_detected=False,
        repeated_header_detected=True,
        repeated_footer_detected=False,
    )
    prompt = _make_classification_prompt()

    result = _classify_page_continuation(
        llm, prompt, "current page", "previous page"
    )

    assert result["continued_from_previous_page"] is True
    assert result["table_continuation_detected"] is False
    assert result["repeated_header_detected"] is True
    assert result["repeated_footer_detected"] is False
    assert result["contains_page_furniture"] is True


def test_classify_page_continuation_section_continuation():
    """Section continuation set when continued but not table."""
    llm = MagicMock()
    llm.call.return_value = _make_classification_response(
        continued_from_previous_page=True,
        table_continuation_detected=False,
    )
    prompt = _make_classification_prompt()

    result = _classify_page_continuation(llm, prompt, "current", "previous")

    assert result["section_continuation_detected"] is True


def test_classify_page_continuation_no_section_when_table():
    """Section continuation false when table continuation."""
    llm = MagicMock()
    llm.call.return_value = _make_classification_response(
        continued_from_previous_page=True,
        table_continuation_detected=True,
    )
    prompt = _make_classification_prompt()

    result = _classify_page_continuation(llm, prompt, "current", "previous")

    assert result["section_continuation_detected"] is False


def test_classify_page_continuation_truncates_inputs():
    """Truncated content appears in the LLM call messages."""
    llm = MagicMock()
    llm.call.return_value = _make_classification_response()
    prompt = _make_classification_prompt()

    long_previous = "P" * (_CONTEXT_MAX_CHARS + 500)
    long_current = (
        "H" * _CONTEXT_MAX_CHARS + "M" * 1000 + "T" * _CONTEXT_MAX_CHARS
    )

    _classify_page_continuation(llm, prompt, long_current, long_previous)

    call_args = llm.call.call_args
    messages = call_args[1]["messages"]
    user_msg = messages[-1]["content"]

    expected_prev_tail = _truncate_tail(long_previous)
    assert expected_prev_tail in user_msg

    expected_curr = _truncate_head_tail(long_current)
    assert expected_curr in user_msg


# ── _classify_continuation_with_retry ─────────────────────────


@patch("ingestion.processors.docx.time.sleep")
def test_classify_continuation_with_retry_first_attempt(
    mock_sleep,
):
    """Succeeds on first attempt without sleeping."""
    llm = MagicMock()
    llm.call.return_value = _make_classification_response(
        continued_from_previous_page=True,
    )
    prompt = _make_classification_prompt()

    result = _classify_continuation_with_retry(
        llm, prompt, "current", "previous"
    )

    assert result["continued_from_previous_page"] is True
    mock_sleep.assert_not_called()


@patch("ingestion.processors.docx.time.sleep")
def test_classify_continuation_with_retry_transient_then_ok(
    mock_sleep,
):
    """Retries on transient error then succeeds."""
    llm = MagicMock()
    llm.call.side_effect = [
        ValueError("transient"),
        _make_classification_response(),
    ]
    prompt = _make_classification_prompt()

    result = _classify_continuation_with_retry(
        llm, prompt, "current", "previous"
    )

    assert result["continued_from_previous_page"] is False
    mock_sleep.assert_called_once()


@patch("ingestion.processors.docx.time.sleep")
def test_classify_continuation_with_retry_exhausted(
    mock_sleep,
):
    """Exhausts retries and raises RuntimeError."""
    llm = MagicMock()
    llm.call.side_effect = ValueError("always fails")
    prompt = _make_classification_prompt()

    with pytest.raises(RuntimeError, match="failed after"):
        _classify_continuation_with_retry(llm, prompt, "current", "previous")
