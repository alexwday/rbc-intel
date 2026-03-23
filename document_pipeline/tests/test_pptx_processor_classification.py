"""Tests for PPTX slide classification helpers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ingestion.processors.pptx import (
    _VALID_SLIDE_TYPES,
    _classify_slide,
    _classify_slide_with_retry,
    _parse_slide_classification,
)


def _make_slide_classification_response(**overrides):
    """Build an LLM response for slide classification.

    Params: overrides (kwargs). Returns: dict.
    """
    defaults = {
        "slide_type": "content_slide",
        "contains_chart": False,
        "contains_dashboard": False,
        "contains_comparison_layout": False,
        "has_dense_visual_content": False,
        "confidence": 0.9,
        "rationale": "Standard content slide.",
    }
    defaults.update(overrides)
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "classify_slide",
                                "arguments": json.dumps(defaults),
                            }
                        }
                    ]
                }
            }
        ],
    }


def _make_classification_prompt():
    """Build a minimal classification prompt.

    Params: none. Returns: dict.
    """
    return {
        "stage": "pptx_classification",
        "system_prompt": "You are a presentation content analyst.",
        "user_prompt": "Classify this slide.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "classify_slide",
                    "parameters": {},
                },
            }
        ],
        "tool_choice": "required",
    }


# ── _parse_slide_classification ──────────────────────────


@pytest.mark.parametrize("slide_type", sorted(_VALID_SLIDE_TYPES))
def test_parse_valid_slide_types(slide_type):
    """Each valid slide type is parsed without fallback."""
    response = _make_slide_classification_response(slide_type=slide_type)
    result = _parse_slide_classification(response)
    assert result["slide_type_guess"] == slide_type


def test_parse_unknown_slide_type_falls_back():
    """Unknown slide_type falls back to content_slide."""
    response = _make_slide_classification_response(slide_type="unknown_type")
    result = _parse_slide_classification(response)
    assert result["slide_type_guess"] == "content_slide"


def test_parse_missing_slide_type_defaults():
    """Missing slide_type defaults to content_slide."""
    response = _make_slide_classification_response()
    raw = json.loads(
        response["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
    )
    del raw["slide_type"]
    response["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] = json.dumps(raw)

    result = _parse_slide_classification(response)
    assert result["slide_type_guess"] == "content_slide"


def test_parse_boolean_flags_all_true():
    """Boolean flags parsed correctly when all True."""
    response = _make_slide_classification_response(
        contains_chart=True,
        contains_dashboard=True,
        contains_comparison_layout=True,
        has_dense_visual_content=True,
    )
    result = _parse_slide_classification(response)
    assert result["contains_chart"] is True
    assert result["contains_dashboard"] is True
    assert result["contains_comparison_layout"] is True
    assert result["has_dense_visual_content"] is True


def test_parse_missing_choices_raises():
    """Missing choices raises ValueError."""
    with pytest.raises(ValueError, match="missing choices"):
        _parse_slide_classification({})


def test_parse_empty_choices_raises():
    """Empty choices list raises ValueError."""
    with pytest.raises(ValueError, match="missing choices"):
        _parse_slide_classification({"choices": []})


def test_parse_missing_tool_calls_raises():
    """Missing tool_calls raises ValueError."""
    response = {"choices": [{"message": {}}]}
    with pytest.raises(ValueError, match="missing tool calls"):
        _parse_slide_classification(response)


def test_parse_missing_message_raises():
    """Missing message raises ValueError."""
    response = {"choices": [{}]}
    with pytest.raises(ValueError, match="missing message"):
        _parse_slide_classification(response)


def test_parse_missing_function_raises():
    """Missing function payload raises ValueError."""
    response = {"choices": [{"message": {"tool_calls": [{}]}}]}
    with pytest.raises(ValueError, match="missing function"):
        _parse_slide_classification(response)


def test_parse_missing_arguments_raises():
    """Missing arguments string raises ValueError."""
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "classify_slide"}}]
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="missing function arg"):
        _parse_slide_classification(response)


# ── _classify_slide ──────────────────────────────────────


def test_classify_slide_valid():
    """Valid classification returns correct slide_type_guess."""
    llm = MagicMock()
    llm.call.return_value = _make_slide_classification_response(
        slide_type="chart_slide"
    )
    prompt = _make_classification_prompt()

    result = _classify_slide(
        llm, prompt, page_number=3, page_title="Revenue", content="Q3"
    )
    assert result["slide_type_guess"] == "chart_slide"

    call_args = llm.call.call_args
    messages = call_args.kwargs["messages"]
    user_content = messages[-1]["content"]
    assert "Slide number: 3" in user_content
    assert "Revenue" in user_content
    assert "Q3" in user_content


# ── _classify_slide_with_retry ───────────────────────────


@patch("ingestion.processors.pptx.time.sleep")
def test_retry_succeeds_first_attempt(mock_sleep):
    """Succeeds on first attempt without retrying."""
    llm = MagicMock()
    llm.call.return_value = _make_slide_classification_response()
    prompt = _make_classification_prompt()

    result = _classify_slide_with_retry(llm, prompt, 1, "Title", "Content")
    assert result["slide_type_guess"] == "content_slide"
    mock_sleep.assert_not_called()


@patch("ingestion.processors.pptx.time.sleep")
def test_retry_succeeds_after_failure(mock_sleep):
    """Retries on ValueError then succeeds."""
    llm = MagicMock()
    llm.call.side_effect = [
        ValueError("bad response"),
        _make_slide_classification_response(slide_type="agenda_slide"),
    ]
    prompt = _make_classification_prompt()

    result = _classify_slide_with_retry(llm, prompt, 2, "Agenda", "Items")
    assert result["slide_type_guess"] == "agenda_slide"
    mock_sleep.assert_called_once()


@patch("ingestion.processors.pptx.time.sleep")
def test_retry_exhausted_raises_runtime_error(mock_sleep):
    """Exhausts retries and raises RuntimeError."""
    llm = MagicMock()
    llm.call.side_effect = ValueError("always fails")
    prompt = _make_classification_prompt()

    with pytest.raises(RuntimeError, match="failed after"):
        _classify_slide_with_retry(llm, prompt, 5, "Broken", "Bad")
    assert mock_sleep.call_count == 2
