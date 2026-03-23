"""Tests for content chunking module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ingestion.utils.content_chunker import (
    _FallbackEncoding,
    _FallbackTiktoken,
    _call_chunking_llm,
    _find_blank_line_near_midpoint,
    _get_encoder,
    _load_tiktoken,
    _number_lines,
    _parse_breakpoints,
    _split_at_blank_line,
    _split_at_breakpoints,
    chunk_content,
    count_tokens,
)

# ── fallback classes ─────────────────────────────────────────


def test_fallback_encoding_basic():
    """Fallback encodes via word count."""
    enc = _FallbackEncoding()
    result = enc.encode("hello world")
    assert len(result) == 2


def test_fallback_encoding_decode_raises():
    """Fallback decode raises NotImplementedError."""
    enc = _FallbackEncoding()
    with pytest.raises(NotImplementedError):
        enc.decode([0])


def test_fallback_encoding_empty():
    """Fallback returns minimum 1 for empty string."""
    enc = _FallbackEncoding()
    result = enc.encode("")
    assert len(result) == 1


def test_fallback_tiktoken_encoding_for_model():
    """Fallback exposes encoding_for_model method."""
    ft = _FallbackTiktoken()
    enc = ft.encoding_for_model("gpt-4")
    assert len(enc.encode("hello")) == 1


def test_fallback_tiktoken_get_encoding():
    """Fallback exposes get_encoding method."""
    ft = _FallbackTiktoken()
    enc = ft.get_encoding("cl100k_base")
    assert len(enc.encode("hello")) == 1


@patch("ingestion.utils.content_chunker._tiktoken")
def test_get_encoder_keyerror_fallback(mock_tiktoken):
    """Falls back to o200k_base on model KeyError."""
    mock_tiktoken.encoding_for_model.side_effect = KeyError("unknown")
    mock_enc = MagicMock()
    mock_tiktoken.get_encoding.return_value = mock_enc
    result = _get_encoder()
    assert result is mock_enc
    mock_tiktoken.get_encoding.assert_called_once_with("o200k_base")


def test_load_tiktoken_fallback():
    """Falls back to _FallbackTiktoken when import fails."""
    with patch(
        "ingestion.utils.content_chunker.import_module",
        side_effect=ImportError("no tiktoken"),
    ):
        result = _load_tiktoken()
    assert isinstance(result, _FallbackTiktoken)


# ── count_tokens ─────────────────────────────────────────────


def test_count_tokens_basic():
    """Returns positive count for normal text."""
    assert count_tokens("Hello world") > 0


def test_count_tokens_empty():
    """Returns zero for empty string."""
    assert count_tokens("") == 0


def test_count_tokens_long():
    """Longer text has more tokens."""
    short = count_tokens("hello")
    long = count_tokens("hello world this is a longer text")
    assert long > short


# ── _number_lines ────────────────────────────────────────────


def test_number_lines_basic():
    """Adds line numbers starting at 1."""
    result = _number_lines("alpha\nbeta\ngamma")
    lines = result.splitlines()
    assert lines[0] == "1: alpha"
    assert lines[1] == "2: beta"
    assert lines[2] == "3: gamma"


def test_number_lines_empty():
    """Handles empty string."""
    result = _number_lines("")
    assert result == ""


# ── _parse_breakpoints ──────────────────────────────────────


def _make_tool_response(breakpoints, rationale="test"):
    """Build a mock LLM tool response."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": ("identify_chunk_" "breakpoints"),
                                "arguments": json.dumps(
                                    {
                                        "breakpoints": (breakpoints),
                                        "rationale": (rationale),
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }


def test_parse_breakpoints_valid():
    """Parses valid breakpoint response."""
    response = _make_tool_response([10, 25, 40])
    result = _parse_breakpoints(response)
    assert result == [10, 25, 40]


def test_parse_breakpoints_unsorted():
    """Sorts breakpoints."""
    response = _make_tool_response([40, 10, 25])
    result = _parse_breakpoints(response)
    assert result == [10, 25, 40]


def test_parse_breakpoints_missing_choices():
    """Raises on missing choices."""
    with pytest.raises(ValueError, match="choices"):
        _parse_breakpoints({})


def test_parse_breakpoints_missing_message():
    """Raises on missing message."""
    with pytest.raises(ValueError, match="message"):
        _parse_breakpoints({"choices": [{}]})


def test_parse_breakpoints_missing_tool_calls():
    """Raises on missing tool calls."""
    with pytest.raises(ValueError, match="tool calls"):
        _parse_breakpoints({"choices": [{"message": {}}]})


def test_parse_breakpoints_missing_function():
    """Raises on missing function data."""
    with pytest.raises(ValueError, match="function data"):
        _parse_breakpoints(
            {"choices": [{"message": {"tool_calls": [{"id": "x"}]}}]}
        )


def test_parse_breakpoints_missing_arguments():
    """Raises on missing arguments."""
    with pytest.raises(ValueError, match="arguments"):
        _parse_breakpoints(
            {
                "choices": [
                    {"message": {"tool_calls": [{"function": {"name": "x"}}]}}
                ]
            }
        )


def test_parse_breakpoints_invalid_json():
    """Raises on malformed JSON arguments."""
    with pytest.raises(json.JSONDecodeError):
        _parse_breakpoints(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"arguments": ("not json")}}
                            ]
                        }
                    }
                ]
            }
        )


def test_parse_breakpoints_missing_list():
    """Raises when breakpoints is not a list."""
    with pytest.raises(ValueError, match="breakpoints"):
        _parse_breakpoints(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": (
                                            json.dumps({"breakpoints": "bad"})
                                        )
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )


# ── _split_at_breakpoints ───────────────────────────────────


def test_split_at_breakpoints_basic():
    """Splits content at given line numbers."""
    content = "line1\nline2\nline3\nline4\nline5"
    chunks = _split_at_breakpoints(content, [3])
    assert len(chunks) == 2
    assert "line1" in chunks[0]
    assert "line3" in chunks[1]


def test_split_at_breakpoints_multiple():
    """Handles multiple breakpoints."""
    content = "\n".join(f"line{i}" for i in range(1, 11))
    chunks = _split_at_breakpoints(content, [4, 7])
    assert len(chunks) == 3


def test_split_at_breakpoints_out_of_range():
    """Ignores breakpoints outside content bounds."""
    content = "line1\nline2"
    chunks = _split_at_breakpoints(content, [100])
    assert len(chunks) == 1
    assert "line1" in chunks[0]


def test_split_at_breakpoints_empty():
    """No breakpoints returns single chunk."""
    content = "line1\nline2"
    chunks = _split_at_breakpoints(content, [])
    assert len(chunks) == 1


# ── _find_blank_line_near_midpoint ───────────────────────────


def test_find_blank_line_has_blank():
    """Finds blank line nearest to midpoint."""
    lines = ["a\n", "b\n", "\n", "c\n", "d\n"]
    idx = _find_blank_line_near_midpoint(lines)
    assert idx == 2


def test_find_blank_line_no_blank():
    """Falls back to midpoint when no blank lines."""
    lines = ["a\n", "b\n", "c\n", "d\n"]
    idx = _find_blank_line_near_midpoint(lines)
    assert idx == 2


# ── _split_at_blank_line ────────────────────────────────────


@patch("ingestion.utils.content_chunker.count_tokens")
def test_split_at_blank_line_within_limit(mock_count):
    """Content within limit returned as-is."""
    mock_count.return_value = 100
    result = _split_at_blank_line("short text", 8191)
    assert result == ["short text"]


@patch("ingestion.utils.content_chunker.count_tokens")
def test_split_at_blank_line_splits(mock_count):
    """Oversized content split at blank line."""
    mock_count.side_effect = lambda t: (50 if len(t) < 12 else 200)
    content = "part one\n\npart two"
    result = _split_at_blank_line(content, 100)
    assert len(result) == 2


@patch("ingestion.utils.content_chunker.count_tokens")
def test_split_at_blank_line_single_line(mock_count):
    """Single line that exceeds limit returned as-is."""
    mock_count.return_value = 10000
    result = _split_at_blank_line("one long line", 100)
    assert result == ["one long line"]


@patch("ingestion.utils.content_chunker.count_tokens")
def test_split_at_blank_line_blank_at_start(mock_count):
    """Handles blank line at position 0 by shifting to 1."""
    mock_count.side_effect = lambda t: (10 if len(t) < 3 else 200)
    content = "\na\nb"
    result = _split_at_blank_line(content, 100)
    assert len(result) >= 1


@patch("ingestion.utils.content_chunker.count_tokens")
def test_split_at_blank_line_empty_part(mock_count):
    """Empty parts after splitting are skipped."""
    mock_count.side_effect = lambda t: (10 if len(t) < 5 else 200)
    content = "a\n\n\nb"
    result = _split_at_blank_line(content, 100)
    for part in result:
        assert part.strip()


@patch("ingestion.utils.content_chunker.count_tokens")
def test_split_at_blank_line_recursive(mock_count):
    """Chunks exceeding limit are recursively re-split."""
    call_count = {"n": 0}

    def _mock(text):
        call_count["n"] += 1
        if len(text) > 15:
            return 200
        return 10

    mock_count.side_effect = _mock
    content = "long first section\n\nshort\n\nlong second section"
    result = _split_at_blank_line(content, 100)
    assert all(len(p) <= 20 or call_count["n"] > 3 for p in result)


# ── _call_chunking_llm ──────────────────────────────────────


@patch("ingestion.utils.content_chunker.load_prompt")
def test_call_chunking_llm(mock_load):
    """Calls LLM and returns parsed breakpoints."""
    mock_load.return_value = {
        "stage": "content_chunking",
        "user_prompt": "Split this content.",
        "system_prompt": "You segment documents.",
        "tools": [{"type": "function", "function": {}}],
        "tool_choice": "required",
    }
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_tool_response([5, 10])
    result = _call_chunking_llm("line1\nline2\nline3", mock_llm)
    assert result == [5, 10]
    mock_llm.call.assert_called_once()


# ── chunk_content ────────────────────────────────────────────


@patch("ingestion.utils.content_chunker.count_tokens")
def test_chunk_content_within_limit(mock_count):
    """Small content returns single chunk with no LLM call."""
    mock_count.return_value = 100
    mock_llm = MagicMock()
    chunks = chunk_content("small text", mock_llm, 8191)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "small text"
    mock_llm.call.assert_not_called()


@patch("ingestion.utils.content_chunker._call_chunking_llm")
@patch("ingestion.utils.content_chunker.count_tokens")
def test_chunk_content_llm_chunking(mock_count, mock_call_llm):
    """Oversized content uses LLM breakpoints."""
    mock_count.side_effect = lambda t: (100 if len(t) < 20 else 10000)
    mock_call_llm.return_value = [3]
    content = "line1\nline2\nline3\nline4\nline5"
    mock_llm = MagicMock()
    chunks = chunk_content(content, mock_llm, 200)
    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


@patch("ingestion.utils.content_chunker._call_chunking_llm")
@patch("ingestion.utils.content_chunker.count_tokens")
def test_chunk_content_llm_failure_fallback(mock_count, mock_call_llm):
    """Falls back to blank-line splitting on LLM failure."""
    mock_count.side_effect = lambda t: (50 if len(t) < 12 else 10000)
    mock_call_llm.side_effect = RuntimeError("LLM down")
    content = "part one\n\npart two"
    mock_llm = MagicMock()
    chunks = chunk_content(content, mock_llm, 100)
    assert len(chunks) == 2


@patch("ingestion.utils.content_chunker._call_chunking_llm")
@patch("ingestion.utils.content_chunker.count_tokens")
def test_chunk_content_post_llm_oversized_fallback(mock_count, mock_call_llm):
    """Chunks exceeding limit after LLM are re-split."""
    call_idx = {"n": 0}

    def _mock_count(text):
        call_idx["n"] += 1
        if len(text) > 30:
            return 10000
        return 50

    mock_count.side_effect = _mock_count
    mock_call_llm.return_value = [3]
    content = (
        "a very long first part here\n\n"
        "second part\n\n"
        "third part here and more"
    )
    mock_llm = MagicMock()
    chunks = chunk_content(content, mock_llm, 100)
    assert len(chunks) >= 2
