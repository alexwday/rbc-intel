"""LLM-based content chunking for oversized pages."""

import json
import logging
from importlib import import_module
from typing import Any, List

from ..connections.llm import LLMClient
from .content_types import ContentChunk
from .prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class _FallbackEncoding:
    """Minimal tokenizer when tiktoken is unavailable."""

    def encode(self, text: str) -> list[int]:
        """Split text into coarse token IDs."""
        return list(range(max(1, len(text.split()))))

    def decode(self, tokens: list[int]) -> str:
        """Decode is unsupported on the fallback tokenizer."""
        raise NotImplementedError("Fallback tokenizer does not support decode")


class _FallbackTiktoken:
    """Compatibility shim for tiktoken methods."""

    def encoding_for_model(self, _model: str) -> _FallbackEncoding:
        """Return a coarse tokenizer. Params: _model. Returns: enc."""
        return _FallbackEncoding()

    def get_encoding(self, _name: str) -> _FallbackEncoding:
        """Return a coarse tokenizer. Params: _name. Returns: enc."""
        return _FallbackEncoding()


def _load_tiktoken() -> Any:
    """Import tiktoken or fall back to word-count estimator."""
    try:
        return import_module("tiktoken")
    except ImportError:
        return _FallbackTiktoken()


_tiktoken = _load_tiktoken()
_EMBEDDING_MODEL = "text-embedding-3-large"


def _get_encoder() -> Any:
    """Get a tokenizer for the embedding model."""
    try:
        return _tiktoken.encoding_for_model(_EMBEDDING_MODEL)
    except KeyError:
        return _tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Count tokens using the embedding model tokenizer.

    Params:
        text: Input text to tokenize

    Returns:
        int — number of tokens

    Example:
        >>> count_tokens("Hello world")
        2
    """
    return len(_get_encoder().encode(text))


def _number_lines(content: str) -> str:
    """Add line numbers for breakpoint identification."""
    lines = content.splitlines()
    if not lines:
        return ""
    numbered = [f"{i}: {line}" for i, line in enumerate(lines, 1)]
    return "\n".join(numbered)


def _parse_breakpoints(response: dict[str, Any]) -> list[int]:
    """Extract breakpoint line numbers from LLM tool response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message")

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("LLM response missing tool calls")

    function_data = tool_calls[0].get("function")
    if not isinstance(function_data, dict):
        raise ValueError("LLM response missing function data")

    arguments = function_data.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("LLM response missing arguments")

    parsed = json.loads(arguments)
    breakpoints = parsed.get("breakpoints")
    if not isinstance(breakpoints, list):
        raise ValueError("Response missing breakpoints list")

    return sorted(int(bp) for bp in breakpoints)


def _call_chunking_llm(content: str, llm: LLMClient) -> list[int]:
    """Send numbered content to LLM for breakpoint detection."""
    prompt = load_prompt("content_chunking")
    numbered = _number_lines(content)
    messages: list[dict[str, str]] = []
    if prompt.get("system_prompt"):
        messages.append(
            {
                "role": "system",
                "content": prompt["system_prompt"],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": (f"{prompt['user_prompt']}\n\n{numbered}"),
        }
    )
    response = llm.call(
        messages=messages,
        stage=prompt["stage"],
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
    )
    return _parse_breakpoints(response)


def _split_at_breakpoints(content: str, breakpoints: list[int]) -> List[str]:
    """Split content into chunks at given line numbers."""
    lines = content.splitlines(keepends=True)
    chunks: list[str] = []
    start = 0
    for bp in breakpoints:
        idx = bp - 1
        if 0 < idx < len(lines):
            chunk = "".join(lines[start:idx]).rstrip()
            if chunk.strip():
                chunks.append(chunk)
            start = idx
    remainder = "".join(lines[start:]).rstrip()
    if remainder.strip():
        chunks.append(remainder)
    return chunks


def _find_blank_line_near_midpoint(
    lines: list[str],
) -> int:
    """Find blank line nearest to midpoint. Returns: line index."""
    mid = len(lines) // 2
    for offset in range(len(lines)):
        for candidate in (mid + offset, mid - offset):
            if 0 < candidate < len(lines):
                if not lines[candidate].strip():
                    return candidate
    return mid


def _split_at_blank_line(content: str, max_tokens: int) -> List[str]:
    """Emergency split at nearest blank line to midpoint.

    Recursively splits until all chunks are within the
    token limit. Falls back to midpoint split when no
    blank lines are available.

    Params:
        content: Text to split
        max_tokens: Maximum tokens per chunk

    Returns:
        list[str] — content pieces within token limit

    Example:
        >>> parts = _split_at_blank_line(big_text, 8191)
        >>> all(count_tokens(p) <= 8191 for p in parts)
        True
    """
    if count_tokens(content) <= max_tokens:
        return [content]

    lines = content.splitlines(keepends=True)
    if len(lines) <= 1:
        return [content]

    split_idx = _find_blank_line_near_midpoint(lines)

    first = "".join(lines[:split_idx]).rstrip()
    second = "".join(lines[split_idx:]).rstrip()

    chunks: list[str] = []
    for part in (first, second):
        if not part.strip():
            continue
        if count_tokens(part) > max_tokens:
            chunks.extend(_split_at_blank_line(part, max_tokens))
        else:
            chunks.append(part)
    return chunks


def chunk_content(
    content: str,
    llm: LLMClient,
    max_chunk_tokens: int = 8191,
) -> List[ContentChunk]:
    """Split content into embeddable chunks if oversized.

    If content fits within the token limit, returns a single
    chunk with no LLM call. Otherwise sends numbered lines
    to the LLM for breakpoint detection. Falls back to blank
    line splitting if the LLM call fails.

    Params:
        content: Page content to chunk
        llm: LLMClient instance for breakpoint detection
        max_chunk_tokens: Maximum tokens per chunk

    Returns:
        list[ContentChunk] — one or more chunks

    Example:
        >>> chunks = chunk_content(small_text, llm)
        >>> len(chunks)
        1
    """
    token_count = count_tokens(content)

    if token_count <= max_chunk_tokens:
        return [
            ContentChunk(
                chunk_index=0,
                content=content,
                token_count=token_count,
            )
        ]

    try:
        breakpoints = _call_chunking_llm(content, llm)
        chunks_text = _split_at_breakpoints(content, breakpoints)
    except (
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        KeyError,
    ):
        logger.warning("LLM chunking failed, using blank-line fallback")
        chunks_text = _split_at_blank_line(content, max_chunk_tokens)

    final_chunks: list[str] = []
    for chunk_text in chunks_text:
        if count_tokens(chunk_text) > max_chunk_tokens:
            final_chunks.extend(
                _split_at_blank_line(chunk_text, max_chunk_tokens)
            )
        else:
            final_chunks.append(chunk_text)

    return [
        ContentChunk(
            chunk_index=i,
            content=text,
            token_count=count_tokens(text),
        )
        for i, text in enumerate(final_chunks)
    ]
