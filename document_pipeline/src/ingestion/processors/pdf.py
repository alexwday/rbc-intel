"""PDF processor — renders pages to PNG, extracts via LLM vision."""

import logging
import time
from pathlib import Path
from typing import Any, List

import openai

from ..connections.llm import LLMClient
from ..utils.config import get_vision_dpi_scale
from ..utils.file_types import ExtractionResult, PageResult
from ..utils.prompt_loader import load_prompt
from .vision import (
    open_rendered_pdf,
    parse_tool_arguments,
    process_page,
    render_page,
)

logger = logging.getLogger(__name__)

_CONTEXT_MAX_CHARS = 800
_CONTEXT_MIN_CHARS = 50
_CLASSIFICATION_MAX_RETRIES = 5
_CLASSIFICATION_RETRY_DELAY_S = 2.0
_RETRYABLE_CLASSIFICATION_ERRORS = (
    openai.OpenAIError,
    RuntimeError,
    ValueError,
)


def _build_context_prompt(prompt: dict, previous_content: str) -> dict:
    """Augment the PDF prompt with prior-page extraction context."""
    content = previous_content
    if len(content) > _CONTEXT_MAX_CHARS:
        tail = content[-_CONTEXT_MAX_CHARS:]
        newline_index = tail.find("\n")
        if (
            newline_index != -1
            and len(tail) - newline_index - 1 >= _CONTEXT_MIN_CHARS
        ):
            content = tail[newline_index + 1 :]
        else:
            content = tail

    augmented = dict(prompt)
    augmented["user_prompt"] = (
        "CONTEXT - The previous PDF page ended with this extracted content. "
        "Use it to detect continued prose, tables, list items, footnotes, "
        "and repeated page furniture.\n\n"
        "---\n"
        f"{content}\n"
        "---\n\n" + prompt["user_prompt"]
    )
    return augmented


def _truncate_tail(content: str) -> str:
    """Truncate content to the last _CONTEXT_MAX_CHARS at a newline boundary."""
    if len(content) <= _CONTEXT_MAX_CHARS:
        return content
    tail = content[-_CONTEXT_MAX_CHARS:]
    newline_index = tail.find("\n")
    if (
        newline_index != -1
        and len(tail) - newline_index - 1 >= _CONTEXT_MIN_CHARS
    ):
        return tail[newline_index + 1 :]
    return tail


def _truncate_head_tail(content: str) -> str:
    """Truncate content to first and last _CONTEXT_MAX_CHARS for classification."""
    if len(content) <= _CONTEXT_MAX_CHARS * 2:
        return content
    head = content[:_CONTEXT_MAX_CHARS]
    tail = content[-_CONTEXT_MAX_CHARS:]
    return head + "\n\n... (middle content omitted) ...\n\n" + tail


def _default_page_metadata() -> dict[str, bool]:
    """Return default metadata for the first page (no previous content)."""
    return {
        "continued_from_previous_page": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "table_continuation_detected": False,
        "contains_page_furniture": False,
    }


def _classify_page_continuation(
    llm: LLMClient,
    prompt: dict[str, Any],
    current_content: str,
    previous_content: str,
) -> dict[str, bool]:
    """Classify page continuation and furniture via LLM.

    Params:
        llm: LLMClient instance
        prompt: Loaded page_continuation_classification prompt
        current_content: Extracted markdown from the current page
        previous_content: Extracted markdown from the previous page

    Returns:
        dict with continuation and furniture detection flags
    """
    previous_tail = _truncate_tail(previous_content)
    current_summary = _truncate_head_tail(current_content)

    messages = []
    if prompt.get("system_prompt"):
        messages.append({"role": "system", "content": prompt["system_prompt"]})
    messages.append(
        {
            "role": "user",
            "content": (
                f"{prompt['user_prompt']}\n\n"
                f"## Previous page content (tail)\n"
                f"---\n{previous_tail}\n---\n\n"
                f"## Current page content\n"
                f"---\n{current_summary}\n---"
            ),
        }
    )
    response = llm.call(
        messages=messages,
        stage=prompt["stage"],
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
    )
    parsed = parse_tool_arguments(response)

    continued = bool(parsed.get("continued_from_previous_page", False))
    table_cont = bool(parsed.get("table_continuation_detected", False))
    header_rep = bool(parsed.get("repeated_header_detected", False))
    footer_rep = bool(parsed.get("repeated_footer_detected", False))

    return {
        "continued_from_previous_page": continued,
        "repeated_header_detected": header_rep,
        "repeated_footer_detected": footer_rep,
        "table_continuation_detected": table_cont,
        "contains_page_furniture": header_rep or footer_rep,
    }


def _classify_continuation_with_retry(
    llm: LLMClient,
    prompt: dict[str, Any],
    current_content: str,
    previous_content: str,
) -> dict[str, bool]:
    """Classify page continuation with retry logic."""
    for attempt in range(1, _CLASSIFICATION_MAX_RETRIES + 1):
        try:
            return _classify_page_continuation(
                llm,
                prompt,
                current_content,
                previous_content,
            )
        except _RETRYABLE_CLASSIFICATION_ERRORS as exc:
            if attempt == _CLASSIFICATION_MAX_RETRIES:
                raise RuntimeError(
                    f"Page continuation classification failed after "
                    f"{_CLASSIFICATION_MAX_RETRIES} attempts: {exc}"
                ) from exc
            wait = _CLASSIFICATION_RETRY_DELAY_S * attempt
            logger.warning(
                "Page continuation classification retry %d/%d "
                "after %.1fs: %s",
                attempt,
                _CLASSIFICATION_MAX_RETRIES,
                wait,
                exc,
            )
            time.sleep(wait)
    raise RuntimeError("Page continuation classification exited retry loop")


def process_pdf(file_path: str, llm: LLMClient) -> ExtractionResult:
    """Extract content from a PDF file via vision processing.

    Renders all pages to PNG, processes each sequentially
    through the LLM vision pipeline, classifies page
    continuation metadata via a separate LLM call, and
    returns an ExtractionResult with per-page results.

    Params:
        file_path: Absolute path to the PDF file
        llm: LLMClient instance

    Returns:
        ExtractionResult with page-level extraction details

    Example:
        >>> result = process_pdf("/data/report.pdf", llm)
        >>> result.total_pages
        10
    """
    pdf_path = Path(file_path)
    extraction_prompt = load_prompt("pdf_extraction_vision")
    classification_prompt = load_prompt("page_continuation_classification")

    pages: List[PageResult] = []
    previous_content = ""
    with open_rendered_pdf(pdf_path, get_vision_dpi_scale()) as rendered:
        for page_num in range(1, rendered.total_pages + 1):
            page_prompt = (
                _build_context_prompt(extraction_prompt, previous_content)
                if previous_content
                else extraction_prompt
            )
            try:
                page_result = process_page(
                    llm,
                    render_page(rendered, page_num),
                    page_num,
                    rendered.total_pages,
                    page_prompt,
                )
                if previous_content:
                    page_result.metadata.update(
                        _classify_continuation_with_retry(
                            llm,
                            classification_prompt,
                            page_result.content,
                            previous_content,
                        )
                    )
                else:
                    page_result.metadata.update(_default_page_metadata())
                previous_content = page_result.content
            except (RuntimeError, ValueError, OSError) as exc:
                raise RuntimeError(
                    "PDF extraction failed for "
                    f"'{pdf_path.name}' on page "
                    f"{page_num}/{rendered.total_pages}: "
                    f"{exc}"
                ) from exc
            logger.info(
                "PDF page %d/%d extracted",
                page_num,
                rendered.total_pages,
            )
            pages.append(page_result)

    return ExtractionResult(
        file_path=file_path,
        filetype="pdf",
        pages=pages,
        total_pages=len(pages),
        pages_succeeded=len(pages),
        pages_failed=0,
    )
