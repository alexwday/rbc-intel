"""Shared test fixtures and helpers."""

from unittest.mock import MagicMock

from ingestion.utils.file_types import (
    ExtractionResult,
    FileRecord,
    PageResult,
)


def make_file_record(**overrides):
    """Build a FileRecord with sensible defaults."""
    defaults = {
        "data_source": "src",
        "filter_1": "",
        "filter_2": "",
        "filter_3": "",
        "filename": "test.pdf",
        "filetype": "pdf",
        "file_size": 1024,
        "date_last_modified": 1700000000.0,
        "file_hash": "",
        "file_path": "/data/src/test.pdf",
    }
    defaults.update(overrides)
    return FileRecord(**defaults)


def make_extraction_prompt() -> dict:
    """Build a minimal extraction prompt. Params: none. Returns: dict."""
    return {
        "stage": "extraction",
        "system_prompt": "You are a test agent.",
        "user_prompt": "Extract content.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "extract_page_content",
                    "parameters": {},
                },
            }
        ],
        "tool_choice": "required",
    }


def make_rendered_pdf(total_pages: int):
    """Build a mock rendered PDF context manager.

    Params: total_pages. Returns: MagicMock.
    """
    rendered_pdf = MagicMock()
    rendered_pdf.total_pages = total_pages
    context_manager = MagicMock()
    context_manager.__enter__.return_value = rendered_pdf
    context_manager.__exit__.return_value = False
    return context_manager


def make_extraction_result(**overrides):
    """Build a minimal ExtractionResult for testing."""
    defaults = {
        "file_path": "/data/src/test.pdf",
        "filetype": "pdf",
        "pages": [PageResult(1, "T", "C", "full_dpi")],
        "total_pages": 1,
        "pages_succeeded": 1,
        "pages_failed": 0,
    }
    defaults.update(overrides)
    return ExtractionResult(**defaults)


def make_partial_failure_results(error_message: str = "vision timeout"):
    """Build a standard partial-failure page sequence.

    Params: error_message. Returns: list.
    """
    return [
        PageResult(1, "Title 1", "Content 1", "full_dpi"),
        RuntimeError(error_message),
        PageResult(3, "Title 3", "Content 3", "full_dpi"),
    ]
