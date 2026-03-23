"""DOCX processor — converts to PDF, then extracts via LLM vision."""

import logging
import shutil
import subprocess
import tempfile
import threading
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

_SOFFICE_PATHS = [
    "soffice",
    "libreoffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/usr/bin/libreoffice",
    "/usr/bin/soffice",
]

_CONVERSION_TIMEOUT_S = 120
_CONTEXT_MAX_CHARS = 800
_CONTEXT_MIN_CHARS = 50
_CONVERSION_LOCK = threading.Lock()
_CLASSIFICATION_MAX_RETRIES = 5
_CLASSIFICATION_RETRY_DELAY_S = 2.0
_RETRYABLE_CLASSIFICATION_ERRORS = (
    openai.OpenAIError,
    RuntimeError,
    ValueError,
)


def _find_soffice() -> str:
    """Locate the LibreOffice/soffice binary.

    Checks common paths and returns the first one found.

    Returns:
        str — command name or absolute path to soffice binary

    Example:
        >>> _find_soffice()
        "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    """
    for candidate in _SOFFICE_PATHS:
        if shutil.which(candidate):
            return candidate
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "LibreOffice not found. Install LibreOffice or add "
        "soffice to PATH for DOCX processing."
    )


def _build_user_installation_arg(profile_dir: Path) -> str:
    """Build an isolated LibreOffice profile argument.

    Params: profile_dir (Path). Returns: str.
    """
    return f"-env:UserInstallation={profile_dir.resolve().as_uri()}"


def _format_conversion_error(
    result: subprocess.CompletedProcess,
) -> str:
    """Format LibreOffice subprocess diagnostics.

    Params: result (CompletedProcess). Returns: str.
    """
    details = [f"return code {result.returncode}"]
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        details.append(f"stdout: {stdout}")
    if stderr:
        details.append(f"stderr: {stderr}")
    return "; ".join(details)


def _convert_to_pdf(docx_path: Path, output_dir: Path) -> Path:
    """Convert a DOCX file to PDF using LibreOffice headless.

    Params:
        docx_path: Path to the source DOCX file
        output_dir: Directory to write the converted PDF

    Returns:
        Path to the generated PDF file

    Example:
        >>> pdf = _convert_to_pdf(Path("doc.docx"), Path("/tmp"))
        >>> pdf.suffix
        ".pdf"
    """
    soffice = _find_soffice()
    profile_dir = output_dir / "soffice-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        soffice,
        _build_user_installation_arg(profile_dir),
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(docx_path),
    ]

    logger.info("Converting '%s' to PDF via LibreOffice", docx_path.name)

    with _CONVERSION_LOCK:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_CONVERSION_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"LibreOffice conversion timed out after "
                f"{_CONVERSION_TIMEOUT_S}s for '{docx_path.name}'"
            ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion failed for "
            f"'{docx_path.name}': "
            f"{_format_conversion_error(result)}"
        )

    pdf_path = output_dir / (docx_path.stem + ".pdf")
    if not pdf_path.is_file():
        raise RuntimeError(
            f"LibreOffice conversion produced no output for "
            f"'{docx_path.name}'"
        )

    logger.info(
        "Converted '%s' to PDF (%d bytes)",
        docx_path.name,
        pdf_path.stat().st_size,
    )
    return pdf_path


def _build_context_prompt(prompt: dict, previous_content: str) -> dict:
    """Augment the extraction prompt with previous-page content.

    Truncates previous_content to the last _CONTEXT_MAX_CHARS
    characters (at a newline boundary) and prepends it to the
    user_prompt so the LLM can detect content that continues
    across a page break.

    Params:
        prompt: Original prompt dict from load_prompt
        previous_content: Extracted markdown from the prior page

    Returns:
        dict — new prompt with augmented user_prompt

    Example:
        >>> p = _build_context_prompt(prompt, "### Tables\\n|A|B|")
        >>> "previous page" in p["user_prompt"]
        True
    """
    content = previous_content
    if len(content) > _CONTEXT_MAX_CHARS:
        tail = content[-_CONTEXT_MAX_CHARS:]
        nl_pos = tail.find("\n")
        if nl_pos != -1 and len(tail) - nl_pos - 1 >= _CONTEXT_MIN_CHARS:
            content = tail[nl_pos + 1 :]
        else:
            content = tail

    augmented = dict(prompt)
    augmented["user_prompt"] = (
        "CONTEXT \u2014 The previous page ended with this "
        "content (use it to identify tables, lists, or "
        "sentences that continue onto this page; include the "
        "continuation in your extraction):\n\n"
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
    nl_pos = tail.find("\n")
    if nl_pos != -1 and len(tail) - nl_pos - 1 >= _CONTEXT_MIN_CHARS:
        return tail[nl_pos + 1 :]
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
        "section_continuation_detected": False,
        "table_continuation_detected": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
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
        "section_continuation_detected": continued and not table_cont,
        "table_continuation_detected": table_cont,
        "repeated_header_detected": header_rep,
        "repeated_footer_detected": footer_rep,
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


def process_docx(file_path: str, llm: LLMClient) -> ExtractionResult:
    """Extract content from a DOCX file via vision processing.

    Converts DOCX to PDF using LibreOffice headless, renders
    all pages to PNG, processes each sequentially through the
    LLM vision pipeline, classifies page continuation metadata
    via a separate LLM call, and returns an ExtractionResult
    with per-page results.

    Params:
        file_path: Absolute path to the DOCX file
        llm: LLMClient instance

    Returns:
        ExtractionResult with page-level extraction details

    Example:
        >>> result = process_docx("/data/report.docx", llm)
        >>> result.total_pages
        5
    """
    extraction_prompt = load_prompt("docx_extraction_vision")
    classification_prompt = load_prompt("page_continuation_classification")

    pages: List[PageResult] = []
    previous_content = ""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = _convert_to_pdf(Path(file_path), Path(tmp_dir))
        with open_rendered_pdf(pdf_path, get_vision_dpi_scale()) as rendered:
            for page_num in range(1, rendered.total_pages + 1):
                page_prompt = (
                    _build_context_prompt(
                        extraction_prompt,
                        previous_content,
                    )
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
                except (
                    RuntimeError,
                    ValueError,
                    OSError,
                ) as exc:
                    raise RuntimeError(
                        "DOCX extraction failed for "
                        f"'{Path(file_path).name}' on page "
                        f"{page_num}/{rendered.total_pages}: {exc}"
                    ) from exc
                pages.append(page_result)

    return ExtractionResult(
        file_path=file_path,
        filetype="docx",
        pages=pages,
        total_pages=len(pages),
        pages_succeeded=len(pages),
        pages_failed=0,
    )
