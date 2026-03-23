"""PPTX processor — converts to PDF, then extracts via LLM vision."""

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
_CONVERSION_LOCK = threading.Lock()
_CLASSIFICATION_MAX_RETRIES = 3
_CLASSIFICATION_RETRY_DELAY_S = 2.0
_RETRYABLE_CLASSIFICATION_ERRORS = (
    openai.OpenAIError,
    RuntimeError,
    ValueError,
)

_VALID_SLIDE_TYPES = {
    "title_slide",
    "agenda_slide",
    "appendix_data_slide",
    "dashboard_slide",
    "comparison_slide",
    "chart_slide",
    "content_slide",
}


def _classify_slide(
    llm: LLMClient,
    prompt: dict[str, Any],
    page_number: int,
    page_title: str,
    content: str,
) -> dict[str, Any]:
    """Classify a slide via LLM based on extracted content.

    Params:
        llm: LLMClient instance
        prompt: Loaded pptx_slide_classification prompt
        page_number: 1-indexed slide number
        page_title: Extracted slide title
        content: Extracted slide markdown

    Returns:
        dict with slide_type_guess and visual content flags
    """
    messages = []
    if prompt.get("system_prompt"):
        messages.append({"role": "system", "content": prompt["system_prompt"]})
    messages.append(
        {
            "role": "user",
            "content": (
                f"{prompt['user_prompt']}\n\n"
                f"## Slide context\n"
                f"- Slide number: {page_number}\n"
                f"- Slide title: {page_title}\n\n"
                f"## Extracted slide content\n"
                f"{content}"
            ),
        }
    )
    response = llm.call(
        messages=messages,
        stage=prompt["stage"],
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
    )
    return _parse_slide_classification(response)


def _parse_slide_classification(
    response: dict[str, Any],
) -> dict[str, Any]:
    """Parse and validate the slide classification tool response."""
    parsed = parse_tool_arguments(response)

    slide_type = parsed.get("slide_type", "content_slide")
    if slide_type not in _VALID_SLIDE_TYPES:
        slide_type = "content_slide"

    contains_chart = parsed.get("contains_chart", False)
    contains_dashboard = parsed.get("contains_dashboard", False)
    contains_comparison = parsed.get("contains_comparison_layout", False)
    has_dense_visual = parsed.get("has_dense_visual_content", False)

    return {
        "slide_type_guess": slide_type,
        "contains_chart": bool(contains_chart),
        "chart_count_estimate": 0,
        "contains_dashboard": bool(contains_dashboard),
        "contains_comparison_layout": bool(contains_comparison),
        "has_dense_visual_content": bool(has_dense_visual),
    }


def _classify_slide_with_retry(
    llm: LLMClient,
    prompt: dict[str, Any],
    page_number: int,
    page_title: str,
    content: str,
) -> dict[str, Any]:
    """Classify a slide with retry logic before failing the file."""
    for attempt in range(1, _CLASSIFICATION_MAX_RETRIES + 1):
        try:
            return _classify_slide(
                llm, prompt, page_number, page_title, content
            )
        except _RETRYABLE_CLASSIFICATION_ERRORS as exc:
            if attempt == _CLASSIFICATION_MAX_RETRIES:
                raise RuntimeError(
                    f"Slide classification failed after "
                    f"{_CLASSIFICATION_MAX_RETRIES} attempts for "
                    f"slide {page_number}: {exc}"
                ) from exc
            wait = _CLASSIFICATION_RETRY_DELAY_S * attempt
            logger.warning(
                "Slide classification retry %d/%d after %.1fs for "
                "slide %d: %s",
                attempt,
                _CLASSIFICATION_MAX_RETRIES,
                wait,
                page_number,
                exc,
            )
            time.sleep(wait)
    raise RuntimeError(
        "Slide classification exited retry loop without a response"
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
        "soffice to PATH for PPTX processing."
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


def _convert_to_pdf(pptx_path: Path, output_dir: Path) -> Path:
    """Convert a PPTX file to PDF using LibreOffice headless.

    Params:
        pptx_path: Path to the source PPTX file
        output_dir: Directory to write the converted PDF

    Returns:
        Path to the generated PDF file

    Example:
        >>> pdf = _convert_to_pdf(Path("deck.pptx"), Path("/tmp"))
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
        str(pptx_path),
    ]

    logger.info("Converting '%s' to PDF via LibreOffice", pptx_path.name)

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
                f"{_CONVERSION_TIMEOUT_S}s for '{pptx_path.name}'"
            ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion failed for "
            f"'{pptx_path.name}': "
            f"{_format_conversion_error(result)}"
        )

    pdf_path = output_dir / (pptx_path.stem + ".pdf")
    if not pdf_path.is_file():
        raise RuntimeError(
            f"LibreOffice conversion produced no output for "
            f"'{pptx_path.name}'"
        )

    logger.info(
        "Converted '%s' to PDF (%d bytes)",
        pptx_path.name,
        pdf_path.stat().st_size,
    )
    return pdf_path


def process_pptx(file_path: str, llm: LLMClient) -> ExtractionResult:
    """Extract content from a PPTX file via vision processing.

    Converts PPTX to PDF using LibreOffice headless, renders
    each slide to PNG, processes sequentially through the LLM
    vision pipeline, classifies each slide via a separate LLM
    call, and returns an ExtractionResult with per-slide results.

    Params:
        file_path: Absolute path to the PPTX file
        llm: LLMClient instance

    Returns:
        ExtractionResult with slide-level extraction details

    Example:
        >>> result = process_pptx("/data/deck.pptx", llm)
        >>> result.total_pages
        12
    """
    extraction_prompt = load_prompt("pptx_extraction_vision")
    classification_prompt = load_prompt("pptx_slide_classification")

    pages: List[PageResult] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = _convert_to_pdf(Path(file_path), Path(tmp_dir))
        with open_rendered_pdf(pdf_path, get_vision_dpi_scale()) as rendered:
            for page_num in range(1, rendered.total_pages + 1):
                try:
                    page_result = process_page(
                        llm,
                        render_page(rendered, page_num),
                        page_num,
                        rendered.total_pages,
                        extraction_prompt,
                    )
                    page_result.metadata.update(
                        _classify_slide_with_retry(
                            llm=llm,
                            prompt=classification_prompt,
                            page_number=page_num,
                            page_title=page_result.page_title,
                            content=page_result.content,
                        )
                    )
                except (
                    RuntimeError,
                    ValueError,
                    OSError,
                ) as exc:
                    raise RuntimeError(
                        "PPTX extraction failed for "
                        f"'{Path(file_path).name}' on slide "
                        f"{page_num}/{rendered.total_pages}: {exc}"
                    ) from exc
                pages.append(page_result)

    return ExtractionResult(
        file_path=file_path,
        filetype="pptx",
        pages=pages,
        total_pages=len(pages),
        pages_succeeded=len(pages),
        pages_failed=0,
    )
