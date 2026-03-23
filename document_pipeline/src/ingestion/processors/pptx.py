"""PPTX processor — converts to PDF, then extracts via LLM vision."""

import base64
import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import openai
import fitz

from ..connections.llm import LLMClient
from ..utils.config import (
    get_pptx_classification_max_retries,
    get_pptx_classification_retry_delay,
    get_pptx_vision_max_retries,
    get_pptx_vision_retry_delay,
    get_vision_dpi_scale,
)
from ..utils.file_types import ExtractionResult, PageResult
from ..utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_FITZ_ERRORS = (RuntimeError, ValueError, OSError)
_FALLBACK_ERRORS = (openai.OpenAIError, RuntimeError, ValueError, OSError)
_RENDER_LOCK = threading.Lock()


@dataclass
class RenderedPdf:
    """Open PDF handle configured for page-by-page rendering.

    Params:
        pdf_path: Path to the source PDF
        document: Open fitz document handle
        matrix: Rendering matrix derived from the DPI scale
        total_pages: Total number of pages in the document
    """

    pdf_path: Path
    document: Any
    matrix: Any
    total_pages: int


@contextmanager
def open_rendered_pdf(
    pdf_path: Path, dpi_scale: float
) -> Iterator[RenderedPdf]:
    """Open PDF once for streaming page rendering.

    Suppresses MuPDF C-level stderr during PDF open and yields
    a handle that can render individual pages on demand.

    Params:
        pdf_path: Path to the PDF file
        dpi_scale: DPI multiplier (e.g. 2.0 for 144 DPI)

    Returns:
        Iterator yielding a RenderedPdf handle

    Example:
        >>> with open_rendered_pdf(Path("doc.pdf"), 2.0) as rendered:
        ...     rendered.total_pages
        10
    """
    with _RENDER_LOCK:
        fitz.TOOLS.mupdf_display_errors(False)
        try:
            try:
                document = fitz.open(str(pdf_path))
            except _FITZ_ERRORS as exc:
                raise RuntimeError(
                    f"Failed to open PDF '{pdf_path.name}': {exc}"
                ) from exc
            fitz.TOOLS.mupdf_warnings()
        finally:
            fitz.TOOLS.mupdf_display_errors(True)

    rendered = RenderedPdf(
        pdf_path=pdf_path,
        document=document,
        matrix=fitz.Matrix(dpi_scale, dpi_scale),
        total_pages=document.page_count,
    )
    try:
        yield rendered
    finally:
        document.close()


def render_page(rendered_pdf: RenderedPdf, page_number: int) -> bytes:
    """Render a single 1-indexed PDF page to PNG bytes.

    Params:
        rendered_pdf: Open RenderedPdf handle
        page_number: 1-indexed page number to render

    Returns:
        bytes — rendered PNG payload for the page

    Example:
        >>> with open_rendered_pdf(Path("doc.pdf"), 2.0) as rendered:
        ...     page = render_page(rendered, 1)
        >>> isinstance(page, bytes)
        True
    """
    if page_number < 1 or page_number > rendered_pdf.total_pages:
        raise ValueError(
            f"Page {page_number} is out of range for "
            f"'{rendered_pdf.pdf_path.name}'"
        )

    page_index = page_number - 1
    with _RENDER_LOCK:
        fitz.TOOLS.mupdf_display_errors(False)
        try:
            try:
                page = rendered_pdf.document.load_page(page_index)
                pix = page.get_pixmap(matrix=rendered_pdf.matrix, alpha=False)
                return pix.tobytes("png")
            except _FITZ_ERRORS as exc:
                raise RuntimeError(
                    f"Failed to render page {page_number}"
                    f" of '{rendered_pdf.pdf_path.name}': {exc}"
                ) from exc
            finally:
                fitz.TOOLS.mupdf_warnings()
        finally:
            fitz.TOOLS.mupdf_display_errors(True)


def shrink_image(img_bytes: bytes) -> bytes:
    """Halve image resolution via fitz.Pixmap.shrink(1).

    Params:
        img_bytes: Original PNG bytes

    Returns:
        bytes — shrunk PNG bytes
    """
    pix = fitz.Pixmap(img_bytes)
    pix.shrink(1)
    return pix.tobytes("png")


_MIN_SPLIT_DIMENSION = 64
_LANDSCAPE_RATIO = 1.5


def split_image(img_bytes: bytes) -> tuple:
    """Split PNG into two halves based on orientation.

    Portrait or square images split top/bottom. Landscape images
    (width > height * 1.5) split left/right. Raises ValueError
    when the splitting dimension is too small for reliable
    extraction.

    Params:
        img_bytes: Original PNG bytes

    Returns:
        tuple of (first_bytes, second_bytes, orientation)
        where orientation is "vertical" or "horizontal"
    """
    src = fitz.Pixmap(img_bytes)
    if src.width > src.height * _LANDSCAPE_RATIO:
        if src.width < _MIN_SPLIT_DIMENSION:
            raise ValueError(f"Image too narrow ({src.width}px) to split")
        mid = src.width // 2
        left = fitz.Pixmap(src, fitz.IRect(0, 0, mid, src.height))
        right = fitz.Pixmap(src, fitz.IRect(mid, 0, src.width, src.height))
        return left.tobytes("png"), right.tobytes("png"), "horizontal"
    if src.height < _MIN_SPLIT_DIMENSION:
        raise ValueError(f"Image too short ({src.height}px) to split")
    mid = src.height // 2
    top = fitz.Pixmap(src, fitz.IRect(0, 0, src.width, mid))
    bot = fitz.Pixmap(src, fitz.IRect(0, mid, src.width, src.height))
    return top.tobytes("png"), bot.tobytes("png"), "vertical"


def parse_tool_arguments(response: dict) -> dict:
    """Extract parsed tool-call arguments from an LLM response.

    Params: response (dict). Returns: dict.
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message payload")

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("LLM response missing tool calls")

    function_data = tool_calls[0].get("function")
    if not isinstance(function_data, dict):
        raise ValueError("LLM response missing function payload")

    arguments = function_data.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("LLM response missing function arguments")

    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("LLM tool arguments must decode to an object")
    return parsed


def parse_vision_response(response: dict) -> tuple[str, str]:
    """Parse the first tool call into page title and content.

    Params: response (dict). Returns: tuple[str, str].
    """
    parsed = parse_tool_arguments(response)
    page_title = parsed.get("page_title")
    content = parsed.get("content")
    if not isinstance(page_title, str) or not isinstance(content, str):
        raise ValueError(
            "LLM tool arguments missing string page_title/content"
        )
    return page_title, content


def call_vision(llm, img_bytes, prompt, tool_choice, detail="auto") -> tuple:
    """Send a page image to the LLM and extract content.

    Builds a multi-modal message with system prompt and user
    prompt containing the base64 image. Parses tool call
    response for page_title and content. Retries on transient
    OpenAI errors.

    Params:
        llm: LLMClient instance
        img_bytes: PNG image bytes
        prompt: Loaded prompt dict with system_prompt,
            user_prompt, stage, tools, tool_choice
        tool_choice: Tool choice override (prompt value
            or constructed dict)
        detail: Vision detail level — "auto" (default),
            "low", or "high"

    Returns:
        tuple of (page_title, content)

    Example:
        >>> title, md = call_vision(llm, img, p, "required")
        >>> title
        "Q3 Financial Results"
    """
    max_retries = get_pptx_vision_max_retries()
    retry_delay = get_pptx_vision_retry_delay()
    b64 = base64.b64encode(img_bytes).decode()

    messages = [
        {"role": "system", "content": prompt["system_prompt"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": detail,
                    },
                },
                {
                    "type": "text",
                    "text": prompt["user_prompt"],
                },
            ],
        },
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = llm.call(
                messages=messages,
                stage=prompt["stage"],
                tools=prompt["tools"],
                tool_choice=tool_choice,
            )
            return parse_vision_response(response)

        except RETRYABLE_ERRORS as exc:
            if attempt == max_retries:
                logger.error(
                    "Vision call failed after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise
            wait = retry_delay * attempt
            logger.warning(
                "Vision retry %d/%d after %.1fs: %s",
                attempt,
                max_retries,
                wait,
                exc,
            )
            time.sleep(wait)
    raise RuntimeError("Vision call exited retry loop without a response")


def _try_full_dpi(llm, img_bytes, prompt, tc) -> PageResult:
    """Attempt extraction at full DPI.

    Params: llm, img_bytes, prompt, tc. Returns: PageResult.
    """
    title, content = call_vision(llm, img_bytes, prompt, tc)
    return PageResult(
        page_number=0,
        page_title=title,
        content=content,
        method="full_dpi",
    )


def _try_high_detail(llm, img_bytes, prompt, tc) -> PageResult:
    """Attempt extraction with detail=high.

    Params: llm, img_bytes, prompt, tc. Returns: PageResult.
    """
    title, content = call_vision(llm, img_bytes, prompt, tc, detail="high")
    return PageResult(
        page_number=0,
        page_title=title,
        content=content,
        method="high_detail",
    )


def _try_half_dpi(llm, img_bytes, prompt, tc) -> PageResult:
    """Attempt extraction at half resolution.

    Params: llm, img_bytes, prompt, tc. Returns: PageResult.
    """
    small_img = shrink_image(img_bytes)
    title, content = call_vision(llm, small_img, prompt, tc)
    return PageResult(
        page_number=0,
        page_title=title,
        content=content,
        method="half_dpi",
    )


def _try_split_halves(llm, img_bytes, prompt, tc) -> PageResult:
    """Attempt extraction by splitting into halves.

    Portrait/square pages split top/bottom; landscape pages split
    left/right. The first half is extracted first and its content
    is passed as context for the second half.

    Params: llm, img_bytes, prompt, tc. Returns: PageResult.
    """
    first_img, second_img, orientation = split_image(img_bytes)
    if orientation == "vertical":
        first_label, second_label = "TOP HALF", "BOTTOM HALF"
    else:
        first_label, second_label = "LEFT HALF", "RIGHT HALF"

    first_title, first_content = call_vision(llm, first_img, prompt, tc)

    second_prompt = dict(prompt)
    second_prompt["user_prompt"] = (
        "The following markdown was extracted from the "
        f"{first_label} of this page:\n\n"
        f"{first_content}\n\n"
        "---\n\n"
        f"Now extract the {second_label} of the page shown "
        "in the image below, continuing from where the "
        f"{first_label.lower()} left off. Use the "
        f"{first_label.lower()} content for context (e.g. "
        "table column headers, section titles).\n\n" + prompt["user_prompt"]
    )
    _, second_content = call_vision(llm, second_img, second_prompt, tc)

    return PageResult(
        page_number=0,
        page_title=first_title,
        content=first_content + "\n\n" + second_content,
        method="split_halves",
    )


def process_page(
    llm: LLMClient,
    img_bytes: bytes,
    page_number: int,
    total_pages: int,
    prompt: dict,
) -> PageResult:
    """Extract content from a single page with fallback.

    Fallback chain: full DPI (auto detail) -> high detail
    -> half resolution -> split halves with context passing.
    Raises if all attempts fail — partial file extraction is
    not useful.

    Params:
        llm: LLMClient instance
        img_bytes: Rendered PNG bytes
        page_number: 1-indexed page number
        total_pages: Total pages in the document
        prompt: Loaded prompt dict

    Returns:
        PageResult with extraction results and method

    Example:
        >>> result = process_page(llm, img, 1, 10, prompt)
        >>> result.method
        "full_dpi"
    """
    tc = prompt.get("tool_choice")
    attempts = [
        ("full DPI", "high detail", _try_full_dpi),
        ("high detail", "half resolution", _try_high_detail),
        ("half resolution", "split halves", _try_half_dpi),
        ("split halves", None, _try_split_halves),
    ]

    last_exc: Exception = RuntimeError("No extraction attempts made")
    for label, fallback_label, attempt_fn in attempts:
        try:
            result = attempt_fn(llm, img_bytes, prompt, tc)
            result.page_number = page_number
            logger.info(
                "Page %d/%d extracted via %s (%d chars)",
                page_number,
                total_pages,
                label,
                len(result.content),
            )
            return result
        except _FALLBACK_ERRORS as exc:
            last_exc = exc
            if fallback_label:
                logger.warning(
                    "Page %d/%d %s failed (%s), next: %s",
                    page_number,
                    total_pages,
                    label,
                    exc,
                    fallback_label,
                )

    raise RuntimeError(
        f"Page {page_number}/{total_pages} failed "
        f"all extraction attempts: {last_exc}"
    ) from last_exc


_SOFFICE_PATHS = [
    "soffice",
    "libreoffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/usr/bin/libreoffice",
    "/usr/bin/soffice",
]

_CONVERSION_TIMEOUT_S = 120
_CONVERSION_LOCK = threading.Lock()
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
    max_retries = get_pptx_classification_max_retries()
    retry_delay = get_pptx_classification_retry_delay()
    for attempt in range(1, max_retries + 1):
        try:
            return _classify_slide(
                llm, prompt, page_number, page_title, content
            )
        except _RETRYABLE_CLASSIFICATION_ERRORS as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Slide classification failed after "
                    f"{max_retries} attempts for "
                    f"slide {page_number}: {exc}"
                ) from exc
            wait = retry_delay * attempt
            logger.warning(
                "Slide classification retry %d/%d after %.1fs for "
                "slide %d: %s",
                attempt,
                max_retries,
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
