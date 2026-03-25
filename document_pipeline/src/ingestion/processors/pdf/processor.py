"""PDF processor — renders pages to PNG, extracts via LLM vision."""

import base64
import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import openai
import fitz

from ...utils.llm import LLMClient
from ...utils.config import (
    get_pdf_classification_max_retries,
    get_pdf_classification_retry_delay,
    get_pdf_vision_max_retries,
    get_pdf_vision_retry_delay,
    get_vision_dpi_scale,
)
from ...utils.file_types import ExtractionResult, PageResult
from ...utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

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


def extract_page_text(rendered_pdf: RenderedPdf, page_number: int) -> str:
    """Extract raw text from a PDF page via PyMuPDF.

    Params:
        rendered_pdf: Open RenderedPdf handle
        page_number: 1-indexed page number

    Returns:
        str — extracted text, empty string on failure
    """
    if page_number < 1 or page_number > rendered_pdf.total_pages:
        return ""
    try:
        page = rendered_pdf.document.load_page(page_number - 1)
        return page.get_text("text").strip()
    except _FITZ_ERRORS:
        return ""


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


def _page_log_context(
    file_label: str,
    page_number: int,
    total_pages: int,
) -> str:
    """Build a concise filename/page label for logs."""
    return f"{file_label} page {page_number}/{total_pages}"


def call_vision(
    llm,
    img_bytes,
    prompt,
    tool_choice,
    detail="auto",
    context: str = "",
    extracted_text: str = "",
) -> tuple:
    """Send a page image to the LLM and extract content.

    Builds a multi-modal message with system prompt and user
    prompt containing the base64 image. When extracted_text is
    provided, it is included as supplementary context so the
    LLM can cross-reference programmatic text against the
    rendered image.

    Params:
        llm: LLMClient instance
        img_bytes: PNG image bytes
        prompt: Loaded prompt dict with system_prompt,
            user_prompt, stage, tools, tool_choice
        tool_choice: Tool choice override (prompt value
            or constructed dict)
        detail: Vision detail level — "auto" (default),
            "low", or "high"
        context: Short log label for the request
        extracted_text: Programmatic text extracted from
            the page (supplementary, may be empty)

    Returns:
        tuple of (page_title, content)

    Example:
        >>> title, md = call_vision(llm, img, p, "required")
        >>> title
        "Q3 Financial Results"
    """
    max_retries = get_pdf_vision_max_retries()
    retry_delay = get_pdf_vision_retry_delay()
    b64 = base64.b64encode(img_bytes).decode()

    user_text = prompt["user_prompt"]
    if extracted_text:
        user_text = (
            f"{prompt['user_prompt']}\n\n"
            "## Programmatic Text Extraction\n"
            "The following text was extracted programmatically "
            "from this page. Use it to verify your visual "
            "reading and improve accuracy on small or dense "
            "text. The visual image is authoritative for "
            "layout, tables, and charts.\n\n"
            f"---\n{extracted_text}\n---"
        )

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
                    "text": user_text,
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
                context=context,
            )
            return parse_vision_response(response)

        except RETRYABLE_ERRORS as exc:
            if attempt == max_retries:
                logger.error(
                    "%s vision call failed after %d attempts: %s",
                    context or "Vision",
                    max_retries,
                    exc,
                )
                raise
            wait = retry_delay * attempt
            logger.warning(
                "%s retry %d/%d after %.1fs: %s",
                context or "Vision",
                attempt,
                max_retries,
                wait,
                exc,
            )
            time.sleep(wait)
    raise RuntimeError("Vision call exited retry loop without a response")


def _try_full_dpi(
    llm, img_bytes, prompt, tc, context: str, extracted_text: str
) -> PageResult:
    """Attempt extraction at full DPI.

    Params: llm, img_bytes, prompt, tc. Returns: PageResult.
    """
    title, content = call_vision(
        llm,
        img_bytes,
        prompt,
        tc,
        context=context,
        extracted_text=extracted_text,
    )
    return PageResult(
        page_number=0,
        page_title=title,
        content=content,
        method="full_dpi",
    )


def _try_high_detail(
    llm,
    img_bytes,
    prompt,
    tc,
    context: str,
    extracted_text: str,
) -> PageResult:
    """Attempt extraction with detail=high.

    Params: llm, img_bytes, prompt, tc. Returns: PageResult.
    """
    title, content = call_vision(
        llm,
        img_bytes,
        prompt,
        tc,
        detail="high",
        context=f"{context} high-detail",
        extracted_text=extracted_text,
    )
    return PageResult(
        page_number=0,
        page_title=title,
        content=content,
        method="high_detail",
    )


def _try_half_dpi(
    llm, img_bytes, prompt, tc, context: str, extracted_text: str
) -> PageResult:
    """Attempt extraction at half resolution.

    Params: llm, img_bytes, prompt, tc. Returns: PageResult.
    """
    small_img = shrink_image(img_bytes)
    title, content = call_vision(
        llm,
        small_img,
        prompt,
        tc,
        context=f"{context} half-resolution",
        extracted_text=extracted_text,
    )
    return PageResult(
        page_number=0,
        page_title=title,
        content=content,
        method="half_dpi",
    )


def _try_split_halves(
    llm,
    img_bytes,
    prompt,
    tc,
    context: str,
    _extracted_text: str,
) -> PageResult:
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

    first_title, first_content = call_vision(
        llm,
        first_img,
        prompt,
        tc,
        context=f"{context} {first_label.lower()}",
    )

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
    _, second_content = call_vision(
        llm,
        second_img,
        second_prompt,
        tc,
        context=f"{context} {second_label.lower()}",
    )

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
    file_label: str = "",
    extracted_text: str = "",
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
    context = _page_log_context(file_label, page_number, total_pages)

    last_exc: Exception = RuntimeError("No extraction attempts made")
    for label, fallback_label, attempt_fn in attempts:
        try:
            result = attempt_fn(
                llm,
                img_bytes,
                prompt,
                tc,
                context,
                extracted_text,
            )
            result.page_number = page_number
            logger.info(
                "%s extracted via %s (%d chars)",
                context,
                label,
                len(result.content),
            )
            return result
        except _FALLBACK_ERRORS as exc:
            last_exc = exc
            if fallback_label:
                logger.warning(
                    "%s %s failed (%s), next: %s",
                    context,
                    label,
                    exc,
                    fallback_label,
                )

    raise RuntimeError(
        f"Page {page_number}/{total_pages} failed "
        f"all extraction attempts: {last_exc}"
    ) from last_exc


_CONTEXT_MAX_CHARS = 800
_CONTEXT_MIN_CHARS = 50
_RETRYABLE_CLASSIFICATION_ERRORS = (
    openai.OpenAIError,
    RuntimeError,
    ValueError,
)


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
        "content. Use it only to decide whether the current "
        "page begins mid-table, mid-list, or mid-sentence. "
        "Do not copy or restate any text that is not visible "
        "on the current page image. Extract only the content "
        "that appears on the current page itself:\n\n"
        "---\n"
        f"{content}\n"
        "---\n\n" + prompt["user_prompt"]
    )
    return augmented


def _truncate_tail(content: str) -> str:
    """Truncate content to last _CONTEXT_MAX_CHARS at a newline."""
    if len(content) <= _CONTEXT_MAX_CHARS:
        return content
    tail = content[-_CONTEXT_MAX_CHARS:]
    nl_pos = tail.find("\n")
    if nl_pos != -1 and len(tail) - nl_pos - 1 >= _CONTEXT_MIN_CHARS:
        return tail[nl_pos + 1 :]
    return tail


def _truncate_head_tail(content: str) -> str:
    """Truncate to first and last _CONTEXT_MAX_CHARS chars."""
    if len(content) <= _CONTEXT_MAX_CHARS * 2:
        return content
    head = content[:_CONTEXT_MAX_CHARS]
    tail = content[-_CONTEXT_MAX_CHARS:]
    return head + "\n\n... (middle content omitted) ...\n\n" + tail


def _base_page_metadata() -> dict[str, Any]:
    """Return shared page metadata defaults for all fields.

    Every processor outputs this same set of base fields so the
    enrichment stage can consume pages uniformly.  Processor-
    specific classification calls then overwrite the subset they
    populate.

    Returns:
        dict — all shared metadata fields with safe defaults
    """
    return {
        "continued_from_previous_page": False,
        "section_continuation_detected": False,
        "table_continuation_detected": False,
        "repeated_header_detected": False,
        "repeated_footer_detected": False,
        "contains_page_furniture": False,
        "page_type": "content_page",
        "contains_chart": False,
        "contains_dashboard": False,
        "contains_comparison_layout": False,
        "has_dense_visual_content": False,
    }


def _classify_page_continuation(
    llm: LLMClient,
    prompt: dict[str, Any],
    current_content: str,
    previous_content: str,
    context: str = "",
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
        context=f"{context} continuation".strip(),
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
    context: str = "",
) -> dict[str, bool]:
    """Classify page continuation with retry logic."""
    max_retries = get_pdf_classification_max_retries()
    retry_delay = get_pdf_classification_retry_delay()
    for attempt in range(1, max_retries + 1):
        try:
            return _classify_page_continuation(
                llm,
                prompt,
                current_content,
                previous_content,
                context,
            )
        except _RETRYABLE_CLASSIFICATION_ERRORS as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Page continuation classification failed after "
                    f"{max_retries} attempts: {exc}"
                ) from exc
            wait = retry_delay * attempt
            logger.warning(
                "%s continuation classification retry %d/%d "
                "after %.1fs: %s",
                context or "Page continuation classification",
                attempt,
                max_retries,
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
    extraction_prompt = load_prompt("page_extraction", _PROMPTS_DIR)
    classification_prompt = load_prompt("page_continuation", _PROMPTS_DIR)

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
                page_text = extract_page_text(rendered, page_num)
                page_result = process_page(
                    llm,
                    render_page(rendered, page_num),
                    page_num,
                    rendered.total_pages,
                    page_prompt,
                    file_label=pdf_path.name,
                    extracted_text=page_text,
                )
                page_result.metadata.update(_base_page_metadata())
                if previous_content:
                    page_result.metadata.update(
                        _classify_continuation_with_retry(
                            llm,
                            classification_prompt,
                            page_result.content,
                            previous_content,
                            _page_log_context(
                                pdf_path.name,
                                page_num,
                                rendered.total_pages,
                            ),
                        )
                    )
                previous_content = page_result.content
            except (RuntimeError, ValueError, OSError) as exc:
                raise RuntimeError(
                    "PDF extraction failed for "
                    f"'{pdf_path.name}' on page "
                    f"{page_num}/{rendered.total_pages}: "
                    f"{exc}"
                ) from exc
            pages.append(page_result)

    return ExtractionResult(
        file_path=file_path,
        filetype="pdf",
        pages=pages,
        total_pages=len(pages),
        pages_succeeded=len(pages),
        pages_failed=0,
    )
