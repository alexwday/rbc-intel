"""Shared helpers for Stage 5 finalization."""

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import openai

from .finalization_quality import (
    _extract_date_from_content,
    _extract_date_from_docx_metadata,
    _extract_metrics_from_text,
    _extract_structured_metrics_from_pages,
    _normalize_key_metrics,
)
from ..utils.config import (
    get_finalization_chunk_summary_batch_size,
    get_finalization_context_chain_depth,
    get_finalization_max_classification_pages,
    get_finalization_max_retries,
    get_finalization_metadata_page_count,
    get_finalization_retry_delay,
)
from ..utils.file_types import (
    DocumentChunk,
    DocumentSection,
    DocumentSubsection,
)
from ..utils.llm import LLMClient
from ..utils.logging_setup import get_stage_logger
from ..utils.prompt_loader import load_prompt

STAGE = "5-FINALIZATION"
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)
STRUCTURE_DEFAULT = ("semantic", "low")
BOILERPLATE_TITLES = {
    "references",
    "bibliography",
    "works cited",
    "acknowledgements",
    "acknowledgments",
    "abstract",
    "table of contents",
    "list of figures",
    "list of tables",
    "appendix",
}
CONTEXT_MARKERS = (
    "continued",
    "continuation",
    "continued from",
    "see previous",
    "prior sheet",
    "prior page",
    "same definitions",
    "same basis as",
    "carried forward",
)
PROCESSOR_PROMPTS = {
    "pdf": Path(__file__).resolve().parent.parent
    / "processors"
    / "pdf"
    / "prompts",
    "docx": Path(__file__).resolve().parent.parent
    / "processors"
    / "docx"
    / "prompts",
    "pptx": Path(__file__).resolve().parent.parent
    / "processors"
    / "pptx"
    / "prompts",
    "xlsx": Path(__file__).resolve().parent.parent
    / "processors"
    / "xlsx"
    / "prompts",
}


def _get_prompt(filetype: str, prompt_name: str) -> dict[str, Any]:
    """Load a processor-specific finalization prompt."""
    prompts_dir = PROCESSOR_PROMPTS.get(filetype)
    if prompts_dir is None:
        raise ValueError(f"Unsupported finalization filetype: {filetype}")
    return load_prompt(prompt_name, prompts_dir)


def _build_prompt_messages(
    prompt: dict[str, Any],
    replacements: dict[str, str],
) -> list[dict[str, str]]:
    """Format prompt messages by replacing template variables."""
    user_prompt = str(prompt["user_prompt"])
    for key, value in replacements.items():
        user_prompt = user_prompt.replace("{" + key + "}", value)

    messages: list[dict[str, str]] = []
    if prompt.get("system_prompt"):
        messages.append({"role": "system", "content": prompt["system_prompt"]})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _parse_tool_arguments(response: dict[str, Any]) -> dict[str, Any]:
    """Parse tool-call arguments from an LLM response."""
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
    if not isinstance(parsed, dict):
        raise ValueError("LLM tool arguments must decode to an object")
    return parsed


def _call_prompt_with_retry(
    llm: LLMClient,
    prompt: dict[str, Any],
    messages: list[dict[str, str]],
    context: str,
) -> dict[str, Any]:
    """Retry a finalization LLM prompt before failing."""
    logger = get_stage_logger(__name__, STAGE)
    max_retries = get_finalization_max_retries()
    retry_delay = get_finalization_retry_delay()

    for attempt in range(1, max_retries + 1):
        try:
            return llm.call(
                messages=messages,
                stage=prompt["stage"],
                tools=prompt.get("tools"),
                tool_choice=prompt.get("tool_choice"),
                context=context,
            )
        except RETRYABLE_ERRORS as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    "Finalization prompt failed after "
                    f"{max_retries} attempts for {context}: {exc}"
                ) from exc
            wait = retry_delay * attempt
            logger.warning(
                "%s retry %d/%d after %.1fs: %s",
                context,
                attempt,
                max_retries,
                wait,
                exc,
            )
            time.sleep(wait)
    raise RuntimeError("Finalization prompt exited retry loop")


def _coerce_int(value: Any, default: int) -> int:
    """Convert a value to int or return the default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncate_text(value: str, limit: int) -> str:
    """Return text truncated to a character limit."""
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _page_metadata(page: dict[str, Any]) -> dict[str, Any]:
    """Return safe page metadata from an enrichment page."""
    metadata = page.get("metadata", {})
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _page_hierarchy_title(page: dict[str, Any], level: int) -> str:
    """Return the first heading title at a given level."""
    hierarchy = page.get("section_hierarchy", [])
    if not isinstance(hierarchy, list):
        return ""

    for item in hierarchy:
        if not isinstance(item, dict):
            continue
        if _coerce_int(item.get("level"), 0) == level:
            title = str(item.get("title", "")).strip()
            if title:
                return title
    return ""


def _default_primary_title(pages: list[dict[str, Any]]) -> str:
    """Return a fallback primary section title."""
    if not pages:
        return "Document"
    first_page = pages[0]
    title = _page_hierarchy_title(first_page, 1).strip()
    if title:
        return title
    return str(first_page.get("page_title", "Document")).strip() or "Document"


def _normalize_string_list(raw_values: Any) -> list[str]:
    """Normalize a string list while preserving order and uniqueness."""
    if not isinstance(raw_values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _empty_document_metadata() -> dict[str, Any]:
    """Build an empty document metadata payload."""
    return {
        "title": "",
        "authors": [],
        "publication_date": "",
        "document_type": "",
        "abstract": "",
    }


def _should_expand_title(title: str) -> bool:
    """Return whether a short title likely needs more context."""
    words = [word for word in re.split(r"\s+", title.strip()) if word]
    if len(words) > 4:
        return False
    return not bool(
        re.search(r"\b(?:q[1-4]|fy)?20\d{2}\b|\bnorthbridge\b", title, re.I)
    )


def _docx_title_context_candidate(page: dict[str, Any]) -> str:
    """Return a useful secondary heading for a short DOCX title."""
    hierarchy = page.get("section_hierarchy", [])
    if not isinstance(hierarchy, list):
        return ""

    skip_prefixes = ("diagram", "notes", "appendix")
    generic_words = {
        "card",
        "cards",
        "chart",
        "figure",
        "grid",
        "key",
        "kpi",
        "snapshot",
        "summary",
        "table",
        "visual",
    }
    for item in hierarchy:
        if not isinstance(item, dict):
            continue
        if _coerce_int(item.get("level"), 0) < 2:
            continue
        candidate = str(item.get("title", "")).strip()
        if not candidate:
            continue
        normalized_candidate = re.sub(
            r"^(?:table|figure)\s*:\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        if not normalized_candidate:
            continue
        if normalized_candidate.lower().startswith(skip_prefixes):
            continue
        words = re.findall(r"[A-Za-z]+", normalized_candidate.lower())
        if words and all(word in generic_words for word in words):
            continue
        return normalized_candidate
    return ""


def _normalize_document_title(
    raw_title: str,
    pages: list[dict[str, Any]] | None = None,
    filetype: str = "",
) -> str:
    """Normalize the document title with a conservative DOCX fallback."""
    title = raw_title.strip()
    if not title or not pages:
        return title
    if filetype != "docx" or not _should_expand_title(title):
        return title

    candidate = _docx_title_context_candidate(pages[0])
    if candidate and candidate.lower() not in title.lower():
        return f"{title} - {candidate}"
    return title


def _normalize_document_metadata(
    parsed: dict[str, Any],
    pages: list[dict[str, Any]] | None = None,
    filetype: str = "",
    file_path: str = "",
) -> dict[str, Any]:
    """Normalize document metadata returned by the LLM."""
    metadata = _empty_document_metadata()
    metadata["title"] = _normalize_document_title(
        str(parsed.get("title", "")).strip(),
        pages,
        filetype=filetype,
    )
    metadata["authors"] = _normalize_string_list(parsed.get("authors", []))
    metadata["publication_date"] = str(
        parsed.get("publication_date", "")
    ).strip()
    metadata["document_type"] = str(parsed.get("document_type", "")).strip()
    metadata["abstract"] = str(parsed.get("abstract", "")).strip()

    if not metadata["publication_date"] and filetype == "docx" and file_path:
        metadata["publication_date"] = _extract_date_from_docx_metadata(
            file_path
        )

    if not metadata["publication_date"] and pages:
        metadata["publication_date"] = _extract_date_from_content(pages)

    return metadata


def _page_prompt_payload(
    page: dict[str, Any],
    include_content: bool,
) -> dict[str, Any]:
    """Build a serialized prompt payload for one page."""
    payload = {
        "page_number": _coerce_int(page.get("page_number"), 0),
        "page_title": str(page.get("page_title", "")),
        "summary": str(page.get("summary", "")),
        "usage_description": str(page.get("usage_description", "")),
        "page_type": str(_page_metadata(page).get("page_type", "unknown")),
        "sheet_name": str(_page_metadata(page).get("sheet_name", "")),
        "section_hierarchy": [
            dict(item)
            for item in page.get("section_hierarchy", [])
            if isinstance(item, dict)
        ],
    }
    if include_content:
        content = str(page.get("content", ""))
        payload["content"] = _truncate_text(content, 4000)
    return payload


def _format_page_payload_json(
    pages: list[dict[str, Any]],
    include_content: bool = False,
) -> str:
    """Serialize ordered page payloads for prompts."""
    payload = [
        _page_prompt_payload(page, include_content=include_content)
        for page in pages
    ]
    return json.dumps(payload, indent=2)


def _extract_document_metadata(
    pages: list[dict[str, Any]],
    filetype: str,
    llm: LLMClient,
    file_label: str,
    file_path: str = "",
) -> dict[str, Any]:
    """Extract structured document metadata from the first pages."""
    if not pages:
        return _empty_document_metadata()

    prompt = _get_prompt(filetype, "metadata_extraction")
    page_count = min(len(pages), get_finalization_metadata_page_count())
    messages = _build_prompt_messages(
        prompt=prompt,
        replacements={
            "filetype": filetype,
            "total_pages": str(len(pages)),
            "page_summaries_json": _format_page_payload_json(
                pages[:page_count],
                include_content=True,
            ),
        },
    )
    response = _call_prompt_with_retry(
        llm=llm,
        prompt=prompt,
        messages=messages,
        context=f"{file_label} metadata extraction",
    )
    return _normalize_document_metadata(
        _parse_tool_arguments(response),
        pages,
        filetype=filetype,
        file_path=file_path,
    )


def _classify_structure(
    pages: list[dict[str, Any]],
    filetype: str,
    llm: LLMClient,
    file_label: str,
) -> tuple[str, str]:
    """Classify the overall document structure type."""
    if filetype == "pptx":
        return "slides", "high"
    if filetype == "xlsx":
        return "sheets", "high"
    if not pages:
        return STRUCTURE_DEFAULT

    prompt = _get_prompt(filetype, "structure_classification")
    page_count = min(len(pages), get_finalization_max_classification_pages())
    messages = _build_prompt_messages(
        prompt=prompt,
        replacements={
            "filetype": filetype,
            "total_pages": str(len(pages)),
            "page_summaries_json": _format_page_payload_json(
                pages[:page_count],
                include_content=False,
            ),
        },
    )
    try:
        response = _call_prompt_with_retry(
            llm=llm,
            prompt=prompt,
            messages=messages,
            context=f"{file_label} structure classification",
        )
        parsed = _parse_tool_arguments(response)
    except (RuntimeError, ValueError, openai.OpenAIError):
        return STRUCTURE_DEFAULT

    structure_type = str(parsed.get("structure_type", "")).strip().lower()
    confidence = str(parsed.get("confidence", "")).strip().lower()
    if not structure_type:
        return STRUCTURE_DEFAULT
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    return structure_type, confidence


def _normalize_boundaries(
    items: list[dict[str, Any]],
    page_start: int,
    page_end: int,
    fallback_title: Callable[[int], str],
) -> list[dict[str, Any]]:
    """Normalize and de-duplicate boundary items by start page."""
    normalized: list[dict[str, Any]] = []
    seen_starts: set[int] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        start = _coerce_int(item.get("page_start"), page_start)
        start = max(page_start, min(start, page_end))
        if start in seen_starts:
            continue
        seen_starts.add(start)
        normalized.append(
            {
                "title": title or fallback_title(len(normalized) + 1),
                "page_start": start,
            }
        )

    normalized.sort(key=lambda item: item["page_start"])
    if not normalized:
        return [{"title": fallback_title(1), "page_start": page_start}]
    normalized[0]["page_start"] = page_start
    return normalized


def _build_primary_sections_from_boundaries(
    items: list[dict[str, Any]],
    total_pages: int,
) -> list[DocumentSection]:
    """Build primary sections from normalized boundary starts."""
    sections: list[DocumentSection] = []
    for index, item in enumerate(items, start=1):
        next_start = (
            items[index]["page_start"]
            if index < len(items)
            else total_pages + 1
        )
        start = item["page_start"]
        end = max(start, next_start - 1)
        sections.append(
            DocumentSection(
                section_number=index,
                title=item["title"],
                page_start=start,
                page_end=end,
                page_count=end - start + 1,
            )
        )
    return sections


def _build_subsections_from_boundaries(
    items: list[dict[str, Any]],
    page_end: int,
) -> list[DocumentSubsection]:
    """Build subsection dataclasses from normalized boundary starts."""
    subsections: list[DocumentSubsection] = []
    for index, item in enumerate(items, start=1):
        next_start = (
            items[index]["page_start"] if index < len(items) else page_end + 1
        )
        start = item["page_start"]
        end = max(start, min(page_end, next_start - 1))
        subsections.append(
            DocumentSubsection(
                subsection_number=index,
                title=item["title"],
                page_start=start,
                page_end=end,
                page_count=end - start + 1,
            )
        )
    return subsections


def _fallback_primary_boundaries(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Infer primary-section boundaries from per-page hierarchy hints."""
    if not pages:
        return []

    boundaries = [
        {
            "title": _default_primary_title(pages),
            "page_start": _coerce_int(pages[0].get("page_number"), 1),
        }
    ]
    previous_key = str(boundaries[0]["title"]).strip().lower()

    for page in pages[1:]:
        title = _page_hierarchy_title(page, 1).strip()
        if not title:
            continue
        key = title.lower()
        if key == previous_key:
            continue
        boundaries.append(
            {
                "title": title,
                "page_start": _coerce_int(page.get("page_number"), 1),
            }
        )
        previous_key = key
    return boundaries


def _fallback_subsection_boundaries(
    section: DocumentSection,
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Infer subsection boundaries from per-page hierarchy hints."""
    boundaries: list[dict[str, Any]] = []
    previous_key = ""

    for page in pages:
        title = _page_hierarchy_title(page, 2).strip()
        if not title:
            continue
        key = title.lower()
        if key == previous_key:
            continue
        boundaries.append(
            {
                "title": title,
                "page_start": _coerce_int(
                    page.get("page_number"), section.page_start
                ),
            }
        )
        previous_key = key
    return boundaries


def _subsection_title_builder(section_title: str) -> Callable[[int], str]:
    """Build fallback subsection titles for one section title."""
    return lambda index: f"{section_title} subsection {index}"


def _section_pages(
    pages: list[dict[str, Any]],
    section: DocumentSection,
) -> list[dict[str, Any]]:
    """Return the pages covered by a section."""
    return [
        page
        for page in pages
        if section.page_start
        <= _coerce_int(page.get("page_number"), 0)
        <= section.page_end
    ]


def _subsection_pages(
    pages: list[dict[str, Any]],
    subsection: DocumentSubsection,
) -> list[dict[str, Any]]:
    """Return the pages covered by a subsection."""
    return [
        page
        for page in pages
        if subsection.page_start
        <= _coerce_int(page.get("page_number"), 0)
        <= subsection.page_end
    ]


def _detect_subsections_with_llm(
    pages: list[dict[str, Any]],
    section: DocumentSection,
    filetype: str,
    llm: LLMClient,
    file_label: str,
) -> list[DocumentSubsection]:
    """Detect subsections within one primary section."""
    prompt = _get_prompt(filetype, "subsection_detection")
    section_pages = _section_pages(pages, section)
    messages = _build_prompt_messages(
        prompt=prompt,
        replacements={
            "filetype": filetype,
            "section_title": section.title,
            "page_start": str(section.page_start),
            "page_end": str(section.page_end),
            "page_summaries_json": _format_page_payload_json(
                section_pages,
                include_content=False,
            ),
        },
    )
    response = _call_prompt_with_retry(
        llm=llm,
        prompt=prompt,
        messages=messages,
        context=f"{file_label} subsection detection '{section.title}'",
    )
    parsed = _parse_tool_arguments(response)
    raw_items = parsed.get("subsections", [])
    if not isinstance(raw_items, list):
        raise ValueError("Subsection detection missing subsections list")

    boundaries = _normalize_boundaries(
        items=raw_items,
        page_start=section.page_start,
        page_end=section.page_end,
        fallback_title=lambda index: f"{section.title} subsection {index}",
    )
    return _build_subsections_from_boundaries(
        items=boundaries,
        page_end=section.page_end,
    )


def _detect_pdf_docx_sections(
    pages: list[dict[str, Any]],
    filetype: str,
    llm: LLMClient,
    file_label: str,
) -> list[DocumentSection]:
    """Detect sections for PDF and DOCX files."""
    total_pages = len(pages)
    if total_pages == 0:
        return []

    max_pages = get_finalization_max_classification_pages()
    if total_pages > max_pages:
        boundaries = _fallback_primary_boundaries(pages)
    else:
        prompt = _get_prompt(filetype, "section_detection")
        messages = _build_prompt_messages(
            prompt=prompt,
            replacements={
                "filetype": filetype,
                "total_pages": str(total_pages),
                "page_summaries_json": _format_page_payload_json(
                    pages,
                    include_content=False,
                ),
            },
        )
        response = _call_prompt_with_retry(
            llm=llm,
            prompt=prompt,
            messages=messages,
            context=f"{file_label} section detection",
        )
        parsed = _parse_tool_arguments(response)
        raw_items = parsed.get("sections", [])
        if not isinstance(raw_items, list):
            raise ValueError("Section detection missing sections list")
        boundaries = _normalize_boundaries(
            items=raw_items,
            page_start=1,
            page_end=total_pages,
            fallback_title=lambda index: f"Section {index}",
        )

    sections = _build_primary_sections_from_boundaries(
        items=boundaries,
        total_pages=total_pages,
    )
    if not sections:
        return []

    for section in sections:
        if section.page_count <= 1:
            continue
        section_pages = _section_pages(pages, section)
        if section.page_count > max_pages:
            fallback_items = _fallback_subsection_boundaries(
                section, section_pages
            )
            if fallback_items:
                normalized = _normalize_boundaries(
                    items=fallback_items,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    fallback_title=_subsection_title_builder(section.title),
                )
                section.subsections = _build_subsections_from_boundaries(
                    items=normalized,
                    page_end=section.page_end,
                )
            continue
        section.subsections = _detect_subsections_with_llm(
            pages=pages,
            section=section,
            filetype=filetype,
            llm=llm,
            file_label=file_label,
        )
    return sections


def _new_subsection(
    title: str,
    subsection_number: int,
    page_number: int,
) -> DocumentSubsection:
    """Create a single-page subsection."""
    return DocumentSubsection(
        subsection_number=subsection_number,
        title=title,
        page_start=page_number,
        page_end=page_number,
        page_count=1,
    )


def _finalize_pptx_section(
    section_number: int,
    section_title: str,
    section_pages: list[dict[str, Any]],
) -> DocumentSection:
    """Build one PPTX section with per-slide subsections."""
    start_page = _coerce_int(section_pages[0].get("page_number"), 1)
    end_page = _coerce_int(section_pages[-1].get("page_number"), start_page)
    subsections = [
        _new_subsection(
            title=str(page.get("page_title", "")).strip()
            or f"Slide {_coerce_int(page.get('page_number'), 0)}",
            subsection_number=index,
            page_number=_coerce_int(page.get("page_number"), start_page),
        )
        for index, page in enumerate(section_pages, start=1)
    ]
    return DocumentSection(
        section_number=section_number,
        title=section_title,
        page_start=start_page,
        page_end=end_page,
        page_count=end_page - start_page + 1,
        subsections=subsections,
    )


def _detect_pptx_sections(
    pages: list[dict[str, Any]],
) -> list[DocumentSection]:
    """Detect sections programmatically for PPTX files."""
    if not pages:
        return []

    sections: list[DocumentSection] = []
    current_pages: list[dict[str, Any]] = []
    current_title = ""

    for page in pages:
        page_number = _coerce_int(page.get("page_number"), 1)
        page_title = (
            str(page.get("page_title", "")).strip() or f"Slide {page_number}"
        )
        page_type = str(_page_metadata(page).get("page_type", ""))
        starts_section = page_type in {"title_slide", "agenda_slide"}

        if not current_pages:
            current_pages = [page]
            current_title = page_title
            continue

        if starts_section:
            sections.append(
                _finalize_pptx_section(
                    section_number=len(sections) + 1,
                    section_title=current_title,
                    section_pages=current_pages,
                )
            )
            current_pages = [page]
            current_title = page_title
            continue

        current_pages.append(page)

    if current_pages:
        sections.append(
            _finalize_pptx_section(
                section_number=len(sections) + 1,
                section_title=current_title,
                section_pages=current_pages,
            )
        )
    return sections


def _dense_table_subsection_title(
    page_title: str,
    dense_table: dict[str, Any],
    index: int,
) -> str:
    """Build a stable subsection title for an XLSX dense table."""
    routing = dense_table.get("routing_metadata", {})
    if not isinstance(routing, dict):
        routing = {}
    used_range = str(routing.get("used_range", "")).strip()
    if not used_range:
        used_range = str(dense_table.get("used_range", "")).strip()
    if used_range:
        return f"{page_title} [{used_range}]"
    region_id = str(routing.get("selected_region_id", "")).strip()
    if not region_id:
        region_id = str(dense_table.get("region_id", "")).strip()
    if region_id:
        return f"{page_title} [{region_id}]"
    return f"{page_title} dense table {index}"


def _detect_xlsx_sections(
    pages: list[dict[str, Any]],
) -> list[DocumentSection]:
    """Detect sections programmatically for XLSX files."""
    sections: list[DocumentSection] = []

    for index, page in enumerate(pages, start=1):
        page_number = _coerce_int(page.get("page_number"), index)
        page_title = (
            str(page.get("page_title", "")).strip() or f"Sheet {index}"
        )
        dense_tables = page.get("dense_tables", [])
        if not isinstance(dense_tables, list):
            dense_tables = []

        if (
            dense_tables
            and str(page.get("method", "")) == "dense_table_replaced"
        ):
            subsections = [
                _new_subsection(
                    title=_dense_table_subsection_title(
                        page_title,
                        dense_table,
                        dense_index,
                    ),
                    subsection_number=dense_index,
                    page_number=page_number,
                )
                for dense_index, dense_table in enumerate(
                    dense_tables, start=1
                )
                if isinstance(dense_table, dict)
            ]
        else:
            subsections = [
                _new_subsection(
                    title=page_title,
                    subsection_number=1,
                    page_number=page_number,
                )
            ]

        sections.append(
            DocumentSection(
                section_number=index,
                title=page_title,
                page_start=page_number,
                page_end=page_number,
                page_count=1,
                subsections=subsections,
            )
        )
    return sections


def _detect_sections(
    pages: list[dict[str, Any]],
    filetype: str,
    llm: LLMClient,
    file_label: str = "",
) -> list[DocumentSection]:
    """Detect document sections, dispatching by filetype."""
    sorted_pages = sorted(
        pages,
        key=lambda page: _coerce_int(page.get("page_number"), 0),
    )
    if filetype in {"pdf", "docx"}:
        return _detect_pdf_docx_sections(
            pages=sorted_pages,
            filetype=filetype,
            llm=llm,
            file_label=file_label or "document",
        )
    if filetype == "pptx":
        return _detect_pptx_sections(sorted_pages)
    if filetype == "xlsx":
        return _detect_xlsx_sections(sorted_pages)
    raise ValueError(f"Unsupported finalization filetype: {filetype}")


def _normalize_section_title(title: str) -> str:
    """Normalize a section title for boilerplate checks."""
    cleaned = re.sub(r"^[\d.\sIVXivx-]+", "", title).strip().lower()
    cleaned = cleaned.rstrip(":")
    return " ".join(cleaned.split())


def _is_boilerplate_section(title: str) -> bool:
    """Return whether a section title should skip summary prompts."""
    normalized = _normalize_section_title(title)
    if normalized == "appendix":
        return True
    return normalized in BOILERPLATE_TITLES


def _boilerplate_summary(
    title: str,
    document_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a deterministic summary for known boilerplate sections."""
    normalized = _normalize_section_title(title)
    if normalized not in BOILERPLATE_TITLES:
        return None

    overview = f"Standard {title} section"
    if normalized == "abstract":
        overview = (
            str(document_metadata.get("abstract", "")).strip() or overview
        )

    return {
        "overview": overview,
        "key_topics": [],
        "key_metrics": {},
        "key_findings": [],
        "notable_facts": [],
        "is_fallback": False,
    }


def _default_structured_summary(
    overview: str,
    is_fallback: bool,
) -> dict[str, Any]:
    """Build a structured summary payload with required keys."""
    return {
        "overview": overview,
        "key_topics": [],
        "key_metrics": {},
        "key_findings": [],
        "notable_facts": [],
        "is_fallback": is_fallback,
    }


def _normalize_summary_dict(
    parsed: dict[str, Any],
    fallback_overview: str,
    default_is_fallback: bool = False,
    pages: list[dict[str, Any]] | None = None,
    filetype: str = "",
) -> dict[str, Any]:
    """Normalize an LLM section-summary payload."""
    normalized_metrics = _normalize_key_metrics(parsed.get("key_metrics", {}))
    structured_metrics = _extract_structured_metrics_from_pages(
        pages or [],
        filetype=filetype,
    )
    if structured_metrics:
        merged_metrics = dict(normalized_metrics)
        merged_metrics.update(structured_metrics)
        normalized_metrics = _normalize_key_metrics(merged_metrics)
    key_findings = _normalize_string_list(parsed.get("key_findings", []))
    notable_facts = _normalize_string_list(parsed.get("notable_facts", []))

    if not normalized_metrics:
        overview = str(parsed.get("overview", "")).strip() or fallback_overview
        fallback_texts = [overview] + key_findings + notable_facts
        if filetype == "pptx" and pages:
            fallback_texts.extend(
                str(page.get("content", "")).strip()
                for page in pages
                if str(page.get("content", "")).strip()
            )
        normalized_metrics = _extract_metrics_from_text(fallback_texts)

    overview = str(parsed.get("overview", "")).strip() or fallback_overview
    is_fallback = parsed.get("is_fallback", default_is_fallback)
    return {
        "overview": overview,
        "key_topics": _normalize_string_list(parsed.get("key_topics", [])),
        "key_metrics": normalized_metrics,
        "key_findings": key_findings,
        "notable_facts": notable_facts,
        "is_fallback": bool(is_fallback),
    }


def _derive_page_summary(
    page: dict[str, Any],
    fallback_overview: str,
) -> dict[str, Any]:
    """Build a minimal summary from page-level enrichment."""
    return {
        "overview": str(page.get("summary", "")).strip() or fallback_overview,
        "key_topics": _normalize_string_list(page.get("keywords", []))[:8],
        "key_metrics": {},
        "key_findings": [],
        "notable_facts": [],
        "is_fallback": False,
    }


def _generate_structured_summary(
    pages: list[dict[str, Any]],
    title: str,
    filetype: str,
    llm: LLMClient,
    file_label: str,
    context_label: str,
) -> dict[str, Any]:
    """Generate a structured summary for one section-like range."""
    if not pages:
        return _default_structured_summary(title, True)

    prompt = _get_prompt(filetype, "section_summary")
    messages = _build_prompt_messages(
        prompt=prompt,
        replacements={
            "filetype": filetype,
            "section_title": title,
            "page_start": str(_coerce_int(pages[0].get("page_number"), 1)),
            "page_end": str(_coerce_int(pages[-1].get("page_number"), 1)),
            "page_summaries_json": _format_page_payload_json(
                pages,
                include_content=True,
            ),
        },
    )
    try:
        response = _call_prompt_with_retry(
            llm=llm,
            prompt=prompt,
            messages=messages,
            context=f"{file_label} {context_label} summary",
        )
        return _normalize_summary_dict(
            _parse_tool_arguments(response),
            title,
            pages=pages,
            filetype=filetype,
        )
    except (RuntimeError, ValueError, openai.OpenAIError):
        return _normalize_summary_dict(
            parsed={},
            fallback_overview=title,
            default_is_fallback=True,
            pages=pages,
            filetype=filetype,
        )


def _summarize_sections(
    sections: list[DocumentSection],
    pages: list[dict[str, Any]],
    filetype: str,
    llm: LLMClient,
    file_label: str,
    document_metadata: dict[str, Any],
) -> list[DocumentSection]:
    """Populate section summaries in place."""
    for section in sections:
        boilerplate = _boilerplate_summary(section.title, document_metadata)
        if boilerplate is not None:
            section.summary = boilerplate
            continue
        section.summary = _generate_structured_summary(
            pages=_section_pages(pages, section),
            title=section.title,
            filetype=filetype,
            llm=llm,
            file_label=file_label,
            context_label=f"section '{section.title}'",
        )
    return sections


def _summarize_subsections(
    sections: list[DocumentSection],
    pages: list[dict[str, Any]],
    filetype: str,
    llm: LLMClient,
    file_label: str,
    document_metadata: dict[str, Any],
) -> list[DocumentSection]:
    """Populate subsection summaries in place."""
    for section in sections:
        section_boilerplate = _boilerplate_summary(
            section.title,
            document_metadata,
        )
        for subsection in section.subsections:
            subsection_pages = _subsection_pages(pages, subsection)
            if section_boilerplate is not None:
                subsection.summary = dict(section_boilerplate)
                continue
            if subsection.page_count <= 1 and subsection_pages:
                subsection.summary = _derive_page_summary(
                    subsection_pages[0],
                    subsection.title,
                )
                continue
            subsection.summary = _generate_structured_summary(
                pages=subsection_pages,
                title=subsection.title,
                filetype=filetype,
                llm=llm,
                file_label=file_label,
                context_label=f"subsection '{subsection.title}'",
            )
    return sections


def _format_page_range(page_start: int, page_end: int) -> str:
    """Format a page range label."""
    if page_start == page_end:
        return f"page {page_start}"
    return f"pages {page_start}-{page_end}"


def _join_key_metrics(summary: dict[str, Any]) -> str:
    """Serialize key metrics from a structured summary."""
    metrics = summary.get("key_metrics", {})
    if not isinstance(metrics, dict):
        return ""
    parts = [
        f"{str(key).strip()} = {str(value).strip()}"
        for key, value in metrics.items()
        if str(key).strip() and str(value).strip()
    ]
    return ", ".join(parts)


def _metadata_summary_rows(
    document_metadata: dict[str, Any],
    page_count: int,
) -> list[tuple[str, str]]:
    """Build the metadata rows for the assembled document summary."""
    title = str(document_metadata.get("title", "")).strip()
    authors = _normalize_string_list(document_metadata.get("authors", []))
    publication_date = str(
        document_metadata.get("publication_date", "")
    ).strip()
    document_type = str(document_metadata.get("document_type", "")).strip()
    return [
        ("Title", title),
        ("Authors", ", ".join(authors)),
        ("Date", publication_date),
        ("Type", document_type),
        ("Pages", str(page_count)),
    ]


def _append_section_summary_lines(
    parts: list[str],
    section: DocumentSection,
) -> None:
    """Append one section's summary lines to the assembled summary."""
    parts.append("")
    parts.append(
        "## Section "
        f"{section.section_number}: {section.title} "
        f"({_format_page_range(section.page_start, section.page_end)})"
    )

    summary = section.summary or _default_structured_summary(
        section.title, True
    )
    overview = str(summary.get("overview", "")).strip()
    if overview:
        parts.append(f"**Overview:** {overview}")

    key_topics = _normalize_string_list(summary.get("key_topics", []))
    if key_topics:
        parts.append(f"**Key Topics:** {', '.join(key_topics)}")

    key_metrics = _join_key_metrics(summary)
    if key_metrics:
        parts.append(f"**Key Metrics:** {key_metrics}")

    key_findings = _normalize_string_list(summary.get("key_findings", []))
    if key_findings:
        parts.append(f"**Key Findings:** {'; '.join(key_findings)}")

    notable_facts = _normalize_string_list(summary.get("notable_facts", []))
    if notable_facts:
        parts.append(f"**Notable Facts:** {'; '.join(notable_facts)}")

    if section.subsections:
        parts.append("**Subsections:**")
        for subsection in section.subsections:
            page_range = _format_page_range(
                subsection.page_start,
                subsection.page_end,
            )
            parts.append(
                "- "
                f"{section.section_number}.{subsection.subsection_number} "
                f"{subsection.title} "
                f"({page_range})"
            )


def _build_document_summary(
    document_metadata: dict[str, Any],
    sections: list[DocumentSection],
    page_count: int,
) -> str:
    """Assemble the document summary from metadata and section summaries."""
    parts = ["# Document Metadata"]
    for label, value in _metadata_summary_rows(document_metadata, page_count):
        if value:
            parts.append(f"- {label}: {value}")

    abstract = str(document_metadata.get("abstract", "")).strip()
    if abstract:
        parts.extend(["", "## Abstract", abstract])

    for section in sections:
        _append_section_summary_lines(parts, section)

    return "\n".join(parts).strip()


def _generate_document_fields(
    document_summary: str,
    pages: list[dict[str, Any]],
    filetype: str,
    llm: LLMClient,
    file_label: str,
    fallback_title: str,
) -> tuple[str, str, bool]:
    """Generate document description and usage with fallbacks."""
    prompt = _get_prompt(filetype, "document_rollup")
    default_title = fallback_title or _default_primary_title(pages)
    fallback_description = default_title or "Document"
    try:
        response = _call_prompt_with_retry(
            llm=llm,
            prompt=prompt,
            messages=_build_prompt_messages(
                prompt=prompt,
                replacements={
                    "filetype": filetype,
                    "total_pages": str(len(pages)),
                    "page_summaries_json": document_summary,
                },
            ),
            context=f"{file_label} document fields",
        )
        parsed = _parse_tool_arguments(response)
    except (RuntimeError, ValueError, openai.OpenAIError):
        return fallback_description, "", True

    description = str(parsed.get("document_description", "")).strip()
    usage = str(parsed.get("document_usage", "")).strip()
    if not description:
        description = fallback_description
    used_fallback_description = description == fallback_description
    return description, usage, used_fallback_description


def _build_sheet_summaries(
    pages: list[dict[str, Any]],
    filetype: str,
) -> list[dict[str, Any]]:
    """Build XLSX sheet summaries from per-page enrichment."""
    if filetype != "xlsx":
        return []

    summaries: list[dict[str, Any]] = []
    for page in pages:
        metadata = _page_metadata(page)
        summaries.append(
            {
                "sheet_name": str(
                    metadata.get("sheet_name", page.get("page_title", ""))
                ),
                "summary": str(page.get("summary", "")),
                "usage": str(page.get("usage_description", "")),
                "handling_mode": str(
                    metadata.get("handling_mode", "page_like")
                ),
            }
        )
    return summaries


def _sheet_requires_prior_context(page: dict[str, Any]) -> bool:
    """Return whether a sheet looks like it depends on earlier sheets."""
    metadata = _page_metadata(page)
    if bool(metadata.get("requires_prior_context")):
        return True

    text = " ".join(
        part
        for part in (
            str(page.get("page_title", "")).lower(),
            str(page.get("summary", "")).lower(),
            str(page.get("usage_description", "")).lower(),
            _truncate_text(str(page.get("content", "")).lower(), 1200),
        )
        if part
    )
    return any(marker in text for marker in CONTEXT_MARKERS)


def _build_sheet_context_chains(
    pages: list[dict[str, Any]],
    filetype: str,
) -> list[dict[str, Any]]:
    """Build XLSX sheet context chains from continuation signals."""
    if filetype != "xlsx":
        return []

    sorted_pages = sorted(
        pages,
        key=lambda page: _coerce_int(page.get("page_number"), 0),
    )
    requires_context = {
        _coerce_int(page.get("page_number"), 0): _sheet_requires_prior_context(
            page
        )
        for page in sorted_pages
    }
    chains: list[dict[str, Any]] = []
    max_depth = get_finalization_context_chain_depth()

    for page in sorted_pages:
        page_number = _coerce_int(page.get("page_number"), 0)
        if not requires_context.get(page_number):
            continue

        context_indices: list[int] = []
        prior_page_number = page_number - 1
        while prior_page_number >= 1 and len(context_indices) < max_depth:
            context_indices.append(prior_page_number)
            if not requires_context.get(prior_page_number, False):
                break
            prior_page_number -= 1

        if not context_indices:
            continue

        chains.append(
            {
                "sheet_index": page_number,
                "sheet_name": str(
                    _page_metadata(page).get(
                        "sheet_name",
                        page.get("page_title", ""),
                    )
                ),
                "context_sheet_indices": list(reversed(context_indices)),
            }
        )
    return chains


def _find_section_for_page(
    page_number: int,
    sections: list[DocumentSection],
) -> DocumentSection | None:
    """Return the section containing a page."""
    for section in sections:
        if section.page_start <= page_number <= section.page_end:
            return section
    return sections[-1] if sections else None


def _find_subsection_for_page(
    page_number: int,
    section: DocumentSection | None,
) -> DocumentSubsection | None:
    """Return the first subsection containing a page."""
    if section is None:
        return None
    for subsection in section.subsections:
        if subsection.page_start <= page_number <= subsection.page_end:
            return subsection
    return None


def _build_hierarchy_path(
    section: DocumentSection | None,
    subsection: DocumentSubsection | None,
) -> str:
    """Build a chunk hierarchy path from section metadata."""
    if section is None:
        return ""
    if subsection is None or subsection.title == section.title:
        return section.title
    return f"{section.title} > {subsection.title}"


def _build_dense_table_routing(page: dict[str, Any]) -> dict[str, Any]:
    """Build dense-table routing metadata for one chunk."""
    dense_tables = page.get("dense_tables", [])
    if not isinstance(dense_tables, list) or not dense_tables:
        return {}

    routing_entries = []
    dense_region_ids: list[str] = []
    for dense_table in dense_tables:
        if not isinstance(dense_table, dict):
            continue
        routing = dense_table.get("routing_metadata", {})
        if isinstance(routing, dict):
            routing_entries.append(dict(routing))
            for value in routing.get("dense_table_region_ids", []):
                region_id = str(value).strip()
                if region_id and region_id not in dense_region_ids:
                    dense_region_ids.append(region_id)
        region_id = str(dense_table.get("region_id", "")).strip()
        if region_id and region_id not in dense_region_ids:
            dense_region_ids.append(region_id)

    if not routing_entries:
        return {}
    if len(routing_entries) == 1:
        return routing_entries[0]

    merged = dict(routing_entries[0])
    merged["dense_table_region_ids"] = dense_region_ids
    merged["regions"] = routing_entries
    return merged


def _build_chunks(
    pages: list[dict[str, Any]],
    sections: list[DocumentSection],
) -> list[DocumentChunk]:
    """Build one finalized chunk per page with hierarchy metadata."""
    chunks: list[DocumentChunk] = []
    sorted_pages = sorted(
        pages,
        key=lambda page: _coerce_int(page.get("page_number"), 0),
    )

    for chunk_number, page in enumerate(sorted_pages):
        page_number = _coerce_int(page.get("page_number"), chunk_number + 1)
        section = _find_section_for_page(page_number, sections)
        subsection = _find_subsection_for_page(page_number, section)
        hierarchy_path = _build_hierarchy_path(section, subsection)
        section_name = section.title if section is not None else ""
        subsection_name = (
            subsection.title if subsection is not None else section_name
        )
        if section is None:
            prefix = ""
        elif subsection is not None and subsection.title != section.title:
            prefix = f"{section.title} > {subsection.title}: "
        else:
            prefix = f"{section.title}: "

        chunks.append(
            DocumentChunk(
                chunk_number=chunk_number,
                page_number=page_number,
                content=str(page.get("content", "")),
                primary_section_number=(
                    section.section_number if section is not None else 0
                ),
                primary_section_name=section_name,
                subsection_number=(
                    subsection.subsection_number if subsection else 0
                ),
                subsection_name=subsection_name,
                hierarchy_path=hierarchy_path,
                primary_section_page_count=(
                    section.page_count if section is not None else 0
                ),
                subsection_page_count=(
                    subsection.page_count
                    if subsection is not None
                    else (section.page_count if section is not None else 0)
                ),
                embedding_prefix=prefix,
                is_dense_table_description=(
                    str(page.get("method", "")) == "dense_table_replaced"
                ),
                dense_table_routing=_build_dense_table_routing(page),
                metadata=_page_metadata(page),
            )
        )
    return chunks


def _parse_chunk_summary_map(
    parsed: dict[str, Any],
) -> dict[int, str]:
    """Parse batch chunk-summary results keyed by chunk number."""
    raw_items = parsed.get("summaries", [])
    if not isinstance(raw_items, list):
        raise ValueError("Chunk summary payload missing summaries list")

    summaries: dict[int, str] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        chunk_number = _coerce_int(item.get("chunk_number"), -1)
        summary = str(item.get("summary", "")).strip()
        if chunk_number < 0 or not summary:
            continue
        summaries[chunk_number] = summary
    return summaries


def _apply_chunk_summary_prefixes(
    chunks: list[DocumentChunk],
    sections: list[DocumentSection],
    filetype: str,
    llm: LLMClient,
    file_label: str,
) -> list[DocumentChunk]:
    """Replace hierarchy prefixes with LLM-generated chunk summaries."""
    if not chunks:
        return chunks

    prompt = _get_prompt(filetype, "chunk_summary")
    section_lookup = {section.section_number: section for section in sections}
    eligible_chunks = [
        chunk
        for chunk in chunks
        if chunk.primary_section_name
        and not _is_boilerplate_section(chunk.primary_section_name)
    ]
    batch_size = get_finalization_chunk_summary_batch_size()

    for batch_start in range(0, len(eligible_chunks), batch_size):
        batch = eligible_chunks[batch_start : batch_start + batch_size]
        batch_payload = []
        for chunk in batch:
            section = section_lookup.get(chunk.primary_section_number)
            section_overview = ""
            if section is not None:
                section_overview = str(
                    section.summary.get("overview", "")
                ).strip()
            batch_payload.append(
                {
                    "chunk_number": chunk.chunk_number,
                    "section_title": chunk.primary_section_name,
                    "section_overview": section_overview,
                    "subsection_title": chunk.subsection_name,
                    "hierarchy_path": chunk.hierarchy_path,
                    "content": _truncate_text(chunk.content, 4000),
                }
            )

        try:
            response = _call_prompt_with_retry(
                llm=llm,
                prompt=prompt,
                messages=_build_prompt_messages(
                    prompt=prompt,
                    replacements={
                        "filetype": filetype,
                        "chunk_payload_json": json.dumps(
                            batch_payload, indent=2
                        ),
                    },
                ),
                context=(
                    f"{file_label} chunk summaries "
                    f"{batch_start + 1}-{batch_start + len(batch)}"
                ),
            )
            summaries = _parse_chunk_summary_map(
                _parse_tool_arguments(response)
            )
        except (RuntimeError, ValueError, openai.OpenAIError):
            continue

        for chunk in batch:
            summary = summaries.get(chunk.chunk_number, "")
            if summary:
                chunk.embedding_prefix = f"[{summary}]\n\n"

    return chunks


def _build_degradation_signals(
    document_metadata: dict[str, Any],
    structure_type: str,
    structure_confidence: str,
    pages: list[dict[str, Any]],
    summary_embedding: list[float],
    sections: list[DocumentSection],
    used_fallback_description: bool,
) -> list[str]:
    """Collect degradation signals for the finalized document."""
    signals: list[str] = []
    if not str(document_metadata.get("title", "")).strip():
        signals.append("empty metadata")
    if (
        len(pages) > 5
        and structure_type == STRUCTURE_DEFAULT[0]
        and structure_confidence == STRUCTURE_DEFAULT[1]
    ):
        signals.append("default structure classification")
    if not summary_embedding:
        signals.append("missing summary embedding")
    if used_fallback_description:
        signals.append("fallback document description")
    if sections:
        fallback_sections = sum(
            1
            for section in sections
            if bool(section.summary.get("is_fallback"))
        )
        if fallback_sections / len(sections) > 0.5:
            signals.append("empty section summaries")
    return signals
