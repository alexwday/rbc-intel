"""
Reference Processor - Citation linking for LLM responses.

Transforms [REF:X] markers in LLM output into formatted citation links.
Consolidates reference indices from multiple data source searches, assigns
sequential IDs, and replaces markers with structured references.

Supports individual refs [REF:1], comma-separated [REF:1,2,3], and
ranges [REF:1-5].
"""

import logging
import re
from typing import Any, Dict, Generator, List, Tuple

logger = logging.getLogger(__name__)

REF_PATTERN = re.compile(r"\[REF:([\d,\s\-]+)\]", re.IGNORECASE)
REF_INCOMPLETE_PATTERN = re.compile(
    r"\[(?:R(?:E(?:F(?::?[0-9,\s\-]*)?)?)?)?$", re.IGNORECASE
)


def _build_reference_link_text(
    source_filename: str,
    page: int,
) -> str:
    """Format citation display text as 'Filename, Pg. Y'."""
    return f"{source_filename}, Pg. {page}"


def _build_reference_text(
    ref_data: Dict[str, Any],
) -> str:
    """Construct a formatted citation string for a finding reference."""
    file_name = ref_data.get("file_name") or ""
    try:
        page = int(ref_data.get("page", 1))
    except (TypeError, ValueError):
        page = 1
    doc_name = ref_data.get("doc_name") or "Unknown Document"
    source_filename = ref_data.get("source_filename") or doc_name

    link_text = _build_reference_link_text(source_filename, page)
    return f"[{link_text}]({file_name}#page={page})"


def _parse_reference_ids(ref_text: str) -> List[str]:
    """Expand reference text into individual IDs, handling commas and ranges."""
    ref_ids: List[str] = []

    for part in ref_text.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start_num = int(start_str.strip())
                end_num = int(end_str.strip())
                if start_num <= end_num:
                    ref_ids.extend(
                        str(i) for i in range(start_num, end_num + 1)
                    )
                else:
                    ref_ids.extend(
                        str(i) for i in range(end_num, start_num + 1)
                    )
            except ValueError:
                ref_ids.append(part)
        else:
            ref_ids.append(part)

    return ref_ids


def _generate_reference_links(
    ref_ids: List[str],
    reference_index: Dict[str, Dict[str, Any]],
    deduplicate_by_page: bool = True,
) -> Tuple[List[str], List[str]]:
    """Build formatted links for reference IDs, optionally deduplicating by page."""
    page_links: Dict[Tuple[str, int], str] = {}
    found_refs: List[str] = []
    missing_refs: List[str] = []

    for ref_id in ref_ids:
        if ref_id not in reference_index:
            missing_refs.append(ref_id)
            continue

        found_refs.append(ref_id)
        ref_data = reference_index[ref_id]

        if deduplicate_by_page:
            doc_name = ref_data.get("doc_name", "Unknown Document")
            page = ref_data.get("page", 1)
            page_key = (doc_name, page)

            if page_key not in page_links:
                page_links[page_key] = _build_reference_text(ref_data)
        else:
            ref_text = _build_reference_text(ref_data)
            page_key = (ref_id, 0)
            page_links[page_key] = ref_text

    if missing_refs:
        logger.warning("References not found in index: %s", missing_refs)

    return list(page_links.values()), found_refs


def _replace_reference_markers_in_text(
    text_content: str,
    reference_index: Dict[str, Dict[str, Any]],
) -> str:
    """Substitute all [REF:X] patterns in text with formatted links."""

    def replace_match(match: re.Match) -> str:
        """Convert a single [REF:...] match to links or remove if unresolved."""
        ref_ids = _parse_reference_ids(match.group(1))
        links, found_refs = _generate_reference_links(ref_ids, reference_index)

        if not links:
            logger.warning(
                "Removing unresolved reference marker: %s", match.group(0)
            )
            return ""

        logger.debug(
            "Replaced %s with %d link(s) for refs: %s",
            match.group(0),
            len(links),
            found_refs,
        )
        return " ".join(links)

    return REF_PATTERN.sub(replace_match, text_content)


def finalize_reference_replacements(
    buffer: str, reference_index: Dict[str, Dict[str, Any]]
) -> Generator[str, None, None]:
    """Process remaining buffer at end of stream, replacing all [REF:X] markers.

    Args:
        buffer: Remaining accumulated content to process.
        reference_index: Master reference index for link generation.

    Yields:
        Processed content with markers converted to formatted links.
    """
    if not buffer:
        return

    if not reference_index:
        yield buffer
        return

    processed = _replace_reference_markers_in_text(buffer, reference_index)
    yield processed


def process_streaming_reference_buffer(
    buffer: str,
    reference_index: Dict[str, Dict[str, Any]],
    buffer_size: int = 80,
) -> Tuple[str, str]:
    """Process streaming buffer, replacing complete [REF:X] patterns immediately.

    Args:
        buffer: Accumulated content from stream chunks.
        reference_index: Master reference index for link generation.
        buffer_size: Maximum buffer before forcing output.

    Returns:
        Tuple of (processed_content, remaining_buffer).
    """
    if not buffer:
        return "", ""

    all_matches = list(REF_PATTERN.finditer(buffer))

    if not all_matches:
        incomplete_match = REF_INCOMPLETE_PATTERN.search(buffer)

        if incomplete_match:
            return (
                buffer[: incomplete_match.start()],
                buffer[incomplete_match.start() :],
            )

        if len(buffer) < buffer_size:
            return buffer, ""
        potential_ref_start = buffer.rfind("[")
        if potential_ref_start != -1 and potential_ref_start > len(buffer) - 15:
            return buffer[:potential_ref_start], buffer[potential_ref_start:]
        keep_chars = min(10, len(buffer) // 3)
        if keep_chars > 0:
            return buffer[:-keep_chars], buffer[-keep_chars:]
        return buffer, ""

    last_ref_end = max(m.end() for m in all_matches)

    trailing = buffer[last_ref_end:]
    incomplete_in_trailing = REF_INCOMPLETE_PATTERN.search(trailing)

    if incomplete_in_trailing:
        split_point = last_ref_end + incomplete_in_trailing.start()
        content_to_process = buffer[:split_point]
        remaining = buffer[split_point:]
    else:
        content_to_process = buffer
        remaining = ""

    processed = _replace_reference_markers_in_text(
        content_to_process, reference_index
    )

    return processed, remaining
