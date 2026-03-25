"""Stage 3: Content preparation for enrichment and storage."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, List

from ..processors.xlsx import prepare_xlsx_page
from ..utils.config import get_max_workers
from ..utils.llm import LLMClient
from ..utils.file_types import ContentPreparationResult, PreparedPage
from ..utils.logging_setup import get_stage_logger
from .startup import PROCESSING_DIR

STAGE = "3-CONTENT-PREP"
EXTRACTION_DIR = PROCESSING_DIR / "extraction"
CONTENT_PREP_DIR = PROCESSING_DIR / "content_preparation"


def _load_extraction_results() -> List[dict[str, Any]]:
    """Load all extraction JSON files from processing/extraction/.

    Returns:
        list[dict] — parsed extraction results

    Example:
        >>> results = _load_extraction_results()
        >>> len(results)
        3
    """
    logger = get_stage_logger(__name__, STAGE)

    if not EXTRACTION_DIR.is_dir():
        logger.info("No extraction directory found")
        return []

    results: list[dict[str, Any]] = []
    for json_path in sorted(EXTRACTION_DIR.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            results.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Skipping malformed extraction file %s: %s",
                json_path.name,
                exc,
            )
    return results


def _prepare_page(
    page: dict[str, Any],
    filetype: str,
    llm: LLMClient,
    file_label: str = "",
) -> PreparedPage:
    """Process a single page through content preparation.

    Params:
        page: Page dict from extraction result
        filetype: Source filetype for processor dispatch
        llm: LLMClient instance

    Returns:
        PreparedPage with full content and optional dense table data
    """
    if filetype == "xlsx":
        return prepare_xlsx_page(page, llm, file_label=file_label)

    return PreparedPage(
        page_number=page["page_number"],
        page_title=page["page_title"],
        content=page["content"],
        method="passthrough",
        metadata=page.get("metadata", {}),
    )


def _prepare_file(
    extraction: dict[str, Any],
    llm: LLMClient,
) -> ContentPreparationResult:
    """Process all pages of a file through preparation.

    All-or-nothing per file: one page failure fails the file.

    Params:
        extraction: Parsed extraction JSON
        llm: LLMClient instance

    Returns:
        ContentPreparationResult for the file
    """
    file_path = extraction["file_path"]
    filetype = extraction["filetype"]
    pages_data = extraction.get("pages", [])
    file_label = _extract_filename(extraction)

    prepared_pages: list[PreparedPage] = []
    dense_count = 0

    for page_data in pages_data:
        prepared = _prepare_page(
            page_data,
            filetype,
            llm,
            file_label=file_label,
        )
        prepared_pages.append(prepared)
        dense_count += len(prepared.dense_tables)

    return ContentPreparationResult(
        file_path=file_path,
        filetype=filetype,
        pages=prepared_pages,
        dense_tables_spliced=dense_count,
    )


def _write_result(result: ContentPreparationResult) -> None:
    """Write content preparation result to JSON.

    Params:
        result: ContentPreparationResult to persist

    Returns:
        None
    """
    CONTENT_PREP_DIR.mkdir(parents=True, exist_ok=True)
    stem = result.file_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    path_hash = hashlib.sha256(result.file_path.encode()).hexdigest()[:12]
    output_path = CONTENT_PREP_DIR / f"{stem}_{path_hash}.json"
    output_path.write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )


def run_content_preparation(llm: LLMClient) -> None:
    """Orchestrate content preparation for extracted files.

    Loads extraction JSONs, processes each file in parallel,
    and writes per-file JSON results. Per-file failures are
    logged but do not crash the stage.

    Params:
        llm: LLMClient instance

    Returns:
        None

    Example:
        >>> run_content_preparation(llm)
    """
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Starting content preparation")

    extractions = _load_extraction_results()
    if not extractions:
        logger.info("No extraction results to prepare")
        return

    logger.info("Preparing %d files", len(extractions))

    succeeded = 0
    failed = 0

    def _prepare(
        extraction: dict[str, Any],
    ) -> ContentPreparationResult:
        return _prepare_file(extraction, llm)

    max_workers = get_max_workers()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_ext = {
            pool.submit(_prepare, ext): ext for ext in extractions
        }
        for future in as_completed(future_to_ext):
            ext = future_to_ext[future]
            filename = _extract_filename(ext)
            try:
                result = future.result()
                _write_result(result)
                succeeded += 1
                logger.debug(
                    "Prepared %s — %d dense tables spliced across %d pages",
                    filename,
                    result.dense_tables_spliced,
                    len(result.pages),
                )
            except (
                ValueError,
                OSError,
                RuntimeError,
            ) as exc:
                failed += 1
                logger.error(
                    "Failed to prepare %s: %s",
                    filename,
                    exc,
                )

    logger.info(
        "Content preparation — %d succeeded, %d failed",
        succeeded,
        failed,
    )


def _extract_filename(
    extraction: dict[str, Any],
) -> str:
    """Get a display filename from an extraction dict."""
    file_path = extraction.get("file_path", "unknown")
    return file_path.rsplit("/", 1)[-1]
