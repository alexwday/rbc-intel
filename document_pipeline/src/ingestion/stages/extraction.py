"""Stage 2: Content extraction via filetype-specific processors."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import List

from ..connections.llm import LLMClient
from ..processors.docx import process_docx
from ..processors.pdf import process_pdf
from ..processors.pptx import process_pptx
from ..processors.xlsx import process_xlsx
from ..utils.config import get_max_workers
from ..utils.file_types import ExtractionResult, FileRecord
from ..utils.logging_setup import get_stage_logger
from .startup import PROCESSING_DIR

STAGE = "2-EXTRACTION"
EXTRACTION_DIR = PROCESSING_DIR / "extraction"


def _load_discovery() -> List[FileRecord]:
    """Read discovery.json and return files to process.

    Combines new and modified file lists, filters to
    supported filetypes only.

    Returns:
        list[FileRecord] — files requiring extraction

    Example:
        >>> files = _load_discovery()
        >>> len(files)
        5
    """
    logger = get_stage_logger(__name__, STAGE)
    discovery_path = PROCESSING_DIR / "discovery.json"
    data = json.loads(discovery_path.read_text(encoding="utf-8"))

    records: List[FileRecord] = []
    for category in ("new", "modified"):
        items = data.get(category, [])
        if not isinstance(items, list):
            logger.warning(
                "Skipping discovery category '%s' because it is not a list",
                category,
            )
            continue

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping discovery record %s[%d] because it is "
                    "not an object",
                    category,
                    idx,
                )
                continue

            record_data = {
                key: value for key, value in item.items() if key != "supported"
            }
            try:
                record = FileRecord(**record_data)
            except TypeError as exc:
                logger.warning(
                    "Skipping malformed discovery record %s[%d]: %s",
                    category,
                    idx,
                    exc,
                )
                continue
            if record.supported:
                records.append(record)
    return records


def _process_file(record: FileRecord, llm: LLMClient) -> ExtractionResult:
    """Route a file to its filetype-specific processor.

    Params:
        record: FileRecord to process
        llm: LLMClient instance

    Returns:
        ExtractionResult from the processor

    Example:
        >>> result = _process_file(pdf_record, llm)
        >>> result.filetype
        "pdf"
    """
    if record.filetype == "pdf":
        return process_pdf(record.file_path, llm)
    if record.filetype == "docx":
        return process_docx(record.file_path, llm)
    if record.filetype == "pptx":
        return process_pptx(record.file_path, llm)
    if record.filetype == "xlsx":
        return process_xlsx(record.file_path, llm)
    raise ValueError(f"Unsupported filetype: {record.filetype}")


def _write_result(result: ExtractionResult) -> None:
    """Write extraction result to JSON in processing/extraction/.

    Filename uses the file stem and a truncated path hash
    to avoid collisions across directories.

    Params:
        result: ExtractionResult to persist

    Returns:
        None
    """
    EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    stem = result.file_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    path_hash = hashlib.sha256(result.file_path.encode()).hexdigest()[:12]
    output_path = EXTRACTION_DIR / f"{stem}_{path_hash}.json"
    output_path.write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )


def run_extraction(llm: LLMClient) -> None:
    """Orchestrate content extraction for discovered files.

    Reads discovery.json, routes each file to its processor
    in parallel via ThreadPoolExecutor, writes per-file JSON
    results. Per-file failures are logged but do not crash
    the stage.

    Params:
        llm: LLMClient instance

    Returns:
        None

    Example:
        >>> run_extraction(llm)
    """
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Starting content extraction")

    records = _load_discovery()
    if not records:
        logger.info("No files to extract")
        return

    logger.info("Extracting %d files", len(records))

    succeeded = 0
    unusable = 0
    failed = 0

    def _extract(record: FileRecord) -> ExtractionResult:
        return _process_file(record, llm)

    max_workers = get_max_workers()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_record = {pool.submit(_extract, rec): rec for rec in records}
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                result = future.result()
                _write_result(result)
                if result.pages_succeeded > 0:
                    succeeded += 1
                    logger.debug(
                        "Extracted %s — %d/%d pages succeeded",
                        record.filename,
                        result.pages_succeeded,
                        result.total_pages,
                    )
                else:
                    unusable += 1
                    logger.debug(
                        "Extracted %s but produced no usable pages",
                        record.filename,
                    )
            except (ValueError, OSError, RuntimeError) as exc:
                failed += 1
                logger.error(
                    "Failed to extract %s: %s",
                    record.filename,
                    exc,
                )

    logger.info(
        "Extraction complete — files: %d succeeded, %d unusable, %d failed",
        succeeded,
        unusable,
        failed,
    )
