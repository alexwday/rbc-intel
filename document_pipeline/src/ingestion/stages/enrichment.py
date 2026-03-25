"""Stage 4: Page-level enrichment for prepared document content."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, List

import openai

from ..utils.llm import LLMClient
from ..utils.config import (
    get_enrichment_max_retries,
    get_enrichment_retry_delay,
    get_max_workers,
)
from ..utils.file_types import EnrichedPage, EnrichmentResult
from ..utils.logging_setup import get_stage_logger
from ..utils.prompt_loader import load_prompt
from ..processors.xlsx.types import (
    ColumnProfile,
    DenseTableDescription,
    PreparedDenseTable,
    TableEDA,
)
from .startup import PROCESSING_DIR

STAGE = "4-ENRICHMENT"
CONTENT_PREP_DIR = PROCESSING_DIR / "content_preparation"
ENRICHMENT_DIR = PROCESSING_DIR / "enrichment"
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)
_PROCESSOR_PROMPTS = {
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


def _load_content_preparation_results() -> List[dict[str, Any]]:
    """Load all content-preparation JSON files.

    Returns:
        list[dict] — parsed content-preparation results

    Example:
        >>> results = _load_content_preparation_results()
        >>> len(results)
        2
    """
    logger = get_stage_logger(__name__, STAGE)

    if not CONTENT_PREP_DIR.is_dir():
        logger.info("No content preparation directory found")
        return []

    results: list[dict[str, Any]] = []
    for json_path in sorted(CONTENT_PREP_DIR.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            results.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Skipping malformed content-prep file %s: %s",
                json_path.name,
                exc,
            )
    return results


def _deserialize_column_profile(
    data: dict[str, Any] | None,
) -> ColumnProfile | None:
    """Rebuild a ColumnProfile from serialized JSON data."""
    if not isinstance(data, dict):
        return None
    return ColumnProfile(
        name=str(data.get("name", "")),
        position=str(data.get("position", "")),
        dtype=str(data.get("dtype", "")),
        stats=(
            dict(data.get("stats", {}))
            if isinstance(data.get("stats"), dict)
            else {}
        ),
        sample_values=(
            [str(value) for value in data.get("sample_values", [])]
            if isinstance(data.get("sample_values"), list)
            else []
        ),
        non_null_count=int(data.get("non_null_count", 0)),
        null_count=int(data.get("null_count", 0)),
        unique_count=int(data.get("unique_count", 0)),
    )


def _deserialize_table_eda(data: dict[str, Any] | None) -> TableEDA | None:
    """Rebuild a TableEDA dataclass from serialized JSON data."""
    if not isinstance(data, dict):
        return None

    columns = [
        column
        for column in (
            _deserialize_column_profile(item)
            for item in data.get("columns", [])
        )
        if column is not None
    ]
    sample_rows = (
        [str(row) for row in data.get("sample_rows", [])]
        if isinstance(data.get("sample_rows"), list)
        else []
    )
    return TableEDA(
        row_count=int(data.get("row_count", 0)),
        columns=columns,
        header_row=int(data.get("header_row", 0)),
        framing_context=str(data.get("framing_context", "")),
        sample_rows=sample_rows,
        token_count=int(data.get("token_count", 0)),
        used_range=str(data.get("used_range", "")),
        header_mode=str(data.get("header_mode", "header_row")),
        source_region_id=str(data.get("source_region_id", "")),
    )


def _deserialize_dense_table_description(
    data: dict[str, Any] | None,
) -> DenseTableDescription | None:
    """Rebuild a DenseTableDescription from serialized JSON data."""
    if not isinstance(data, dict):
        return None
    return DenseTableDescription(
        description=str(data.get("description", "")),
        column_descriptions=[
            {
                "position": str(item.get("position", "")),
                "name": str(item.get("name", "")),
                "description": str(item.get("description", "")),
            }
            for item in data.get("column_descriptions", [])
            if isinstance(item, dict)
        ],
        filter_columns=(
            [str(value) for value in data.get("filter_columns", [])]
            if isinstance(data.get("filter_columns"), list)
            else []
        ),
        identifier_columns=(
            [str(value) for value in data.get("identifier_columns", [])]
            if isinstance(data.get("identifier_columns"), list)
            else []
        ),
        measure_columns=(
            [str(value) for value in data.get("measure_columns", [])]
            if isinstance(data.get("measure_columns"), list)
            else []
        ),
        text_content_columns=(
            [str(value) for value in data.get("text_content_columns", [])]
            if isinstance(data.get("text_content_columns"), list)
            else []
        ),
        sample_queries=(
            [str(value) for value in data.get("sample_queries", [])]
            if isinstance(data.get("sample_queries"), list)
            else []
        ),
    )


def _deserialize_prepared_dense_table(
    data: dict[str, Any] | None,
) -> PreparedDenseTable | None:
    """Rebuild a PreparedDenseTable from serialized JSON data."""
    if not isinstance(data, dict):
        return None
    return PreparedDenseTable(
        region_id=str(data.get("region_id", "")),
        used_range=str(data.get("used_range", "")),
        routing_metadata=(
            dict(data.get("routing_metadata", {}))
            if isinstance(data.get("routing_metadata"), dict)
            else {}
        ),
        raw_content=(
            [dict(row) for row in data.get("raw_content", [])]
            if isinstance(data.get("raw_content"), list)
            else []
        ),
        replacement_content=str(data.get("replacement_content", "")),
        dense_table_eda=_deserialize_table_eda(data.get("dense_table_eda")),
        dense_table_description=_deserialize_dense_table_description(
            data.get("dense_table_description")
        ),
        description_generation_mode=str(
            data.get("description_generation_mode", "")
        ),
    )


def _get_enrichment_prompt(filetype: str) -> dict[str, Any]:
    """Load the enrichment prompt for a processor filetype."""
    prompts_dir = _PROCESSOR_PROMPTS.get(filetype)
    if prompts_dir is None:
        raise ValueError(f"Unsupported enrichment filetype: {filetype}")
    return load_prompt("page_enrichment", prompts_dir)


def _parse_enrichment_response(response: dict[str, Any]) -> dict[str, Any]:
    """Parse the enrichment tool-call response payload."""
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

    summary = parsed.get("summary", "")
    usage_description = parsed.get("usage_description", "")
    if not isinstance(summary, str):
        raise ValueError("Missing summary string")
    if not isinstance(usage_description, str):
        raise ValueError("Missing usage_description string")

    def _string_list(key: str) -> list[str]:
        value = parsed.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"Missing {key} list")
        return [str(item) for item in value]

    entities_data = parsed.get("entities", [])
    if not isinstance(entities_data, list):
        raise ValueError("Missing entities list")

    section_hierarchy_data = parsed.get("section_hierarchy", [])
    if not isinstance(section_hierarchy_data, list):
        raise ValueError("Missing section_hierarchy list")

    entities = [
        {
            "type": str(item.get("type", "")),
            "value": str(item.get("value", "")),
        }
        for item in entities_data
        if isinstance(item, dict)
    ]
    section_hierarchy = [
        {
            "level": int(item.get("level", 0)),
            "title": str(item.get("title", "")),
        }
        for item in section_hierarchy_data
        if isinstance(item, dict)
    ]

    return {
        "summary": summary,
        "usage_description": usage_description,
        "keywords": _string_list("keywords"),
        "classifications": _string_list("classifications"),
        "entities": entities,
        "section_hierarchy": section_hierarchy,
    }


def _build_enrichment_messages(
    page: dict[str, Any],
    filetype: str,
    total_pages: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the prompt and formatted messages for one page."""
    prompt = _get_enrichment_prompt(filetype)
    page_metadata = page.get("metadata", {})
    if not isinstance(page_metadata, dict):
        page_metadata = {}

    template = prompt["user_prompt"]
    replacements = {
        "filetype": filetype,
        "page_number": str(page["page_number"]),
        "total_pages": str(total_pages),
        "page_title": str(page["page_title"]),
        "page_type": str(page_metadata.get("page_type", "unknown")),
        "content": str(page["content"]),
    }
    user_prompt = template
    for key, value in replacements.items():
        user_prompt = user_prompt.replace("{" + key + "}", value)
    messages: list[dict[str, Any]] = []
    if prompt.get("system_prompt"):
        messages.append({"role": "system", "content": prompt["system_prompt"]})
    messages.append({"role": "user", "content": user_prompt})
    return prompt, messages


def _enrich_page(
    page: dict[str, Any],
    filetype: str,
    total_pages: int,
    llm: LLMClient,
    file_label: str = "",
) -> EnrichedPage:
    """Enrich one prepared page with summary and retrieval metadata.

    Params:
        page: Prepared page dict loaded from Stage 3 JSON
        filetype: Source filetype for the containing document
        total_pages: Total pages in the containing document
        llm: LLMClient instance

    Returns:
        EnrichedPage with page content plus enrichment metadata

    Example:
        >>> enriched = _enrich_page(page, "pdf", 3, llm)
        >>> enriched.page_number
        1
    """
    prompt, messages = _build_enrichment_messages(
        page=page,
        filetype=filetype,
        total_pages=total_pages,
    )
    response = llm.call(
        messages=messages,
        stage=prompt["stage"],
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
        context=(
            f"{file_label} page {page.get('page_number', '?')}/{total_pages}"
        ).strip(),
    )
    parsed = _parse_enrichment_response(response)
    dense_tables = [
        dense_table
        for dense_table in (
            _deserialize_prepared_dense_table(item)
            for item in page.get("dense_tables", [])
        )
        if dense_table is not None
    ]
    metadata = page.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return EnrichedPage(
        page_number=int(page["page_number"]),
        page_title=str(page["page_title"]),
        content=str(page["content"]),
        method=str(page["method"]),
        metadata=dict(metadata),
        summary=parsed["summary"],
        usage_description=parsed["usage_description"],
        keywords=parsed["keywords"],
        classifications=parsed["classifications"],
        entities=parsed["entities"],
        section_hierarchy=parsed["section_hierarchy"],
        original_content=str(page.get("original_content", "")),
        dense_tables=dense_tables,
        dense_table_eda=page.get("dense_table_eda"),
        dense_table_description=page.get("dense_table_description"),
        description_generation_mode=str(
            page.get("description_generation_mode", "")
        ),
    )


def _enrich_page_with_retry(
    page: dict[str, Any],
    filetype: str,
    total_pages: int,
    llm: LLMClient,
    file_label: str = "",
) -> EnrichedPage:
    """Retry enrichment before failing the whole file."""
    max_retries = get_enrichment_max_retries()
    retry_delay = get_enrichment_retry_delay()

    for attempt in range(1, max_retries + 1):
        try:
            return _enrich_page(
                page=page,
                filetype=filetype,
                total_pages=total_pages,
                llm=llm,
                file_label=file_label,
            )
        except RETRYABLE_ERRORS as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    "Enrichment failed after "
                    f"{max_retries} attempts for page "
                    f"{page.get('page_number', '?')}: {exc}"
                ) from exc
            logger = get_stage_logger(__name__, STAGE)
            logger.warning(
                "%s page %s/%s enrichment retry %d/%d after %.1fs: %s",
                file_label or "File",
                page.get("page_number", "?"),
                total_pages,
                attempt,
                max_retries,
                retry_delay * attempt,
                exc,
            )
            time.sleep(retry_delay * attempt)
    raise RuntimeError("Enrichment exited retry loop without a response")


def _enrich_file(
    preparation: dict[str, Any],
    llm: LLMClient,
) -> EnrichmentResult:
    """Enrich all pages for one file sequentially.

    All-or-nothing per file: one page failure fails the file.

    Params:
        preparation: Parsed Stage 3 JSON result
        llm: LLMClient instance

    Returns:
        EnrichmentResult for the entire file
    """
    file_path = preparation["file_path"]
    filetype = preparation["filetype"]
    pages_data = preparation.get("pages", [])
    total_pages = len(pages_data)
    file_label = _extract_filename(preparation)

    enriched_pages = [
        _enrich_page_with_retry(
            page,
            filetype,
            total_pages,
            llm,
            file_label=file_label,
        )
        for page in pages_data
    ]
    return EnrichmentResult(
        file_path=file_path,
        filetype=filetype,
        pages=enriched_pages,
        pages_enriched=len(enriched_pages),
        pages_failed=0,
        dense_tables_spliced=int(preparation.get("dense_tables_spliced", 0)),
    )


def _write_result(result: EnrichmentResult) -> None:
    """Write one enrichment result JSON file.

    Params:
        result: EnrichmentResult to persist

    Returns:
        None
    """
    ENRICHMENT_DIR.mkdir(parents=True, exist_ok=True)
    stem = result.file_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    path_hash = hashlib.sha256(result.file_path.encode()).hexdigest()[:12]
    output_path = ENRICHMENT_DIR / f"{stem}_{path_hash}.json"
    output_path.write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )


def run_enrichment(llm: LLMClient) -> None:
    """Orchestrate page-level enrichment for prepared files.

    Loads content-preparation JSONs, enriches files in parallel,
    writes per-file JSON results, and logs per-file failures
    without aborting the stage.

    Params:
        llm: LLMClient instance

    Returns:
        None

    Example:
        >>> run_enrichment(llm)
    """
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Starting enrichment")

    preparations = _load_content_preparation_results()
    if not preparations:
        logger.info("No prepared files to enrich")
        return

    logger.info("Enriching %d files", len(preparations))

    succeeded = 0
    failed = 0
    max_workers = get_max_workers()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_preparation = {
            pool.submit(_enrich_file, preparation, llm): preparation
            for preparation in preparations
        }
        for future in as_completed(future_to_preparation):
            preparation = future_to_preparation[future]
            filename = _extract_filename(preparation)
            try:
                result = future.result()
                _write_result(result)
                succeeded += 1
                logger.debug(
                    "Enriched %s — %d/%d pages",
                    filename,
                    result.pages_enriched,
                    len(result.pages),
                )
            except (
                ValueError,
                OSError,
                RuntimeError,
                openai.OpenAIError,
            ) as exc:
                failed += 1
                logger.error(
                    "Failed to enrich %s: %s",
                    filename,
                    exc,
                )

    logger.info(
        "Enrichment complete — %d succeeded, %d failed",
        succeeded,
        failed,
    )


def _extract_filename(preparation: dict[str, Any]) -> str:
    """Get a display filename from a content-preparation result."""
    file_path = preparation.get("file_path", "unknown")
    return file_path.rsplit("/", 1)[-1]
