"""Stage 5: Finalize enriched files into document-level outputs."""

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import openai

from ..utils.config import (
    get_finalization_degradation_signal_threshold,
    get_finalization_embedding_batch_size,
    get_finalization_embedding_model,
    get_finalization_max_retries,
    get_finalization_retry_delay,
    get_max_workers,
)
from ..utils.file_types import FinalizedDocument
from ..utils.llm import LLMClient
from ..utils.logging_setup import get_stage_logger
from .finalization_support import (
    _apply_chunk_summary_prefixes,
    _build_chunks,
    _build_degradation_signals,
    _build_document_summary,
    _build_prompt_messages,
    _build_sheet_context_chains,
    _build_sheet_summaries,
    _call_prompt_with_retry,
    _classify_structure,
    _coerce_int,
    _default_primary_title,
    _detect_sections,
    _extract_document_metadata,
    _find_section_for_page,
    _generate_document_fields,
    _get_prompt,
    _normalize_string_list,
    _page_metadata,
    _parse_tool_arguments,
    _summarize_sections,
    _summarize_subsections,
)
from .startup import PROCESSING_DIR

STAGE = "5-FINALIZATION"
ENRICHMENT_DIR = PROCESSING_DIR / "enrichment"
FINALIZATION_DIR = PROCESSING_DIR / "finalization"
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def _load_enrichment_results() -> list[dict[str, Any]]:
    """Load all enrichment JSON files.

    Returns:
        list[dict] — parsed enrichment results

    Example:
        >>> results = _load_enrichment_results()
        >>> isinstance(results, list)
        True
    """
    logger = get_stage_logger(__name__, STAGE)

    if not ENRICHMENT_DIR.is_dir():
        logger.info("No enrichment directory found")
        return []

    results: list[dict[str, Any]] = []
    for json_path in sorted(ENRICHMENT_DIR.glob("*.json")):
        try:
            results.append(json.loads(json_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Skipping malformed enrichment file %s: %s",
                json_path.name,
                exc,
            )
    return results


def _extract_filename(enrichment: dict[str, Any]) -> str:
    """Get a display filename from an enrichment result."""
    file_path = str(enrichment.get("file_path", "unknown"))
    return file_path.rsplit("/", 1)[-1]


_MAX_EMBEDDING_CHARS = 20000


def _truncate_embedding_text(text: str) -> str:
    """Cap embedding input to stay within model token limits."""
    if len(text) <= _MAX_EMBEDDING_CHARS:
        return text
    return text[:_MAX_EMBEDDING_CHARS]


def _extract_embeddings(response: Any) -> list[list[float]]:
    """Extract embeddings from an OpenAI-compatible response."""
    data = response.get("data") if isinstance(response, dict) else None
    if data is None:
        data = getattr(response, "data", None)
    if not isinstance(data, list):
        raise ValueError("Embedding response missing data list")

    embeddings: list[list[float]] = []
    for item in data:
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if embedding is None:
            embedding = getattr(item, "embedding", None)
        if not isinstance(embedding, list):
            raise ValueError("Embedding response missing embedding vector")
        embeddings.append([float(value) for value in embedding])
    return embeddings


def _embed_texts_with_retry(
    llm: LLMClient,
    texts: list[str],
    context: str,
) -> list[list[float]]:
    """Generate embeddings for a batch of texts with retries."""
    logger = get_stage_logger(__name__, STAGE)
    max_retries = get_finalization_max_retries()
    retry_delay = get_finalization_retry_delay()
    model = get_finalization_embedding_model()

    for attempt in range(1, max_retries + 1):
        try:
            client = llm.get_client()
            response = client.embeddings.create(model=model, input=texts)
            return _extract_embeddings(response)
        except RETRYABLE_ERRORS as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    "Embedding generation failed after "
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
    raise RuntimeError("Embedding generation exited retry loop")


def _generate_embeddings(
    document_summary: str,
    chunks: list[Any],
    llm: LLMClient,
    file_label: str = "",
) -> tuple[list[float], list[Any]]:
    """Generate summary and chunk embeddings for one file."""
    summary_embedding: list[float] = []
    if document_summary.strip():
        summary_batch = _embed_texts_with_retry(
            llm=llm,
            texts=[document_summary],
            context=f"{file_label} summary embedding".strip(),
        )
        summary_embedding = summary_batch[0]

    batch_size = get_finalization_embedding_batch_size()
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [
            _truncate_embedding_text(
                f"{chunk.embedding_prefix}{chunk.content}"
            )
            for chunk in batch
        ]
        embeddings = _embed_texts_with_retry(
            llm=llm,
            texts=texts,
            context=(
                f"{file_label} chunk embeddings "
                f"{batch_start + 1}-{batch_start + len(batch)}"
            ).strip(),
        )
        if len(embeddings) != len(batch):
            raise ValueError(
                "Embedding count mismatch: "
                f"got {len(embeddings)} for {len(batch)} chunks"
            )
        for chunk, embedding in zip(batch, embeddings):
            chunk.embedding = embedding
    return summary_embedding, chunks


def _keyword_embedding_text(
    page_title: str,
    keyword: str,
    section_title: str,
) -> str:
    """Build one contextualized keyword embedding string."""
    return " | ".join(
        part
        for part in (
            page_title,
            f"keyword: {keyword}",
            f"section: {section_title}" if section_title else "",
        )
        if part
    )


def _build_keyword_embedding_rows(
    pages: list[dict[str, Any]],
    sections: list[Any],
) -> list[dict[str, Any]]:
    """Build per-keyword rows prior to embedding."""
    rows: list[dict[str, Any]] = []
    for page in pages:
        page_number = _coerce_int(page.get("page_number"), 0)
        page_title = str(page.get("page_title", "")).strip()
        section = _find_section_for_page(page_number, sections)
        section_title = section.title if section is not None else page_title
        for keyword in _normalize_string_list(page.get("keywords", [])):
            rows.append(
                {
                    "keyword": keyword,
                    "page_number": page_number,
                    "page_title": page_title,
                    "section": section_title,
                    "_text": _keyword_embedding_text(
                        page_title,
                        keyword,
                        section_title,
                    ),
                }
            )
    return rows


def _embed_row_texts(
    rows: list[dict[str, Any]],
    llm: LLMClient,
    file_label: str,
    context_label: str,
) -> dict[str, list[float]]:
    """Embed unique row texts and return a text-to-vector mapping."""
    unique_texts = list(dict.fromkeys(str(row["_text"]) for row in rows))
    embeddings_by_text: dict[str, list[float]] = {}
    batch_size = get_finalization_embedding_batch_size()

    for batch_start in range(0, len(unique_texts), batch_size):
        batch = unique_texts[batch_start : batch_start + batch_size]
        vectors = _embed_texts_with_retry(
            llm=llm,
            texts=batch,
            context=(
                f"{file_label} {context_label} "
                f"{batch_start + 1}-{batch_start + len(batch)}"
            ),
        )
        if len(vectors) != len(batch):
            raise ValueError(
                f"{context_label.title()} count mismatch: "
                f"got {len(vectors)} for {len(batch)} texts"
            )
        for text, vector in zip(batch, vectors):
            embeddings_by_text[text] = vector
    return embeddings_by_text


def _generate_keyword_embeddings(
    pages: list[dict[str, Any]],
    sections: list[Any],
    llm: LLMClient,
    file_label: str,
) -> list[dict[str, Any]]:
    """Generate per-keyword contextualized embeddings."""
    rows = _build_keyword_embedding_rows(pages, sections)
    if not rows:
        return []

    embeddings_by_text = _embed_row_texts(
        rows=rows,
        llm=llm,
        file_label=file_label,
        context_label="keyword embeddings",
    )
    finalized_rows: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.pop("_text"))
        row["embedding"] = embeddings_by_text[text]
        finalized_rows.append(row)
    return finalized_rows


def _iter_dense_tables(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the dense tables attached to a page."""
    dense_tables = page.get("dense_tables", [])
    if not isinstance(dense_tables, list):
        dense_tables = []
    tables = [dict(item) for item in dense_tables if isinstance(item, dict)]
    if tables:
        return tables

    if page.get("dense_table_eda") or page.get("dense_table_description"):
        return [
            {
                "used_range": "",
                "region_id": "",
                "dense_table_eda": page.get("dense_table_eda"),
                "dense_table_description": page.get("dense_table_description"),
            }
        ]
    return []


def _metric_embedding_text(metric: dict[str, Any]) -> str:
    """Build one metric embedding string."""
    parts = [
        str(metric.get("sheet_name", "")).strip(),
        f"metric: {str(metric.get('metric_name', '')).strip()}",
    ]
    platform = str(metric.get("platform", "")).strip()
    sub_platform = str(metric.get("sub_platform", "")).strip()
    if platform:
        parts.append(f"platform: {platform}")
    if sub_platform:
        parts.append(f"sub-platform: {sub_platform}")
    return " | ".join(part for part in parts if part)


_FORMULA_METRIC_PATTERNS = re.compile(
    r"(?:\bCalc[_\s]?\d+\b|\bFormula(?:[_\s]?\d+)?\b|"
    r"Adjusted (?:Percentage|Ratio).*(?:Variant|Offset)|"
    r"\bOffset \d+\b|ROUND\(|IF\(|SUM\(|VLOOKUP\(|"
    r"\bComputation\b|\bIntermediate\b)",
    re.IGNORECASE,
)


def _is_formula_metric(name: str) -> bool:
    """Check if a metric name looks like a formula/calculation column."""
    return bool(_FORMULA_METRIC_PATTERNS.search(name))


def _looks_like_formula_description(value: str) -> bool:
    """Return whether free text describes a formula/scaffolding column."""
    return bool(
        re.search(
            r"formula text|template|scenario|round\(|if\(|sum\(|vlookup\(",
            value,
            re.IGNORECASE,
        )
    )


def _is_formula_dense_table(dense_table: dict[str, Any]) -> bool:
    """Return whether a dense-table region is formula-only scaffolding."""
    eda = dense_table.get("dense_table_eda", {})
    description = dense_table.get("dense_table_description", {})
    if not isinstance(eda, dict) or not isinstance(description, dict):
        return False

    columns = eda.get("columns", [])
    if not isinstance(columns, list) or not columns:
        return False

    formula_like = 0
    total_columns = 0
    for column in columns:
        if not isinstance(column, dict):
            continue
        total_columns += 1
        if _is_formula_metric(str(column.get("name", "")).strip()):
            formula_like += 1

    column_descriptions = description.get("column_descriptions", [])
    if isinstance(column_descriptions, list):
        for item in column_descriptions:
            if not isinstance(item, dict):
                continue
            if _looks_like_formula_description(
                str(item.get("description", "")).strip()
            ):
                formula_like += 1

    if total_columns == 0:
        return False

    measure_columns = description.get("measure_columns", [])
    has_measure_columns = isinstance(measure_columns, list) and bool(
        measure_columns
    )
    return (formula_like / total_columns) >= 0.6 and not has_measure_columns


def _parse_metric_items(
    parsed: dict[str, Any],
    page: dict[str, Any],
    dense_table: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize metric extraction output for one dense table."""
    raw_metrics = parsed.get("metrics", [])
    if not isinstance(raw_metrics, list):
        raise ValueError("Metric extraction missing metrics list")

    page_number = _coerce_int(page.get("page_number"), 0)
    sheet_name = str(
        _page_metadata(page).get("sheet_name", page.get("page_title", ""))
    ).strip()
    used_range = str(dense_table.get("used_range", "")).strip()
    region_id = str(dense_table.get("region_id", "")).strip()

    metrics: list[dict[str, Any]] = []
    for item in raw_metrics:
        if not isinstance(item, dict):
            continue
        metric_name = str(item.get("metric_name", "")).strip()
        if not metric_name:
            continue
        if _is_formula_metric(metric_name):
            continue
        metrics.append(
            {
                "metric_name": metric_name,
                "platform": str(item.get("platform", "")).strip(),
                "sub_platform": str(item.get("sub_platform", "")).strip(),
                "periods_available": _normalize_string_list(
                    item.get("periods_available", [])
                ),
                "page_number": page_number,
                "sheet_name": sheet_name,
                "used_range": used_range,
                "region_id": region_id,
            }
        )
    return metrics


def _condensed_eda_json(eda: dict[str, Any]) -> str:
    """Build a token-efficient EDA summary for metric extraction."""
    if not isinstance(eda, dict):
        return "{}"
    columns = eda.get("columns", [])
    condensed = {
        "row_count": eda.get("row_count", 0),
        "used_range": eda.get("used_range", ""),
        "header_mode": eda.get("header_mode", ""),
        "columns": [
            {
                "name": col.get("name", ""),
                "position": col.get("position", ""),
                "dtype": col.get("dtype", ""),
            }
            for col in columns[:20]
            if isinstance(col, dict)
        ],
    }
    if len(columns) > 20:
        condensed["columns_truncated"] = len(columns)
    sample_rows = eda.get("sample_rows", [])
    if sample_rows:
        condensed["sample_rows"] = sample_rows[:3]
    return json.dumps(condensed, indent=2)


def _condensed_description_json(desc: dict[str, Any]) -> str:
    """Build a token-efficient description summary for metric extraction."""
    if not isinstance(desc, dict):
        return "{}"
    condensed = {
        "description": str(desc.get("description", "")),
        "sample_queries": desc.get("sample_queries", [])[:5],
        "column_descriptions": desc.get("column_descriptions", [])[:20],
    }
    return json.dumps(condensed, indent=2)


def _extract_metrics(
    pages: list[dict[str, Any]],
    filetype: str,
    llm: LLMClient,
    file_label: str,
) -> list[dict[str, Any]]:
    """Extract XLSX metrics and attach per-metric embeddings."""
    if filetype != "xlsx":
        return []

    prompt = _get_prompt(filetype, "metric_extraction")
    metrics: list[dict[str, Any]] = []
    for page in pages:
        for dense_table in _iter_dense_tables(page):
            if _is_formula_dense_table(dense_table):
                continue
            response = _call_prompt_with_retry(
                llm=llm,
                prompt=prompt,
                messages=_build_prompt_messages(
                    prompt=prompt,
                    replacements={
                        "sheet_name": str(
                            _page_metadata(page).get(
                                "sheet_name",
                                page.get("page_title", ""),
                            )
                        ),
                        "page_number": str(
                            _coerce_int(page.get("page_number"), 0)
                        ),
                        "used_range": str(dense_table.get("used_range", "")),
                        "dense_table_eda_json": _condensed_eda_json(
                            dense_table.get("dense_table_eda", {}),
                        ),
                        "dense_table_description_json": (
                            _condensed_description_json(
                                dense_table.get("dense_table_description", {}),
                            )
                        ),
                    },
                ),
                context=(
                    f"{file_label} metric extraction "
                    f"page {_coerce_int(page.get('page_number'), 0)}"
                ),
            )
            metrics.extend(
                _parse_metric_items(
                    _parse_tool_arguments(response), page, dense_table
                )
            )

    if not metrics:
        return []

    texts = [_metric_embedding_text(metric) for metric in metrics]
    batch_size = get_finalization_embedding_batch_size()
    for batch_start in range(0, len(metrics), batch_size):
        batch_metrics = metrics[batch_start : batch_start + batch_size]
        batch_texts = texts[batch_start : batch_start + batch_size]
        embeddings = _embed_texts_with_retry(
            llm=llm,
            texts=batch_texts,
            context=(
                f"{file_label} metric embeddings "
                f"{batch_start + 1}-{batch_start + len(batch_metrics)}"
            ),
        )
        if len(embeddings) != len(batch_metrics):
            raise ValueError(
                "Metric embedding count mismatch: "
                f"got {len(embeddings)} for {len(batch_metrics)} metrics"
            )
        for metric, embedding in zip(batch_metrics, embeddings):
            metric["embedding"] = embedding
    return metrics


def _build_extraction_metadata(
    enrichment: dict[str, Any],
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve upstream enrichment metadata for downstream stages."""
    page_metadata = []
    for page in pages:
        page_metadata.append(
            {
                "page_number": _coerce_int(page.get("page_number"), 0),
                "page_title": str(page.get("page_title", "")),
                "method": str(page.get("method", "")),
                "metadata": _page_metadata(page),
                "summary": str(page.get("summary", "")),
                "usage_description": str(page.get("usage_description", "")),
                "keywords": [
                    str(item)
                    for item in page.get("keywords", [])
                    if isinstance(item, str)
                ],
                "classifications": [
                    str(item)
                    for item in page.get("classifications", [])
                    if isinstance(item, str)
                ],
                "entities": [
                    dict(item)
                    for item in page.get("entities", [])
                    if isinstance(item, dict)
                ],
                "section_hierarchy": [
                    dict(item)
                    for item in page.get("section_hierarchy", [])
                    if isinstance(item, dict)
                ],
            }
        )

    return {
        "pages_enriched": _coerce_int(
            enrichment.get("pages_enriched"), len(pages)
        ),
        "pages_failed": _coerce_int(enrichment.get("pages_failed"), 0),
        "dense_tables_spliced": _coerce_int(
            enrichment.get("dense_tables_spliced"),
            0,
        ),
        "page_metadata": page_metadata,
    }


def _flatten_dense_tables(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten dense-table payloads preserved on pages."""
    dense_tables: list[dict[str, Any]] = []
    for page in pages:
        page_dense_tables = page.get("dense_tables", [])
        if not isinstance(page_dense_tables, list):
            continue
        dense_tables.extend(
            dict(item) for item in page_dense_tables if isinstance(item, dict)
        )
    return dense_tables


def _collect_finalization_inputs(
    pages: list[dict[str, Any]],
    filetype: str,
    llm: LLMClient,
    file_name: str,
    file_path: str,
) -> dict[str, Any]:
    """Run the non-embedding finalization steps for one file."""
    document_metadata = _extract_document_metadata(
        pages=pages,
        filetype=filetype,
        llm=llm,
        file_label=file_name,
        file_path=file_path,
    )
    structure_type, structure_confidence = _classify_structure(
        pages=pages,
        filetype=filetype,
        llm=llm,
        file_label=file_name,
    )
    sections = _detect_sections(
        pages=pages,
        filetype=filetype,
        llm=llm,
        file_label=file_name,
    )
    if (
        filetype in {"docx", "pdf"}
        and len(sections) >= 2
        and structure_type in {"semantic", "topic_based"}
    ):
        structure_type = "sections"
        if structure_confidence == "low":
            structure_confidence = "medium"
    sections = _summarize_sections(
        sections=sections,
        pages=pages,
        filetype=filetype,
        llm=llm,
        file_label=file_name,
        document_metadata=document_metadata,
    )
    sections = _summarize_subsections(
        sections=sections,
        pages=pages,
        filetype=filetype,
        llm=llm,
        file_label=file_name,
        document_metadata=document_metadata,
    )
    document_summary = _build_document_summary(
        document_metadata=document_metadata,
        sections=sections,
        page_count=len(pages),
    )
    document_description, document_usage, used_fallback_description = (
        _generate_document_fields(
            document_summary=document_summary,
            pages=pages,
            filetype=filetype,
            llm=llm,
            file_label=file_name,
            fallback_title=str(document_metadata.get("title", "")).strip()
            or _default_primary_title(pages),
        )
    )
    return {
        "document_metadata": document_metadata,
        "structure_type": structure_type,
        "structure_confidence": structure_confidence,
        "sections": sections,
        "document_summary": document_summary,
        "document_description": document_description,
        "document_usage": document_usage,
        "used_fallback_description": used_fallback_description,
        "sheet_context_chains": _build_sheet_context_chains(
            pages=pages,
            filetype=filetype,
        ),
        "extracted_metrics": _extract_metrics(
            pages=pages,
            filetype=filetype,
            llm=llm,
            file_label=file_name,
        ),
    }


def _build_embedding_outputs(
    pages: list[dict[str, Any]],
    sections: list[Any],
    filetype: str,
    llm: LLMClient,
    file_name: str,
    document_summary: str,
) -> dict[str, Any]:
    """Generate chunk and keyword embedding outputs for one file."""
    chunks = _build_chunks(pages, sections)
    chunks = _apply_chunk_summary_prefixes(
        chunks=chunks,
        sections=sections,
        filetype=filetype,
        llm=llm,
        file_label=file_name,
    )
    summary_embedding, chunks = _generate_embeddings(
        document_summary=document_summary,
        chunks=chunks,
        llm=llm,
        file_label=file_name,
    )
    keyword_embeddings = _generate_keyword_embeddings(
        pages=pages,
        sections=sections,
        llm=llm,
        file_label=file_name,
    )
    return {
        "chunks": chunks,
        "summary_embedding": summary_embedding,
        "keyword_embeddings": keyword_embeddings,
    }


def _finalize_file(
    enrichment: dict[str, Any],
    llm: LLMClient,
) -> FinalizedDocument:
    """Finalize one enrichment result into document-level output.

    Params:
        enrichment: Parsed Stage 4 JSON result
        llm: LLMClient instance

    Returns:
        FinalizedDocument for the entire file
    """
    file_path = str(enrichment["file_path"])
    filetype = str(enrichment["filetype"])
    pages = sorted(
        enrichment.get("pages", []),
        key=lambda page: _coerce_int(page.get("page_number"), 0),
    )
    file_name = Path(file_path).name

    document_inputs = _collect_finalization_inputs(
        pages=pages,
        filetype=filetype,
        llm=llm,
        file_name=file_name,
        file_path=file_path,
    )
    embedding_outputs = _build_embedding_outputs(
        pages=pages,
        sections=document_inputs["sections"],
        filetype=filetype,
        llm=llm,
        file_name=file_name,
        document_summary=str(document_inputs["document_summary"]),
    )
    degradation_signals = _build_degradation_signals(
        document_metadata=document_inputs["document_metadata"],
        structure_type=str(document_inputs["structure_type"]),
        structure_confidence=str(document_inputs["structure_confidence"]),
        pages=pages,
        summary_embedding=embedding_outputs["summary_embedding"],
        sections=document_inputs["sections"],
        used_fallback_description=bool(
            document_inputs["used_fallback_description"]
        ),
    )

    result = FinalizedDocument(
        file_path=file_path,
        filetype=filetype,
        file_name=file_name,
        document_summary=str(document_inputs["document_summary"]),
        document_description=str(document_inputs["document_description"]),
        document_usage=str(document_inputs["document_usage"]),
        document_metadata=document_inputs["document_metadata"],
        structure_type=str(document_inputs["structure_type"]),
        structure_confidence=str(document_inputs["structure_confidence"]),
        degradation_signals=degradation_signals,
        summary_embedding=embedding_outputs["summary_embedding"],
        sections=document_inputs["sections"],
        chunks=embedding_outputs["chunks"],
        dense_tables=_flatten_dense_tables(pages),
        sheet_summaries=_build_sheet_summaries(pages, filetype),
        keyword_embeddings=embedding_outputs["keyword_embeddings"],
        extracted_metrics=document_inputs["extracted_metrics"],
        sheet_context_chains=document_inputs["sheet_context_chains"],
        extraction_metadata=_build_extraction_metadata(enrichment, pages),
    )

    threshold = get_finalization_degradation_signal_threshold()
    if len(degradation_signals) >= threshold:
        raise RuntimeError(
            "Degraded processing: " + ", ".join(degradation_signals)
        )

    return result


def _serialize_finalized_document(
    result: FinalizedDocument,
) -> dict[str, Any]:
    """Serialize a finalized document and include derived counts."""
    serialized = asdict(result)
    serialized.update(
        {
            "page_count": result.page_count,
            "primary_section_count": result.primary_section_count,
            "subsection_count": result.subsection_count,
            "chunk_count": result.chunk_count,
            "dense_table_count": result.dense_table_count,
        }
    )
    return serialized


def _write_result(result: FinalizedDocument) -> None:
    """Write one finalized document JSON file."""
    FINALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    stem = result.file_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    path_hash = hashlib.sha256(result.file_path.encode()).hexdigest()[:12]
    output_path = FINALIZATION_DIR / f"{stem}_{path_hash}.json"
    output_path.write_text(
        json.dumps(_serialize_finalized_document(result), indent=2),
        encoding="utf-8",
    )


def run_finalization(llm: LLMClient) -> None:
    """Orchestrate document finalization for enriched files.

    Loads enrichment JSONs, finalizes files in parallel, writes
    per-file JSON results, and logs per-file failures without
    aborting the stage.

    Params:
        llm: LLMClient instance

    Returns:
        None

    Example:
        >>> run_finalization(llm)
    """
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Starting finalization")

    enrichments = _load_enrichment_results()
    if not enrichments:
        logger.info("No enriched files to finalize")
        return

    logger.info("Finalizing %d files", len(enrichments))

    succeeded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=get_max_workers()) as pool:
        future_to_enrichment = {
            pool.submit(_finalize_file, enrichment, llm): enrichment
            for enrichment in enrichments
        }
        for future in as_completed(future_to_enrichment):
            enrichment = future_to_enrichment[future]
            filename = _extract_filename(enrichment)
            try:
                result = future.result()
                _write_result(result)
                succeeded += 1
                logger.debug(
                    "Finalized %s — %d sections, %d chunks",
                    filename,
                    result.primary_section_count,
                    result.chunk_count,
                )
            except (
                ValueError,
                OSError,
                RuntimeError,
                openai.OpenAIError,
            ) as exc:
                failed += 1
                logger.error("Failed to finalize %s: %s", filename, exc)

    logger.info(
        "Finalization complete — %d succeeded, %d failed",
        succeeded,
        failed,
    )
