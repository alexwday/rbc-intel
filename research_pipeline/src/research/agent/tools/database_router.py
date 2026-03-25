"""
Database Router — cascading retrieval architecture with three paths.

**Path A - Selective (non-source-wide queries):**
    - Mode: file_selection (top 1 chunk per file)
    - LLM selects files from catalog (binary yes/no)
    - Deep research ALL selected files

**Path B - Source-wide + Deep Research Approved:**
    - Mode: metadata_research (top 3 chunks per file)
    - LLM makes 3-way decisions: answered/irrelevant/needs_deep_research
    - Deep research only files that NEED it

**Path C - Source-wide + Metadata Only:**
    - Mode: metadata_research (top 3 chunks per file)
    - LLM makes 3-way decisions
    - NO deep research
"""

import json
import logging
from typing import Any, Dict, List, Optional, TypedDict

from .database_metadata import DatabaseMetadataCache, fetch_available_data_sources
from .file_research_subagent import execute_file_research_sync
from .metadata_subagent import execute_unified_metadata_query
from .research_types import FindingsList


class DatabaseRouterError(Exception):
    """Exception raised for database router errors."""


class DatabaseRouterResult(TypedDict):
    """Result structure from database router."""

    findings: FindingsList
    status_summary: str
    path: str
    needs_deeper_analysis_count: int


QueryContext = Dict[str, Any]

logger = logging.getLogger(__name__)


def route_query_with_cascading_retrieval(
    data_source: str,
    token: Optional[str] = None,
    process_monitor: Optional[Any] = None,
    query_stage_name: Optional[str] = None,
    query_context: Optional[QueryContext] = None,
) -> DatabaseRouterResult:
    """Route a query using the cascading retrieval architecture.

    Args:
        data_source: Data source identifier.
        token: Authentication token for downstream services.
        process_monitor: Optional monitor for reporting stage progress.
        query_stage_name: Optional stage name for instrumentation.
        query_context: Query metadata with ``research_statement``,
            ``query_embedding``, ``is_db_wide``, ``deep_research_approved``,
            and optional ``filters``.

    Returns:
        DatabaseRouterResult with unified findings, status summary, and path.

    Raises:
        DatabaseRouterError: If query context is missing or routing fails.
    """
    if query_context is None:
        raise DatabaseRouterError(
            "query_context is required for cascading retrieval"
        )

    research_statement = query_context.get("research_statement", "")
    query_embedding = query_context.get("query_embedding")
    is_db_wide = query_context.get("is_db_wide", False)
    deep_research_approved = query_context.get(
        "deep_research_approved", False
    )
    filters = query_context.get("filters")

    research_config = DatabaseMetadataCache().get_research_config(data_source)
    enable_db_wide_deep_research = research_config[
        "enable_db_wide_deep_research"
    ]

    if is_db_wide:
        if deep_research_approved and enable_db_wide_deep_research:
            path = "B"
            mode = "metadata_research"
        else:
            path = "C"
            mode = "metadata_research"
    else:
        path = "A"
        mode = "file_selection"

    logger.info(
        "Cascading query to %s: Path %s (is_db_wide=%s, "
        "deep_research_approved=%s, enable_db_wide_deep_research=%s, "
        "mode=%s)",
        data_source,
        path,
        is_db_wide,
        deep_research_approved,
        enable_db_wide_deep_research,
        mode,
    )
    stage_name = query_stage_name or f"ds_cascading_{data_source}"

    if data_source not in fetch_available_data_sources():
        logger.error("Unknown data source: %s", data_source)
        if process_monitor:
            process_monitor.add_stage_details(
                stage_name,
                error=f"Unknown data source: {data_source}",
            )
        raise DatabaseRouterError(
            f"Unknown data source: {data_source}"
        )

    try:
        logger.info(
            "Stage 1: Metadata query for %s (mode=%s)",
            data_source,
            mode,
        )

        metadata_stage = f"{stage_name}_metadata"
        if process_monitor:
            process_monitor.start_stage(metadata_stage)

        unified_result = execute_unified_metadata_query(
            research_statement=research_statement,
            data_source=data_source,
            query_context={
                "token": token,
                "process_monitor": process_monitor,
                "stage_name": metadata_stage,
                "query_embedding": query_embedding,
                "filters": filters,
            },
            mode=mode,
        )

        if process_monitor:
            process_monitor.end_stage(metadata_stage)

        metadata_findings: FindingsList = unified_result.get("findings", [])
        needs_research_doc_ids = unified_result.get(
            "needs_research_doc_ids", []
        )
        irrelevant_count = unified_result.get("irrelevant_count", 0)

        answered_count = sum(
            1 for f in metadata_findings if f.get("source") == "metadata"
        )

        logger.info(
            "Stage 1 complete (Path %s): %d metadata findings, "
            "%d need research, %d irrelevant",
            path,
            answered_count,
            len(needs_research_doc_ids),
            irrelevant_count,
        )

        if process_monitor:
            process_monitor.add_stage_details(
                metadata_stage,
                path=path,
                mode=mode,
                answered_count=answered_count,
                needs_research_count=len(needs_research_doc_ids),
                irrelevant_count=irrelevant_count,
            )

        combined_findings: FindingsList = []
        needs_deeper_analysis_count = 0

        if path == "B" and needs_research_doc_ids:
            research_doc_ids_set = set(needs_research_doc_ids)
            combined_findings = [
                f
                for f in metadata_findings
                if f.get("document_id") not in research_doc_ids_set
            ]
            logger.info(
                "Path B: Keeping %d metadata findings for answered-only "
                "docs, %d docs will get file research",
                len(combined_findings),
                len(needs_research_doc_ids),
            )
        else:
            combined_findings = list(metadata_findings)

        if path == "C":
            if needs_research_doc_ids:
                logger.info(
                    "Path C: Skipping deep research for %d documents "
                    "in %s (metadata only)",
                    len(needs_research_doc_ids),
                    data_source,
                )
                needs_deeper_analysis_count = len(needs_research_doc_ids)
        elif needs_research_doc_ids:
            research_label = (
                "ALL selected" if path == "A" else "flagged"
            )
            logger.info(
                "Stage 2: Deep research on %d %s documents in %s",
                len(needs_research_doc_ids),
                research_label,
                data_source,
            )

            file_research_stage = f"{stage_name}_file_research"
            if process_monitor:
                process_monitor.start_stage(file_research_stage)

            file_research_result = execute_file_research_sync(
                research_statement=research_statement,
                document_ids=needs_research_doc_ids,
                data_source=data_source,
                research_context={
                    "token": token,
                    "process_monitor": process_monitor,
                    "stage_name": file_research_stage,
                    "query_embedding": query_embedding,
                },
            )

            if process_monitor:
                process_monitor.end_stage(file_research_stage)
                retrieval_paths = file_research_result.get(
                    "retrieval_paths", {}
                )
                process_monitor.add_stage_details(
                    file_research_stage,
                    documents_researched=len(needs_research_doc_ids),
                    research_type=research_label,
                    retrieval_paths=retrieval_paths,
                )

            file_findings: FindingsList = file_research_result.get(
                "findings", []
            )
            combined_findings.extend(file_findings)

            logger.info(
                "Stage 2 complete: %d file research findings added",
                len(file_findings),
            )
        else:
            logger.info("No file research needed for %s", data_source)

        logger.debug(
            "RESEARCH_OUTPUT [database_router_%s]: Combined findings "
            "(Path %s):\n%s",
            data_source,
            path,
            json.dumps(
                [
                    {
                        "document": f["document_name"],
                        "page": f.get("page"),
                        "source": f.get("source"),
                        "finding": f.get("finding", "")[:500],
                    }
                    for f in combined_findings
                ],
                indent=2,
            ),
        )

        # Build status summary
        status_parts: List[str] = []
        if path == "A":
            if needs_research_doc_ids:
                status_parts.append(
                    f"{len(needs_research_doc_ids)} files researched"
                )
            if irrelevant_count > 0:
                status_parts.append(f"{irrelevant_count} not relevant")
        else:
            if answered_count > 0:
                status_parts.append(
                    f"{answered_count} answered from metadata"
                )
            if path == "B" and needs_research_doc_ids:
                status_parts.append(
                    f"{len(needs_research_doc_ids)} from deep research"
                )
            elif path == "C" and needs_research_doc_ids:
                status_parts.append(
                    f"{len(needs_research_doc_ids)} need deeper analysis"
                )
            if irrelevant_count > 0:
                status_parts.append(f"{irrelevant_count} not relevant")

        status_summary = (
            f"{data_source}: " + ", ".join(status_parts)
            if status_parts
            else f"Queried {data_source}"
        )

        return {
            "findings": combined_findings,
            "status_summary": status_summary,
            "path": path,
            "needs_deeper_analysis_count": needs_deeper_analysis_count,
        }

    except Exception as exc:
        error_msg = (
            f"Error in cascading query for {data_source}: {exc}"
        )
        logger.error(error_msg, exc_info=True)

        if process_monitor:
            process_monitor.add_stage_details(
                stage_name, error=error_msg
            )

        raise DatabaseRouterError(error_msg) from exc
