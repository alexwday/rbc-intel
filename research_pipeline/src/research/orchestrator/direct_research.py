"""
Direct Research Orchestrator — combo-based retrieval for Aegis integration.

Receives a research statement and a list of combos (data_source + period +
bank).  Each combo is researched in parallel, findings are consolidated,
and a single synthesized response is streamed.

Skips: router, clarifier, planner, filter resolver.
Uses:  metadata subagent, file research subagent, summarizer.
"""

import concurrent.futures
import logging
import time
import uuid
from typing import Any, Dict, Generator, List, Optional

from ..agent.summarizer import stream_research_summary
from ..agent.tools.database_metadata import fetch_available_data_sources
from ..agent.tools.database_router import route_query_with_cascading_retrieval
from ..agent.tools.research_types import FindingsList, IndexedFindingsList
from ..connections.oauth import fetch_oauth_token
from ..utils.prompt_loader import load_all_prompts
from ..utils.reference_processor import (
    finalize_reference_replacements,
    process_streaming_reference_buffer,
)

logger = logging.getLogger(__name__)


class ResearchCombo(Dict[str, str]):
    """A single research combo: data_source + period + bank."""


def consolidate_findings_with_refs(
    all_findings: FindingsList,
) -> tuple[IndexedFindingsList, Dict[str, Dict[str, Any]]]:
    """Assign sequential ref_ids to findings for citations."""
    indexed_findings: IndexedFindingsList = []
    master_reference_index: Dict[str, Dict[str, Any]] = {}

    ref_counter = 1
    for finding in all_findings:
        ref_id = str(ref_counter)
        indexed_findings.append({**finding, "ref_id": ref_id})
        master_reference_index[ref_id] = {
            "doc_name": finding["document_name"],
            "file_link": finding["file_link"],
            "file_name": finding["file_name"],
            "page": finding["page"] or 1,
            "page_reference": str(finding["page"] or 1),
            "source_filename": (
                finding["file_name"] or finding["document_name"]
            ),
            "source_data_source": finding["data_source"],
        }
        ref_counter += 1

    return indexed_findings, master_reference_index


def format_findings_for_summarizer(
    indexed_findings: IndexedFindingsList,
) -> Dict[str, str]:
    """Format indexed findings into research text grouped by data source."""
    from collections import defaultdict

    findings_by_ds: Dict[str, list] = defaultdict(list)
    for finding in indexed_findings:
        findings_by_ds[finding["data_source"]].append(finding)

    formatted: Dict[str, str] = {}
    for ds, ds_findings in findings_by_ds.items():
        findings_by_doc: Dict[str, list] = defaultdict(list)
        for f in ds_findings:
            findings_by_doc[f["document_name"]].append(f)

        parts = []
        for doc_name, doc_findings in findings_by_doc.items():
            parts.append(f"## {doc_name}\n")
            for f in sorted(doc_findings, key=lambda x: x["page"] or 0):
                parts.append(
                    f"**Page {f['page'] or 'N/A'}:** "
                    f"{f['finding']} [REF:{f['ref_id']}]\n"
                )
            parts.append("")
        formatted[ds] = "\n".join(parts)

    return formatted


def _execute_combo_query(
    combo: Dict[str, str],
    research_statement: str,
    query_embedding: Optional[List[float]],
    token: str,
    combo_index: int,
    total_combos: int,
) -> Dict[str, Any]:
    """Execute research for a single combo in a thread pool worker."""
    data_source = combo["data_source"]
    period = combo.get("period", "")
    bank = combo.get("bank", "")

    combo_label = f"{data_source}/{period}/{bank}"
    logger.info(
        "Combo %d/%d: researching %s",
        combo_index + 1,
        total_combos,
        combo_label,
    )

    filters = {}
    if period:
        filters["filter_1"] = period
    if bank:
        filters["filter_2"] = bank

    try:
        result = route_query_with_cascading_retrieval(
            data_source=data_source,
            token=token,
            query_context={
                "research_statement": research_statement,
                "query_embedding": query_embedding,
                "is_db_wide": False,
                "deep_research_approved": False,
                "filters": filters,
            },
        )

        findings = result.get("findings", [])
        logger.info(
            "Combo %d/%d complete: %s — %d findings",
            combo_index + 1,
            total_combos,
            combo_label,
            len(findings),
        )

        return {
            "combo": combo,
            "combo_label": combo_label,
            "findings": findings,
            "status_summary": result.get("status_summary", ""),
            "exception": None,
        }

    except Exception as exc:
        logger.error(
            "Combo %d/%d failed: %s — %s",
            combo_index + 1,
            total_combos,
            combo_label,
            exc,
        )
        return {
            "combo": combo,
            "combo_label": combo_label,
            "findings": [],
            "status_summary": f"Error: {exc}",
            "exception": exc,
        }


def _generate_query_embedding(
    research_statement: str, token: str
) -> Optional[List[float]]:
    """Generate embedding for the research statement."""
    from ..agent.planner import _generate_query_embedding_vector

    embedding, _ = _generate_query_embedding_vector(
        research_statement, token
    )
    return embedding


def execute_direct_research(
    research_statement: str,
    combos: List[Dict[str, str]],
    stream: bool = True,
) -> Generator[str, None, None]:
    """Run direct research across combos and stream the synthesized response.

    Args:
        research_statement: Detailed research query from Aegis.
        combos: List of {data_source, period, bank} dicts.
        stream: Whether to stream the response.

    Yields:
        Response chunks including combo status and final synthesis.
    """
    start_time = time.time()

    def _elapsed() -> str:
        return f"{time.time() - start_time:.1f}s"

    try:
        # --- Setup ---
        load_all_prompts("research")
        token, auth_info = fetch_oauth_token()
        available_data_sources = fetch_available_data_sources()

        logger.info(
            "Direct research: %d combos, statement: '%s...'",
            len(combos),
            research_statement[:100],
        )

        # Validate combos
        valid_combos = []
        for combo in combos:
            ds = combo.get("data_source", "")
            if ds not in available_data_sources:
                logger.warning("Skipping unknown data source: %s", ds)
                continue
            valid_combos.append(combo)

        if not valid_combos:
            yield "No valid combos provided.\n"
            return

        # --- Research plan ---
        yield f"# Research Plan ({_elapsed()})\n\n"
        yield f"{research_statement}\n\n"
        yield f"Researching {len(valid_combos)} combo(s):\n"
        for combo in valid_combos:
            yield (
                f"- {combo.get('bank', '?')} "
                f"{combo.get('period', '?')} "
                f"({combo['data_source']})\n"
            )
        yield "\n"

        # --- Generate embedding ---
        query_embedding = _generate_query_embedding(
            research_statement, token
        )

        # --- Parallel combo research ---
        all_findings: FindingsList = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(valid_combos), 5)
        ) as executor:
            futures = {
                executor.submit(
                    _execute_combo_query,
                    combo,
                    research_statement,
                    query_embedding,
                    token,
                    i,
                    len(valid_combos),
                ): combo
                for i, combo in enumerate(valid_combos)
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                combo_label = result["combo_label"]
                status = result["status_summary"]

                if result["exception"]:
                    yield f"{combo_label}: Error — {status}\n\n"
                else:
                    finding_count = len(result["findings"])
                    all_findings.extend(result["findings"])
                    yield (
                        f"{combo_label}: {finding_count} findings "
                        f"({_elapsed()})\n\n"
                    )

        # --- Synthesize ---
        if not all_findings:
            yield (
                "\n---\n\nNo findings across any combos. "
                "Try broadening the search.\n"
            )
            return

        yield f"\n---\n\n## Research Summary ({_elapsed()})\n"

        indexed_findings, master_reference_index = (
            consolidate_findings_with_refs(all_findings)
        )
        aggregated_research = format_findings_for_summarizer(
            indexed_findings
        )

        summary_context = {
            "research_statement": research_statement,
            "reference_index": master_reference_index,
        }

        summary_stream = stream_research_summary(
            aggregated_detailed_research=aggregated_research,
            token=token,
            available_data_sources=available_data_sources,
            summary_context=summary_context,
        )

        buffer = ""
        for item in summary_stream:
            if isinstance(item, dict) and "usage_details" in item:
                continue
            if isinstance(item, str):
                buffer += item
                processed, buffer = process_streaming_reference_buffer(
                    buffer, master_reference_index
                )
                if processed:
                    yield processed

        for chunk in finalize_reference_replacements(
            buffer, master_reference_index
        ):
            yield chunk

        logger.info(
            "Direct research complete: %d findings, %s",
            len(all_findings),
            _elapsed(),
        )

    except Exception as exc:
        logger.error(
            "Direct research failed: %s", exc, exc_info=True
        )
        yield f"\n**Error:** {exc}\n"
