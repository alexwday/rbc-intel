"""Research model orchestration for routing, research, and summarization."""

import concurrent.futures
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional

from sqlalchemy import text

from ..agent.tools.database_router import route_query_with_cascading_retrieval
from ..agent.tools.research_types import FindingsList, IndexedFinding, IndexedFindingsList
from ..connections.postgres import get_database_session
from ..utils.reference_processor import (
    finalize_reference_replacements,
    process_streaming_reference_buffer,
)

logger = logging.getLogger(__name__)


def consolidate_findings_with_refs(
    all_findings: FindingsList,
) -> tuple[IndexedFindingsList, Dict[str, Dict[str, Any]]]:
    """Consolidate findings from all data sources and assign reference IDs.

    Takes the combined findings from all data source queries and:
    1. Assigns sequential ref_ids starting from 1
    2. Builds a master reference index for the streaming processor

    Args:
        all_findings: Combined list of Finding objects from all data sources.

    Returns:
        Tuple of:
        - IndexedFindingsList: Findings with ref_id assigned
        - Dict: Master reference index for href link generation
    """
    indexed_findings: IndexedFindingsList = []
    master_reference_index: Dict[str, Dict[str, Any]] = {}

    ref_counter = 1
    for finding in all_findings:
        ref_id = str(ref_counter)

        # Create IndexedFinding by adding ref_id
        indexed_finding: IndexedFinding = {
            **finding,
            "ref_id": ref_id,
        }
        indexed_findings.append(indexed_finding)

        # Build reference index entry for href generation
        master_reference_index[ref_id] = {
            "doc_name": finding["document_name"],
            "file_link": finding["file_link"],
            "file_name": finding["file_name"],
            "page": finding["page"] or 1,
            "page_reference": str(finding["page"] or 1),
            "source_filename": finding["file_name"] or finding["document_name"],
            "source_data_source": finding["data_source"],
        }

        ref_counter += 1

    return indexed_findings, master_reference_index


def format_findings_for_summarizer(
    indexed_findings: IndexedFindingsList,
) -> Dict[str, str]:
    """Format indexed findings into research text for the summarizer.

    Groups findings by data source and formats them with [REF:X] markers
    that will be replaced with clickable links during streaming.

    Args:
        indexed_findings: Findings with ref_ids assigned.

    Returns:
        Dict mapping data_source to formatted research text.
    """
    from collections import defaultdict

    # Group findings by data source
    findings_by_ds: Dict[str, List[IndexedFinding]] = defaultdict(list)
    for finding in indexed_findings:
        findings_by_ds[finding["data_source"]].append(finding)

    formatted_research: Dict[str, str] = {}

    for data_source, ds_findings in findings_by_ds.items():
        # Group by document within each data source
        findings_by_doc: Dict[str, List[IndexedFinding]] = defaultdict(list)
        for finding in ds_findings:
            findings_by_doc[finding["document_name"]].append(finding)

        parts = []
        for doc_name, doc_findings in findings_by_doc.items():
            parts.append(f"## {doc_name}\n")

            # Sort by page number
            sorted_findings = sorted(
                doc_findings, key=lambda f: f["page"] or 0
            )

            for finding in sorted_findings:
                page = finding["page"] or "N/A"
                ref_id = finding["ref_id"]
                content = finding["finding"]

                parts.append(f"**Page {page}:** {content} [REF:{ref_id}]\n")

            parts.append("")

        formatted_research[data_source] = "\n".join(parts)

    return formatted_research


def _execute_data_source_query_task(
    data_source_name: str,
    query_text: str,
    token: str,
    ds_display_name: str,
    query_index: int,
    total_queries: int,
    query_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a single data source query in a thread pool worker.

    Uses unified cascading retrieval architecture where metadata subagent makes
    per-document decisions (answered/irrelevant/needs_deep_research), triggering file
    research only for documents that need it.

    Args:
        data_source_name (str): Internal name of the data source.
        query_text (str): The search query to execute.
        token (str): OAuth token for API authentication.
        ds_display_name (str): Human-readable data source name for display.
        query_index (int): Index of this query in the batch (0-based).
        total_queries (int): Total number of queries being executed.
        query_context (Optional[Dict[str, Any]]): Context containing research_statement
            and query_embedding.

    Returns:
        Dict[str, Any]: Query results with findings, status_summary, path info.
    """
    router_result = None
    task_exception = None

    from ..utils.process_monitoring import get_process_monitor_instance

    process_monitor = get_process_monitor_instance()
    query_stage_name = f"ds_query_{data_source_name}_{query_index}"

    process_monitor.start_stage(query_stage_name)
    process_monitor.add_stage_details(
        query_stage_name,
        data_source_name=data_source_name,
        ds_display_name=ds_display_name,
        query_text=query_text,
        query_index=query_index,
        total_queries=total_queries,
    )

    try:
        logger.info(
            "Thread executing query %d/%d for data source: %s",
            query_index + 1,
            total_queries,
            data_source_name,
        )
        if query_context is None:
            query_context = {
                "research_statement": query_text,
            }

        router_result = route_query_with_cascading_retrieval(
            data_source=data_source_name,
            token=token,
            process_monitor=process_monitor,
            query_stage_name=query_stage_name,
            query_context=query_context,
        )

        logger.info("Thread completed query for data source: %s", data_source_name)
        process_monitor.end_stage(query_stage_name)

        process_monitor.add_stage_details(
            query_stage_name,
            status_summary=router_result.get("status_summary", "No status provided"),
            findings_count=len(router_result.get("findings", [])),
            path=router_result.get("path", "unknown"),
        )

    except Exception as e:
        task_exception = e
        logger.error(
            "Thread error executing query for %s: %s",
            data_source_name,
            e,
            exc_info=True,
        )
        process_monitor.end_stage(query_stage_name, "error")
        process_monitor.add_stage_details(query_stage_name, error=str(e))

    finally:
        try:
            import gc

            gc.collect()
        except Exception as cleanup_exc:
            logger.warning("Error during worker cleanup: %s", cleanup_exc)

    return {
        "data_source_name": data_source_name,
        "query_text": query_text,
        "ds_display_name": ds_display_name,
        "query_index": query_index,
        "total_queries": total_queries,
        "router_result": router_result,
        "exception": task_exception,
    }


def _stream_model_workflow(
    conversation: Optional[Dict[str, Any]] = None,
    _html_callback: Optional[Callable] = None,
    debug_mode: bool = False,
    data_source_names: Optional[List[str]] = None,
    filters: Optional[Dict[str, Dict[str, str]]] = None,
) -> Generator[str, None, None]:
    """Run the agent workflow synchronously and yield streaming chunks.

    Orchestrates conversation processing, routing decisions, research planning,
    parallel data source queries, and response generation. Implements process monitoring
    for performance tracking and debugging.

    Args:
        conversation (Optional[Dict[str, Any]]): Conversation dictionary containing
            a messages list.
        _html_callback (Optional[Callable]): Optional callback for HTML rendering
            (deprecated).
        debug_mode (bool): When True, yields legacy DEBUG_DATA JSON at the end.
        data_source_names (Optional[List[str]]): Data sources to restrict queries to.
        filters (Optional[Dict[str, str]]): Optional filters to pass through to
            data source queries and planning.

    Yields:
        str: Streaming response chunks including research plans, status updates, and
            final synthesized answers.

    Raises:
        Exception: Critical errors are caught, logged, and yielded as error messages.
    """
    from ..utils.process_monitoring import (
        get_process_monitor_instance,
        set_process_monitoring_enabled,
    )

    set_process_monitoring_enabled(True)
    process_monitor = get_process_monitor_instance()
    run_uuid_val = uuid.uuid4()
    process_monitor.set_run_uuid(run_uuid_val)
    process_monitor.start_monitoring()

    _pipeline_start = time.time()

    def _elapsed() -> str:
        """Return elapsed time since pipeline start as a string."""
        return f"{time.time() - _pipeline_start:.1f}s"

    debug_data = None
    if debug_mode:
        debug_data = {
            "decisions": [],
            "tokens": {
                "prompt": 0,
                "completion": 0,
                "total": 0,
                "cost": 0.0,
                "stages": {},
            },
            "start_timestamp": datetime.now().isoformat(),
            "error": None,
            "completed": False,
        }

    from ..agent.clarifier import generate_clarifier_decision
    from ..agent.direct_response import stream_direct_response_from_conversation
    from ..agent.planner import generate_data_source_selection_plan
    from ..agent.router import generate_routing_decision
    from ..agent.summarizer import stream_research_summary
    from ..agent.tools.database_metadata import fetch_available_data_sources
    from ..utils.input_sanitizer import sanitize_conversation_history
    from ..connections.oauth import fetch_oauth_token
    from ..utils.prompt_loader import load_all_prompts

    try:
        logger.info("Initializing model...")

        process_monitor.start_stage("prompt_cache")
        prompt_count, loaded_prompts = load_all_prompts("research")
        process_monitor.end_stage("prompt_cache")
        process_monitor.add_stage_details(
            "prompt_cache",
            prompts_loaded=prompt_count,
            prompts=loaded_prompts,
        )

        process_monitor.start_stage("oauth_setup")
        token, auth_info = fetch_oauth_token()
        process_monitor.end_stage("oauth_setup")
        process_monitor.add_stage_details(
            "oauth_setup",
            auth_method=auth_info.get("method"),
            client_id=auth_info.get("client_id"),
        )

        if not conversation:
            logger.warning("No conversation provided.")
            yield "Model initialized, but no conversation provided to process."
            return

        process_monitor.start_stage("conversation_processing")
        try:
            processed_conversation = sanitize_conversation_history(conversation)
            logger.info(
                "Conversation processed: %d messages",
                len(processed_conversation["messages"]),
            )
        except ValueError as e:
            logger.warning("Invalid conversation format: %s", e)
            process_monitor.end_stage("conversation_processing", "error")
            yield f"Model initialized, but conversation format is invalid: {str(e)}"
            return
        except Exception as e:
            logger.error("Error processing conversation: %s", e)
            process_monitor.end_stage("conversation_processing", "error")
            yield f"Error processing conversation: {str(e)}"
            return

        if not processed_conversation["messages"]:
            logger.warning("Processed conversation is empty.")
            process_monitor.end_stage("conversation_processing", "error")
            yield "Model initialized, but processed conversation is empty."
            return

        process_monitor.end_stage("conversation_processing")

        # Extract user query for process monitor
        user_messages = [
            m["content"] for m in processed_conversation["messages"]
            if m.get("role") == "user"
        ]
        user_query = user_messages[-1] if user_messages else "No user message"

        process_monitor.add_stage_details(
            "conversation_processing",
            user_query=user_query,
            original_msg_count=processed_conversation.get("original_count"),
            system_filtered=processed_conversation.get("system_filtered"),
            final_msg_count=processed_conversation.get("final_count"),
        )

        available_data_sources = fetch_available_data_sources()
        if data_source_names is not None:
            logger.info("Filtering data sources to: %s", data_source_names)
            available_data_sources = {
                k: v
                for k, v in available_data_sources.items()
                if k in data_source_names
            }
            if not available_data_sources:
                logger.warning(
                    "All data sources filtered out by data_source_names: %s",
                    data_source_names,
                )
                yield (
                    "None of the requested data sources are available. "
                    "Please check your data source selection and try again."
                )
                process_monitor.end_monitoring()
                return

        process_monitor.start_stage("router")
        logger.info("[%s] Getting routing decision...", _elapsed())
        routing_decision, router_usage_details = generate_routing_decision(
            processed_conversation, token, available_data_sources
        )
        process_monitor.end_stage("router")
        if router_usage_details:
            process_monitor.add_llm_call_details_to_stage(
                "router", router_usage_details
            )
        process_monitor.add_stage_details(
            "router",
            function_name=routing_decision.get("function_name"),
            decision=routing_decision,
        )

        if routing_decision["function_name"] == "direct_response":
            logger.info("Using direct response path")
            process_monitor.start_stage("direct_response")
            direct_response_usage_details = None
            stream_iterator = stream_direct_response_from_conversation(
                processed_conversation, token, available_data_sources
            )
            for chunk in stream_iterator:
                if isinstance(chunk, dict) and "usage_details" in chunk:
                    direct_response_usage_details = chunk["usage_details"]
                else:
                    yield chunk
            process_monitor.end_stage("direct_response")
            if direct_response_usage_details:
                process_monitor.add_llm_call_details_to_stage(
                    "direct_response", direct_response_usage_details
                )
            else:
                logger.debug("No usage details received from direct_response stream")

            logger.debug("Direct response completed, ending monitoring")
            process_monitor.end_monitoring()

        elif routing_decision["function_name"] == "database_research":
            logger.info("Using research path")
            process_monitor.start_stage("clarifier")
            logger.info("[%s] Clarifying research needs...", _elapsed())
            clarifier_decision, clarifier_usage_details = generate_clarifier_decision(
                processed_conversation, token, available_data_sources
            )
            process_monitor.end_stage("clarifier")
            if clarifier_usage_details:
                process_monitor.add_llm_call_details_to_stage(
                    "clarifier", clarifier_usage_details
                )
            process_monitor.add_stage_details(
                "clarifier",
                action=clarifier_decision.get("action"),
                decision=clarifier_decision,
            )

            if clarifier_decision["action"] == "ask_clarification":
                logger.info("Essential context needed")
                questions = clarifier_decision["output"].strip()
                yield "Before proceeding with research, please clarify:\n\n" + questions

                logger.debug("Context request completed, ending monitoring")
                process_monitor.end_monitoring()

            elif clarifier_decision["action"] == "request_deep_research_approval":
                logger.info("Source-wide query detected, requesting approval")
                approval_message = clarifier_decision["output"].strip()
                yield approval_message

                logger.debug("Approval request completed, ending monitoring")
                process_monitor.end_monitoring()

            else:
                research_statement = clarifier_decision.get("output", "")
                is_db_wide = clarifier_decision.get("is_db_wide", False)
                deep_research_approved = clarifier_decision.get(
                    "deep_research_approved", False
                )

                logger.info(
                    "Research statement: %s... (is_db_wide=%s, "
                    "deep_research_approved=%s)",
                    research_statement[:100],
                    is_db_wide,
                    deep_research_approved,
                )

                process_monitor.start_stage("planner")
                logger.info("[%s] Creating data source selection plan...", _elapsed())
                ds_selection_plan, planner_usage_list = (
                    generate_data_source_selection_plan(
                        research_statement,
                        token,
                        available_data_sources,
                        process_monitor=process_monitor,
                    )
                )
                selected_data_sources = ds_selection_plan.get("data_sources", [])
                query_embedding = ds_selection_plan.get("query_embedding")
                logger.info(
                    "Data source selection plan created with %d data sources: %s",
                    len(selected_data_sources),
                    selected_data_sources,
                )
                process_monitor.end_stage("planner")
                for usage_details in planner_usage_list:
                    process_monitor.add_llm_call_details_to_stage(
                        "planner", usage_details
                    )
                process_monitor.add_stage_details(
                    "planner",
                    data_source_count=len(selected_data_sources),
                    selected_data_sources=selected_data_sources,
                    decision=ds_selection_plan,
                )

                logger.info(
                    "[%s] Querying data sources: %s",
                    _elapsed(),
                    selected_data_sources,
                )
                yield f"# Research Plan ({_elapsed()})\n\n"
                yield f"{research_statement}\n\n"
                selected_ds_display_names = [
                    available_data_sources.get(data_source_name, {}).get(
                        "name", data_source_name
                    )
                    for data_source_name in selected_data_sources
                ]
                if selected_ds_display_names:
                    if len(selected_ds_display_names) == 1:
                        names_str = selected_ds_display_names[0]
                    elif len(selected_ds_display_names) == 2:
                        names_str = (
                            f"{selected_ds_display_names[0]} and "
                            f"{selected_ds_display_names[1]}"
                        )
                    else:
                        names_str = (
                            ", ".join(selected_ds_display_names[:-1])
                            + f", and {selected_ds_display_names[-1]}"
                        )
                    yield f"Searching {names_str}.\n\n"
                else:
                    yield "No data sources selected for search.\n\n---\n"

                all_findings: FindingsList = []

                if not selected_data_sources:
                    logger.warning(
                        "Data source selection plan is empty, "
                        "skipping data source search."
                    )
                else:
                    # --- Filter resolution step ---
                    from ..agent.filter_resolver import (
                        resolve_filters as resolve_source_filters,
                        FilterResolverError,
                    )

                    try:
                        filter_resolution, filter_usage = (
                            resolve_source_filters(
                                research_statement=research_statement,
                                selected_sources=selected_data_sources,
                                token=token,
                                pre_provided_filters=filters,
                                process_monitor=process_monitor,
                            )
                        )
                        if filter_usage:
                            logger.debug(
                                "Filter resolver usage: %s", filter_usage
                            )

                        if filter_resolution.needs_clarification:
                            logger.info(
                                "Filter resolver needs user clarification"
                            )
                            yield (
                                "Before proceeding, please clarify the "
                                "following filter selections:\n\n"
                                + filter_resolution.clarification_message
                            )
                            process_monitor.end_monitoring()
                            return

                        resolved_filters = (
                            filter_resolution.resolved_filters
                        )
                        logger.info(
                            "Resolved filters: %s", resolved_filters
                        )

                    except FilterResolverError as exc:
                        logger.warning(
                            "Filter resolution failed, proceeding "
                            "without filters: %s",
                            exc,
                        )
                        resolved_filters = {}

                    logger.info(
                        "Starting %d parallel queries...", len(selected_data_sources)
                    )
                    futures = []

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        for i, data_source_name in enumerate(selected_data_sources):
                            query_text = research_statement
                            ds_display_name = available_data_sources.get(
                                data_source_name, {}
                            ).get("name", data_source_name)
                            query_context = {
                                "research_statement": research_statement,
                                "query_embedding": query_embedding,
                                "is_db_wide": is_db_wide,
                                "deep_research_approved": deep_research_approved,
                            }
                            # Apply per-source resolved filters
                            source_filters = resolved_filters.get(
                                data_source_name
                            )
                            if source_filters:
                                query_context["filters"] = source_filters
                            if i > 0:
                                time.sleep(1)
                            future = executor.submit(
                                _execute_data_source_query_task,
                                data_source_name,
                                query_text,
                                token,
                                ds_display_name,
                                i,
                                len(selected_data_sources),
                                query_context,
                            )
                            futures.append(future)
                        logger.info(
                            "Submitted %d queries to thread pool.", len(futures)
                        )

                        for future in concurrent.futures.as_completed(futures):
                            result_data = future.result()
                            data_source_name = result_data["data_source_name"]
                            ds_display_name = result_data["ds_display_name"]
                            task_exception = result_data["exception"]
                            router_result = result_data.get("router_result")

                            status_summary = "Unknown status (Processing error)."
                            if task_exception:
                                status_summary = f"Error: {str(task_exception)}"
                            elif router_result is not None:
                                status_summary = router_result.get(
                                    "status_summary", "No status"
                                )
                                # Collect findings from this data source
                                ds_findings = router_result.get("findings", [])
                                all_findings.extend(ds_findings)
                                logger.info(
                                    "Collected %d findings from %s",
                                    len(ds_findings),
                                    data_source_name,
                                )

                            status_summary = (
                                status_summary.replace("✅", "•")
                                .replace("📄", "•")
                                .replace("❌", "•")
                                .replace("ℹ️", "•")
                                .replace("⚠️", "•")
                                .replace("❓", "•")
                            )
                            status_block = (
                                f"{ds_display_name}: "
                                f"{status_summary} ({_elapsed()})\n\n"
                            )
                            yield status_block

                    logger.info(
                        "[%s] All data source queries completed",
                        _elapsed(),
                    )

                if all_findings:
                    yield "\n\n---\n"
                    yield f"\n\n## Research Summary ({_elapsed()})\n"
                    process_monitor.start_stage("summary")

                    # Consolidate findings and assign ref_ids
                    indexed_findings, master_reference_index = (
                        consolidate_findings_with_refs(all_findings)
                    )

                    # Format findings for summarizer
                    aggregated_detailed_research = format_findings_for_summarizer(
                        indexed_findings
                    )

                    # Debug: Log findings being passed to summarizer
                    logger.info("=" * 60)
                    logger.info("FINDINGS PASSED TO SUMMARIZER:")
                    logger.info("=" * 60)
                    for data_source_name, research_text in (
                        aggregated_detailed_research.items()
                    ):
                        logger.info("--- %s ---", data_source_name)
                        logger.info(
                            "%s",
                            research_text[:2000]
                            if len(research_text) > 2000
                            else research_text,
                        )
                    logger.info("=" * 60)

                    process_monitor.add_stage_details(
                        "summary",
                        num_findings=len(indexed_findings),
                        sources=list(aggregated_detailed_research.keys()),
                    )

                    try:
                        logger.info("Generating summary...")
                        summary_usage_details = None
                        summary_context = {
                            "research_statement": research_statement,
                            "reference_index": master_reference_index,
                        }
                        summary_stream = stream_research_summary(
                            aggregated_detailed_research,
                            token,
                            available_data_sources,
                            summary_context=summary_context,
                        )

                        buffer = ""

                        for chunk in summary_stream:
                            if isinstance(chunk, dict) and "usage_details" in chunk:
                                summary_usage_details = chunk["usage_details"]
                                if buffer:
                                    yield from finalize_reference_replacements(
                                        buffer, master_reference_index
                                    )
                                    buffer = ""
                            else:
                                buffer += chunk
                                processed, buffer = (
                                    process_streaming_reference_buffer(
                                        buffer, master_reference_index
                                    )
                                )
                                if processed:
                                    yield processed
                        process_monitor.end_stage("summary")
                        if summary_usage_details:
                            process_monitor.add_llm_call_details_to_stage(
                                "summary", summary_usage_details
                            )
                        else:
                            logger.debug(
                                "No usage details received from summary stream"
                            )
                    except Exception as summary_exc:
                        logger.error(
                            "Error during summarization: %s",
                            summary_exc,
                            exc_info=True,
                        )
                        if buffer:
                            yield from finalize_reference_replacements(
                                buffer, master_reference_index
                            )
                            buffer = ""
                        err_msg = str(summary_exc)
                        yield f"\n\n**Error during summarization:** {err_msg}"
                        process_monitor.end_stage("summary", "error")
                        process_monitor.add_stage_details(
                            "summary", error=str(summary_exc)
                        )
                    logger.info("[%s] Research completed", _elapsed())
                else:
                    logger.info("No findings returned from any data source")
                    yield (
                        "\n\n---\n\n"
                        "No relevant results were found across the searched "
                        "data sources. "
                        "Try rephrasing your question or broadening the "
                        "search criteria."
                    )

                logger.debug("Research completed, ending monitoring")
                process_monitor.end_monitoring()

        else:
            logger.error(
                "Unknown routing function: %s", routing_decision["function_name"]
            )
            yield "Error: Unable to process query due to internal routing error."
            logger.debug("Ending monitoring due to routing error")
            process_monitor.end_monitoring()

    except Exception as e:
        error_msg = f"Critical error processing request: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if process_monitor.enabled and (
            not hasattr(process_monitor, "end_time") or not process_monitor.end_time
        ):
            process_monitor.end_monitoring()
        process_monitor.add_stage_details("_global", error=error_msg)
        yield f"**Error:** {error_msg}"

    finally:
        if process_monitor.enabled and (
            not hasattr(process_monitor, "end_time") or not process_monitor.end_time
        ):
            logger.debug("Setting process monitoring end_time in finally block")
            process_monitor.end_monitoring()

        if process_monitor.enabled:
            try:
                logger.info(
                    "Logging process monitor data for run %s",
                    process_monitor.run_uuid,
                )

                with get_database_session() as session:
                    result = session.execute(
                        text(
                            """
                            SELECT EXISTS (
                               SELECT FROM information_schema.tables
                               WHERE table_schema = 'public'
                               AND table_name = 'process_monitor_logs'
                            )
                        """
                        )
                    )
                    table_exists = result.scalar()
                    if not table_exists:
                        logger.warning("process_monitor_logs table does not exist")

                    process_monitor.log_to_database(session)
                    logger.info("Process monitor data logged to database")

            except Exception as log_exc:
                logger.error(
                    "Failed to log process monitor data: %s",
                    log_exc,
                    exc_info=True,
                )

            finally:
                import gc

                gc.collect()

        if debug_mode and debug_data is not None and not debug_data.get("error"):
            debug_data["completed"] = True
            if "end_timestamp" not in debug_data:
                final_agent_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                }
                try:
                    for stage_usage in (
                        debug_data.get("tokens", {}).get("stages", {}).values()
                    ):
                        final_agent_usage["prompt_tokens"] += stage_usage.get(
                            "prompt", 0
                        )
                        final_agent_usage["completion_tokens"] += stage_usage.get(
                            "completion", 0
                        )
                        final_agent_usage["total_tokens"] += stage_usage.get(
                            "total", 0
                        )
                        final_agent_usage["cost"] += stage_usage.get("cost", 0.0)
                    debug_data["tokens"]["prompt"] = final_agent_usage[
                        "prompt_tokens"
                    ]
                    debug_data["tokens"]["completion"] = final_agent_usage[
                        "completion_tokens"
                    ]
                    debug_data["tokens"]["total"] = final_agent_usage[
                        "total_tokens"
                    ]
                    debug_data["cost"] = final_agent_usage["cost"]
                except Exception:
                    logger.debug("Could not calculate legacy debug token totals")
                debug_data["end_timestamp"] = datetime.now().isoformat()
            yield f"\n\nDEBUG_DATA:{json.dumps(debug_data)}"


def stream_model_response(
    conversation: Optional[Dict[str, Any]] = None,
    html_callback: Optional[Callable] = None,
    debug_mode: bool = False,
    data_source_names: Optional[List[str]] = None,
    filters: Optional[Dict[str, str]] = None,
) -> Generator[str, None, None]:
    """Process conversations and yield streaming responses.

    Args:
        conversation (Optional[Dict[str, Any]]): Conversation history with a messages
            list.
        html_callback (Optional[Callable]): Deprecated HTML rendering callback.
        debug_mode (bool): When True, appends DEBUG_DATA JSON at the end of the stream.
        data_source_names (Optional[List[str]]): Data sources to restrict queries to.
        filters (Optional[Dict[str, str]]): Optional filters to pass through to
            data source queries and planning.

    Yields:
        str: Streaming research plans, data source status updates, and final answers.
    """
    try:
        yield from _stream_model_workflow(
            conversation, html_callback, debug_mode, data_source_names, filters
        )
    except Exception as e:
        error_msg = f"Error during model execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield f"**Error:** {error_msg}"


async def process_conversation_request_async(
    conversation: List[Dict[str, str]],
    stream: bool = False,
    data_source_names: Optional[List[str]] = None,
    filters: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Async wrapper for FastAPI that processes a conversation request.

    Runs the synchronous model in a thread pool executor to avoid blocking the event
    loop. Collects all streaming chunks and returns a complete response.

    Args:
        conversation (List[Dict[str, str]]): Conversation messages with role/content.
        stream (bool): Whether to enable streaming (reserved for future use).
        data_source_names (Optional[List[str]]): Data sources to restrict queries to.
        filters (Optional[Dict[str, str]]): Optional filters to pass through to
            data source queries and planning.

    Returns:
        Dict[str, Any]: Response text, agent_used, processing_time_ms, token_usage,
            and run_uuid when available.
    """
    import asyncio

    if "_stream" in kwargs:
        stream = kwargs.pop("_stream")

    logger.info("Processing async request: %d messages", len(conversation))

    start_time = time.time()

    def run_sync_model():
        try:
            conversation_dict = {"messages": conversation}
            response_chunks = []
            agent_used = None
            run_uuid = None
            token_usage = None

            for chunk in stream_model_response(
                conversation_dict,
                debug_mode=False,
                data_source_names=data_source_names,
                filters=filters,
            ):
                if isinstance(chunk, str):
                    response_chunks.append(chunk)
                elif isinstance(chunk, dict):
                    if "agent_used" in chunk:
                        agent_used = chunk.get("agent_used")
                    if "run_uuid" in chunk:
                        run_uuid = chunk.get("run_uuid")
                    if "token_usage" in chunk:
                        token_usage = chunk.get("token_usage")

            full_response = "".join(response_chunks)

            if not full_response.strip():
                full_response = "No response was generated. Please try again."

            return {
                "response": full_response,
                "agent_used": agent_used,
                "run_uuid": str(run_uuid) if run_uuid else None,
                "token_usage": token_usage,
            }

        except Exception as e:
            logger.error("Error in sync model execution: %s", e, exc_info=True)
            return {
                "response": f"Error processing request: {str(e)}",
                "agent_used": None,
                "run_uuid": None,
                "token_usage": None,
            }

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, run_sync_model)

    processing_time_ms = int((time.time() - start_time) * 1000)
    result["processing_time_ms"] = processing_time_ms

    logger.info("Request completed: %dms", processing_time_ms)

    return result
