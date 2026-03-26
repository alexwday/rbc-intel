"""
FastAPI REST API for the Research Pipeline.

This module provides a FastAPI interface to the Research Pipeline,
exposing the core functionality as REST endpoints.

Endpoints:
    POST /chat: Process a conversation through Research Pipeline agents
    GET /health: Health check endpoint
    GET /: Root endpoint with API info
    GET /data-sources: List available data sources
    GET /filters: Return distinct filter values from documents
    GET /db-inspector: Preview core database tables
    POST /reset: Clear server caches
    GET /process-monitor/runs: List recent process monitor runs
    GET /process-monitor/run/{run_uuid}: Get detailed stages for a run

Functions:
    get_chat_processor: Lazy import for async chat model
    get_streaming_chat_processor: Lazy import for streaming chat model
    stream_chat_response: Async generator for streaming responses
    chat_endpoint: Main chat endpoint handler
    health_check: Health check handler
    root: Root endpoint handler
    get_data_sources: Data source listing handler
    get_filters: Filter values handler
    get_db_inspector: Database table preview handler
    reset_server: Cache reset handler
    startup_event: FastAPI startup hook
    shutdown_event: FastAPI shutdown hook

Classes:
    ChatMessage: Single message in a conversation
    ChatRequest: Incoming chat request payload
    ChatResponse: Non-streaming response payload
    HealthResponse: Health check response payload
"""

from typing import Any, Callable, Dict, List, Optional
import asyncio
import importlib
import logging
import queue
import threading
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .utils.logging_format import configure_root_logger
from .utils.config import config

API_VERSION = "1.0.0"

configure_root_logger()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Pipeline API",
    description="Research Pipeline - AI-powered research and analysis assistant",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    """Single message in a conversation."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    messages: List[ChatMessage] = Field(..., description="Conversation history")
    stream: bool = Field(default=False, description="Enable streaming response")
    data_sources: Optional[List[str]] = Field(
        default=None, description="List of data source names to query"
    )
    filters: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description=(
            "Optional per-source subfolder filters, keyed by data_source. "
            "Example: {\"refined_suite\": {\"filter_1\": \"core\"}}. "
            "Omit to let the pipeline auto-resolve filters from context."
        ),
    )


class ResearchCombo(BaseModel):
    """A single research combo — data source + period + bank."""

    data_source: str = Field(..., description="Data source (e.g., investor_slides, pillar3_disclosure)")
    period: str = Field(default="", description="Reporting period filter (e.g., 2026_Q1)")
    bank: str = Field(default="", description="Bank filter (e.g., RBC, TD)")


class ResearchRequest(BaseModel):
    """Direct research request from Aegis."""

    research_statement: str = Field(..., description="Detailed research query from Aegis")
    combos: List[ResearchCombo] = Field(..., description="List of data_source/period/bank combos to research")


class ChatResponse(BaseModel):
    """Non-streaming chat response payload."""

    response: str = Field(..., description="Research Pipeline response")


class HealthResponse(BaseModel):
    """Health check response payload."""

    status: str
    version: str


def _serialize_db_value(value: Any) -> Any:
    """Convert DB values into JSON-serializable representations."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()

    if isinstance(value, dict):
        return {str(k): _serialize_db_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_serialize_db_value(v) for v in value]

    return str(value)


def _lazy_import(module_path: str, attr_name: str) -> Any:
    """
    Perform a lazy import to avoid circular dependencies.

    Args:
        module_path: The module path to import from.
        attr_name: The attribute name to import from the module.

    Returns:
        The imported attribute.

    Raises:
        ImportError: If the module or attribute cannot be imported.
    """
    module = importlib.import_module(module_path, package="research")
    return getattr(module, attr_name)


def get_chat_processor() -> Callable:
    """
    Lazily import the async chat model to avoid circular dependencies.

    Returns:
        The process_conversation_request_async function from orchestrator.model.

    Raises:
        ImportError: If the chat model module cannot be imported.
    """
    try:
        return _lazy_import(
            ".orchestrator.model", "process_conversation_request_async"
        )
    except (ImportError, AttributeError) as exc:
        logger.error(
            "Failed to import chat model. "
            "Make sure to add the async wrapper to model.py"
        )
        raise ImportError(
            "Chat model not properly configured for async operation"
        ) from exc


def get_streaming_chat_processor() -> Callable:
    """
    Lazily import the streaming chat model to avoid circular dependencies.

    Returns:
        The streaming model generator function from orchestrator.model.

    Raises:
        ImportError: If the streaming chat model module cannot be imported.
    """
    try:
        return _lazy_import(".orchestrator.model", "stream_model_response")
    except (ImportError, AttributeError) as exc:
        logger.error("Failed to import streaming chat model")
        raise ImportError("Streaming chat model not properly configured") from exc


class StreamingError(Exception):
    """Exception raised during streaming response generation."""


async def stream_chat_response(
    conversation: List[Dict[str, str]],
    data_source_names: Optional[List[str]] = None,
    filters: Optional[Dict[str, str]] = None,
):
    """
    Async generator that streams chat responses from the Research Pipeline model.

    Uses a thread-based queue to bridge the synchronous model generator
    with the async FastAPI streaming response.

    Args:
        conversation: List of message dictionaries with 'role' and 'content' keys.
        data_source_names: Optional list of data source names to restrict scope.
        filters: Optional filter constraints (filter_1, filter_2, filter_3).

    Yields:
        String chunks of the response as they are generated.
    """
    try:
        model_func = get_streaming_chat_processor()
        conversation_dict = {"messages": conversation}
        chunk_queue: queue.Queue = queue.Queue()
        exception_container: List[Optional[Exception]] = [None]

        def run_sync_generator():
            """Execute the synchronous model generator in a background thread."""
            try:
                for chunk in model_func(
                    conversation_dict,
                    debug_mode=False,
                    data_source_names=data_source_names,
                    filters=filters,
                ):
                    if isinstance(chunk, str):
                        chunk_queue.put(chunk)
                chunk_queue.put(None)
            except (
                ImportError,
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                OSError,
                ConnectionError,
                TimeoutError,
            ) as exc:
                exception_container[0] = exc
                chunk_queue.put(None)

        thread = threading.Thread(target=run_sync_generator)
        thread.start()

        while True:
            try:
                chunk = chunk_queue.get(timeout=0.1)
                if chunk is None:
                    break
                yield chunk
                await asyncio.sleep(0)
            except queue.Empty:
                stored_exception = exception_container[0]
                if stored_exception is not None:
                    raise StreamingError(str(stored_exception)) from stored_exception
                await asyncio.sleep(0.01)
                continue

        thread.join(timeout=1)

        stored_exception = exception_container[0]
        if stored_exception is not None:
            raise StreamingError(str(stored_exception)) from stored_exception

    except (ImportError, RuntimeError, ValueError, StreamingError) as exc:
        logger.error("Streaming error: %s", str(exc), exc_info=True)
        yield f"Error: {exc}"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Process a conversation through the Research Pipeline.

    Accepts a conversation history and routes it through the appropriate
    agents to generate a response.

    Args:
        request: ChatRequest containing messages, stream flag, optional
                 data_sources and filters.

    Returns:
        StreamingResponse if stream=True, otherwise ChatResponse with full response.

    Raises:
        HTTPException: 500 error if processing fails.
    """
    try:
        logger.info(
            "Received chat request with %d messages, stream=%s",
            len(request.messages),
            request.stream,
        )

        conversation = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        if request.stream:
            logger.info("Returning streaming response")
            return StreamingResponse(
                stream_chat_response(
                    conversation, request.data_sources, request.filters
                ),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        logger.info("Returning complete response")
        process_conversation_request_async = get_chat_processor()
        result = await process_conversation_request_async(
            conversation,
            stream=False,
            data_source_names=request.data_sources,
            filters=request.filters,
        )

        logger.info("Chat request processed successfully")
        return ChatResponse(response=result.get("response", ""))

    except (ImportError, RuntimeError, ValueError) as exc:
        logger.error("Chat endpoint error: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {exc}",
        ) from exc


@app.post("/research")
async def research_endpoint(request: ResearchRequest):
    """Direct research endpoint for Aegis integration.

    Receives a research statement and combos (data_source + period + bank),
    researches each combo in parallel, and returns a synthesized response.
    """
    try:
        logger.info(
            "Research request: %d combos, statement: '%s...'",
            len(request.combos),
            request.research_statement[:80],
        )

        from .orchestrator.direct_research import execute_direct_research

        combos = [combo.model_dump() for combo in request.combos]

        async def stream_research():
            """Bridge sync generator to async streaming."""
            import asyncio

            loop = asyncio.get_event_loop()
            chunk_queue: queue.Queue = queue.Queue()
            exception_container: List[Optional[Exception]] = [None]

            def run_sync():
                try:
                    for chunk in execute_direct_research(
                        request.research_statement, combos
                    ):
                        if isinstance(chunk, str):
                            chunk_queue.put(chunk)
                    chunk_queue.put(None)
                except Exception as exc:
                    exception_container[0] = exc
                    chunk_queue.put(None)

            thread = threading.Thread(target=run_sync)
            thread.start()

            while True:
                try:
                    chunk = chunk_queue.get(timeout=0.1)
                    if chunk is None:
                        break
                    yield chunk
                    await asyncio.sleep(0)
                except queue.Empty:
                    if exception_container[0] is not None:
                        yield f"\nError: {exception_container[0]}\n"
                        break
                    await asyncio.sleep(0.01)

            thread.join(timeout=1)
            if exception_container[0] is not None:
                yield f"\nError: {exception_container[0]}\n"

        return StreamingResponse(
            stream_research(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as exc:
        logger.error("Research endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {exc}",
        ) from exc


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Verify the API is running and properly configured.

    Returns:
        HealthResponse with status and version.

    Raises:
        HTTPException: 503 error if configuration validation fails.
    """
    try:
        config.validate_required_environment()

        return HealthResponse(status="healthy", version=API_VERSION)
    except (ValueError, RuntimeError, AttributeError) as exc:
        logger.error("Health check failed: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {exc}",
        ) from exc


@app.get("/")
async def root():
    """
    Return basic information about the API.

    Returns:
        Dictionary with API name, documentation URL, health URL, and version.
    """
    return {
        "message": "Research Pipeline API",
        "docs": "/docs",
        "health": "/health",
        "version": API_VERSION,
    }


@app.get("/data-sources")
async def get_data_sources():
    """
    Return available data sources from the registry.

    Used by the frontend to dynamically populate data source filter checkboxes.

    Returns:
        Dictionary with 'data_sources' key containing list of data source info dicts.

    Raises:
        HTTPException: 500 error if data source retrieval fails.
    """
    try:
        fetch_available_data_sources = _lazy_import(
            ".agent.tools.database_metadata", "fetch_available_data_sources"
        )
        data_sources = fetch_available_data_sources()

        result = []
        for data_source, ds_config in data_sources.items():
            result.append(
                {
                    "id": data_source,
                    "display_name": ds_config.get("name", data_source),
                }
            )

        result.sort(key=lambda x: x["display_name"])

        logger.info("Returning %d data sources", len(result))
        return {"data_sources": result}

    except (ImportError, RuntimeError, ValueError) as exc:
        logger.error("Failed to get data sources: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve data sources: {exc}",
        ) from exc


@app.get("/filters")
async def get_filters(data_source: Optional[str] = None):
    """
    Return distinct filter values from the documents table.

    Args:
        data_source: Optional data source name to filter by.

    Returns:
        Dictionary with 'filters' key containing list of filter value dicts.

    Raises:
        HTTPException: 500 error if filter retrieval fails.
    """
    from sqlalchemy import text
    from .connections.postgres import get_database_session, get_database_schema

    schema = get_database_schema()

    try:
        with get_database_session() as session:
            if data_source is not None:
                query = text(
                    f"SELECT DISTINCT filter_1, filter_2, filter_3 "
                    f"FROM {schema}.documents "
                    f"WHERE data_source = :data_source "
                    f"ORDER BY filter_1, filter_2, filter_3"
                )
                rows = session.execute(
                    query, {"data_source": data_source}
                ).mappings().all()
            else:
                query = text(
                    f"SELECT DISTINCT filter_1, filter_2, filter_3 "
                    f"FROM {schema}.documents "
                    f"ORDER BY filter_1, filter_2, filter_3"
                )
                rows = session.execute(query).mappings().all()

            filters = []
            for row in rows:
                filters.append(
                    {
                        "filter_1": row["filter_1"],
                        "filter_2": row["filter_2"],
                        "filter_3": row["filter_3"],
                    }
                )

            logger.info("Returning %d filter combinations", len(filters))
            return {"filters": filters}

    except Exception as exc:
        logger.error("Failed to get filters: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve filters: {exc}",
        ) from exc


@app.get("/db-inspector")
async def get_db_inspector(limit: int = 50, offset: int = 0):
    """
    Return preview rows from core database tables.

    Uses the same PostgreSQL connection configured at server startup.

    Args:
        limit: Number of rows per table to return (1-200).
        offset: Pagination offset (>=0).

    Returns:
        Dictionary with connection metadata and table previews.

    Raises:
        HTTPException: 500 error if inspection query fails.
    """
    from sqlalchemy import text
    from .connections.postgres import get_database_session, get_database_schema

    schema = get_database_schema()
    bounded_limit = max(1, min(limit, 200))
    bounded_offset = max(0, offset)

    table_map = {
        "research_registry": "research_registry",
        "documents": "documents",
        "document_chunks": "document_chunks",
    }

    try:
        tables: Dict[str, Dict[str, Any]] = {}
        with get_database_session() as session:
            for table_key, table_name in table_map.items():
                qualified_name = f"{schema}.{table_name}"

                column_rows = session.execute(
                    text(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = :schema AND table_name = :table_name
                        ORDER BY ordinal_position
                        """
                    ),
                    {"schema": schema, "table_name": table_name},
                ).fetchall()
                columns = [row[0] for row in column_rows]

                total_rows = session.execute(
                    text(f"SELECT COUNT(*) FROM {qualified_name}")
                ).scalar_one()

                preview_rows = session.execute(
                    text(
                        f"SELECT * FROM {qualified_name} "
                        f"LIMIT :limit OFFSET :offset"
                    ),
                    {"limit": bounded_limit, "offset": bounded_offset},
                ).mappings().all()

                serialized_rows: List[Dict[str, Any]] = []
                for row in preview_rows:
                    serialized_rows.append(
                        {key: _serialize_db_value(value) for key, value in row.items()}
                    )

                tables[table_key] = {
                    "table_name": qualified_name,
                    "columns": columns,
                    "rows": serialized_rows,
                    "total_rows": total_rows,
                    "limit": bounded_limit,
                    "offset": bounded_offset,
                }

        return {
            "connection": {
                "host": config.DB_HOST,
                "port": config.DB_PORT,
                "database": config.DB_NAME,
                "user": config.DB_USER,
            },
            "tables": tables,
        }

    except Exception as exc:
        logger.error("Failed to inspect database tables: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to inspect database tables: {exc}",
        ) from exc


@app.post("/reset")
async def reset_server():
    """
    Clear all server caches and reload configurations.

    Invalidates the database metadata cache, forcing a fresh load
    from PostgreSQL on the next request.

    Returns:
        Dictionary with reset status and message.

    Raises:
        HTTPException: 500 error if cache invalidation fails.
    """
    try:
        get_metadata_repository = _lazy_import(
            ".agent.tools.database_metadata", "get_metadata_repository"
        )
        repo = get_metadata_repository()
        repo.invalidate_cache()

        logger.info("Server caches cleared successfully")
        return {"status": "reset", "message": "Server caches cleared successfully"}

    except (ImportError, RuntimeError, ValueError, AttributeError) as exc:
        logger.error("Failed to reset server: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset server: {exc}",
        ) from exc


@app.get("/process-monitor/runs")
async def get_process_monitor_runs(limit: int = 20):
    """Get recent process monitor runs with their first user message."""
    from .connections.postgres import get_database_session, get_database_schema
    from sqlalchemy import text

    schema = get_database_schema()

    try:
        with get_database_session() as session:
            result = session.execute(
                text(f"""
                    SELECT
                        run_uuid,
                        MIN(stage_start_time) as start_time,
                        SUM(duration_ms) as total_duration_ms,
                        MAX(custom_metadata->>'user_query') as user_query
                    FROM {schema}.process_monitor_logs
                    GROUP BY run_uuid
                    ORDER BY MIN(stage_start_time) DESC
                    LIMIT :limit
                """),
                {"limit": limit},
            )
            rows = result.mappings().all()

            runs = []
            for row in rows:
                runs.append(
                    {
                        "run_uuid": str(row["run_uuid"]),
                        "start_time": (
                            row["start_time"].isoformat()
                            if row["start_time"]
                            else None
                        ),
                        "total_duration_ms": row["total_duration_ms"],
                        "user_query": row["user_query"] or "No query recorded",
                    }
                )

            return {"runs": runs}

    except Exception as exc:
        logger.error("Failed to get process monitor runs: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/process-monitor/run/{run_uuid}")
async def get_process_monitor_run(run_uuid: str):
    """Get detailed stages for a specific run."""
    from .connections.postgres import get_database_session, get_database_schema
    from sqlalchemy import text

    schema = get_database_schema()

    try:
        with get_database_session() as session:
            result = session.execute(
                text(f"""
                    SELECT
                        stage_name,
                        stage_start_time,
                        stage_end_time,
                        duration_ms,
                        total_tokens,
                        total_cost,
                        status,
                        decision_details,
                        custom_metadata
                    FROM {schema}.process_monitor_logs
                    WHERE run_uuid = :run_uuid
                    ORDER BY stage_start_time
                """),
                {"run_uuid": run_uuid},
            )
            rows = result.mappings().all()

            stages = []
            for row in rows:
                stages.append(
                    {
                        "stage_name": row["stage_name"],
                        "start_time": (
                            row["stage_start_time"].isoformat()
                            if row["stage_start_time"]
                            else None
                        ),
                        "end_time": (
                            row["stage_end_time"].isoformat()
                            if row["stage_end_time"]
                            else None
                        ),
                        "duration_ms": row["duration_ms"],
                        "total_tokens": row["total_tokens"],
                        "total_cost": (
                            float(row["total_cost"])
                            if row["total_cost"]
                            else None
                        ),
                        "status": row["status"],
                        "decision_details": row["decision_details"],
                        "custom_metadata": row["custom_metadata"],
                    }
                )

            return {"run_uuid": run_uuid, "stages": stages}

    except Exception as exc:
        logger.error("Failed to get process monitor run: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.on_event("startup")
async def startup_event():
    """
    Perform startup validation and initialization.

    Validates environment configuration and logs startup status.

    Raises:
        ValueError: If configuration validation fails.
    """
    logger.info("Starting Research Pipeline API...")

    if not config.validate_required_environment():
        raise ValueError("Configuration validation failed")

    logger.info("Research Pipeline API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Perform cleanup on application shutdown.

    Logs shutdown message for monitoring purposes.
    """
    logger.info("Shutting down Research Pipeline API...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "research.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
