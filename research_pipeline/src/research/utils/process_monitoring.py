"""
Process Monitoring - Pipeline execution tracking and metrics collection.

Provides timing, token usage, and cost tracking for research pipeline stages.
Records when each stage starts and ends, accumulates LLM call metrics, and
persists the data to PostgreSQL for observability and cost analysis.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class ProcessStageMetrics:
    """Metrics container for a single pipeline stage's execution."""

    name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status: str = "not_started"
    llm_calls_data: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark stage as in_progress and record start timestamp."""
        self.start_time = datetime.now(timezone.utc)
        self.status = "in_progress"

    def end(self, status: str = "completed") -> None:
        """Record end timestamp and calculate duration in seconds."""
        self.end_time = datetime.now(timezone.utc)
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = status

    def add_llm_call_details(self, call_details: Dict[str, Any]) -> None:
        """Append an LLM call record (model, tokens, cost, latency)."""
        self.llm_calls_data.append(call_details)

    def add_details(self, **kwargs: Any) -> None:
        """Merge additional key-value pairs into stage details."""
        self.details.update(kwargs)

    def get_total_tokens(self) -> int:
        """Sum prompt and completion tokens across all LLM calls."""
        return sum(
            call.get("prompt_tokens", 0) + call.get("completion_tokens", 0)
            for call in self.llm_calls_data
        )

    def get_total_cost(self) -> float:
        """Sum costs across all LLM calls for this stage."""
        return sum(call.get("cost", 0.0) for call in self.llm_calls_data)


class ProcessMonitoringManager:
    """Coordinates stage-level metrics collection for a single pipeline execution."""

    def __init__(self, enabled: bool = False) -> None:
        """Initialize the monitoring manager."""
        self.enabled = enabled
        self.stages: Dict[str, ProcessStageMetrics] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.run_uuid: Optional[uuid.UUID] = None

    def set_run_uuid(self, run_uuid: uuid.UUID) -> None:
        """Assign an external UUID to correlate with request tracing."""
        if self.enabled:
            self.run_uuid = run_uuid

    def start_monitoring(self) -> None:
        """Initialize monitoring state and generate run UUID if not set."""
        if not self.enabled:
            return
        self.start_time = datetime.now(timezone.utc)
        if self.run_uuid is None:
            self.run_uuid = uuid.uuid4()
        self.stages = {}
        self.end_time = None

    def end_monitoring(self) -> None:
        """Record the overall end timestamp for the pipeline run."""
        if self.enabled:
            self.end_time = datetime.now(timezone.utc)

    def start_stage(self, stage_name: str) -> None:
        """Begin timing a named stage, creating the metrics object if needed."""
        if not self.enabled:
            return
        if stage_name not in self.stages:
            self.stages[stage_name] = ProcessStageMetrics(stage_name)
        self.stages[stage_name].start()

    def end_stage(self, stage_name: str, status: str = "completed") -> None:
        """Finalize timing for a stage with the given completion status."""
        if self.enabled and stage_name in self.stages:
            self.stages[stage_name].end(status)

    def add_llm_call_details_to_stage(
        self, stage_name: str, call_details: Dict[str, Any]
    ) -> None:
        """Record an LLM call's metrics under the specified stage."""
        if self.enabled and stage_name in self.stages:
            self.stages[stage_name].add_llm_call_details(call_details)

    def add_stage_details(self, stage_name: str, **kwargs: Any) -> None:
        """Attach arbitrary metadata to a stage for later analysis."""
        if self.enabled and stage_name in self.stages:
            self.stages[stage_name].add_details(**kwargs)

    def _extract_custom_metadata(
        self, details: Dict[str, Any]
    ) -> Optional[str]:
        """Extract custom metadata from stage details, excluding reserved keys."""
        reserved_keys = {"decision_details", "error"}
        custom = {k: v for k, v in details.items() if k not in reserved_keys}
        if not custom:
            return None
        return json.dumps(custom, default=str)

    def log_to_database(self, session: Any) -> None:
        """Persist all stage metrics to the process_monitor_logs table."""
        if not self.enabled:
            return
        if not self.run_uuid:
            logger.error(
                "Run UUID not set, cannot log process monitor data."
            )
            return
        if not self.stages:
            logger.warning("No stages recorded, skipping database logging.")
            return

        logger.info(
            "Logging process monitor data for run_uuid: %s", self.run_uuid
        )

        insert_query = text(
            """
            INSERT INTO process_monitor_logs (
                run_uuid, model_name, stage_name, stage_start_time,
                stage_end_time, duration_ms, llm_calls, total_tokens,
                total_cost, status, decision_details, error_message,
                custom_metadata
            ) VALUES (
                :run_uuid, :model_name, :stage_name, :stage_start_time,
                :stage_end_time, :duration_ms, :llm_calls, :total_tokens,
                :total_cost, :status, :decision_details, :error_message,
                :custom_metadata
            )
            """
        )

        records = self._prepare_records_for_db()
        if not records:
            logger.warning("No valid stage records prepared for DB logging.")
            return

        try:
            session.execute(insert_query, records)
        except (
            SQLAlchemyError,
            ValueError,
            TypeError,
            RuntimeError,
            OSError,
        ) as err:
            logger.error(
                "Database error during process monitor logging: %s", err
            )
            raise

    def _prepare_records_for_db(self) -> List[Dict[str, Any]]:
        """Transform stage metrics into insert-ready dictionaries."""
        records = []
        for stage in self.stages.values():
            try:
                total_tokens = stage.get_total_tokens()
                total_cost = stage.get_total_cost()
                records.append(
                    {
                        "run_uuid": str(self.run_uuid),
                        "model_name": config.PROCESS_MONITOR_MODEL_NAME,
                        "stage_name": stage.name,
                        "stage_start_time": stage.start_time,
                        "stage_end_time": stage.end_time,
                        "duration_ms": (
                            int(stage.duration * 1000)
                            if stage.duration
                            else None
                        ),
                        "llm_calls": (
                            json.dumps(stage.llm_calls_data)
                            if stage.llm_calls_data
                            else None
                        ),
                        "total_tokens": (
                            total_tokens if total_tokens > 0 else None
                        ),
                        "total_cost": (
                            total_cost if total_cost > 0 else None
                        ),
                        "status": stage.status,
                        "decision_details": stage.details.get(
                            "decision_details"
                        ),
                        "error_message": (
                            stage.details.get("error")
                            if stage.status == "error"
                            else None
                        ),
                        "custom_metadata": self._extract_custom_metadata(
                            stage.details
                        ),
                    }
                )
            except (KeyError, TypeError, AttributeError) as exc:
                logger.error(
                    "Error preparing stage '%s' for DB: %s", stage.name, exc
                )
        return records


_process_monitor: ProcessMonitoringManager = ProcessMonitoringManager(
    enabled=False
)


def set_process_monitoring_enabled(enabled: bool = True) -> None:
    """Toggle process monitoring on or off, reinitializing if state changes."""
    global _process_monitor  # pylint: disable=global-statement
    if _process_monitor.enabled != enabled:
        _process_monitor = ProcessMonitoringManager(enabled=enabled)


def get_process_monitor_instance() -> ProcessMonitoringManager:
    """Return the global ProcessMonitoringManager singleton."""
    return _process_monitor
