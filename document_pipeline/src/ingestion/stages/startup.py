"""Stage 0: Pipeline startup, lock management, and cleanup."""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.llm import LLMClient
from ..utils.postgres import get_connection, verify_connection
from ..utils.config import get_retention_count, load_config
from ..utils.logging_setup import (
    LOGS_DIR,
    get_stage_logger,
    setup_logging,
)
from ..utils.ssl_certificates import setup_ssl

STAGE = "0-STARTUP"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSING_DIR = PROJECT_ROOT / "processing"
LOCK_FILE = PROCESSING_DIR / "pipeline.lock"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

LOCK_EXPIRY_HOURS = 6


def _lock_age_hours(timestamp: float) -> float:
    """Get lock age in hours. Params: timestamp. Returns: float."""
    return (time.time() - timestamp) / 3600


def _safe_unlink(path: Path) -> bool:
    """Delete a path if it exists. Params: path. Returns: bool."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def _read_lock_timestamp() -> float:
    """Read the lock timestamp from JSON. Returns: float."""
    lock_data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    return float(lock_data["timestamp"])


def _write_lock_file(lock_data: dict) -> None:
    """Atomically create the lock file. Params: lock_data. Returns: None."""
    lock_json = json.dumps(lock_data)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(LOCK_FILE, flags)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
            lock_file.write(lock_json)
            lock_file.flush()
            os.fsync(lock_file.fileno())
    except Exception:
        _safe_unlink(LOCK_FILE)
        raise


def _acquire_lock() -> None:
    """Create a lock file or abort if one is active.

    Ensures the processing directory exists, then checks for
    an existing lock. Fresh locks cause the pipeline to abort.
    Stale locks (older than LOCK_EXPIRY_HOURS) are removed.

    Returns:
        None

    Example:
        >>> _acquire_lock()
    """
    logger = get_stage_logger(__name__, STAGE)

    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        lock_data = {
            "timestamp": time.time(),
            "started_at": datetime.now().isoformat(),
            "pid": os.getpid(),
        }
        try:
            _write_lock_file(lock_data)
            logger.info("Pipeline lock acquired")
            return
        except FileExistsError as exc:
            try:
                lock_timestamp = _read_lock_timestamp()
            except FileNotFoundError:
                continue
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as invalid_exc:
                try:
                    modified_at = LOCK_FILE.stat().st_mtime
                except FileNotFoundError:
                    continue
                lock_age_hours = _lock_age_hours(modified_at)
                if lock_age_hours < LOCK_EXPIRY_HOURS:
                    raise RuntimeError(
                        f"Pipeline lock is active but unreadable "
                        f"(age: {lock_age_hours:.1f}h). "
                        f"Another run may be in progress."
                    ) from invalid_exc
                logger.warning(
                    "Unreadable stale lock found (%.1fh old), removing",
                    lock_age_hours,
                )
                _safe_unlink(LOCK_FILE)
                continue

            lock_age_hours = _lock_age_hours(lock_timestamp)
            if lock_age_hours < LOCK_EXPIRY_HOURS:
                raise RuntimeError(
                    f"Pipeline lock is active "
                    f"(age: {lock_age_hours:.1f}h). "
                    f"Another run may be in progress."
                ) from exc
            logger.warning(
                "Stale lock found (%.1fh old), removing",
                lock_age_hours,
            )
            _safe_unlink(LOCK_FILE)


def release_lock() -> None:
    """Remove the lock file. Returns: None."""
    _safe_unlink(LOCK_FILE)


def _processing_contents() -> list:
    """List processing dir items, excluding lock and staging dirs."""
    return [
        item
        for item in PROCESSING_DIR.iterdir()
        if item.name != LOCK_FILE.name
        and not item.name.startswith(".staging_")
    ]


def _clean_stale_processing() -> None:
    """Remove leftover files from a crashed previous run.

    If the processing directory contains files other than
    the lock, a previous run crashed without archiving.
    Archives those files for debugging, then cleans the
    directory so the current run starts fresh.

    Returns:
        None

    Example:
        >>> _clean_stale_processing()
    """
    logger = get_stage_logger(__name__, STAGE)

    contents = _processing_contents()
    if not contents:
        return

    logger.warning("Stale processing files found, archiving")
    _archive_and_clean(contents, "crashed")


def archive_run() -> None:
    """Archive the current run's processing output.

    Zips the processing directory contents (excluding the
    lock file) into the archive directory with a timestamped
    name, then cleans the processing directory. Also prunes
    old archives beyond the retention limit.

    Returns:
        None

    Example:
        >>> archive_run()
    """
    logger = get_stage_logger(__name__, STAGE)

    contents = _processing_contents()
    if not contents:
        return

    _archive_and_clean(contents, "run")
    _prune_old_files(ARCHIVE_DIR, "*.zip", "archives")
    _prune_old_files(LOGS_DIR, "*.log", "logs")
    logger.info("Run archived")


def _archive_and_clean(contents: list, prefix: str) -> None:
    """Zip processing contents to archive and remove them.

    Copies only the specified files to a temp directory
    before zipping, so the lock file is never included.

    Params:
        contents: List of Path objects to archive
        prefix: Archive filename prefix ("run" or "crashed")
    """
    logger = get_stage_logger(__name__, STAGE)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{prefix}_{timestamp}"
    archive_path = ARCHIVE_DIR / archive_name

    staging = PROCESSING_DIR / f".staging_{timestamp}"
    staging.mkdir()
    for item in contents:
        dest = staging / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    shutil.make_archive(str(archive_path), "zip", staging)
    shutil.rmtree(staging)
    logger.info("Archived to %s.zip", archive_name)

    for item in contents:
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def _prune_old_files(directory: Path, pattern: str, label: str) -> None:
    """Delete old files beyond the retention limit.

    Keeps the most recent RETENTION_COUNT files matching
    the pattern, sorted by name (which includes timestamp).

    Params:
        directory: Directory to prune
        pattern: Glob pattern to match files
        label: Label for log message (e.g. "archives", "logs")
    """
    if not directory.exists():
        return

    retention = get_retention_count()
    files = sorted(directory.glob(pattern))
    if len(files) <= retention:
        return

    logger = get_stage_logger(__name__, STAGE)
    to_remove = files[: len(files) - retention]
    for old_file in to_remove:
        old_file.unlink()
    logger.info("Pruned %d old %s", len(to_remove), label)


def run_startup(require_llm: bool = True) -> tuple[Any, LLMClient | None]:
    """Initialize the pipeline and return connections.

    Handles config loading, logging setup, SSL, lock
    acquisition, crash recovery cleanup, and verification
    of database connectivity. LLM connectivity is verified
    only when required for downstream stages.

    Params:
        require_llm: Whether to initialize and verify the LLM client

    Returns:
        tuple of (psycopg2 connection, LLMClient | None)

    Example:
        >>> conn, llm = run_startup(require_llm=False)
    """
    load_config()
    setup_logging()
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Ingestion pipeline starting")

    setup_ssl()

    conn = None
    llm = None
    _acquire_lock()
    try:
        _clean_stale_processing()

        if require_llm:
            logger.info("Initializing LLM client")
            llm = LLMClient()
            llm.test_connection()
        else:
            logger.info("Skipping LLM initialization")

        logger.info("Connecting to database")
        conn = get_connection()
        verify_connection(conn)
    except Exception:
        if conn is not None:
            conn.close()
        release_lock()
        raise

    logger.info("Startup complete")
    return conn, llm
