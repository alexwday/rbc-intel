"""Ingestion pipeline entry point."""

from .stages.content_preparation import run_content_preparation
from .stages.discovery import run_discovery
from .stages.extraction import run_extraction
from .stages.startup import archive_run, release_lock, run_startup


def main() -> None:
    """Run the ingestion pipeline.

    Delegates to stage functions: startup initializes
    connections and validates infrastructure, then each
    pipeline stage runs in sequence. The lock is always
    released on exit, whether the run succeeds or fails.

    Returns:
        None

    Example:
        >>> main()
    """
    conn = None
    try:
        conn, llm = run_startup()
        run_discovery(conn)
        run_extraction(llm)
        run_content_preparation(llm)
        archive_run()
    finally:
        try:
            if conn is not None:
                conn.close()
        finally:
            release_lock()


if __name__ == "__main__":  # pragma: no cover
    main()
