"""Ingestion pipeline entry point."""

import argparse
import sys
from collections.abc import Sequence

from .stages.content_preparation import run_content_preparation
from .stages.discovery import run_discovery
from .stages.enrichment import run_enrichment
from .stages.extraction import run_extraction
from .stages.finalization import run_finalization
from .stages.storage import run_storage
from .stages.startup import archive_run, release_lock, run_startup


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the ingestion entry point.

    Params:
        argv: Optional argument sequence, excluding the executable name

    Returns:
        argparse.Namespace with parsed flags

    Example:
        >>> parse_args(["--storage-only"]).storage_only
        True
    """
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.main",
        description="Run the document ingestion pipeline.",
    )
    parser.add_argument(
        "--storage-only",
        action="store_true",
        help=(
            "Skip discovery through finalization and run only the storage "
            "stage against the current source snapshot and existing outputs."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else [])


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ingestion pipeline or storage-only sync.

    Delegates to stage functions: startup initializes
    connections and validates infrastructure, then each
    pipeline stage runs in sequence. The lock is always
    released on exit, whether the run succeeds or fails.

    Params:
        argv: Optional CLI args, excluding executable name

    Returns:
        None

    Example:
        >>> main(["--storage-only"])
    """
    args = parse_args(argv)
    conn = None
    try:
        conn, llm = run_startup(require_llm=not args.storage_only)
        if not args.storage_only:
            assert llm is not None
            run_discovery(conn)
            run_extraction(llm)
            run_content_preparation(llm)
            run_enrichment(llm)
            run_finalization(llm)
        run_storage(conn)
        archive_run()
    finally:
        try:
            if conn is not None:
                conn.close()
        finally:
            release_lock()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
