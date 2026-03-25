"""Tests for main module."""

from unittest.mock import MagicMock, patch

import pytest

from ingestion.main import main, parse_args


def test_main_calls_stages_in_order():
    """main() runs startup, ingestion stages, and archive in order."""
    mock_conn = MagicMock()
    mock_llm = MagicMock()
    with (
        patch(
            "ingestion.main.run_startup",
            return_value=(mock_conn, mock_llm),
        ) as mock_startup,
        patch("ingestion.main.run_discovery") as mock_discovery,
        patch("ingestion.main.run_extraction") as mock_extraction,
        patch("ingestion.main.run_content_preparation") as mock_prep,
        patch("ingestion.main.run_enrichment") as mock_enrichment,
        patch("ingestion.main.run_finalization") as mock_finalization,
        patch("ingestion.main.run_storage") as mock_storage,
        patch("ingestion.main.archive_run") as mock_archive,
        patch("ingestion.main.release_lock") as mock_release,
    ):
        main()

        mock_startup.assert_called_once_with(require_llm=True)
        mock_discovery.assert_called_once_with(mock_conn)
        mock_extraction.assert_called_once_with(mock_llm)
        mock_prep.assert_called_once_with(mock_llm)
        mock_enrichment.assert_called_once_with(mock_llm)
        mock_finalization.assert_called_once_with(mock_llm)
        mock_storage.assert_called_once_with(mock_conn)
        mock_archive.assert_called_once()
        mock_conn.close.assert_called_once()
        mock_release.assert_called_once()


def test_main_releases_lock_on_failure():
    """Lock is released even when a stage fails."""
    mock_conn = MagicMock()
    mock_llm = MagicMock()
    with (
        patch(
            "ingestion.main.run_startup",
            return_value=(mock_conn, mock_llm),
        ),
        patch(
            "ingestion.main.run_discovery",
            side_effect=RuntimeError("boom"),
        ),
        patch("ingestion.main.run_extraction"),
        patch("ingestion.main.run_content_preparation"),
        patch("ingestion.main.run_enrichment"),
        patch("ingestion.main.run_finalization"),
        patch("ingestion.main.run_storage"),
        patch("ingestion.main.archive_run"),
        patch("ingestion.main.release_lock") as mock_release,
    ):
        try:
            main()
        except RuntimeError:
            pass

        mock_conn.close.assert_called_once()
        mock_release.assert_called_once()


def test_main_storage_only_skips_llm_stages():
    """--storage-only runs startup and storage without LLM stages."""
    mock_conn = MagicMock()
    with (
        patch(
            "ingestion.main.run_startup",
            return_value=(mock_conn, None),
        ) as mock_startup,
        patch("ingestion.main.run_discovery") as mock_discovery,
        patch("ingestion.main.run_extraction") as mock_extraction,
        patch("ingestion.main.run_content_preparation") as mock_prep,
        patch("ingestion.main.run_enrichment") as mock_enrichment,
        patch("ingestion.main.run_finalization") as mock_finalization,
        patch("ingestion.main.run_storage") as mock_storage,
        patch("ingestion.main.archive_run") as mock_archive,
        patch("ingestion.main.release_lock") as mock_release,
    ):
        main(["--storage-only"])

    mock_startup.assert_called_once_with(require_llm=False)
    mock_discovery.assert_not_called()
    mock_extraction.assert_not_called()
    mock_prep.assert_not_called()
    mock_enrichment.assert_not_called()
    mock_finalization.assert_not_called()
    mock_storage.assert_called_once_with(mock_conn)
    mock_archive.assert_called_once()
    mock_conn.close.assert_called_once()
    mock_release.assert_called_once()


def test_main_releases_lock_if_startup_fails():
    """Lock release still runs when startup itself raises."""
    with (
        patch(
            "ingestion.main.run_startup",
            side_effect=RuntimeError("startup failed"),
        ),
        patch("ingestion.main.run_storage"),
        patch("ingestion.main.release_lock") as mock_release,
    ):
        with pytest.raises(RuntimeError, match="startup failed"):
            main()

    mock_release.assert_called_once()


def test_parse_args_storage_only_flag():
    """parse_args reads the storage-only CLI flag."""
    args = parse_args(["--storage-only"])

    assert args.storage_only is True
