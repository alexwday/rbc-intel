"""Tests for main module."""

from unittest.mock import MagicMock, patch

import pytest

from ingestion.main import main


def test_main_calls_stages_in_order():
    """main() runs startup, discovery, extraction, archive, release."""
    mock_conn = MagicMock()
    mock_llm = MagicMock()
    with (
        patch(
            "ingestion.main.run_startup",
            return_value=(mock_conn, mock_llm),
        ) as mock_startup,
        patch("ingestion.main.run_discovery") as mock_discovery,
        patch("ingestion.main.run_extraction") as mock_extraction,
        patch("ingestion.main.archive_run") as mock_archive,
        patch("ingestion.main.release_lock") as mock_release,
    ):
        main()

        mock_startup.assert_called_once()
        mock_discovery.assert_called_once_with(mock_conn)
        mock_extraction.assert_called_once_with(mock_llm)
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
        patch("ingestion.main.archive_run"),
        patch("ingestion.main.release_lock") as mock_release,
    ):
        try:
            main()
        except RuntimeError:
            pass

        mock_conn.close.assert_called_once()
        mock_release.assert_called_once()


def test_main_releases_lock_if_startup_fails():
    """Lock release still runs when startup itself raises."""
    with (
        patch(
            "ingestion.main.run_startup",
            side_effect=RuntimeError("startup failed"),
        ),
        patch("ingestion.main.release_lock") as mock_release,
    ):
        with pytest.raises(RuntimeError, match="startup failed"):
            main()

    mock_release.assert_called_once()
