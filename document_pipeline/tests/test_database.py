"""Tests for ingestion.connections.postgres."""

from unittest.mock import MagicMock, patch

import pytest

from ingestion.connections.postgres import (
    DELETE_CATALOG_SQL,
    FETCH_CATALOG_SQL,
    delete_catalog_records,
    fetch_catalog_records,
    get_connection,
    verify_connection,
)


@patch("ingestion.connections.postgres.get_database_config")
@patch("ingestion.connections.postgres.psycopg2")
def test_get_connection(mock_psycopg2, mock_config):
    """get_connection calls psycopg2.connect with config params."""
    db_config = {
        "host": "localhost",
        "port": "5432",
        "dbname": "test",
        "user": "dev",
        "password": "",
    }
    mock_config.return_value = db_config
    mock_conn = MagicMock()
    mock_psycopg2.connect.return_value = mock_conn

    result = get_connection()

    mock_psycopg2.connect.assert_called_once_with(**db_config)
    assert result is mock_conn


def test_verify_connection_success():
    """verify_connection passes when DB is up and table exists."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (1,)
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    assert verify_connection(mock_conn) is True


def test_verify_connection_db_failure():
    """verify_connection raises when DB is unreachable."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = Exception("connection refused")
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with pytest.raises(Exception, match="connection refused"):
        verify_connection(mock_conn)


def test_verify_connection_missing_table():
    """verify_connection raises when table does not exist."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match="document_catalog"):
        verify_connection(mock_conn)


def test_fetch_catalog_records_empty():
    """Returns empty list when table has no rows."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    result = fetch_catalog_records(mock_conn)

    mock_cursor.execute.assert_called_once_with(FETCH_CATALOG_SQL)
    assert result == []


def test_fetch_catalog_records_returns_file_records():
    """Rows are converted to FileRecord objects."""
    row = (
        "policy",
        "2026",
        "Q1",
        "",
        "doc.pdf",
        "pdf",
        1024,
        1700000000.0,
        "abc123",
        "/data/policy/2026/Q1/doc.pdf",
    )
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [row]
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    result = fetch_catalog_records(mock_conn)

    assert len(result) == 1
    assert result[0].data_source == "policy"
    assert result[0].filter_1 == "2026"
    assert result[0].filter_2 == "Q1"
    assert result[0].file_hash == "abc123"
    assert result[0].supported is True


def test_delete_catalog_records_empty_list():
    """No-op when paths list is empty."""
    mock_conn = MagicMock()
    count = delete_catalog_records(mock_conn, [])
    assert count == 0
    mock_conn.cursor.assert_not_called()


def test_delete_catalog_records_deletes_by_path():
    """Deletes rows matching the given paths."""
    paths = ["/data/old.pdf", "/data/gone.docx"]
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 2
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    count = delete_catalog_records(mock_conn, paths)

    mock_cursor.execute.assert_called_once_with(DELETE_CATALOG_SQL, (paths,))
    mock_conn.commit.assert_called_once()
    assert count == 2
