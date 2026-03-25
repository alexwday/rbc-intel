"""Tests for ingestion.utils.postgres."""

from unittest.mock import MagicMock, patch

import pytest

from ingestion.utils.postgres import (
    CATALOG_TABLE_DDL,
    STORAGE_TABLE_COLUMNS,
    STORAGE_TABLE_ORDER,
    VERIFY_TABLE_SQL,
    build_delete_catalog_sql,
    build_fetch_catalog_sql,
    delete_catalog_records,
    ensure_storage_tables,
    fetch_catalog_records,
    get_connection,
    refresh_storage_tables,
    verify_connection,
)


@patch("ingestion.utils.postgres.get_database_config")
@patch("ingestion.utils.postgres.psycopg2")
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
    assert mock_conn.commit.call_count == 1
    executed = [call.args for call in mock_cursor.execute.call_args_list]
    assert executed[1][0].startswith("CREATE SCHEMA IF NOT EXISTS")
    assert executed[2][0] == CATALOG_TABLE_DDL.format(schema="public")
    assert executed[3] == (VERIFY_TABLE_SQL, ("public",))


def test_verify_connection_db_failure():
    """verify_connection raises when DB is unreachable."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = Exception("connection refused")
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with pytest.raises(Exception, match="connection refused"):
        verify_connection(mock_conn)


def test_verify_connection_missing_table_after_create():
    """verify_connection raises when the table still cannot be found."""
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

    mock_cursor.execute.assert_called_once_with(
        build_fetch_catalog_sql("public")
    )
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

    mock_cursor.execute.assert_called_once_with(
        build_delete_catalog_sql("public"), (paths,)
    )
    mock_conn.commit.assert_called_once()
    assert count == 2


def test_fetch_catalog_records_uses_configured_schema(monkeypatch):
    """Catalog reads honor DB_SCHEMA."""
    monkeypatch.setenv("DB_SCHEMA", "analytics")
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    fetch_catalog_records(mock_conn)

    mock_cursor.execute.assert_called_once_with(
        build_fetch_catalog_sql("analytics")
    )


def test_ensure_storage_tables_creates_all_tables():
    """Storage DDL covers the catalog and storage tables."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    ensure_storage_tables(mock_conn)

    executed_sql = [
        call.args[0] for call in mock_cursor.execute.call_args_list
    ]
    assert executed_sql[0] == "CREATE SCHEMA IF NOT EXISTS public;"
    assert executed_sql[1] == "CREATE EXTENSION IF NOT EXISTS vector;"
    assert executed_sql[2] == CATALOG_TABLE_DDL.format(schema="public")
    expected_count = 3 + len(STORAGE_TABLE_COLUMNS) - 1
    assert len(executed_sql) == expected_count


def test_refresh_storage_tables_truncates_and_copies(tmp_path):
    """Storage sync replaces all tables from the CSV masters."""
    csv_paths = {}
    for table_name in STORAGE_TABLE_ORDER:
        csv_path = tmp_path / f"{table_name}.csv"
        csv_path.write_text(
            ",".join(STORAGE_TABLE_COLUMNS[table_name]) + "\n",
            encoding="utf-8",
        )
        csv_paths[table_name] = csv_path

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    refresh_storage_tables(mock_conn, csv_paths)

    truncate_sql = mock_cursor.execute.call_args_list[-1].args[0]
    assert truncate_sql.startswith("TRUNCATE TABLE public.document_catalog")
    assert mock_cursor.copy_expert.call_count == len(STORAGE_TABLE_ORDER)
    assert mock_conn.commit.call_count == 1


def test_refresh_storage_tables_rolls_back_on_copy_failure(tmp_path):
    """Storage sync rolls back if COPY fails."""
    csv_path = tmp_path / "document_catalog.csv"
    csv_path.write_text(
        ",".join(STORAGE_TABLE_COLUMNS["document_catalog"]) + "\n",
        encoding="utf-8",
    )
    csv_paths = {table_name: csv_path for table_name in STORAGE_TABLE_ORDER}
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.copy_expert.side_effect = RuntimeError("copy failed")
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match="copy failed"):
        refresh_storage_tables(mock_conn, csv_paths)

    mock_conn.rollback.assert_called_once()
