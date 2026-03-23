"""PostgreSQL connection and catalog table operations."""

import logging

import psycopg2

from ..utils.config import get_database_config
from ..utils.file_types import FileRecord

logger = logging.getLogger(__name__)

FETCH_CATALOG_SQL = """
SELECT data_source, filter_1, filter_2, filter_3,
       filename, filetype, file_size, date_last_modified,
       file_hash, file_path
FROM document_catalog;
"""

DELETE_CATALOG_SQL = """
DELETE FROM document_catalog WHERE file_path = ANY(%s);
"""


def get_connection():
    """Open a psycopg2 connection using DATABASE_URL.

    Returns:
        psycopg2 connection object

    Example:
        >>> conn = get_connection()
        >>> conn.closed
        0
    """
    return psycopg2.connect(**get_database_config())


VERIFY_TABLE_SQL = """
SELECT 1 FROM information_schema.tables
WHERE table_name = 'document_catalog';
"""


def verify_connection(conn) -> bool:
    """Validate database connectivity and required tables.

    Checks that the connection is alive and that the
    document_catalog table exists.

    Params:
        conn: psycopg2 connection

    Returns:
        bool — True if all checks pass

    Example:
        >>> verify_connection(conn)
        True
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        logger.info("Database connection test passed")
    except Exception:
        logger.error("Database connection test failed")
        raise

    with conn.cursor() as cur:
        cur.execute(VERIFY_TABLE_SQL)
        if not cur.fetchone():
            raise RuntimeError("document_catalog table does not exist")
    logger.info("document_catalog table verified")
    return True


def fetch_catalog_records(conn) -> list:
    """Fetch all rows from document_catalog as FileRecords.

    Params:
        conn: psycopg2 connection

    Returns:
        list[FileRecord] — one per catalog row

    Example:
        >>> records = fetch_catalog_records(conn)
        >>> len(records)
        42
    """
    with conn.cursor() as cur:
        cur.execute(FETCH_CATALOG_SQL)
        rows = cur.fetchall()
    return [
        FileRecord(
            data_source=row[0],
            filter_1=row[1],
            filter_2=row[2],
            filter_3=row[3],
            filename=row[4],
            filetype=row[5],
            file_size=row[6],
            date_last_modified=row[7],
            file_hash=row[8],
            file_path=row[9],
        )
        for row in rows
    ]


def delete_catalog_records(conn, paths: list) -> int:
    """Delete catalog rows by file_path list.

    Params:
        conn: psycopg2 connection
        paths: list of file_path strings to remove

    Returns:
        int — number of rows deleted

    Example:
        >>> delete_catalog_records(conn, ["/data/old.pdf"])
        1
    """
    if not paths:
        return 0
    with conn.cursor() as cur:
        cur.execute(DELETE_CATALOG_SQL, (paths,))
        count = cur.rowcount
    conn.commit()
    return count
