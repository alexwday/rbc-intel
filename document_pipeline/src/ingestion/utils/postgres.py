"""PostgreSQL connection and pipeline storage operations."""

import logging
from pathlib import Path

import psycopg2

from .config import (
    get_database_config,
    get_database_schema,
)
from .file_types import FileRecord

logger = logging.getLogger(__name__)

FETCH_CATALOG_SQL = """
SELECT data_source, filter_1, filter_2, filter_3,
       filename, filetype, file_size, date_last_modified,
       file_hash, file_path
FROM {schema}.document_catalog;
"""

DELETE_CATALOG_SQL = """
DELETE FROM {schema}.document_catalog WHERE file_path = ANY(%s);
"""

VERIFY_TABLE_SQL = """
SELECT 1 FROM information_schema.tables
WHERE table_schema = %s AND table_name = 'document_catalog';
"""

CATALOG_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {schema}.document_catalog (
    data_source TEXT NOT NULL,
    filter_1 TEXT NOT NULL,
    filter_2 TEXT NOT NULL,
    filter_3 TEXT NOT NULL,
    filename TEXT NOT NULL,
    filetype TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    date_last_modified DOUBLE PRECISION NOT NULL,
    file_hash TEXT NOT NULL,
    file_path TEXT PRIMARY KEY
);
"""

STORAGE_TABLE_COLUMNS = {
    "document_catalog": [
        "data_source",
        "filter_1",
        "filter_2",
        "filter_3",
        "filename",
        "filetype",
        "file_size",
        "date_last_modified",
        "file_hash",
        "file_path",
    ],
    "documents": [
        "document_id",
        "file_path",
        "file_name",
        "filetype",
        "data_source",
        "filter_1",
        "filter_2",
        "filter_3",
        "file_size",
        "date_last_modified",
        "file_hash",
        "title",
        "publication_date",
        "authors_text",
        "document_type",
        "abstract",
        "metadata_json",
        "document_summary",
        "document_description",
        "document_usage",
        "structure_type",
        "structure_confidence",
        "degradation_signals_json",
        "summary_embedding",
        "page_count",
        "primary_section_count",
        "subsection_count",
        "chunk_count",
        "dense_table_count",
        "extraction_metadata_json",
    ],
    "document_sections": [
        "section_id",
        "document_id",
        "file_path",
        "section_number",
        "title",
        "page_start",
        "page_end",
        "page_count",
        "overview",
        "key_topics_json",
        "key_metrics_json",
        "key_findings_json",
        "notable_facts_json",
        "is_fallback",
        "summary_json",
    ],
    "document_subsections": [
        "subsection_id",
        "section_id",
        "document_id",
        "file_path",
        "section_number",
        "subsection_number",
        "title",
        "page_start",
        "page_end",
        "page_count",
        "overview",
        "key_topics_json",
        "key_metrics_json",
        "key_findings_json",
        "notable_facts_json",
        "is_fallback",
        "summary_json",
    ],
    "document_chunks": [
        "chunk_id",
        "document_id",
        "file_path",
        "chunk_number",
        "page_number",
        "content",
        "primary_section_number",
        "primary_section_name",
        "subsection_number",
        "subsection_name",
        "hierarchy_path",
        "primary_section_page_count",
        "subsection_page_count",
        "embedding_prefix",
        "embedding",
        "is_dense_table_description",
        "dense_table_routing_json",
        "metadata_json",
    ],
    "document_dense_tables": [
        "dense_table_id",
        "document_id",
        "file_path",
        "region_id",
        "used_range",
        "sheet_name",
        "page_title",
        "description_generation_mode",
        "replacement_content",
        "routing_metadata_json",
        "dense_table_description_json",
        "dense_table_eda_json",
        "raw_content_json",
    ],
    "document_keywords": [
        "keyword_id",
        "document_id",
        "file_path",
        "keyword",
        "page_number",
        "page_title",
        "section",
        "embedding",
    ],
    "document_metrics": [
        "metric_id",
        "document_id",
        "file_path",
        "metric_name",
        "page_number",
        "sheet_name",
        "region_id",
        "used_range",
        "platform",
        "sub_platform",
        "periods_available_json",
        "embedding",
    ],
    "document_sheet_summaries": [
        "sheet_summary_id",
        "document_id",
        "file_path",
        "sheet_name",
        "handling_mode",
        "summary",
        "usage",
    ],
    "document_sheet_context_chains": [
        "chain_id",
        "document_id",
        "file_path",
        "sheet_index",
        "sheet_name",
        "context_sheet_indices_json",
    ],
}

STORAGE_TABLE_DDLS = {
    "documents": """
CREATE TABLE IF NOT EXISTS {schema}.documents (
    document_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    filetype TEXT NOT NULL,
    data_source TEXT NOT NULL,
    filter_1 TEXT NOT NULL,
    filter_2 TEXT NOT NULL,
    filter_3 TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    date_last_modified DOUBLE PRECISION NOT NULL,
    file_hash TEXT NOT NULL,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL,
    authors_text TEXT NOT NULL,
    document_type TEXT NOT NULL,
    abstract TEXT NOT NULL,
    metadata_json JSONB NOT NULL,
    document_summary TEXT NOT NULL,
    document_description TEXT NOT NULL,
    document_usage TEXT NOT NULL,
    structure_type TEXT NOT NULL,
    structure_confidence TEXT NOT NULL,
    degradation_signals_json JSONB NOT NULL,
    summary_embedding VECTOR(3072),
    page_count INTEGER NOT NULL,
    primary_section_count INTEGER NOT NULL,
    subsection_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    dense_table_count INTEGER NOT NULL,
    extraction_metadata_json JSONB NOT NULL
);
""",
    "document_sections": """
CREATE TABLE IF NOT EXISTS {schema}.document_sections (
    section_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    section_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    page_count INTEGER NOT NULL,
    overview TEXT NOT NULL,
    key_topics_json JSONB NOT NULL,
    key_metrics_json JSONB NOT NULL,
    key_findings_json JSONB NOT NULL,
    notable_facts_json JSONB NOT NULL,
    is_fallback BOOLEAN NOT NULL,
    summary_json JSONB NOT NULL
);
""",
    "document_subsections": """
CREATE TABLE IF NOT EXISTS {schema}.document_subsections (
    subsection_id TEXT PRIMARY KEY,
    section_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    section_number INTEGER NOT NULL,
    subsection_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    page_count INTEGER NOT NULL,
    overview TEXT NOT NULL,
    key_topics_json JSONB NOT NULL,
    key_metrics_json JSONB NOT NULL,
    key_findings_json JSONB NOT NULL,
    notable_facts_json JSONB NOT NULL,
    is_fallback BOOLEAN NOT NULL,
    summary_json JSONB NOT NULL
);
""",
    "document_chunks": """
CREATE TABLE IF NOT EXISTS {schema}.document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    primary_section_number INTEGER NOT NULL,
    primary_section_name TEXT NOT NULL,
    subsection_number INTEGER NOT NULL,
    subsection_name TEXT NOT NULL,
    hierarchy_path TEXT NOT NULL,
    primary_section_page_count INTEGER NOT NULL,
    subsection_page_count INTEGER NOT NULL,
    embedding_prefix TEXT NOT NULL,
    embedding VECTOR(3072),
    is_dense_table_description BOOLEAN NOT NULL,
    dense_table_routing_json JSONB NOT NULL,
    metadata_json JSONB NOT NULL
);
""",
    "document_dense_tables": """
CREATE TABLE IF NOT EXISTS {schema}.document_dense_tables (
    dense_table_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    region_id TEXT NOT NULL,
    used_range TEXT NOT NULL,
    sheet_name TEXT NOT NULL,
    page_title TEXT NOT NULL,
    description_generation_mode TEXT NOT NULL,
    replacement_content TEXT NOT NULL,
    routing_metadata_json JSONB NOT NULL,
    dense_table_description_json JSONB NOT NULL,
    dense_table_eda_json JSONB NOT NULL,
    raw_content_json JSONB NOT NULL
);
""",
    "document_keywords": """
CREATE TABLE IF NOT EXISTS {schema}.document_keywords (
    keyword_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    keyword TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    page_title TEXT NOT NULL,
    section TEXT NOT NULL,
    embedding VECTOR(3072)
);
""",
    "document_metrics": """
CREATE TABLE IF NOT EXISTS {schema}.document_metrics (
    metric_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    sheet_name TEXT NOT NULL,
    region_id TEXT NOT NULL,
    used_range TEXT NOT NULL,
    platform TEXT NOT NULL,
    sub_platform TEXT NOT NULL,
    periods_available_json JSONB NOT NULL,
    embedding VECTOR(3072)
);
""",
    "document_sheet_summaries": """
CREATE TABLE IF NOT EXISTS {schema}.document_sheet_summaries (
    sheet_summary_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    sheet_name TEXT NOT NULL,
    handling_mode TEXT NOT NULL,
    summary TEXT NOT NULL,
    usage TEXT NOT NULL
);
""",
    "document_sheet_context_chains": """
CREATE TABLE IF NOT EXISTS {schema}.document_sheet_context_chains (
    chain_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    sheet_index INTEGER NOT NULL,
    sheet_name TEXT NOT NULL,
    context_sheet_indices_json JSONB NOT NULL
);
""",
}

STORAGE_TABLE_ORDER = list(STORAGE_TABLE_COLUMNS.keys())


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


def _qualified_table_name(schema: str, table_name: str) -> str:
    """Build a validated schema-qualified table name."""
    return f"{schema}.{table_name}"


def build_fetch_catalog_sql(schema: str) -> str:
    """Build the catalog SELECT statement. Params: schema. Returns: str."""
    return FETCH_CATALOG_SQL.format(schema=schema)


def build_delete_catalog_sql(schema: str) -> str:
    """Build the catalog DELETE statement. Params: schema. Returns: str."""
    return DELETE_CATALOG_SQL.format(schema=schema)


def _create_catalog_table(conn) -> None:
    """Ensure the catalog schema and table exist.

    Params: conn. Returns: None.
    """
    schema = get_database_schema()
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
        cur.execute(CATALOG_TABLE_DDL.format(schema=schema))


def verify_connection(conn) -> bool:
    """Validate database connectivity and required tables.

    Checks that the connection is alive and ensures the
    document_catalog table exists in the configured schema.

    Params:
        conn: psycopg2 connection

    Returns:
        bool — True if all checks pass

    Example:
        >>> verify_connection(conn)
        True
    """
    schema = get_database_schema()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        logger.info("Database connection test passed")
    except Exception:
        logger.error("Database connection test failed")
        raise

    _create_catalog_table(conn)
    with conn.cursor() as cur:
        cur.execute(VERIFY_TABLE_SQL, (schema,))
        if not cur.fetchone():
            raise RuntimeError("document_catalog table does not exist")
    conn.commit()
    logger.info("document_catalog table verified in schema %s", schema)
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
    schema = get_database_schema()
    with conn.cursor() as cur:
        cur.execute(build_fetch_catalog_sql(schema))
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
    schema = get_database_schema()
    with conn.cursor() as cur:
        cur.execute(build_delete_catalog_sql(schema), (paths,))
        count = cur.rowcount
    conn.commit()
    return count


def ensure_storage_tables(conn) -> None:
    """Ensure the storage schema and tables exist.

    Params: conn. Returns: None.
    """
    schema = get_database_schema()
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(CATALOG_TABLE_DDL.format(schema=schema))
        for ddl in STORAGE_TABLE_DDLS.values():
            cur.execute(ddl.format(schema=schema))


def refresh_storage_tables(
    conn,
    csv_paths: dict[str, Path],
) -> None:
    """Replace storage tables with the current CSV masters.

    Params:
        conn: psycopg2 connection
        csv_paths: mapping of table name to CSV path

    Returns:
        None
    """
    schema = get_database_schema()
    missing = [
        table_name
        for table_name in STORAGE_TABLE_ORDER
        if table_name not in csv_paths
    ]
    if missing:
        raise ValueError(
            f"Missing CSV paths for storage tables: {', '.join(missing)}"
        )

    try:
        ensure_storage_tables(conn)
        qualified_tables = ", ".join(
            _qualified_table_name(schema, table_name)
            for table_name in STORAGE_TABLE_ORDER
        )
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {qualified_tables};")
            for table_name in STORAGE_TABLE_ORDER:
                csv_path = Path(csv_paths[table_name])
                columns = ", ".join(STORAGE_TABLE_COLUMNS[table_name])
                copy_sql = (
                    f"COPY {_qualified_table_name(schema, table_name)} "
                    f"({columns}) FROM STDIN WITH "
                    f"(FORMAT CSV, HEADER TRUE, NULL '\\\\N')"
                )
                with csv_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    cur.copy_expert(copy_sql, handle)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
