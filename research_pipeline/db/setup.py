"""
Database setup script for the research pipeline.

Creates required tables (research_registry, prompts) and loads seed data.
Run from the research_pipeline/ directory:

    python -m db.setup
    python -m db.setup --seed-only   # skip table creation
    python -m db.setup --schema myschema
"""

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent
SCHEMAS_DIR = DB_DIR / "schemas"
SEED_DIR = DB_DIR / "seed"


def _get_connection_string() -> str:
    """Build psycopg2 connection string from environment variables."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "rbc_intel")
    user = os.getenv("DB_USER", "")
    password = os.getenv("DB_PASSWORD", "")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def _execute_sql_file(cursor: "psycopg2.cursor", filepath: Path, schema: str) -> None:
    """Execute a SQL file with schema substitution."""
    sql = filepath.read_text()
    if schema != "public":
        sql = sql.replace("research_registry", f"{schema}.research_registry")
        sql = sql.replace("INTO prompts", f"INTO {schema}.prompts")
        sql = sql.replace("FROM prompts", f"FROM {schema}.prompts")
        sql = sql.replace(
            "CREATE TABLE IF NOT EXISTS prompts",
            f"CREATE TABLE IF NOT EXISTS {schema}.prompts",
        )
        sql = sql.replace(
            "CREATE TABLE IF NOT EXISTS research_registry",
            f"CREATE TABLE IF NOT EXISTS {schema}.research_registry",
        )
    cursor.execute(sql)
    logger.info("Executed %s", filepath.name)


def run_setup(schema: str = "public", seed_only: bool = False) -> None:
    """Create tables and load seed data.

    Args:
        schema: Database schema to use.
        seed_only: If True, skip table creation and only load seed data.
    """
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed — run: pip install psycopg2-binary")
        sys.exit(1)

    conn_string = _get_connection_string()
    logger.info("Connecting to database...")

    with psycopg2.connect(conn_string) as conn:
        with conn.cursor() as cursor:
            if not seed_only:
                logger.info("Creating tables in schema '%s'...", schema)
                for sql_file in sorted(SCHEMAS_DIR.glob("*.sql")):
                    _execute_sql_file(cursor, sql_file, schema)

            logger.info("Loading seed data...")
            for sql_file in sorted(SEED_DIR.glob("*.sql")):
                _execute_sql_file(cursor, sql_file, schema)

        conn.commit()
        logger.info("Setup complete.")


def main() -> None:
    """Parse arguments and run setup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Research pipeline database setup")
    parser.add_argument(
        "--schema",
        default=os.getenv("DB_SCHEMA", "public"),
        help="Database schema (default: DB_SCHEMA env var or 'public')",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Skip table creation, only load seed data",
    )
    args = parser.parse_args()
    run_setup(schema=args.schema, seed_only=args.seed_only)


if __name__ == "__main__":
    main()
