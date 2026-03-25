"""PostgreSQL database connections via SQLAlchemy with connection pooling."""

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from ..utils.config import config

logger = logging.getLogger(__name__)


class _ConnectionManager:
    """Manages singleton database connections."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        self._engine: Optional[Engine] = None
        self._factory: Optional[Callable[[], Session]] = None

    def get_database_engine(self) -> Engine:
        """Return a pooled SQLAlchemy engine.

        Returns:
            Shared engine instance.
        """
        if self._engine is None:
            params = _get_database_params()
            dsn = build_database_dsn(params)
            logger.info("Creating SQLAlchemy engine with connection pooling")
            self._engine = create_engine(
                dsn,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False,
            )
        return self._engine

    def get_session_factory(self) -> Callable[[], Session]:
        """Return a session factory bound to the engine.

        Returns:
            Factory that yields sessions.
        """
        if self._factory is None:
            engine = self.get_database_engine()
            self._factory = sessionmaker(bind=engine, expire_on_commit=False)
        return self._factory


_connection_manager = _ConnectionManager()


def _get_database_params() -> Dict[str, Any]:
    """Return database connection parameters from environment configuration."""
    logger.debug("Getting database parameters from environment configuration")
    return config.get_database_params()


def build_database_dsn(params: Dict[str, Any]) -> str:
    """Build SQLAlchemy DSN from database parameters.

    Args:
        params: Database connection parameters.

    Returns:
        SQLAlchemy connection string.

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    raw_hosts = params.get("host", "")
    raw_ports = str(params.get("port", "")).strip()
    database = params.get("dbname")
    user = params.get("user")
    password = params.get("password")

    if not raw_hosts:
        raise ValueError("Host is not set or is empty.")
    hosts = [host.strip() for host in raw_hosts.split(",") if host.strip()]
    if not hosts:
        raise ValueError("Host is not set or is empty.")

    if not raw_ports:
        raise ValueError("Port is not set or is empty.")
    if "," in raw_ports:
        ports = [port.strip() for port in raw_ports.split(",") if port.strip()]
    else:
        ports = [raw_ports] * len(hosts)
    if len(ports) != len(hosts):
        raise ValueError("The number of ports must match the number of hosts.")

    if not database:
        raise ValueError("Database name is not set.")
    if not user:
        raise ValueError("Database user is not set.")
    if password is None:
        raise ValueError("Database password is not set.")

    host_port = ",".join(f"{host}:{port}" for host, port in zip(hosts, ports))
    hosts_csv = ",".join(hosts)
    ports_csv = ",".join(ports)
    safe_user = quote_plus(str(user))
    safe_password = quote_plus(str(password))
    safe_database = quote_plus(str(database))

    logger.info("Using database: %s", database)
    logger.info("Using host(s): %s", host_port)
    logger.info("Using port(s): %s", ports)

    is_local = all(h in ("localhost", "127.0.0.1") for h in hosts)
    sslmode = "prefer" if is_local else "require"
    logger.info("Using SSL mode: %s (local=%s)", sslmode, is_local)

    if len(hosts) == 1:
        return (
            f"postgresql+psycopg2://{safe_user}:{safe_password}"
            f"@{host_port}/{safe_database}"
            f"?sslmode={sslmode}&target_session_attrs=read-write"
        )
    return (
        f"postgresql+psycopg2://{safe_user}:{safe_password}@/{safe_database}"
        f"?host={quote_plus(hosts_csv)}&port={quote_plus(ports_csv)}"
        f"&sslmode={sslmode}&target_session_attrs=read-write"
    )


def get_database_schema() -> str:
    """Return the configured database schema name.

    Returns:
        Schema name from DB_SCHEMA env var, defaults to 'public'.
    """
    return config.DB_SCHEMA


@contextmanager
def get_database_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session with automatic commit/rollback handling.

    Yields:
        Active SQLAlchemy session.
    """
    factory = _connection_manager.get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        raise exc
    finally:
        session.close()


def get_database_engine() -> Engine:
    """Return the shared SQLAlchemy engine for direct access.

    Returns:
        Shared SQLAlchemy engine.
    """
    return _connection_manager.get_database_engine()
