"""Tests for stages.startup."""

import json
import logging
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from ingestion.stages.startup import (
    LOCK_EXPIRY_HOURS,
    _acquire_lock,
    _archive_and_clean,
    _clean_stale_processing,
    _prune_old_files,
    _write_lock_file,
    archive_run,
    release_lock,
    run_startup,
)

# --- _acquire_lock ---


def test_acquire_lock_creates_file(tmp_path, monkeypatch):
    """Creates a lock file with timestamp and PID."""
    proc = tmp_path / "processing"
    lock = proc / "pipeline.lock"
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)

    _acquire_lock()

    assert lock.exists()
    data = json.loads(lock.read_text())
    assert "timestamp" in data
    assert "pid" in data


def test_acquire_lock_aborts_if_active(tmp_path, monkeypatch):
    """Raises if a fresh lock exists."""
    proc = tmp_path / "processing"
    proc.mkdir()
    lock = proc / "pipeline.lock"
    lock.write_text(json.dumps({"timestamp": time.time()}))
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)

    with pytest.raises(RuntimeError, match="lock is active"):
        _acquire_lock()


def test_acquire_lock_removes_stale(tmp_path, monkeypatch):
    """Replaces a stale lock older than expiry threshold."""
    proc = tmp_path / "processing"
    proc.mkdir()
    lock = proc / "pipeline.lock"
    old_time = time.time() - (LOCK_EXPIRY_HOURS + 1) * 3600
    lock.write_text(json.dumps({"timestamp": old_time}))
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)

    _acquire_lock()

    data = json.loads(lock.read_text())
    assert data["timestamp"] > old_time


def test_acquire_lock_replaces_unreadable_stale_lock(tmp_path, monkeypatch):
    """Unreadable stale locks are removed and replaced."""
    proc = tmp_path / "processing"
    proc.mkdir()
    lock = proc / "pipeline.lock"
    lock.write_text("{bad json")
    old_time = time.time() - (LOCK_EXPIRY_HOURS + 1) * 3600
    os.utime(lock, (old_time, old_time))
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)

    _acquire_lock()

    data = json.loads(lock.read_text())
    assert data["timestamp"] > old_time


def test_acquire_lock_aborts_if_unreadable_but_fresh(tmp_path, monkeypatch):
    """Unreadable recent locks are treated as active."""
    proc = tmp_path / "processing"
    proc.mkdir()
    lock = proc / "pipeline.lock"
    lock.write_text("{bad json")
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)

    with pytest.raises(RuntimeError, match="unreadable"):
        _acquire_lock()


def test_acquire_lock_retries_if_lock_disappears(tmp_path, monkeypatch):
    """Retries if the lock vanishes during collision handling."""
    proc = tmp_path / "processing"
    lock = proc / "pipeline.lock"
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)
    with (
        patch(
            "ingestion.stages.startup._write_lock_file",
            side_effect=[FileExistsError, None],
        ) as mock_write,
        patch(
            "ingestion.stages.startup._read_lock_timestamp",
            side_effect=FileNotFoundError,
        ),
    ):
        _acquire_lock()

    assert mock_write.call_count == 2


def test_acquire_lock_retries_if_invalid_lock_disappears(
    tmp_path, monkeypatch
):
    """Retries if an unreadable lock disappears before stat succeeds."""
    proc = tmp_path / "processing"
    lock = proc / "pipeline.lock"
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)
    with (
        patch(
            "ingestion.stages.startup._write_lock_file",
            side_effect=[FileExistsError, None],
        ) as mock_write,
        patch(
            "ingestion.stages.startup._read_lock_timestamp",
            side_effect=json.JSONDecodeError("bad", "", 0),
        ),
        patch(
            "pathlib.Path.stat",
            side_effect=FileNotFoundError,
        ),
    ):
        _acquire_lock()

    assert mock_write.call_count == 2


# --- release_lock ---


def test_release_lock_removes_file(tmp_path, monkeypatch):
    """Removes the lock file."""
    lock = tmp_path / "pipeline.lock"
    lock.write_text("lock")
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)

    release_lock()

    assert not lock.exists()


def test_release_lock_noop_if_missing(tmp_path, monkeypatch):
    """No error if lock file does not exist."""
    lock = tmp_path / "pipeline.lock"
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)

    release_lock()


# --- _clean_stale_processing ---


def test_clean_stale_noop_if_empty(tmp_path, monkeypatch):
    """No action when processing dir has only lock file."""
    proc = tmp_path / "processing"
    proc.mkdir()
    lock = proc / "pipeline.lock"
    lock.write_text("{}")
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)
    monkeypatch.setattr(
        "ingestion.stages.startup.ARCHIVE_DIR", tmp_path / "archive"
    )

    _clean_stale_processing()

    assert not (tmp_path / "archive").exists()


def test_clean_stale_archives_leftovers(tmp_path, monkeypatch):
    """Archives stale files from a crashed run."""
    proc = tmp_path / "processing"
    proc.mkdir()
    lock = proc / "pipeline.lock"
    lock.write_text("{}")
    (proc / "discovery.json").write_text("{}")
    archive = tmp_path / "archive"
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)
    monkeypatch.setattr("ingestion.stages.startup.ARCHIVE_DIR", archive)

    _clean_stale_processing()

    zips = list(archive.glob("crashed_*.zip"))
    assert len(zips) == 1
    remaining = [f for f in proc.iterdir() if f.name != "pipeline.lock"]
    assert not remaining


# --- archive_run ---


def test_archive_run_zips_and_cleans(tmp_path, monkeypatch):
    """Archives current run output and prunes old archives."""
    proc = tmp_path / "processing"
    proc.mkdir()
    lock = proc / "pipeline.lock"
    lock.write_text("{}")
    (proc / "discovery.json").write_text("{}")
    archive = tmp_path / "archive"
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)
    monkeypatch.setattr("ingestion.stages.startup.ARCHIVE_DIR", archive)

    archive_run()

    zips = list(archive.glob("run_*.zip"))
    assert len(zips) == 1
    remaining = [f for f in proc.iterdir() if f.name != "pipeline.lock"]
    assert not remaining


def test_archive_run_noop_if_empty(tmp_path, monkeypatch):
    """No archive when processing dir has only lock file."""
    proc = tmp_path / "processing"
    proc.mkdir()
    lock = proc / "pipeline.lock"
    lock.write_text("{}")
    archive = tmp_path / "archive"
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)
    monkeypatch.setattr("ingestion.stages.startup.ARCHIVE_DIR", archive)

    archive_run()

    assert not archive.exists()


# --- _archive_and_clean ---


def test_archive_and_clean_handles_subdirs(tmp_path, monkeypatch):
    """Cleans subdirectories in processing dir."""
    proc = tmp_path / "processing"
    subdir = proc / "traces"
    subdir.mkdir(parents=True)
    (subdir / "trace.json").write_text("{}")
    archive = tmp_path / "archive"
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.ARCHIVE_DIR", archive)

    contents = [subdir]
    _archive_and_clean(contents, "test")

    assert not list(proc.iterdir())
    assert len(list(archive.glob("test_*.zip"))) == 1


def test_write_lock_file_cleans_up_on_write_failure(tmp_path, monkeypatch):
    """Partial lock writes are removed on failure."""
    lock = tmp_path / "pipeline.lock"
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)
    with patch(
        "ingestion.stages.startup.os.fsync",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError, match="disk full"):
            _write_lock_file({"timestamp": time.time()})

    assert not lock.exists()


# --- _prune_old_files ---


def test_prune_noop_under_limit(tmp_path, monkeypatch):
    """No pruning when file count is within retention."""
    monkeypatch.setenv("RETENTION_COUNT", "5")
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "run_001.zip").write_bytes(b"z")

    _prune_old_files(archive, "*.zip", "archives")

    assert len(list(archive.glob("*.zip"))) == 1


def test_prune_removes_oldest(tmp_path, monkeypatch):
    """Removes oldest files beyond retention limit."""
    monkeypatch.setenv("RETENTION_COUNT", "3")
    archive = tmp_path / "archive"
    archive.mkdir()
    for i in range(6):
        (archive / f"run_{i:03d}.zip").write_bytes(b"z")

    _prune_old_files(archive, "*.zip", "archives")

    remaining = sorted(archive.glob("*.zip"))
    assert len(remaining) == 3
    assert remaining[0].name == "run_003.zip"


def test_prune_noop_if_dir_missing(tmp_path):
    """No error if directory does not exist."""
    _prune_old_files(tmp_path / "nope", "*.zip", "archives")


# --- run_startup ---


def test_run_startup_releases_lock_on_failure(tmp_path, monkeypatch):
    """Lock is released when a startup check fails."""
    monkeypatch.setattr("ingestion.utils.logging_setup.LOGS_DIR", tmp_path)
    proc = tmp_path / "processing"
    lock = proc / "pipeline.lock"
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)
    monkeypatch.setattr(
        "ingestion.stages.startup.ARCHIVE_DIR",
        tmp_path / "archive",
    )
    with (
        patch("ingestion.stages.startup.load_config"),
        patch("ingestion.stages.startup.setup_ssl"),
        patch("ingestion.stages.startup.LLMClient") as mock_llm_cls,
    ):
        mock_llm_cls.return_value.test_connection.side_effect = RuntimeError(
            "LLM down"
        )

        with pytest.raises(RuntimeError, match="LLM down"):
            run_startup()

    assert not lock.exists()


def test_run_startup_closes_conn_on_verify_failure(tmp_path, monkeypatch):
    """DB connections are closed if verification fails."""
    monkeypatch.setattr("ingestion.utils.logging_setup.LOGS_DIR", tmp_path)
    proc = tmp_path / "processing"
    lock = proc / "pipeline.lock"
    monkeypatch.setattr("ingestion.stages.startup.PROCESSING_DIR", proc)
    monkeypatch.setattr("ingestion.stages.startup.LOCK_FILE", lock)
    monkeypatch.setattr(
        "ingestion.stages.startup.ARCHIVE_DIR",
        tmp_path / "archive",
    )
    with (
        patch("ingestion.stages.startup.load_config"),
        patch("ingestion.stages.startup.setup_ssl"),
        patch("ingestion.stages.startup.LLMClient") as mock_llm_cls,
        patch("ingestion.stages.startup.get_connection") as mock_get_conn,
        patch(
            "ingestion.stages.startup.verify_connection",
            side_effect=RuntimeError("db verify failed"),
        ),
    ):
        mock_llm_cls.return_value = MagicMock()
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        with pytest.raises(RuntimeError, match="db verify failed"):
            run_startup()

    mock_conn.close.assert_called_once()
    assert not lock.exists()


def test_run_startup_returns_conn_and_llm(tmp_path, monkeypatch):
    """run_startup returns (conn, llm) tuple."""
    monkeypatch.setattr("ingestion.utils.logging_setup.LOGS_DIR", tmp_path)
    monkeypatch.setattr(
        "ingestion.stages.startup.PROCESSING_DIR", tmp_path / "processing"
    )
    monkeypatch.setattr(
        "ingestion.stages.startup.LOCK_FILE",
        tmp_path / "processing" / "pipeline.lock",
    )
    monkeypatch.setattr(
        "ingestion.stages.startup.ARCHIVE_DIR", tmp_path / "archive"
    )
    with (
        patch("ingestion.stages.startup.load_config"),
        patch("ingestion.stages.startup.setup_ssl"),
        patch("ingestion.stages.startup.LLMClient") as mock_llm_cls,
        patch("ingestion.stages.startup.get_connection") as mock_get_conn,
        patch("ingestion.stages.startup.verify_connection"),
    ):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        conn, llm = run_startup()

    assert conn is mock_conn
    assert llm is mock_llm
    logging.getLogger().handlers.clear()


def test_run_startup_skips_llm_when_not_required(tmp_path, monkeypatch):
    """Storage-only startup returns a DB connection without an LLM client."""
    monkeypatch.setattr("ingestion.utils.logging_setup.LOGS_DIR", tmp_path)
    monkeypatch.setattr(
        "ingestion.stages.startup.PROCESSING_DIR", tmp_path / "processing"
    )
    monkeypatch.setattr(
        "ingestion.stages.startup.LOCK_FILE",
        tmp_path / "processing" / "pipeline.lock",
    )
    monkeypatch.setattr(
        "ingestion.stages.startup.ARCHIVE_DIR", tmp_path / "archive"
    )
    with (
        patch("ingestion.stages.startup.load_config"),
        patch("ingestion.stages.startup.setup_ssl"),
        patch("ingestion.stages.startup.LLMClient") as mock_llm_cls,
        patch("ingestion.stages.startup.get_connection") as mock_get_conn,
        patch("ingestion.stages.startup.verify_connection"),
    ):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        conn, llm = run_startup(require_llm=False)

    assert conn is mock_conn
    assert llm is None
    mock_llm_cls.assert_not_called()
    logging.getLogger().handlers.clear()
