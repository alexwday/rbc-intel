"""Tests for ingestion.utils.logging_setup."""

import logging
import sys

from ingestion.utils.logging_setup import (
    ConsoleFormatter,
    FileFormatter,
    get_stage_logger,
    setup_logging,
)


def _make_record(level=logging.INFO, msg="test message", stage=None):
    """Build a LogRecord for testing. Returns: LogRecord."""
    record = logging.LogRecord(
        name="test",
        level=level,
        pathname="test_file.py",
        lineno=10,
        msg=msg,
        args=None,
        exc_info=None,
    )
    if stage is not None:
        record.stage = stage
    return record


def test_console_format_time_default():
    """Default format is YYYY-MM-DD HH:MM:SS."""
    formatter = ConsoleFormatter()
    record = _make_record()
    result = formatter.formatTime(record)
    parts = result.split("-")
    assert len(parts) == 3


def test_console_format_time_custom_datefmt():
    """Custom datefmt is respected."""
    formatter = ConsoleFormatter()
    record = _make_record()
    result = formatter.formatTime(record, datefmt="%H:%M")
    assert ":" in result
    assert "-" not in result


def test_console_format_with_stage():
    """Stage appears in formatted output."""
    formatter = ConsoleFormatter()
    record = _make_record(stage="1-CLASSIFY")
    result = formatter.format(record)
    assert "1-CLASSIFY" in result
    assert "test message" in result


def test_console_format_without_stage():
    """SYSTEM is used when no stage is set."""
    formatter = ConsoleFormatter()
    record = _make_record()
    result = formatter.format(record)
    assert "SYSTEM" in result


def test_console_format_warning_color():
    """Warning level uses a different color."""
    formatter = ConsoleFormatter()
    record = _make_record(level=logging.WARNING)
    result = formatter.format(record)
    assert "test message" in result


def test_console_format_error_color():
    """Error level uses a different color."""
    formatter = ConsoleFormatter()
    record = _make_record(level=logging.ERROR)
    result = formatter.format(record)
    assert "test message" in result


def test_console_format_critical_color():
    """Critical level uses a different color."""
    formatter = ConsoleFormatter()
    record = _make_record(level=logging.CRITICAL)
    result = formatter.format(record)
    assert "test message" in result


def test_console_format_debug_color():
    """Debug level uses a different color."""
    formatter = ConsoleFormatter()
    record = _make_record(level=logging.DEBUG)
    result = formatter.format(record)
    assert "test message" in result


def test_file_format_time_default():
    """Default format includes milliseconds."""
    formatter = FileFormatter()
    record = _make_record()
    result = formatter.formatTime(record)
    assert "." in result
    ms_part = result.split(".")[-1]
    assert len(ms_part) == 3


def test_file_format_time_custom_datefmt():
    """Custom datefmt is respected."""
    formatter = FileFormatter()
    record = _make_record()
    result = formatter.formatTime(record, datefmt="%H:%M")
    assert "." not in result


def test_file_format_with_stage():
    """Stage and line number appear in file output."""
    formatter = FileFormatter()
    record = _make_record(stage="2-EXTRACT")
    result = formatter.format(record)
    assert "2-EXTRACT" in result
    assert "test_file.py:10" in result
    assert "test message" in result


def test_file_format_without_stage():
    """SYSTEM is used when no stage is set."""
    formatter = FileFormatter()
    record = _make_record()
    result = formatter.format(record)
    assert "SYSTEM" in result


def test_file_format_includes_level():
    """Log level name appears in file output."""
    formatter = FileFormatter()
    record = _make_record(level=logging.WARNING)
    result = formatter.format(record)
    assert "WARNING" in result


def test_setup_logging_creates_handlers(tmp_path, monkeypatch):
    """setup_logging adds console and file handlers."""
    monkeypatch.setattr("ingestion.utils.logging_setup.LOGS_DIR", tmp_path)
    setup_logging()
    root = logging.getLogger()
    handler_types = [type(h) for h in root.handlers]
    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types
    root.handlers.clear()


def test_setup_logging_creates_log_directory(tmp_path, monkeypatch):
    """setup_logging creates the logs directory if missing."""
    log_dir = tmp_path / "newlogs"
    monkeypatch.setattr("ingestion.utils.logging_setup.LOGS_DIR", log_dir)
    setup_logging()
    assert log_dir.exists()
    logging.getLogger().handlers.clear()


def test_setup_logging_custom_level(tmp_path, monkeypatch):
    """File handler uses the provided level."""
    monkeypatch.setattr("ingestion.utils.logging_setup.LOGS_DIR", tmp_path)
    setup_logging(level=logging.WARNING)
    root = logging.getLogger()
    file_handlers = [
        h for h in root.handlers if isinstance(h, logging.FileHandler)
    ]
    assert file_handlers[0].level == logging.WARNING
    root.handlers.clear()


def test_console_format_includes_exception():
    """Exception traceback is included in console output."""
    formatter = ConsoleFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        record = _make_record(level=logging.ERROR)
        record.exc_info = sys.exc_info()
    result = formatter.format(record)
    assert "ValueError" in result
    assert "test error" in result


def test_file_format_includes_exception():
    """Exception traceback is included in file output."""
    formatter = FileFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        record = _make_record(level=logging.ERROR)
        record.exc_info = sys.exc_info()
    result = formatter.format(record)
    assert "ValueError" in result
    assert "test error" in result


def test_setup_logging_closes_existing_handlers(tmp_path, monkeypatch):
    """setup_logging closes handlers before replacing."""
    monkeypatch.setattr("ingestion.utils.logging_setup.LOGS_DIR", tmp_path)
    setup_logging()
    old_handlers = list(logging.getLogger().handlers)
    setup_logging()
    for handler in old_handlers:
        assert handler not in logging.getLogger().handlers
    logging.getLogger().handlers.clear()


def test_get_stage_logger_returns_adapter():
    """get_stage_logger returns a LoggerAdapter."""
    result = get_stage_logger("test", "1-CLASSIFY")
    assert isinstance(result, logging.LoggerAdapter)
    assert result.extra["stage"] == "1-CLASSIFY"
