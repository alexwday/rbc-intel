"""Pipeline logging with colored console and file output."""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"

STAGE_WIDTH = 20
SOURCE_WIDTH = 28

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
BRIGHT_RED = "\033[91m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_CYAN = "\033[96m"


def _short_source_path(pathname: str) -> str:
    """Return a compact source path for log output."""
    path = Path(pathname)
    parts = path.parts
    if "ingestion" in parts:
        index = parts.index("ingestion")
        relative_parts = parts[index + 1 :]
        if len(relative_parts) >= 2:
            return "/".join(relative_parts[-2:])
        if relative_parts:
            return relative_parts[-1]
    return path.name


class ConsoleFormatter(logging.Formatter):
    """Colored, column-aligned formatter for console output.

    Columns: timestamp | stage | source file | message.
    Colors vary by log level. Exception tracebacks are
    appended when present.

    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(ConsoleFormatter())
    """

    LEVEL_COLORS = {
        logging.DEBUG: DIM,
        logging.INFO: WHITE,
        logging.WARNING: BRIGHT_YELLOW,
        logging.ERROR: BRIGHT_RED,
        logging.CRITICAL: BOLD + BRIGHT_RED,
    }

    def formatTime(self, record, datefmt=None):
        """Timestamp as YYYY-MM-DD HH:MM:SS."""
        created = datetime.fromtimestamp(record.created)
        if datefmt:
            return created.strftime(datefmt)
        return created.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record):
        """Build colored, column-aligned log line."""
        timestamp = self.formatTime(record)
        stage = getattr(record, "stage", "SYSTEM")
        source = _short_source_path(record.pathname)
        color = self.LEVEL_COLORS.get(record.levelno, WHITE)
        line = (
            f"{DIM}{timestamp}{RESET} \u2502 "
            f"{BOLD}{BRIGHT_CYAN}"
            f"{stage:<{STAGE_WIDTH}}{RESET} \u2502 "
            f"{YELLOW}"
            f"{source:<{SOURCE_WIDTH}}{RESET} \u2502 "
            f"{color}{record.getMessage()}{RESET}"
        )
        if record.exc_info and record.exc_info[1]:
            line += "\n" + self.formatException(record.exc_info)
        return line


class FileFormatter(logging.Formatter):
    """Verbose plain-text formatter for log files.

    Includes milliseconds, log level, source line numbers,
    and full exception tracebacks when present.

    Example:
        >>> handler = logging.FileHandler("pipeline.log")
        >>> handler.setFormatter(FileFormatter())
    """

    def formatTime(self, record, datefmt=None):
        """Timestamp with milliseconds."""
        created = datetime.fromtimestamp(record.created)
        if datefmt:
            return created.strftime(datefmt)
        base = created.strftime("%Y-%m-%d %H:%M:%S")
        return f"{base}.{int(record.msecs):03d}"

    def format(self, record):
        """Build plain-text log line with full detail."""
        timestamp = self.formatTime(record)
        level = record.levelname
        stage = getattr(record, "stage", "SYSTEM")
        source = f"{_short_source_path(record.pathname)}:{record.lineno}"
        line = (
            f"{timestamp} | {level:<8} "
            f"| {stage:<{STAGE_WIDTH}} "
            f"| {source:<25} "
            f"| {record.getMessage()}"
        )
        if record.exc_info and record.exc_info[1]:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(level: int = logging.DEBUG) -> None:
    """Configure root logger with console and file handlers.

    Console shows INFO+ with color and column alignment.
    File captures all levels with full detail. Closes any
    existing handlers before replacing them.

    Params:
        level: Minimum level for file output (default DEBUG).
            Console always logs INFO and above.

    Returns:
        None

    Example:
        >>> setup_logging()
        >>> setup_logging(logging.INFO)
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"pipeline_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(FileFormatter())
    root.addHandler(file_handler)

    noisy_loggers = [
        "openai",
        "httpx",
        "httpcore",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_stage_logger(name: str, stage: str) -> logging.LoggerAdapter:
    """Get a logger with pipeline stage context attached.

    Params:
        name: Logger name (typically __name__)
        stage: Stage label (e.g. "1-CLASSIFY")

    Returns:
        logging.LoggerAdapter with stage in extra dict

    Example:
        >>> log = get_stage_logger(__name__, "1-CLASSIFY")
        >>> log.info("Processing 12 files")
    """
    logger = logging.getLogger(name)
    return logging.LoggerAdapter(logger, {"stage": stage})
