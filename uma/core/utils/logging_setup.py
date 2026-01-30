import logging
import os
from datetime import datetime

_ORIGINAL_RECORD_FACTORY = logging.getLogRecordFactory()


def _uma_record_factory(*args, **kwargs):
    record = _ORIGINAL_RECORD_FACTORY(*args, **kwargs)
    if not hasattr(record, "request_id"):
        record.request_id = "-"
    if not hasattr(record, "trace_id"):
        record.trace_id = "-"
    return record


logging.setLogRecordFactory(_uma_record_factory)

# -------------------------------------------------------------------
# UMA Logging Setup
# -------------------------------------------------------------------
# This module configures a dedicated logger for the "uma" namespace.
# It does not rely on logging.basicConfig(), so it plays nicely with
# pytest or frameworks that configure root logging first.
#
# It writes logs to:
#   - a daily-rotated file: uma_YYYY-MM-DD.log (by default)
#   - stderr console stream
#
# You can override the log path via UMA_LOG_PATH if desired.
# -------------------------------------------------------------------

DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_LOG_FILENAME = f"uma_{DATE}.log"
LOG_PATH = os.environ.get("UMA_LOG_PATH", DEFAULT_LOG_FILENAME)
LOG_TO_FILE = os.environ.get("UMA_LOG_TO_FILE", "1").lower() not in ("0", "false", "no")


def _configure_uma_logger() -> logging.Logger:
    """
    Configure and return the root UMA logger.

    This attaches:
      - FileHandler(LOG_PATH)
      - StreamHandler()
    with a consistent format and DEBUG level for UMA code,
    while not interfering with root logging configuration.
    """
    logger = logging.getLogger("uma")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if this module is imported more than once.
    if logger.handlers:
        return logger

    # Common formatter
    class _ContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                from ...adapters.observability.context import get_request_id, get_trace_id
                record.request_id = get_request_id()
                record.trace_id = get_trace_id()
            except Exception:
                record.request_id = "-"
                record.trace_id = "-"
            return True

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [req=%(request_id)s trace=%(trace_id)s] %(name)s: %(message)s"
    )

    # File handler (daily log)
    file_handler = None
    if LOG_TO_FILE and LOG_PATH not in ("stdout", "stderr", "-"):
        try:
            log_dir = os.path.dirname(LOG_PATH)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
        except Exception:
            file_handler = None

    # Console handler
    stream = None
    if LOG_PATH == "stdout":
        stream = __import__("sys").stdout
    elif LOG_PATH == "stderr":
        stream = __import__("sys").stderr
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addFilter(_ContextFilter())
    if file_handler is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Reduce noise from common noisy libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("UMA logging initialized. Log file: %s", LOG_PATH)
    return logger


# This is the logger exported for use via `from uma import logger`
logger = _configure_uma_logger()
