import logging
import os
from datetime import datetime

# -------------------------------------------------------------------
# UMA-3 Logging Setup
# -------------------------------------------------------------------
# This module configures a dedicated logger for the "uma3" namespace.
# It does not rely on logging.basicConfig(), so it plays nicely with
# pytest or frameworks that configure root logging first.
#
# It writes logs to:
#   - a daily-rotated file: uma3_YYYY-MM-DD.log (by default)
#   - stderr console stream
#
# You can override the log path via UMA3_LOG_PATH if desired.
# -------------------------------------------------------------------

DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_LOG_FILENAME = f"uma3_{DATE}.log"
LOG_PATH = os.environ.get("UMA3_LOG_PATH", DEFAULT_LOG_FILENAME)


def _configure_uma3_logger() -> logging.Logger:
    """
    Configure and return the root UMA-3 logger.

    This attaches:
      - FileHandler(LOG_PATH)
      - StreamHandler()
    with a consistent format and DEBUG level for UMA-3 code,
    while not interfering with root logging configuration.
    """
    logger = logging.getLogger("uma3")
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times if this module is imported more than once.
    if logger.handlers:
        return logger

    # Common formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # File handler (daily log)
    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Reduce noise from common noisy libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("UMA-3 logging initialized. Log file: %s", LOG_PATH)
    return logger


# This is the logger exported for use via `from uma3 import logger`
logger = _configure_uma3_logger()