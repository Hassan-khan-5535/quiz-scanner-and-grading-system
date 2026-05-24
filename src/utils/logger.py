"""
logger.py — Centralized Logging Setup
=======================================

PURPOSE:
    Provides a consistent way to log messages across ALL modules.
    Instead of print() scattered everywhere, we use structured logging.

WHY USE LOGGING INSTEAD OF PRINT?
    1. print() can't be turned off easily — logging can (via LOG_LEVEL)
    2. print() doesn't show which module produced the message
    3. print() mixes with program output — logging is separate
    4. Logging adds timestamps automatically

HOW TO USE IN OTHER MODULES:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.debug("Detailed info for debugging")    # Only shows if LOG_LEVEL = "DEBUG"
    logger.info("General progress updates")        # Normal operation info
    logger.warning("Something unexpected but OK")   # Potential issues
    logger.error("Something went wrong!")           # Actual errors
"""

import logging
import sys

# Import our config to get the log level setting
# We use a try/except because config.py is in the parent directory
try:
    from config import LOG_LEVEL
except ImportError:
    # Fallback if config can't be imported (e.g., running tests)
    LOG_LEVEL = "INFO"


def get_logger(module_name: str) -> logging.Logger:
    """
    Create and return a logger for the given module.

    Args:
        module_name: Usually pass __name__ so the logger is named
                     after the file that's using it.
                     Example: "src.qr_decoder.qr_reader"

    Returns:
        A configured Logger instance ready to use.

    Example:
        logger = get_logger(__name__)
        logger.info("QR code decoded successfully")
        # Output: [INFO] src.qr_decoder.qr_reader — QR code decoded successfully
    """
    # Create a logger with the module's name
    logger = logging.getLogger(module_name)

    # Only add handlers if none exist (prevents duplicate messages)
    # WHY: If get_logger() is called twice for the same module,
    #      we don't want to add another handler — that would print
    #      each message twice.
    if not logger.handlers:
        # Set the minimum severity level from our config
        # "DEBUG" shows everything, "ERROR" shows only errors
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

        # Create a handler that prints to the console (terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

        # Define the format of log messages
        # Example output: [INFO] src.qr_decoder.qr_reader — QR code found
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        # Attach the handler to the logger
        logger.addHandler(console_handler)

    return logger
