"""SSL certificate setup for RBC environment."""

import importlib
import logging

logger = logging.getLogger(__name__)


def setup_ssl() -> None:
    """Enable RBC SSL certificates if available.

    Attempts to load rbc_security for corporate
    environment. Falls back silently for local development.

    Returns:
        None

    Example:
        >>> setup_ssl()
    """
    try:
        module = importlib.import_module("rbc_security")
        module.enable_certs()
        logger.info("RBC SSL certificates enabled")
    except ImportError:
        msg = "rbc_security not available, using system certificates"
        logger.info(msg)
