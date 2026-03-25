"""
Authentication — simplified auth for local development.

In production, this module would validate OAuth tokens against an
external auth service.  For local development, all requests are allowed.
"""

import logging

from fastapi import Request

logger = logging.getLogger(__name__)


async def validate_request(request: Request) -> None:
    """Validate an incoming request.

    For local development this is a no-op.  In production, add token
    validation and access control here.

    Args:
        request: The incoming FastAPI request.
    """
    logger.debug("Auth validation (local dev): pass-through for %s", request.url.path)
