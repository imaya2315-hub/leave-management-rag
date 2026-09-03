"""
NOTE: Actual JWT verification for protected routes lives in
app/api/deps.py as a FastAPI dependency (get_current_user) — that's
the idiomatic way to do per-route auth in FastAPI, since it lets each
route opt in individually and shows up correctly in /docs.

This file holds true ASGI middleware instead — logic that should run
on EVERY request regardless of route, such as request logging.
"""
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.utils.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)"
        )
        return response
