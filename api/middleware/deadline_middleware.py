"""Deadline/Time-Budget Middleware für REST.

Setzt pro eingehendem Request eine Deadline anhand des Headers
`X-Deadline-Ms` (absolute Millisekunden seit jetzt) oder `X-Time-Budget-Ms`
(Restbudget ab jetzt in Millisekunden). Exportiert verbleibendes Budget als
Response-Header `X-Time-Budget-Remaining`.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from kei_logging import get_logger
from observability.deadline import clear_deadline, get_remaining_budget_ms, set_deadline_ms_from_now

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.types import ASGIApp

logger = get_logger(__name__)


class DeadlineMiddleware(BaseHTTPMiddleware):
    """Middleware zur Verwaltung und Propagation von Deadlines."""

    def __init__(self, app: ASGIApp, default_budget_ms: int | None = None) -> None:
        super().__init__(app)
        self.default_budget_ms = default_budget_ms

    async def dispatch(self, request: Request, call_next):
        response = None  # Initialize response to avoid unbound variable in finally block
        try:
            # Budget aus Header priorisieren
            budget_ms: int | None = None
            # Absolutes Budget (ab jetzt, in ms)
            if request.headers.get("X-Time-Budget-Ms"):
                try:
                    budget_ms = int(request.headers["X-Time-Budget-Ms"])  # type: ignore[index]
                except Exception:
                    budget_ms = None
            # Falls explizit gesetzt, anwenden; sonst Default
            if budget_ms is None and self.default_budget_ms is not None:
                budget_ms = int(self.default_budget_ms)
            if budget_ms is not None and budget_ms >= 0:
                set_deadline_ms_from_now(budget_ms)
            response = await call_next(request)
        finally:
            # Budget Remaining setzen (best effort)
            rem = get_remaining_budget_ms()
            if rem is not None and response is not None:
                with contextlib.suppress(Exception):
                    response.headers["X-Time-Budget-Remaining"] = str(rem)
            clear_deadline()
        return response


__all__ = ["DeadlineMiddleware"]
