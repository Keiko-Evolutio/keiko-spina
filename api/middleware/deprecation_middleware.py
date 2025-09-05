"""Middleware für Deprecation-/Sunset-Header auf Basis konfigurierter Regeln."""

from __future__ import annotations

from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from config.api_versioning_config import DEPRECATION_RULES
from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request
    from starlette.responses import Response

logger = get_logger(__name__)


class DeprecationHeadersMiddleware(BaseHTTPMiddleware):
    """Ergänzt Deprecation-/Sunset-Header für passende Pfade."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:  # type: ignore[override]
        response = await call_next(request)

        try:
            path = request.url.path
            for rule in DEPRECATION_RULES:
                if path.startswith(rule.path_prefix):
                    if rule.deprecation:
                        response.headers["Deprecation"] = rule.deprecation
                    if rule.sunset:
                        response.headers["Sunset"] = rule.sunset
                    if rule.link:
                        response.headers.setdefault("Link", f"<{rule.link}>; rel=deprecation")
                    break
        except Exception as e:  # pragma: no cover - weich fehlertolerant
            logger.debug(f"Deprecation Header Handling Fehler: {e}")

        return response


__all__ = ["DeprecationHeadersMiddleware"]
