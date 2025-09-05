"""Versionierungs‑Middleware.

Erkennt API‑Version aus URL‑Prefix, Accept‑Header und Custom‑Header. Setzt
Response‑Header zur Version und unterstützt Fallback von v2→v1 abhängig von
Konfiguration. Nutzt Prometheus‑Counter zur Messung der Versionnutzung.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from prometheus_client import Counter
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings
from kei_logging import get_logger, structured_msg

from .constants import (
    API_VERSION_V1,
    API_VERSION_V2,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_KEIKO_V2,
    HEADER_ACCEPT,
    HEADER_API_VERSION,
    HEADER_API_VERSION_LOWER,
    HEADER_WARNING,
    PROMETHEUS_COUNTER_DESCRIPTION,
    PROMETHEUS_COUNTER_NAME,
    URL_PREFIX_V1,
    URL_PREFIX_V2,
    WARNING_V1_DEPRECATED,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request
    from starlette.responses import Response

logger = get_logger(__name__)

API_VERSION_REQUESTS = Counter(
    PROMETHEUS_COUNTER_NAME,
    PROMETHEUS_COUNTER_DESCRIPTION,
    ["version"],
)


class VersioningMiddleware(BaseHTTPMiddleware):
    """Middleware zur API‑Versionserkennung und Negotiation."""

    def __init__(self, app) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.api_versioning_enabled:
            return await call_next(request)

        version = self._detect_version(request)
        request.state.api_version = version

        with contextlib.suppress(Exception):
            API_VERSION_REQUESTS.labels(version=version).inc()

        try:
            response = await call_next(request)
        except Exception as exc:
            if version == API_VERSION_V2 and settings.api_version_fallback_enabled:
                logger.warning(structured_msg("v2 fallback invoked", error=str(exc)))
                scope = dict(request.scope)
                scope["path"] = request.url.path.replace(URL_PREFIX_V2, URL_PREFIX_V1, 1)
                response = await self.app(scope, request.receive, self._send_dummy)  # type: ignore[arg-type]
            else:
                raise

        try:
            response.headers[HEADER_API_VERSION] = version
            if settings.api_deprecation_warnings and version == API_VERSION_V1:
                response.headers.setdefault(HEADER_WARNING, WARNING_V1_DEPRECATED)
        except Exception:
            pass

        return response

    def _detect_version(self, request: Request) -> str:
        """Erkennt API-Version aus Request-Informationen.

        Args:
            request: HTTP-Request

        Returns:
            Erkannte API-Version (v1 oder v2)
        """
        path = request.url.path
        accept = request.headers.get(HEADER_ACCEPT, CONTENT_TYPE_JSON).lower()
        x_ver = (
            request.headers.get(HEADER_API_VERSION) or
            request.headers.get(HEADER_API_VERSION_LOWER) or
            ""
        ).lower().strip()

        if CONTENT_TYPE_KEIKO_V2 in accept:
            return API_VERSION_V2
        if x_ver in {API_VERSION_V1, API_VERSION_V2}:
            return x_ver
        if path.startswith(URL_PREFIX_V2):
            return API_VERSION_V2
        if path.startswith(URL_PREFIX_V1):
            return API_VERSION_V1
        return settings.api_default_version

    async def _send_dummy(self, _message):  # pragma: no cover
        return None


__all__ = ["VersioningMiddleware"]
