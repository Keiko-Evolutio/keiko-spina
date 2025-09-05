"""DDoS Protection Middleware.

Implementiert leichtgewichtige IP-basierte Anomalie-Erkennung, Whitelist/Blacklist
mit Redis-Persistierung und Metrik-/Alert-Integration. Diese Middleware ergänzt
Reverse-Proxy-Rate-Limiting (Nginx/HAProxy) und arbeitet eng mit Redis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings
from kei_logging import get_logger
from monitoring import record_custom_metric
from services.webhooks.alerting import emit_warning

from .utils import MiddlewareErrorBuilder, RedisClientHelper

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request
    from starlette.responses import Response


@dataclass
class RateLimitResult:
    """Ergebnis der Rate-Limit-Prüfung."""

    minute_count: int
    burst_count: int
    should_block: bool


logger = get_logger(__name__)

# Redis-Schlüssel für DDoS-Protection
WHITELIST_KEY = "kei:ddos:whitelist"
BLACKLIST_KEY = "kei:ddos:blacklist"

# Ausgeschlossene Pfade (nicht blockieren)
EXCLUDED_PATHS = ["/metrics", "/health", "/ws", "/stream"]


def _extract_client_ip(request: Request) -> str:
    """Ermittelt die Client-IP unter Berücksichtigung von Proxy-Headern.

    Args:
        request: FastAPI Request-Objekt

    Returns:
        Client-IP-Adresse als String
    """
    if settings.ddos_trust_proxy_headers:
        # X-Forwarded-For Header prüfen
        xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip()

        # X-Real-IP Header prüfen
        xri = request.headers.get("x-real-ip") or request.headers.get("X-Real-IP")
        if xri:
            return xri.strip()

    # Fallback auf direkte Client-IP
    return request.client.host if request.client else "0.0.0.0"


def _is_excluded_path(path: str) -> bool:
    """Prüft, ob ein Pfad von DDoS-Protection ausgeschlossen ist.

    Args:
        path: Request-Pfad

    Returns:
        True wenn Pfad ausgeschlossen ist
    """
    return any(path.startswith(excluded) for excluded in EXCLUDED_PATHS)


class DDoSMiddleware(BaseHTTPMiddleware):
    """Einfaches IP-Blocking und Anomalie-Erkennung auf App-Ebene.

    Implementiert Redis-basiertes Rate-Limiting mit Whitelist/Blacklist-Funktionalität
    und automatische Anomalie-Erkennung für verdächtige Traffic-Patterns.
    """

    def __init__(self, app) -> None:
        """Initialisiert DDoS-Protection-Middleware.

        Args:
            app: FastAPI-Anwendung
        """
        super().__init__(app)
        self.enabled = settings.ddos_enabled
        self.req_per_min_threshold = settings.ddos_anomaly_requests_per_minute
        self.burst_threshold = settings.ddos_burst_threshold
        self.block_duration = settings.ddos_ip_block_duration_seconds

        # Utility-Instanzen
        self.redis_helper = RedisClientHelper("ddos_middleware")
        self.error_builder = MiddlewareErrorBuilder("ddos_middleware")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Hauptmethode für DDoS-Protection-Verarbeitung.

        Args:
            request: Eingehender HTTP-Request
            call_next: Nächste Middleware/Handler in der Kette

        Returns:
            HTTP-Response
        """
        if not self.enabled:
            return await call_next(request)

        # Client-IP extrahieren
        client_ip = _extract_client_ip(request)

        # Ausgeschlossene Pfade überspringen
        if _is_excluded_path(request.url.path or ""):
            return await call_next(request)

        # Whitelist-Prüfung
        if await self._is_whitelisted(client_ip):
            return await call_next(request)

        # Blacklist-Prüfung
        if await self._is_blacklisted(client_ip):
            return self.error_builder.forbidden(
                message="IP address is blocked",
                request=request,
                ip=client_ip,
                reason="blacklisted"
            )

        # Rate-Limiting und Anomalie-Erkennung
        rate_limit_result = await self._check_rate_limits(client_ip)
        if rate_limit_result.should_block:
            await self._handle_anomaly_detection(client_ip, rate_limit_result)
            return self.error_builder.rate_limited(
                message="Rate limit exceeded",
                request=request,
                retry_after=60,
                ip=client_ip,
                requests_per_minute=rate_limit_result.minute_count,
                burst_count=rate_limit_result.burst_count
            )

        return await call_next(request)

    async def _is_whitelisted(self, ip: str) -> bool:
        """Prüft, ob IP-Adresse auf der Whitelist steht.

        Args:
            ip: Client-IP-Adresse

        Returns:
            True wenn IP auf Whitelist steht
        """
        result = await self.redis_helper.safe_sismember(WHITELIST_KEY, ip)
        return result.success and result.value

    async def _is_blacklisted(self, ip: str) -> bool:
        """Prüft, ob IP-Adresse auf der Blacklist steht.

        Args:
            ip: Client-IP-Adresse

        Returns:
            True wenn IP auf Blacklist steht
        """
        result = await self.redis_helper.safe_sismember(BLACKLIST_KEY, ip)
        return result.success and result.value

    async def _check_rate_limits(self, ip: str) -> RateLimitResult:
        """Prüft Rate-Limits für eine IP-Adresse.

        Args:
            ip: Client-IP-Adresse

        Returns:
            RateLimitResult mit Zählern und Block-Entscheidung
        """
        minute_key = f"kei:ddos:cnt:{ip}"
        burst_key = f"kei:ddos:burst:{ip}"

        # Minute Counter (TTL: 70 Sekunden)
        minute_result = await self.redis_helper.safe_incr(minute_key)
        minute_count = minute_result.value if minute_result.success else 0

        if minute_result.success and minute_count == 1:
            await self.redis_helper.safe_expire(minute_key, 70)

        # Burst Counter (TTL: 5 Sekunden)
        burst_result = await self.redis_helper.safe_incr(burst_key)
        burst_count = burst_result.value if burst_result.success else 0

        if burst_result.success and burst_count == 1:
            await self.redis_helper.safe_expire(burst_key, 5)

        # Anomalie-Erkennung
        should_block = (
            minute_count > self.req_per_min_threshold or
            burst_count > self.burst_threshold
        )

        return RateLimitResult(
            minute_count=minute_count,
            burst_count=burst_count,
            should_block=should_block
        )

    async def _handle_anomaly_detection(self, ip: str, rate_limit_result: RateLimitResult) -> None:
        """Behandelt erkannte Anomalien (Blacklisting, Metriken, Alerts).

        Args:
            ip: Client-IP-Adresse
            rate_limit_result: Ergebnis der Rate-Limit-Prüfung
        """
        # IP zur Blacklist hinzufügen
        await self.redis_helper.safe_sadd(BLACKLIST_KEY, ip)
        await self.redis_helper.safe_expire(BLACKLIST_KEY, self.block_duration)

        # Metriken erfassen
        try:
            record_custom_metric("ddos.blocked", 1, {"ip": ip})
        except Exception as e:
            logger.debug(f"Metrik-Erfassung fehlgeschlagen: {e}")

        # Alert senden
        try:
            await emit_warning(
                "Verdächtiges Traffic-Muster erkannt",
                {
                    "ip": ip,
                    "requests_per_minute": rate_limit_result.minute_count,
                    "burst_count": rate_limit_result.burst_count,
                    "threshold_minute": self.req_per_min_threshold,
                    "threshold_burst": self.burst_threshold
                }
            )
        except Exception as e:
            logger.debug(f"Alert-Versendung fehlgeschlagen: {e}")


__all__ = ["BLACKLIST_KEY", "WHITELIST_KEY", "DDoSMiddleware", "RateLimitResult"]
