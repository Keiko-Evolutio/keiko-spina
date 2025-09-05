"""Production-ready Rate Limiting Middleware für KEI-MCP API.

Implementiert Redis-basiertes Rate Limiting mit Fallback auf In-Memory,
verschiedene Tiers und umfassende Monitoring-Capabilities.
"""

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config.unified_rate_limiting import (
    RateLimitBackend,
    RateLimitConfig,
    RateLimitTier,
    get_rate_limit_config,
)
from core.container import get_container
from kei_logging import get_logger
from observability.kei_mcp_metrics import kei_mcp_metrics
from services.interfaces.rate_limiter import RateLimiterBackend, RateLimitResult
from services.webhooks.audit_logger import WebhookAuditEventType, webhook_audit

from .constants import PathConstants, RateLimitConstants
from .utils.client_identification import ClientIdentificationUtils

logger = get_logger(__name__)


class RateLimitManager:
    """Manager für Rate Limiting mit verschiedenen Backends."""

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or get_rate_limit_config()
        self._current_backend: RateLimiterBackend | None = None
        self._memory_backend: RateLimiterBackend | None = None
        self._cleanup_task: asyncio.Task | None = None

        # API-Key zu Tier Mapping (würde normalerweise aus Datenbank kommen)
        self._api_key_tiers: dict[str, RateLimitTier] = {}

        # IP-Whitelist für Bypass
        self._ip_whitelist: set = set()

    async def initialize(self):
        """Initialisiert Rate Limiting Backends."""
        try:
            # Backend aus dem DI-Container auflösen
            backend = get_container().resolve(RateLimiterBackend)
            if await backend.health_check():
                self._current_backend = backend
                logger.info(f"✅ Rate Limiting Backend aktiviert: {type(backend).__name__}")
            else:
                # Fallback: Fail-Open; Konfiguration entscheidet, ob Exception geworfen wird
                logger.warning("⚠️ Rate Limiting Backend nicht verfügbar")
                if self.config.backend == RateLimitBackend.REDIS:
                    raise Exception("Redis Backend erforderlich aber nicht verfügbar")

            # Initialisiere Memory Backend als Fallback
            try:
                from services.redis_rate_limiter import MemoryRateLimiter
                self._memory_backend = MemoryRateLimiter()
                logger.info("✅ Memory Backend als Fallback initialisiert")
            except ImportError:
                # Definiere MemoryRateLimiter als None im except Block
                MemoryRateLimiter = None
                logger.warning("⚠️ Memory Backend nicht verfügbar")
                self._memory_backend = None

            if not self._current_backend:
                raise Exception("Kein Rate Limiting Backend verfügbar")

            # Cleanup-Task starten
            if self.config.cleanup_interval_seconds > 0:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            # Lade API-Key-Tiers (Beispiel-Implementation)
            await self._load_api_key_tiers()

            logger.info(f"Rate Limiting Manager initialisiert: Backend={type(self._current_backend).__name__}")

        except Exception as e:
            logger.exception(f"Fehler bei Rate Limiting Initialisierung: {e}")
            raise

    async def shutdown(self):
        """Beendet Rate Limiting Manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        logger.info("Rate Limiting Manager beendet")

    async def check_rate_limit(
        self,
        request: Request,
        operation: str = "default"
    ) -> RateLimitResult:
        """Prüft Rate Limit für Request."""
        try:
            # Extrahiere Client-Identifikation
            client_id, tier = await self._get_client_info(request)

            # Prüfe IP-Whitelist
            client_ip = self._get_client_ip(request)
            if client_ip in self._ip_whitelist:
                logger.debug(f"IP {client_ip} in Whitelist - Rate Limit übersprungen")
                return self._create_bypass_result()

            # Prüfe Tier-spezifische Bypass-Einstellungen
            tier_config = self.config.get_tier_config(tier)
            if tier_config.bypass_rate_limits:
                logger.debug(f"Tier {tier} hat Rate Limit Bypass - übersprungen")
                return self._create_bypass_result()

            # Rate Limit Key erstellen
            rate_limit_key = f"{client_id}:{operation}"

            # Policy für Operation abrufen
            policy = tier_config.get_policy(operation)

            # Rate Limit prüfen
            # Wenn Backend noch nicht initialisiert ist, Fail-Open
            if self._current_backend is None:
                return self._create_bypass_result()

            result = await self._current_backend.check_rate_limit(
                rate_limit_key,
                policy,
                time.time()
            )

            # Metriken aktualisieren
            if self.config.enable_metrics:
                self._update_metrics(client_id, tier, operation, result)

            # Logging
            if self.config.enable_logging:
                self._log_rate_limit_check(client_id, operation, result)

            return result

        except Exception as e:
            logger.exception(f"Fehler bei Rate Limit Check: {e}")
            # Bei Fehlern erlauben (Fail-Open)
            return self._create_bypass_result()

    async def reset_rate_limit(self, client_id: str, operation: str | None = None) -> bool:
        """Setzt Rate Limit für Client zurück."""
        try:
            if operation:
                key = f"{client_id}:{operation}"
                return await self._current_backend.reset_rate_limit(key)
            # Alle Operationen für Client zurücksetzen
            success = True
            for op in ["default", "register", "invoke", "discovery", "stats"]:
                key = f"{client_id}:{op}"
                result = await self._current_backend.reset_rate_limit(key)
                success = success and result
            return success

        except Exception as e:
            logger.exception(f"Fehler beim Zurücksetzen von Rate Limit für {client_id}: {e}")
            return False

    async def get_rate_limit_info(self, client_id: str, operation: str = "default") -> dict[str, Any] | None:
        """Gibt Rate Limit Informationen für Client zurück."""
        try:
            key = f"{client_id}:{operation}"
            return await self._current_backend.get_rate_limit_info(key)
        except Exception as e:
            logger.exception(f"Fehler beim Abrufen von Rate Limit Info für {client_id}: {e}")
            return None

    async def add_api_key_tier(self, api_key: str, tier: RateLimitTier):
        """Fügt API-Key-Tier-Mapping hinzu."""
        self._api_key_tiers[api_key] = tier
        logger.info(f"API-Key-Tier hinzugefügt: {api_key[:8]}... -> {tier}")

    async def remove_api_key_tier(self, api_key: str):
        """Entfernt API-Key-Tier-Mapping."""
        if api_key in self._api_key_tiers:
            del self._api_key_tiers[api_key]
            logger.info(f"API-Key-Tier entfernt: {api_key[:8]}...")

    def add_ip_to_whitelist(self, ip: str):
        """Fügt IP zur Whitelist hinzu."""
        self._ip_whitelist.add(ip)
        logger.info(f"IP zur Whitelist hinzugefügt: {ip}")

    def remove_ip_from_whitelist(self, ip: str):
        """Entfernt IP von Whitelist."""
        self._ip_whitelist.discard(ip)
        logger.info(f"IP von Whitelist entfernt: {ip}")

    async def _get_client_info(self, request: Request) -> tuple[str, RateLimitTier]:
        """Extrahiert Client-ID und Tier aus Request."""
        client_id, tier_str = ClientIdentificationUtils.get_client_info_with_tier(
            request,
            self._api_key_tiers,
            self.config.default_tier.value
        )
        # Konvertiere String zu RateLimitTier wenn nötig
        if isinstance(tier_str, str):
            tier = RateLimitTier(tier_str)
        else:
            tier = tier_str
        return client_id, tier

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extrahiert Client-IP aus Request."""
        return ClientIdentificationUtils.extract_client_ip(request)

    @staticmethod
    def _create_bypass_result() -> RateLimitResult:
        """Erstellt Bypass-Result für Whitelist/Admin."""
        return RateLimitResult(
            allowed=True,
            remaining=RateLimitConstants.BYPASS_REMAINING,
            reset_time=int(time.time() + RateLimitConstants.DEFAULT_TTL_SECONDS),
            current_usage=0,
            limit=RateLimitConstants.BYPASS_LIMIT
        )

    @staticmethod
    def _update_metrics(
        _client_id: str,
        tier: RateLimitTier,
        operation: str,
        result: RateLimitResult
    ):
        """Aktualisiert Rate Limit Metriken."""
        try:
            # Rate Limit Hits - verwende Dictionary statt Prometheus Labels
            key = f"{tier.value}:{operation}:{'allowed' if result.allowed else 'rejected'}"
            if key not in kei_mcp_metrics.rate_limit_checks:
                kei_mcp_metrics.rate_limit_checks[key] = 0
            kei_mcp_metrics.rate_limit_checks[key] += 1

            # Soft Limit Warnings
            if result.soft_limit_exceeded:
                warning_key = f"{tier.value}:{operation}"
                if warning_key not in kei_mcp_metrics.rate_limit_soft_limit_warnings:
                    kei_mcp_metrics.rate_limit_soft_limit_warnings[warning_key] = 0
                kei_mcp_metrics.rate_limit_soft_limit_warnings[warning_key] += 1

            # Current Usage Gauge
            usage_key = f"{tier.value}:{operation}"
            kei_mcp_metrics.rate_limit_current_usage[usage_key] = result.current_usage

        except Exception as e:
            logger.warning(f"Fehler beim Aktualisieren der Rate Limit Metriken: {e}")

    @staticmethod
    def _log_rate_limit_check(
        client_id: str,
        operation: str,
        result: RateLimitResult
    ):
        """Loggt Rate Limit Check."""
        if not result.allowed:
            logger.warning(
                f"Rate Limit überschritten: {client_id} für {operation} "
                f"({result.current_usage}/{result.limit}), "
                f"Retry-After: {result.retry_after}s"
            )
        elif result.soft_limit_exceeded:
            logger.info(
                f"Rate Limit Soft-Limit erreicht: {client_id} für {operation} "
                f"({result.current_usage}/{result.limit})"
            )
        else:
            logger.debug(
                f"Rate Limit OK: {client_id} für {operation} "
                f"({result.current_usage}/{result.limit}), "
                f"Remaining: {result.remaining}"
            )

    async def _load_api_key_tiers(self):
        """Lädt API-Key-Tier-Mappings (Beispiel-Implementation)."""
        # In einer echten Implementation würde dies aus einer Datenbank kommen
        example_mappings = {
            "admin_key_123": RateLimitTier.ADMIN,
            "enterprise_key_456": RateLimitTier.ENTERPRISE,
            "premium_key_789": RateLimitTier.PREMIUM,
            "basic_key_000": RateLimitTier.BASIC
        }

        self._api_key_tiers.update(example_mappings)
        logger.info(f"API-Key-Tiers geladen: {len(self._api_key_tiers)} Mappings")

    async def _cleanup_loop(self):
        """Periodische Bereinigung abgelaufener Rate Limit Einträge."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)

                if self._current_backend:
                    deleted = await self._current_backend.cleanup_expired()
                    if deleted > 0:
                        logger.info(f"Rate Limit Cleanup: {deleted} Einträge bereinigt")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Fehler im Rate Limit Cleanup: {e}")
                await asyncio.sleep(60)  # Kurze Pause bei Fehlern

    async def health_check(self) -> dict[str, Any]:
        """Gibt Gesundheitsstatus des Rate Limiting zurück."""
        health_status = {
            "backend_type": type(self._current_backend).__name__ if self._current_backend else "None",
            "config_backend": self.config.backend.value,
            "redis_healthy": False,
            "memory_available": self._memory_backend is not None,
            "cleanup_running": self._cleanup_task is not None and not self._cleanup_task.done()
        }

        # Redis-Gesundheit prüfen
        if self._current_backend:
            health_status["redis_healthy"] = await self._current_backend.health_check()

        return health_status


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware für Rate Limiting."""

    def __init__(self, app, rate_limiter: RateLimitManager):
        super().__init__(app)
        self.rate_limit_manager = rate_limiter

        # Ausgeschlossene Pfade (keine Rate Limits)
        self.excluded_paths = set(PathConstants.EXCLUDED_PATHS)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Middleware-Dispatch mit Rate Limiting."""
        # Prüfe ausgeschlossene Pfade
        if self._is_excluded_path(request.url.path):
            return await call_next(request)

        # Bestimme Operation basierend auf Pfad
        operation = self._get_operation_from_path(request.url.path)

        try:
            # Rate Limit prüfen
            result = await self.rate_limit_manager.check_rate_limit(request, operation)

            if not result.allowed:
                # Rate Limit überschritten
                headers = {}

                if self.rate_limit_manager.config.include_rate_limit_headers:
                    headers.update({
                        "X-RateLimit-Limit": str(result.limit),
                        "X-RateLimit-Remaining": str(result.remaining),
                        "X-RateLimit-Reset": str(result.reset_time),
                        "X-RateLimit-Reset-After": str(result.reset_time - int(time.time()))
                    })

                    if result.retry_after:
                        headers["Retry-After"] = str(result.retry_after)

                # Audit: Security Event (Rate Limit exceeded)
                with contextlib.suppress(Exception):
                    await webhook_audit.security_event(
                        event=WebhookAuditEventType.SECURITY_RATE_LIMIT_EXCEEDED,
                        correlation_id=getattr(request.state, "correlation_id", None),
                        tenant_id=request.headers.get("x-tenant") if hasattr(request, "headers") else None,
                        user_id=getattr(request.state, "user_id", None),
                        error_details={
                            "operation": operation,
                            "current_usage": result.current_usage,
                            "limit": result.limit,
                            "retry_after": result.retry_after,
                        },
                    )

                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate Limit Exceeded",
                        "message": f"Rate Limit überschritten für Operation '{operation}'",
                        "type": "rate_limit_error",
                        "limit": result.limit,
                        "remaining": result.remaining,
                        "reset_time": result.reset_time,
                        "retry_after": result.retry_after
                    },
                    headers=headers
                )

            # Request verarbeiten
            response = await call_next(request)

            # Rate Limit Headers hinzufügen
            if self.rate_limit_manager.config.include_rate_limit_headers:
                response.headers.update({
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": str(result.remaining),
                    "X-RateLimit-Reset": str(result.reset_time)
                })

                # Soft-Limit-Warnung
                if result.soft_limit_exceeded:
                    response.headers["X-RateLimit-Warning"] = "Soft limit exceeded"

            return response

        except Exception as e:
            logger.exception(f"Fehler in Rate Limit Middleware: {e}")
            # Bei Fehlern Request durchlassen (Fail-Open)
            return await call_next(request)

    def _is_excluded_path(self, path: str) -> bool:
        """Prüft ob Pfad von Rate Limiting ausgeschlossen ist."""
        return any(excluded in path for excluded in self.excluded_paths)

    @staticmethod
    def _get_operation_from_path(path: str) -> str:
        """Bestimmt Operation basierend auf Request-Pfad."""
        if "/servers/register" in path:
            return "register"
        if "/tools/invoke" in path:
            return "invoke"
        if "/tools" in path:
            return "discovery"
        if "/stats" in path or "/metrics" in path:
            return "stats"
        return "default"


# Globale Rate Limit Manager Instanz
rate_limit_manager = RateLimitManager()


async def initialize_rate_limiting():
    """Initialisiert globales Rate Limiting."""
    await rate_limit_manager.initialize()


async def shutdown_rate_limiting():
    """Beendet globales Rate Limiting."""
    await rate_limit_manager.shutdown()


def get_rate_limit_manager() -> RateLimitManager:
    """Gibt globalen Rate Limit Manager zurück."""
    return rate_limit_manager
