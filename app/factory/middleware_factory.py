"""Middleware Factory für die Keiko-Anwendung.

Diese Factory kapselt die Middleware-Setup-Logik um die Komplexität
der KeikoApplication zu reduzieren.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.logger_utils import get_module_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ..service_container import ServiceContainer

logger = get_module_logger(__name__)


class MiddlewareFactory:
    """Factory für Middleware-Setup."""

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container

    def setup_middleware(self, app: FastAPI) -> None:
        """Richtet alle Middleware-Komponenten ein.

        Args:
            app: FastAPI-Anwendungsinstanz
        """
        # Core Middleware
        self._setup_core_middleware(app)

        # Security Middleware
        self._setup_security_middleware(app)

        # Monitoring Middleware
        self._setup_monitoring_middleware(app)

        # Rate Limiting Middleware
        self._setup_rate_limiting_middleware(app)

        # mTLS Middleware (optional)
        self._setup_mtls_middleware(app)

        logger.info("Middleware-Setup abgeschlossen")

    def _setup_core_middleware(self, app: FastAPI) -> None:
        """Richtet Core-Middleware ein."""
        from api.middleware import setup_middleware

        middleware_config = self.container.middleware_config
        setup_middleware(app, middleware_config)
        logger.debug("Core-Middleware eingerichtet")

    def _setup_security_middleware(self, app: FastAPI) -> None:
        """Richtet Security-Middleware ein."""
        try:
            from api.middleware.deprecation_middleware import DeprecationHeadersMiddleware
            from api.middleware.idempotency_middleware import IdempotencyMiddleware
            from audit_system import AuditMiddleware
            from audit_system.audit_middleware import AuditConfig

            # Erstelle Standard-Audit-Konfiguration
            audit_config = AuditConfig()
            app.add_middleware(AuditMiddleware, config=audit_config)
            app.add_middleware(DeprecationHeadersMiddleware)
            app.add_middleware(IdempotencyMiddleware)

            logger.debug("Security-Middleware eingerichtet")
        except ImportError as exc:
            logger.warning(f"Security-Middleware konnte nicht geladen werden: {exc}")

    def _setup_monitoring_middleware(self, app: FastAPI) -> None:
        """Richtet Monitoring-Middleware ein."""
        try:
            from api.middleware.prometheus_middleware import PrometheusMiddleware
            from api.middleware.response_cache_middleware import ResponseCacheMiddleware

            app.add_middleware(PrometheusMiddleware)
            app.add_middleware(ResponseCacheMiddleware)

            logger.debug("Monitoring-Middleware eingerichtet")
        except ImportError as exc:
            logger.warning(f"Monitoring-Middleware konnte nicht geladen werden: {exc}")

    def _setup_rate_limiting_middleware(self, app: FastAPI) -> None:
        """Richtet Rate Limiting Middleware ein."""
        try:
            from middleware.kei_stream_rate_limiting import (
                KEIStreamRateLimitingMiddleware,
                get_kei_stream_rate_limiting_config,
            )

            get_kei_stream_rate_limiting_config()
            app.add_middleware(
                KEIStreamRateLimitingMiddleware,
                redis_url="redis://localhost:6379",
                default_config=None  # Verwende Standard-Konfiguration
            )

            logger.debug("Rate Limiting Middleware eingerichtet")
        except ImportError as exc:
            logger.warning(f"Rate Limiting Middleware konnte nicht geladen werden: {exc}")

    def _setup_mtls_middleware(self, app: FastAPI) -> None:
        """Richtet mTLS Middleware ein (optional)."""
        if not self.container.is_mtls_enabled:
            logger.debug("mTLS ist deaktiviert")
            return

        mtls_middleware_class = self.container.get_mtls_middleware_class()
        if mtls_middleware_class:
            app.add_middleware(mtls_middleware_class)
            logger.info("mTLS Middleware eingerichtet")
        else:
            logger.warning("mTLS ist aktiviert, aber Middleware-Klasse nicht verfügbar")


class MiddlewareConfigFactory:
    """Factory für Middleware-Konfiguration."""

    @staticmethod
    def create_cors_config(origins: list[str] | None = None) -> dict[str, Any]:
        """Erstellt CORS-Konfiguration.

        Args:
            origins: Liste der erlaubten Origins

        Returns:
            CORS-Konfiguration
        """
        from ..common.constants import DEFAULT_CORS_ORIGINS

        if origins is None:
            origins = DEFAULT_CORS_ORIGINS

        return {
            "allow_origins": origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

    @staticmethod
    def create_rate_limit_config(
        requests_per_minute: int = 60,
        burst_size: int = 10
    ) -> dict[str, Any]:
        """Erstellt Rate Limiting Konfiguration.

        Args:
            requests_per_minute: Anfragen pro Minute
            burst_size: Burst-Größe

        Returns:
            Rate Limiting Konfiguration
        """
        return {
            "requests_per_minute": requests_per_minute,
            "burst_size": burst_size,
            "fail_open": True,  # Fail-Open im Fehlerfall
        }

    @staticmethod
    def create_security_config(
        enable_audit: bool = True,
        enable_deprecation_headers: bool = True,
        enable_idempotency: bool = True
    ) -> dict[str, bool]:
        """Erstellt Security-Konfiguration.

        Args:
            enable_audit: Audit-Middleware aktivieren
            enable_deprecation_headers: Deprecation Headers aktivieren
            enable_idempotency: Idempotency-Middleware aktivieren

        Returns:
            Security-Konfiguration
        """
        return {
            "audit": enable_audit,
            "deprecation_headers": enable_deprecation_headers,
            "idempotency": enable_idempotency,
        }


class MiddlewareOrderManager:
    """Verwaltet die Reihenfolge der Middleware."""

    # Middleware-Reihenfolge (von außen nach innen)
    MIDDLEWARE_ORDER = [
        "cors",           # CORS muss ganz außen sein
        "security",       # Security-Middleware
        "rate_limiting",  # Rate Limiting
        "monitoring",     # Monitoring und Metriken
        "caching",        # Response Caching
        "audit",          # Audit-Logging
        "mtls",           # mTLS (falls aktiviert)
    ]

    @classmethod
    def get_middleware_order(cls) -> list[str]:
        """Gibt die empfohlene Middleware-Reihenfolge zurück."""
        return cls.MIDDLEWARE_ORDER.copy()

    @classmethod
    def validate_middleware_order(cls, middleware_list: list[str]) -> bool:
        """Validiert die Middleware-Reihenfolge.

        Args:
            middleware_list: Liste der Middleware-Namen

        Returns:
            True wenn die Reihenfolge korrekt ist
        """
        expected_order = cls.MIDDLEWARE_ORDER

        # Prüfe dass alle Middleware in der richtigen Reihenfolge sind
        last_index = -1
        for middleware in middleware_list:
            if middleware in expected_order:
                current_index = expected_order.index(middleware)
                if current_index < last_index:
                    return False
                last_index = current_index

        return True


__all__ = [
    "MiddlewareConfigFactory",
    "MiddlewareFactory",
    "MiddlewareOrderManager",
]
