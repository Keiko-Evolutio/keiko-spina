"""Service Container für die Keiko-Anwendung.

Der Container verwaltet zentrale Abhängigkeiten, Konfigurationen und Factories
zur Entkopplung der App-Komponenten. Alle Kommentare und Docstrings sind in
Deutsch, während Identifier auf Englisch bleiben.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from typing import Any

from api.middleware import MiddlewareConfig
from config.settings import Settings, settings
from core.container import bootstrap_defaults, get_container
from kei_logging import get_logger

# Service-Imports
try:
    from services.interfaces.domain_revalidation_service import DomainRevalidationServiceInterface
    from services.unified_domain_revalidation_service import (
        UnifiedDomainRevalidationService as DomainRevalidationService,
    )
    DOMAIN_REVALIDATION_AVAILABLE = True
except ImportError:
    DomainRevalidationServiceInterface = None
    DomainRevalidationService = None
    DOMAIN_REVALIDATION_AVAILABLE = False

try:
    from config.unified_rate_limiting import RateLimitBackend as RLBackendKind
    from config.unified_rate_limiting import get_unified_rate_limit_config as get_rate_limit_config
    from services.interfaces.rate_limiter import RateLimiterBackend
    from services.redis_rate_limiter import MemoryRateLimiter, RedisRateLimiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    get_rate_limit_config = None
    RLBackendKind = None
    RateLimiterBackend = None
    RedisRateLimiter = None
    MemoryRateLimiter = None
    RATE_LIMITER_AVAILABLE = False

try:
    from services.core.manager import ServiceManager as ConcreteServiceManager
    from services.interfaces.service_manager import ServiceManagerInterface
    SERVICE_MANAGER_AVAILABLE = True
except ImportError:
    ServiceManagerInterface = None
    ConcreteServiceManager = None
    SERVICE_MANAGER_AVAILABLE = False

try:
    from kei_webhook import get_webhook_manager as legacy_get_webhook_manager

    from services.interfaces.webhook_manager import WebhookManagerInterface
    WEBHOOK_MANAGER_AVAILABLE = True
except ImportError:
    WebhookManagerInterface = None
    legacy_get_webhook_manager = None
    WEBHOOK_MANAGER_AVAILABLE = False

try:
    from services.interfaces import BusService as IBusService
    BUS_SERVICE_AVAILABLE = True
except ImportError:
    IBusService = None
    BUS_SERVICE_AVAILABLE = False

try:
    from config.mtls_config import MTLS_SETTINGS
    MTLS_AVAILABLE = True
except ImportError:
    MTLS_SETTINGS = None
    MTLS_AVAILABLE = False

try:
    from services import service_manager as _service_manager
    LEGACY_SERVICE_MANAGER_AVAILABLE = True
except ImportError:
    _service_manager = None
    LEGACY_SERVICE_MANAGER_AVAILABLE = False


logger = get_logger(__name__)


class ServiceContainer:
    """Verwaltet Konfigurationen und Service-Factories für die App.

    Der Container kapselt die Erstellung und Bereitstellung von Abhängigkeiten
    (z. B. Settings, Middleware-Konfiguration, Service Manager, KEI‑Bus,
    Rate Limiting), damit diese zentral verwaltet und im Startup/Shutdown
    koordiniert werden können.

    Attribute:
        settings: Global konfigurierte Anwendungseinstellungen.
        jwt_secret: Zufälliges Secret für JWT, falls nicht vorgegeben.
    """

    def __init__(self, app_settings: Settings | None = None) -> None:
        # Settings initialisieren (Singleton aus config.settings)
        self._settings: Settings = app_settings or settings

        # JWT Secret sicherstellen und im Environment bereitstellen
        self._jwt_secret: str = os.getenv("JWT_SECRET_KEY", "") or secrets.token_urlsafe(32)
        os.environ["JWT_SECRET_KEY"] = self._jwt_secret

        # Globalen DI-Container bootstrappen (Standard-Registrierungen)
        try:
            bootstrap_defaults()
        except Exception:
            # Bootstrap darf nicht kritisch fehlschlagen
            pass

        # Lazy-initialisierte Referenzen (werden bei Bedarf gesetzt)
        self._service_manager = None
        self._rate_limit_manager_factory: Callable[[], Any] | None = None
        self._bus_service_factory: Callable[[], Any] | None = None
        self._webhook_manager_factory: Callable[[], Any] | None = None

        # Externe, optionale Komponenten (werden in StartupManager gesetzt)
        self.grpc_server: Any | None = None
        self.work_stealer: Any | None = None

        # Interface-Registrierungen für erste Leaf-Services (DI)
        try:
            if not DOMAIN_REVALIDATION_AVAILABLE:
                raise ImportError("Domain revalidation services not available")

            container = get_container()

            def domain_reval_factory(_: Any) -> DomainRevalidationServiceInterface:
                # Leaf-Service ohne weitere Dependencies
                return DomainRevalidationService()

            container.register(DomainRevalidationServiceInterface, domain_reval_factory, singleton=True)

            # Rate Limiter Backend Registrierung (Redis/Memory via Manager)
            if not RATE_LIMITER_AVAILABLE:
                raise ImportError("Rate limiter services not available")

            def rate_limiter_factory(_: Any) -> RateLimiterBackend:
                # Wählt Backend basierend auf Konfiguration
                if get_rate_limit_config is None:
                    # Fallback auf Memory-Backend wenn Konfiguration nicht verfügbar
                    return MemoryRateLimiter()
                cfg = get_rate_limit_config()
                if cfg.backend in (RLBackendKind.REDIS, RLBackendKind.HYBRID):
                    return RedisRateLimiter(cfg)
                return MemoryRateLimiter()

            container.register(RateLimiterBackend, rate_limiter_factory, singleton=True)

            # Service Manager Registrierung über bestehende Implementierung
            if not SERVICE_MANAGER_AVAILABLE:
                raise ImportError("Service manager not available")

            def service_manager_factory(_: Any) -> ServiceManagerInterface:
                return ConcreteServiceManager()

            container.register(ServiceManagerInterface, service_manager_factory, singleton=True)

            # Webhook Manager Registrierung via bestehender Factory
            if not WEBHOOK_MANAGER_AVAILABLE:
                raise ImportError("Webhook manager not available")

            def webhook_manager_factory(_: Any) -> WebhookManagerInterface:
                if legacy_get_webhook_manager is None:
                    raise RuntimeError("Webhook manager factory not available")
                return legacy_get_webhook_manager()

            container.register(WebhookManagerInterface, webhook_manager_factory, singleton=True)
        except Exception as exc:
            logger.debug(f"DomainRevalidation Interface-Registrierung übersprungen: {exc}")

    @property
    def settings(self) -> Settings:
        """Gibt die Anwendungseinstellungen zurück."""
        return self._settings

    @property
    def jwt_secret(self) -> str:
        """Gibt das JWT Secret zurück (aus ENV oder generiert)."""
        return self._jwt_secret

    @property
    def middleware_config(self) -> MiddlewareConfig:
        """Erstellt eine Middleware-Konfiguration basierend auf Settings.

        Returns:
            MiddlewareConfig: Konfiguration für den Middleware-Stack.
        """
        cors_origins = getattr(self._settings, "cors_allowed_origins_list", [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ])
        return MiddlewareConfig(
            cors_origins=cors_origins,
            cors_enabled=True,
            auth_enabled=getattr(self._settings, "auth_enabled", False),  # Aus Settings laden
            environment=getattr(self._settings, "environment", "development"),
        )

    # --- Service Manager -------------------------------------------------
    @property
    def service_manager(self) -> Any:
        """Zugriff auf den globalen Service Manager.

        Import erfolgt lazy, um Zyklen beim App‑Bootstrap zu vermeiden.
        """
        if self._service_manager is None:
            try:
                if not LEGACY_SERVICE_MANAGER_AVAILABLE:
                    raise ImportError("Legacy service manager not available")
                self._service_manager = _service_manager
            except Exception as exc:  # pragma: no cover
                logger.error(f"Service Manager konnte nicht geladen werden: {exc}")
                raise
        return self._service_manager

    # --- KEI‑Bus ---------------------------------------------------------
    def get_bus_service(self) -> Any:
        """Gibt die KEI‑Bus Service‑Instanz zurück.

        Returns:
            Any: KEI‑Bus Service.
        """
        # Bevorzugt DI-Container-Auflösung, fallback auf Legacy-Factory
        try:
            if not BUS_SERVICE_AVAILABLE:
                raise ImportError("Bus service not available")
            container = get_container()
            svc = container.resolve(IBusService)
            return svc
        except Exception:
            if self._bus_service_factory is None:
                try:
                    from kei_bus.service import get_bus_service as factory  # type: ignore
                    self._bus_service_factory = factory
                except Exception as exc:  # pragma: no cover
                    logger.error(f"KEI‑Bus Factory konnte nicht geladen werden: {exc}")
                    raise
            return self._bus_service_factory()

    # --- Rate Limiting ---------------------------------------------------
    def get_rate_limit_manager(self) -> Any:
        """Gibt den Rate Limit Manager zurück.

        Returns:
            Any: Rate Limit Manager Instanz.
        """
        if self._rate_limit_manager_factory is None:
            try:
                from middleware.rate_limiting import (
                    get_rate_limit_manager as factory,  # type: ignore
                )
                self._rate_limit_manager_factory = factory
            except Exception as exc:  # pragma: no cover
                logger.error(f"Rate Limit Factory konnte nicht geladen werden: {exc}")
                raise
        return self._rate_limit_manager_factory()

    # --- Webhook Manager -------------------------------------------------
    def get_webhook_manager(self) -> Any:
        """Gibt den Webhook Manager zurück."""
        if self._webhook_manager_factory is None:
            try:
                from kei_webhook import get_webhook_manager as factory  # type: ignore
                self._webhook_manager_factory = factory
            except Exception as exc:  # pragma: no cover
                logger.error(f"Webhook Manager Factory konnte nicht geladen werden: {exc}")
                raise
        return self._webhook_manager_factory()

    # --- mTLS ------------------------------------------------------------
    @property
    def is_mtls_enabled(self) -> bool:
        """Prüft, ob mTLS inbound aktiviert ist."""
        try:
            if not MTLS_AVAILABLE:
                return False
            return bool(MTLS_SETTINGS.inbound.enabled)  # type: ignore
        except Exception:
            return False

    def get_mtls_middleware_class(self) -> type | None:
        """Gibt die mTLS Middleware‑Klasse zurück, falls verfügbar."""
        if not self.is_mtls_enabled:
            return None
        try:
            from security.mtls_middleware import MTLSMiddleware  # type: ignore
            return MTLSMiddleware
        except Exception:
            return None

    # --- Observability ---------------------------------------------------
    def initialize_tracing(self) -> None:
        """Initialisiert Tracing, falls konfiguriert."""
        try:
            from observability.tracing import init_tracing  # type: ignore
            init_tracing()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Tracing konnte nicht initialisiert werden: {exc}")


__all__ = [
    "ServiceContainer",
]
