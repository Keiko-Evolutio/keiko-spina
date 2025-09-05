"""Startup-Koordinator für orchestrierte Service-Initialisierung.

Dieser Koordinator verwendet die modularisierten Service-Initializer und
Shutdown-Handler um eine saubere Trennung der Verantwortlichkeiten zu erreichen.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..common.constants import (
    DEFAULT_GRPC_BIND,
    ENV_KEI_RPC_ENABLE_GRPC,
    ENV_KEI_RPC_GRPC_BIND,
)
from ..common.logger_utils import get_module_logger, safe_log_exception

# Service-Imports
try:
    from services.lifecycle import lifecycle
    LIFECYCLE_AVAILABLE = True
except ImportError:
    lifecycle = None
    LIFECYCLE_AVAILABLE = False

try:
    from config.settings import settings
    from services.scheduling.work_stealing import WorkStealer
    WORK_STEALER_AVAILABLE = True
except ImportError:
    settings = None
    WorkStealer = None
    WORK_STEALER_AVAILABLE = False
from .logfire_initializer import LogfireInitializer, LogfireShutdownHandler
from .service_initializers import (
    AgentCircuitBreakerInitializer,
    AgentsInitializer,
    BusServiceInitializer,
    DomainRevalidationInitializer,
    MCPServerInitializer,
    MonitoringServiceInitializer,
    ServiceInitializer,
    ServiceManagerInitializer,
    VoicePerformanceInitializer,
    VoiceRateLimitingInitializer,
    VoiceSystemInitializer,
    WebhookAndRateLimitingInitializer,
)
from .shutdown_handlers import (
    AgentsShutdownHandler,
    DomainRevalidationShutdownHandler,
    GrpcShutdownHandler,
    LifecycleShutdownHandler,
    MCPRegistryShutdownHandler,
    RateLimitingShutdownHandler,
    RedisShutdownHandler,
    ShutdownHandler,
    WebhookAndServicesShutdownHandler,
    WebSocketShutdownHandler,
    WorkStealerShutdownHandler,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ..service_container import ServiceContainer

logger = get_module_logger(__name__)


class StartupCoordinator:
    """Koordiniert die Startup- und Shutdown-Sequenzen."""

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container
        self._service_initializers: list[ServiceInitializer] = []
        self._shutdown_handlers: list[ShutdownHandler] = []
        self._setup_initializers()
        self._setup_shutdown_handlers()

    def _setup_initializers(self) -> None:
        """Erstellt die Service-Initializer in der richtigen Reihenfolge."""
        self._service_initializers = [
            MonitoringServiceInitializer(self.container),  # Früh initialisieren für Monitoring
            LogfireInitializer(self.container),  # Logfire nach Monitoring starten
            VoiceRateLimitingInitializer(self.container),  # Rate Limiting vor Voice Services
            AgentCircuitBreakerInitializer(self.container),  # Circuit Breaker vor Agent Services
            VoicePerformanceInitializer(self.container),  # Performance Optimization vor Voice System
            BusServiceInitializer(self.container),
            ServiceManagerInitializer(self.container),
            AgentsInitializer(self.container),
            VoiceSystemInitializer(self.container),
            MCPServerInitializer(self.container),
            DomainRevalidationInitializer(self.container),
            WebhookAndRateLimitingInitializer(self.container),
        ]

    def _setup_shutdown_handlers(self) -> None:
        """Erstellt die Shutdown-Handler in der richtigen Reihenfolge."""
        self._shutdown_handlers = [
            WebSocketShutdownHandler(self.container),
            DomainRevalidationShutdownHandler(self.container),
            RateLimitingShutdownHandler(self.container),
            GrpcShutdownHandler(self.container),
            RedisShutdownHandler(self.container),
            AgentsShutdownHandler(self.container),
            WebhookAndServicesShutdownHandler(self.container),
            WorkStealerShutdownHandler(self.container),
            LifecycleShutdownHandler(self.container),
            MCPRegistryShutdownHandler(self.container),
            LogfireShutdownHandler(self.container),  # Logfire als letztes beenden
        ]

    async def startup(self, app: FastAPI) -> None:
        """Führt die komplette Startup-Sequenz aus."""
        logger.info("Starte Startup-Sequenz...")

        # 0) DI-Container mit Standard-Services initialisieren
        await self._bootstrap_di_container()

        # 1) Environment-Validierung
        await self._validate_environment()

        # 2) Tracing initialisieren
        self._initialize_tracing()

        # 3) Warmup Hooks
        await self._run_warmup()

        # 4) Service-Initializer ausführen
        await self._run_service_initializers(app)

        # 5) Externe Services starten
        await self._start_external_services(app)

        logger.info("Startup-Sequenz abgeschlossen")

    async def _bootstrap_di_container(self) -> None:
        """Initialisiert DI-Container mit Standard-Services."""
        try:
            from core.container import bootstrap_defaults
            bootstrap_defaults()
            logger.info("✅ DI-Container mit Standard-Services initialisiert")
        except Exception as exc:
            logger.warning(f"⚠️ DI-Container-Initialisierung fehlgeschlagen: {exc}")

    async def shutdown(self, app: FastAPI) -> None:
        """Führt die komplette Shutdown-Sequenz aus."""
        logger.info("Starte Shutdown-Sequenz...")

        # Shutdown-Handler in umgekehrter Reihenfolge ausführen
        for handler in self._shutdown_handlers:
            try:
                await handler.shutdown(app)
            except Exception as exc:
                safe_log_exception(
                    logger,
                    exc,
                    f"Fehler in {handler.__class__.__name__}",
                    handler=handler.__class__.__name__
                )

        logger.info("Shutdown-Sequenz abgeschlossen")

    async def _validate_environment(self) -> None:
        """Validiert die Environment-Konfiguration."""
        try:
            # Environment-Validierung wird in application.py durchgeführt
            logger.info("Environment-Validierung übersprungen (bereits in application.py)")
        except Exception as exc:
            logger.critical(f"Startup abgebrochen: {exc}")
            raise

    def _initialize_tracing(self) -> None:
        """Initialisiert Tracing."""
        self.container.initialize_tracing()

    async def _run_warmup(self) -> None:
        """Führt Warmup-Hooks aus."""
        try:
            if not LIFECYCLE_AVAILABLE:
                raise ImportError("Lifecycle service not available")
            await lifecycle.run_warmup()
            logger.info("Warmup abgeschlossen")
        except Exception as exc:
            logger.warning(f"Warmup konnte nicht vollständig abgeschlossen werden: {exc}")

    async def _run_service_initializers(self, app) -> None:
        """Führt alle Service-Initializer aus."""
        success_count = 0

        for initializer in self._service_initializers:
            try:
                # Spezielle Behandlung für LogfireInitializer
                if initializer.__class__.__name__ == "LogfireInitializer":
                    success = await initializer.initialize(app)
                else:
                    success = await initializer.initialize()

                if success:
                    success_count += 1
                    logger.debug(f"{initializer.__class__.__name__} erfolgreich")
                else:
                    logger.warning(f"{initializer.__class__.__name__} fehlgeschlagen")
            except Exception as exc:
                safe_log_exception(
                    logger,
                    exc,
                    f"Fehler in {initializer.__class__.__name__}",
                    initializer=initializer.__class__.__name__
                )

        logger.info(f"Service-Initialisierung: {success_count}/{len(self._service_initializers)} erfolgreich")

    async def _start_external_services(self, app: FastAPI) -> None:
        """Startet externe Services (gRPC, Work-Stealer)."""
        await self._maybe_start_grpc(app)
        await self._start_work_stealer(app)

    async def _maybe_start_grpc(self, app: FastAPI) -> None:
        """Startet optional den gRPC Server."""
        app.state.grpc_server = None  # type: ignore[attr-defined]

        if os.getenv(ENV_KEI_RPC_ENABLE_GRPC, "false").lower() == "true":
            try:
                from api.grpc.grpc_server import serve_grpc
                bind_addr = os.getenv(ENV_KEI_RPC_GRPC_BIND, DEFAULT_GRPC_BIND)
                app.state.grpc_server = await serve_grpc(bind_addr)  # type: ignore[attr-defined]
                self.container.grpc_server = app.state.grpc_server  # type: ignore[attr-defined]
                logger.info("gRPC Server gestartet")
            except Exception as exc:
                safe_log_exception(logger, exc, "gRPC Server konnte nicht gestartet werden", service="grpc")

    async def _start_work_stealer(self, app: FastAPI) -> None:
        """Startet den Work-Stealer."""
        try:
            if not WORK_STEALER_AVAILABLE:
                raise ImportError("Work stealer services not available")

            local_queue = getattr(settings, "agent_orchestrator_id", None) or "orchestrator"
            app.state.work_stealer = WorkStealer(local_queue=local_queue)  # type: ignore[attr-defined]
            await app.state.work_stealer.start()  # type: ignore[attr-defined]
            self.container.work_stealer = app.state.work_stealer  # type: ignore[attr-defined]
            logger.info("Work-Stealer gestartet")
        except Exception as exc:
            safe_log_exception(logger, exc, "Work-Stealer konnte nicht gestartet werden", service="work_stealer")


__all__ = [
    "StartupCoordinator",
]
