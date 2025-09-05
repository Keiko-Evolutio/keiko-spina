"""Shutdown-Handler für das Startup-Management.

Dieses Modul extrahiert die Shutdown-Logik aus StartupManager
um die Komplexität zu reduzieren und Single Responsibility Principle zu befolgen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from ..common.constants import (
    SERVICE_AGENTS,
    SERVICE_DOMAIN_REVALIDATION,
    SERVICE_GRPC,
    SERVICE_RATE_LIMITING,
    SERVICE_REDIS,
    SERVICE_WEBHOOK,
    SERVICE_WORK_STEALER,
    WEBSOCKET_CLOSE_CODE_SHUTDOWN,
    WEBSOCKET_CLOSE_REASON_SHUTDOWN,
)
from ..common.error_handlers import handle_async_service_errors
from ..common.logger_utils import log_service_operation, safe_log_exception

# Service-Imports
try:
    from services.streaming import websocket_manager
    STREAMING_AVAILABLE = True
except ImportError:
    websocket_manager = None
    STREAMING_AVAILABLE = False

try:
    from services.unified_domain_revalidation_service import UnifiedDomainRevalidationService
    DOMAIN_REVALIDATION_AVAILABLE = True
except ImportError:
    UnifiedDomainRevalidationService = None
    DOMAIN_REVALIDATION_AVAILABLE = False

try:
    from core.container import get_container
    from services.interfaces.service_manager import ServiceManagerInterface
    from services.interfaces.webhook_manager import WebhookManagerInterface
    CONTAINER_SERVICES_AVAILABLE = True
except ImportError:
    get_container = None
    WebhookManagerInterface = None
    ServiceManagerInterface = None
    CONTAINER_SERVICES_AVAILABLE = False

try:
    from services.lifecycle import lifecycle
    LIFECYCLE_AVAILABLE = True
except ImportError:
    lifecycle = None
    LIFECYCLE_AVAILABLE = False

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ..service_container import ServiceContainer

logger = get_logger(__name__)


class ShutdownHandler:
    """Basis-Klasse für Shutdown-Handler."""

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container

    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Führt den Shutdown durch.

        Args:
            app: Optional FastAPI-Instanz

        Returns:
            True wenn erfolgreich, False bei Fehlern
        """
        raise NotImplementedError("Subclasses must implement shutdown()")


class WebSocketShutdownHandler(ShutdownHandler):
    """Handler für WebSocket-Verbindungen Shutdown."""

    @handle_async_service_errors("websockets", fallback_value=False)
    @log_service_operation(logger, "websockets", "shutdown")
    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Schließt alle WebSocket-Verbindungen."""
        try:
            if not STREAMING_AVAILABLE or websocket_manager is None:
                raise ImportError("WebSocket manager not available")

            stats = websocket_manager.get_stats()
            active_count = stats.get("active_connections", 0)

            if active_count > 0:
                logger.info(f"Schließe {active_count} WebSocket-Verbindungen")
                await self._close_all_connections(websocket_manager)

            return True
        except Exception as exc:
            logger.debug(f"WebSocket Manager nicht verfügbar: {exc}")
            return False

    @staticmethod
    async def _close_all_connections(websocket_manager: Any) -> None:
        """Schließt alle aktiven WebSocket-Verbindungen."""
        from fastapi.websockets import WebSocketState

        for connection_id in list(websocket_manager.connections.keys()):
            try:
                connection = websocket_manager.get_connection(connection_id)
                if connection and connection.websocket.client_state == WebSocketState.CONNECTED:
                    await connection.websocket.close(
                        code=WEBSOCKET_CLOSE_CODE_SHUTDOWN,
                        reason=WEBSOCKET_CLOSE_REASON_SHUTDOWN
                    )
            except Exception as exc:
                logger.debug(f"WebSocket Cleanup Fehler: {exc}")
            finally:
                await websocket_manager.disconnect(connection_id)


class DomainRevalidationShutdownHandler(ShutdownHandler):
    """Handler für Domain-Revalidierung Shutdown."""

    @handle_async_service_errors(SERVICE_DOMAIN_REVALIDATION, fallback_value=False)
    @log_service_operation(logger, SERVICE_DOMAIN_REVALIDATION, "shutdown")
    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Stoppt den Domain-Revalidierung Service."""
        if not DOMAIN_REVALIDATION_AVAILABLE:
            return False
        service = UnifiedDomainRevalidationService()
        await service.stop()
        return True


class RateLimitingShutdownHandler(ShutdownHandler):
    """Handler für Rate Limiting Shutdown."""

    @handle_async_service_errors(SERVICE_RATE_LIMITING, fallback_value=False)
    @log_service_operation(logger, SERVICE_RATE_LIMITING, "shutdown")
    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Beendet das Rate Limiting System."""
        from middleware.rate_limiting import shutdown_rate_limiting
        await shutdown_rate_limiting()
        return True


class GrpcShutdownHandler(ShutdownHandler):
    """Handler für gRPC Server Shutdown."""

    @handle_async_service_errors(SERVICE_GRPC, fallback_value=False)
    @log_service_operation(logger, SERVICE_GRPC, "shutdown")
    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Beendet den gRPC Server."""
        if not app:
            return False

        try:
            if getattr(app.state, "grpc_server", None):  # type: ignore[attr-defined]
                from api.grpc.grpc_server import shutdown_grpc
                await shutdown_grpc(app.state.grpc_server)  # type: ignore[attr-defined]
            return True
        except Exception as exc:
            safe_log_exception(logger, exc, "Fehler beim Beenden des gRPC Servers", service=SERVICE_GRPC)
            return False


class RedisShutdownHandler(ShutdownHandler):
    """Handler für Redis Cache Shutdown."""

    @handle_async_service_errors(SERVICE_REDIS, fallback_value=False)
    @log_service_operation(logger, SERVICE_REDIS, "shutdown")
    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Schließt die Redis-Verbindung."""
        try:
            from storage.cache.redis_cache import close_redis_connection
            if close_redis_connection:
                await close_redis_connection()
                return True
        except Exception as exc:
            logger.debug(f"Redis Close Fehler: {exc}")
        return False


class AgentsShutdownHandler(ShutdownHandler):
    """Handler für Agent System Shutdown."""

    @handle_async_service_errors(SERVICE_AGENTS, fallback_value=False)
    @log_service_operation(logger, SERVICE_AGENTS, "shutdown")
    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Beendet das Agent System."""
        from agents.common import shutdown_agents
        await shutdown_agents()
        return True


class WebhookAndServicesShutdownHandler(ShutdownHandler):
    """Handler für Webhook Worker-Pool und Services Cleanup."""

    @handle_async_service_errors(SERVICE_WEBHOOK, fallback_value=False)
    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Stoppt Webhook Worker-Pool und bereinigt Services."""
        webhook_success = await self._stop_webhook_pool()
        services_success = await self._cleanup_services()
        return webhook_success or services_success

    @staticmethod
    async def _stop_webhook_pool() -> bool:
        """Stoppt den Webhook Worker-Pool."""
        try:
            if not CONTAINER_SERVICES_AVAILABLE:
                raise ImportError("Container services not available")
            webhook_mgr = get_container().resolve(WebhookManagerInterface)
            await webhook_mgr.stop_worker_pool()
            return True
        except Exception as exc:
            logger.debug(f"Webhook Worker-Pool Stop Fehler: {exc}")
            return False

    @log_service_operation(logger, "service_manager", "cleanup")
    async def _cleanup_services(self) -> bool:
        """Bereinigt den Service Manager."""
        try:
            if not CONTAINER_SERVICES_AVAILABLE:
                raise ImportError("Container services not available")
            svc = get_container().resolve(ServiceManagerInterface)
            if svc is not None:
                await svc.cleanup()
                return True
        except Exception as exc:
            safe_log_exception(logger, exc, "Fehler beim Bereinigen des Service Managers")
        return False


class WorkStealerShutdownHandler(ShutdownHandler):
    """Handler für Work-Stealer Shutdown."""

    @handle_async_service_errors(SERVICE_WORK_STEALER, fallback_value=False)
    @log_service_operation(logger, SERVICE_WORK_STEALER, "shutdown")
    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Stoppt den Work-Stealer."""
        if not app:
            return False

        try:
            work_stealer = getattr(app.state, "work_stealer", None)  # type: ignore[attr-defined]
            if work_stealer is not None:
                await work_stealer.stop()
                # Work Stealer aus app.state entfernen
                delattr(app.state, "work_stealer")  # type: ignore[attr-defined]
            return True
        except Exception as exc:
            safe_log_exception(logger, exc, "Fehler beim Stoppen des Work-Stealers", service=SERVICE_WORK_STEALER)
            return False


class LifecycleShutdownHandler(ShutdownHandler):
    """Handler für Lifecycle Shutdown Hooks."""

    @handle_async_service_errors("lifecycle", fallback_value=False)
    @log_service_operation(logger, "lifecycle", "shutdown_hooks")
    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Führt Graceful Shutdown Hooks aus."""
        if not LIFECYCLE_AVAILABLE:
            return False
        await lifecycle.run_shutdown()
        return True


class MCPRegistryShutdownHandler(ShutdownHandler):
    """Handler für MCP Registry Shutdown."""

    @handle_async_service_errors("mcp_registry", fallback_value=False)
    @log_service_operation(logger, "mcp_registry", "shutdown")
    async def shutdown(self, app: FastAPI | None = None) -> bool:
        """Beendet die externe MCP Registry."""
        try:
            from agents.tools.mcp import mcp_registry
            await mcp_registry.shutdown()
            return True
        except Exception as exc:
            logger.debug(f"MCP Registry Shutdown Hinweis: {exc}")
            return False


__all__ = [
    "AgentsShutdownHandler",
    "DomainRevalidationShutdownHandler",
    "GrpcShutdownHandler",
    "LifecycleShutdownHandler",
    "MCPRegistryShutdownHandler",
    "RateLimitingShutdownHandler",
    "RedisShutdownHandler",
    "ShutdownHandler",
    "WebSocketShutdownHandler",
    "WebhookAndServicesShutdownHandler",
    "WorkStealerShutdownHandler",
]
