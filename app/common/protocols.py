"""Protocol Definitions für externe Dependencies.

Diese Datei definiert Protocols für externe Dependencies um Type Safety
zu verbessern ohne direkte Abhängigkeiten zu schaffen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable


class LoggerProtocol(Protocol):
    """Protocol für Logger-Objekte."""

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Loggt eine Debug-Nachricht."""
        ...

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Loggt eine Info-Nachricht."""
        ...

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Loggt eine Warning-Nachricht."""
        ...

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Loggt eine Error-Nachricht."""
        ...

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Loggt eine Critical-Nachricht."""
        ...


class ServiceProtocol(Protocol):
    """Protocol für Service-Objekte."""

    async def initialize(self) -> None:
        """Initialisiert den Service."""
        ...

    async def cleanup(self) -> None:
        """Bereinigt den Service."""
        ...

    def is_healthy(self) -> bool:
        """Prüft ob der Service gesund ist."""
        ...


class BusServiceProtocol(Protocol):
    """Protocol für KEI-Bus Service."""

    async def initialize(self) -> None:
        """Initialisiert den Bus Service."""
        ...

    async def publish(self, topic: str, message: Any) -> None:
        """Publiziert eine Nachricht."""
        ...

    async def subscribe(self, topic: str, handler: Any) -> None:
        """Abonniert ein Topic."""
        ...


class ServiceManagerProtocol(Protocol):
    """Protocol für Service Manager."""

    async def initialize(self) -> None:
        """Initialisiert den Service Manager."""
        ...

    async def cleanup(self) -> None:
        """Bereinigt den Service Manager."""
        ...

    def get_service(self, service_name: str) -> Any | None:
        """Gibt einen Service zurück."""
        ...


class WebSocketManagerProtocol(Protocol):
    """Protocol für WebSocket Manager."""

    def get_stats(self) -> dict[str, Any]:
        """Gibt WebSocket-Statistiken zurück."""
        ...

    def get_connection(self, connection_id: str) -> Any | None:
        """Gibt eine WebSocket-Verbindung zurück."""
        ...

    async def disconnect(self, connection_id: str) -> None:
        """Trennt eine WebSocket-Verbindung."""
        ...

    async def send_json_to_connection(
        self,
        connection_id: str,
        data: dict[str, Any]
    ) -> None:
        """Sendet JSON-Daten an eine Verbindung."""
        ...


class MiddlewareConfigProtocol(Protocol):
    """Protocol für Middleware-Konfiguration."""

    cors_origins: list[str]
    enable_cors: bool
    enable_gzip: bool
    enable_security_headers: bool


class SettingsProtocol(Protocol):
    """Protocol für Settings-Objekte."""

    cors_allowed_origins_list: list[str] | None
    agent_orchestrator_id: str | None

    def get(self, key: str, default: Any = None) -> Any:
        """Gibt einen Konfigurationswert zurück."""
        ...


class ContainerProtocol(Protocol):
    """Protocol für DI-Container."""

    def resolve(self, interface: type) -> Any:
        """Löst eine Abhängigkeit auf."""
        ...

    def register(self, interface: type, implementation: type) -> None:
        """Registriert eine Implementierung."""
        ...


class WorkStealerProtocol(Protocol):
    """Protocol für Work-Stealer."""

    async def start(self) -> None:
        """Startet den Work-Stealer."""
        ...

    async def stop(self) -> None:
        """Stoppt den Work-Stealer."""
        ...


class GrpcServerProtocol(Protocol):
    """Protocol für gRPC Server."""

    async def start(self) -> None:
        """Startet den gRPC Server."""
        ...

    async def stop(self) -> None:
        """Stoppt den gRPC Server."""
        ...


class HealthCheckProtocol(Protocol):
    """Protocol für Health Check Funktionen."""

    def __call__(self) -> bool | Awaitable[bool]:
        """Führt einen Health Check durch."""
        ...


class VoiceHealthMonitorProtocol(Protocol):
    """Protocol für Voice Health Monitor."""

    async def perform_startup_validation(self) -> Any:
        """Führt Startup-Validierung durch."""
        ...


class VoiceMonitoringManagerProtocol(Protocol):
    """Protocol für Voice Monitoring Manager."""

    async def start_monitoring(self) -> None:
        """Startet das Monitoring."""
        ...


class MCPRegistryProtocol(Protocol):
    """Protocol für MCP Registry."""

    async def register_server(self, config: Any, domain_validated: bool = False) -> bool:
        """Registriert einen MCP Server."""
        ...

    async def shutdown(self) -> None:
        """Beendet die Registry."""
        ...


class RateLimitingConfigProtocol(Protocol):
    """Protocol für Rate Limiting Konfiguration."""

    def get_tenant_config(self, tenant_id: str) -> dict[str, Any] | None:
        """Gibt Tenant-Konfiguration zurück."""
        ...

    def get_endpoint_config(self, pattern: str) -> dict[str, Any] | None:
        """Gibt Endpoint-Konfiguration zurück."""
        ...


class LifecycleProtocol(Protocol):
    """Protocol für Lifecycle Manager."""

    async def run_warmup(self) -> None:
        """Führt Warmup aus."""
        ...

    async def run_shutdown(self) -> None:
        """Führt Shutdown aus."""
        ...


__all__ = [
    "BusServiceProtocol",
    "ContainerProtocol",
    "GrpcServerProtocol",
    "HealthCheckProtocol",
    "LifecycleProtocol",
    "LoggerProtocol",
    "MCPRegistryProtocol",
    "MiddlewareConfigProtocol",
    "RateLimitingConfigProtocol",
    "ServiceManagerProtocol",
    "ServiceProtocol",
    "SettingsProtocol",
    "VoiceHealthMonitorProtocol",
    "VoiceMonitoringManagerProtocol",
    "WebSocketManagerProtocol",
    "WorkStealerProtocol",
]
