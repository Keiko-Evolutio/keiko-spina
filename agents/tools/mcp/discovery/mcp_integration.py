# backend/kei_mcp/discovery/mcp_integration.py
"""Erweiterte MCP-Integration für KEI-MCP Interface.

Verbindet Discovery-System mit Agent-System, implementiert automatische
Server-Registration, Health-Checks und Lifecycle-Management.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function

from .prompt_discovery import DiscoveredPrompt, prompt_discovery_engine
from .resource_discovery import DiscoveredResource, resource_discovery_engine
from .tool_discovery import DiscoveredTool, tool_discovery_engine

if TYPE_CHECKING:
    from ..kei_mcp_registry import KEIMCPRegistry, RegisteredMCPServer

logger = get_logger(__name__)


class IntegrationStatus(str, Enum):
    """Status der MCP-Integration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    STOPPED = "stopped"


class ServerHealthStatus(str, Enum):
    """Gesundheitsstatus eines MCP-Servers."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class DiscoveryMetrics:
    """Metriken für Discovery-Operationen."""
    total_servers: int = 0
    healthy_servers: int = 0
    total_tools: int = 0
    available_tools: int = 0
    total_resources: int = 0
    available_resources: int = 0
    total_prompts: int = 0
    active_prompts: int = 0
    last_discovery_duration_ms: float = 0.0
    discovery_success_rate: float = 1.0
    avg_server_response_time_ms: float = 0.0


@dataclass
class ServerHealthMetrics:
    """Gesundheitsmetriken für MCP-Server."""
    server_name: str
    status: ServerHealthStatus
    last_check: datetime
    response_time_ms: float
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    error_rate: float = 0.0
    last_error: str | None = None


@dataclass
class IntegrationConfig:
    """Konfiguration für MCP-Integration."""
    discovery_interval_seconds: int = 300  # 5 Minuten
    health_check_interval_seconds: int = 60  # 1 Minute
    server_timeout_seconds: int = 30
    max_consecutive_failures: int = 3
    auto_recovery_enabled: bool = True
    failover_enabled: bool = True
    cache_enabled: bool = True
    metrics_retention_hours: int = 24


class MCPIntegrationEngine:
    """Engine für erweiterte MCP-Integration."""

    def __init__(self, config: IntegrationConfig | None = None) -> None:
        """Initialisiert MCP-Integration-Engine."""
        self._config = config or IntegrationConfig()
        self._registry: KEIMCPRegistry | None = None
        self._status = IntegrationStatus.INITIALIZING
        self._discovery_metrics = DiscoveryMetrics()
        self._server_health_metrics: dict[str, ServerHealthMetrics] = {}
        self._discovery_task: asyncio.Task | None = None
        self._health_check_task: asyncio.Task | None = None
        self._agent_integrations: dict[str, Any] = {}
        self._failover_mappings: dict[str, list[str]] = {}
        self._last_full_discovery: datetime | None = None

    async def initialize(self, registry: KEIMCPRegistry) -> None:
        """Initialisiert Integration-Engine mit Registry.

        Args:
            registry: KEI-MCP Registry-Instanz
        """
        self._registry = registry
        self._status = IntegrationStatus.INITIALIZING

        logger.info("Initialisiere MCP-Integration-Engine")

        try:
            # Initiale Discovery durchführen
            await self._perform_full_discovery()

            # Background-Tasks starten
            await self._start_background_tasks()

            self._status = IntegrationStatus.ACTIVE
            logger.info("MCP-Integration-Engine erfolgreich initialisiert")

        except Exception as e:
            self._status = IntegrationStatus.ERROR
            logger.exception(f"MCP-Integration-Engine Initialisierung fehlgeschlagen: {e}")
            raise

    async def shutdown(self) -> None:
        """Fährt Integration-Engine herunter."""
        logger.info("Fahre MCP-Integration-Engine herunter")

        self._status = IntegrationStatus.STOPPED

        # Background-Tasks stoppen
        if self._discovery_task:
            self._discovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._discovery_task

        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        logger.info("MCP-Integration-Engine heruntergefahren")

    @trace_function("mcp.integration.full_discovery")
    async def _perform_full_discovery(self) -> None:
        """Führt vollständige Discovery aller registrierten Server durch."""
        if not self._registry:
            raise RuntimeError("Registry nicht initialisiert")

        start_time = datetime.now(UTC)

        try:
            servers = self._registry.get_all_servers()

            logger.info(f"Starte vollständige Discovery für {len(servers)} Server")

            # Parallele Discovery für alle Server-Typen
            discovery_tasks = [
                tool_discovery_engine.discover_all_tools(servers, force_refresh=True),
                resource_discovery_engine.discover_all_resources(servers, force_refresh=True),
                prompt_discovery_engine.discover_all_prompts(servers, force_refresh=True)
            ]

            tools, resources, prompts = await asyncio.gather(*discovery_tasks)

            # Metriken aktualisieren
            self._update_discovery_metrics(servers, tools, resources, prompts, start_time)

            # Agent-Integrationen aktualisieren
            await self._update_agent_integrations(tools, resources, prompts)

            self._last_full_discovery = datetime.now(UTC)

            logger.info(
                f"Vollständige Discovery abgeschlossen: "
                f"{len(tools)} Tools, {len(resources)} Ressourcen, {len(prompts)} Prompts"
            )

        except Exception as e:
            logger.exception(f"Vollständige Discovery fehlgeschlagen: {e}")
            self._status = IntegrationStatus.ERROR
            raise

    def _update_discovery_metrics(
        self,
        servers: dict[str, RegisteredMCPServer],
        tools: list[DiscoveredTool],
        resources: list[DiscoveredResource],
        prompts: list[DiscoveredPrompt],
        start_time: datetime
    ) -> None:
        """Aktualisiert Discovery-Metriken."""
        duration = (datetime.now(UTC) - start_time).total_seconds() * 1000

        self._discovery_metrics.total_servers = len(servers)
        self._discovery_metrics.healthy_servers = sum(1 for s in servers.values() if s.is_healthy)
        self._discovery_metrics.total_tools = len(tools)
        self._discovery_metrics.available_tools = sum(1 for t in tools if t.is_available())
        self._discovery_metrics.total_resources = len(resources)
        self._discovery_metrics.available_resources = sum(
            1 for r in resources if r.status.value == "available"
        )
        self._discovery_metrics.total_prompts = len(prompts)
        self._discovery_metrics.active_prompts = sum(
            1 for p in prompts if p.status.value == "active"
        )
        self._discovery_metrics.last_discovery_duration_ms = duration

        # Erfolgsrate berechnen
        if self._discovery_metrics.total_servers > 0:
            self._discovery_metrics.discovery_success_rate = (
                self._discovery_metrics.healthy_servers / self._discovery_metrics.total_servers
            )

        # Durchschnittliche Server-Antwortzeit
        response_times = [s.avg_response_time_ms for s in servers.values() if s.avg_response_time_ms > 0]
        if response_times:
            self._discovery_metrics.avg_server_response_time_ms = sum(response_times) / len(response_times)

    async def _update_agent_integrations(
        self,
        tools: list[DiscoveredTool],
        resources: list[DiscoveredResource],
        prompts: list[DiscoveredPrompt]
    ) -> None:
        """Aktualisiert Agent-Integrationen mit entdeckten Komponenten."""
        try:
            # Tools nach Capabilities gruppieren
            tools_by_capability = {}
            for tool in tools:
                if tool.is_available():
                    for tag in tool.metadata.tags:
                        if tag not in tools_by_capability:
                            tools_by_capability[tag] = []
                        tools_by_capability[tag].append(tool)

            # Ressourcen nach Typ gruppieren
            resources_by_type = {}
            for resource in resources:
                resource_type = resource.metadata.resource_type.value
                if resource_type not in resources_by_type:
                    resources_by_type[resource_type] = []
                resources_by_type[resource_type].append(resource)

            # Prompts nach Kategorie gruppieren
            prompts_by_category = {}
            for prompt in prompts:
                category = prompt.metadata.category.value
                if category not in prompts_by_category:
                    prompts_by_category[category] = []
                prompts_by_category[category].append(prompt)

            # Integration-Mappings aktualisieren
            self._agent_integrations = {
                "tools_by_capability": tools_by_capability,
                "resources_by_type": resources_by_type,
                "prompts_by_category": prompts_by_category,
                "last_updated": datetime.now(UTC).isoformat()
            }

            logger.debug(f"Agent-Integrationen aktualisiert: {len(tools_by_capability)} Tool-Capabilities")

        except Exception as e:
            logger.exception(f"Agent-Integration-Update fehlgeschlagen: {e}")

    async def _start_background_tasks(self) -> None:
        """Startet Background-Tasks für Discovery und Health-Checks."""
        # Discovery-Task
        self._discovery_task = asyncio.create_task(self._discovery_loop())

        # Health-Check-Task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Background-Tasks für MCP-Integration gestartet")

    async def _discovery_loop(self) -> None:
        """Background-Loop für regelmäßige Discovery."""
        while self._status in [IntegrationStatus.ACTIVE, IntegrationStatus.DEGRADED]:
            try:
                await asyncio.sleep(self._config.discovery_interval_seconds)

                if self._status == IntegrationStatus.ACTIVE:
                    await self._perform_full_discovery()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Discovery-Loop Fehler: {e}")
                self._status = IntegrationStatus.DEGRADED
                await asyncio.sleep(60)  # Warte vor Retry

    async def _health_check_loop(self) -> None:
        """Background-Loop für Server-Health-Checks."""
        while self._status in [IntegrationStatus.ACTIVE, IntegrationStatus.DEGRADED]:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                if self._registry:
                    await self._perform_health_checks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Health-Check-Loop Fehler: {e}")
                await asyncio.sleep(30)  # Warte vor Retry

    async def _perform_health_checks(self) -> None:
        """Führt Health-Checks für alle Server durch."""
        if not self._registry:
            return

        servers = self._registry.get_all_servers()

        for server_name in servers:
            try:
                server = self._registry.get_server(server_name)
                if server:
                    await self._check_server_health(server_name, server)
            except Exception as e:
                logger.exception(f"Health-Check für Server {server_name} fehlgeschlagen: {e}")

    async def _check_server_health(self, server_name: str, server: RegisteredMCPServer) -> None:
        """Führt Health-Check für einzelnen Server durch."""
        start_time = datetime.now(UTC)

        try:
            # Vereinfachter Health-Check (kann erweitert werden)
            is_healthy = server.is_healthy
            response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            # Metriken aktualisieren
            if server_name not in self._server_health_metrics:
                self._server_health_metrics[server_name] = ServerHealthMetrics(
                    server_name=server_name,
                    status=ServerHealthStatus.UNKNOWN,
                    last_check=start_time,
                    response_time_ms=response_time
                )

            metrics = self._server_health_metrics[server_name]
            metrics.last_check = start_time
            metrics.response_time_ms = response_time

            if is_healthy:
                metrics.status = ServerHealthStatus.HEALTHY
                metrics.consecutive_failures = 0
            else:
                metrics.consecutive_failures += 1

                if metrics.consecutive_failures >= self._config.max_consecutive_failures:
                    metrics.status = ServerHealthStatus.FAILED

                    # Auto-Recovery versuchen
                    if self._config.auto_recovery_enabled:
                        await self._attempt_server_recovery(server_name, server)
                else:
                    metrics.status = ServerHealthStatus.UNHEALTHY

            # Uptime-Prozentsatz aktualisieren (vereinfacht)
            if metrics.consecutive_failures == 0:
                metrics.uptime_percentage = min(100.0, metrics.uptime_percentage + 0.1)
            else:
                metrics.uptime_percentage = max(0.0, metrics.uptime_percentage - 1.0)

        except Exception as e:
            logger.exception(f"Health-Check für Server {server_name} fehlgeschlagen: {e}")

            if server_name in self._server_health_metrics:
                metrics = self._server_health_metrics[server_name]
                metrics.status = ServerHealthStatus.FAILED
                metrics.last_error = str(e)
                metrics.consecutive_failures += 1

    async def _attempt_server_recovery(self, server_name: str, server: RegisteredMCPServer) -> None:
        """Versucht Server-Recovery."""
        logger.info(f"Versuche Recovery für Server {server_name}")

        try:
            # Vereinfachter Recovery-Versuch
            # In echter Implementierung könnte hier Server-Neustart, Reconnect etc. stattfinden

            if server_name in self._server_health_metrics:
                self._server_health_metrics[server_name].status = ServerHealthStatus.RECOVERING

            # Simuliere Recovery-Versuch
            await asyncio.sleep(5)

            # Prüfe, ob Recovery erfolgreich war
            if server.is_healthy:
                logger.info(f"Server {server_name} erfolgreich recovered")
                if server_name in self._server_health_metrics:
                    self._server_health_metrics[server_name].status = ServerHealthStatus.HEALTHY
                    self._server_health_metrics[server_name].consecutive_failures = 0
            else:
                logger.warning(f"Recovery für Server {server_name} fehlgeschlagen")

        except Exception as e:
            logger.exception(f"Recovery-Versuch für Server {server_name} fehlgeschlagen: {e}")

    # Public Interface Methods

    def get_integration_status(self) -> IntegrationStatus:
        """Gibt aktuellen Integration-Status zurück."""
        return self._status

    def get_discovery_metrics(self) -> DiscoveryMetrics:
        """Gibt Discovery-Metriken zurück."""
        return self._discovery_metrics

    def get_server_health_metrics(self) -> dict[str, ServerHealthMetrics]:
        """Gibt Server-Health-Metriken zurück."""
        return self._server_health_metrics.copy()

    def get_agent_integrations(self) -> dict[str, Any]:
        """Gibt Agent-Integration-Mappings zurück."""
        return self._agent_integrations.copy()

    async def force_discovery(self) -> None:
        """Erzwingt sofortige Discovery."""
        if self._status == IntegrationStatus.ACTIVE:
            await self._perform_full_discovery()

    async def register_server_auto(self, server_config: dict[str, Any]) -> bool:
        """Registriert Server automatisch mit Discovery.

        Args:
            server_config: Server-Konfiguration

        Returns:
            True wenn erfolgreich registriert
        """
        if not self._registry:
            return False

        try:
            # Server in Registry registrieren
            success = await self._registry.register_server(server_config)

            if success:
                # Sofortige Discovery für neuen Server
                await self._perform_full_discovery()
                logger.info(f"Server automatisch registriert und entdeckt: {server_config.get('name')}")

            return success

        except Exception as e:
            logger.exception(f"Automatische Server-Registrierung fehlgeschlagen: {e}")
            return False

    def get_tools_for_capability(self, capability: str) -> list[DiscoveredTool]:
        """Gibt Tools für spezifische Capability zurück."""
        tools_by_capability = self._agent_integrations.get("tools_by_capability", {})
        return tools_by_capability.get(capability, [])

    def get_resources_for_type(self, resource_type: str) -> list[DiscoveredResource]:
        """Gibt Ressourcen für spezifischen Typ zurück."""
        resources_by_type = self._agent_integrations.get("resources_by_type", {})
        return resources_by_type.get(resource_type, [])

    def get_prompts_for_category(self, category: str) -> list[DiscoveredPrompt]:
        """Gibt Prompts für spezifische Kategorie zurück."""
        prompts_by_category = self._agent_integrations.get("prompts_by_category", {})
        return prompts_by_category.get(category, [])


# Globale MCP-Integration-Engine
mcp_integration_engine = MCPIntegrationEngine()
