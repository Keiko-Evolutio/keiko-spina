"""Registry für externe MCP Server mit automatischer Discovery und Health Monitoring.

Diese Registry verwaltet alle registrierten externen MCP Server, überwacht deren
Gesundheit und stellt eine einheitliche Schnittstelle für Tool-Discovery und
-Ausführung bereit.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .kei_mcp_client import ExternalMCPConfig, KEIMCPClient, MCPToolDefinition, MCPToolResult
from .schema_validator import ValidationResult, schema_validator

logger = get_logger(__name__)


@dataclass
class MCPResourceDefinition:
    """Definition einer MCP Resource."""

    id: str
    name: str
    type: str
    description: str
    size_bytes: int | None = None
    last_modified: str | None = None
    etag: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class MCPPromptDefinition:
    """Definition eines MCP Prompt-Templates."""

    name: str
    description: str
    version: str
    parameters: dict[str, Any] | None = None
    tags: list[str] | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class MCPResourceResult:
    """Ergebnis eines Resource-Abrufs."""

    success: bool
    content: bytes | None = None
    content_type: str | None = None
    content_length: int | None = None
    etag: str | None = None
    last_modified: str | None = None
    status_code: int = 200
    headers: dict[str, str] | None = None
    error: str | None = None


@dataclass
class RegisteredMCPServer:
    """Registrierter externer MCP Server.

    Attributes:
        config: Konfiguration des Servers
        client: HTTP-Client für Kommunikation
        available_tools: Liste verfügbarer Tools
        is_healthy: Aktueller Gesundheitsstatus
        last_health_check: Zeitstempel des letzten Health Checks
        registration_time: Zeitstempel der Registrierung
        total_requests: Gesamtanzahl der Requests
        failed_requests: Anzahl fehlgeschlagener Requests
        avg_response_time_ms: Durchschnittliche Antwortzeit
        domain_validated: bool = False
        domain_validation_time: Optional[float] = None
        last_domain_revalidation: Optional[float] = None
    """

    config: ExternalMCPConfig
    client: KEIMCPClient
    available_tools: list[MCPToolDefinition] = field(default_factory=list)
    available_resources: list[MCPResourceDefinition] = field(default_factory=list)
    available_prompts: list[MCPPromptDefinition] = field(default_factory=list)
    is_healthy: bool = False
    last_health_check: float | None = None
    registration_time: float = field(default_factory=time.time)
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    # Domain-Validierung-Status (nur bei Registrierung geprüft)
    domain_validated: bool = False
    domain_validation_time: float | None = None
    last_domain_revalidation: float | None = None

    def update_stats(self, success: bool, response_time_ms: float):
        """Aktualisiert Server-Statistiken."""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1

        # Gleitender Durchschnitt der Antwortzeit
        if self.total_requests == 1:
            self.avg_response_time_ms = response_time_ms
        else:
            alpha = 0.1  # Gewichtung für neue Werte
            self.avg_response_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * self.avg_response_time_ms
            )

    @property
    def error_rate(self) -> float:
        """Berechnet die Fehlerrate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def uptime_seconds(self) -> float:
        """Berechnet die Uptime seit Registrierung."""
        return time.time() - self.registration_time


class KEIMCPRegistry:
    """Registry für externe MCP Server.

    Diese Klasse verwaltet alle registrierten externen MCP Server und bietet
    Funktionen für Discovery, Health Monitoring und Tool-Ausführung.
    """

    def __init__(self, health_check_interval: float = 60.0):
        """Initialisiert die Registry.

        Args:
            health_check_interval: Intervall für Health Checks in Sekunden
        """
        self._servers: dict[str, RegisteredMCPServer] = {}
        self._health_check_interval = health_check_interval
        self._health_check_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    async def register_server(self, config: ExternalMCPConfig, domain_validated: bool = False) -> bool:
        """Registriert einen externen MCP Server.

        Args:
            config: Konfiguration des zu registrierenden Servers
            domain_validated: Ob die Domain bereits validiert wurde (von Registrierungs-Endpoint)

        Returns:
            True bei erfolgreicher Registrierung, False sonst
        """
        try:
            # Prüfe ob Server bereits registriert
            if config.server_name in self._servers:
                logger.warning(f"MCP Server {config.server_name} ist bereits registriert")
                return False

            client = KEIMCPClient(config)

            # Initiale Verbindung testen
            async with client:
                is_healthy = await client.health_check()
                if not is_healthy:
                    logger.warning(f"MCP Server {config.server_name} ist nicht erreichbar")
                    return False

                # Tools entdecken
                tools = await client.discover_tools()
                logger.info(f"MCP Server {config.server_name} registriert mit {len(tools)} Tools")

                # Tool-Schemas cachen für Validierung
                await KEIMCPRegistry._cache_tool_schemas(config.server_name, tools)

                # Server-Info abrufen
                server_info = await client.get_server_info()
                logger.debug(f"Server-Info für {config.server_name}: {server_info}")

            # In Registry aufnehmen
            current_time = time.time()
            self._servers[config.server_name] = RegisteredMCPServer(
                config=config,
                client=KEIMCPClient(config),  # Neue Instanz für Registry
                available_tools=tools,
                is_healthy=is_healthy,
                last_health_check=current_time,
                domain_validated=domain_validated,
                domain_validation_time=current_time if domain_validated else None
            )

            # Health Check Task starten falls noch nicht aktiv
            if self._health_check_task is None or self._health_check_task.done():
                self._health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info(f"MCP Server {config.server_name} erfolgreich registriert")
            return True

        except Exception as exc:
            logger.exception(f"Registrierung von MCP Server {config.server_name} fehlgeschlagen: {exc}")
            return False

    async def unregister_server(self, server_name: str) -> bool:
        """Entfernt einen MCP Server aus der Registry.

        Args:
            server_name: Name des zu entfernenden Servers

        Returns:
            True bei erfolgreicher Entfernung, False sonst
        """
        if server_name in self._servers:
            server = self._servers[server_name]

            # Client schließen
            try:
                await server.client.aclose()
            except Exception as exc:
                logger.warning(f"Fehler beim Schließen des Clients für {server_name}: {exc}")

            del self._servers[server_name]
            logger.info(f"MCP Server {server_name} entfernt")
            return True

        logger.warning(f"MCP Server {server_name} nicht gefunden")
        return False

    def is_server_domain_validated(self, server_name: str) -> bool:
        """Prüft ob ein Server domain-validiert ist.

        Args:
            server_name: Name des Servers

        Returns:
            True wenn Server domain-validiert ist, False sonst
        """
        if server_name not in self._servers:
            return False

        return self._servers[server_name].domain_validated

    def mark_server_domain_validated(self, server_name: str) -> bool:
        """Markiert einen Server als domain-validiert.

        Args:
            server_name: Name des Servers

        Returns:
            True wenn erfolgreich markiert, False wenn Server nicht existiert
        """
        if server_name not in self._servers:
            return False

        server = self._servers[server_name]
        server.domain_validated = True
        server.domain_validation_time = time.time()

        logger.info(f"Server {server_name} als domain-validiert markiert")
        return True

    def get_domain_validation_status(self, server_name: str) -> dict[str, Any]:
        """Gibt Domain-Validierung-Status für einen Server zurück.

        Args:
            server_name: Name des Servers

        Returns:
            Dictionary mit Validierung-Status-Informationen
        """
        if server_name not in self._servers:
            return {
                "exists": False,
                "domain_validated": False,
                "validation_time": None,
                "last_revalidation": None
            }

        server = self._servers[server_name]
        return {
            "exists": True,
            "domain_validated": server.domain_validated,
            "validation_time": server.domain_validation_time,
            "last_revalidation": server.last_domain_revalidation,
            "server_url": server.config.base_url
        }

    async def revalidate_domains_if_needed(self, revalidation_interval_hours: int = 24) -> dict[str, bool]:
        """Führt periodische Domain-Revalidierung durch falls erforderlich.

        Args:
            revalidation_interval_hours: Intervall für Revalidierung in Stunden

        Returns:
            Dictionary mit Revalidierung-Ergebnissen pro Server
        """
        from security.kei_mcp_auth import (
            validate_server_domain_for_registration as validate_server_domain,
        )

        current_time = time.time()
        revalidation_interval_seconds = revalidation_interval_hours * 3600
        results = {}

        for server_name, server in self._servers.items():
            # Prüfe ob Revalidierung erforderlich ist
            needs_revalidation = False

            if not server.domain_validated:
                # Server war nie validiert - überspringe (sollte bei Registrierung passieren)
                results[server_name] = False
                continue

            if server.last_domain_revalidation is None:
                # Erste Revalidierung seit Registrierung
                if server.domain_validation_time and (current_time - server.domain_validation_time) > revalidation_interval_seconds:
                    needs_revalidation = True
            # Prüfe ob Revalidierung-Intervall erreicht ist
            elif (current_time - server.last_domain_revalidation) > revalidation_interval_seconds:
                needs_revalidation = True

            if needs_revalidation:
                try:
                    # Führe Domain-Revalidierung durch
                    is_valid = await validate_server_domain(server.config.base_url)

                    if is_valid:
                        server.last_domain_revalidation = current_time
                        results[server_name] = True
                        logger.info(f"Domain-Revalidierung für {server_name} erfolgreich")
                    else:
                        # Domain nicht mehr in Whitelist - markiere als nicht validiert
                        server.domain_validated = False
                        server.last_domain_revalidation = current_time
                        results[server_name] = False
                        logger.warning(f"Domain-Revalidierung für {server_name} fehlgeschlagen - Server als nicht validiert markiert")

                        # Optional: Server aus Registry entfernen
                        # await self.unregister_server(server_name)

                except Exception as e:
                    logger.exception(f"Fehler bei Domain-Revalidierung für {server_name}: {e}")
                    results[server_name] = False
            else:
                results[server_name] = True  # Keine Revalidierung erforderlich

        return results

    def get_available_servers(self) -> list[str]:
        """Gibt Liste verfügbarer (gesunder) Server zurück."""
        return [name for name, server in self._servers.items() if server.is_healthy]

    def get_all_servers(self) -> list[str]:
        """Gibt Liste aller registrierten Server zurück."""
        return list(self._servers.keys())

    def get_server_stats(self, server_name: str) -> dict[str, Any] | None:
        """Gibt Statistiken für einen Server zurück."""
        if server_name not in self._servers:
            return None

        server = self._servers[server_name]
        return {
            "server_name": server_name,
            "is_healthy": server.is_healthy,
            "uptime_seconds": server.uptime_seconds,
            "total_requests": server.total_requests,
            "failed_requests": server.failed_requests,
            "error_rate": server.error_rate,
            "avg_response_time_ms": server.avg_response_time_ms,
            "available_tools": len(server.available_tools),
            "last_health_check": server.last_health_check
        }

    def get_all_tools(self) -> dict[str, list[dict[str, Any]]]:
        """Gibt alle verfügbaren Tools aller gesunden Server zurück."""
        result = {}
        for name, server in self._servers.items():
            if server.is_healthy:
                result[name] = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "required": tool.required,
                        "examples": tool.examples
                    }
                    for tool in server.available_tools
                ]
        return result

    def find_tools_by_name(self, tool_name: str) -> list[dict[str, Any]]:
        """Findet Tools nach Namen über alle Server."""
        matching_tools = []
        for server_name, server in self._servers.items():
            if not server.is_healthy:
                continue

            for tool in server.available_tools:
                if tool.name == tool_name:
                    matching_tools.append({
                        "server": server_name,
                        "tool": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                            "required": tool.required,
                            "examples": tool.examples
                        }
                    })
        return matching_tools

    @trace_function("external_mcp.invoke_tool_on_server")
    async def invoke_tool(self, server_name: str, tool_name: str, parameters: dict[str, Any]) -> MCPToolResult:
        """Ruft Tool auf spezifischem Server auf.

        Args:
            server_name: Name des Servers
            tool_name: Name des Tools
            parameters: Parameter für den Tool-Aufruf

        Returns:
            Ergebnis des Tool-Aufrufs
        """
        if server_name not in self._servers:
            return MCPToolResult(
                success=False,
                error=f"Server {server_name} nicht registriert"
            )

        server = self._servers[server_name]
        if not server.is_healthy:
            return MCPToolResult(
                success=False,
                error=f"Server {server_name} ist nicht verfügbar"
            )

        # Tool-Aufruf
        async with server.client:
            result = await server.client.invoke_tool(tool_name, parameters)

        # Statistiken aktualisieren
        if result.execution_time_ms is not None:
            server.update_stats(result.success, result.execution_time_ms)

        return result

    async def shutdown(self):
        """Fährt die Registry herunter und schließt alle Verbindungen."""
        logger.info("Fahre External MCP Registry herunter...")

        # Health Check Task stoppen
        self._shutdown_event.set()
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        # Alle Server-Clients schließen
        for server_name, server in self._servers.items():
            try:
                await server.client.aclose()
                logger.debug(f"Client für {server_name} geschlossen")
            except Exception as exc:
                logger.warning(f"Fehler beim Schließen des Clients für {server_name}: {exc}")

        self._servers.clear()
        logger.info("External MCP Registry heruntergefahren")

    async def _health_check_loop(self):
        """Kontinuierliche Health Checks für alle Server."""
        logger.info("Health Check Loop gestartet")

        while not self._shutdown_event.is_set():
            try:
                # Health Checks für alle Server
                for server_name, server in list(self._servers.items()):
                    current_time = time.time()
                    old_status = server.is_healthy

                    try:
                        # Health Check durchführen
                        async with server.client:
                            is_healthy = await server.client.health_check()

                        # Status-Änderung verarbeiten
                        if old_status != is_healthy:
                            status_text = "healthy" if is_healthy else "unhealthy"
                            logger.info(f"MCP Server {server_name} Status-Änderung: {old_status} -> {status_text}")

                            # Tools neu entdecken NUR wenn Server von unhealthy zu healthy wechselt
                            if is_healthy and not old_status:
                                await self._rediscover_tools_safely(server_name, server)

                        # Status und Zeitstempel aktualisieren
                        server.is_healthy = is_healthy
                        server.last_health_check = current_time

                        # Erfolgreichen Health Check loggen (nur bei Debug-Level)
                        if is_healthy:
                            logger.debug(f"Health Check erfolgreich für {server_name}")

                    except Exception as exc:
                        # Health Check fehlgeschlagen
                        logger.warning(f"Health Check für {server_name} fehlgeschlagen: {exc}")

                        # Status nur ändern wenn vorher healthy
                        if old_status:
                            logger.exception(f"MCP Server {server_name} Status-Änderung: healthy -> unhealthy")

                        server.is_healthy = False
                        server.last_health_check = current_time

                # Warten bis zum nächsten Check oder Shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self._health_check_interval
                    )
                    break  # Shutdown angefordert
                except TimeoutError:
                    continue  # Normaler Health Check Zyklus

            except Exception as exc:
                logger.exception(f"Health Check Loop Fehler: {exc}")
                await asyncio.sleep(self._health_check_interval)

        logger.info("Health Check Loop beendet")

    @staticmethod
    async def _cache_tool_schemas(server_name: str, tools: list[MCPToolDefinition]):
        """Cached JSON-Schemas für Tools eines Servers.

        Args:
            server_name: Name des MCP Servers
            tools: Liste der Tool-Definitionen
        """
        try:
            cached_count = 0

            for tool in tools:
                if tool.parameters:
                    # Schema in Validator cachen
                    success = schema_validator.cache_schema(
                        server_name=server_name,
                        tool_name=tool.name,
                        schema=tool.parameters
                    )

                    if success:
                        cached_count += 1
                    else:
                        logger.warning(f"Schema-Caching fehlgeschlagen für {server_name}:{tool.name}")
                else:
                    logger.debug(f"Kein Schema verfügbar für Tool {server_name}:{tool.name}")

            logger.info(f"Tool-Schemas gecacht für {server_name}: {cached_count}/{len(tools)} erfolgreich")

        except Exception as e:
            logger.exception(f"Fehler beim Cachen der Tool-Schemas für {server_name}: {e}")

    @trace_function("registry.validate_tool_parameters")
    async def validate_tool_parameters(
        self,
        server_name: str,
        tool_name: str,
        parameters: dict[str, Any]
    ) -> ValidationResult:
        """Validiert Tool-Parameter gegen gecachtes JSON-Schema.

        Args:
            server_name: Name des MCP Servers
            tool_name: Name des Tools
            parameters: Zu validierende Parameter

        Returns:
            Validierungsergebnis
        """
        try:
            # Prüfe ob Server registriert ist
            if server_name not in self._servers:
                return ValidationResult(
                    valid=False,
                    errors=[f"Server '{server_name}' ist nicht registriert"]
                )

            # Prüfe ob Tool existiert
            server = self._servers[server_name]
            tool_exists = any(tool.name == tool_name for tool in server.available_tools)

            if not tool_exists:
                return ValidationResult(
                    valid=False,
                    errors=[f"Tool '{tool_name}' existiert nicht auf Server '{server_name}'"]
                )

            # Parameter validieren
            return schema_validator.validate_parameters(
                server_name=server_name,
                tool_name=tool_name,
                parameters=parameters
            )

        except Exception as e:
            logger.exception(f"Fehler bei Tool-Parameter-Validierung für {server_name}:{tool_name}: {e}")
            return ValidationResult(
                valid=False,
                errors=[f"Validierungsfehler: {e!s}"]
            )

    async def _rediscover_tools_safely(self, server_name: str, server: RegisteredMCPServer):
        """Entdeckt Tools sicher neu wenn Server wieder healthy wird.

        Args:
            server_name: Name des Servers
            server: Server-Instanz
        """
        try:
            logger.info(f"Starte Tool-Re-Discovery für {server_name}")

            # Separate Client-Instanz für Tool-Discovery verwenden
            async with server.client:
                new_tools = await server.client.discover_tools()

                # Tools nur aktualisieren wenn Discovery erfolgreich
                if new_tools is not None:
                    old_tool_count = len(server.available_tools)
                    server.available_tools = new_tools

                    # Tool-Schemas neu cachen
                    await KEIMCPRegistry._cache_tool_schemas(server_name, new_tools)
                    new_tool_count = len(new_tools)

                    logger.info(f"Tool-Re-Discovery für {server_name} erfolgreich: "
                               f"{old_tool_count} -> {new_tool_count} Tools")

                    # Tool-Namen für Debug-Logging
                    tool_names = [tool.name for tool in new_tools]
                    logger.debug(f"Verfügbare Tools für {server_name}: {tool_names}")
                else:
                    logger.warning(f"Tool-Discovery für {server_name} gab None zurück")

        except Exception as tool_exc:
            logger.exception(f"Tool-Re-Discovery für {server_name} fehlgeschlagen: {tool_exc}")

            # Server bleibt healthy, aber Tools werden nicht aktualisiert
            # Das ist besser als den Server als unhealthy zu markieren
            logger.info(f"Server {server_name} bleibt healthy trotz Tool-Discovery-Fehler")

    # ========================================================================
    # RESOURCE METHODS
    # ========================================================================

    async def get_all_resources(self) -> dict[str, list[dict[str, Any]]]:
        """Gibt alle verfügbaren Resources zurück.

        Returns:
            Dictionary mit Server-Namen als Keys und Resource-Listen als Values
        """
        resources_by_server = {}

        for server_name, server in self._servers.items():
            if server.is_healthy:
                # Resources in Dictionary-Format konvertieren
                resources = []
                for resource in server.available_resources:
                    resources.append({
                        "id": resource.id,
                        "name": resource.name,
                        "type": resource.type,
                        "description": resource.description,
                        "size_bytes": resource.size_bytes,
                        "last_modified": resource.last_modified,
                        "etag": resource.etag,
                        "metadata": resource.metadata
                    })
                resources_by_server[server_name] = resources
            else:
                resources_by_server[server_name] = []

        return resources_by_server

    @trace_function("external_mcp.get_resource")
    async def get_resource(
        self,
        server_name: str,
        resource_id: str,
        if_none_match: str | None = None,
        range_header: str | None = None
    ) -> MCPResourceResult:
        """Ruft eine spezifische Resource ab.

        Args:
            server_name: Name des Servers
            resource_id: ID der Resource
            if_none_match: ETag für Conditional Requests
            range_header: Range-Header für partielle Inhalte

        Returns:
            Ergebnis des Resource-Abrufs
        """
        if server_name not in self._servers:
            return MCPResourceResult(
                success=False,
                error=f"Server {server_name} nicht registriert"
            )

        server = self._servers[server_name]
        if not server.is_healthy:
            return MCPResourceResult(
                success=False,
                error=f"Server {server_name} ist nicht verfügbar"
            )

        # Resource-Abruf über Client
        async with server.client:
            try:
                # Hier würde der tatsächliche HTTP-Request an den MCP Server gehen
                # Für jetzt simulieren wir eine erfolgreiche Antwort
                return await server.client.get_resource(
                    resource_id,
                    if_none_match=if_none_match,
                    range_header=range_header
                )
            except Exception as exc:
                logger.exception(f"Resource-Abruf fehlgeschlagen: {server_name}:{resource_id} - {exc}")
                return MCPResourceResult(
                    success=False,
                    error=str(exc)
                )

    # ========================================================================
    # PROMPT METHODS
    # ========================================================================

    async def get_all_prompts(self) -> dict[str, list[dict[str, Any]]]:
        """Gibt alle verfügbaren Prompts zurück.

        Returns:
            Dictionary mit Server-Namen als Keys und Prompt-Listen als Values
        """
        prompts_by_server = {}

        for server_name, server in self._servers.items():
            if server.is_healthy:
                # Prompts in Dictionary-Format konvertieren
                prompts = []
                for prompt in server.available_prompts:
                    prompts.append({
                        "name": prompt.name,
                        "description": prompt.description,
                        "version": prompt.version,
                        "parameters": prompt.parameters,
                        "tags": prompt.tags,
                        "created_at": prompt.created_at,
                        "updated_at": prompt.updated_at
                    })
                prompts_by_server[server_name] = prompts
            else:
                prompts_by_server[server_name] = []

        return prompts_by_server

    @trace_function("external_mcp.get_prompt")
    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        version: str | None = None
    ) -> dict[str, Any]:
        """Ruft ein spezifisches Prompt-Template ab.

        Args:
            server_name: Name des Servers
            prompt_name: Name des Prompts
            version: Spezifische Version (optional)

        Returns:
            Prompt-Template oder Fehler-Dictionary
        """
        if server_name not in self._servers:
            return {
                "success": False,
                "error": f"Server {server_name} nicht registriert"
            }

        server = self._servers[server_name]
        if not server.is_healthy:
            return {
                "success": False,
                "error": f"Server {server_name} ist nicht verfügbar"
            }

        # Prompt-Abruf über Client
        async with server.client:
            try:
                # Hier würde der tatsächliche HTTP-Request an den MCP Server gehen
                return await server.client.get_prompt(prompt_name, version=version)
            except Exception as exc:
                logger.exception(f"Prompt-Abruf fehlgeschlagen: {server_name}:{prompt_name} - {exc}")
                return {
                    "success": False,
                    "error": str(exc)
                }


# Globale Registry-Instanz
kei_mcp_registry = KEIMCPRegistry()


__all__ = [
    "KEIMCPRegistry",
    "RegisteredMCPServer",
    "kei_mcp_registry"
]
