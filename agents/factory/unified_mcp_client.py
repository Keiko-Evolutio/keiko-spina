# backend/agents/factory/unified_mcp_client.py
"""Konsolidierter MCP Client für das Factory-Modul.

Merge von ProductionMCPClient und KEIMCPclient zu einer einzigen, robusten
Implementierung mit Enterprise-Grade Features und Clean Code Prinzipien.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from agents.protocols.core import MCPCapabilityType, MCPTransportType
from kei_logging import get_logger

from .constants import (
    CONNECTION_TIMEOUT,
    DEFAULT_MCP_ENDPOINT,
    DEFAULT_TIMEOUT,
    ERROR_SESSION_NOT_FOUND,
    READ_TIMEOUT,
    ClientState,
    LogLevel,
)
from .error_handlers import (
    ErrorHandler,
    MCPClientError,
    SessionError,
    error_context,
    retry_on_error,
)
from .metrics_manager import MetricsManager
from .singleton_mixin import SingletonMixin

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)


# =============================================================================
# Session Configuration und State Management
# =============================================================================

@dataclass
class MCPSessionConfig:
    """Konfiguration für MCP-Sessions."""
    session_id: str
    server_endpoint: str
    transport_type: MCPTransportType
    auth_config: dict[str, Any] | None
    created_at: datetime
    timeout: float = DEFAULT_TIMEOUT

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Konfiguration zu Dictionary."""
        return {
            "session_id": self.session_id,
            "server_endpoint": self.server_endpoint,
            "transport_type": self.transport_type.value,
            "auth_config": self.auth_config,
            "created_at": self.created_at.isoformat(),
            "timeout": self.timeout,
        }


class MCPTransportHandler:
    """Handler für MCP-Transport-Operationen mit verbesserter Fehlerbehandlung."""

    def __init__(self) -> None:
        self._connections: dict[str, httpx.AsyncClient] = {}
        self._error_handler = ErrorHandler("MCPTransportHandler")
        self._metrics = MetricsManager()

    async def establish_connection(self, config: MCPSessionConfig) -> bool:
        """Stellt eine Verbindung basierend auf der Konfiguration her."""
        try:
            async with error_context("establish_connection"):
                # HTTP Client für MCP-Verbindung erstellen
                client = httpx.AsyncClient(
                    base_url=config.server_endpoint,
                    timeout=httpx.Timeout(
                        connect=CONNECTION_TIMEOUT,
                        read=READ_TIMEOUT,
                        write=config.timeout,
                        pool=config.timeout
                    ),
                    headers=self._build_headers(config.auth_config)
                )

                # Verbindungstest durchführen
                start_time = datetime.now(UTC)
                response = await client.get("/health", timeout=CONNECTION_TIMEOUT)
                latency = (datetime.now(UTC) - start_time).total_seconds()

                if response.status_code == 200:
                    self._connections[config.session_id] = client
                    self._metrics.record_request(
                        config.session_id,
                        latency,
                        success=True,
                        metadata={"operation": "establish_connection"}
                    )
                    return True
                await client.aclose()
                return False

        except Exception as e:
            self._error_handler.handle_error(
                e,
                "establish_connection",
                additional_details={
                    "session_id": config.session_id,
                    "endpoint": config.server_endpoint
                }
            )
            return False

    async def close_connection(self, session_id: str) -> None:
        """Schließt eine Verbindung für eine Session."""
        if session_id in self._connections:
            client = self._connections.pop(session_id)
            try:
                await client.aclose()
                logger.debug(f"MCP-Verbindung geschlossen: {session_id}")
            except Exception as e:
                self._error_handler.handle_error(
                    e,
                    "close_connection",
                    suppress=True,
                    additional_details={"session_id": session_id}
                )

    @staticmethod
    def _build_headers(auth_config: dict[str, Any] | None) -> dict[str, str]:
        """Erstellt HTTP-Headers basierend auf Auth-Konfiguration.

        Args:
            auth_config: Authentifizierungs-Konfiguration

        Returns:
            HTTP-Headers für MCP-Client
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Keiko-MCP-Client/1.0"
        }

        if auth_config:
            if "bearer_token" in auth_config:
                headers["Authorization"] = f"Bearer {auth_config['bearer_token']}"
            elif "api_key" in auth_config:
                headers["X-API-Key"] = auth_config["api_key"]

        return headers

    async def send_request(
        self,
        session_id: str,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None
    ) -> Any:
        """Sendet HTTP-Request über eine bestehende Verbindung.

        Args:
            session_id: Session-ID für die Verbindung
            method: HTTP-Methode (GET, POST, etc.)
            endpoint: Endpoint-Pfad
            data: Request-Body-Daten
            params: Query-Parameter

        Returns:
            Response-Daten

        Raises:
            SessionError: Wenn Session nicht gefunden wird
            MCPClientError: Bei Request-Fehlern
        """
        if session_id not in self._connections:
            raise SessionError(
                f"Keine Verbindung für Session {session_id} gefunden",
                details={"session_id": session_id}
            )

        client = self._connections[session_id]

        try:
            # Request durchführen
            response = await client.request(
                method=method,
                url=endpoint,
                json=data,
                params=params
            )

            response.raise_for_status()

            # JSON-Response parsen falls möglich
            try:
                return response.json()
            except Exception:
                return response.text

        except httpx.HTTPStatusError as e:
            raise MCPClientError(
                f"HTTP-Fehler bei {method} {endpoint}: {e.response.status_code}",
                details={
                    "session_id": session_id,
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": e.response.status_code
                }
            )
        except Exception as e:
            raise MCPClientError(
                f"Request-Fehler bei {method} {endpoint}: {e}",
                details={
                    "session_id": session_id,
                    "method": method,
                    "endpoint": endpoint
                },
                original_error=e
            )


class UnifiedMCPClient:
    """Konsolidierter MCP Client mit Enterprise-Features.

    Kombiniert die Funktionalität von ProductionMCPClient und KEIMCPclient
    zu einer einzigen, robusten Implementierung.
    """

    def __init__(self, client_id: str | None = None) -> None:
        """Initialisiert den Unified MCP Client."""
        self.client_id = client_id or str(uuid.uuid4())
        self._sessions: dict[str, MCPSessionConfig] = {}
        self._session_states: dict[str, ClientState] = {}
        self._capability_cache: dict[str, list[MCPCapabilityType]] = {}
        self._transport_handler = MCPTransportHandler()
        self._error_handler = ErrorHandler(f"UnifiedMCPClient-{self.client_id}")
        self._metrics = MetricsManager()

        # Client bei Metrics registrieren
        self._metrics.register_component(self.client_id, "mcp_client")

        logger.debug(
            f"UnifiedMCPClient initialisiert: {self.client_id}",
            extra={
                "client_id": self.client_id,
                "log_level": LogLevel.DEBUG
            }
        )

    async def __aenter__(self) -> UnifiedMCPClient:
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit mit automatischem Cleanup."""
        await self._cleanup_all_resources()

    @retry_on_error(
        max_attempts=3,
        exceptions=(ConnectionError, httpx.RequestError),
        context="initialize_session"
    )
    async def initialize_session(
        self,
        server_endpoint: str,
        transport_type: MCPTransportType = MCPTransportType.STREAMABLE_HTTP,
        auth_config: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT
    ) -> str:
        """Initialisiert eine neue MCP-Session.

        Args:
            server_endpoint: Server-Endpunkt URL
            transport_type: Transport-Typ
            auth_config: Authentifizierungs-Konfiguration
            timeout: Timeout für Operationen

        Returns:
            Session-ID

        Raises:
            MCPClientError: Bei Initialisierungsfehlern
        """
        session_config = MCPSessionConfig(
            session_id=str(uuid.uuid4()),
            server_endpoint=server_endpoint,
            transport_type=transport_type,
            auth_config=auth_config,
            created_at=datetime.now(UTC),
            timeout=timeout
        )

        try:
            # Verbindung herstellen
            if not await self._transport_handler.establish_connection(session_config):
                raise MCPClientError(
                    "Verbindung zu MCP-Server fehlgeschlagen",
                    details={
                        "server_endpoint": server_endpoint,
                        "transport_type": transport_type.value
                    }
                )

            # Session speichern
            self._sessions[session_config.session_id] = session_config
            self._session_states[session_config.session_id] = ClientState.CONNECTED

            logger.info(
                f"MCP-Session initialisiert: {session_config.session_id}",
                extra={
                    "session_id": session_config.session_id,
                    "server_endpoint": server_endpoint,
                    "log_level": LogLevel.INFO
                }
            )

            return session_config.session_id

        except Exception as e:
            error = self._error_handler.handle_error(
                e,
                "initialize_session",
                additional_details={
                    "server_endpoint": server_endpoint,
                    "transport_type": transport_type.value
                }
            )
            raise MCPClientError(
                f"Session-Initialisierung fehlgeschlagen: {error.error_message}",
                original_error=e
            )

    async def cleanup_session(self, session_id: str) -> None:
        """Bereinigt eine spezifische Session."""
        if session_id in self._sessions:
            # Transport-Verbindung schließen
            await self._transport_handler.close_connection(session_id)

            # Session-Daten entfernen
            self._sessions.pop(session_id, None)
            self._session_states.pop(session_id, None)
            self._capability_cache.pop(session_id, None)

            logger.debug(
                f"MCP-Session bereinigt: {session_id}",
                extra={
                    "session_id": session_id,
                    "log_level": LogLevel.DEBUG
                }
            )

    async def get_session_state(self, session_id: str) -> ClientState:
        """Gibt den Status einer Session zurück."""
        return self._session_states.get(session_id, ClientState.DISCONNECTED)

    def get_sessions_info(self) -> dict[str, Any]:
        """Public API für Session-Informationen.

        Returns:
            Dictionary mit Session-Informationen
        """
        return {
            "active_sessions": len(self._sessions),
            "session_ids": list(self._sessions.keys())
        }

    def get_session_states_info(self) -> dict[str, str]:
        """Public API für Session-States.

        Returns:
            Dictionary mit Session-States
        """
        return {
            sid: state.value for sid, state in self._session_states.items()
        }

    async def invoke_tool(
        self,
        session_id: str,
        tool_name: str,
        parameters: dict[str, Any]
    ) -> Any:
        """Führt ein Tool über eine MCP-Session aus."""
        if session_id not in self._sessions:
            raise SessionError(
                ERROR_SESSION_NOT_FOUND,
                details={"session_id": session_id}
            )

        start_time = datetime.now(UTC)

        try:
            # Tool-Aufruf über Transport-Handler
            result = await self._transport_handler.send_request(
                session_id,
                "POST",
                f"/tools/{tool_name}/invoke",
                data=parameters
            )

            latency = (datetime.now(UTC) - start_time).total_seconds()
            self._metrics.record_request(
                session_id,
                latency,
                success=True,
                metadata={
                    "operation": "invoke_tool",
                    "tool_name": tool_name
                }
            )

            return result

        except Exception as e:
            latency = (datetime.now(UTC) - start_time).total_seconds()
            self._metrics.record_request(
                session_id,
                latency,
                success=False,
                metadata={
                    "operation": "invoke_tool",
                    "tool_name": tool_name,
                    "error": str(e)
                }
            )
            raise

    async def discover_capabilities(self, session_id: str) -> list[MCPCapabilityType]:
        """Entdeckt Capabilities für eine Session."""
        if session_id not in self._sessions:
            raise SessionError(
                ERROR_SESSION_NOT_FOUND,
                details={"session_id": session_id}
            )

        # Cache prüfen
        if session_id in self._capability_cache:
            return self._capability_cache[session_id]

        try:
            # Capabilities vom Server abfragen
            response = await self._transport_handler.send_request(
                session_id,
                "GET",
                "/capabilities"
            )

            capabilities = [
                MCPCapabilityType(cap) for cap in response.get("capabilities", [])
            ]

            # Cache aktualisieren
            self._capability_cache[session_id] = capabilities

            return capabilities

        except Exception as e:
            self._error_handler.handle_error(
                e,
                "discover_capabilities",
                additional_details={"session_id": session_id}
            )
            # Fallback auf Standard-Capabilities
            return [MCPCapabilityType.TOOL_INVOCATION]

    async def _cleanup_all_resources(self) -> None:
        """Bereinigt alle Ressourcen des Clients."""
        session_ids = list(self._sessions.keys())

        for session_id in session_ids:
            try:
                await self.cleanup_session(session_id)
            except Exception as e:
                self._error_handler.handle_error(
                    e,
                    "cleanup_session",
                    suppress=True,
                    additional_details={"session_id": session_id}
                )

        # Metrics-Komponente bereinigen
        self._metrics.cleanup_component(self.client_id)

        logger.debug(
            f"UnifiedMCPClient Ressourcen bereinigt: {self.client_id}",
            extra={
                "client_id": self.client_id,
                "cleaned_sessions": len(session_ids),
                "log_level": LogLevel.DEBUG
            }
        )


# =============================================================================
# Unified MCP Client Factory
# =============================================================================

class UnifiedMCPClientFactory(SingletonMixin):
    """Konsolidierte Factory für MCP Client-Erstellung.

    Ersetzt MCPClientFactory und bietet einheitliche Client-Verwaltung
    mit verbessertem Resource Management und Monitoring.
    """

    def _initialize_singleton(self, *args, **kwargs) -> None:
        """Initialisiert die MCP Client Factory."""
        self._client_instances: dict[str, UnifiedMCPClient] = {}
        self._server_registry: dict[str, dict[str, Any]] = {}
        self._error_handler = ErrorHandler("UnifiedMCPClientFactory")
        self._metrics = MetricsManager()

        # Factory bei Metrics registrieren
        self._metrics.register_component("mcp_client_factory", "factory")

        logger.info(
            "UnifiedMCPClientFactory initialisiert",
            extra={
                "component": "UnifiedMCPClientFactory",
                "log_level": LogLevel.INFO
            }
        )

    async def create_client(
        self,
        agent_id: str | None = None,
        agent_metadata: dict[str, Any] | None = None
    ) -> UnifiedMCPClient:
        """Erstellt einen MCP Client (Backward-Compatibility-Methode).

        Args:
            agent_id: Optionale Agent-ID (wird generiert falls nicht angegeben)
            agent_metadata: Optionale Agent-Metadaten

        Returns:
            UnifiedMCPClient-Instanz
        """
        if agent_id is None:
            agent_id = str(uuid.uuid4())

        return await self.create_client_for_agent(agent_id, agent_metadata)

    async def create_client_for_agent(
        self,
        agent_id: str,
        agent_metadata: dict[str, Any] | None = None
    ) -> UnifiedMCPClient:
        """Erstellt einen MCP Client für einen Agent.

        Args:
            agent_id: Eindeutige Agent-ID
            agent_metadata: Optionale Agent-Metadaten

        Returns:
            UnifiedMCPClient-Instanz
        """
        # Bestehenden Client zurückgeben falls vorhanden
        if agent_id in self._client_instances:
            logger.debug(
                f"Bestehender MCP Client zurückgegeben: {agent_id}",
                extra={
                    "agent_id": agent_id,
                    "log_level": LogLevel.DEBUG
                }
            )
            return self._client_instances[agent_id]

        try:
            # Neuen Client erstellen
            client = UnifiedMCPClient(client_id=f"agent-{agent_id}")

            # Context Manager aktivieren
            await client.__aenter__()

            # Server-Discovery durchführen
            await self._discover_servers_for_agent(client, agent_id, agent_metadata)

            # Client registrieren
            self._client_instances[agent_id] = client

            logger.info(
                f"MCP Client für Agent erstellt: {agent_id}",
                extra={
                    "agent_id": agent_id,
                    "client_id": client.client_id,
                    "log_level": LogLevel.INFO
                }
            )

            return client

        except Exception as e:
            error = self._error_handler.handle_error(
                e,
                "create_client_for_agent",
                additional_details={
                    "agent_id": agent_id,
                    "agent_metadata": agent_metadata
                }
            )
            raise MCPClientError(
                f"Client-Erstellung für Agent fehlgeschlagen: {error.error_message}",
                original_error=e
            )

    async def create_client_for_servers(
        self,
        agent_id: str,
        server_ids: list[str]
    ) -> UnifiedMCPClient:
        """Erstellt einen Client und initialisiert Sessions für Server.

        Args:
            agent_id: Agent-ID
            server_ids: Liste von Server-IDs

        Returns:
            UnifiedMCPClient mit initialisierten Sessions
        """
        try:
            client = UnifiedMCPClient(client_id=f"agent-{agent_id}")

            # Sessions für alle Server initialisieren
            for server_id in server_ids:
                server_info = self._server_registry.get(server_id)
                if server_info:
                    try:
                        await client.initialize_session(
                            server_endpoint=server_info.get("endpoint", DEFAULT_MCP_ENDPOINT),
                            auth_config=server_info.get("auth_config")
                        )
                    except Exception as e:
                        # Partielle Fehler tolerieren
                        self._error_handler.handle_error(
                            e,
                            "initialize_server_session",
                            suppress=True,
                            additional_details={
                                "server_id": server_id,
                                "agent_id": agent_id
                            }
                        )

            self._client_instances[agent_id] = client
            return client

        except Exception as e:
            error = self._error_handler.handle_error(
                e,
                "create_client_for_servers",
                additional_details={
                    "agent_id": agent_id,
                    "server_ids": server_ids
                }
            )
            raise MCPClientError(
                f"Client-Erstellung für Server fehlgeschlagen: {error.error_message}",
                original_error=e
            )

    async def cleanup_client(self, agent_id: str) -> bool:
        """Bereinigt einen spezifischen Client.

        Args:
            agent_id: Agent-ID des zu bereinigenden Clients

        Returns:
            True wenn Client bereinigt wurde, False wenn nicht gefunden
        """
        if agent_id not in self._client_instances:
            return False

        client = self._client_instances.pop(agent_id)

        try:
            await client.__aexit__(None, None, None)
            logger.debug(
                f"MCP Client bereinigt: {agent_id}",
                extra={
                    "agent_id": agent_id,
                    "log_level": LogLevel.DEBUG
                }
            )
            return True

        except Exception as e:
            self._error_handler.handle_error(
                e,
                "cleanup_client",
                suppress=True,
                additional_details={"agent_id": agent_id}
            )
            return False

    async def cleanup_all_clients(self) -> None:
        """Bereinigt alle Clients."""
        agent_ids = list(self._client_instances.keys())

        for agent_id in agent_ids:
            try:
                await self.cleanup_client(agent_id)
            except Exception as e:
                self._error_handler.handle_error(
                    e,
                    "cleanup_all_clients",
                    suppress=True,
                    additional_details={"agent_id": agent_id}
                )

        logger.info(
            f"Alle MCP Clients bereinigt: {len(agent_ids)} Clients",
            extra={
                "cleanup_count": len(agent_ids),
                "log_level": LogLevel.INFO
            }
        )

    async def register_server(
        self,
        server_id: str,
        server_config: dict[str, Any]
    ) -> None:
        """Registriert einen MCP Server.

        Args:
            server_id: Eindeutige Server-ID
            server_config: Server-Konfiguration
        """
        self._server_registry[server_id] = {
            "server_id": server_id,
            "endpoint": server_config.get("endpoint", DEFAULT_MCP_ENDPOINT),
            "auth_config": server_config.get("auth_config"),
            "capabilities": server_config.get("capabilities", []),
            "registered_at": datetime.now(UTC).isoformat(),
            **server_config
        }

        logger.debug(
            f"MCP Server registriert: {server_id}",
            extra={
                "server_id": server_id,
                "endpoint": server_config.get("endpoint"),
                "log_level": LogLevel.DEBUG
            }
        )

    def get_registered_servers(self) -> dict[str, dict[str, Any]]:
        """Gibt alle registrierten Server zurück."""
        return self._server_registry.copy()

    async def health_check(self) -> dict[str, Any]:
        """Führt einen Health Check der Factory durch."""
        active_clients = len(self._client_instances)
        registered_servers = len(self._server_registry)

        # Detaillierte Client-Statistiken
        client_stats = {}
        for agent_id, client in self._client_instances.items():
            sessions_info = client.get_sessions_info()
            session_states_info = client.get_session_states_info()
            client_stats[agent_id] = {
                "client_id": client.client_id,
                "active_sessions": sessions_info["active_sessions"],
                "session_states": session_states_info
            }

        return {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "active_clients": active_clients,
            "registered_servers": registered_servers,
            "client_details": client_stats,
            "server_registry": list(self._server_registry.keys())
        }

    async def get_performance_metrics(self, agent_id: str) -> dict[str, Any]:
        """Gibt Performance-Metriken für einen Agent zurück."""
        if agent_id not in self._client_instances:
            return {}

        client = self._client_instances[agent_id]
        component_metrics = self._metrics.get_component_metrics(client.client_id)

        if component_metrics:
            return {
                "component_id": component_metrics.component_id,
                "total_requests": component_metrics.total_requests,
                "successful_requests": component_metrics.successful_requests,
                "failed_requests": component_metrics.failed_requests,
                "success_rate": component_metrics.success_rate,
                "error_rate": component_metrics.error_rate,
                "average_latency": component_metrics.average_latency,
                "last_activity": component_metrics.last_activity.isoformat()
            }

        return {}

    async def _discover_servers_for_agent(
        self,
        client: UnifiedMCPClient,
        agent_id: str,
        agent_metadata: dict[str, Any] | None
    ) -> None:
        """Entdeckt und konfiguriert Server für einen Agent."""
        try:
            # Standard-Server für alle Agents
            default_servers = ["default-mcp-server"]

            # Agent-spezifische Server aus Metadaten
            if agent_metadata and "mcp_servers" in agent_metadata:
                default_servers.extend(agent_metadata["mcp_servers"])

            # Sessions für verfügbare Server initialisieren
            for server_id in default_servers:
                if server_id in self._server_registry:
                    server_config = self._server_registry[server_id]
                    try:
                        await client.initialize_session(
                            server_endpoint=server_config["endpoint"],
                            auth_config=server_config.get("auth_config")
                        )
                    except Exception as e:
                        # Server-Discovery-Fehler tolerieren
                        self._error_handler.handle_error(
                            e,
                            "server_discovery",
                            suppress=True,
                            additional_details={
                                "server_id": server_id,
                                "agent_id": agent_id
                            }
                        )

        except Exception as e:
            self._error_handler.handle_error(
                e,
                "discover_servers_for_agent",
                suppress=True,
                additional_details={
                    "agent_id": agent_id,
                    "agent_metadata": agent_metadata
                }
            )


# =============================================================================
# Context Manager für MCP Client Sessions
# =============================================================================

@asynccontextmanager
async def unified_mcp_client_session(
    agent_id: str,
    server_endpoint: str,
    transport_type: MCPTransportType = MCPTransportType.STREAMABLE_HTTP,
    auth_config: dict[str, Any] | None = None,
    agent_metadata: dict[str, Any] | None = None
) -> AsyncGenerator[tuple[UnifiedMCPClient, str], None]:
    """Context Manager für MCP Client Sessions mit automatischem Cleanup.

    Konsolidiert die Funktionalität von mcp_client_session und bietet
    verbesserte Fehlerbehandlung und Resource Management.

    Args:
        agent_id: Agent-ID für Client-Erstellung
        server_endpoint: Server-Endpunkt URL
        transport_type: Transport-Typ (default: STREAMABLE_HTTP)
        auth_config: Authentifizierungs-Konfiguration
        agent_metadata: Agent-Metadaten für Server-Discovery

    Yields:
        Tuple[client, session_id]: MCP Client und Session-ID
    """
    factory = UnifiedMCPClientFactory()
    client = None
    session_id = None

    try:
        # MCP Client für Agent erstellen
        client = await factory.create_client_for_agent(agent_id, agent_metadata)

        # Session initialisieren
        session_id = await client.initialize_session(
            server_endpoint,
            transport_type,
            auth_config
        )

        logger.debug(
            f"MCP Client Session erstellt: {agent_id}",
            extra={
                "agent_id": agent_id,
                "session_id": session_id,
                "server_endpoint": server_endpoint,
                "log_level": LogLevel.DEBUG
            }
        )

        yield client, session_id

    except Exception as e:
        logger.exception(
            f"Fehler in MCP Client Session: {e}",
            extra={
                "agent_id": agent_id,
                "server_endpoint": server_endpoint,
                "error": str(e),
                "log_level": LogLevel.ERROR
            }
        )
        raise

    finally:
        # Cleanup Session und Client
        if client and session_id:
            try:
                await client.cleanup_session(session_id)
            except Exception as e:
                logger.warning(
                    f"Fehler beim Session-Cleanup: {e}",
                    extra={
                        "session_id": session_id,
                        "error": str(e),
                        "log_level": LogLevel.WARNING
                    }
                )

        # Factory-Cleanup
        try:
            await factory.cleanup_client(agent_id)
        except Exception as e:
            logger.warning(
                f"Fehler beim Client-Cleanup: {e}",
                extra={
                    "agent_id": agent_id,
                    "error": str(e),
                    "log_level": LogLevel.WARNING
                }
            )


# =============================================================================
# Convenience Functions
# =============================================================================

def get_unified_mcp_client_factory() -> UnifiedMCPClientFactory:
    """Gibt die Singleton-Instanz der UnifiedMCPClientFactory zurück."""
    return UnifiedMCPClientFactory()


async def create_mcp_client_for_agent(
    agent_id: str,
    agent_metadata: dict[str, Any] | None = None
) -> UnifiedMCPClient:
    """Convenience-Funktion für MCP Client-Erstellung."""
    factory = get_unified_mcp_client_factory()
    return await factory.create_client_for_agent(agent_id, agent_metadata)


async def create_mcp_client_for_servers(
    agent_id: str,
    server_ids: list[str]
) -> UnifiedMCPClient:
    """Convenience-Funktion für MCP Client-Erstellung mit spezifischen Servern."""
    factory = get_unified_mcp_client_factory()
    return await factory.create_client_for_servers(agent_id, server_ids)


async def cleanup_mcp_client(agent_id: str) -> bool:
    """Convenience-Funktion für MCP Client-Cleanup."""
    factory = get_unified_mcp_client_factory()
    return await factory.cleanup_client(agent_id)


async def cleanup_all_mcp_clients() -> None:
    """Convenience-Funktion für Cleanup aller MCP Clients."""
    factory = get_unified_mcp_client_factory()
    await factory.cleanup_all_clients()


# =============================================================================
# Backward Compatibility Layer
# =============================================================================

# Legacy-Aliases für bestehenden Code
ProductionMCPClient = UnifiedMCPClient
MCPClientFactory = UnifiedMCPClientFactory
mcp_client_session = unified_mcp_client_session


# =============================================================================
# Export für einfachen Import
# =============================================================================

__all__ = [
    "MCPClientFactory",
    # Core Classes
    "MCPSessionConfig",
    "MCPTransportHandler",
    # Legacy Compatibility
    "ProductionMCPClient",
    "UnifiedMCPClient",
    "UnifiedMCPClientFactory",
    "cleanup_all_mcp_clients",
    "cleanup_mcp_client",
    "create_mcp_client_for_agent",
    "create_mcp_client_for_servers",
    # Convenience Functions
    "get_unified_mcp_client_factory",
    "mcp_client_session",
    # Context Managers
    "unified_mcp_client_session",
]
