"""Protocol Interfaces."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from .dataclasses import AgentExecutionContext, AgentOperationResult, MCPCapability
from .enums import AgentCapabilityType, MCPTransportType

# TYPE_CHECKING Block entfernt - keine spezifischen Typen verwendet


@runtime_checkable
class MCPClientProtocol(Protocol):
    """Protocol für MCP Client-Implementierungen."""

    async def initialize_session(
        self,
        server_endpoint: str,
        transport: MCPTransportType,
        auth_config: dict[str, Any] | None = None,
    ) -> str:
        """Initialisiert MCP Session.

        Args:
            server_endpoint: MCP Server-Endpunkt
            transport: Transport-Mechanismus
            auth_config: Authentifizierungs-Konfiguration

        Returns:
            Session-ID
        """
        ...

    async def discover_capabilities(self, session_id: str) -> list[MCPCapability]:
        """Entdeckt verfügbare MCP Tools, Resources und Prompts.

        Args:
            session_id: MCP Session-ID

        Returns:
            Liste verfügbarer Capabilities
        """
        ...

    async def invoke_tool(
        self, session_id: str, tool_id: str, parameters: dict[str, Any]
    ) -> AgentOperationResult:
        """Führt MCP Tool-Aufruf aus.

        Args:
            session_id: MCP Session-ID
            tool_id: Tool-Identifier
            parameters: Tool-Parameter

        Returns:
            Ausführungsergebnis
        """
        ...

    async def access_resource(
        self, session_id: str, resource_id: str, filters: dict[str, Any] | None = None
    ) -> AgentOperationResult:
        """Greift auf MCP Resource zu.

        Args:
            session_id: MCP Session-ID
            resource_id: Resource-Identifier
            filters: Optionale Filter

        Returns:
            Resource-Daten
        """
        ...

    async def execute_prompt(
        self, session_id: str, prompt_id: str, variables: dict[str, Any]
    ) -> AgentOperationResult:
        """Führt MCP Prompt-Template aus.

        Args:
            session_id: MCP Session-ID
            prompt_id: Prompt-Identifier
            variables: Template-Variablen

        Returns:
            Prompt-Ergebnis
        """
        ...

    async def cleanup_session(self, session_id: str) -> None:
        """Bereinigt MCP Session und Resources.

        Args:
            session_id: MCP Session-ID
        """
        ...


@runtime_checkable
class ConnectedAgentProtocol(Protocol):
    """Protocol für Connected Agents - Multi-Agent Orchestrierung."""

    async def delegate_task(
        self,
        context: AgentExecutionContext,
        task_description: str,
        target_agent_id: str | None = None,
    ) -> AgentOperationResult:
        """Delegiert Task an spezialisierten Sub-Agent.

        Args:
            context: Execution-Context
            task_description: Task-Beschreibung
            target_agent_id: Ziel-Agent (optional)

        Returns:
            Delegations-Ergebnis
        """
        ...

    async def coordinate_workflow(
        self, context: AgentExecutionContext, workflow_spec: dict[str, Any]
    ) -> AsyncIterator[AgentOperationResult]:
        """Koordiniert Multi-Agent Workflow.

        Args:
            context: Execution-Context
            workflow_spec: Workflow-Spezifikation

        Yields:
            Workflow-Schritte
        """
        ...

    async def register_capability(
        self, capability: AgentCapabilityType, spec: dict[str, Any]
    ) -> bool:
        """Registriert neue Agent-Capability.

        Args:
            capability: Capability-Typ
            spec: Capability-Spezifikation

        Returns:
            True wenn erfolgreich registriert
        """
        ...


@runtime_checkable
class A2AInteropProtocol(Protocol):
    """Protocol für Agent2Agent (A2A) Interoperabilität."""

    async def establish_a2a_connection(
        self, remote_agent_endpoint: str, protocol_version: str = "v1"
    ) -> str:
        """Etabliert A2A Verbindung zu externem Agent.

        Args:
            remote_agent_endpoint: Remote-Agent-Endpunkt
            protocol_version: Protocol-Version

        Returns:
            Connection-ID
        """
        ...

    async def send_a2a_message(
        self, connection_id: str, message: dict[str, Any]
    ) -> AgentOperationResult:
        """Sendet strukturierte Nachricht über A2A Protocol.

        Args:
            connection_id: Verbindungs-ID
            message: Nachricht

        Returns:
            Sende-Ergebnis
        """
        ...

    async def handle_a2a_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Behandelt eingehende A2A Requests.

        Args:
            request: Eingehender Request

        Returns:
            Response-Daten
        """
        ...


@runtime_checkable
class BaseAgentProtocol(Protocol):
    """Production-Ready Protocol-Interface für multi-agent Systeme."""

    # Agent Lifecycle Management
    @abstractmethod
    async def initialize(
        self, agent: Any, services: Any, mcp_config: dict[str, Any] | None = None
    ) -> AgentExecutionContext:
        """Initialisiert Agent mit MCP-Konfiguration und Azure-Services.

        Args:
            agent: Agent-Instanz
            services: Service-Container
            mcp_config: MCP-Konfiguration

        Returns:
            Execution-Context
        """
        ...

    @abstractmethod
    async def execute(
        self,
        context: AgentExecutionContext,
        input_data: dict[str, Any],
        callback: Any | None = None,
    ) -> AsyncIterator[AgentOperationResult]:
        """Führt Agent-Operation aus mit MCP-Integration.

        Args:
            context: Execution-Context
            input_data: Input-Daten
            callback: Update-Callback

        Yields:
            Ausführungsergebnisse
        """
        ...

    @abstractmethod
    async def suspend(self, context: AgentExecutionContext) -> bool:
        """Suspendiert Agent-Execution.

        Args:
            context: Execution-Context

        Returns:
            True wenn erfolgreich suspendiert
        """
        ...

    @abstractmethod
    async def resume(self, context: AgentExecutionContext) -> bool:
        """Setzt suspendierte Agent-Execution fort.

        Args:
            context: Execution-Context

        Returns:
            True wenn erfolgreich fortgesetzt
        """
        ...

    @abstractmethod
    async def terminate(self, context: AgentExecutionContext) -> None:
        """Terminiert Agent gracefully.

        Args:
            context: Execution-Context
        """
        ...

    # MCP Client Capabilities
    @property
    @abstractmethod
    def mcp_client(self) -> MCPClientProtocol:
        """MCP Client für Tools, Resources und Prompts."""
        ...

    @abstractmethod
    async def discover_mcp_capabilities(
        self, context: AgentExecutionContext
    ) -> list[MCPCapability]:
        """Entdeckt verfügbare MCP-Capabilities.

        Args:
            context: Execution-Context

        Returns:
            Liste verfügbarer Capabilities
        """
        ...

    @abstractmethod
    async def register_mcp_server(
        self, context: AgentExecutionContext, server_config: dict[str, Any]
    ) -> str:
        """Registriert neuen MCP-Server zur Laufzeit.

        Args:
            context: Execution-Context
            server_config: Server-Konfiguration

        Returns:
            Server-ID
        """
        ...

    # Multi-Agent Orchestrierung
    @property
    @abstractmethod
    def connected_agent(self) -> ConnectedAgentProtocol:
        """Connected Agent für Multi-Agent-Orchestrierung."""
        ...

    @abstractmethod
    async def discover_agent_network(self, context: AgentExecutionContext) -> list[Any]:
        """Entdeckt verfügbare Agents im Netzwerk.

        Args:
            context: Execution-Context

        Returns:
            Liste verfügbarer Agents
        """
        ...

    # Interoperabilität und Extensibility
    @property
    @abstractmethod
    def a2a_interop(self) -> A2AInteropProtocol:
        """Agent2Agent Interoperabilität."""
        ...

    @abstractmethod
    async def register_extension(
        self, extension_type: str, _extension_impl: Any, metadata: dict[str, Any]
    ) -> bool:
        """Registriert Framework-Extension zur Laufzeit.

        Args:
            extension_type: Extension-Typ
            extension_impl: Extension-Implementierung
            metadata: Extension-Metadaten

        Returns:
            True wenn erfolgreich registriert
        """
        ...

    @abstractmethod
    async def validate_capability(
        self, capability: AgentCapabilityType, spec: dict[str, Any]
    ) -> bool:
        """Validiert Agent-Capability gegen Spezifikation.

        Args:
            capability: Capability-Typ
            spec: Capability-Spezifikation

        Returns:
            True wenn valide
        """
        ...

    # Observability und Health Monitoring
    @abstractmethod
    async def get_health_status(self, context: AgentExecutionContext) -> dict[str, Any]:
        """Ermittelt detaillierten Health-Status des Agents.

        Args:
            context: Execution-Context

        Returns:
            Health-Status-Daten
        """
        ...

    @abstractmethod
    async def get_performance_metrics(self, context: AgentExecutionContext) -> dict[str, float]:
        """Sammelt Performance-Metriken für Monitoring.

        Args:
            context: Execution-Context

        Returns:
            Performance-Metriken
        """
        ...

    @abstractmethod
    async def export_telemetry(
        self, context: AgentExecutionContext, output_format: str = "json"
    ) -> dict[str, Any]:
        """Exportiert umfassende Telemetrie-Daten.

        Args:
            context: Execution-Context
            output_format: Export-Format

        Returns:
            Telemetrie-Daten
        """
        ...


__all__ = [
    "A2AInteropProtocol",
    "BaseAgentProtocol",
    "ConnectedAgentProtocol",
    "MCPClientProtocol",
]
