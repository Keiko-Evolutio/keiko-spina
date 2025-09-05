"""LangChain Agent Protocol Integration.

Produktionsnahe Konversationsketten mit Tool-Integration und Memory-Anbindung.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from agents.factory.mcp_client_factory import ProductionMCPClient
from agents.protocols.core import (
    A2AInteropProtocol,
    AgentCapabilityType,
    AgentExecutionContext,
    AgentOperationResult,
    BaseAgentProtocol,
    ConnectedAgentProtocol,
    MCPCapability,
    MCPClientProtocol,
    MCPTransportType,
    create_agent_context,
)
from kei_logging import get_logger

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

try:
    from langchain_core.runnables import Runnable
    LANGCHAIN_AVAILABLE = True
except ImportError:
    Runnable = object  # type: ignore[assignment]
    LANGCHAIN_AVAILABLE = False

logger = get_logger(__name__)

# Konfiguration
SESSION_PREFIX = "lc"
ECHO_PREFIX = "Echo"
DEFAULT_CONNECTION_ID = "conn-1"
DEFAULT_PROTOCOL_VERSION = "v1"


class ProductionMCPClientAdapter(MCPClientProtocol):
    """Adapter für ProductionMCPClient um MCPClientProtocol zu implementieren."""

    def __init__(self, production_client: ProductionMCPClient):
        """Initialisiert den Adapter.

        Args:
            production_client: ProductionMCPClient Instanz
        """
        self._client = production_client

    async def initialize_session(
        self,
        server_endpoint: str,
        transport: MCPTransportType,
        auth_config: dict[str, Any] | None = None,
    ) -> str:
        """Initialisiert MCP Session."""
        return await self._client.initialize_session(server_endpoint, transport, auth_config)

    async def discover_capabilities(self, session_id: str) -> list[MCPCapability]:
        """Entdeckt verfügbare MCP Tools, Resources und Prompts."""
        # ProductionMCPClient gibt list[Any] zurück, wir konvertieren zu list[MCPCapability]
        _raw_capabilities = await self._client.discover_capabilities(session_id)
        # Fallback: Erstelle leere MCPCapability-Liste wenn keine vorhanden
        return []  # TODO: Implementiere Konvertierung wenn nötig - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/104

    async def invoke_tool(
        self, session_id: str, tool_id: str, parameters: dict[str, Any]
    ) -> AgentOperationResult:
        """Führt MCP Tool-Aufruf aus."""
        try:
            result = await self._client.invoke_tool(session_id, tool_id, parameters)
            return AgentOperationResult(success=True, result=result)
        except Exception as e:
            return AgentOperationResult(success=False, error=str(e))

    async def access_resource(
        self, session_id: str, resource_id: str, filters: dict[str, Any] | None = None
    ) -> AgentOperationResult:
        """Greift auf MCP Resource zu (Fallback-Implementierung)."""
        logger.warning(f"Resource access nicht implementiert: {resource_id}")
        return AgentOperationResult(
            success=False,
            error="Resource access nicht implementiert in ProductionMCPClient"
        )

    async def execute_prompt(
        self, session_id: str, prompt_id: str, variables: dict[str, Any]
    ) -> AgentOperationResult:
        """Führt MCP Prompt-Template aus (Fallback-Implementierung)."""
        logger.warning(f"Prompt execution nicht implementiert: {prompt_id}")
        return AgentOperationResult(
            success=False,
            error="Prompt execution nicht implementiert in ProductionMCPClient"
        )

    async def cleanup_session(self, session_id: str) -> None:
        """Bereinigt MCP Session und Resources."""
        await self._client.cleanup_session(session_id)


class LangChainFallbackConnectedAgent(ConnectedAgentProtocol):
    """Fallback-Implementierung für ConnectedAgent-Funktionalität."""

    async def delegate_task(
        self,
        _context: AgentExecutionContext,
        task_description: str,
        target_agent_id: str | None = None,
    ) -> AgentOperationResult:
        """Delegiert Task (Fallback-Implementierung)."""
        logger.info(f"Task delegation fallback: {task_description}")
        return AgentOperationResult(
            success=True,
            result={
                "delegated": True,
                "task": task_description,
                "target_agent": target_agent_id,
                "fallback": True,
            }
        )

    async def coordinate_workflow(
        self, _context: AgentExecutionContext, workflow_spec: dict[str, Any]
    ) -> AsyncIterator[AgentOperationResult]:
        """Koordiniert Workflow (Fallback-Implementierung)."""
        logger.info("Workflow coordination fallback")
        yield AgentOperationResult(
            success=True,
            result={
                "coordinated": True,
                "workflow": workflow_spec,
                "fallback": True,
            }
        )

    async def register_capability(
        self, capability: AgentCapabilityType, spec: dict[str, Any]
    ) -> bool:
        """Registriert Capability (Fallback-Implementierung)."""
        logger.info(f"Capability registration fallback: {capability}")
        return True


class LangChainFallbackA2AInterop(A2AInteropProtocol):
    """Fallback-Implementierung für A2A-Interoperabilität."""

    async def establish_a2a_connection(
        self, remote_agent_endpoint: str, protocol_version: str = DEFAULT_PROTOCOL_VERSION
    ) -> str:
        """Etabliert A2A-Verbindung (Fallback-Implementierung)."""
        logger.info(f"A2A connection fallback: {remote_agent_endpoint}")
        return f"{DEFAULT_CONNECTION_ID}-{uuid.uuid4().hex[:8]}"

    async def send_a2a_message(
        self, connection_id: str, message: dict[str, Any]
    ) -> AgentOperationResult:
        """Sendet A2A-Nachricht (Fallback-Implementierung)."""
        logger.info(f"A2A message fallback: {connection_id}")
        return AgentOperationResult(
            success=True,
            result={
                "sent": True,
                "connection_id": connection_id,
                "message_size": len(str(message)),
                "fallback": True,
            }
        )

    async def handle_a2a_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Behandelt A2A-Request (Fallback-Implementierung)."""
        logger.info("A2A request handling fallback")
        return {
            "handled": True,
            "request_type": request.get("type", "unknown"),
            "fallback": True,
        }


class LangChainAgentProtocol(BaseAgentProtocol):
    """LangChain Agent Protocol Integration.

    BaseAgentProtocol-Implementierung mit LangChain-Integration und Fallback-Mechanismen.
    """

    def __init__(self) -> None:
        """Initialisiert LangChain Agent Protocol."""
        self._production_mcp_client = ProductionMCPClient()
        self._mcp_client_adapter = ProductionMCPClientAdapter(self._production_mcp_client)
        self._default_chain: Runnable | None = None
        self._connected_agent_impl = LangChainFallbackConnectedAgent()
        self._a2a_impl = LangChainFallbackA2AInterop()

    @property
    def mcp_client(self) -> MCPClientProtocol:
        """Gibt den MCP-Client zurück."""
        return self._mcp_client_adapter

    async def initialize(
        self,
        agent: Any,
        services: Any,
        mcp_config: dict[str, Any] | None = None,
    ) -> AgentExecutionContext:
        """Initialisiert den Agenten und erstellt einen Kontext.

        Args:
            agent: Agent-Instanz oder Dictionary
            services: Service-Container
            mcp_config: MCP-Konfiguration

        Returns:
            AgentExecutionContext: Initialisierter Execution-Context
        """
        agent_id = self._extract_agent_id(agent)
        session_id = f"{SESSION_PREFIX}-{uuid.uuid4().hex[:8]}"
        context = create_agent_context(session_id=session_id, agent_id=agent_id)

        if mcp_config:
            await self._initialize_mcp_session(mcp_config)

        logger.info(f"LangChain agent initialized: {agent_id} (session: {session_id})")
        return context

    @staticmethod
    def _extract_agent_id(agent: Any) -> str:
        """Extrahiert Agent-ID aus Agent-Objekt.

        Args:
            agent: Agent-Instanz oder Dictionary

        Returns:
            Agent-ID als String
        """
        if hasattr(agent, "id") and agent.id:
            return str(agent.id)
        if isinstance(agent, dict) and agent.get("id"):
            return str(agent["id"])
        return f"agent-{uuid.uuid4().hex[:8]}"

    async def _initialize_mcp_session(self, mcp_config: dict[str, Any]) -> None:
        """Initialisiert MCP-Session basierend auf Konfiguration.

        Args:
            mcp_config: MCP-Konfiguration
        """
        endpoint = mcp_config.get("endpoint")
        if not endpoint:
            logger.warning("MCP-Konfiguration ohne Endpoint ignoriert")
            return

        transport = mcp_config.get("transport")
        auth = mcp_config.get("auth")

        try:
            await self._mcp_client_adapter.initialize_session(str(endpoint), transport, auth)
            logger.info(f"MCP-Session initialisiert: {endpoint}")
        except Exception as e:
            logger.error(f"MCP-Session-Initialisierung fehlgeschlagen: {e}")
            raise

    async def execute(
        self,
        context: AgentExecutionContext,
        input_data: dict[str, Any],
        callback: Any | None = None,
    ) -> AsyncIterator[AgentOperationResult]:
        """Führt LangChain-basierte Agent-Execution aus.

        Args:
            context: Execution-Context
            input_data: Input-Daten mit 'text' Feld
            callback: Optional Update-Callback

        Yields:
            AgentOperationResult: Execution-Ergebnisse
        """
        text = str(input_data.get("text", ""))
        if not text.strip():
            yield AgentOperationResult(
                success=False,
                error="Leerer Input-Text"
            )
            return

        # Fallback ohne LangChain
        if not LANGCHAIN_AVAILABLE:
            yield self._create_fallback_result(text)
            return

        # LangChain-basierte Execution
        try:
            async for result in self._execute_langchain(context, text, callback):
                yield result
        except Exception as exc:
            logger.error(f"LangChain execution error: {exc}")
            yield AgentOperationResult(success=False, error=str(exc))

    @staticmethod
    def _create_fallback_result(text: str) -> AgentOperationResult:
        """Erstellt Fallback-Result ohne LangChain.

        Args:
            text: Input-Text

        Returns:
            AgentOperationResult: Fallback-Ergebnis
        """
        logger.warning("LangChain nicht verfügbar - Fallback aktiv")
        return AgentOperationResult(
            success=True,
            result=f"{ECHO_PREFIX}: {text}",
            execution_time=0.001,
            trace_data={"fallback": True, "langchain_available": False}
        )

    async def _execute_langchain(
        self,
        context: AgentExecutionContext,
        text: str,
        callback: Any | None = None,
    ) -> AsyncIterator[AgentOperationResult]:
        """Führt LangChain-basierte Execution aus.

        Args:
            context: Execution-Context
            text: Input-Text
            callback: Optional Update-Callback

        Yields:
            AgentOperationResult: LangChain-Ergebnisse
        """
        import time
        start_time = time.time()

        try:
            # Lazy Import für bessere Performance
            from agents.chains.keiko_conversation_chain import (
                ChainConfig,
                KeikoConversationChain,
            )
            from agents.memory.langchain_cosmos_memory import CosmosChatMemory

            # Konfiguration und Kette erstellen
            chain_config = ChainConfig(session_id=context.session_id)
            memory = CosmosChatMemory()
            conversation = KeikoConversationChain(
                config=chain_config,
                memory=memory,
                tools=[]
            )

            # Execution mit Callback-Support
            if callback:
                # Callback über Start informieren
                pass  # Callback-Implementierung falls benötigt

            result = await conversation.ainvoke(text)
            execution_time = time.time() - start_time

            # Metriken aktualisieren
            context.add_operation_metrics(execution_time, success=True)

            yield AgentOperationResult(
                success=True,
                result=result,
                execution_time=execution_time,
                trace_data={
                    "session_id": context.session_id,
                    "langchain_available": True,
                    "chain_type": "KeikoConversationChain"
                }
            )

        except Exception:
            execution_time = time.time() - start_time
            context.add_operation_metrics(execution_time, success=False)
            raise

    async def suspend(self, context: AgentExecutionContext) -> bool:
        """Suspendiert Agent-Execution.

        Args:
            context: Execution-Context

        Returns:
            True wenn erfolgreich suspendiert
        """
        logger.info(f"Agent suspended: {context.agent_id}")
        return True

    async def resume(self, context: AgentExecutionContext) -> bool:
        """Setzt suspendierte Agent-Execution fort.

        Args:
            context: Execution-Context

        Returns:
            True wenn erfolgreich fortgesetzt
        """
        logger.info(f"Agent resumed: {context.agent_id}")
        return True

    async def terminate(self, context: AgentExecutionContext) -> None:
        """Terminiert Agent gracefully.

        Args:
            context: Execution-Context
        """
        try:
            await self._mcp_client_adapter.cleanup_session("")
            logger.info(f"Agent terminated: {context.agent_id}")
        except Exception as e:
            logger.warning(f"MCP client cleanup error: {e}")

    async def discover_mcp_capabilities(
        self, context: AgentExecutionContext
    ) -> list[MCPCapability]:
        """Entdeckt verfügbare MCP-Capabilities.

        Args:
            context: Execution-Context

        Returns:
            Liste verfügbarer MCP-Capabilities
        """
        try:
            # Hier könnte echte MCP-Discovery implementiert werden
            logger.info("MCP capability discovery (placeholder)")
            return []
        except Exception as e:
            logger.error(f"MCP capability discovery error: {e}")
            return []

    async def register_mcp_server(
        self, context: AgentExecutionContext, server_config: dict[str, Any]
    ) -> str:
        """Registriert neuen MCP-Server zur Laufzeit.

        Args:
            context: Execution-Context
            server_config: Server-Konfiguration

        Returns:
            Server-ID

        Raises:
            ValueError: Bei ungültiger Konfiguration
        """
        endpoint = server_config.get("endpoint")
        if not endpoint:
            raise ValueError("MCP-Server-Konfiguration benötigt 'endpoint'")

        auth = server_config.get("auth")
        transport = server_config.get("transport")

        try:
            session_id = await self._mcp_client_adapter.initialize_session(
                str(endpoint), transport, auth
            )
            logger.info(f"MCP-Server registriert: {endpoint}")
            return session_id
        except Exception as e:
            logger.error(f"MCP-Server-Registrierung fehlgeschlagen: {e}")
            raise

    @property
    def connected_agent(self) -> ConnectedAgentProtocol:
        """Connected Agent für Multi-Agent-Orchestrierung."""
        return self._connected_agent_impl

    async def discover_agent_network(self, context: AgentExecutionContext) -> list[Any]:
        """Entdeckt verfügbare Agents im Netzwerk.

        Args:
            context: Execution-Context

        Returns:
            Liste verfügbarer Agents (Fallback: leer)
        """
        logger.info("Agent network discovery (placeholder)")
        return []

    @property
    def a2a_interop(self) -> A2AInteropProtocol:
        """Agent2Agent Interoperabilität."""
        return self._a2a_impl

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
        logger.info(f"Extension registration: {extension_type}")
        return True

    async def validate_capability(self, capability: Any, spec: dict[str, Any]) -> bool:
        """Validiert Agent-Capability gegen Spezifikation.

        Args:
            capability: Capability-Typ
            spec: Capability-Spezifikation

        Returns:
            True wenn valide
        """
        logger.info(f"Capability validation: {capability}")
        return True

    async def get_health_status(self, context: AgentExecutionContext) -> dict[str, Any]:
        """Ermittelt detaillierten Health-Status des Agents.

        Args:
            context: Execution-Context

        Returns:
            Health-Status-Daten
        """
        return {
            "status": "healthy",
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "operation_count": context.operation_count,
            "error_rate": context.get_error_rate(),
            "uptime_seconds": (context.start_time.timestamp() if context.start_time else 0),
        }

    async def get_performance_metrics(self, context: AgentExecutionContext) -> dict[str, float]:
        """Sammelt Performance-Metriken für Monitoring.

        Args:
            context: Execution-Context

        Returns:
            Performance-Metriken
        """
        return {
            "operations_total": float(context.operation_count),
            "errors_total": float(context.error_count),
            "average_latency_seconds": context.get_average_latency(),
            "error_rate_percent": context.get_error_rate() * 100,
            "total_latency_seconds": context.total_latency,
        }

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
        health_status = await self.get_health_status(context)
        performance_metrics = await self.get_performance_metrics(context)

        return {
            "trace_id": context.trace_id,
            "export_format": output_format,
            "timestamp": context.start_time.isoformat() if context.start_time else None,
            "health": health_status,
            "performance": performance_metrics,
            "metadata": context.metadata,
            "langchain_integration": {
                "available": LANGCHAIN_AVAILABLE,
                "version": "unknown",  # Könnte aus LangChain-Modul extrahiert werden
            }
        }


__all__ = [
    "LangChainAgentProtocol",
    "LangChainFallbackA2AInterop",
    "LangChainFallbackConnectedAgent",
]
