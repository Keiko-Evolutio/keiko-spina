"""Bridge zwischen Agenten und n8n Workflows.

Integriert n8n-Workflow-Trigger in bestehende AgentExecutionContext-Logik und
wählt Workflows basierend auf Agent-Capabilities automatisch aus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function
from services.n8n import N8nClient
from services.n8n.bidirectional_sync import N8nStateSynchronizer
from services.n8n.models import ExecutionResult, TriggerResult

if TYPE_CHECKING:
    from agents.core import BaseAgent as AgentExecutionContext

logger = get_logger(__name__)


@dataclass(slots=True)
class N8nBridgeConfig:
    """Konfiguration für die Workflow-Bridge."""

    default_mode: str = "rest"


class N8nWorkflowBridge:
    """Ermöglicht bidirektionale Integration zwischen Agenten und n8n."""

    def __init__(self, config: N8nBridgeConfig | None = None) -> None:
        self.config = config or N8nBridgeConfig()
        # Synchronizer kann pro Bridge-Instanz wiederverwendet werden
        self._synchronizer: N8nStateSynchronizer | None = None

    @trace_function("agents.n8n_bridge.select_workflow")
    async def select_workflow_for_capability(
        self, context: AgentExecutionContext, capability: str
    ) -> str | None:
        """Wählt Workflow-ID anhand einer Capability.

        Aktuell: einfache Heuristik via Mapping in `context.resource_cache`.
        """
        # Einfache Lookup-Strategie: `context.resource_cache["n8n_workflows"][capability]`
        mapping: dict[str, str] = (
            context.resource_cache.get("n8n_workflows", {})
            if context and hasattr(context, "resource_cache") and context.resource_cache
            else {}
        )
        return mapping.get(capability)

    @trace_function("agents.n8n_bridge.trigger")
    async def trigger(
        self,
        context: AgentExecutionContext,
        capability: str,
        payload: dict[str, Any],
        *,
        mode: str | None = None,
        webhook_path: str | None = None,
    ) -> TriggerResult:
        """Triggert passenden n8n-Workflow für eine Capability."""
        workflow_id = await self.select_workflow_for_capability(context, capability)
        if not workflow_id:
            logger.warning(f"Kein n8n-Workflow für Capability '{capability}' gefunden")
            return TriggerResult(execution_id=None, started=False, raw={"error": "no_workflow"})

        client = N8nClient()
        try:
            result = await client.trigger_workflow(
                workflow_id=workflow_id,
                payload=payload,
                mode=mode or self.config.default_mode,
                webhook_path=webhook_path,
            )
            # Synchronisation starten, wenn Execution-ID vorhanden ist
            if result.execution_id and hasattr(context, "thread_id") and context.thread_id:
                self._synchronizer = N8nStateSynchronizer(
                    state_update_webhook_path=webhook_path,
                )
                await self._synchronizer.start(
                    execution_id=result.execution_id,
                    thread_id=context.thread_id,
                    initial_agent_state={"payload": payload, "capability": capability},
                )
            return result
        finally:
            await client.aclose()

    @trace_function("agents.n8n_bridge.poll_status")
    async def poll_status(self, execution_id: str) -> ExecutionResult:
        """Pollt den Status einer n8n-Execution."""
        client = N8nClient()
        try:
            return await client.poll_execution_status(execution_id)
        finally:
            await client.aclose()

    async def update_agent_state(self, state: dict[str, Any]) -> None:
        """Aktualisiert Agent-State in laufender Synchronisation (falls aktiv)."""
        if self._synchronizer is None:
            return
        await self._synchronizer.push_agent_state(state)

    async def stop_sync(self) -> None:
        """Stoppt die laufende Synchronisation (falls aktiv)."""
        if self._synchronizer is None:
            return
        await self._synchronizer.stop()
        self._synchronizer = None


__all__ = ["N8nBridgeConfig", "N8nWorkflowBridge"]
