"""LangGraph State Manager mit Azure Cosmos Persistierung.

Verantwortlich für:
- Typisierten Workflow-State (über `WorkflowState` Komposition)
- Persistierung/Resume über `CosmosCheckpointSaver`
- OpenTelemetry-Tracing-Integration
- Einheitliches Exception-Handling und Validierung
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from kei_logging import get_logger

from .state_constants import (
    LOG_WORKFLOW_REGISTRATION,
    LOG_WORKFLOW_RESUME,
    LOG_WORKFLOW_START,
    TRACE_WORKFLOW_RESUME,
    TRACE_WORKFLOW_START,
    WORKFLOW_NOT_FOUND_ERROR,
)
from .state_utils import (
    handle_workflow_operation,
    validate_workflow_config,
    validate_workflow_name,
)

if TYPE_CHECKING:
    from .langgraph_state_bridge import WorkflowState

logger = get_logger(__name__)


# =============================================================================
# Protocols für Dependency Injection
# =============================================================================


class CheckpointSaver(Protocol):
    """Protokoll für Checkpoint-Saver-Implementierungen."""

    def get(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Lädt Checkpoint für gegebene Konfiguration."""
        ...

    def put(
        self,
        config: dict[str, Any],
        value: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Speichert Checkpoint mit gegebener Konfiguration."""
        ...


# =============================================================================
# Workflow-Container mit verbesserter Type Safety
# =============================================================================


@dataclass
class ManagedWorkflow:
    """Container für kompilierten Graph und State-Management.

    Attributes:
        graph: Kompilierter LangGraph Graph (Runnable)
        checkpointer: Checkpoint-Saver für Persistierung
        name: Logischer Name des Workflows
    """

    graph: Any  # LangGraph Graph - Any wegen optionaler Abhängigkeit
    checkpointer: CheckpointSaver
    name: str


# =============================================================================
# State Manager mit verbesserter Architektur
# =============================================================================


class LangGraphStateManager:
    """State-Manager für LangGraph-Workflows mit Enterprise-Features.

    Bietet vereinfachte API für Start, Fortschritt und Resume von Workflows
    mit einheitlichem Exception-Handling, Validierung und Tracing.

    Features:
    - Typisierte Workflow-Registration
    - Automatische Input-Validierung
    - Einheitliches Exception-Handling
    - OpenTelemetry-Tracing
    - Dependency Injection für Checkpointer
    """

    def __init__(self) -> None:
        """Initialisiert den State-Manager."""
        self._workflows: dict[str, ManagedWorkflow] = {}

    def register(
        self, name: str, graph: Any, checkpointer: CheckpointSaver | None = None
    ) -> None:
        """Registriert kompilierten Graphen unter einem Namen.

        Args:
            name: Logischer Name des Workflows
            graph: Kompilierter Graph (z. B. `StateGraph(...).compile(...)`)
            checkpointer: Optional expliziter Checkpointer

        Raises:
            ValueError: Bei ungültigem Workflow-Namen
        """
        validate_workflow_name(name)

        # Lazy Import um zirkuläre Abhängigkeiten zu vermeiden
        if checkpointer is None:
            from agents.memory.langgraph_cosmos_checkpointer import CosmosCheckpointSaver
            checkpointer = CosmosCheckpointSaver()

        self._workflows[name] = ManagedWorkflow(
            graph=graph,
            checkpointer=checkpointer,
            name=name
        )

        logger.info(LOG_WORKFLOW_REGISTRATION.format(name=name))

    async def start(self, name: str, state: WorkflowState, *, thread_id: str) -> dict[str, Any]:
        """Startet einen Workflow neu mit Validierung und Tracing.

        Args:
            name: Workflow-Name
            state: Startzustand
            thread_id: Thread-ID für Checkpointing

        Returns:
            Dict mit 'state' und 'checkpoint' Feldern

        Raises:
            ValueError: Bei ungültigen Parametern
            KeikoNotFoundError: Wenn Workflow nicht registriert ist
        """
        validate_workflow_name(name)

        if name not in self._workflows:
            from core.exceptions import KeikoNotFoundError
            raise KeikoNotFoundError(
                WORKFLOW_NOT_FOUND_ERROR.format(name=name),
                details={"workflow": name}
            )

        managed = self._workflows[name]
        config = {"configurable": {"thread_id": thread_id}}
        validate_workflow_config(config)

        async def _start_operation() -> dict[str, Any]:
            result = await managed.graph.ainvoke(state.to_dict(), config=config)
            checkpoint = managed.checkpointer.get(config)
            logger.info(LOG_WORKFLOW_START.format(name=name, thread_id=thread_id))
            return {"state": result, "checkpoint": checkpoint}

        return await handle_workflow_operation(
            _start_operation,
            TRACE_WORKFLOW_START,
            name=name,
            thread_id=thread_id
        )

    async def resume(self, name: str, *, thread_id: str) -> dict[str, Any]:
        """Setzt einen Workflow anhand Checkpoints fort.

        Args:
            name: Workflow-Name
            thread_id: Thread-ID für Checkpoint-Wiederherstellung

        Returns:
            Dict mit 'state' und 'checkpoint' Feldern

        Raises:
            ValueError: Bei ungültigen Parametern
            KeikoNotFoundError: Wenn Workflow nicht registriert ist
        """
        validate_workflow_name(name)

        if name not in self._workflows:
            from core.exceptions import KeikoNotFoundError
            raise KeikoNotFoundError(
                WORKFLOW_NOT_FOUND_ERROR.format(name=name),
                details={"workflow": name}
            )

        managed = self._workflows[name]
        config = {"configurable": {"thread_id": thread_id}}
        validate_workflow_config(config)

        async def _resume_operation() -> dict[str, Any]:
            # Ein erneuter Aufruf mit identischem Thread führt LangGraph fort
            result = await managed.graph.ainvoke({}, config=config)
            checkpoint = managed.checkpointer.get(config)
            logger.info(LOG_WORKFLOW_RESUME.format(name=name, thread_id=thread_id))
            return {"state": result, "checkpoint": checkpoint}

        return await handle_workflow_operation(
            _resume_operation,
            TRACE_WORKFLOW_RESUME,
            name=name,
            thread_id=thread_id
        )

    def list_workflows(self) -> dict[str, str]:
        """Listet alle registrierten Workflows auf.

        Returns:
            Dict mit Workflow-Namen als Keys und Beschreibungen als Values
        """
        return {name: f"Workflow: {workflow.name}" for name, workflow in self._workflows.items()}

    def is_registered(self, name: str) -> bool:
        """Prüft ob ein Workflow registriert ist.

        Args:
            name: Workflow-Name

        Returns:
            True wenn registriert, False sonst
        """
        return name in self._workflows


__all__ = [
    "CheckpointSaver",  # Protocol für externe Verwendung
    "LangGraphStateManager",
    "ManagedWorkflow",
]
