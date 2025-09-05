# backend/services/orchestrator/state_store.py
"""State Store für Orchestrator Service mit Plan-Persistierung.

Implementiert persistente Speicherung von Task-Plänen, Saga-States und
Orchestration-Zuständen für Wiederaufnahme nach Ausfällen.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agents.memory.langgraph_cosmos_checkpointer import CosmosCheckpointSaver
from kei_logging import get_logger
from services.enhanced_security_integration import PlanPersistenceManager, SecurityLevel

from ..enhanced_real_time_monitoring.data_models import SagaStatus, SagaStep, SagaTransaction
from ..task_decomposition.data_models import DecompositionPlan, SubtaskDefinition

if TYPE_CHECKING:
    from agents.state import CheckpointSaver

logger = get_logger(__name__)


class OrchestratorStateStore:
    """State Store für Orchestrator Service mit Enterprise-Grade Persistierung.

    Features:
    - Plan-Persistierung mit Verschlüsselung
    - Saga-State-Management
    - Checkpoint-basierte Recovery
    - Distributed State Synchronization
    - Automatic Cleanup und Archivierung
    """

    def __init__(
        self,
        plan_persistence_manager: PlanPersistenceManager | None = None,
        checkpoint_saver: CheckpointSaver | None = None,
        cleanup_interval_hours: int = 24,
        retention_days: int = 30
    ):
        """Initialisiert State Store.

        Args:
            plan_persistence_manager: Manager für verschlüsselte Plan-Persistierung
            checkpoint_saver: Checkpoint-Saver für State-Snapshots
            cleanup_interval_hours: Intervall für automatische Bereinigung
            retention_days: Aufbewahrungszeit für States
        """
        self.plan_persistence_manager = plan_persistence_manager or PlanPersistenceManager()
        self.checkpoint_saver = checkpoint_saver or CosmosCheckpointSaver()
        self.cleanup_interval_hours = cleanup_interval_hours
        self.retention_days = retention_days

        # In-Memory Cache für aktive States
        self._active_plans: dict[str, DecompositionPlan] = {}
        self._active_sagas: dict[str, SagaTransaction] = {}
        self._plan_metadata: dict[str, dict[str, Any]] = {}

        # Background-Tasks
        self._cleanup_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("Orchestrator State Store initialisiert")

    async def start(self) -> None:
        """Startet State Store."""
        if self._is_running:
            return

        try:
            # Starte Plan Persistence Manager
            await self.plan_persistence_manager.start()

            # Starte Background-Cleanup
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._is_running = True
            logger.info("Orchestrator State Store gestartet")

        except Exception as e:
            logger.exception(f"State Store Start fehlgeschlagen: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stoppt State Store."""
        self._is_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Stoppe Plan Persistence Manager
        await self.plan_persistence_manager.stop()

        logger.info("Orchestrator State Store gestoppt")

    # =========================================================================
    # Plan-Persistierung
    # =========================================================================

    async def persist_plan(
        self,
        plan: DecompositionPlan,
        tenant_id: str | None = None,
        security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL
    ) -> str:
        """Persistiert Decomposition-Plan.

        Args:
            plan: Plan zum Persistieren
            tenant_id: Tenant-ID für Multi-Tenancy
            security_level: Security-Level für Verschlüsselung

        Returns:
            State-ID für Plan-Recovery
        """
        start_time = time.time()

        try:
            # Serialisiere Plan
            plan_data = {
                "plan_id": plan.plan_id,
                "subtasks": [self._serialize_subtask(subtask) for subtask in plan.subtasks],
                "execution_strategy": plan.execution_strategy.value,
                "agent_assignments": {k: self._serialize_agent_match(v) for k, v in plan.agent_assignments.items()},
                "dependency_graph": plan.dependency_graph,
                "estimated_total_duration_minutes": plan.estimated_total_duration_minutes,
                "estimated_parallel_duration_minutes": plan.estimated_parallel_duration_minutes,
                "parallelization_efficiency": plan.parallelization_efficiency,
                "plan_confidence": plan.plan_confidence,
                "created_at": plan.created_at.isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            # Persistiere verschlüsselt
            state_id = f"plan_{plan.plan_id}_{uuid4().hex[:8]}"
            await self.plan_persistence_manager.encrypt_and_store_state(
                state_id=state_id,
                state_data=plan_data,
                tenant_id=tenant_id,
                security_level=security_level
            )

            # Cache Plan
            self._active_plans[plan.plan_id] = plan
            self._plan_metadata[plan.plan_id] = {
                "state_id": state_id,
                "tenant_id": tenant_id,
                "security_level": security_level.value,
                "persisted_at": datetime.utcnow().isoformat(),
                "size_bytes": len(json.dumps(plan_data).encode("utf-8"))
            }

            persist_time_ms = (time.time() - start_time) * 1000

            logger.info({
                "event": "plan_persisted",
                "plan_id": plan.plan_id,
                "state_id": state_id,
                "persist_time_ms": persist_time_ms,
                "subtasks_count": len(plan.subtasks),
                "security_level": security_level.value
            })

            return state_id

        except Exception as e:
            logger.exception(f"Plan-Persistierung fehlgeschlagen: {e}")
            raise

    async def recover_plan(
        self,
        plan_id: str,
        tenant_id: str | None = None
    ) -> DecompositionPlan | None:
        """Stellt Plan aus Persistierung wieder her.

        Args:
            plan_id: Plan-ID
            tenant_id: Tenant-ID für Access-Control

        Returns:
            Wiederhergestellter Plan oder None
        """
        start_time = time.time()

        try:
            # Prüfe Cache
            if plan_id in self._active_plans:
                logger.debug(f"Plan {plan_id} aus Cache geladen")
                return self._active_plans[plan_id]

            # Hole Metadata
            if plan_id not in self._plan_metadata:
                logger.warning(f"Plan {plan_id} nicht gefunden")
                return None

            metadata = self._plan_metadata[plan_id]
            state_id = metadata["state_id"]

            # Lade verschlüsselte Daten
            plan_data = await self.plan_persistence_manager.retrieve_and_decrypt_state(
                state_id=state_id,
                tenant_id=tenant_id,
                verify_integrity=True
            )

            if not plan_data:
                logger.warning(f"Plan-Daten für {plan_id} nicht gefunden")
                return None

            # Deserialisiere Plan
            plan = self._deserialize_plan(plan_data)

            # Cache Plan
            self._active_plans[plan_id] = plan

            recovery_time_ms = (time.time() - start_time) * 1000

            logger.info({
                "event": "plan_recovered",
                "plan_id": plan_id,
                "state_id": state_id,
                "recovery_time_ms": recovery_time_ms,
                "subtasks_count": len(plan.subtasks)
            })

            return plan

        except Exception as e:
            logger.exception(f"Plan-Recovery fehlgeschlagen: {e}")
            return None

    # =========================================================================
    # Saga-State-Management
    # =========================================================================

    async def persist_saga_state(
        self,
        saga: SagaTransaction,
        tenant_id: str | None = None
    ) -> str:
        """Persistiert Saga-State.

        Args:
            saga: Saga-Transaction
            tenant_id: Tenant-ID

        Returns:
            State-ID für Saga-Recovery
        """
        try:
            # Serialisiere Saga
            saga_data = {
                "saga_id": saga.saga_id,
                "steps": [self._serialize_saga_step(step) for step in saga.steps],
                "compensation_strategy": saga.compensation_strategy,
                "status": saga.status.value,
                "current_step_index": saga.current_step_index,
                "started_at": saga.started_at.isoformat() if saga.started_at else None,
                "completed_at": saga.completed_at.isoformat() if saga.completed_at else None,
                "error_message": saga.error_message,
                "compensation_executed": saga.compensation_executed,
                "updated_at": datetime.utcnow().isoformat()
            }

            # Persistiere verschlüsselt
            state_id = f"saga_{saga.saga_id}_{uuid4().hex[:8]}"
            await self.plan_persistence_manager.encrypt_and_store_state(
                state_id=state_id,
                state_data=saga_data,
                tenant_id=tenant_id,
                security_level=SecurityLevel.CONFIDENTIAL
            )

            # Cache Saga
            self._active_sagas[saga.saga_id] = saga

            logger.info({
                "event": "saga_state_persisted",
                "saga_id": saga.saga_id,
                "state_id": state_id,
                "status": saga.status.value,
                "steps_count": len(saga.steps)
            })

            return state_id

        except Exception as e:
            logger.exception(f"Saga-State-Persistierung fehlgeschlagen: {e}")
            raise

    async def recover_saga_state(
        self,
        saga_id: str,
        tenant_id: str | None = None
    ) -> SagaTransaction | None:
        """Stellt Saga-State wieder her.

        Args:
            saga_id: Saga-ID
            tenant_id: Tenant-ID

        Returns:
            Wiederhergestellte Saga oder None
        """
        try:
            # Prüfe Cache
            if saga_id in self._active_sagas:
                return self._active_sagas[saga_id]

            # Suche State-ID (vereinfacht - in Production würde man Index verwenden)
            state_id = f"saga_{saga_id}"  # Vereinfachte Suche

            # Lade verschlüsselte Daten
            saga_data = await self.plan_persistence_manager.retrieve_and_decrypt_state(
                state_id=state_id,
                tenant_id=tenant_id
            )

            if not saga_data:
                return None

            # Deserialisiere Saga
            saga = self._deserialize_saga(saga_data)

            # Cache Saga
            self._active_sagas[saga_id] = saga

            logger.info({
                "event": "saga_state_recovered",
                "saga_id": saga_id,
                "status": saga.status.value,
                "steps_count": len(saga.steps)
            })

            return saga

        except Exception as e:
            logger.exception(f"Saga-State-Recovery fehlgeschlagen: {e}")
            return None

    # =========================================================================
    # Checkpoint-Management
    # =========================================================================

    async def create_checkpoint(
        self,
        orchestration_id: str,
        state_data: dict[str, Any],
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Erstellt Checkpoint für Orchestration.

        Args:
            orchestration_id: Orchestration-ID
            state_data: State-Daten
            metadata: Zusätzliche Metadaten

        Returns:
            Checkpoint-ID
        """
        try:
            config = {
                "configurable": {
                    "thread_id": f"orchestration_{orchestration_id}",
                    "checkpoint_ns": "orchestrator"
                }
            }

            checkpoint_metadata = {
                "orchestration_id": orchestration_id,
                "created_at": datetime.utcnow().isoformat(),
                "checkpoint_type": "orchestration_state",
                **(metadata or {})
            }

            result = self.checkpoint_saver.put(config, state_data, checkpoint_metadata)
            checkpoint_id = result.get("checkpoint_id", str(uuid4()))

            logger.info({
                "event": "checkpoint_created",
                "orchestration_id": orchestration_id,
                "checkpoint_id": checkpoint_id
            })

            return checkpoint_id

        except Exception as e:
            logger.exception(f"Checkpoint-Erstellung fehlgeschlagen: {e}")
            raise

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _serialize_subtask(self, subtask: SubtaskDefinition) -> dict[str, Any]:
        """Serialisiert Subtask."""
        return {
            "subtask_id": subtask.subtask_id,
            "name": subtask.name,
            "description": subtask.description,
            "required_capabilities": subtask.required_capabilities,
            "optional_capabilities": subtask.optional_capabilities,
            "depends_on": subtask.depends_on,
            "estimated_duration_minutes": subtask.estimated_duration_minutes,
            "priority": subtask.priority.value,
            "payload": subtask.payload
        }

    def _serialize_agent_match(self, agent_match) -> dict[str, Any]:
        """Serialisiert Agent-Match."""
        return {
            "agent_id": agent_match.agent_id,
            "match_score": agent_match.match_score,
            "capability_coverage": agent_match.capability_coverage,
            "estimated_performance": agent_match.estimated_performance
        }

    def _serialize_saga_step(self, step: SagaStep) -> dict[str, Any]:
        """Serialisiert Saga-Step."""
        return {
            "step_id": step.step_id,
            "action": step.action,
            "compensation_action": step.compensation_action.value,
            "status": step.status.value,
            "started_at": step.started_at.isoformat() if step.started_at else None,
            "completed_at": step.completed_at.isoformat() if step.completed_at else None,
            "error_message": step.error_message,
            "retry_count": step.retry_count,
            "context": step.context
        }

    def _deserialize_plan(self, plan_data: dict[str, Any]) -> DecompositionPlan:
        """Deserialisiert Plan."""
        # Vereinfachte Deserialisierung - in Production würde man vollständige Objektrekonstruktion implementieren
        from ..task_decomposition.data_models import DecompositionStrategy

        return DecompositionPlan(
            plan_id=plan_data["plan_id"],
            subtasks=[],  # Würde vollständig deserialisiert werden
            execution_strategy=DecompositionStrategy(plan_data["execution_strategy"]),
            agent_assignments={},  # Würde vollständig deserialisiert werden
            dependency_graph=plan_data["dependency_graph"],
            estimated_total_duration_minutes=plan_data["estimated_total_duration_minutes"],
            estimated_parallel_duration_minutes=plan_data["estimated_parallel_duration_minutes"],
            parallelization_efficiency=plan_data["parallelization_efficiency"],
            plan_confidence=plan_data["plan_confidence"]
        )

    def _deserialize_saga(self, saga_data: dict[str, Any]) -> SagaTransaction:
        """Deserialisiert Saga."""
        # Vereinfachte Deserialisierung
        return SagaTransaction(
            saga_id=saga_data["saga_id"],
            steps=[],  # Würde vollständig deserialisiert werden
            compensation_strategy=saga_data["compensation_strategy"],
            status=SagaStatus(saga_data["status"])
        )

    async def _cleanup_loop(self) -> None:
        """Background-Loop für automatische Bereinigung."""
        while self._is_running:
            try:
                await asyncio.sleep(self.cleanup_interval_hours * 3600)
                await self._cleanup_old_states()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Cleanup-Loop Fehler: {e}")

    async def _cleanup_old_states(self) -> None:
        """Bereinigt alte States."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

            # Bereinige Plan-Metadaten
            plans_to_remove = []
            for plan_id, metadata in self._plan_metadata.items():
                persisted_at = datetime.fromisoformat(metadata["persisted_at"])
                if persisted_at < cutoff_date:
                    plans_to_remove.append(plan_id)

            for plan_id in plans_to_remove:
                del self._plan_metadata[plan_id]
                if plan_id in self._active_plans:
                    del self._active_plans[plan_id]

            logger.info(f"Cleanup: {len(plans_to_remove)} alte Plans entfernt")

        except Exception as e:
            logger.exception(f"State-Cleanup fehlgeschlagen: {e}")
