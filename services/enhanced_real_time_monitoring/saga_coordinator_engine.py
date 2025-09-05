# backend/services/enhanced_real_time_monitoring/saga_coordinator_engine.py
"""Saga Coordinator Engine für Enterprise-Grade Reliability.

Implementiert Saga-Pattern mit Compensation-Logic für robuste
Distributed-Transaction-Management und Enterprise-Reliability.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from kei_logging import get_logger
from services.enhanced_security_integration import SecurityContext

from .data_models import CompensationAction, SagaStatus, SagaStep, SagaTransaction

logger = get_logger(__name__)


class SagaCoordinatorEngine:
    """Saga Coordinator Engine für Enterprise-Grade Reliability."""

    def __init__(self):
        """Initialisiert Saga Coordinator Engine."""
        # Saga-Storage
        self._active_sagas: dict[str, SagaTransaction] = {}
        self._completed_sagas: dict[str, SagaTransaction] = {}
        self._saga_execution_history: dict[str, list[dict[str, Any]]] = {}

        # Compensation-Logic
        self._compensation_strategies = {
            "reverse_order": self._compensate_reverse_order,
            "parallel": self._compensate_parallel,
            "custom": self._compensate_custom
        }

        # Performance-Tracking
        self._saga_performance_stats = {
            "total_sagas_executed": 0,
            "successful_sagas": 0,
            "failed_sagas": 0,
            "compensated_sagas": 0,
            "avg_saga_execution_time_ms": 0.0,
            "avg_compensation_time_ms": 0.0
        }

        # Background-Tasks
        self._coordinator_tasks: list[asyncio.Task] = []
        self._is_running = False

        # Timeout-Management
        self._saga_timeouts: dict[str, asyncio.Task] = {}

        logger.info("Saga Coordinator Engine initialisiert")

    async def start(self) -> None:
        """Startet Saga Coordinator Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._coordinator_tasks = [
            asyncio.create_task(self._saga_monitoring_loop()),
            asyncio.create_task(self._timeout_management_loop()),
            asyncio.create_task(self._compensation_monitoring_loop())
        ]

        logger.info("Saga Coordinator Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Saga Coordinator Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._coordinator_tasks:
            task.cancel()

        await asyncio.gather(*self._coordinator_tasks, return_exceptions=True)
        self._coordinator_tasks.clear()

        # Stoppe Timeout-Tasks
        for timeout_task in self._saga_timeouts.values():
            timeout_task.cancel()
        self._saga_timeouts.clear()

        logger.info("Saga Coordinator Engine gestoppt")

    async def create_saga(
        self,
        saga_name: str,
        description: str,
        steps: list[dict[str, Any]],
        compensation_strategy: str = "reverse_order",
        timeout_seconds: int = 1800,
        security_context: SecurityContext | None = None
    ) -> str:
        """Erstellt neue Saga-Transaction.

        Args:
            saga_name: Name der Saga
            description: Beschreibung der Saga
            steps: Liste von Saga-Steps
            compensation_strategy: Compensation-Strategie
            timeout_seconds: Timeout in Sekunden
            security_context: Security-Context

        Returns:
            Saga-ID
        """
        try:
            import uuid

            saga_id = str(uuid.uuid4())

            # Erstelle Saga-Steps
            saga_steps = []
            for i, step_config in enumerate(steps):
                step = SagaStep(
                    step_id=str(uuid.uuid4()),
                    saga_id=saga_id,
                    step_name=step_config["step_name"],
                    step_order=i,
                    service_name=step_config["service_name"],
                    operation=step_config["operation"],
                    parameters=step_config.get("parameters", {}),
                    compensation_operation=step_config.get("compensation_operation"),
                    compensation_parameters=step_config.get("compensation_parameters", {}),
                    compensation_action=CompensationAction(step_config.get("compensation_action", "rollback")),
                    timeout_seconds=step_config.get("timeout_seconds", 300),
                    max_retries=step_config.get("max_retries", 3)
                )
                saga_steps.append(step)

            # Erstelle Saga-Transaction
            saga = SagaTransaction(
                saga_id=saga_id,
                saga_name=saga_name,
                description=description,
                steps=saga_steps,
                compensation_strategy=compensation_strategy,
                timeout_seconds=timeout_seconds,
                orchestration_id=security_context.request_id if security_context else None,
                user_id=security_context.user_id if security_context else None,
                tenant_id=security_context.tenant_id if security_context else None,
                security_level=security_context.security_level if security_context else None
            )

            self._active_sagas[saga_id] = saga
            self._saga_execution_history[saga_id] = []

            logger.info({
                "event": "saga_created",
                "saga_id": saga_id,
                "saga_name": saga_name,
                "steps_count": len(saga_steps),
                "compensation_strategy": compensation_strategy
            })

            return saga_id

        except Exception as e:
            logger.error(f"Saga creation fehlgeschlagen: {e}")
            raise

    async def execute_saga(
        self,
        saga_id: str,
        security_context: SecurityContext | None = None
    ) -> dict[str, Any]:
        """Führt Saga-Transaction aus.

        Args:
            saga_id: Saga-ID
            security_context: Security-Context

        Returns:
            Execution-Result
        """
        start_time = time.time()

        try:
            saga = self._active_sagas.get(saga_id)
            if not saga:
                raise ValueError(f"Saga {saga_id} nicht gefunden")

            logger.info({
                "event": "saga_execution_started",
                "saga_id": saga_id,
                "saga_name": saga.saga_name,
                "steps_count": len(saga.steps)
            })

            # Update Saga-Status
            saga.status = SagaStatus.RUNNING
            saga.started_at = datetime.utcnow()

            # Starte Timeout-Management
            await self._start_saga_timeout(saga_id, saga.timeout_seconds)

            # Führe Steps sequenziell aus
            execution_result = await self._execute_saga_steps(saga, security_context)

            # Berechne Execution-Zeit
            execution_time_ms = (time.time() - start_time) * 1000

            if execution_result["success"]:
                saga.status = SagaStatus.COMPLETED
                saga.completed_at = datetime.utcnow()
                saga.final_result = execution_result

                # Move zu completed sagas
                self._completed_sagas[saga_id] = saga
                del self._active_sagas[saga_id]

                self._saga_performance_stats["successful_sagas"] += 1

                logger.info({
                    "event": "saga_execution_completed",
                    "saga_id": saga_id,
                    "execution_time_ms": execution_time_ms,
                    "completed_steps": len(saga.completed_steps)
                })

            else:
                # Saga fehlgeschlagen - starte Compensation
                saga.status = SagaStatus.COMPENSATING
                saga.error = execution_result.get("error", "Unknown error")

                compensation_result = await self._compensate_saga(saga, security_context)

                if compensation_result["success"]:
                    saga.status = SagaStatus.FAILED
                    saga.compensation_result = compensation_result
                    self._saga_performance_stats["compensated_sagas"] += 1
                else:
                    saga.status = SagaStatus.FAILED
                    saga.error = f"Compensation failed: {compensation_result.get('error', 'Unknown error')}"

                saga.completed_at = datetime.utcnow()

                # Move zu completed sagas
                self._completed_sagas[saga_id] = saga
                del self._active_sagas[saga_id]

                self._saga_performance_stats["failed_sagas"] += 1

                logger.error({
                    "event": "saga_execution_failed",
                    "saga_id": saga_id,
                    "execution_time_ms": execution_time_ms,
                    "error": saga.error,
                    "compensation_success": compensation_result["success"]
                })

            # Update Performance-Stats
            self._update_saga_performance_stats(execution_time_ms)

            # Cleanup Timeout
            await self._cleanup_saga_timeout(saga_id)

            return {
                "success": execution_result["success"],
                "saga_id": saga_id,
                "execution_time_ms": execution_time_ms,
                "completed_steps": len(saga.completed_steps),
                "final_result": saga.final_result,
                "compensation_result": saga.compensation_result,
                "error": saga.error
            }

        except Exception as e:
            logger.error(f"Saga execution fehlgeschlagen: {e}")

            # Cleanup bei Fehler
            if saga_id in self._active_sagas:
                saga = self._active_sagas[saga_id]
                saga.status = SagaStatus.FAILED
                saga.error = str(e)
                saga.completed_at = datetime.utcnow()

                self._completed_sagas[saga_id] = saga
                del self._active_sagas[saga_id]

                self._saga_performance_stats["failed_sagas"] += 1

            await self._cleanup_saga_timeout(saga_id)

            return {
                "success": False,
                "saga_id": saga_id,
                "error": str(e)
            }

    async def _execute_saga_steps(
        self,
        saga: SagaTransaction,
        security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Führt Saga-Steps aus."""
        try:
            executed_steps = []

            for step in saga.steps:
                step_result = await self._execute_single_step(step, security_context)

                if step_result["success"]:
                    saga.completed_steps.add(step.step_id)
                    executed_steps.append(step.step_id)

                    # Log Step-Execution
                    self._saga_execution_history[saga.saga_id].append({
                        "step_id": step.step_id,
                        "step_name": step.step_name,
                        "action": "executed",
                        "timestamp": datetime.utcnow().isoformat(),
                        "result": step_result
                    })

                else:
                    # Step fehlgeschlagen
                    step.error = step_result.get("error", "Unknown error")

                    self._saga_execution_history[saga.saga_id].append({
                        "step_id": step.step_id,
                        "step_name": step.step_name,
                        "action": "failed",
                        "timestamp": datetime.utcnow().isoformat(),
                        "error": step.error
                    })

                    return {
                        "success": False,
                        "error": f"Step {step.step_name} failed: {step.error}",
                        "executed_steps": executed_steps,
                        "failed_step": step.step_id
                    }

            return {
                "success": True,
                "executed_steps": executed_steps,
                "total_steps": len(saga.steps)
            }

        except Exception as e:
            logger.error(f"Saga steps execution fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_single_step(
        self,
        step: SagaStep,
        _security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Führt einzelnen Saga-Step aus."""
        try:
            step.status = SagaStatus.RUNNING
            step.started_at = datetime.utcnow()

            # Simuliere Step-Execution (in Realität würde hier der echte Service-Call stattfinden)
            await asyncio.sleep(0.01)  # Simuliere Arbeit

            # Simuliere Success/Failure basierend auf Step-Name
            if "fail" in step.step_name.lower():
                step.status = SagaStatus.FAILED
                step.error = f"Simulated failure for step {step.step_name}"
                return {
                    "success": False,
                    "error": step.error
                }

            step.status = SagaStatus.COMPLETED
            step.completed_at = datetime.utcnow()
            step.result = {
                "step_id": step.step_id,
                "operation": step.operation,
                "result": "success",
                "timestamp": step.completed_at.isoformat()
            }

            return {
                "success": True,
                "result": step.result
            }

        except Exception as e:
            step.status = SagaStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()

            return {
                "success": False,
                "error": str(e)
            }

    async def _compensate_saga(
        self,
        saga: SagaTransaction,
        security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Führt Saga-Compensation aus."""
        try:
            logger.info({
                "event": "saga_compensation_started",
                "saga_id": saga.saga_id,
                "compensation_strategy": saga.compensation_strategy,
                "completed_steps": len(saga.completed_steps)
            })

            # Wähle Compensation-Strategie
            compensation_handler = self._compensation_strategies.get(
                saga.compensation_strategy,
                self._compensate_reverse_order
            )

            compensation_result = await compensation_handler(saga, security_context)

            logger.info({
                "event": "saga_compensation_completed",
                "saga_id": saga.saga_id,
                "compensation_success": compensation_result["success"],
                "compensated_steps": compensation_result.get("compensated_steps", 0)
            })

            return compensation_result

        except Exception as e:
            logger.error(f"Saga compensation fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _compensate_reverse_order(
        self,
        saga: SagaTransaction,
        security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Kompensiert Saga in umgekehrter Reihenfolge."""
        try:
            compensated_steps = []

            # Kompensiere Steps in umgekehrter Reihenfolge
            completed_steps = [step for step in saga.steps if step.step_id in saga.completed_steps]
            completed_steps.reverse()

            for step in completed_steps:
                if step.compensation_operation:
                    compensation_result = await self._execute_compensation_step(step, security_context)

                    if compensation_result["success"]:
                        saga.compensated_steps.add(step.step_id)
                        compensated_steps.append(step.step_id)

                        self._saga_execution_history[saga.saga_id].append({
                            "step_id": step.step_id,
                            "step_name": step.step_name,
                            "action": "compensated",
                            "timestamp": datetime.utcnow().isoformat(),
                            "compensation_result": compensation_result
                        })
                    else:
                        logger.error({
                            "event": "compensation_step_failed",
                            "saga_id": saga.saga_id,
                            "step_id": step.step_id,
                            "error": compensation_result.get("error")
                        })

                        # Bei Compensation-Fehler: Entscheide basierend auf Compensation-Action
                        if step.compensation_action == CompensationAction.ESCALATE:
                            return {
                                "success": False,
                                "error": f"Compensation escalated for step {step.step_name}",
                                "compensated_steps": compensated_steps
                            }
                        # Bei SKIP oder anderen: Fortfahren

            return {
                "success": True,
                "compensated_steps": compensated_steps,
                "total_compensated": len(compensated_steps)
            }

        except Exception as e:
            logger.error(f"Reverse order compensation fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _compensate_parallel(
        self,
        saga: SagaTransaction,
        security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Kompensiert Saga parallel."""
        try:
            completed_steps = [step for step in saga.steps if step.step_id in saga.completed_steps]

            # Erstelle Compensation-Tasks
            compensation_tasks = []
            for step in completed_steps:
                if step.compensation_operation:
                    task = asyncio.create_task(
                        self._execute_compensation_step(step, security_context)
                    )
                    compensation_tasks.append((step, task))

            # Warte auf alle Compensation-Tasks
            compensated_steps = []
            for step, task in compensation_tasks:
                try:
                    compensation_result = await task

                    if compensation_result["success"]:
                        saga.compensated_steps.add(step.step_id)
                        compensated_steps.append(step.step_id)

                except Exception as e:
                    logger.error(f"Parallel compensation step failed: {e}")

            return {
                "success": True,
                "compensated_steps": compensated_steps,
                "total_compensated": len(compensated_steps)
            }

        except Exception as e:
            logger.error(f"Parallel compensation fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _compensate_custom(
        self,
        saga: SagaTransaction,
        security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Kompensiert Saga mit Custom-Strategie."""
        # Fallback zu reverse order
        return await self._compensate_reverse_order(saga, security_context)

    async def _execute_compensation_step(
        self,
        step: SagaStep,
        _security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Führt Compensation-Step aus."""
        try:
            # Simuliere Compensation-Execution
            await asyncio.sleep(0.005)  # Simuliere Compensation-Arbeit

            return {
                "success": True,
                "compensation_operation": step.compensation_operation,
                "step_id": step.step_id
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _start_saga_timeout(self, saga_id: str, timeout_seconds: int) -> None:
        """Startet Saga-Timeout."""
        try:
            async def timeout_handler():
                await asyncio.sleep(timeout_seconds)
                await self._handle_saga_timeout(saga_id)

            timeout_task = asyncio.create_task(timeout_handler())
            self._saga_timeouts[saga_id] = timeout_task

        except Exception as e:
            logger.error(f"Saga timeout start fehlgeschlagen: {e}")

    async def _handle_saga_timeout(self, saga_id: str) -> None:
        """Behandelt Saga-Timeout."""
        try:
            if saga_id in self._active_sagas:
                saga = self._active_sagas[saga_id]
                saga.status = SagaStatus.FAILED
                saga.error = "Saga timeout exceeded"
                saga.completed_at = datetime.utcnow()

                logger.warning({
                    "event": "saga_timeout",
                    "saga_id": saga_id,
                    "saga_name": saga.saga_name
                })

                # Starte Compensation
                await self._compensate_saga(saga, None)

                # Move zu completed sagas
                self._completed_sagas[saga_id] = saga
                del self._active_sagas[saga_id]

                self._saga_performance_stats["failed_sagas"] += 1

        except Exception as e:
            logger.error(f"Saga timeout handling fehlgeschlagen: {e}")

    async def _cleanup_saga_timeout(self, saga_id: str) -> None:
        """Bereinigt Saga-Timeout."""
        try:
            if saga_id in self._saga_timeouts:
                timeout_task = self._saga_timeouts[saga_id]
                timeout_task.cancel()
                del self._saga_timeouts[saga_id]

        except Exception as e:
            logger.error(f"Saga timeout cleanup fehlgeschlagen: {e}")

    async def _saga_monitoring_loop(self) -> None:
        """Background-Loop für Saga-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # Alle 30 Sekunden

                if self._is_running:
                    await self._monitor_active_sagas()

            except Exception as e:
                logger.error(f"Saga monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(30)

    async def _timeout_management_loop(self) -> None:
        """Background-Loop für Timeout-Management."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Jede Minute

                if self._is_running:
                    await self._check_saga_timeouts()

            except Exception as e:
                logger.error(f"Timeout management loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _compensation_monitoring_loop(self) -> None:
        """Background-Loop für Compensation-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(45)  # Alle 45 Sekunden

                if self._is_running:
                    await self._monitor_compensation_status()

            except Exception as e:
                logger.error(f"Compensation monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(45)

    async def _monitor_active_sagas(self) -> None:
        """Monitort aktive Sagas."""
        try:
            for saga_id, saga in self._active_sagas.items():
                # Prüfe Saga-Health
                if saga.status == SagaStatus.RUNNING:
                    runtime = (datetime.utcnow() - saga.started_at).total_seconds() if saga.started_at else 0

                    if runtime > saga.timeout_seconds * 0.8:  # 80% des Timeouts
                        logger.warning({
                            "event": "saga_approaching_timeout",
                            "saga_id": saga_id,
                            "runtime_seconds": runtime,
                            "timeout_seconds": saga.timeout_seconds
                        })

        except Exception as e:
            logger.error(f"Active sagas monitoring fehlgeschlagen: {e}")

    async def _check_saga_timeouts(self) -> None:
        """Prüft Saga-Timeouts."""
        try:
            current_time = datetime.utcnow()

            for saga_id, saga in list(self._active_sagas.items()):
                if saga.started_at:
                    runtime = (current_time - saga.started_at).total_seconds()

                    if runtime > saga.timeout_seconds:
                        await self._handle_saga_timeout(saga_id)

        except Exception as e:
            logger.error(f"Saga timeouts check fehlgeschlagen: {e}")

    async def _monitor_compensation_status(self) -> None:
        """Monitort Compensation-Status."""
        try:
            compensating_sagas = [
                saga for saga in self._active_sagas.values()
                if saga.status == SagaStatus.COMPENSATING
            ]

            for saga in compensating_sagas:
                logger.info({
                    "event": "saga_compensation_in_progress",
                    "saga_id": saga.saga_id,
                    "compensated_steps": len(saga.compensated_steps),
                    "total_completed_steps": len(saga.completed_steps)
                })

        except Exception as e:
            logger.error(f"Compensation status monitoring fehlgeschlagen: {e}")

    def _update_saga_performance_stats(self, execution_time_ms: float) -> None:
        """Aktualisiert Saga-Performance-Statistiken."""
        try:
            self._saga_performance_stats["total_sagas_executed"] += 1

            current_avg = self._saga_performance_stats["avg_saga_execution_time_ms"]
            total_count = self._saga_performance_stats["total_sagas_executed"]
            new_avg = ((current_avg * (total_count - 1)) + execution_time_ms) / total_count
            self._saga_performance_stats["avg_saga_execution_time_ms"] = new_avg

        except Exception as e:
            logger.error(f"Saga performance stats update fehlgeschlagen: {e}")

    def get_saga_status(self, saga_id: str) -> dict[str, Any] | None:
        """Holt Saga-Status.

        Args:
            saga_id: Saga-ID

        Returns:
            Saga-Status oder None
        """
        try:
            saga = self._active_sagas.get(saga_id) or self._completed_sagas.get(saga_id)

            if not saga:
                return None

            return {
                "saga_id": saga.saga_id,
                "saga_name": saga.saga_name,
                "status": saga.status.value,
                "current_step": saga.current_step,
                "completed_steps": len(saga.completed_steps),
                "total_steps": len(saga.steps),
                "compensated_steps": len(saga.compensated_steps),
                "started_at": saga.started_at.isoformat() if saga.started_at else None,
                "completed_at": saga.completed_at.isoformat() if saga.completed_at else None,
                "error": saga.error,
                "execution_history": self._saga_execution_history.get(saga_id, [])
            }

        except Exception as e:
            logger.error(f"Saga status retrieval fehlgeschlagen: {e}")
            return None

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            stats = self._saga_performance_stats.copy()

            # Berechne Success-Rate
            total_sagas = stats["total_sagas_executed"]
            if total_sagas > 0:
                stats["saga_success_rate"] = stats["successful_sagas"] / total_sagas
                stats["saga_compensation_rate"] = stats["compensated_sagas"] / total_sagas
            else:
                stats["saga_success_rate"] = 0.0
                stats["saga_compensation_rate"] = 0.0

            # Active Sagas
            stats["active_sagas"] = len(self._active_sagas)
            stats["completed_sagas"] = len(self._completed_sagas)

            # Compensation-Strategien
            stats["available_compensation_strategies"] = list(self._compensation_strategies.keys())

            return stats

        except Exception as e:
            logger.error(f"Saga performance stats retrieval fehlgeschlagen: {e}")
            return {}
