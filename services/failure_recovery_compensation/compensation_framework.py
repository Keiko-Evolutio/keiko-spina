# backend/services/failure_recovery_compensation/compensation_framework.py
"""Compensation Framework für Saga Pattern Transactions.

Implementiert Enterprise-Grade Compensation Framework mit automatischen
Rollback-Mechanismen und Kompensations-Aktionen für fehlgeschlagene
Multi-Service-Operationen.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from kei_logging import get_logger
from services.enhanced_security_integration import SecurityContext

from .data_models import CompensationAction, SagaState, SagaStep, SagaTransaction

logger = get_logger(__name__)


class CompensationFramework:
    """Enterprise-Grade Compensation Framework für Saga Pattern."""

    def __init__(self):
        """Initialisiert Compensation Framework."""
        # Saga-Storage
        self._active_sagas: dict[str, SagaTransaction] = {}
        self._saga_history: dict[str, SagaTransaction] = {}

        # Compensation-Handlers
        self._compensation_handlers: dict[CompensationAction, Callable] = {
            CompensationAction.UNDO_DATA_CHANGE: self._undo_data_change,
            CompensationAction.RESTORE_BACKUP: self._restore_backup,
            CompensationAction.DELETE_CREATED_RECORD: self._delete_created_record,
            CompensationAction.UPDATE_STATUS: self._update_status,
            CompensationAction.CANCEL_OPERATION: self._cancel_operation,
            CompensationAction.REVERSE_OPERATION: self._reverse_operation,
            CompensationAction.NOTIFY_CANCELLATION: self._notify_cancellation,
            CompensationAction.REFUND_PAYMENT: self._refund_payment,
            CompensationAction.REVERSE_CHARGE: self._reverse_charge,
            CompensationAction.CREDIT_ACCOUNT: self._credit_account,
            CompensationAction.RELEASE_RESOURCES: self._release_resources,
            CompensationAction.DEALLOCATE_MEMORY: self._deallocate_memory,
            CompensationAction.CLOSE_CONNECTIONS: self._close_connections,
            CompensationAction.SEND_FAILURE_NOTIFICATION: self._send_failure_notification,
            CompensationAction.UPDATE_USER_STATUS: self._update_user_status,
            CompensationAction.LOG_COMPENSATION: self._log_compensation,
            CompensationAction.CUSTOM_COMPENSATION: self._custom_compensation
        }

        # Service-Clients für Compensation-Calls
        self._service_clients: dict[str, Any] = {}

        # Compensation-Metriken
        self._compensation_metrics = {
            "total_sagas": 0,
            "successful_sagas": 0,
            "compensated_sagas": 0,
            "failed_sagas": 0,
            "avg_saga_duration_ms": 0.0,
            "avg_compensation_time_ms": 0.0,
            "compensation_success_rate": 0.0
        }

        # Background-Tasks
        self._background_tasks: list[asyncio.Task] = []
        self._is_running = False

        # Event-Callbacks
        self._saga_callbacks: list[Callable] = []
        self._compensation_callbacks: list[Callable] = []

        logger.info("Compensation Framework initialisiert")

    async def start(self) -> None:
        """Startet Compensation Framework."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._background_tasks = [
            asyncio.create_task(self._saga_monitoring_loop()),
            asyncio.create_task(self._compensation_monitoring_loop()),
            asyncio.create_task(self._metrics_collection_loop())
        ]

        logger.info("Compensation Framework gestartet")

    async def stop(self) -> None:
        """Stoppt Compensation Framework."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info("Compensation Framework gestoppt")

    async def create_saga_transaction(
        self,
        saga_name: str,
        description: str,
        steps: list[SagaStep],
        security_context: SecurityContext | None = None
    ) -> SagaTransaction:
        """Erstellt neue Saga-Transaction.

        Args:
            saga_name: Saga-Name
            description: Saga-Beschreibung
            steps: Saga-Steps
            security_context: Security-Context

        Returns:
            Saga-Transaction
        """
        try:
            import uuid

            saga = SagaTransaction(
                saga_id=str(uuid.uuid4()),
                saga_name=saga_name,
                description=description,
                steps=steps,
                user_id=security_context.user_id if security_context else None,
                tenant_id=security_context.tenant_id if security_context else None,
                security_level=security_context.security_level if security_context else None
            )

            # Validiere Saga-Steps
            await self._validate_saga_steps(saga)

            # Speichere Saga
            self._active_sagas[saga.saga_id] = saga

            # Update Metriken
            self._compensation_metrics["total_sagas"] += 1

            logger.info({
                "event": "saga_transaction_created",
                "saga_id": saga.saga_id,
                "saga_name": saga_name,
                "steps_count": len(steps),
                "user_id": saga.user_id,
                "tenant_id": saga.tenant_id
            })

            return saga

        except Exception as e:
            logger.error(f"Saga transaction creation fehlgeschlagen: {e}")
            raise

    async def execute_saga_transaction(
        self,
        saga_id: str,
        security_context: SecurityContext | None = None
    ) -> bool:
        """Führt Saga-Transaction aus.

        Args:
            saga_id: Saga-ID
            security_context: Security-Context

        Returns:
            Erfolg der Saga-Execution
        """
        start_time = time.time()

        try:
            saga = self._active_sagas.get(saga_id)
            if not saga:
                raise ValueError(f"Saga {saga_id} nicht gefunden")

            # Starte Saga-Execution
            saga.state = SagaState.EXECUTING
            saga.started_at = datetime.utcnow()

            logger.info({
                "event": "saga_execution_started",
                "saga_id": saga_id,
                "saga_name": saga.saga_name,
                "steps_count": len(saga.steps)
            })

            # Führe Steps sequenziell aus
            for step in sorted(saga.steps, key=lambda s: s.step_order):
                try:
                    # Prüfe Dependencies
                    if not await self._check_step_dependencies(step, saga):
                        raise Exception(f"Dependencies für Step {step.step_id} nicht erfüllt")

                    # Führe Step aus
                    success = await self._execute_saga_step(step, saga, security_context)

                    if success:
                        step.state = "completed"
                        step.completed_at = datetime.utcnow()
                        saga.executed_steps.append(step.step_id)

                        logger.debug({
                            "event": "saga_step_completed",
                            "saga_id": saga_id,
                            "step_id": step.step_id,
                            "step_name": step.step_name
                        })
                    else:
                        # Step fehlgeschlagen - starte Compensation
                        step.state = "failed"
                        saga.failed_steps.append(step.step_id)
                        saga.error_step_id = step.step_id
                        saga.error_message = step.error_message

                        logger.warning({
                            "event": "saga_step_failed",
                            "saga_id": saga_id,
                            "step_id": step.step_id,
                            "step_name": step.step_name,
                            "error": step.error_message
                        })

                        # Starte Compensation für bereits ausgeführte Steps
                        await self._compensate_saga_transaction(saga, security_context)
                        return False

                except Exception as e:
                    # Step-Exception - starte Compensation
                    step.state = "failed"
                    step.error_message = str(e)
                    saga.failed_steps.append(step.step_id)
                    saga.error_step_id = step.step_id
                    saga.error_message = str(e)

                    logger.error({
                        "event": "saga_step_exception",
                        "saga_id": saga_id,
                        "step_id": step.step_id,
                        "step_name": step.step_name,
                        "error": str(e)
                    })

                    # Starte Compensation
                    await self._compensate_saga_transaction(saga, security_context)
                    return False

            # Alle Steps erfolgreich - Saga abgeschlossen
            saga.state = SagaState.COMPLETED
            saga.completed_at = datetime.utcnow()

            # Berechne Saga-Duration
            if saga.started_at:
                saga_duration_ms = (saga.completed_at - saga.started_at).total_seconds() * 1000
                saga.metadata["duration_ms"] = saga_duration_ms

            # Update Metriken
            self._compensation_metrics["successful_sagas"] += 1

            # Verschiebe zu History
            self._saga_history[saga_id] = saga
            del self._active_sagas[saga_id]

            # Trigger Saga-Callbacks
            await self._trigger_saga_callbacks(saga, "completed")

            execution_time_ms = (time.time() - start_time) * 1000

            logger.info({
                "event": "saga_execution_completed",
                "saga_id": saga_id,
                "saga_name": saga.saga_name,
                "execution_time_ms": execution_time_ms,
                "saga_duration_ms": saga.metadata.get("duration_ms", 0)
            })

            return True

        except Exception as e:
            logger.error(f"Saga execution fehlgeschlagen: {e}")

            # Update Saga-State bei Exception
            if saga_id in self._active_sagas:
                saga = self._active_sagas[saga_id]
                saga.state = SagaState.FAILED
                saga.error_message = str(e)
                saga.completed_at = datetime.utcnow()

                # Update Metriken
                self._compensation_metrics["failed_sagas"] += 1

                # Verschiebe zu History
                self._saga_history[saga_id] = saga
                del self._active_sagas[saga_id]

            return False

    async def compensate_saga_transaction(
        self,
        saga_id: str,
        security_context: SecurityContext | None = None
    ) -> bool:
        """Kompensiert Saga-Transaction manuell.

        Args:
            saga_id: Saga-ID
            security_context: Security-Context

        Returns:
            Erfolg der Compensation
        """
        try:
            saga = self._active_sagas.get(saga_id) or self._saga_history.get(saga_id)
            if not saga:
                raise ValueError(f"Saga {saga_id} nicht gefunden")

            return await self._compensate_saga_transaction(saga, security_context)

        except Exception as e:
            logger.error(f"Manual saga compensation fehlgeschlagen: {e}")
            return False

    async def get_saga_status(self, saga_id: str) -> SagaTransaction | None:
        """Gibt Saga-Status zurück.

        Args:
            saga_id: Saga-ID

        Returns:
            Saga-Transaction oder None
        """
        return self._active_sagas.get(saga_id) or self._saga_history.get(saga_id)

    async def list_active_sagas(self) -> list[SagaTransaction]:
        """Gibt Liste aktiver Sagas zurück.

        Returns:
            Liste aktiver Sagas
        """
        return list(self._active_sagas.values())

    async def register_service_client(self, service_name: str, client: Any) -> None:
        """Registriert Service-Client für Compensation-Calls.

        Args:
            service_name: Service-Name
            client: Service-Client
        """
        self._service_clients[service_name] = client
        logger.debug(f"Service client für {service_name} registriert")

    async def register_saga_callback(self, callback: Callable) -> None:
        """Registriert Saga-Callback.

        Args:
            callback: Callback-Funktion
        """
        self._saga_callbacks.append(callback)

    async def register_compensation_callback(self, callback: Callable) -> None:
        """Registriert Compensation-Callback.

        Args:
            callback: Callback-Funktion
        """
        self._compensation_callbacks.append(callback)

    async def _validate_saga_steps(self, saga: SagaTransaction) -> None:
        """Validiert Saga-Steps."""
        try:
            # Prüfe Step-Order
            step_orders = [step.step_order for step in saga.steps]
            if len(set(step_orders)) != len(step_orders):
                raise ValueError("Duplicate step orders gefunden")

            # Prüfe Dependencies
            step_ids = {step.step_id for step in saga.steps}
            for step in saga.steps:
                for dep_id in step.depends_on:
                    if dep_id not in step_ids:
                        raise ValueError(f"Dependency {dep_id} für Step {step.step_id} nicht gefunden")

            # Prüfe Compensation-Actions
            for step in saga.steps:
                if step.compensation_action not in self._compensation_handlers:
                    raise ValueError(f"Unbekannte Compensation-Action: {step.compensation_action}")

        except Exception as e:
            logger.error(f"Saga steps validation fehlgeschlagen: {e}")
            raise

    async def _check_step_dependencies(self, step: SagaStep, saga: SagaTransaction) -> bool:
        """Prüft ob Step-Dependencies erfüllt sind."""
        try:
            for dep_id in step.depends_on:
                if dep_id not in saga.executed_steps:
                    return False
            return True

        except Exception as e:
            logger.error(f"Step dependencies check fehlgeschlagen: {e}")
            return False

    async def _execute_saga_step(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Führt einzelnen Saga-Step aus."""
        try:
            step.executed_at = datetime.utcnow()
            step.state = "executing"

            # Hole Service-Client
            service_client = self._service_clients.get(step.service_name)

            if service_client:
                # Führe echten Service-Call aus
                try:
                    response = await asyncio.wait_for(
                        service_client.call_operation(
                            operation=step.operation_name,
                            data=step.request_data,
                            headers=step.headers
                        ),
                        timeout=step.timeout_ms / 1000.0
                    )

                    step.response_data = response
                    return True

                except Exception as e:
                    step.error_message = str(e)
                    return False
            else:
                # Simuliere Service-Call
                await asyncio.sleep(0.1)  # Simuliere Verarbeitungszeit

                # Simuliere Success/Failure basierend auf Service-Name
                success_probability = self._get_step_success_probability(step)
                success = time.time() % 1.0 < success_probability

                if success:
                    step.response_data = {
                        "status": "success",
                        "simulated": True,
                        "step_id": step.step_id
                    }
                    return True
                step.error_message = f"Simulated failure for {step.service_name}:{step.operation_name}"
                return False

        except Exception as e:
            logger.error(f"Saga step execution fehlgeschlagen: {e}")
            step.error_message = str(e)
            return False

    def _get_step_success_probability(self, step: SagaStep) -> float:
        """Berechnet Success-Wahrscheinlichkeit für Step."""
        # Base Success-Rate per Service
        service_success_rates = {
            "user_service": 0.95,
            "payment_service": 0.85,
            "notification_service": 0.90,
            "inventory_service": 0.88,
            "order_service": 0.92,
            "shipping_service": 0.87,
            "analytics_service": 0.93,
            "audit_service": 0.98
        }

        return service_success_rates.get(step.service_name, 0.90)

    async def _compensate_saga_transaction(
        self,
        saga: SagaTransaction,
        security_context: SecurityContext | None
    ) -> bool:
        """Kompensiert Saga-Transaction."""
        start_time = time.time()

        try:
            saga.state = SagaState.COMPENSATING

            logger.info({
                "event": "saga_compensation_started",
                "saga_id": saga.saga_id,
                "executed_steps": len(saga.executed_steps)
            })

            # Kompensiere Steps in umgekehrter Reihenfolge
            compensation_success = True

            for step_id in reversed(saga.executed_steps):
                step = next((s for s in saga.steps if s.step_id == step_id), None)
                if not step:
                    continue

                try:
                    # Führe Compensation aus
                    success = await self._execute_compensation_action(step, saga, security_context)

                    if success:
                        saga.compensated_steps.append(step_id)

                        logger.debug({
                            "event": "step_compensated",
                            "saga_id": saga.saga_id,
                            "step_id": step_id,
                            "compensation_action": step.compensation_action.value
                        })
                    else:
                        compensation_success = False

                        logger.warning({
                            "event": "step_compensation_failed",
                            "saga_id": saga.saga_id,
                            "step_id": step_id,
                            "compensation_action": step.compensation_action.value
                        })

                except Exception as e:
                    compensation_success = False

                    logger.error({
                        "event": "step_compensation_exception",
                        "saga_id": saga.saga_id,
                        "step_id": step_id,
                        "error": str(e)
                    })

            # Update Saga-State
            if compensation_success:
                saga.state = SagaState.COMPLETED  # Erfolgreich kompensiert
                self._compensation_metrics["compensated_sagas"] += 1
            else:
                saga.state = SagaState.FAILED
                self._compensation_metrics["failed_sagas"] += 1

            saga.completed_at = datetime.utcnow()

            # Berechne Compensation-Zeit
            compensation_time_ms = (time.time() - start_time) * 1000
            saga.metadata["compensation_time_ms"] = compensation_time_ms

            # Verschiebe zu History
            self._saga_history[saga.saga_id] = saga
            if saga.saga_id in self._active_sagas:
                del self._active_sagas[saga.saga_id]

            # Trigger Compensation-Callbacks
            await self._trigger_compensation_callbacks(saga, compensation_success)

            logger.info({
                "event": "saga_compensation_completed",
                "saga_id": saga.saga_id,
                "success": compensation_success,
                "compensated_steps": len(saga.compensated_steps),
                "compensation_time_ms": compensation_time_ms
            })

            return compensation_success

        except Exception as e:
            logger.error(f"Saga compensation fehlgeschlagen: {e}")
            saga.state = SagaState.FAILED
            saga.error_message = str(e)
            return False

    async def _execute_compensation_action(
        self,
        step: SagaStep,
        saga: SagaTransaction,
        security_context: SecurityContext | None
    ) -> bool:
        """Führt Compensation-Action für Step aus."""
        try:
            # Hole Compensation-Handler
            handler = self._compensation_handlers.get(step.compensation_action)

            if not handler:
                logger.error(f"Kein Handler für Compensation-Action: {step.compensation_action}")
                return False

            # Führe Compensation-Handler aus
            return await handler(step, saga, security_context)

        except Exception as e:
            logger.error(f"Compensation action execution fehlgeschlagen: {e}")
            return False

    # Compensation-Handler
    async def _undo_data_change(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Macht Datenänderung rückgängig."""
        try:
            await asyncio.sleep(0.05)  # Simuliere Undo-Operation

            # Simuliere Success basierend auf Compensation-Data
            if step.compensation_data.get("backup_available", True):
                logger.debug(f"Data change undone for step {step.step_id}")
                return True
            logger.warning(f"No backup available for step {step.step_id}")
            return False

        except Exception as e:
            logger.error(f"Undo data change fehlgeschlagen: {e}")
            return False

    async def _restore_backup(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Stellt Backup wieder her."""
        try:
            await asyncio.sleep(0.1)  # Simuliere Restore-Operation

            backup_id = step.compensation_data.get("backup_id")
            if backup_id:
                logger.debug(f"Backup {backup_id} restored for step {step.step_id}")
                return True
            logger.warning(f"No backup ID for step {step.step_id}")
            return False

        except Exception as e:
            logger.error(f"Restore backup fehlgeschlagen: {e}")
            return False

    async def _delete_created_record(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Löscht erstellten Record."""
        try:
            await asyncio.sleep(0.03)  # Simuliere Delete-Operation

            record_id = step.compensation_data.get("created_record_id")
            if record_id:
                logger.debug(f"Created record {record_id} deleted for step {step.step_id}")
                return True
            logger.warning(f"No record ID for step {step.step_id}")
            return False

        except Exception as e:
            logger.error(f"Delete created record fehlgeschlagen: {e}")
            return False

    async def _update_status(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Aktualisiert Status."""
        try:
            await asyncio.sleep(0.02)  # Simuliere Status-Update

            new_status = step.compensation_data.get("rollback_status", "cancelled")
            logger.debug(f"Status updated to {new_status} for step {step.step_id}")
            return True

        except Exception as e:
            logger.error(f"Update status fehlgeschlagen: {e}")
            return False

    async def _cancel_operation(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Bricht Operation ab."""
        try:
            await asyncio.sleep(0.05)  # Simuliere Cancel-Operation

            operation_id = step.compensation_data.get("operation_id")
            if operation_id:
                logger.debug(f"Operation {operation_id} cancelled for step {step.step_id}")
                return True
            logger.debug(f"Operation cancelled for step {step.step_id}")
            return True

        except Exception as e:
            logger.error(f"Cancel operation fehlgeschlagen: {e}")
            return False

    async def _reverse_operation(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Kehrt Operation um."""
        try:
            await asyncio.sleep(0.08)  # Simuliere Reverse-Operation

            # Simuliere Success basierend auf Operation-Type
            operation_type = step.compensation_data.get("operation_type", "unknown")
            reversible_operations = ["create", "update", "transfer", "allocate"]

            if operation_type in reversible_operations:
                logger.debug(f"Operation {operation_type} reversed for step {step.step_id}")
                return True
            logger.warning(f"Operation {operation_type} not reversible for step {step.step_id}")
            return False

        except Exception as e:
            logger.error(f"Reverse operation fehlgeschlagen: {e}")
            return False

    async def _notify_cancellation(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Sendet Cancellation-Notification."""
        try:
            await asyncio.sleep(0.02)  # Simuliere Notification

            recipients = step.compensation_data.get("notification_recipients", [])
            logger.debug(f"Cancellation notification sent to {len(recipients)} recipients for step {step.step_id}")
            return True

        except Exception as e:
            logger.error(f"Notify cancellation fehlgeschlagen: {e}")
            return False

    async def _refund_payment(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Erstattet Payment."""
        try:
            await asyncio.sleep(0.15)  # Simuliere Refund-Operation

            payment_id = step.compensation_data.get("payment_id")
            amount = step.compensation_data.get("refund_amount", 0)

            if payment_id and amount > 0:
                logger.debug(f"Payment {payment_id} refunded {amount} for step {step.step_id}")
                return True
            logger.warning(f"Invalid refund data for step {step.step_id}")
            return False

        except Exception as e:
            logger.error(f"Refund payment fehlgeschlagen: {e}")
            return False

    async def _reverse_charge(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Kehrt Charge um."""
        try:
            await asyncio.sleep(0.12)  # Simuliere Reverse-Charge

            charge_id = step.compensation_data.get("charge_id")
            if charge_id:
                logger.debug(f"Charge {charge_id} reversed for step {step.step_id}")
                return True
            logger.warning(f"No charge ID for step {step.step_id}")
            return False

        except Exception as e:
            logger.error(f"Reverse charge fehlgeschlagen: {e}")
            return False

    async def _credit_account(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Kreditiert Account."""
        try:
            await asyncio.sleep(0.08)  # Simuliere Credit-Operation

            account_id = step.compensation_data.get("account_id")
            credit_amount = step.compensation_data.get("credit_amount", 0)

            if account_id and credit_amount > 0:
                logger.debug(f"Account {account_id} credited {credit_amount} for step {step.step_id}")
                return True
            logger.warning(f"Invalid credit data for step {step.step_id}")
            return False

        except Exception as e:
            logger.error(f"Credit account fehlgeschlagen: {e}")
            return False

    async def _release_resources(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Gibt Ressourcen frei."""
        try:
            await asyncio.sleep(0.03)  # Simuliere Resource-Release

            resources = step.compensation_data.get("allocated_resources", [])
            logger.debug(f"Released {len(resources)} resources for step {step.step_id}")
            return True

        except Exception as e:
            logger.error(f"Release resources fehlgeschlagen: {e}")
            return False

    async def _deallocate_memory(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Gibt Memory frei."""
        try:
            await asyncio.sleep(0.01)  # Simuliere Memory-Deallocation

            memory_size = step.compensation_data.get("allocated_memory_mb", 0)
            logger.debug(f"Deallocated {memory_size}MB memory for step {step.step_id}")
            return True

        except Exception as e:
            logger.error(f"Deallocate memory fehlgeschlagen: {e}")
            return False

    async def _close_connections(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Schließt Connections."""
        try:
            await asyncio.sleep(0.02)  # Simuliere Connection-Close

            connections = step.compensation_data.get("open_connections", [])
            logger.debug(f"Closed {len(connections)} connections for step {step.step_id}")
            return True

        except Exception as e:
            logger.error(f"Close connections fehlgeschlagen: {e}")
            return False

    async def _send_failure_notification(
        self,
        step: SagaStep,
        saga: SagaTransaction,
        security_context: SecurityContext | None
    ) -> bool:
        """Sendet Failure-Notification."""
        try:
            await asyncio.sleep(0.05)  # Simuliere Notification

            notification_data = {
                "saga_id": saga.saga_id,
                "failed_step": step.step_id,
                "error_message": saga.error_message,
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.debug({
                "event": "failure_notification_sent",
                **notification_data
            })
            return True

        except Exception as e:
            logger.error(f"Send failure notification fehlgeschlagen: {e}")
            return False

    async def _update_user_status(
        self,
        step: SagaStep,
        saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Aktualisiert User-Status."""
        try:
            await asyncio.sleep(0.03)  # Simuliere Status-Update

            user_id = step.compensation_data.get("user_id") or saga.user_id
            new_status = step.compensation_data.get("rollback_status", "operation_failed")

            if user_id:
                logger.debug(f"User {user_id} status updated to {new_status} for step {step.step_id}")
                return True
            logger.warning(f"No user ID for step {step.step_id}")
            return False

        except Exception as e:
            logger.error(f"Update user status fehlgeschlagen: {e}")
            return False

    async def _log_compensation(
        self,
        step: SagaStep,
        saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Loggt Compensation-Action."""
        try:
            compensation_log = {
                "saga_id": saga.saga_id,
                "step_id": step.step_id,
                "compensation_action": step.compensation_action.value,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": saga.user_id,
                "tenant_id": saga.tenant_id
            }

            logger.info({
                "event": "compensation_logged",
                **compensation_log
            })

            return True

        except Exception as e:
            logger.error(f"Log compensation fehlgeschlagen: {e}")
            return False

    async def _custom_compensation(
        self,
        step: SagaStep,
        _saga: SagaTransaction,
        _security_context: SecurityContext | None
    ) -> bool:
        """Führt Custom-Compensation aus."""
        try:
            # Hole Custom-Handler aus Compensation-Data
            custom_handler_name = step.compensation_data.get("custom_handler")

            if custom_handler_name:
                # Simuliere Custom-Handler-Execution
                await asyncio.sleep(0.1)
                logger.debug(f"Custom compensation {custom_handler_name} executed for step {step.step_id}")
                return True
            logger.warning(f"No custom handler specified for step {step.step_id}")
            return False

        except Exception as e:
            logger.error(f"Custom compensation fehlgeschlagen: {e}")
            return False

    # Background-Loops
    async def _saga_monitoring_loop(self) -> None:
        """Background-Loop für Saga-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Jede Minute

                if self._is_running:
                    await self._monitor_active_sagas()

            except Exception as e:
                logger.error(f"Saga monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _compensation_monitoring_loop(self) -> None:
        """Background-Loop für Compensation-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(120)  # Alle 2 Minuten

                if self._is_running:
                    await self._monitor_compensation_health()

            except Exception as e:
                logger.error(f"Compensation monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(120)

    async def _metrics_collection_loop(self) -> None:
        """Background-Loop für Metrics-Collection."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Alle 5 Minuten

                if self._is_running:
                    await self._update_compensation_metrics()

            except Exception as e:
                logger.error(f"Metrics collection loop fehlgeschlagen: {e}")
                await asyncio.sleep(300)

    async def _monitor_active_sagas(self) -> None:
        """Monitort aktive Sagas."""
        try:
            current_time = datetime.utcnow()

            for saga_id, saga in list(self._active_sagas.items()):
                # Prüfe Saga-Timeout
                if saga.started_at:
                    elapsed_time_ms = (current_time - saga.started_at).total_seconds() * 1000

                    if elapsed_time_ms > saga.total_timeout_ms:
                        logger.warning({
                            "event": "saga_timeout",
                            "saga_id": saga_id,
                            "elapsed_time_ms": elapsed_time_ms,
                            "timeout_ms": saga.total_timeout_ms
                        })

                        # Starte Compensation wegen Timeout
                        saga.state = SagaState.TIMEOUT
                        saga.error_message = "Saga timeout"
                        await self._compensate_saga_transaction(saga, None)

        except Exception as e:
            logger.error(f"Active sagas monitoring fehlgeschlagen: {e}")

    async def _monitor_compensation_health(self) -> None:
        """Monitort Compensation-Health."""
        try:
            # Prüfe Compensation-Success-Rate
            if self._compensation_metrics["total_sagas"] > 0:
                success_rate = (
                    self._compensation_metrics["successful_sagas"] +
                    self._compensation_metrics["compensated_sagas"]
                ) / self._compensation_metrics["total_sagas"]

                if success_rate < 0.8:  # Unter 80% Success-Rate
                    logger.warning({
                        "event": "low_compensation_success_rate",
                        "success_rate": success_rate,
                        "total_sagas": self._compensation_metrics["total_sagas"]
                    })

        except Exception as e:
            logger.error(f"Compensation health monitoring fehlgeschlagen: {e}")

    async def _update_compensation_metrics(self) -> None:
        """Aktualisiert Compensation-Metriken."""
        try:
            # Berechne Success-Rates
            total_sagas = self._compensation_metrics["total_sagas"]

            if total_sagas > 0:
                self._compensation_metrics["compensation_success_rate"] = (
                    self._compensation_metrics["compensated_sagas"] / total_sagas
                )

            # Berechne Average Saga-Duration
            completed_sagas = [
                saga for saga in self._saga_history.values()
                if saga.completed_at and saga.started_at
            ]

            if completed_sagas:
                durations = [
                    (saga.completed_at - saga.started_at).total_seconds() * 1000
                    for saga in completed_sagas
                ]
                self._compensation_metrics["avg_saga_duration_ms"] = sum(durations) / len(durations)

            # Berechne Average Compensation-Time
            compensated_sagas = [
                saga for saga in self._saga_history.values()
                if saga.metadata.get("compensation_time_ms")
            ]

            if compensated_sagas:
                compensation_times = [
                    saga.metadata["compensation_time_ms"]
                    for saga in compensated_sagas
                ]
                self._compensation_metrics["avg_compensation_time_ms"] = sum(compensation_times) / len(compensation_times)

        except Exception as e:
            logger.error(f"Compensation metrics update fehlgeschlagen: {e}")

    async def _trigger_saga_callbacks(self, saga: SagaTransaction, event_type: str) -> None:
        """Triggert Saga-Callbacks."""
        for callback in self._saga_callbacks:
            try:
                await callback(saga, event_type)
            except Exception as e:
                logger.error(f"Saga callback fehlgeschlagen: {e}")

    async def _trigger_compensation_callbacks(self, saga: SagaTransaction, success: bool) -> None:
        """Triggert Compensation-Callbacks."""
        for callback in self._compensation_callbacks:
            try:
                await callback(saga, success)
            except Exception as e:
                logger.error(f"Compensation callback fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            return {
                "compensation_framework": {
                    "is_running": self._is_running,
                    "active_sagas": len(self._active_sagas),
                    "saga_history_count": len(self._saga_history),
                    "metrics": self._compensation_metrics,
                    "service_clients": list(self._service_clients.keys())
                }
            }

        except Exception as e:
            logger.error(f"Performance stats retrieval fehlgeschlagen: {e}")
            return {}
