# backend/services/failure_recovery_compensation/failure_recovery_integration.py
"""Failure Recovery & Compensation Integration Layer.

Integriert Failure Recovery System und Compensation Framework mit allen
Enhanced Services für Enterprise-Grade Resilience und Observability.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from kei_logging import get_logger
from services.enhanced_security_integration import SecurityContext

from .compensation_framework import CompensationFramework
from .data_models import (
    DistributedSystemHealth,
    FailureContext,
    FailureType,
    RecoveryConfiguration,
    RecoveryStrategy,
    SagaStep,
    SagaTransaction,
)
from .failure_recovery_system import FailureRecoverySystem

logger = get_logger(__name__)


class FailureRecoveryIntegration:
    """Enterprise-Grade Failure Recovery & Compensation Integration."""

    def __init__(
        self,
        monitoring_engine=None,
        dependency_resolution_engine=None,
        quota_management_engine=None,
        security_integration_engine=None,
        performance_analytics_engine=None,
        real_time_monitoring_engine=None
    ):
        """Initialisiert Failure Recovery Integration.

        Args:
            monitoring_engine: Real-time Monitoring Engine
            dependency_resolution_engine: Dependency Resolution Engine
            quota_management_engine: Quota Management Engine
            security_integration_engine: Security Integration Engine
            performance_analytics_engine: Performance Analytics Engine
            real_time_monitoring_engine: Real-time Monitoring Engine
        """
        # Enhanced Services Integration
        self.monitoring_engine = monitoring_engine
        self.dependency_resolution_engine = dependency_resolution_engine
        self.quota_management_engine = quota_management_engine
        self.security_integration_engine = security_integration_engine
        self.performance_analytics_engine = performance_analytics_engine
        self.real_time_monitoring_engine = real_time_monitoring_engine

        # Core Components
        self.failure_recovery_system = FailureRecoverySystem()
        self.compensation_framework = CompensationFramework()

        # Integration State
        self._is_running = False
        self._integration_metrics = {
            "total_integrations": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "avg_integration_time_ms": 0.0
        }

        # Service Registry
        self._service_registry: dict[str, Any] = {}

        logger.info("Failure Recovery Integration initialisiert")

    async def start(self) -> None:
        """Startet Failure Recovery Integration."""
        if self._is_running:
            return

        try:
            # Starte Core Components
            await self.failure_recovery_system.start()
            await self.compensation_framework.start()

            # Registriere Enhanced Services Integration
            await self._register_enhanced_services_integration()

            # Registriere Default Recovery-Konfigurationen
            await self._register_default_recovery_configurations()

            # Registriere Service-Clients für Compensation
            await self._register_service_clients()

            # Registriere Event-Callbacks
            await self._register_event_callbacks()

            self._is_running = True

            logger.info("Failure Recovery Integration gestartet")

        except Exception as e:
            logger.error(f"Failure Recovery Integration start fehlgeschlagen: {e}")
            raise

    async def stop(self) -> None:
        """Stoppt Failure Recovery Integration."""
        self._is_running = False

        try:
            # Stoppe Core Components
            await self.failure_recovery_system.stop()
            await self.compensation_framework.stop()

            logger.info("Failure Recovery Integration gestoppt")

        except Exception as e:
            logger.error(f"Failure Recovery Integration stop fehlgeschlagen: {e}")

    async def execute_with_failure_recovery(
        self,
        operation: Any,
        service_name: str,
        operation_name: str,
        security_context: SecurityContext | None = None,
        *args,
        **kwargs
    ) -> Any:
        """Führt Operation mit Failure Recovery aus.

        Args:
            operation: Auszuführende Operation
            service_name: Service-Name
            operation_name: Operation-Name
            security_context: Security-Context
            *args: Operation-Argumente
            **kwargs: Operation-Keyword-Argumente

        Returns:
            Operation-Result
        """
        start_time = time.time()

        try:
            # Führe Operation mit Failure Recovery aus
            result = await self.failure_recovery_system.execute_with_recovery(
                operation=operation,
                service_name=service_name,
                operation_name=operation_name,
                security_context=security_context,
                *args,
                **kwargs
            )

            # Update Integration-Metriken
            self._integration_metrics["total_integrations"] += 1
            self._integration_metrics["successful_integrations"] += 1

            # Trigger Performance Analytics
            if self.performance_analytics_engine:
                await self._trigger_performance_analytics(
                    service_name=service_name,
                    operation_name=operation_name,
                    response_time_ms=(time.time() - start_time) * 1000,
                    success=True,
                    security_context=security_context
                )

            return result

        except Exception as e:
            # Update Integration-Metriken
            self._integration_metrics["total_integrations"] += 1
            self._integration_metrics["failed_integrations"] += 1

            # Trigger Performance Analytics
            if self.performance_analytics_engine:
                await self._trigger_performance_analytics(
                    service_name=service_name,
                    operation_name=operation_name,
                    response_time_ms=(time.time() - start_time) * 1000,
                    success=False,
                    security_context=security_context
                )

            logger.error(f"Operation mit Failure Recovery fehlgeschlagen: {e}")
            raise

    async def execute_saga_transaction(
        self,
        saga_name: str,
        description: str,
        steps: list[SagaStep],
        security_context: SecurityContext | None = None
    ) -> bool:
        """Führt Saga-Transaction mit Compensation aus.

        Args:
            saga_name: Saga-Name
            description: Saga-Beschreibung
            steps: Saga-Steps
            security_context: Security-Context

        Returns:
            Erfolg der Saga-Execution
        """
        try:
            # Erstelle Saga-Transaction
            saga = await self.compensation_framework.create_saga_transaction(
                saga_name=saga_name,
                description=description,
                steps=steps,
                security_context=security_context
            )

            # Führe Saga-Transaction aus
            success = await self.compensation_framework.execute_saga_transaction(
                saga_id=saga.saga_id,
                security_context=security_context
            )

            # Trigger Enhanced Services Integration
            await self._trigger_saga_integration_events(saga, success, security_context)

            return success

        except Exception as e:
            logger.error(f"Saga transaction execution fehlgeschlagen: {e}")
            return False

    async def get_distributed_system_health(self) -> DistributedSystemHealth:
        """Gibt Distributed System Health zurück.

        Returns:
            Distributed System Health
        """
        try:
            # Hole System-Health vom Failure Recovery System
            system_health = await self.failure_recovery_system.get_system_health()

            # Erweitere mit Enhanced Services Health
            if self.real_time_monitoring_engine:
                monitoring_health = await self._get_monitoring_health()
                system_health.service_health["real_time_monitoring"] = monitoring_health

            if self.security_integration_engine:
                security_health = await self._get_security_health()
                system_health.service_health["security_integration"] = security_health

            if self.performance_analytics_engine:
                analytics_health = await self._get_analytics_health()
                system_health.service_health["performance_analytics"] = analytics_health

            return system_health

        except Exception as e:
            logger.error(f"Distributed system health retrieval fehlgeschlagen: {e}")
            raise

    async def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Gibt umfassende Metriken zurück.

        Returns:
            Comprehensive Metrics
        """
        try:
            # Hole Recovery-Metriken
            recovery_metrics = await self.failure_recovery_system.get_recovery_metrics()

            # Hole Compensation-Metriken
            compensation_stats = self.compensation_framework.get_performance_stats()

            # Hole Integration-Metriken
            integration_metrics = self._integration_metrics.copy()

            # Berechne Integration-Success-Rate
            if integration_metrics["total_integrations"] > 0:
                integration_metrics["integration_success_rate"] = (
                    integration_metrics["successful_integrations"] /
                    integration_metrics["total_integrations"]
                )
            else:
                integration_metrics["integration_success_rate"] = 0.0

            return {
                "failure_recovery_metrics": {
                    "total_failures": recovery_metrics.total_failures,
                    "total_recovery_attempts": recovery_metrics.total_recovery_attempts,
                    "recovery_success_rate": recovery_metrics.recovery_success_rate,
                    "avg_recovery_time_ms": recovery_metrics.avg_recovery_time_ms,
                    "system_availability": recovery_metrics.system_availability,
                    "failures_by_type": recovery_metrics.failures_by_type,
                    "failures_by_service": recovery_metrics.failures_by_service
                },
                "compensation_metrics": compensation_stats["compensation_framework"],
                "integration_metrics": integration_metrics,
                "enhanced_services_integration": {
                    "monitoring_engine": self.monitoring_engine is not None,
                    "dependency_resolution_engine": self.dependency_resolution_engine is not None,
                    "quota_management_engine": self.quota_management_engine is not None,
                    "security_integration_engine": self.security_integration_engine is not None,
                    "performance_analytics_engine": self.performance_analytics_engine is not None,
                    "real_time_monitoring_engine": self.real_time_monitoring_engine is not None
                }
            }

        except Exception as e:
            logger.error(f"Comprehensive metrics retrieval fehlgeschlagen: {e}")
            return {}

    async def register_service(self, service_name: str, service_client: Any) -> None:
        """Registriert Service für Integration.

        Args:
            service_name: Service-Name
            service_client: Service-Client
        """
        try:
            self._service_registry[service_name] = service_client

            # Registriere Service-Client für Compensation
            await self.compensation_framework.register_service_client(
                service_name=service_name,
                client=service_client
            )

            logger.info(f"Service {service_name} für Integration registriert")

        except Exception as e:
            logger.error(f"Service registration fehlgeschlagen: {e}")
            raise

    async def _register_enhanced_services_integration(self) -> None:
        """Registriert Enhanced Services Integration."""
        try:
            # Registriere Failure-Callbacks für Enhanced Services
            if self.performance_analytics_engine:
                await self.failure_recovery_system.register_failure_callback(
                    self._performance_analytics_failure_callback
                )

            if self.real_time_monitoring_engine:
                await self.failure_recovery_system.register_failure_callback(
                    self._monitoring_failure_callback
                )

            # Registriere Recovery-Callbacks
            if self.performance_analytics_engine:
                await self.failure_recovery_system.register_recovery_callback(
                    self._performance_analytics_recovery_callback
                )

            # Registriere Saga-Callbacks
            if self.performance_analytics_engine:
                await self.compensation_framework.register_saga_callback(
                    self._performance_analytics_saga_callback
                )

            logger.debug("Enhanced Services Integration registriert")

        except Exception as e:
            logger.error(f"Enhanced Services Integration registration fehlgeschlagen: {e}")

    async def _register_default_recovery_configurations(self) -> None:
        """Registriert Default Recovery-Konfigurationen."""
        try:
            # Standard-Services mit Recovery-Konfigurationen
            default_configs = [
                {
                    "service_name": "user_service",
                    "operation_name": "get_user_profile",
                    "primary_strategy": RecoveryStrategy.EXPONENTIAL_BACKOFF,
                    "fallback_strategies": [RecoveryStrategy.CACHED_RESPONSE, RecoveryStrategy.DEFAULT_RESPONSE]
                },
                {
                    "service_name": "payment_service",
                    "operation_name": "process_payment",
                    "primary_strategy": RecoveryStrategy.CIRCUIT_BREAKER,
                    "fallback_strategies": [RecoveryStrategy.FALLBACK_SERVICE]
                },
                {
                    "service_name": "notification_service",
                    "operation_name": "send_notification",
                    "primary_strategy": RecoveryStrategy.LINEAR_BACKOFF,
                    "fallback_strategies": [RecoveryStrategy.DEGRADED_SERVICE]
                },
                {
                    "service_name": "analytics_service",
                    "operation_name": "collect_metrics",
                    "primary_strategy": RecoveryStrategy.IMMEDIATE_RETRY,
                    "fallback_strategies": [RecoveryStrategy.CACHED_RESPONSE]
                }
            ]

            for config_data in default_configs:
                import uuid

                config = RecoveryConfiguration(
                    config_id=str(uuid.uuid4()),
                    service_name=config_data["service_name"],
                    operation_name=config_data["operation_name"],
                    primary_strategy=config_data["primary_strategy"],
                    fallback_strategies=config_data["fallback_strategies"],
                    max_retry_attempts=3,
                    initial_retry_delay_ms=1000,
                    max_retry_delay_ms=30000,
                    retry_multiplier=2.0,
                    retry_jitter=True,
                    failure_threshold=5,
                    success_threshold=3,
                    circuit_timeout_ms=60000
                )

                await self.failure_recovery_system.register_recovery_configuration(config)

            logger.debug("Default Recovery-Konfigurationen registriert")

        except Exception as e:
            logger.error(f"Default Recovery-Konfigurationen registration fehlgeschlagen: {e}")

    async def _register_service_clients(self) -> None:
        """Registriert Service-Clients für Compensation."""
        try:
            # Registriere Enhanced Services als Service-Clients
            if self.monitoring_engine:
                await self.compensation_framework.register_service_client(
                    "monitoring_service",
                    self.monitoring_engine
                )

            if self.security_integration_engine:
                await self.compensation_framework.register_service_client(
                    "security_service",
                    self.security_integration_engine
                )

            if self.performance_analytics_engine:
                await self.compensation_framework.register_service_client(
                    "analytics_service",
                    self.performance_analytics_engine
                )

            logger.debug("Service-Clients für Compensation registriert")

        except Exception as e:
            logger.error(f"Service-Clients registration fehlgeschlagen: {e}")

    async def _register_event_callbacks(self) -> None:
        """Registriert Event-Callbacks."""
        try:
            # Registriere Compensation-Callbacks
            await self.compensation_framework.register_compensation_callback(
                self._compensation_completed_callback
            )

            logger.debug("Event-Callbacks registriert")

        except Exception as e:
            logger.error(f"Event-Callbacks registration fehlgeschlagen: {e}")

    # Event-Callbacks
    async def _performance_analytics_failure_callback(
        self,
        failure_context: FailureContext,
        _recovery_attempt: Any
    ) -> None:
        """Performance Analytics Failure-Callback."""
        try:
            if self.performance_analytics_engine:
                # Erstelle Performance-Datenpunkt für Failure
                await self.performance_analytics_engine.collect_performance_data_point({
                    "data_point_id": f"failure_{failure_context.failure_id}",
                    "metric_name": "failure_event",
                    "scope": "SERVICE",
                    "scope_id": failure_context.service_name,
                    "value": 1.0,
                    "unit": "count",
                    "dimensions": {
                        "SERVICE": failure_context.service_name,
                        "OPERATION": failure_context.operation_name,
                        "FAILURE_TYPE": failure_context.failure_type.value
                    },
                    "service_name": failure_context.service_name,
                    "user_id": failure_context.user_id,
                    "tenant_id": failure_context.tenant_id,
                    "labels": {"event_type": "failure"},
                    "metadata": {"failure_context": failure_context.failure_id}
                })

        except Exception as e:
            logger.error(f"Performance Analytics Failure-Callback fehlgeschlagen: {e}")

    async def _monitoring_failure_callback(
        self,
        failure_context: FailureContext,
        _recovery_attempt: Any
    ) -> None:
        """Monitoring Failure-Callback."""
        try:
            if self.real_time_monitoring_engine:
                # Trigger Monitoring-Alert
                await self.real_time_monitoring_engine.trigger_alert({
                    "alert_type": "service_failure",
                    "service_name": failure_context.service_name,
                    "operation_name": failure_context.operation_name,
                    "failure_type": failure_context.failure_type.value,
                    "error_message": failure_context.error_message,
                    "timestamp": failure_context.occurred_at.isoformat(),
                    "severity": "high" if failure_context.failure_type in [
                        FailureType.SERVICE_UNAVAILABLE,
                        FailureType.AUTHENTICATION_FAILED,
                        FailureType.AUTHORIZATION_FAILED
                    ] else "medium"
                })

        except Exception as e:
            logger.error(f"Monitoring Failure-Callback fehlgeschlagen: {e}")

    async def _performance_analytics_recovery_callback(
        self,
        recovery_attempt: Any,
        failure_context: FailureContext
    ) -> None:
        """Performance Analytics Recovery-Callback."""
        try:
            if self.performance_analytics_engine:
                # Erstelle Performance-Datenpunkt für Recovery
                await self.performance_analytics_engine.collect_performance_data_point({
                    "data_point_id": f"recovery_{recovery_attempt.attempt_id}",
                    "metric_name": "recovery_event",
                    "scope": "SERVICE",
                    "scope_id": failure_context.service_name,
                    "value": 1.0 if recovery_attempt.success else 0.0,
                    "unit": "success_rate",
                    "dimensions": {
                        "SERVICE": failure_context.service_name,
                        "OPERATION": failure_context.operation_name,
                        "RECOVERY_STRATEGY": recovery_attempt.strategy.value
                    },
                    "service_name": failure_context.service_name,
                    "user_id": failure_context.user_id,
                    "tenant_id": failure_context.tenant_id,
                    "labels": {"event_type": "recovery"},
                    "metadata": {
                        "recovery_attempt": recovery_attempt.attempt_id,
                        "recovery_time_ms": recovery_attempt.recovery_time_ms
                    }
                })

        except Exception as e:
            logger.error(f"Performance Analytics Recovery-Callback fehlgeschlagen: {e}")

    async def _performance_analytics_saga_callback(
        self,
        saga: SagaTransaction,
        event_type: str
    ) -> None:
        """Performance Analytics Saga-Callback."""
        try:
            if self.performance_analytics_engine:
                # Erstelle Performance-Datenpunkt für Saga
                await self.performance_analytics_engine.collect_performance_data_point({
                    "data_point_id": f"saga_{saga.saga_id}_{event_type}",
                    "metric_name": "saga_event",
                    "scope": "ORCHESTRATION",
                    "scope_id": saga.saga_name,
                    "value": 1.0,
                    "unit": "count",
                    "dimensions": {
                        "ORCHESTRATION": saga.saga_name,
                        "EVENT_TYPE": event_type,
                        "SAGA_STATE": saga.state.value
                    },
                    "service_name": "compensation_framework",
                    "user_id": saga.user_id,
                    "tenant_id": saga.tenant_id,
                    "labels": {"event_type": "saga"},
                    "metadata": {
                        "saga_id": saga.saga_id,
                        "steps_count": len(saga.steps),
                        "executed_steps": len(saga.executed_steps),
                        "compensated_steps": len(saga.compensated_steps)
                    }
                })

        except Exception as e:
            logger.error(f"Performance Analytics Saga-Callback fehlgeschlagen: {e}")

    async def _compensation_completed_callback(
        self,
        saga: SagaTransaction,
        success: bool
    ) -> None:
        """Compensation Completed-Callback."""
        try:
            # Trigger Enhanced Services Integration Events
            await self._trigger_saga_integration_events(saga, success, None)

        except Exception as e:
            logger.error(f"Compensation Completed-Callback fehlgeschlagen: {e}")

    async def _trigger_performance_analytics(
        self,
        service_name: str,
        operation_name: str,
        response_time_ms: float,
        success: bool,
        security_context: SecurityContext | None
    ) -> None:
        """Triggert Performance Analytics."""
        try:
            if self.performance_analytics_engine:
                await self.performance_analytics_engine.analyze_service_performance(
                    service_name=service_name,
                    operation=operation_name,
                    request_data={"integration": "failure_recovery"},
                    result_data={"success": success},
                    response_time_ms=response_time_ms,
                    success=success,
                    security_context=security_context,
                    metadata={"source": "failure_recovery_integration"}
                )

        except Exception as e:
            logger.error(f"Performance Analytics trigger fehlgeschlagen: {e}")

    async def _trigger_saga_integration_events(
        self,
        saga: SagaTransaction,
        success: bool,
        security_context: SecurityContext | None
    ) -> None:
        """Triggert Saga Integration Events."""
        try:
            # Trigger Security Integration
            if self.security_integration_engine and security_context:
                await self.security_integration_engine.log_security_event({
                    "event_type": "saga_transaction",
                    "saga_id": saga.saga_id,
                    "saga_name": saga.saga_name,
                    "success": success,
                    "user_id": saga.user_id,
                    "tenant_id": saga.tenant_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "steps_executed": len(saga.executed_steps),
                    "steps_compensated": len(saga.compensated_steps)
                })

            # Trigger Quota Management
            if self.quota_management_engine:
                await self.quota_management_engine.update_quota_usage({
                    "resource_type": "saga_transactions",
                    "usage_delta": 1,
                    "user_id": saga.user_id,
                    "tenant_id": saga.tenant_id,
                    "metadata": {
                        "saga_id": saga.saga_id,
                        "success": success
                    }
                })

        except Exception as e:
            logger.error(f"Saga Integration Events trigger fehlgeschlagen: {e}")

    async def _get_monitoring_health(self) -> dict[str, Any]:
        """Gibt Monitoring-Health zurück."""
        try:
            if self.real_time_monitoring_engine:
                return {
                    "status": "healthy",
                    "last_check": datetime.utcnow().isoformat(),
                    "metrics_collected": True,
                    "alerts_active": 0
                }
            return {"status": "not_available"}

        except Exception as e:
            logger.error(f"Monitoring health check fehlgeschlagen: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_security_health(self) -> dict[str, Any]:
        """Gibt Security-Health zurück."""
        try:
            if self.security_integration_engine:
                return {
                    "status": "healthy",
                    "last_check": datetime.utcnow().isoformat(),
                    "security_checks_active": True,
                    "audit_trail_enabled": True
                }
            return {"status": "not_available"}

        except Exception as e:
            logger.error(f"Security health check fehlgeschlagen: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_analytics_health(self) -> dict[str, Any]:
        """Gibt Analytics-Health zurück."""
        try:
            if self.performance_analytics_engine:
                return {
                    "status": "healthy",
                    "last_check": datetime.utcnow().isoformat(),
                    "analytics_enabled": True,
                    "ml_predictions_enabled": True
                }
            return {"status": "not_available"}

        except Exception as e:
            logger.error(f"Analytics health check fehlgeschlagen: {e}")
            return {"status": "error", "error": str(e)}

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            failure_recovery_stats = self.failure_recovery_system.get_performance_stats()
            compensation_stats = self.compensation_framework.get_performance_stats()

            return {
                "failure_recovery_integration": {
                    "is_running": self._is_running,
                    "integration_metrics": self._integration_metrics,
                    "service_registry": list(self._service_registry.keys()),
                    "enhanced_services_connected": {
                        "monitoring_engine": self.monitoring_engine is not None,
                        "dependency_resolution_engine": self.dependency_resolution_engine is not None,
                        "quota_management_engine": self.quota_management_engine is not None,
                        "security_integration_engine": self.security_integration_engine is not None,
                        "performance_analytics_engine": self.performance_analytics_engine is not None,
                        "real_time_monitoring_engine": self.real_time_monitoring_engine is not None
                    }
                },
                **failure_recovery_stats,
                **compensation_stats
            }

        except Exception as e:
            logger.error(f"Performance stats retrieval fehlgeschlagen: {e}")
            return {}
