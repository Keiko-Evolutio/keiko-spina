# backend/services/failure_recovery_compensation/__init__.py
"""Failure Recovery & Compensation System.

Enterprise-Grade Failure Recovery System mit automatischer Fehlerbehandlung,
Retry-Mechanismen, Circuit Breaker Patterns und Saga Pattern Compensation Framework
für alle kritischen Services im Keiko-Personal-Assistant Backend.
"""

from .compensation_framework import CompensationFramework
from .data_models import (
    CompensationAction,
    DistributedSystemHealth,
    # Data Models
    FailureContext,
    FailureRecoveryMetrics,
    # Enums
    FailureType,
    RecoveryAttempt,
    RecoveryConfiguration,
    RecoveryState,
    RecoveryStrategy,
    SagaState,
    SagaStep,
    SagaTransaction,
)
from .failure_recovery_integration import FailureRecoveryIntegration
from .failure_recovery_system import CircuitBreakerState, FailureRecoverySystem


# Factory Functions
def create_failure_recovery_system() -> FailureRecoverySystem:
    """Erstellt Failure Recovery System.

    Returns:
        Failure Recovery System
    """
    return FailureRecoverySystem()


def create_compensation_framework() -> CompensationFramework:
    """Erstellt Compensation Framework.

    Returns:
        Compensation Framework
    """
    return CompensationFramework()


def create_failure_recovery_integration(
    monitoring_engine=None,
    dependency_resolution_engine=None,
    quota_management_engine=None,
    security_integration_engine=None,
    performance_analytics_engine=None,
    real_time_monitoring_engine=None
) -> FailureRecoveryIntegration:
    """Erstellt Failure Recovery Integration.

    Args:
        monitoring_engine: Real-time Monitoring Engine
        dependency_resolution_engine: Dependency Resolution Engine
        quota_management_engine: Quota Management Engine
        security_integration_engine: Security Integration Engine
        performance_analytics_engine: Performance Analytics Engine
        real_time_monitoring_engine: Real-time Monitoring Engine

    Returns:
        Failure Recovery Integration
    """
    return FailureRecoveryIntegration(
        monitoring_engine=monitoring_engine,
        dependency_resolution_engine=dependency_resolution_engine,
        quota_management_engine=quota_management_engine,
        security_integration_engine=security_integration_engine,
        performance_analytics_engine=performance_analytics_engine,
        real_time_monitoring_engine=real_time_monitoring_engine
    )


def create_integrated_failure_recovery_system(
    monitoring_engine=None,
    dependency_resolution_engine=None,
    quota_management_engine=None,
    security_integration_engine=None,
    performance_analytics_engine=None,
    real_time_monitoring_engine=None
) -> dict:
    """Erstellt vollständiges integriertes Failure Recovery System.

    Args:
        monitoring_engine: Real-time Monitoring Engine
        dependency_resolution_engine: Dependency Resolution Engine
        quota_management_engine: Quota Management Engine
        security_integration_engine: Security Integration Engine
        performance_analytics_engine: Performance Analytics Engine
        real_time_monitoring_engine: Real-time Monitoring Engine

    Returns:
        Dictionary mit allen Failure Recovery Komponenten
    """
    # Erstelle Core Components
    failure_recovery_system = create_failure_recovery_system()
    compensation_framework = create_compensation_framework()

    # Erstelle Integration Layer
    failure_recovery_integration = create_failure_recovery_integration(
        monitoring_engine=monitoring_engine,
        dependency_resolution_engine=dependency_resolution_engine,
        quota_management_engine=quota_management_engine,
        security_integration_engine=security_integration_engine,
        performance_analytics_engine=performance_analytics_engine,
        real_time_monitoring_engine=real_time_monitoring_engine
    )

    return {
        "failure_recovery_system": failure_recovery_system,
        "compensation_framework": compensation_framework,
        "failure_recovery_integration": failure_recovery_integration
    }


# Utility Functions
def create_default_recovery_configuration(
    service_name: str,
    operation_name: str,
    primary_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF,
    fallback_strategies: list = None,
    max_retry_attempts: int = 3,
    initial_retry_delay_ms: int = 1000,
    max_retry_delay_ms: int = 30000,
    retry_multiplier: float = 2.0,
    retry_jitter: bool = True,
    failure_threshold: int = 5,
    success_threshold: int = 3,
    circuit_timeout_ms: int = 60000
) -> RecoveryConfiguration:
    """Erstellt Default Recovery-Konfiguration.

    Args:
        service_name: Service-Name
        operation_name: Operation-Name
        primary_strategy: Primäre Recovery-Strategie
        fallback_strategies: Fallback-Strategien
        max_retry_attempts: Maximale Retry-Versuche
        initial_retry_delay_ms: Initiale Retry-Delay in ms
        max_retry_delay_ms: Maximale Retry-Delay in ms
        retry_multiplier: Retry-Multiplier für Exponential Backoff
        retry_jitter: Retry-Jitter aktiviert
        failure_threshold: Circuit Breaker Failure-Threshold
        success_threshold: Circuit Breaker Success-Threshold
        circuit_timeout_ms: Circuit Breaker Timeout in ms

    Returns:
        Recovery-Konfiguration
    """
    import uuid

    if fallback_strategies is None:
        fallback_strategies = [
            RecoveryStrategy.FALLBACK_SERVICE,
            RecoveryStrategy.CACHED_RESPONSE,
            RecoveryStrategy.DEFAULT_RESPONSE
        ]

    return RecoveryConfiguration(
        config_id=str(uuid.uuid4()),
        service_name=service_name,
        operation_name=operation_name,
        primary_strategy=primary_strategy,
        fallback_strategies=fallback_strategies,
        max_retry_attempts=max_retry_attempts,
        initial_retry_delay_ms=initial_retry_delay_ms,
        max_retry_delay_ms=max_retry_delay_ms,
        retry_multiplier=retry_multiplier,
        retry_jitter=retry_jitter,
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        circuit_timeout_ms=circuit_timeout_ms
    )


def create_saga_step(
    step_name: str,
    service_name: str,
    operation_name: str,
    endpoint_url: str,
    request_data: dict,
    compensation_action: CompensationAction,
    step_order: int = 1,
    headers: dict = None,
    compensation_service: str = None,
    compensation_operation: str = None,
    compensation_data: dict = None,
    timeout_ms: int = 30000,
    retry_enabled: bool = True,
    max_retries: int = 3,
    depends_on: list = None,
    metadata: dict = None
) -> SagaStep:
    """Erstellt Saga-Step.

    Args:
        step_name: Step-Name
        service_name: Service-Name
        operation_name: Operation-Name
        endpoint_url: Endpoint-URL
        request_data: Request-Daten
        compensation_action: Compensation-Action
        step_order: Step-Reihenfolge
        headers: HTTP-Headers
        compensation_service: Compensation-Service
        compensation_operation: Compensation-Operation
        compensation_data: Compensation-Daten
        timeout_ms: Timeout in ms
        retry_enabled: Retry aktiviert
        max_retries: Maximale Retries
        depends_on: Abhängigkeiten
        metadata: Metadata

    Returns:
        Saga-Step
    """
    import uuid

    if headers is None:
        headers = {}
    if compensation_data is None:
        compensation_data = {}
    if depends_on is None:
        depends_on = []
    if metadata is None:
        metadata = {}

    return SagaStep(
        step_id=str(uuid.uuid4()),
        step_name=step_name,
        step_order=step_order,
        service_name=service_name,
        operation_name=operation_name,
        endpoint_url=endpoint_url,
        request_data=request_data,
        headers=headers,
        compensation_action=compensation_action,
        compensation_service=compensation_service,
        compensation_operation=compensation_operation,
        compensation_data=compensation_data,
        timeout_ms=timeout_ms,
        retry_enabled=retry_enabled,
        max_retries=max_retries,
        depends_on=depends_on,
        metadata=metadata
    )


def create_saga_transaction(
    saga_name: str,
    description: str,
    steps: list,
    compensation_strategy: str = "automatic",
    compensation_timeout_ms: int = 300000,
    total_timeout_ms: int = 1800000,
    metadata: dict = None,
    tags: list = None
) -> SagaTransaction:
    """Erstellt Saga-Transaction.

    Args:
        saga_name: Saga-Name
        description: Saga-Beschreibung
        steps: Saga-Steps
        compensation_strategy: Compensation-Strategie
        compensation_timeout_ms: Compensation-Timeout in ms
        total_timeout_ms: Gesamt-Timeout in ms
        metadata: Metadata
        tags: Tags

    Returns:
        Saga-Transaction
    """
    import uuid

    if metadata is None:
        metadata = {}
    if tags is None:
        tags = []

    return SagaTransaction(
        saga_id=str(uuid.uuid4()),
        saga_name=saga_name,
        description=description,
        steps=steps,
        compensation_strategy=compensation_strategy,
        compensation_timeout_ms=compensation_timeout_ms,
        total_timeout_ms=total_timeout_ms,
        metadata=metadata,
        tags=tags
    )


# Export all public components
__all__ = [
    # Enums
    "FailureType",
    "RecoveryStrategy",
    "CompensationAction",
    "SagaState",
    "RecoveryState",

    # Data Models
    "FailureContext",
    "RecoveryConfiguration",
    "RecoveryAttempt",
    "SagaStep",
    "SagaTransaction",
    "DistributedSystemHealth",
    "FailureRecoveryMetrics",

    # Core Components
    "CircuitBreakerState",
    "FailureRecoverySystem",
    "CompensationFramework",
    "FailureRecoveryIntegration",

    # Factory Functions
    "create_failure_recovery_system",
    "create_compensation_framework",
    "create_failure_recovery_integration",
    "create_integrated_failure_recovery_system",

    # Utility Functions
    "create_default_recovery_configuration",
    "create_saga_step",
    "create_saga_transaction"
]
