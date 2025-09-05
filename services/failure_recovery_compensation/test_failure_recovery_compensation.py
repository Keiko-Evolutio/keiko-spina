# backend/services/failure_recovery_compensation/test_failure_recovery_compensation.py
"""Tests für Failure Recovery & Compensation System.

Umfassende Tests für Enterprise-Grade Failure Recovery System,
Compensation Framework und Integration Layer.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.enhanced_security_integration import SecurityContext, SecurityLevel

from . import (
    CompensationAction,
    CompensationFramework,
    FailureRecoveryIntegration,
    FailureRecoverySystem,
    RecoveryStrategy,
    SagaState,
    create_default_recovery_configuration,
    create_integrated_failure_recovery_system,
    create_saga_step,
)


@pytest.fixture
def security_context():
    """Security Context für Tests."""
    return SecurityContext(
        user_id="test_user",
        tenant_id="test_tenant",
        authentication_method="oauth2",
        token_type="bearer",
        token_claims={"sub": "test_user", "tenant": "test_tenant"},
        roles=["user", "failure_recovery_manager"],
        permissions=["read", "write", "execute"],
        scopes=["failure_recovery_management", "system_access"],
        security_level=SecurityLevel.INTERNAL,
        clearances=["internal_clearance"],
        request_id="test_request_123",
        source_ip="192.168.1.100",
        user_agent="TestClient/1.0"
    )


@pytest.fixture
def mock_enhanced_services():
    """Mock Enhanced Services für Tests."""
    return {
        "monitoring_engine": MagicMock(),
        "dependency_resolution_engine": MagicMock(),
        "quota_management_engine": MagicMock(),
        "security_integration_engine": MagicMock(),
        "performance_analytics_engine": MagicMock(),
        "real_time_monitoring_engine": MagicMock()
    }


class TestFailureRecoverySystem:
    """Tests für Failure Recovery System."""

    @pytest.mark.asyncio
    async def test_failure_recovery_system_initialization(self):
        """Test Failure Recovery System Initialisierung."""
        system = FailureRecoverySystem()

        assert not system._is_running
        assert len(system._recovery_configs) == 0
        assert len(system._circuit_breakers) == 0
        assert len(system._active_recoveries) == 0

        await system.start()
        assert system._is_running

        await system.stop()
        assert not system._is_running

    @pytest.mark.asyncio
    async def test_recovery_configuration_registration(self):
        """Test Recovery-Konfiguration Registration."""
        system = FailureRecoverySystem()
        await system.start()

        try:
            # Erstelle Recovery-Konfiguration
            config = create_default_recovery_configuration(
                service_name="test_service",
                operation_name="test_operation",
                primary_strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF
            )

            # Registriere Konfiguration
            await system.register_recovery_configuration(config)

            # Prüfe Registration
            config_key = f"{config.service_name}:{config.operation_name}"
            assert config_key in system._recovery_configs
            assert system._recovery_configs[config_key] == config

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_execute_with_recovery_success(self, security_context):
        """Test Execute with Recovery - Success."""
        system = FailureRecoverySystem()
        await system.start()

        try:
            # Mock Operation
            async def mock_operation():
                return {"status": "success", "data": "test_data"}

            # Führe Operation mit Recovery aus
            result = await system.execute_with_recovery(
                operation=mock_operation,
                service_name="test_service",
                operation_name="test_operation",
                security_context=security_context
            )

            # Prüfe Ergebnis
            assert result["status"] == "success"
            assert result["data"] == "test_data"

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_execute_with_recovery_failure_and_retry(self, security_context):
        """Test Execute with Recovery - Failure and Retry."""
        system = FailureRecoverySystem()
        await system.start()

        try:
            # Registriere Recovery-Konfiguration
            config = create_default_recovery_configuration(
                service_name="test_service",
                operation_name="test_operation",
                max_retry_attempts=2,
                initial_retry_delay_ms=10
            )
            await system.register_recovery_configuration(config)

            # Mock Operation die beim ersten Mal fehlschlägt
            call_count = 0

            async def mock_operation():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Temporary failure")
                return {"status": "success", "retry": call_count}

            # Führe Operation mit Recovery aus
            result = await system.execute_with_recovery(
                operation=mock_operation,
                service_name="test_service",
                operation_name="test_operation",
                security_context=security_context
            )

            # Prüfe Ergebnis
            assert result["status"] == "success"
            assert result["retry"] == 2
            assert call_count == 2

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self):
        """Test System Health Monitoring."""
        system = FailureRecoverySystem()
        await system.start()

        try:
            # Hole System-Health
            health = await system.get_system_health()

            # Prüfe Health-Status
            assert health.system_id == "keiko_personal_assistant"
            assert health.system_name == "Keiko Personal Assistant"
            assert health.overall_health in ["healthy", "degraded", "unhealthy", "critical"]
            assert 0.0 <= health.health_score <= 1.0
            assert isinstance(health.last_updated, datetime)

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_recovery_metrics(self):
        """Test Recovery Metrics."""
        system = FailureRecoverySystem()
        await system.start()

        try:
            # Hole Recovery-Metriken
            metrics = await system.get_recovery_metrics()

            # Prüfe Metriken
            assert metrics.system_id == "keiko_personal_assistant"
            assert metrics.total_failures >= 0
            assert metrics.total_recovery_attempts >= 0
            assert 0.0 <= metrics.recovery_success_rate <= 1.0
            assert metrics.avg_recovery_time_ms >= 0.0
            assert 0.0 <= metrics.system_availability <= 100.0

        finally:
            await system.stop()


class TestCompensationFramework:
    """Tests für Compensation Framework."""

    @pytest.mark.asyncio
    async def test_compensation_framework_initialization(self):
        """Test Compensation Framework Initialisierung."""
        framework = CompensationFramework()

        assert not framework._is_running
        assert len(framework._active_sagas) == 0
        assert len(framework._saga_history) == 0

        await framework.start()
        assert framework._is_running

        await framework.stop()
        assert not framework._is_running

    @pytest.mark.asyncio
    async def test_saga_transaction_creation(self, security_context):
        """Test Saga Transaction Creation."""
        framework = CompensationFramework()
        await framework.start()

        try:
            # Erstelle Saga-Steps
            steps = [
                create_saga_step(
                    step_name="create_user",
                    service_name="user_service",
                    operation_name="create_user",
                    endpoint_url="/api/users",
                    request_data={"name": "Test User", "email": "test@example.com"},
                    compensation_action=CompensationAction.DELETE_CREATED_RECORD,
                    step_order=1
                ),
                create_saga_step(
                    step_name="send_welcome_email",
                    service_name="notification_service",
                    operation_name="send_email",
                    endpoint_url="/api/notifications/email",
                    request_data={"to": "test@example.com", "template": "welcome"},
                    compensation_action=CompensationAction.NOTIFY_CANCELLATION,
                    step_order=2,
                    depends_on=["create_user"]
                )
            ]

            # Erstelle Saga-Transaction
            saga = await framework.create_saga_transaction(
                saga_name="user_onboarding",
                description="User Onboarding Process",
                steps=steps,
                security_context=security_context
            )

            # Prüfe Saga
            assert saga.saga_name == "user_onboarding"
            assert saga.description == "User Onboarding Process"
            assert len(saga.steps) == 2
            assert saga.state == SagaState.CREATED
            assert saga.user_id == security_context.user_id
            assert saga.tenant_id == security_context.tenant_id

        finally:
            await framework.stop()

    @pytest.mark.asyncio
    async def test_saga_execution_success(self, security_context):
        """Test Saga Execution - Success."""
        framework = CompensationFramework()
        await framework.start()

        try:
            # Erstelle einfache Saga
            steps = [
                create_saga_step(
                    step_name="step1",
                    service_name="test_service",
                    operation_name="operation1",
                    endpoint_url="/api/test1",
                    request_data={"data": "test1"},
                    compensation_action=CompensationAction.UNDO_DATA_CHANGE,
                    step_order=1
                )
            ]

            saga = await framework.create_saga_transaction(
                saga_name="test_saga",
                description="Test Saga",
                steps=steps,
                security_context=security_context
            )

            # Führe Saga aus
            success = await framework.execute_saga_transaction(
                saga_id=saga.saga_id,
                security_context=security_context
            )

            # Prüfe Ergebnis
            assert success

            # Prüfe Saga-Status
            completed_saga = await framework.get_saga_status(saga.saga_id)
            assert completed_saga is None  # Moved to history

            # Prüfe History
            assert saga.saga_id in framework._saga_history
            historical_saga = framework._saga_history[saga.saga_id]
            assert historical_saga.state == SagaState.COMPLETED

        finally:
            await framework.stop()

    @pytest.mark.asyncio
    async def test_saga_execution_with_compensation(self, security_context):
        """Test Saga Execution with Compensation."""
        framework = CompensationFramework()
        await framework.start()

        try:
            # Erstelle Saga mit Steps die fehlschlagen werden
            steps = [
                create_saga_step(
                    step_name="step1",
                    service_name="user_service",  # Hohe Success-Rate
                    operation_name="operation1",
                    endpoint_url="/api/test1",
                    request_data={"data": "test1"},
                    compensation_action=CompensationAction.UNDO_DATA_CHANGE,
                    step_order=1
                ),
                create_saga_step(
                    step_name="step2",
                    service_name="payment_service",  # Niedrigere Success-Rate
                    operation_name="operation2",
                    endpoint_url="/api/test2",
                    request_data={"data": "test2"},
                    compensation_action=CompensationAction.REFUND_PAYMENT,
                    step_order=2
                )
            ]

            saga = await framework.create_saga_transaction(
                saga_name="test_saga_with_failure",
                description="Test Saga with Potential Failure",
                steps=steps,
                security_context=security_context
            )

            # Führe Saga aus (kann fehlschlagen und Compensation auslösen)
            success = await framework.execute_saga_transaction(
                saga_id=saga.saga_id,
                security_context=security_context
            )

            # Prüfe dass Saga verarbeitet wurde (erfolgreich oder kompensiert)
            assert saga.saga_id in framework._saga_history
            historical_saga = framework._saga_history[saga.saga_id]
            assert historical_saga.state in [SagaState.COMPLETED, SagaState.FAILED]

        finally:
            await framework.stop()

    @pytest.mark.asyncio
    async def test_compensation_framework_metrics(self):
        """Test Compensation Framework Metrics."""
        framework = CompensationFramework()
        await framework.start()

        try:
            # Hole Performance-Stats
            stats = framework.get_performance_stats()

            # Prüfe Stats
            assert "compensation_framework" in stats
            framework_stats = stats["compensation_framework"]

            assert framework_stats["is_running"]
            assert framework_stats["active_sagas"] >= 0
            assert framework_stats["saga_history_count"] >= 0
            assert "metrics" in framework_stats

        finally:
            await framework.stop()


class TestFailureRecoveryIntegration:
    """Tests für Failure Recovery Integration."""

    @pytest.mark.asyncio
    async def test_integration_initialization(self, mock_enhanced_services):
        """Test Integration Initialisierung."""
        integration = FailureRecoveryIntegration(**mock_enhanced_services)

        assert not integration._is_running
        assert integration.monitoring_engine is not None
        assert integration.security_integration_engine is not None
        assert integration.performance_analytics_engine is not None

        await integration.start()
        assert integration._is_running

        await integration.stop()
        assert not integration._is_running

    @pytest.mark.asyncio
    async def test_execute_with_failure_recovery_integration(self, mock_enhanced_services, security_context):
        """Test Execute with Failure Recovery Integration."""
        # Setup Mocks
        mock_enhanced_services["performance_analytics_engine"].analyze_service_performance = AsyncMock()

        integration = FailureRecoveryIntegration(**mock_enhanced_services)
        await integration.start()

        try:
            # Mock Operation
            async def mock_operation():
                return {"status": "success", "integration": "tested"}

            # Führe Operation mit Integration aus
            result = await integration.execute_with_failure_recovery(
                operation=mock_operation,
                service_name="test_service",
                operation_name="test_operation",
                security_context=security_context
            )

            # Prüfe Ergebnis
            assert result["status"] == "success"
            assert result["integration"] == "tested"

            # Prüfe dass Performance Analytics aufgerufen wurde
            mock_enhanced_services["performance_analytics_engine"].analyze_service_performance.assert_called_once()

        finally:
            await integration.stop()

    @pytest.mark.asyncio
    async def test_saga_transaction_integration(self, mock_enhanced_services, security_context):
        """Test Saga Transaction Integration."""
        integration = FailureRecoveryIntegration(**mock_enhanced_services)
        await integration.start()

        try:
            # Erstelle Saga-Steps
            steps = [
                create_saga_step(
                    step_name="integration_test",
                    service_name="test_service",
                    operation_name="test_operation",
                    endpoint_url="/api/test",
                    request_data={"test": "integration"},
                    compensation_action=CompensationAction.LOG_COMPENSATION,
                    step_order=1
                )
            ]

            # Führe Saga-Transaction aus
            success = await integration.execute_saga_transaction(
                saga_name="integration_test_saga",
                description="Integration Test Saga",
                steps=steps,
                security_context=security_context
            )

            # Prüfe dass Saga verarbeitet wurde
            assert isinstance(success, bool)

        finally:
            await integration.stop()

    @pytest.mark.asyncio
    async def test_distributed_system_health(self, mock_enhanced_services):
        """Test Distributed System Health."""
        integration = FailureRecoveryIntegration(**mock_enhanced_services)
        await integration.start()

        try:
            # Hole Distributed System Health
            health = await integration.get_distributed_system_health()

            # Prüfe Health
            assert health.system_id == "keiko_personal_assistant"
            assert health.overall_health in ["healthy", "degraded", "unhealthy", "critical"]
            assert 0.0 <= health.health_score <= 1.0
            assert isinstance(health.service_health, dict)

        finally:
            await integration.stop()

    @pytest.mark.asyncio
    async def test_comprehensive_metrics(self, mock_enhanced_services):
        """Test Comprehensive Metrics."""
        integration = FailureRecoveryIntegration(**mock_enhanced_services)
        await integration.start()

        try:
            # Hole Comprehensive Metrics
            metrics = await integration.get_comprehensive_metrics()

            # Prüfe Metrics
            assert "failure_recovery_metrics" in metrics
            assert "compensation_metrics" in metrics
            assert "integration_metrics" in metrics
            assert "enhanced_services_integration" in metrics

            # Prüfe Integration Metrics
            integration_metrics = metrics["integration_metrics"]
            assert integration_metrics["total_integrations"] >= 0
            assert 0.0 <= integration_metrics.get("integration_success_rate", 0.0) <= 1.0

            # Prüfe Enhanced Services Integration
            enhanced_services = metrics["enhanced_services_integration"]
            assert enhanced_services["monitoring_engine"]
            assert enhanced_services["security_integration_engine"]
            assert enhanced_services["performance_analytics_engine"]

        finally:
            await integration.stop()


class TestIntegratedSystem:
    """Tests für Integrated Failure Recovery System."""

    @pytest.mark.asyncio
    async def test_integrated_system_creation(self, mock_enhanced_services):
        """Test Integrated System Creation."""
        # Erstelle Integrated System
        system = create_integrated_failure_recovery_system(**mock_enhanced_services)

        # Prüfe Components
        assert "failure_recovery_system" in system
        assert "compensation_framework" in system
        assert "failure_recovery_integration" in system

        # Prüfe Component-Types
        assert isinstance(system["failure_recovery_system"], FailureRecoverySystem)
        assert isinstance(system["compensation_framework"], CompensationFramework)
        assert isinstance(system["failure_recovery_integration"], FailureRecoveryIntegration)

    @pytest.mark.asyncio
    async def test_integrated_system_lifecycle(self, mock_enhanced_services):
        """Test Integrated System Lifecycle."""
        system = create_integrated_failure_recovery_system(**mock_enhanced_services)

        # Starte alle Components
        await system["failure_recovery_system"].start()
        await system["compensation_framework"].start()
        await system["failure_recovery_integration"].start()

        try:
            # Prüfe dass alle Components laufen
            assert system["failure_recovery_system"]._is_running
            assert system["compensation_framework"]._is_running
            assert system["failure_recovery_integration"]._is_running

        finally:
            # Stoppe alle Components
            await system["failure_recovery_integration"].stop()
            await system["compensation_framework"].stop()
            await system["failure_recovery_system"].stop()

            # Prüfe dass alle Components gestoppt sind
            assert not system["failure_recovery_system"]._is_running
            assert not system["compensation_framework"]._is_running
            assert not system["failure_recovery_integration"]._is_running


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
