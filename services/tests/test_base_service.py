"""Unit-Tests für services.core.base_service.

Testet alle Basis-Service-Klassen und deren Funktionalität.
"""

import asyncio

import pytest

from services.core.base_service import BaseService, MonitoringService, PeriodicService
from services.core.constants import (
    SERVICE_STATUS_AVAILABLE,
    SERVICE_STATUS_ERROR,
    SERVICE_STATUS_UNAVAILABLE,
)


class TestBaseService:
    """Tests für BaseService."""

    class ConcreteService(BaseService):
        """Konkrete Implementierung für Tests."""

        def __init__(self, service_name: str = "TestService"):
            super().__init__(service_name)
            self.initialize_called = False
            self.cleanup_called = False
            self.should_fail_initialize = False
            self.should_fail_cleanup = False

        async def _initialize(self) -> None:
            """Test-Implementierung der Initialisierung."""
            self.initialize_called = True
            if self.should_fail_initialize:
                raise RuntimeError("Initialize failed")

        async def _cleanup(self) -> None:
            """Test-Implementierung der Bereinigung."""
            self.cleanup_called = True
            if self.should_fail_cleanup:
                raise RuntimeError("Cleanup failed")

    def test_service_initialization(self):
        """Testet Service-Initialisierung."""
        service = self.ConcreteService("TestService")
        assert service.service_name == "TestService"
        assert not service.running
        assert service.status == SERVICE_STATUS_UNAVAILABLE
        assert service.last_health_check is None

    @pytest.mark.asyncio
    async def test_successful_start(self):
        """Testet erfolgreichen Service-Start."""
        service = self.ConcreteService()

        await service.start()

        assert service.running
        assert service.status == SERVICE_STATUS_AVAILABLE
        assert service.initialize_called

    @pytest.mark.asyncio
    async def test_failed_start(self):
        """Testet fehlgeschlagenen Service-Start."""
        service = self.ConcreteService()
        service.should_fail_initialize = True

        with pytest.raises(RuntimeError, match="Initialize failed"):
            await service.start()

        assert not service.running
        assert service.status == SERVICE_STATUS_ERROR

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Testet Start eines bereits laufenden Services."""
        service = self.ConcreteService()
        await service.start()

        # Reset für zweiten Start
        service.initialize_called = False

        await service.start()  # Sollte keine Exception werfen
        assert not service.initialize_called  # Sollte nicht erneut initialisiert werden

    @pytest.mark.asyncio
    async def test_successful_stop(self):
        """Testet erfolgreichen Service-Stop."""
        service = self.ConcreteService()
        await service.start()

        await service.stop()

        assert not service.running
        assert service.status == SERVICE_STATUS_UNAVAILABLE
        assert service.cleanup_called

    @pytest.mark.asyncio
    async def test_failed_stop(self):
        """Testet fehlgeschlagenen Service-Stop."""
        service = self.ConcreteService()
        await service.start()
        service.should_fail_cleanup = True

        with pytest.raises(RuntimeError, match="Cleanup failed"):
            await service.stop()

        assert service.status == SERVICE_STATUS_ERROR

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Testet Stop eines nicht laufenden Services."""
        service = self.ConcreteService()

        await service.stop()  # Sollte keine Exception werfen
        assert not service.cleanup_called

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Testet Health-Check für gesunden Service."""
        service = self.ConcreteService()
        await service.start()

        health = await service.health_check()

        assert health["service"] == "TestService"
        assert health["status"] == SERVICE_STATUS_AVAILABLE
        assert health["running"] is True
        assert "response_time_ms" in health
        assert "last_check" in health
        assert service.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Testet Health-Check für ungesunden Service."""
        service = self.ConcreteService()
        # Service nicht starten

        health = await service.health_check()

        assert health["service"] == "TestService"
        assert health["status"] == SERVICE_STATUS_ERROR
        assert health["running"] is False

    def test_get_status(self):
        """Testet Status-Abfrage."""
        service = self.ConcreteService()

        status = service.get_status()

        assert status["service"] == "TestService"
        assert status["status"] == SERVICE_STATUS_UNAVAILABLE
        assert status["running"] is False
        assert status["last_health_check"] is None


class TestPeriodicService:
    """Tests für PeriodicService."""

    class ConcretePeriodicService(PeriodicService):
        """Konkrete Implementierung für Tests."""

        def __init__(self, service_name: str = "TestPeriodicService", interval: float = 0.1):
            super().__init__(service_name, interval)
            self.periodic_task_calls = 0
            self.should_fail_periodic = False

        async def _initialize(self) -> None:
            """Test-Implementierung der Initialisierung."""

        async def _cleanup(self) -> None:
            """Test-Implementierung der Bereinigung."""

        async def _perform_periodic_task(self) -> None:
            """Test-Implementierung des periodischen Tasks."""
            self.periodic_task_calls += 1
            if self.should_fail_periodic:
                raise RuntimeError("Periodic task failed")

    @pytest.mark.asyncio
    async def test_periodic_task_execution(self):
        """Testet Ausführung periodischer Tasks."""
        service = self.ConcretePeriodicService(interval=0.05)  # Sehr kurzes Intervall für Tests

        await service.start()
        await asyncio.sleep(0.2)  # Warte auf mehrere Ausführungen
        await service.stop()

        assert service.periodic_task_calls > 0

    @pytest.mark.asyncio
    async def test_periodic_task_error_handling(self):
        """Testet Error-Handling in periodischen Tasks."""
        service = self.ConcretePeriodicService(interval=0.05)
        service.should_fail_periodic = True

        await service.start()
        await asyncio.sleep(0.2)  # Warte trotz Fehlern
        await service.stop()

        # Service sollte trotz Fehlern weiterlaufen
        assert service.running is False  # Nach stop()


class TestMonitoringService:
    """Tests für MonitoringService."""

    class ConcreteMonitoringService(MonitoringService):
        """Konkrete Implementierung für Tests."""

        def __init__(self, service_name: str = "TestMonitoringService"):
            super().__init__(service_name, interval_seconds=0.1, max_failures=2)
            self.escalations = []

        async def _initialize(self) -> None:
            """Test-Implementierung der Initialisierung."""

        async def _cleanup(self) -> None:
            """Test-Implementierung der Bereinigung."""

        async def _perform_periodic_task(self) -> None:
            """Test-Implementierung des periodischen Tasks."""

        async def _escalate_failure(self, target: str) -> None:
            """Test-Implementierung der Eskalation."""
            self.escalations.append(target)

    def test_failure_count_management(self):
        """Testet Failure-Count-Management."""
        service = self.ConcreteMonitoringService()

        # Test Increment
        count1 = service._increment_failure_count("target1")
        assert count1 == 1

        count2 = service._increment_failure_count("target1")
        assert count2 == 2

        # Test Reset
        service._reset_failure_count("target1")
        assert service.failure_counts.get("target1", 0) == 0

    def test_escalation_threshold(self):
        """Testet Eskalations-Schwellenwert."""
        service = self.ConcreteMonitoringService()

        # Unter Schwellenwert
        service._increment_failure_count("target1")
        assert not service._should_escalate("target1")

        # Über Schwellenwert
        service._increment_failure_count("target1")
        assert service._should_escalate("target1")

    @pytest.mark.asyncio
    async def test_monitoring_success_handling(self):
        """Testet Behandlung erfolgreicher Monitoring-Checks."""
        service = self.ConcreteMonitoringService()

        # Setze Failure-Count
        service._increment_failure_count("target1")

        await service._handle_monitoring_success("target1")

        # Failure-Count sollte zurückgesetzt sein
        assert service.failure_counts.get("target1", 0) == 0

    @pytest.mark.asyncio
    async def test_monitoring_failure_handling(self):
        """Testet Behandlung fehlgeschlagener Monitoring-Checks."""
        service = self.ConcreteMonitoringService()

        # Erste Fehler
        await service._handle_monitoring_failure("target1", "Error 1")
        assert service.failure_counts["target1"] == 1
        assert len(service.escalations) == 0

        # Zweiter Fehler - sollte eskalieren
        await service._handle_monitoring_failure("target1", "Error 2")
        assert service.failure_counts["target1"] == 2
        assert "target1" in service.escalations
