"""Tests für Basis-Klassen und Service-Hierarchie."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from services.interfaces._base import (
    CoreService,
    FeatureService,
    InfrastructureService,
    LifecycleService,
    ServiceStatus,
    UtilityService,
)


class ConcreteLifecycleService(LifecycleService):
    """Konkrete Implementierung für Tests."""

    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_called = False

    async def initialize(self) -> None:
        """Initialisiert den Test-Service."""
        self.initialized = True

    async def shutdown(self) -> None:
        """Shutdown des Test-Services."""
        self.shutdown_called = True


class TestLifecycleService:
    """Tests für LifecycleService Basis-Klasse."""

    @pytest.fixture
    def service(self) -> ConcreteLifecycleService:
        """Erstellt Test-Service-Instanz."""
        return ConcreteLifecycleService()

    @pytest.mark.asyncio
    async def test_initialize(self, service: ConcreteLifecycleService) -> None:
        """Testet Service-Initialisierung."""
        assert not service.initialized
        await service.initialize()
        assert service.initialized

    @pytest.mark.asyncio
    async def test_shutdown(self, service: ConcreteLifecycleService) -> None:
        """Testet Service-Shutdown."""
        assert not service.shutdown_called
        await service.shutdown()
        assert service.shutdown_called

    @pytest.mark.asyncio
    async def test_health_check_default(self, service: ConcreteLifecycleService) -> None:
        """Testet Standard-Health-Check."""
        health = await service.health_check()

        assert isinstance(health, dict)
        assert health["status"] == "healthy"
        assert health["service"] == "ConcreteLifecycleService"

    def test_get_status_default(self, service: ConcreteLifecycleService) -> None:
        """Testet Standard-Status."""
        status = service.get_status()
        assert status == ServiceStatus.RUNNING

    def test_lifecycle_service_is_abstract(self) -> None:
        """Testet, dass LifecycleService abstrakt ist."""
        with pytest.raises(TypeError):
            LifecycleService()  # type: ignore[abstract]


class TestServiceHierarchy:
    """Tests für Service-Hierarchie."""

    def test_core_service_inheritance(self) -> None:
        """Testet CoreService Vererbung."""
        assert issubclass(CoreService, LifecycleService)

        with pytest.raises(TypeError):
            CoreService()  # type: ignore[abstract]

    def test_infrastructure_service_inheritance(self) -> None:
        """Testet InfrastructureService Vererbung."""
        assert issubclass(InfrastructureService, LifecycleService)

        with pytest.raises(TypeError):
            InfrastructureService()  # type: ignore[abstract]

    def test_feature_service_inheritance(self) -> None:
        """Testet FeatureService Vererbung."""
        assert issubclass(FeatureService, LifecycleService)

        with pytest.raises(TypeError):
            FeatureService()  # type: ignore[abstract]

    def test_utility_service_inheritance(self) -> None:
        """Testet UtilityService Vererbung."""
        assert issubclass(UtilityService, LifecycleService)

        with pytest.raises(TypeError):
            UtilityService()  # type: ignore[abstract]


class TestServiceStatus:
    """Tests für ServiceStatus Enum."""

    def test_all_status_values(self) -> None:
        """Testet alle ServiceStatus Werte."""
        expected_values = {
            "uninitialized",
            "initializing",
            "running",
            "stopping",
            "stopped",
            "error"
        }

        actual_values = {status.value for status in ServiceStatus}
        assert actual_values == expected_values

    def test_status_enum_members(self) -> None:
        """Testet ServiceStatus Enum-Member."""
        assert ServiceStatus.UNINITIALIZED.value == "uninitialized"
        assert ServiceStatus.INITIALIZING.value == "initializing"
        assert ServiceStatus.RUNNING.value == "running"
        assert ServiceStatus.STOPPING.value == "stopping"
        assert ServiceStatus.STOPPED.value == "stopped"
        assert ServiceStatus.ERROR.value == "error"

    def test_status_comparison(self) -> None:
        """Testet ServiceStatus Vergleiche."""
        assert ServiceStatus.UNINITIALIZED != ServiceStatus.RUNNING
        assert ServiceStatus.RUNNING == ServiceStatus.RUNNING

    def test_status_string_representation(self) -> None:
        """Testet String-Repräsentation."""
        assert str(ServiceStatus.RUNNING) == "ServiceStatus.RUNNING"


class MockService(LifecycleService):
    """Mock-Service für erweiterte Tests."""

    def __init__(self) -> None:
        self.initialize_mock: AsyncMock = AsyncMock()
        self.shutdown_mock: AsyncMock = AsyncMock()

    async def initialize(self) -> None:
        """Mock-Initialisierung."""
        await self.initialize_mock()  # type: ignore[misc]

    async def shutdown(self) -> None:
        """Mock-Shutdown."""
        await self.shutdown_mock()  # type: ignore[misc]


class TestServiceLifecycle:
    """Tests für Service-Lifecycle-Management."""

    @pytest.fixture
    def mock_service(self) -> MockService:
        """Erstellt Mock-Service."""
        return MockService()

    @pytest.mark.asyncio
    async def test_lifecycle_methods_called(self, mock_service: MockService) -> None:
        """Testet, dass Lifecycle-Methoden aufgerufen werden."""
        await mock_service.initialize()
        mock_service.initialize_mock.assert_called_once()

        await mock_service.shutdown()
        mock_service.shutdown_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_customizable(self, mock_service: MockService) -> None:
        """Testet, dass Health-Check anpassbar ist."""
        # Standard Health-Check
        health = await mock_service.health_check()
        assert health["status"] == "healthy"
        assert health["service"] == "MockService"

    def test_status_management(self, mock_service: MockService) -> None:
        """Testet Status-Management."""
        # Standard-Status
        status = mock_service.get_status()
        assert status == ServiceStatus.RUNNING


if __name__ == "__main__":
    pytest.main([__file__])
