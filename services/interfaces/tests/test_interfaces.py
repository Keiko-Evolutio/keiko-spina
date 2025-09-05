"""Tests für Service-Interface-Definitionen."""

from __future__ import annotations

from abc import ABC
from typing import get_type_hints

import pytest

from services.interfaces import (
    AgentService,
    BusService,
    CoreService,
    DomainRevalidationService,
    FeatureService,
    InfrastructureService,
    LifecycleService,
    RateLimiterService,
    ServiceManagerService,
    ServiceStatus,
    StreamService,
    UtilityService,
    VoiceService,
    WebhookManagerService,
)


class TestInterfaceDefinitions:
    """Tests für korrekte Interface-Definitionen."""

    def test_all_services_are_abstract(self) -> None:
        """Testet, dass alle Service-Interfaces abstrakt sind."""
        services = [
            AgentService,
            BusService,
            StreamService,
            VoiceService,
            ServiceManagerService,
            DomainRevalidationService,
            WebhookManagerService,
            RateLimiterService,
        ]

        for service in services:
            assert issubclass(service, ABC), f"{service.__name__} sollte von ABC erben"

            # Teste, dass Service nicht direkt instanziiert werden kann
            with pytest.raises(TypeError):
                service()  # type: ignore[abstract]

    def test_lifecycle_service_inheritance(self) -> None:
        """Testet, dass alle Services von LifecycleService erben."""
        services = [
            AgentService,
            BusService,
            StreamService,
            VoiceService,
            ServiceManagerService,
            DomainRevalidationService,
            WebhookManagerService,
            RateLimiterService,
        ]

        for service in services:
            assert issubclass(service, LifecycleService), f"{service.__name__} sollte von LifecycleService erben"

    def test_service_categorization(self) -> None:
        """Testet korrekte Service-Kategorisierung."""
        # Core Services
        assert issubclass(AgentService, CoreService)
        assert issubclass(BusService, CoreService)
        assert issubclass(StreamService, CoreService)

        # Infrastructure Services
        assert issubclass(ServiceManagerService, InfrastructureService)
        assert issubclass(DomainRevalidationService, InfrastructureService)

        # Feature Services
        assert issubclass(VoiceService, FeatureService)
        assert issubclass(WebhookManagerService, FeatureService)

        # Utility Services
        assert issubclass(RateLimiterService, UtilityService)

    def test_service_status_enum(self) -> None:
        """Testet ServiceStatus Enum."""
        assert ServiceStatus.UNINITIALIZED.value == "uninitialized"
        assert ServiceStatus.INITIALIZING.value == "initializing"
        assert ServiceStatus.RUNNING.value == "running"
        assert ServiceStatus.STOPPING.value == "stopping"
        assert ServiceStatus.STOPPED.value == "stopped"
        assert ServiceStatus.ERROR.value == "error"

    def test_abstract_methods_present(self) -> None:
        """Testet, dass alle Services die erforderlichen abstrakten Methoden haben."""
        # LifecycleService Methoden
        lifecycle_methods = {"initialize", "shutdown"}

        services = [
            AgentService,
            BusService,
            StreamService,
            VoiceService,
            ServiceManagerService,
            DomainRevalidationService,
            WebhookManagerService,
            RateLimiterService,
        ]

        for service in services:
            abstract_methods = service.__abstractmethods__

            # Alle Services müssen LifecycleService Methoden implementieren
            for method in lifecycle_methods:
                assert method in abstract_methods, f"{service.__name__} fehlt abstrakte Methode: {method}"

    def test_type_hints_present(self) -> None:
        """Testet, dass alle abstrakten Methoden Type Hints haben."""
        services = [
            AgentService,
            BusService,
            StreamService,
            VoiceService,
            ServiceManagerService,
            DomainRevalidationService,
            WebhookManagerService,
            RateLimiterService,
        ]

        for service in services:
            for method_name in service.__abstractmethods__:
                method = getattr(service, method_name)
                hints = get_type_hints(method)

                # Mindestens return type sollte vorhanden sein
                assert hints, f"{service.__name__}.{method_name} fehlen Type Hints"


class TestImportStructure:
    """Tests für korrekte Import-Struktur."""

    def test_all_exports_importable(self) -> None:
        """Testet, dass alle __all__ Exports importierbar sind."""
        from services.interfaces import __all__

        # Teste, dass alle Exports tatsächlich importiert werden können
        for export_name in __all__:
            try:
                from services.interfaces import __dict__
                assert export_name in __dict__, f"Export {export_name} nicht verfügbar"
            except ImportError as e:
                __dict__ = {}  # Fallback für __dict__ import
                pytest.fail(f"Konnte {export_name} nicht importieren: {e}")

    def test_backward_compatibility(self) -> None:
        """Testet Backward Compatibility für alte Interface-Namen."""
        from services.interfaces import RateLimiterBackend, RateLimiterService

        # RateLimiterBackend sollte Alias für RateLimiterService sein
        assert RateLimiterBackend is RateLimiterService

    def test_no_circular_imports(self) -> None:
        """Testet, dass keine zirkulären Imports existieren."""
        # Dieser Test wird durch erfolgreichen Import bereits validiert

        # Wenn wir hier ankommen, gibt es keine zirkulären Imports
        assert True


class TestDocstrings:
    """Tests für Docstring-Qualität."""

    def test_all_services_have_docstrings(self) -> None:
        """Testet, dass alle Services Docstrings haben."""
        services = [
            AgentService,
            BusService,
            StreamService,
            VoiceService,
            ServiceManagerService,
            DomainRevalidationService,
            WebhookManagerService,
            RateLimiterService,
        ]

        for service in services:
            assert service.__doc__, f"{service.__name__} fehlt Docstring"
            assert len(service.__doc__.strip()) > 10, f"{service.__name__} Docstring zu kurz"

    def test_abstract_methods_have_docstrings(self) -> None:
        """Testet, dass abstrakte Methoden Docstrings haben."""
        services = [
            AgentService,
            BusService,
            StreamService,
            VoiceService,
            ServiceManagerService,
            DomainRevalidationService,
            WebhookManagerService,
            RateLimiterService,
        ]

        for service in services:
            for method_name in service.__abstractmethods__:
                method = getattr(service, method_name)
                assert method.__doc__, f"{service.__name__}.{method_name} fehlt Docstring"


if __name__ == "__main__":
    pytest.main([__file__])
